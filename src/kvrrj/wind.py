import calendar
import json
import urllib
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
from honeybee.config import folders as hb_folders
from ladybug.datatype.angle import WindDirection
from ladybug.datatype.speed import WindSpeed
from ladybug.dt import Date
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.sunpath import DateTime, Location
from ladybug.windrose import WindRose
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter

from kvrrj.geometry.util import (
    angle_clockwise_from_north,
    angle_to_vector,
    cardinality,
    circular_weighted_mean,
)
from kvrrj.ladybug.analysisperiod import (
    _analysis_period_to_string,
    _iterable_datetimes_to_lb_analysis_period,
)
from kvrrj.ladybug.location import (
    _is_location_time_zone_valid_for_location,
    average_location,
    get_timezone_str_from_location,
    get_tzinfo_from_location,
    get_utc_offset_from_location,
)
from kvrrj.util import (
    _datetimes_span_at_least_1_year,
    _is_iterable_single_dtype,
    _is_leap_year,
)
from kvrrj.viz.color import contrasting_color, to_hex

from .logging import CONSOLE_LOGGER  # noqa: F401


class TerrainType(Enum):
    """A class to represent the terrain type for wind data."""

    CITY = auto()
    SUBURBAN = auto()
    COUNTRY = auto()
    WATER = auto()

    @property
    def roughness_length(self) -> float:
        d = {
            TerrainType.CITY.name: 1.0,
            TerrainType.SUBURBAN.name: 0.5,
            TerrainType.COUNTRY.name: 0.1,
            TerrainType.WATER.name: 0.03,
        }
        return d[self.name]

    @property
    def boundary_layer_height(self) -> float:
        d = {
            TerrainType.CITY.name: 460,
            TerrainType.SUBURBAN.name: 370,
            TerrainType.COUNTRY.name: 270,
            TerrainType.WATER.name: 210,
        }
        return d[self.name]

    @property
    def power_law_exponent(self) -> float:
        d = {
            TerrainType.CITY.name: 0.33,
            TerrainType.SUBURBAN.name: 0.22,
            TerrainType.COUNTRY.name: 0.14,
            TerrainType.WATER.name: 0.1,
        }
        return d[self.name]

    @classmethod
    def from_roughness_length(cls, roughness_length: float) -> "TerrainType":
        """Get the terrain type from a roughness length."""
        return abs(
            pd.Series({tt: tt.roughness_length for tt in cls}) - roughness_length
        ).idxmin()

    def translate_wind_speed(
        self,
        reference_value: float,
        reference_height: float,
        target_height: float,
        log_law: bool = False,
        target_terrain_type: "TerrainType" = None,
    ) -> float:
        if target_terrain_type is None:
            target_terrain_type = self

        if log_law:
            if target_height <= target_terrain_type.roughness_length:
                return 0
            ref_h_ratio = reference_height / self.roughness_length
            ref_log_denom = np.log(ref_h_ratio)
            ref_log_num = np.log(target_height / target_terrain_type.roughness_length)
            return float(reference_value * (ref_log_num / ref_log_denom))

        ref_h_ratio = self.boundary_layer_height / reference_height
        ref_power_denom = ref_h_ratio**self.power_law_exponent
        target_h_ratio = (
            target_height / target_terrain_type.boundary_layer_height
        ) ** target_terrain_type.power_law_exponent
        return float(target_h_ratio * (reference_value * ref_power_denom))


@dataclass
class Wind:
    """An object containing wind data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (list[datetime]):
            An iterable of datetime-like objects.
        wind_speed (list[float]):
            A list of wind speeds, in m/s.
        wind_direction (list[float]):
            A list of wind directions, in degrees clockwise from north (at 0).
        height_above_ground (float):
            The height above ground (in m) where the input wind speeds and
            directions were collected.
        terrain_type (TerrainType):
            The terrain associtaed with this wind data.
    """

    # NOTE - BE STRICT WITH THE TYPING!
    # NOTE - Conversions happen in class methods.
    # NOTE - Validation happens at instantiation.

    location: Location
    datetimes: list[datetime]
    wind_speed: list[float]
    wind_direction: list[float]
    height_above_ground: float
    terrain_type: TerrainType = None

    # region: DUNDER METHODS

    def __post_init__(self):
        """Check for validation of the inputs."""

        # location validation
        if not isinstance(self.location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if self.location.source is None:
            warnings.warn(
                'The source field of the Location input is None. This means that things are a bit ambiguous! A default value of "somewhere ... be more specific!" has been added.'
            )
            self.location.source = "UnknownSource"
        if not _is_location_time_zone_valid_for_location(self.location):
            warnings.warn(
                f"The time zone of the location ({self.location.time_zone}) does not match the time zone of the lat/lon ({self.location.latitude}, {self.location.longitude})."
            )

        # datetimes validation
        if not _is_iterable_single_dtype(self.datetimes, datetime):
            raise ValueError("datetimes must be a list of datetime-like objects.")

        # timezone validation
        tzi = get_tzinfo_from_location(self.location)
        if self.datetimes[0].tzinfo is None:
            self.datetimes = (
                pd.to_datetime(self.datetimes)
                .tz_localize(tzi, ambiguous="NaT")
                .to_pydatetime()
                .tolist()
            )
        target_utc_offset = tzi.utcoffset(self.datetimes[0]).seconds / 3600
        actual_utc_offset = (
            self.datetimes[0].tzinfo.utcoffset(self.datetimes[0]).seconds / 3600
        )
        if target_utc_offset != actual_utc_offset:
            raise ValueError(
                f"datetimes time zone does not match location time zone. Expected {target_utc_offset}, got {actual_utc_offset}."
            )

        # height above ground validation
        if not isinstance(self.height_above_ground, (int, float)):
            raise ValueError("height_above_ground must be a number.")
        if self.height_above_ground < 0.1:
            raise ValueError(
                "height_above_ground must be greater than or equal to 0.1."
            )

        # terrain type validation
        if self.terrain_type is None:
            self.terrain_type = TerrainType.COUNTRY
            warnings.warn(
                "terrain_type was not provided. Defaulting to TerrainType.COUNTRY."
            )
        if not isinstance(self.terrain_type, TerrainType):
            raise ValueError("terrain_type must be a TerrainType object.")

        # data validation
        array_names = [
            "wind_speed",
            "wind_direction",
        ]
        for name in array_names:
            if len(getattr(self, name)) != len(self.datetimes):
                raise ValueError(
                    f"{name} must be the same length as datetimes. {len(getattr(self, name))} != {len(self.datetimes)}."
                )
            if not _is_iterable_single_dtype(
                getattr(self, name), (int, float, np.float64)
            ):
                raise ValueError(f"{name} must be a list of numeric values.")
            if any(np.isnan(getattr(self, name))):
                raise ValueError(f"{name} cannot contain null values.")
        if any(i < 0 for i in self.wind_speed):
            raise ValueError("Wind speeds cannot be negative.")
        if any(i < 0 or i > 360 for i in self.wind_direction):
            raise ValueError("Wind directions must be between 0 and 360 degrees.")

    def __len__(self) -> int:
        return len(self.datetimes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} data from {self.location.source}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.location,
                tuple(self.datetimes),
                tuple(self.wind_speed),
                tuple(self.wind_direction),
                str(self.terrain_type),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Wind):
            return False
        return (
            self.location == other.location
            and self.datetimes == other.datetimes
            and self.wind_speed == other.wind_speed
            and self.wind_direction == other.wind_direction
            and self.terrain_type == other.terrain_type
        )

    def __iter__(self) -> iter:
        return (
            (
                self.datetimes[i],
                self.wind_speed[i],
                self.wind_direction[i],
            )
            for i in range(len(self))
        )

    def __getitem__(self, idx: int) -> dict[str, datetime | float]:
        return {
            "datetime": self.datetimes[idx],
            "wind_speed": self.wind_speed[idx],
            "wind_direction": self.wind_direction[idx],
        }

    def __copy__(self) -> "Wind":
        return Wind(
            location=self.location.duplicate(),
            datetimes=self.datetimes,
            wind_speed=self.wind_speed,
            wind_direction=self.wind_direction,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    # endregion: DUNDER METHODS

    # region: STATIC METHODS

    @staticmethod
    def _direction_bin_centers(
        directions: int = 36,
    ) -> list[float]:
        """Calculate the bin centers for a given number of directions.
        This returns a list the length of the number of directions, with each
        bin center representing the centroid of a directional bin. The first
        value is always 0 (or north), and centers move clockwise from there.

        Args:
            directions (int):
                The number of directions to calculate bin centers for.

        Returns:
            list[float]:
                A list of bin centers.
        """
        return np.linspace(0, 360, directions + 1)[:-1].tolist()

    @staticmethod
    def _direction_bin_edges(
        directions: int = 36,
    ) -> list[float]:
        """Calculate the bin edges for a given number of directions.
        The returned list includes half bins for the ranfges about 0/360, so
        that the first and last pair of values the list are "half-bins".

        Args:
            directions (int):
                The number of directions to calculate bin edges for.

        Returns:
            list[float]:
                A list of bin edges.
        """

        bin_width = 360 / directions
        if bin_width == 360:
            bin_edges = np.array([0, 360])
        else:
            bin_edges = np.array(Wind._direction_bin_centers(directions=directions)) - (
                bin_width / 2
            )
            bin_edges = np.where(bin_edges < 0, 360 + bin_edges, bin_edges)
            bin_edges = np.append(bin_edges, bin_edges[0])
            bin_edges[0] = 0
            bin_edges = np.append(bin_edges, 360)
        return bin_edges.tolist()

    # endregion: STATIC METHODS

    # region: PROPERTIES

    @property
    def dates(self) -> list[date]:
        return [dt.date() for dt in self.datetimes]

    @property
    def start_date(self) -> date:
        return min(self.dates)

    @property
    def end_date(self) -> date:
        return max(self.dates)

    @property
    def lb_datetimes(self) -> list[date]:
        return [
            DateTime(
                month=dt.month,
                day=dt.day,
                hour=dt.hour,
                minute=dt.minute,
                leap_year=_is_leap_year(dt.year),
            )
            for dt in self.datetimes
        ]

    @property
    def lb_dates(self) -> list[Date]:
        return [
            Date(
                month=d.month,
                day=d.day,
                leap_year=_is_leap_year(d.year),
            )
            for d in self.dates
        ]

    @property
    def datetimeindex(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.datetimes, freq="infer")

    @property
    def wind_speed_series(self) -> pd.Series:
        return pd.Series(
            data=self.wind_speed,
            index=self.datetimeindex,
            name="Wind Speed (m/s)",
        )

    @property
    def wind_speed_collection(self) -> HourlyContinuousCollection:
        data = self.wind_speed_series
        values = (
            data.groupby([data.index.month, data.index.day, data.index.time])
            .mean()
            .values
        ).tolist()
        ap = self.analysis_period
        header = Header(
            data_type=WindSpeed(),
            unit="m/s",
            analysis_period=ap,
            metadata={"source": self.location.source},
        )
        return HourlyContinuousCollection(header=header, values=values)

    @property
    def wind_direction_series(self) -> pd.Series:
        return pd.Series(
            data=self.wind_direction,
            index=self.datetimeindex,
            name="Wind Direction (degrees)",
        )

    @property
    def wind_direction_collection(self) -> HourlyContinuousCollection:
        data = self.wind_direction_series
        values = (
            data.groupby([data.index.month, data.index.day, data.index.time])
            .mean()
            .values
        ).tolist()
        ap = self.analysis_period
        header = Header(
            data_type=WindDirection(),
            unit="degrees",
            analysis_period=ap,
            metadata={"source": self.location.source},
        )
        return HourlyContinuousCollection(header=header, values=values)

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.wind_speed_series,
                self.wind_direction_series,
            ],
            axis=1,
        )

    @property
    def analysis_period(self) -> AnalysisPeriod:
        idx = self.datetimeindex
        data = pd.Series(index=idx, data=np.zeros_like(idx))
        data = data.groupby([data.index.month, data.index.day, data.index.time]).mean()
        midx = data.index
        year = 2016 if [2, 29] in midx.to_frame(index=False)[[0, 1]].values else 2017
        dts = (
            pd.to_datetime(
                [
                    datetime(year, month, day, time.hour, time.minute, time.second)
                    for month, day, time in midx
                ]
            )
            .to_pydatetime()
            .tolist()
        )
        return _iterable_datetimes_to_lb_analysis_period(dts)

    @property
    def uv(self) -> list[tuple[float, float]]:
        """Return the U and V wind components in m/s."""
        uvs = []
        for wd, ws in zip(*[self.wind_direction, self.wind_speed]):
            if ws == 0:
                uvs.append((0, 0))
            else:
                u, v = angle_to_vector(wd)
                uvs.append((float(u * ws), float(v * ws)))
        return uvs

    # endregion: PROPERTIES

    # region: CLASS METHODS

    @classmethod
    def from_epw(
        cls, epw: Path | EPW, terrain_type: TerrainType = TerrainType.COUNTRY
    ) -> "Wind":
        """Create a Wind object from an EPW file or object.

        Args:
            epw (Path | EPW):
                The path to the EPW file, or an EPW object.
        """

        if isinstance(epw, (str, Path)):
            epw = EPW(epw)

        # modify location to state the EPW file in the source field
        location = epw.location
        location.source = f"{Path(epw.file_path).name}"

        # obtain the datetimes
        datetimes = pd.to_datetime(
            epw.dry_bulb_temperature.header.analysis_period.datetimes
        )
        # add timezone information to the datetimes
        datetimes = [
            dt.replace(
                tzinfo=timezone(timedelta(hours=epw.location.time_zone))
            ).to_pydatetime()
            for dt in datetimes
        ]

        return cls(
            location=location,
            datetimes=datetimes,
            wind_speed=epw.wind_speed.values,
            wind_direction=epw.wind_direction.values,
            height_above_ground=10,
            terrain_type=terrain_type,
        )

    def to_dict(self) -> dict:
        """Represent the object as a python-native dtype dictionary."""

        return {
            "type": "Wind",
            "location": self.location.to_dict(),
            "datetimes": [i.isoformat() for i in self.datetimes],
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "height_above_ground": self.height_above_ground,
            "terrain_type": self.terrain_type.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Wind":
        """Create this object from a dictionary."""

        if d.get("type", None) != "Wind":
            raise ValueError("The dictionary cannot be converted Wind object.")

        return cls(
            location=Location.from_dict(d["location"]),
            datetimes=pd.to_datetime(d["datetimes"]),
            wind_speed=d["wind_speed"],
            wind_direction=d["wind_direction"],  #
            height_above_ground=d["height_above_ground"],
            terrain_type=TerrainType[d["terrain_type"]],  # type: ignore[call-arg]
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Wind":
        """Create this object from a JSON string."""
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        location: Location,
        terrain_type: TerrainType = TerrainType.COUNTRY,
        wind_speed_column: str = None,
        wind_direction_column: str = None,
        height_above_ground: float = 10,
    ) -> "Wind":
        """Create this object from a DataFrame.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing the wind data.
            location (Location, optional):
                A ladybug Location object. If not provided, the location data
                will be extracted from the DataFrame if present.
        """

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame's index must be of type pd.DatetimeIndex.")
        if not isinstance(location, Location):
            raise ValueError("location must be a ladybug Location object.")

        # get the columns from a best guess
        if wind_speed_column is None:
            for col in [
                "wind_speed",
                "speed",
                "Wind Speed (m/s)",
                "ws",
            ]:
                if col in df.columns:
                    wind_speed_column = col
                    break
            if wind_speed_column is None:
                raise ValueError(
                    "wind_speed_column not found in DataFrame. You'll need to provide a specific column label rather than relying on a best-guess."
                )
        if wind_direction_column is None:
            for col in [
                "wind_direction",
                "direction",
                "Wind Direction (degrees)",
                "wd",
            ]:
                if col in df.columns:
                    wind_direction_column = col
                    break
            if wind_direction_column is None:
                raise ValueError(
                    "wind_direction_column not found in DataFrame. You'll need to provide a specific column label rather than relying on a best-guess."
                )

        return cls(
            location=location,
            datetimes=df.index.to_pydatetime().tolist(),
            wind_speed=df[wind_speed_column].tolist(),
            wind_direction=df[wind_direction_column].tolist(),
            height_above_ground=height_above_ground,
            terrain_type=terrain_type,
        )

    @classmethod
    def from_average(cls, objects: list["Wind"], weights: list[float] = None) -> "Wind":
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""

        # validation
        if not _is_iterable_single_dtype(objects, Wind):
            raise ValueError("objects must be a list of Wind objects.")
        if len(objects) == 0:
            raise ValueError("objects cannot be empty.")
        if len(objects) == 1:
            return objects[0]

        # check datetimes are the same
        for obj in objects:
            if obj.datetimes != objects[0].datetimes:
                raise ValueError("All objects must share the same datetimes.")

        # create default weightings if None
        if weights is None:
            weights = [1 / len(objects)] * len(objects)
        else:
            if sum(weights) != 1:
                raise ValueError("weights must total 1.")

        # create average location
        avg_location = average_location([i.location for i in objects], weights=weights)

        # align collections so that intersection only is created
        df_ws = pd.concat([i.wind_speed_series for i in objects], axis=1).dropna()
        df_wd = pd.concat([i.wind_direction_series for i in objects], axis=1).dropna()

        # construct the weighted means
        wd_avg = np.array(
            [circular_weighted_mean(i, weights) for _, i in df_wd.iterrows()]
        )
        ws_avg = np.average(df_ws, axis=1, weights=weights)

        # construct the avg height above ground
        avg_height_above_ground = np.average(
            [i.height_above_ground for i in objects], weights=weights
        )

        # construct the new terrain type, based on the average of the input objects
        avg_roughness_length = np.average(
            [i.terrain_type.roughness_length for i in objects], weights=weights
        )
        terrain_type = TerrainType.from_roughness_length(avg_roughness_length)

        # return the new averaged object
        return cls(
            wind_speed=ws_avg.tolist(),
            wind_direction=wd_avg.tolist(),
            datetimes=objects[0].datetimes,
            height_above_ground=avg_height_above_ground,
            location=avg_location,
            terrain_type=terrain_type,
        )

    @classmethod
    def from_uv(
        cls,
        u: list[float],
        v: list[float],
        location: Location,
        datetimes: list[datetime],
        height_above_ground: float = 10,
        terrain_type: TerrainType = TerrainType.COUNTRY,
    ) -> "Wind":
        """Create a Wind object from a set of U, V wind components.

        Args:
            u (list[float]):
                An iterable of U (eastward) wind components in m/s.
            v (list[float]):
                An iterable of V (northward) wind components in m/s.
            datetimes (list[datetime]):
                An iterable of datetime-like objects.
            height_above_ground (float, optional):
                The height above ground (in m) where the input wind speeds and
                directions were collected.
                Defaults to 10m.
            source (str, optional):
                A source string to describe where the input data comes from.
                Defaults to None.

        Returns:
            Wind:
                A Wind object!
        """

        # convert UV into angle and magnitude
        wind_direction = angle_clockwise_from_north(np.stack([u, v]))
        wind_speed = np.sqrt(np.square(u) + np.square(v))

        if any(wind_direction[wind_speed == 0] == 90):
            warning_message = "Some input vectors have velocity of 0. This is not bad, but can mean directions may be misreported."
            warnings.warn(warning_message, UserWarning)

        return cls(
            wind_speed=wind_speed.tolist(),
            wind_direction=wind_direction.tolist(),
            datetimes=datetimes,
            height_above_ground=height_above_ground,
            location=location,
            terrain_type=terrain_type,
        )

    @classmethod
    def from_openmeteo(
        cls,
        location: Location,
        start_date: str | date,
        end_date: str | date,
        terrain_type: TerrainType = TerrainType.COUNTRY,
    ) -> "Wind":
        """Query Open Meteo for wind data."""

        if not isinstance(location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).to_pydatetime().date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).to_pydatetime().date()
        if not isinstance(start_date, date):
            raise ValueError("start_date must be a date object.")
        if not isinstance(end_date, date):
            raise ValueError("end_date must be a date object.")
        if start_date > end_date:
            raise ValueError("start_date must be less than end_date.")

        # create the list of datetimes being queried
        dates = [
            i.date()
            for i in pd.date_range(
                start=start_date, end=end_date, freq="D", inclusive="both"
            )
        ]
        _datetimes = pd.date_range(
            start=start_date,
            end=end_date + timedelta(days=1),
            freq="h",
            inclusive="both",
        )[:-1]
        if not _datetimes_span_at_least_1_year(_datetimes):
            raise ValueError("The dates must span an entire year.")
        if start_date.year == end_date.year:
            # single year covered
            if _is_leap_year(start_date.year):
                # leap year
                if len(dates) < 366:
                    raise ValueError(
                        "The requested date range must be at least a full year."
                    )
            else:
                # non-leap year
                if len(dates) < 365:
                    raise ValueError(
                        "The requested date range must be at least a full year."
                    )

        # get time zone from the location and check that it matches the location
        time_zone_str = get_timezone_str_from_location(location)
        time_zone_hours = get_utc_offset_from_location(location).seconds / 3600
        if not _is_location_time_zone_valid_for_location(location):
            raise ValueError(
                f"The time zone of the location ({location.time_zone}) does not match the time zone of the lat/lon ({time_zone_hours})."
            )

        # modify the location so that its source is OpenMeteo ERA5
        location = location.duplicate()
        location.source = (
            f"OpenMeteo ERA5 at {location.latitude}°, {location.longitude}°"
        )

        # set the output directory
        _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_openmeteo"
        _dir.mkdir(exist_ok=True, parents=True)

        # variables to query
        vars = [
            {
                "openmeteo_name": "winddirection_10m",
                "openmeteo_unit": "°",
                "target_name": "Wind Direction",
                "target_unit": "degrees",
                "target_multiplier": 1,
            },
            {
                "openmeteo_name": "windspeed_10m",
                "openmeteo_unit": "km/h",
                "target_name": "Wind Speed",
                "target_unit": "m/s",
                "target_multiplier": 1 / 3.6,
            },
        ]

        # create the savepath for the returned data
        sp = (
            _dir
            / f"wind_{location.latitude}_{location.longitude}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.json"
        )
        if sp.exists():
            CONSOLE_LOGGER.info(f"Loading data from {sp.name}")
            with open(sp, "r") as f:
                data = json.load(f)
        else:
            # construct url query string
            var_strings = ",".join([i["openmeteo_name"] for i in vars])
            query_string = "".join(
                [
                    "https://archive-api.open-meteo.com/v1/era5",
                    f"?latitude={location.latitude}",
                    f"&longitude={location.longitude}",
                    f"&timezone={time_zone_str}",
                    f"&start_date={start_date:%Y-%m-%d}",
                    f"&end_date={end_date:%Y-%m-%d}",
                    f"&hourly={var_strings}",
                ]
            )

            # query the data, and save to file
            CONSOLE_LOGGER.info(f"Querying data from {query_string}")
            with urllib.request.urlopen(query_string) as url:
                data = json.load(url)
            with open(sp, "w") as f:
                json.dump(data, f)

        # get the datetimes
        datetimes: list[datetime] = (
            pd.to_datetime(data["hourly"]["time"])
            .tz_localize(pytz.FixedOffset(data["utc_offset_seconds"] / 60))
            .to_pydatetime()
            .tolist()
        )
        # get the wind speed
        ws = [i * 1 / 3.6 for i in data["hourly"]["windspeed_10m"]]
        # get the wind direction
        wd = data["hourly"]["winddirection_10m"]
        return cls(
            location=location,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=10,
            terrain_type=terrain_type,
        )

    # endregion: CLASS METHODS

    # region: FILTER METHODS

    def filter_by_boolean_mask(self, mask: list[bool] = None) -> "Wind":
        """Filter the current object by a boolean mask.

        Args:
            mask (list[bool]):
                A boolean mask to filter the current object.

        Returns:
            Wind:
                A dataset describing solar radiation.
        """

        if mask is None:
            mask = [True] * len(self)

        # validations
        if not _is_iterable_single_dtype(mask, bool):
            raise ValueError("mask must be a list of booleans.")
        if len(mask) != len(self):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )
        if sum(mask) == 0:
            raise ValueError("No data remains within the given boolean filters.")
        if sum(mask) == len(self):
            return self

        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered)"

        return Wind(
            location=loc,
            datetimes=[i for i, j in zip(*[self.datetimes, mask]) if j],
            wind_speed=[i for i, j in zip(*[self.wind_speed, mask]) if j],
            wind_direction=[i for i, j in zip(*[self.wind_direction, mask]) if j],
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> "Wind":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Wind:
                A dataset describing wind.
        """
        mask = []
        for n, i in enumerate(self.lb_datetimes):
            mask.append(i in analysis_period.datetimes)

        # create new data
        loc = self.location.duplicate()
        loc.source = (
            f"{self.location.source} (filtered to {_analysis_period_to_string(analysis_period)})",
        )
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_direction=wd,
            wind_speed=ws,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_time(
        self,
        years: list[int] = None,
        months: list[float] = None,
        days: list[float] = None,
        hours: list[int] = None,
    ) -> "Wind":
        """Filter the current object by months, days, hours.

        Args:
            years (list[int], optional):
                A list of years to include.
                Default to all years.
            months (list[int], optional):
                A list of months.
                Defaults to all possible months.
            days (list[int], optional):
                A list of days.
                Defaults to all possible days.
            hours (list[int], optional):
                A list of hours.
                Defaults to all possible hours.

        Returns:
            Wind:
                A dataset describing historic wind data.
        """
        idx = self.datetimeindex
        filtered_by = []
        if years is None:
            years = idx.year.unique().tolist()
        else:
            filtered_by.append("year")
        if months is None:
            months = list(range(1, 13))
        else:
            filtered_by.append("month")
        if days is None:
            days = list(range(1, 32))
        else:
            filtered_by.append("day")
        if hours is None:
            hours = list(range(0, 24))
        else:
            filtered_by.append("hour")

        if len(filtered_by) > 2:
            filtered_by = ", ".join(filtered_by[:-1]) + ", and " + str(filtered_by[-1])
        elif len(filtered_by) == 2:
            filtered_by = " and ".join(filtered_by)
        elif len(filtered_by) == 1:
            filtered_by = filtered_by[0]

        # construct masks
        year_mask = idx.year.isin(years)
        month_mask = idx.month.isin(months)
        day_mask = idx.day.isin(days)
        hour_mask = idx.hour.isin(hours)
        mask = np.all([year_mask, month_mask, day_mask, hour_mask], axis=0)

        # create new data
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered by {filtered_by})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_direction(
        self,
        left_angle: float = 0,
        right_angle: float = 360,
        include_left: bool = True,
        include_right: bool = True,
    ) -> "Wind":
        """Filter the current object by wind direction, based on the angle as
        observed from a location.

        Args:
            left_angle (float):
                The left-most angle, to the left of which wind speeds and
                directions will be removed.
                Defaults to 0.
            right_angle (float):
                The right-most angle, to the right of which wind speeds and
                directions will be removed.
                Defaults to 360.
            include_left (bool, optional):
                Include values that are exactly the left angle.
                Defaults to True.
            include_right (bool, optional):
                Include values that are exactly the right angle.
                Defaults to True.

        Return:
            Wind:
                A Wind object!
        """

        if left_angle < 0 or right_angle > 360:
            raise ValueError("Angle limits must be between 0 and 360 degrees.")

        if left_angle == 0 and right_angle == 360:
            return self

        if (left_angle == right_angle) or (left_angle == 360 and right_angle == 0):
            raise ValueError("Angle limits cannot be identical.")

        wd = self.wind_direction_series.values

        if include_right:
            right_mask = wd <= right_angle
        else:
            right_mask = wd < right_angle

        if include_left:
            left_mask = wd >= left_angle
        else:
            left_mask = wd > left_angle

        if left_angle > right_angle:
            mask = left_mask | right_mask
        else:
            mask = left_mask & right_mask

        # create new data
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (filtered by direction {'[' if include_left else '('}{left_angle}°-{right_angle}°{']' if include_right else ')'})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    def filter_by_speed(
        self,
        min_speed: float = 0,
        max_speed: float = np.inf,
        include_left: bool = True,
        include_right: bool = True,
    ) -> "Wind":
        """Filter the current object by wind speed, based on given low-high limit values.

        Args:
            min_speed (float):
                The lowest speed to include. Values below this wil be removed.
                Defaults to 0.
            max_speed (float):
                The highest speed to include. Values above this wil be removed.
                Defaults to np.inf.
            include_right (bool, optional):
                Include values that are exactly the max speed.
                Defaults to True.
            include_left (bool, optional):
                Include values that are exactly the min speed.

        Return:
            Wind:
                A Wind object!
        """

        if min_speed < 0:
            raise ValueError("min_speed cannot be negative.")

        if max_speed <= min_speed:
            raise ValueError("min_speed must be less than max_speed.")

        if min_speed == 0 and np.isinf(max_speed):
            return self

        ws = self.wind_speed_series.values
        if include_right:
            right_mask = ws <= max_speed
        else:
            right_mask = ws < max_speed

        if include_left:
            left_mask = ws >= min_speed
        else:
            left_mask = ws > min_speed

        mask = left_mask & right_mask

        # create new data
        loc = self.location.duplicate()
        speed_range = f"{'[' if include_left else '('}{min_speed}m/s-{max_speed}m/s{']' if include_right else ')'}"
        loc.source = f"{self.location.source} (filtered by speed {speed_range})"
        datetimes = [i for i, j in zip(*[self.datetimes, mask]) if j]
        ws = [i for i, j in zip(*[self.wind_speed, mask]) if j]
        wd = [i for i, j in zip(*[self.wind_direction, mask]) if j]

        return Wind(
            location=loc,
            datetimes=datetimes,
            wind_speed=ws,
            wind_direction=wd,
            height_above_ground=self.height_above_ground,
            terrain_type=self.terrain_type,
        )

    # endregion: FILTER METHODS

    # region: INSTANCE METHODS

    def _direction_categories(self, directions: int = 36) -> pd.Categorical:
        edges = self._direction_bin_edges(directions=directions)
        return pd.cut(self.wind_direction, bins=edges, include_lowest=True, right=True)

    def _direction_binned_data(
        self, directions: int = 36, other_data: Any = None
    ) -> dict[str, list[Any]]:
        """Bin data by wind direction."""
        if other_data is None:
            other_data = self.wind_speed
        if len(other_data) != len(self):
            raise ValueError("other_data must be same length as this object")

        binned = self._direction_categories(directions=directions)
        grp = pd.Series(other_data).groupby(binned, observed=True)
        d = {k: table.values.tolist() for k, table in grp}
        # combine the first and last bins
        renamer = {}
        for n, interval in enumerate(binned.categories):
            if n == 0 or n == len(binned.categories) - 1:
                renamer[interval] = (
                    (
                        str(binned.categories[-1]).split(",")[0]
                        if directions != 1
                        else "(0.0"
                    )
                    + ","
                    + str(binned.categories[0]).split(",")[1]
                )
            else:
                renamer[interval] = str(interval)
        # rename the keys in the original dict
        d_renamed = {}
        for k, v in d.items():
            target_key = renamer[k]
            if target_key in d_renamed:
                d_renamed[target_key].extend(v)
            else:
                d_renamed[target_key] = v
        return d_renamed

    def proportion_calm(self, threshold: float = 0.1) -> float:
        """Return the proportion of timesteps "calm" (i.e. less than or equal
        to the threshold).

        Args:
            threshold (float, optional):
                The threshold for calm wind speeds. Defaults to 0.1.

        Returns:
            float:
                The proportion of calm instances.
        """
        s = self.wind_speed_series
        return float((s <= threshold).sum() / len(s))

    def calm_mask(self, threshold: float = 0.1) -> list[bool]:
        """Return a boolean mask of the timesteps that are "calm" (i.e. less than or equal to the threshold).

        Args:
            threshold (float, optional):
                The threshold for calm wind speeds. Defaults to 0.1.

        Returns:
            list[bool]:
                A boolean mask of the timesteps that are calm.
        """
        return (np.array(self.wind_speed) <= threshold).tolist()

    def percentile(
        self, q: float | list[float] = (0.25, 0.5, 0.75, 0.95), directions: int = 8
    ) -> pd.DataFrame:
        """Calculate the wind speed at the given percentiles."""
        q = np.atleast_1d(q)
        dd = self._direction_binned_data(directions=directions)
        return pd.DataFrame(
            {k: np.quantile(v, q).tolist() for k, v in dd.items()}, index=q
        ).T

    def to_height(
        self,
        target_height: float,
        log_law: bool = True,
    ) -> "Wind":
        """Translate the object to a different height above ground.

        Args:
            target_height (float):
                Height to translate to (in m).
            terrain_roughness_length (float, optional):
                Terrain roughness (how big objects are to adjust translation).
                Defaults to 1.
            log_function (bool, optional):
                Whether to use log-function or pow-function. Defaults to True.

        Returns:
            Wind:
                A translated Wind object.
        """
        if self.height_above_ground == target_height:
            return self

        wss = [
            self.terrain_type.translate_wind_speed(
                reference_value=ws,
                reference_height=self.height_above_ground,
                target_height=target_height,
                log_law=log_law,
            )
            for ws in self.wind_speed
        ]
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} translated to {target_height}m"
        return Wind(
            wind_speed=wss,
            wind_direction=self.wind_direction,
            datetimes=self.datetimes,
            height_above_ground=target_height,
            location=loc,
            terrain_type=self.terrain_type,
        )

    def apply_directional_factors(
        self, directions: int, factors: tuple[float]
    ) -> "Wind":
        """Adjust wind speed values by a set of factors per direction.
        Factors start at north, and move clockwise. Right edges are inclusive.

        Example:
            >>> wind = Wind.from_epw(epw_path)
            >>> wind.apply_directional_factors(
            ...     directions=4,
            ...     factors=(0.5, 0.75, 1, 0.75)
            ... )

        Where northern winds would be multiplied by 0.5, eastern winds by 0.75,
        southern winds by 1, and western winds by 0.75.

        Args:
            directions (int):
                The number of directions to bin wind-directions into.
            factors (tuple[float], optional):
                Adjustment factors per direction.

        Returns:
            Wind:
                An adjusted Wind object.
        """
        binned = self._direction_categories(directions=directions)

        if len(binned.categories) - 1 != len(factors):
            raise ValueError("Number of factors must be equal to number of directions.")

        mapping = {k: v for k, v in zip(binned.categories, factors + [factors[0]])}

        wind_speeds = self.wind_speed * binned.map(mapping, na_action="ignore")

        loc = self.location.duplicate()
        loc.source = f"{self.location.source} (adjusted by {directions} directional factors {factors})"

        return Wind(
            wind_speed=wind_speeds.tolist(),
            wind_direction=self.wind_direction,
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
            location=loc,
            terrain_type=self.terrain_type,
        )

    def direction_counts(
        self,
        directions: int = 8,
        as_midpoints: bool = False,
    ) -> dict[str, int]:
        """Calculate the number of values per wind direction (i.e. prevailing directions).

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.

        Returns:
            list[float] | list[str]:
                A list of wind directions.
        """
        dd = {
            k: len(np.array(v)[np.array(v) != 0])
            for k, v in self._direction_binned_data(directions=directions).items()
        }
        if as_midpoints:
            lookup = {}
            for n, (k, v) in enumerate(dd.items()):
                if n == 0:
                    lookup[k] = 0.0
                else:
                    lookup[k] = (
                        float(k[1:-1].split(", ")[0]) + float(k[1:-1].split(", ")[1])
                    ) / 2.0
            dd = {lookup[k]: v for k, v in dd.items()}
        return dd

    def prevailing(
        self, directions: int = 8, n: int = 1, as_cardinal: bool = False
    ) -> list[str]:
        pp = self.direction_counts(directions=directions, as_midpoints=True)
        prevailing_directions = [
            i[0] for i in sorted(pp.items(), key=lambda x: x[1], reverse=True)
        ]
        if as_cardinal:
            x = [cardinality(j, directions=32) for j in prevailing_directions]
            # remove duplicates from x, but retain order
            seen = []
            for i in x:
                if i not in seen:
                    seen.append(i)
                if len(seen) == n:
                    break
            return seen
        return prevailing_directions[:n]

    def month_hour_mean_matrix(
        self, other_data: HourlyContinuousCollection = None
    ) -> pd.DataFrame:
        """Calculate the mean direction and "other_data" for each month and
        hour of in the Wind object.

        Args:
            other_data (HourlyContinuousCollection, optional):
                The other data to calculate the matrix for.
                Defaults to None, which will use the wind speed data.

        Returns:
            pd.DataFrame:
                A DataFrame containing average other_data and direction for each
                month and hour of day.
        """

        # ensure data is suitable for matrixisation
        if other_data is None:
            other_data = self.wind_speed_collection

        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as the wind data.")
        if not isinstance(other_data, HourlyContinuousCollection):
            raise ValueError("other_data must be a HourlyContinuousCollection.")

        idx = self.datetimeindex
        # convert other data to a series
        other_data = pd.Series(
            other_data.values,
            index=idx,
            name=f"{other_data.header.data_type.name} ({other_data.header.unit})",
        )

        # get the average wind direction per-hour, per-month
        wd = self.wind_direction_series
        wind_directions = (
            (
                (
                    wd.groupby([idx.month, idx.hour], axis=0).apply(
                        circular_weighted_mean
                    )
                )
                % 360
            )
            .unstack()
            .T
        )

        _other_data = (
            other_data.groupby([idx.month, idx.hour], axis=0).mean().unstack().T
        )

        df = pd.concat(
            [wind_directions, _other_data],
            axis=1,
            keys=[wd.name, other_data.name],
        )
        df.index.name = "hour"
        df.columns.set_names(names=["variable", "month"], level=[0, 1], inplace=True)

        return df

    def windrose(
        self, other_data: HourlyContinuousCollection = None, directions: int = 36
    ) -> WindRose:
        if other_data is None:
            other_data = self.wind_speed_collection
        return WindRose(
            direction_data_collection=self.wind_direction_collection,
            analysis_data_collection=other_data,
            direction_count=directions,
        )

    def histogram(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] | int = 11,
        density: bool = False,
    ) -> pd.DataFrame:
        """Bin data by direction, returning counts for each direction.

        Args:
            directions (int, optional):
                The number of directions to use.
                Defaults to 8 directions.
            other_data (list[float], optional):
                A list of other data to bin by direction.
                If None, then wind speed will be used.
            other_bins (list[float] | int, optional):
                The other data bins to use for the histogram.
                Defaults to 11 bins between the min and max of the other data.
            density (bool, optional):
                If True, then return the probability density function.
                Defaults to False.
            directions_right (bool, optional):
                Whether to include the right edge of the direction bin.
                Defaults to True.
                If false, the left edge is included instead.
            other_right (bool, optional):
                Whether to include the right edge of the other bin.
                Defaults to True.
                If false, the left edge is included instead

        Returns:
            pd.DataFrame:
                A numpy array, containing the number or probability for each bin,
                for each direction bin.
        """

        # get other data
        if other_data is None:
            other_data = self.wind_speed
        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as wind data")

        # bin per direction
        dd = self._direction_binned_data(directions=directions)

        # create other intervals, and check for invalid edges
        cats = pd.cut(other_data, other_bins, right=True, include_lowest=True)
        if cats.categories[-1].right < max(other_data) or cats.categories[0].left > min(
            other_data
        ):
            raise ValueError(
                f"bin edges must be between {min(other_data)} and {max(other_data)} (inclusive)"
            )

        # iterate binned data, and bin
        new_d = {}
        for k, v in dd.items():
            dv = (
                pd.cut(v, other_bins, right=True, include_lowest=True)
                .value_counts()
                .values
            )
            new_d[k] = {str(i): float(j) for i, j in zip(*[cats.categories, dv])}

        df = pd.DataFrame(new_d).T

        # rename first column
        if float(str(df.columns[0]).split(",")[0][1:]) < min(other_data):
            r = str(df.columns[0]).split(",")[1]
            df.rename(columns={df.columns[0]: f"({min(other_data)},{r}"}, inplace=True)

        # name the index and columns to be used downstream
        df.index.name = "Wind Direction (degrees)"

        if density:
            return df / df.values.sum()

        return df

    # endregion: INSTANCE METHODS

    # region: VIZUALIZATION

    def plot_windprofile(
        self,
        ax: Axes = None,
        max_height: int = 30,
        log_law: bool = True,
        terrain_types: tuple[TerrainType] = None,
    ) -> Axes:
        reference_value = float(np.mean(self.wind_speed))

        if terrain_types is None:
            terrain_types = tuple([i for i in TerrainType])
        if not all(isinstance(tt, TerrainType) for tt in terrain_types):
            raise ValueError("terrain_types must be a list of TerrainType objects.")

        if ax is None:
            ax = plt.gca()

        heights = np.arange(0, max_height, 1)
        speeds = []
        for target_terrain in terrain_types:
            speeds.append(
                [
                    self.terrain_type.translate_wind_speed(
                        reference_value=reference_value,
                        reference_height=self.height_above_ground,
                        target_height=height,
                        log_law=log_law,
                        target_terrain_type=target_terrain,
                    )
                    for height in heights
                ]
            )

        # add the reference value to the plot
        ax.scatter(reference_value, self.height_above_ground, c="k")
        ax.plot(
            [reference_value] * 2,
            [0, self.height_above_ground],
            c="k",
            alpha=0.5,
            lw=2,
            ls="--",
        )
        ax.plot(
            [0, reference_value],
            [self.height_above_ground, self.height_above_ground],
            c="k",
            lw=2,
            ls="--",
            alpha=0.5,
        )
        ax.text(
            reference_value + 0.02,
            0.1,
            f"{reference_value:0.2f} m/s",
            ha="left",
            va="bottom",
        )

        for speed, tt in zip(*[speeds, terrain_types]):
            ax.plot(speed, heights, lw=2, label=tt.name)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Height (m)")
        ax.set_ylim(0, max_height)
        ax.set_xlim(0, np.array(speeds).max() + 0.1)
        ax.set_title(
            f"Wind Profiles (using {'log' if log_law else 'power'}-law)\n{self}"
        )
        ax.legend()

        return ax

    def plot_windmatrix(
        self,
        ax: Axes = None,
        show_values: bool = True,
        show_arrows: bool = True,
        other_data: HourlyContinuousCollection = None,
        **kwargs,
    ) -> Axes:
        """Create a plot showing the annual wind speed and direction bins
        using the month_time_average method.

        Args:
            ax (plt.Axes, optional):
                The axes to plot on. If None, the current axes will be used.
            show_values (bool, optional):
                Whether to show values in the cells.
                Defaults to True.
            show_arrows (bool, optional):
                Whether to show the directional arrows on each patch.
                Defaults to True.
            other_data: (pd.Series, optional):
                The other data to align with the wind direction and speed.
                Defaults to None which uses wind speed.
            **kwargs:
                Additional keyword arguments to pass to the pcolor function.
                title (str, optional):
                    A title for the plot. Defaults to None.

        Returns:
            plt.Axes:
                A matplotlib Axes object.

        """

        if ax is None:
            ax = plt.gca()

        if other_data is None:
            other_data = self.wind_speed_collection

        df = self.month_hour_mean_matrix(other_data=other_data)

        _wind_directions = df[df.columns.get_level_values(0)[0]]
        _other_data = df[df.columns.get_level_values(0)[-1]]

        cmap = kwargs.pop("cmap", "YlGnBu")
        vmin = kwargs.pop("vmin", _other_data.values.min())
        vmax = kwargs.pop("vmax", _other_data.values.max())
        unit = kwargs.pop("unit", other_data.header.unit)
        title = kwargs.pop("title", self.location.source)
        norm = kwargs.pop("norm", plt.Normalize(vmin=vmin, vmax=vmax, clip=True))
        mapper = kwargs.pop("mapper", ScalarMappable(norm=norm, cmap=cmap))
        pc = ax.pcolor(_other_data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        if show_arrows:
            _x, _y = np.array(angle_to_vector(_wind_directions.values))
            arrow_scale = 0.8
            ax.quiver(
                np.arange(1, 13, 1) - 0.5,
                np.arange(0, 24, 1) + 0.5,
                (_x * _other_data.values / 2) * arrow_scale,
                (_y * _other_data.values / 2) * arrow_scale,
                pivot="mid",
                fc="white",
                ec="black",
                lw=0.5,
                alpha=0.5,
            )

        if show_values:
            for _xx, col in enumerate(_wind_directions.values.T):
                for _yy, _ in enumerate(col.T):
                    local_value = _other_data.values[_yy, _xx]
                    cell_color = mapper.to_rgba(local_value)
                    text_color = contrasting_color(cell_color)
                    # direction text
                    ax.text(
                        _xx,
                        _yy,
                        f"{_wind_directions.values[_yy][_xx]:0.0f}°",
                        color=text_color,
                        ha="left",
                        va="bottom",
                        fontsize="xx-small",
                    )
                    # other_data text
                    ax.text(
                        _xx + 1,
                        _yy + 1,
                        f"{_other_data.values[_yy][_xx]:0.1f}{unit}",
                        color=text_color,
                        ha="right",
                        va="top",
                        fontsize="xx-small",
                    )

        # add title and colorbar
        ax.set_title(title)
        ax.set_xticks([i - 0.5 for i in range(1, 13, 1)])
        ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13, 1)])
        ax.set_yticks([i + 0.5 for i in range(24)])
        ax.set_yticklabels([f"{i:02d}:00" for i in range(24)])
        for label in ax.yaxis.get_ticklabels()[1::2]:
            label.set_visible(False)

        cb = plt.colorbar(pc, label=unit, pad=0.01)
        cb.outline.set_visible(False)

        return ax

    def plot_windrose(
        self,
        ax: plt.Axes = None,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] = 11,
        show_legend: bool = True,
        show_label: bool = False,
        remove_calm: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Create a wind rose showing wind speed and direction frequency.

        Args:
            ax (plt.Axes, optional):
                The axes to plot this chart on. Defaults to None.
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction.
                If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.
                If other data is None, then the default Beaufort bins will be used,
                otherwise 11 evenly spaced bins will be used.
            show_legend (bool, optional):
                Whether to show the legend.
                Defaults to True.
            show_label (bool, optional):
                Whether to show the bin labels.
                Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to the plot.

        Returns:
            plt.Axes: The axes object.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        calm_wind_speeds = np.array(self.calm_mask(threshold=0.01))

        if other_data is None:
            other_data = self.wind_speed

        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as wind data")

        # obtain kwarg data
        cmap = kwargs.pop("cmap", "YlGnBu")
        title = kwargs.pop(
            "title",
            f"{self.location.source}"
            + (
                f" ({sum(calm_wind_speeds) / len(self):0.2%} calm)"
                if remove_calm
                else ""
            ),
        )

        # create grouped data for plotting
        binned = self.filter_by_boolean_mask((~calm_wind_speeds).tolist()).histogram(
            directions=directions,
            other_data=np.array(other_data)[~calm_wind_speeds].tolist(),
            other_bins=other_bins,
            density=True,
        )

        ylim = kwargs.pop("ylim", (0, max(binned.sum(axis=1))))
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple of length 2.")

        # obtain colors
        if not isinstance(cmap, Colormap):
            cmap = plt.get_cmap(cmap)
        colors = [to_hex(cmap(i)) for i in np.linspace(0, 1, len(binned.columns))]

        # create the patches
        theta_width = np.deg2rad(360 / directions)
        patches = []
        color_list = []
        x = theta_width / 2
        for _, data_values in binned.iterrows():
            y = 0
            for n, val in enumerate(data_values.values):
                patches.append(
                    Rectangle(
                        xy=(x, y),
                        width=theta_width,
                        height=val,
                        alpha=1,
                    )
                )
                color_list.append(colors[n])
                y += val
            if show_label:
                ax.text(x, y, f"{y:0.1%}", ha="center", va="center", fontsize="x-small")
            x += theta_width
        local_cmap = ListedColormap(np.array(color_list).flatten())
        pc = PatchCollection(patches, cmap=local_cmap)
        pc.set_array(np.arange(len(color_list)))
        ax.add_collection(pc)

        # construct legend
        if show_legend:
            handles = [
                Patch(color=colors[n], label=col)
                for n, col in enumerate(binned.columns)
            ]
            _ = ax.legend(
                handles=handles,
                bbox_to_anchor=(1.1, 0.5),
                loc="center left",
                ncol=1,
                borderaxespad=0,
                frameon=False,
                fontsize="small",
                title=binned.columns.name,
                title_fontsize="small",
            )

        # set y-axis limits
        ax.set_ylim(ylim)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

        # format the plot
        ax.set_title(title)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.spines["polar"].set_visible(False)
        ax.grid(True, which="both", ls="--", zorder=0, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.get_yticklabels(), fontsize="small")
        ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
        ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
        ax.set_xticks(
            np.radians(
                (
                    22.5,
                    45,
                    67.5,
                    112.5,
                    135,
                    157.5,
                    202.5,
                    225,
                    247.5,
                    292.5,
                    315,
                    337.5,
                )
            ),
            minor=True,
        )
        ax.set_xticklabels(
            (
                "NNE",
                "NE",
                "ENE",
                "ESE",
                "SE",
                "SSE",
                "SSW",
                "SW",
                "WSW",
                "WNW",
                "NW",
                "NNW",
            ),
            minor=True,
            **{"fontsize": "x-small"},
        )

        return ax

    def plot_windhistogram(
        self,
        ax: plt.Axes = None,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] | int = 11,
        density: bool = False,
        cmap: str | Colormap = "YlGnBu",
        show_values: bool = True,
        vmin: float = None,
        vmax: float = None,
    ) -> plt.Axes:
        """Plot a 2D-histogram for a collection of wind speeds and directions.

        Args:
            ax (plt.Axes, optional):
                The axis to plot results on. Defaults to None.
            directions (int, optional):
                The number of directions to use. Defaults to 36.
            other_data (list[float], optional):
                A list of other data to bin by direction. If None, then wind speed will be used.
            other_bins (list[float]):
                The other data bins to use for the histogram. These bins are right inclusive.
            density (bool, optional):
                If True, then return the probability density function. Defaults to False.
            cmap (str | Colormap, optional):
                The colormap to use. Defaults to "YlGnBu".
            show_values (bool, optional):
                Whether to show values in the cells. Defaults to True.
            vmin (float, optional):
                The minimum value for the colormap. Defaults to None.
            vmax (float, optional):
                The maximum value for the colormap. Defaults to None.

        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        # FIXME - This method kind-of works, but needs to be fixed to poperly work!

        if ax is None:
            ax = plt.gca()

        hist = self.histogram(
            directions=directions,
            other_data=other_data,
            other_bins=other_bins,
            density=density,
        )

        vmin = hist.values.min() if vmin is None else vmin
        vmax = hist.values.max() if vmax is None else vmax
        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = ScalarMappable(norm=norm, cmap=cmap)

        _xticks = np.roll(hist.index, 1)
        _values = np.roll(hist.values, 1, axis=0).T

        pc = ax.pcolor(_values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(0.5, len(hist.index), 1), labels=_xticks, rotation=90)
        ax.set_xlabel(hist.index.name)
        ax.set_yticks(np.arange(0.5, len(hist.columns), 1), labels=hist.columns)
        ax.set_ylabel(hist.columns.name)

        cb = plt.colorbar(pc, pad=0.01, label="Density" if density else "Count")
        if density:
            cb.ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        cb.outline.set_visible(False)

        ax.set_title(self.location.source)

        if show_values:
            for _xx, row in enumerate(_values):
                for _yy, col in enumerate(row):
                    if (col * 100).round(1) == 0:
                        continue
                    cell_color = mapper.to_rgba(col)
                    text_color = contrasting_color(cell_color)
                    ax.text(
                        _yy + 0.5,
                        _xx + 0.5,
                        f"{col:0.2%}" if density else col,
                        color=text_color,
                        ha="center",
                        va="center",
                        fontsize="xx-small",
                    )

        return ax

    def plot_densityfunction(
        self,
        ax: plt.Axes = None,
        speed_bins: list[float] | int = 11,
        percentiles: tuple[float] = (0.5, 0.95),
        function: str = "pdf",
        ylim: tuple[float] = None,
    ) -> plt.Axes:
        """Create a histogram showing wind speed frequency.

        Args:
            ax (plt.Axes, optional):
                The axes to plot this chart on. Defaults to None.
            speed_bins (list[float], optional):
                The wind speed bins to use for the histogram. These bins are right inclusive.
            percentiles (tuple[float], optional):
                The percentiles to plot. Defaults to (0.5, 0.95).
            function (str, optional):
                The function to use. Either "pdf" or "cdf". Defaults to "pdf".
            ylim (tuple[float], optional):
                The y-axis limits. Defaults to None.

        Returns:
            plt.Axes: The axes object.
        """

        # FIXME - this method kind of works, but could be done better

        if function not in ["pdf", "cdf"]:
            raise ValueError('function must be either "pdf" or "cdf".')

        if ax is None:
            ax = plt.gca()

        ax.set_title(
            f"{str(self)}\n{'Probability Density Function' if function == 'pdf' else 'Cumulative Density Function'}"
        )

        self.wind_speed_series.plot.hist(
            ax=ax,
            density=True,
            bins=speed_bins,
            cumulative=True if function == "cdf" else False,
        )

        for percentile in percentiles:
            x = np.quantile(self.wind_speed_series, percentile)
            ax.axvline(x, 0, 1, ls="--", lw=1, c="black", alpha=0.5)
            ax.text(
                x + 0.05,
                0,
                f"{percentile:0.0%}\n{x:0.2f}m/s",
                ha="left",
                va="bottom",
            )

        ax.set_xlim(0, ax.get_xlim()[-1])
        if ylim:
            ax.set_ylim(ylim)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Frequency")

        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))

        return ax

    # endregion: VIZUALIZATION
