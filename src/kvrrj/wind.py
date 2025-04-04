import calendar
import json
import urllib
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from honeybee.config import folders as hb_folders
from ladybug.datatype.angle import WindDirection
from ladybug.datatype.speed import WindSpeed
from ladybug.dt import Date
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.sunpath import DateTime, Location
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
    _all_timezones_same,
    _is_datetime_location_aligned,
    _is_location_time_zone_valid_for_location,
    average_location,
    get_timezone_str_from_location,
    get_tzinfo_from_location,
    get_utc_offset_from_location,
)
from kvrrj.util import (
    _datetimes_span_at_least_1_year,
    _is_iterable_1d,
    _is_iterable_single_dtype,
    _is_leap_year,
    wind_speed_at_height,
)
from kvrrj.viz.color import contrasting_color, to_hex

from .logging import CONSOLE_LOGGER  # noqa: F401


@dataclass
class Wind:
    """An object containing solar data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (list[datetime]):
            An iterable of datetime-like objects.
        wind_speed (list[float]):
            A list of wind speeds, in m/s.
        wind_direction (list[float]):
            A list of wind directions, in degrees clockwise from north (at 0).

    """

    # NOTE - BE STRICT WITH THE TYPING!
    # NOTE - Conversions happen in class methods.
    # NOTE - Validation happens at instantiation.

    location: Location
    datetimes: list[datetime]
    wind_speed: list[float]
    wind_direction: list[float]
    height_above_ground: float

    # region: DUNDER METHODS

    def __post_init__(self):
        """Check for validation of the inputs."""

        # location validation
        if not isinstance(self.location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if self.location.source is None:
            warnings.warn(
                'The source field of the Location input is None. This means that things are a bit ambiguous! A default value of "somewhere ... be more spceific!" has been added.'
            )
            self.location.source = "UnknownSource"
        if not _is_location_time_zone_valid_for_location(self.location):
            raise ValueError(
                f"The time zone of the location ({self.location.time_zone}) does not match the time zone of the lat/lon ({self.location.latitude}, {self.location.longitude})."
            )

        # datetimes validation
        if not _is_iterable_1d(self.datetimes):
            raise ValueError("datetimes must be a 1D iterable.")
        if not _is_iterable_single_dtype(self.datetimes, datetime):
            raise ValueError("datetimes must be a list of datetime-like objects.")

        # check datetime timezone awareness
        location_tzinfo = get_tzinfo_from_location(self.location)
        dts = []
        for dt in self.datetimes:
            if dt.tzinfo is None:
                warnings.warn(
                    "Timezone information is missing from datetimes. This will be obtained from the Location and added as a UTC offset to the datetimes."
                )
                dts.append(dt.replace(tzinfo=location_tzinfo))
            else:
                if not _is_datetime_location_aligned(dt, self.location):
                    raise ValueError(
                        "The datetimes' timezone must match the location's time_zone."
                    )
                dts.append(dt)

        # check timezines are the same
        if not _all_timezones_same([d.tzinfo for d in dts]):
            raise ValueError(
                "All datetimes must have the same timezone. This is not the case."
            )
        self.datetimes = dts

        # wind data validation
        array_names = [
            "wind_speed",
            "wind_direction",
        ]
        for name in array_names:
            if len(getattr(self, name)) != len(self.datetimes):
                raise ValueError(
                    f"{name} must be the same length as datetimes. {len(getattr(self, name))} != {len(self.datetimes)}."
                )
            if not _is_iterable_1d(getattr(self, name)):
                raise ValueError(f"{name} must be a 1D iterable.")
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

        # height above ground validation
        if not isinstance(self.height_above_ground, (int, float)):
            raise ValueError("height_above_ground must be a number.")
        if self.height_above_ground < 0.1:
            raise ValueError(
                "height_above_ground must be greater than or equal to 0.1."
            )

    def __len__(self) -> int:
        return len(self.datetimes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} data from {self.location.source} (n={len(self)})"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(
            (
                self.location,
                tuple(self.datetimes),
                tuple(self.wind_speed),
                tuple(self.wind_direction),
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
        )

    # endregion: DUNDER METHODS

    # region: STATIC METHODS

    @staticmethod
    def _direction_midpoints(directions: int = 36) -> list[float]:
        """Calculate the midpoints for a given number of directions.

        Args:
            directions (int):
                The number of directions to calculate midpoints for.

        Returns:
            list[float]:
                A list of midpoints.
        """
        if directions <= 2:
            raise ValueError("directions must be > 2.")
        return np.linspace(0, 360, directions + 1)[:-1].tolist()

    @staticmethod
    def _direction_bins(
        directions: int = 36,
    ) -> list[tuple[float, float]]:
        """Calculate the bin edges for a given number of directions.

        Args:
            directions (int):
                The number of directions to calculate bin edges for.

        Returns:
            list[tuple[float, float]]:
                A list of bin edges.
        """

        edges = [
            i + (360 / directions / 2) for i in Wind._direction_midpoints(directions)
        ]
        edges.insert(0, edges[-1])
        return [
            tuple(i)
            for i in np.lib.stride_tricks.sliding_window_view(edges, 2).tolist()
        ]

    # endregion: STATIC METHODS

    # region: PROPERTIES

    @property
    def dates(self) -> list[date]:
        return sorted(list(set([dt.date() for dt in self.datetimes])))

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
        return pd.DatetimeIndex(self.datetimes)

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
    def freq(self) -> str:
        """Return the inferred frequency of the datetimes associated with this object."""
        freq = pd.infer_freq(self.datetimes)
        if freq is None:
            return "inconsistent"
        return freq

    @property
    def pd_datetimeindex(self) -> pd.DatetimeIndex:
        """Get the datetimes as a pandas DateTimeIndex."""
        return pd.to_datetime(self.datetimes)

    @property
    def ws(self) -> pd.Series:
        """Convenience accessor for wind speeds as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_speeds, index=self.index, name="Wind Speed (m/s)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def wd(self) -> pd.Series:
        """Convenience accessor for wind directions as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_directions, index=self.index, name="Wind Direction (degrees)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def uv(self) -> list[tuple[float, float]]:
        """Return the U and V wind components in m/s."""
        return angle_to_vector(self.wd)

    def mean_speed(self, include_zero: bool = True) -> float:
        """Return the mean wind speed for this object.

        Args:
            include_zero (bool, optional):
                If True, include calm wind speeds in the mean.
                Defaults to True.

        Returns:
            float:
                Mean wind speed.

        """
        if include_zero:
            return self.ws[self.ws > 0].mean()
        return self.ws.mean()

    @property
    def mean_uv(self) -> list[float, float]:
        """Calculate the average U and V wind components in m/s.

        Returns:
            list[float, float]:
                A tuple containing the average U and V wind components.
        """
        return self.uv.mean().tolist()

    @property
    def mean_direction(self) -> tuple[float, float]:
        """Calculate the average direction for this object."""
        return angle_clockwise_from_north(self.mean_uv)

    @property
    def min_speed(self, include_zero: bool = False) -> float:
        """Return the min wind speed for this object."""
        if include_zero:
            return self.ws.min()
        return self.ws[self.ws > 0].min()

    @property
    def max_speed(self) -> float:
        """Return the max wind speed for this object."""
        return self.ws.max()

    @property
    def median_speed(self, include_zero: bool = False) -> float:
        """Return the median wind speed for this object."""
        if include_zero:
            return self.ws.median()
        return self.ws[self.ws > 0].median()

    # endregion: PROPERTIES

    # region: CLASS METHODS

    @classmethod
    def from_openmeteo(
        cls,
        location: Location,
        start_date: str | date,
        end_date: str | date,
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
        )

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "Wind":
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
        wind_speed_column: str = None,
        wind_direction_column: str = None,
        height_above_ground: float = 10,
    ) -> "Wind":
        """Create this object from a DataFrame.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing the solar data.
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
                "Wind Directions (degrees)",
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
        )

    @classmethod
    def from_average(cls, objects: list["Wind"], weights: list[float] = None) -> "Wind":
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""

        # validation
        if not _is_iterable_1d(objects):
            raise ValueError("objects must be a 1D list of Wind objects.")
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
        df_ws = pd.concat([i.ws for i in objects], axis=1).dropna()
        df_wd = pd.concat([i.wd for i in objects], axis=1).dropna()

        # construct the weighted means
        wd_avg = np.array(
            [circular_weighted_mean(i, weights) for _, i in df_wd.iterrows()]
        )
        ws_avg = np.average(df_ws, axis=1, weights=weights)

        # construct the avg height above ground
        avg_height_above_ground = np.average(
            [i.height_above_ground for i in objects], weights=weights
        )

        # return the new averaged object
        return cls(
            wind_speeds=ws_avg.tolist(),
            wind_directions=wd_avg.tolist(),
            datetimes=objects[0].datetimes,
            height_above_ground=avg_height_above_ground,
            location=avg_location,
        )

    @classmethod
    def from_uv(
        cls,
        u: list[float],
        v: list[float],
        location: Location,
        datetimes: list[datetime],
        height_above_ground: float = 10,
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
            wind_speeds=wind_speed.tolist(),
            wind_directions=wind_direction.tolist(),
            datetimes=datetimes,
            height_above_ground=height_above_ground,
            location=location,
        )

    # endregion: CLASS METHODS

    # region: INSTANCE METHODS

    def calm(self, threshold: float = 0.1) -> float:
        """Return the proportion of timesteps "calm" (i.e. less than or equal
        to the threshold).

        Args:
            threshold (float, optional):
                The threshold for calm wind speeds. Defaults to 0.1.

        Returns:
            float:
                The proportion of calm instances.
        """
        s = self.ws
        return float((s <= threshold).sum() / len(s))

    def percentile(self, percentile: float | list[float]) -> dict[float, float]:
        """Calculate the wind speed at the given percentiles.

        Args:
            percentile (list[float]):
                The percentile/s to calculate.

        Returns:
            dict[float, float]:
                Wind speed/s at the given percentile/s.
        """
        percentile = np.atleast_1d(percentile)
        return dict(zip(percentile.tolist(), self.ws.quantile(percentile)))

    def to_height(
        self,
        target_height: float,
        terrain_roughness_length: float = 1,
        log_function: bool = True,
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
        ws = wind_speed_at_height(
            reference_value=self.ws,
            reference_height=self.height_above_ground,
            target_height=target_height,
            terrain_roughness_length=terrain_roughness_length,
            log_function=log_function,
        )
        return Wind(
            wind_speeds=ws.tolist(),
            wind_directions=self.wind_direction,
            datetimes=self.datetimes,
            height_above_ground=target_height,
            source=f"{self.location.source} translated to {target_height}m",
        )

    def apply_directional_factors(
        self, directions: int, factors: tuple[float], right: bool = True
    ) -> "Wind":
        """Adjust wind speed values by a set of factors per direction.
        Factors start at north, and move clockwise.

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
            right (bool, optional):
                Whether to include the right edge of the bin.
                Defaults to True.
                If false, the left edge is included.

        Returns:
            Wind:
                An adjusted Wind object.
        """
        if len(factors) != directions:
            raise ValueError(
                f"number of factors ({len(factors)}) must equal number of directions ({directions})"
            )

        _, binned_directions = self._bin_direction_data(
            directions=directions, right=right
        )
        factor_lookup = dict(zip(self._direction_bins(directions=directions), factors))
        mapped_factors = [*map(factor_lookup.get, binned_directions)]

        return Wind(
            wind_speed=(np.array(mapped_factors) * self.wind_speed).tolist(),
            wind_direction=self.wind_direction,
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
            source=f"{self.source} (adjusted by factors {factor_lookup})",
        )

    def prevailing(
        self,
        directions: int = 36,
        n: int = 1,
        as_cardinal: bool = False,
    ) -> tuple[float] | tuple[str]:
        """Calculate the prevailing wind direction/s for this object.

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.
                Defaults to 8 directions.
            n (int, optional):
                The number of prevailing directions to return. Default is 1.
            as_cardinal (bool, optional):
                If True, then return the prevailing directions as cardinal directions.
                Defaults to False.

        Returns:
            list[float] | list[str]:
                A list of wind directions.
        """

        _, binned = self._bin_direction_data(directions=directions)

        prevailing_angles = pd.Series(binned).value_counts().index[:n]

        if as_cardinal:
            card = []
            for i in prevailing_angles:
                if i[0] < i[1]:
                    card.append(
                        cardinality(direction_angle=np.mean(i), directions=directions)
                    )
                else:
                    card.append(cardinality(direction_angle=0, directions=directions))
            return card

        return tuple(prevailing_angles.tolist())

    def wind_matrix(self, other_data: list[float] = None) -> pd.DataFrame:
        """Calculate average wind direction and speed (or aligned other data)
        for each month and hour of in the Wind object.

        Args:
            other_data (pd.Series, optional):
                The other data to calculate the matrix for.

        Returns:
            pd.DataFrame:
                A DataFrame containing average other_data and direction for each
                month and hour of day.
        """

        # ensure data is suitable for matrixisation

        if other_data is None:
            other_data = self.wind_speed

        if len(other_data) != len(self):
            raise ValueError("other_data must be the same length as the wind data.")

        # convert other data to a series
        other_data = pd.Series(other_data, index=self.datetimeindex)

        # get the average wind direction per-hour, per-month
        wind_directions = (
            (
                (
                    self.wd.groupby(
                        [self.datetimeindex, self.datetimeindex], axis=0
                    ).apply(circular_weighted_mean)
                )
                % 360
            )
            .unstack()
            .T
        )
        wind_directions.columns = [
            calendar.month_abbr[i] for i in wind_directions.columns
        ]
        _other_data = (
            other_data.groupby([other_data.index.month, other_data.index.hour], axis=0)
            .mean()
            .unstack()
            .T
        )
        _other_data.columns = [calendar.month_abbr[i] for i in _other_data.columns]

        df = pd.concat(
            [wind_directions, _other_data], axis=1, keys=["direction", "other"]
        )
        df.index.name = "hour"

        return df

    # region: SAMPLING

    def _bin_direction_data(
        self, directions: int = 36, right: bool = True
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Bin the wind direction data into a set of directions.

        Args:
            directions (int):
                The number of directions to bin the data into.
            right (bool, optional):
                Whether to include the right edge of the bin.
                Defaults to True.
                If false, the left edge is included.

        Returns:
            tuple[list[tuple[float, float]], list[tuple[float, float]]]:
                - The possible bin values
                - A list of the bins associated with each wind direction.
        """

        bin_edges = np.concat(
            [[-1], np.unique(self._direction_bins(directions)), [361]]
        ).tolist()

        cat = pd.cut(self.wind_directions, bins=bin_edges, right=right)

        cat_bins = [(float(i.left), float(i.right)) for i in cat.categories]
        cat_cat = [(float(i.left), float(i.right)) for i in cat]

        d = dict(zip(cat_bins, cat_bins))
        d[cat_bins[0]] = (cat_bins[-1][0], cat_bins[0][1])
        d[cat_bins[-1]] = (cat_bins[-1][0], cat_bins[0][1])

        return (list(d.values())[:-1], [d[i] for i in cat_cat])

    def _bin_other_data(
        self,
        other_data: list[float] = None,
        other_bins: list[float] | int = 11,
        right: bool = True,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Bin the "other" data into a set of bins.

        Args:
            other_data (list[float]):
                The "other" data to bin.
                Defaults to the wind speeds associated with the wind directions.
            other_bins (list[float] | int):
                The bin edges to use.
                Defaults to 11 bins between the min and max of the "other" data.
            right (bool, optional):
                Whether to include the right edge of the bin.
                Defaults to True.
                If false, the left edge is included instead.

        Returns:
            tuple[list[tuple[float, float]], list[tuple[float, float]]]:
                - The possible bin values
                - A list of the bins associated with each value in other_data.
        """

        if other_data is None:
            other_data = self.wind_speeds

        if len(other_data) != len(self.wind_directions):
            raise ValueError("other_data must be the same length as the wind data.")

        if not isinstance(other_bins, int):
            if max(other_bins) < max(other_data):
                raise ValueError(
                    f"Bin edges must be greater than the maximum value in the data ({max(other_bins)} < {max(other_data)})."
                )
            if min(other_bins) > min(other_data):
                raise ValueError(
                    f"Bin edges must be less than the minimum value in the data ({min(other_bins)} > {min(other_data)})."
                )

        cat = pd.cut(other_data, bins=other_bins, right=right)
        cat_bins = [(float(i.left), float(i.right)) for i in cat.categories]
        cat_cat = [(float(i.left), float(i.right)) for i in cat]

        d = dict(zip(cat_bins, cat_bins))
        d[cat_bins[0]] = (min(other_data), cat_bins[0][1])
        d[cat_bins[-1]] = (cat_bins[-1][0], max(other_data))

        return (list(d.values()), [d[i] for i in cat_cat])

    def histogram(
        self,
        directions: int = 36,
        other_data: list[float] = None,
        other_bins: list[float] | int = 11,
        density: bool = False,
        directions_right: bool = True,
        other_right: bool = True,
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
        # bin data
        direction_categories, direction_bin_tuples = self._bin_direction_data(
            directions=directions, right=directions_right
        )
        other_categories, other_bin_tuples = self._bin_other_data(
            other_data=other_data, other_bins=other_bins, right=other_right
        )

        # create table
        idx = self.index
        df = pd.concat(
            [
                pd.Series(direction_bin_tuples, index=idx, name="direction"),
                pd.Series(other_bin_tuples, index=idx, name="other"),
            ],
            axis=1,
        )

        # pivot!
        df = (
            df.groupby([df.columns[0], df.columns[1]], observed=True)
            .value_counts()
            .unstack()
            .fillna(0)
            .astype(int)
        )

        # add back missing categories
        df = df.reindex(other_categories, axis=1, fill_value=0).reindex(
            direction_categories, axis=0, fill_value=0
        )

        # move last row to first
        df = pd.concat([df.iloc[-1:], df.iloc[:-1]])

        # normalize
        if density:
            return df / df.values.sum()

        return df

    # endregion: SAMPLING

    # region: FILTERING

    def filter_by_boolean_mask(self, mask: tuple[bool], source: str = None) -> "Wind":
        """Filter the current object by a boolean mask.

        Args:
            mask (tuple[bool]):
                A boolean mask to filter the current object.
            source_ (str, optional):
                The source of the new object.
                Defaults to None, which will append " (filtered)" to the current source.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if len(mask) != len(self):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )

        if sum(mask) == 0:
            raise ValueError("No data remains within the given boolean filters.")

        if source is None:
            source = f"{self.source} (filtered)"

        return Wind(
            wind_speeds=np.array(self.wind_speeds)[mask].tolist(),
            wind_directions=np.array(self.wind_directions)[mask].tolist(),
            datetimes=np.array(self.datetimes)[mask].tolist(),
            height_above_ground=self.height_above_ground,
            source=source,
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
                A dataset describing historic wind speed and direction relationship.
        """

        possible_datetimes = pd.to_datetime(analysis_period.datetimes)
        lookup = pd.DataFrame(
            {
                "month": possible_datetimes.month,
                "day": possible_datetimes.day,
                "hour": possible_datetimes.hour,
            }
        )
        idx = self.ws.index
        reference = pd.DataFrame(
            {
                "month": idx.month,
                "day": idx.day,
                "hour": idx.hour,
            }
        )
        mask = reference.isin(lookup).all(axis=1)

        return self.filter_by_boolean_mask(
            mask,
            source=f"{self.source} (filtered to {_analysis_period_to_string(analysis_period)})",
        )

    def filter_by_time(
        self,
        months: tuple[float] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        days: tuple[float] = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ),
        hours: tuple[int] = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ),
        years: tuple[int] = None,
    ) -> "Wind":
        """Filter the current object by month and hour.

        Args:
            months (list[int], optional):
                A list of months.
                Defaults to all possible months.
            days (list[int], optional):
                A list of days.
                Defaults to all possible days.
            hours (list[int], optional):
                A list of hours.
                Defaults to all possible hours.
            years (tuple[int], optional):
                A list of years to include.
                Default to all years.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        # convert the datetimes to a pandas datetime index
        idx = pd.DatetimeIndex(self.datetimes)

        # construct the masks
        year_mask = (
            np.ones_like(idx).astype(bool) if years is None else idx.year.isin(years)
        )
        month_mask = idx.month.isin(months)
        day_mask = idx.day.isin(days)
        hour_mask = idx.hour.isin(hours)
        mask = np.all([year_mask, month_mask, day_mask, hour_mask], axis=0)

        filtered_by = []
        if sum(year_mask) != 0:
            filtered_by.append("year")
        if sum(month_mask) != 0:
            filtered_by.append("month")
        if sum(day_mask) != 0:
            filtered_by.append("day")
        if sum(hour_mask) != 0:
            filtered_by.append("hour")
        filtered_by = ", ".join(filtered_by)
        source = f"{self.source} (filtered {filtered_by})"

        return self.filter_by_boolean_mask(
            mask,
            source=source,
        )

    def filter_by_direction(
        self,
        left_angle: float = 0,
        right_angle: float = 360,
        right: bool = True,
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
            right (bool, optional):
                Indicates whether the interval includes the rightmost edge or not.
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

        wd = self.wd.values

        if left_angle > right_angle:
            if right:
                mask = (wd > left_angle) | (wd <= right_angle)
            else:
                mask = (wd >= left_angle) | (wd < right_angle)
        else:
            if right:
                mask = (wd > left_angle) & (wd <= right_angle)
            else:
                mask = (self.wd >= left_angle) & (self.wd < right_angle)

        source = f"{self.source} (filtered by direction {'(' if right else '['}{left_angle}-{right_angle}{']' if right else ')'})"

        return self.filter_by_boolean_mask(mask, source=source)

    def filter_by_speed(
        self, min_speed: float = 0, max_speed: float = np.inf, right: bool = True
    ) -> "Wind":
        """Filter the current object by wind speed, based on given low-high limit values.

        Args:
            min_speed (float):
                The lowest speed to include. Values below this wil be removed.
                Defaults to 0.
            max_speed (float):
                The highest speed to include. Values above this wil be removed.
                Defaults to np.inf.
            right (bool, optional):
                Include values that are exactly the min or max speeds.
                Defaults to True.

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

        if right:
            mask = (self.ws > min_speed) & (self.ws <= max_speed)
        else:
            mask = (self.ws >= min_speed) & (self.ws < max_speed)

        # get the new min/max speeds for labelling the source max
        source = f"{self.source} (filtered by speed {min_speed}m/s-{min(self.ws[mask].max(), max_speed)}m/s)"

        return self.filter_by_boolean_mask(mask, source=source)

    # endregion: FILTERING

    # endregion: INSTANCE METHODS

    # region: VIZUALIZATION

    def plot_windmatrix(
        self,
        ax: Axes = None,
        show_values: bool = True,
        show_arrows: bool = True,
        other_data: list[float] = None,
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
            other_data = self.wind_speeds
            kwargs["vmin"] = kwargs.get("vmin", 0)
            kwargs["unit"] = kwargs.get("unit", "m/s")

        df = self.wind_matrix(other_data=other_data)
        _wind_directions = df["direction"]
        _other_data = df["other"]

        cmap = kwargs.pop("cmap", "YlGnBu")
        vmin = kwargs.pop("vmin", _other_data.values.min())
        vmax = kwargs.pop("vmax", _other_data.values.max())
        unit = kwargs.pop("unit", "")
        title = kwargs.pop("title", self.source)
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

        # obtain kwarg data
        cmap = kwargs.pop("cmap", "YlGnBu")
        title = kwargs.pop("title", self.source)

        # create grouped data for plotting
        binned = self.histogram(
            directions=directions,
            other_data=other_data,
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
                Patch(color=colors[n], label=f"{i} to {j}")
                for n, (i, j) in enumerate(binned.columns.values)
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
        other_bins: list[float] = None,
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

        # TODO - update the method below

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

        ax.set_title(self.source)

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

        # TODO - update method below

        if function not in ["pdf", "cdf"]:
            raise ValueError('function must be either "pdf" or "cdf".')

        if ax is None:
            ax = plt.gca()

        ax.set_title(
            f"{str(self)}\n{'Probability Density Function' if function == 'pdf' else 'Cumulative Density Function'}"
        )

        self.ws.plot.hist(
            ax=ax,
            density=True,
            bins=speed_bins,
            cumulative=True if function == "cdf" else False,
        )

        for percentile in percentiles:
            x = np.quantile(self.ws, percentile)
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

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(visible=True, which="major", axis="both", ls="--", lw=1, alpha=0.25)

        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))

        return ax

    # endregion: VIZUALIZATION
