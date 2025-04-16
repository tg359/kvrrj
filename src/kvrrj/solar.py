"""Methods for handling solar data. This module relies heavily on numpy, pandas, and ladybug."""

# TODO - Shade benefit calc (on window) - https://github.com/ladybug-tools/ladybug-grasshopper/blob/master/ladybug_grasshopper/src/LB%20Shade%20Benefit.py
# TODO - Thermal shade benefit calc - https://github.com/ladybug-tools/ladybug-grasshopper/blob/master/ladybug_grasshopper/src//LB%20Thermal%20Shade%20Benefit.py
# TODO - PV calc from pvlib
# TODO - PV with shade objects (from sky matrix, or get total incident radiation on surface using Radiance and then feed into PVLib)
# TODO - Use DirectSun/RadiationStudy to calculate shadedness of a point given context meshes
# todo - glare risk for aperture in direction

import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any

import ephem
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.base import DataTypeBase
from ladybug.datatype.energyintensity import (
    DiffuseHorizontalRadiation,
    DirectNormalRadiation,
    GlobalHorizontalRadiation,
)
from ladybug.dt import Date, DateTime
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.sunpath import Location, Sun, Sunpath
from ladybug.viewsphere import ViewSphere
from ladybug.wea import Wea
from ladybug_geometry.geometry3d import (
    Face3D,
    Mesh3D,
    Plane,
    Point3D,
    Polyline3D,
)

# pylint: enable=E0401
from ladybug_radiance.config import folders as lbr_folders
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.study.radiation import RadiationStudy
from ladybug_radiance.visualize.raddome import RadiationDome
from ladybug_radiance.visualize.radrose import RadiationRose
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.ticker import MultipleLocator
from pvlib.irradiance import campbell_norman, get_extra_radiation
from pvlib.location import Location as PVLocation

from .geometry.to_shapely import to_shapely
from .geometry.util import (
    _create_azimuth_mesh,
    angle_clockwise_from_north,
    vector3d_to_azimuth_altitude,
)
from .ladybug.analysisperiod import (
    _analysis_period_to_string,
    _iterable_datetimes_to_lb_analysis_period,
)
from .ladybug.location import (
    _is_location_time_zone_valid_for_location,
    average_location,
    get_tzinfo_from_location,
)
from .util import (
    _are_iterables_same_length,
    _datetimes_span_at_least_1_year,
    _filter_kwargs_by_allowable,
    _is_iterable_single_dtype,
    _is_leap_year,
)
from .viz.color import contrasting_color


class RadiationType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()
    REFLECTED = auto()


def _lb_location_to_pvlib_location(location: Location) -> PVLocation:
    """Convert a ladybug Location to a pvlib Location."""

    if not isinstance(location, Location):
        raise ValueError("location must be a ladybug Location object.")
    # check time zone is valid for location
    if not _is_location_time_zone_valid_for_location(location):
        raise ValueError(
            f"The time zone of the location ({location.time_zone}) does not match the time zone of the lat/lon ({location.latitude}, {location.longitude})."
        )

    return PVLocation(
        latitude=location.latitude,
        longitude=location.longitude,
        tz=location.time_zone,
        altitude=location.elevation,
        name=location.source,
    )


def sunrise_sunset(dates: list[date], location: Location) -> pd.DataFrame:
    """Get sunrise and sunset times for the given dates in the given location.

    Args:
        dates (list[date]):
            A list of dates for which to get sunrise times.
        location (Location):
            A ladybug Location object.

    Returns:
        pd.DataFrame:
            A DataFrame with sunrise and sunset times for each date.
    """

    # check inputs
    if not all(isinstance(d, date) for d in dates):
        raise ValueError("All items in dates must be date objects.")
    if not isinstance(location, Location):
        raise ValueError("Location must be a ladybug Location object.")
    if not _is_iterable_single_dtype(dates, date):
        raise ValueError("dates must be a 1D list of date objects.")

    sunpath = Sunpath.from_location(location)

    light_types = ["actual", "apparent", "civil", "nautical", "astronomical"]
    depression_angles = [0.5334, 0.833, 6, 12, 18]

    # get solstice and equinox dates, to determine whether a date is considered winter or summer
    years = np.unique([i.year for i in dates])
    d = {}
    for year in years:
        d[year] = {
            "vernal equinox": ephem.next_vernal_equinox(str(year)).datetime().date(),
            "summer solstice": ephem.next_summer_solstice(str(year)).datetime().date(),
            "autumnal equinox": ephem.next_autumnal_equinox(str(year))
            .datetime()
            .date(),
            "winter solstice": ephem.next_winter_solstice(str(year)).datetime().date(),
        }
    se = pd.DataFrame(d).T

    if location.latitude > 0:
        # northern hemisphere
        winter_dates = (dates > se["autumnal equinox"].values[0]) | (
            dates < se["vernal equinox"].values[0]
        )
    else:
        # southern hemisphere
        winter_dates = (dates > se["vernal equinox"].values[0]) | (
            dates < se["autumnal equinox"].values[0]
        )

    # calculate sunrsies and sets
    kk = {}
    for is_winter, dt in zip(*[winter_dates, dates]):
        day_start = pd.to_datetime(dt).to_pydatetime()
        day_end = (pd.to_datetime(dt) + timedelta(days=1)).to_pydatetime()
        rr = []
        for light_type, depression in zip(*[light_types, depression_angles]):
            # calculate sunrise and set times using ladybug
            d = sunpath.calculate_sunrise_sunset(
                month=dt.month, day=dt.day, depression=depression
            )

            # calculate the length (for the given light period) of the day in hours
            length_key = "delta"
            if d["sunrise"] is None and d["sunset"] is None and is_winter:
                # no sunrise or sunset, and during winter
                d[length_key] = 0
            elif d["sunrise"] is None and d["sunset"] is None and not is_winter:
                # no sunrise or sunset, and during summer
                d[length_key] = 24
            elif d["sunrise"] is None:
                # only sunrise is present
                d[length_key] = (day_end - d["sunrise"]).total_seconds() / 3600
            elif d["sunset"] is None:
                d[length_key] = (d["sunset"] - day_start).total_seconds() / 3600
            else:
                d[length_key] = (d["sunset"] - d["sunrise"]).total_seconds() / 3600

            rr.append({(light_type, k): v for k, v in d.items()})

            # rr.append({("night", "hours"): 24 - d["hours"]})
        kk[pd.to_datetime(dt)] = {k: v for x in rr for k, v in x.items()}

    df = pd.DataFrame(kk).T
    df.index = pd.to_datetime(df.index)

    # add the number of hours in each light period
    actual_hours = (df[("actual", length_key)]).rename(("actual", "hours"))
    apparent_hours = (df[("apparent", length_key)] - actual_hours).rename(
        ("apparent", "hours")
    )
    civil_hours = (df[("civil", length_key)] - apparent_hours - actual_hours).rename(
        ("civil", "hours")
    )
    nautical_hours = (
        df[("nautical", length_key)] - civil_hours - apparent_hours - actual_hours
    ).rename(("nautical", "hours"))
    astronomical_hours = (
        df[("astronomical", length_key)]
        - nautical_hours
        - civil_hours
        - apparent_hours
        - actual_hours
    ).rename(("astronomical", "hours"))
    night_hours = (
        24
        - (
            actual_hours
            + apparent_hours
            + civil_hours
            + nautical_hours
            + astronomical_hours
        )
    ).rename(("night", "hours"))

    df = pd.concat(
        [
            df,
            pd.concat(
                [
                    actual_hours,
                    apparent_hours,
                    civil_hours,
                    nautical_hours,
                    astronomical_hours,
                    night_hours,
                ],
                axis=1,
            ),
        ],
        axis=1,
    ).sort_index(axis=1)

    return df


@dataclass
class Solar:
    """An object containing solar data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (list[datetime]):
            An iterable of datetime-like objects.
        direct_normal_radiation (list[float]):
            An iterable of direct normal irradiance values, in Wh/m2.
        diffuse_horizontal_radiation (list[float]):
            An iterable of diffuse horizontal irradiance values, in Wh/m2.
        global_horizontal_radiation (list[float]):
            An iterable of global horizontal irradiance values, in Wh/m2.

    """

    # NOTE - BE STRICT WITH THE TYPING!
    # NOTE - Conversions happen in class methods.
    # NOTE - Validation happens at instantiation.

    location: Location
    datetimes: list[datetime]
    global_horizontal_radiation: list[float]
    direct_normal_radiation: list[float]
    diffuse_horizontal_radiation: list[float]

    # region: DUNDER METHODS

    def __post_init__(self):
        """Check for validation of the inputs."""

        # location validation
        if not isinstance(self.location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if self.location.source is None:
            warnings.warn(
                "The source field of the Location input is None. This means that things are a bit ambiguous!"
            )
        if not _is_location_time_zone_valid_for_location(self.location):
            warnings.warn(
                f"The time zone of the location ({self.location.time_zone}) does not match the expected time zone for the lat/lon ({self.location.latitude}, {self.location.longitude})."
            )

        # datetimes validation
        if not _is_iterable_single_dtype(self.datetimes, datetime):
            raise ValueError("datetimes must be a 1D list of datetime-like objects.")

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

        # irradiance validation
        array_names = [
            "direct_normal_radiation",
            "diffuse_horizontal_radiation",
            "global_horizontal_radiation",
        ]
        for name in array_names:
            _temp = getattr(self, name)
            # length validation
            if len(_temp) != len(self.datetimes):
                raise ValueError(
                    f"{name} must be the same length as datetimes. {len(_temp)} != {len(self.datetimes)}."
                )
            # dtype validation
            if not _is_iterable_single_dtype(_temp, (int, float)):
                raise ValueError(f"{name} must be a list of numeric values.")
            # null validation
            if any(np.isnan(_temp)):
                raise ValueError(f"{name} cannot contain null values.")
            # value limit validation
            if any([i < 0 for i in _temp]):
                raise ValueError(f"{name} must be >= 0")

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
                tuple(self.direct_normal_radiation),
                tuple(self.diffuse_horizontal_radiation),
                tuple(self.global_horizontal_radiation),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Solar):
            return False
        return (
            self.location == other.location
            and self.datetimes == other.datetimes
            and self.direct_normal_radiation == other.direct_normal_radiation
            and self.diffuse_horizontal_radiation == other.diffuse_horizontal_radiation
            and self.global_horizontal_radiation == other.global_horizontal_radiation
        )

    def __iter__(self) -> iter:
        return (
            (
                self.datetimes[i],
                self.global_horizontal_radiation[i],
                self.direct_normal_radiation[i],
                self.diffuse_horizontal_radiation[i],
            )
            for i in range(len(self))
        )

    def __getitem__(self, idx: int) -> dict[str, datetime | float]:
        return {
            "datetime": self.datetimes[idx],
            "global_horizontal_radiation": self.global_horizontal_radiation[idx],
            "direct_normal_radiation": self.direct_normal_radiation[idx],
            "diffuse_horizontal_radiation": self.diffuse_horizontal_radiation[idx],
        }

    def __copy__(self) -> "Solar":
        return Solar(
            location=self.location.duplicate(),
            datetimes=self.datetimes,
            direct_normal_radiation=self.direct_normal_radiation,
            diffuse_horizontal_radiation=self.diffuse_horizontal_radiation,
            global_horizontal_radiation=self.global_horizontal_radiation,
        )

    # endregion: DUNDER METHODS

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
    def direct_normal_radiation_series(self) -> pd.Series:
        return pd.Series(
            data=self.direct_normal_radiation,
            index=self.datetimeindex,
            name="Direct Normal Radiation (Wh/m2)",
        )

    @property
    def direct_normal_radiation_collection(self) -> HourlyContinuousCollection:
        data = self.direct_normal_radiation_series
        values = (
            data.groupby([data.index.month, data.index.day, data.index.time])
            .mean()
            .values
        ).tolist()
        ap = self.analysis_period
        header = Header(
            data_type=DirectNormalRadiation(),
            unit="Wh/m2",
            analysis_period=ap,
            metadata={"source": self.location.source},
        )
        return HourlyContinuousCollection(header=header, values=values)

    @property
    def diffuse_horizontal_radiation_series(self) -> pd.Series:
        return pd.Series(
            data=self.diffuse_horizontal_radiation,
            index=self.datetimeindex,
            name="Diffuse Horizontal Radiation (Wh/m2)",
        )

    @property
    def diffuse_horizontal_radiation_collection(self) -> HourlyContinuousCollection:
        data = self.diffuse_horizontal_radiation_series
        values = (
            data.groupby([data.index.month, data.index.day, data.index.time])
            .mean()
            .values
        ).tolist()
        ap = self.analysis_period
        header = Header(
            data_type=DiffuseHorizontalRadiation(),
            unit="Wh/m2",
            analysis_period=ap,
            metadata={"source": self.location.source},
        )
        return HourlyContinuousCollection(header=header, values=values)

    @property
    def global_horizontal_radiation_series(self) -> pd.Series:
        return pd.Series(
            data=self.global_horizontal_radiation,
            index=self.datetimeindex,
            name="Global Horizontal Radiation (Wh/m2)",
        )

    @property
    def global_horizontal_radiation_collection(self) -> HourlyContinuousCollection:
        data = self.global_horizontal_radiation_series
        values = (
            data.groupby([data.index.month, data.index.day, data.index.time])
            .mean()
            .values
        ).tolist()
        ap = self.analysis_period
        header = Header(
            data_type=GlobalHorizontalRadiation(),
            unit="Wh/m2",
            analysis_period=ap,
            metadata={"source": self.location.source},
        )
        return HourlyContinuousCollection(header=header, values=values)

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.direct_normal_radiation_series,
                self.diffuse_horizontal_radiation_series,
                self.global_horizontal_radiation_series,
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
    def sunpath(self) -> Sunpath:
        return Sunpath.from_location(self.location)

    @property
    def suns(self) -> list[Sun]:
        sp = self.sunpath
        return [sp.calculate_sun_from_hoy(hoy) for hoy in self.analysis_period.hoys]

    @property
    def suns_df(self) -> pd.DataFrame:
        """Get a DataFrame of the sun positions for the analysis period."""
        suns = self.suns
        return pd.DataFrame(
            {
                "azimuth": [s.azimuth for s in suns],
                "altitude": [s.altitude for s in suns],
            },
            index=self.datetimeindex,
        )

    @property
    def sunrise_sunset(self) -> pd.DataFrame:
        """Get sunrise and sunset times for the analysis period.

        Returns:
            pd.DataFrame:
                A DataFrame with sunrise and sunset times for each date in the analysis period.
        """
        return sunrise_sunset(
            dates=np.unique(self.dates),
            location=self.location,
        )

    @property
    def solstices_equinoxes(self) -> pd.DataFrame:
        """Get the solstices and equinoxes for this object.

        Returns:
            pd.DataFrame:
                A DataFrame with solstices and equinoxes for each date in the analysis period.
        """
        years = np.unique([i.year for i in self.dates])
        d = {}
        for year in years:
            d[year] = {
                "vernal equinox": ephem.next_vernal_equinox(str(year))
                .datetime()
                .date(),
                "summer solstice": ephem.next_summer_solstice(str(year))
                .datetime()
                .date(),
                "autumnal equinox": ephem.next_autumnal_equinox(str(year))
                .datetime()
                .date(),
                "winter solstice": ephem.next_winter_solstice(str(year))
                .datetime()
                .date(),
            }
        return pd.DataFrame(d).T

    # endregion: PROPERTIES

    # region: CLASS METHODS
    @classmethod
    def from_wea(cls, wea: Wea) -> "Solar":
        if not isinstance(wea, Wea):
            raise ValueError("wea must be a ladybug Wea object.")

        # modify location to state the Wea object in the source field
        location = wea.location.duplicate()
        location.source = "WEA"

        # obtain the datetimes
        datetimes = pd.to_datetime(wea.analysis_period.datetimes)
        # add timezone information to the datetimes
        datetimes = [
            dt.replace(
                tzinfo=timezone(timedelta(hours=location.time_zone))
            ).to_pydatetime()
            for dt in datetimes
        ]

        return cls(
            location=location,
            datetimes=datetimes,
            direct_normal_radiation=wea.direct_normal_irradiance.values,
            diffuse_horizontal_radiation=wea.diffuse_horizontal_irradiance.values,
            global_horizontal_radiation=wea.global_horizontal_irradiance.values,
        )

    @classmethod
    def from_pvlib(
        cls,
        location: Location,
        start_date: str | date,
        end_date: str | date,
        cloud_cover: float | list[float] = None,
    ) -> "Solar":
        """Construct a Solar object using PVLib."""
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
        datetimes = pd.date_range(
            start=start_date,
            end=end_date + timedelta(days=1),
            freq="h",
            inclusive="both",
            tz=get_tzinfo_from_location(location),
        )[:-1]

        # process cloudcover data
        if cloud_cover is None:
            warnings.warn(
                "cloud_cover is None. This will be set to 0% for all datetimes. This can result in higher than expected radiation values. Try to estimate a rough cloud-cover for the location."
            )
            cloud_cover = [0] * len(datetimes)
        if isinstance(cloud_cover, (float, int)):
            cloud_cover = [cloud_cover] * len(datetimes)
        if isinstance(cloud_cover, (list, tuple)):
            if any([i < 0 or i > 1 for i in cloud_cover]):
                raise ValueError("cloud_cover must be between 0 and 1.")
        if len(cloud_cover) != len(datetimes):
            raise ValueError(
                f"cloud_cover must be the same length as the date range ({len(cloud_cover)} != {len(datetimes)})."
            )
        cloud_cover = np.array(cloud_cover) * 100.0  # convert to percentage

        # modify the location so that its source is pvlib
        location = location.duplicate()
        location.source = f"pvlib-python at {location.latitude}°, {location.longitude}°"
        if sum(cloud_cover) == 0:
            location.source += " (0% cloud cover)"
        elif len(set(cloud_cover)) == 1:
            location.source += f" ({cloud_cover[0] / 100:.0%} constant cloud cover)"
        else:
            location.source += f" ({cloud_cover.mean() / 100:.0%} average cloud cover)"
        # create pvlib location
        pv_location = _lb_location_to_pvlib_location(location)

        # calculate the solar radiation
        solar_position = pv_location.get_solarposition(datetimes)
        dni_extra = get_extra_radiation(datetimes)
        transmittance = ((100.0 - cloud_cover) / 100.0) * 0.75
        irrads = campbell_norman(
            solar_position["apparent_zenith"], transmittance, dni_extra=dni_extra
        )
        irrads = irrads.fillna(0)

        # construct the resulting object
        return cls(
            location=location,
            datetimes=datetimes.to_pydatetime().tolist(),
            direct_normal_radiation=irrads["dni"].tolist(),
            diffuse_horizontal_radiation=irrads["dhi"].tolist(),
            global_horizontal_radiation=irrads["ghi"].tolist(),
        )

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "Solar":
        """Create a Solar object from an EPW file or object.

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
            direct_normal_radiation=epw.direct_normal_radiation.values,
            diffuse_horizontal_radiation=epw.diffuse_horizontal_radiation.values,
            global_horizontal_radiation=epw.global_horizontal_radiation.values,
        )

    def to_dict(self) -> dict:
        """Represent the object as a python-native dtype dictionary."""

        return {
            "type": "Solar",
            "location": self.location.to_dict(),
            "datetimes": [i.isoformat() for i in self.datetimes],
            "direct_normal_radiation": self.direct_normal_radiation,
            "diffuse_horizontal_radiation": self.diffuse_horizontal_radiation,
            "global_horizontal_radiation": self.global_horizontal_radiation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Solar":
        """Create this object from a dictionary."""

        if d.get("type", None) != "Solar":
            raise ValueError("The dictionary cannot be converted Solar object.")

        return cls(
            location=Location.from_dict(d["location"]),
            datetimes=pd.to_datetime(d["datetimes"]),
            direct_normal_radiation=d["direct_normal_radiation"],
            diffuse_horizontal_radiation=d["diffuse_horizontal_radiation"],
            global_horizontal_radiation=d["global_horizontal_radiation"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Solar":
        """Create this object from a JSON string."""
        return cls.from_dict(json.loads(json_string))

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        location: Location,
        direct_normal_radiation_column: str = None,
        diffuse_horizontal_radiation_column: str = None,
        global_horizontal_radiation_column: str = None,
    ) -> "Solar":
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
        if direct_normal_radiation_column is None:
            for col in [
                "direct_normal_radiation",
                "direct_normal_irradiance",
                "dnr",
                "dni",
                "direct",
                "Direct Normal Radiation (Wh/m2)",
                "Direct Normal Irradiance (W/m2)",
            ]:
                if col in df.columns:
                    direct_normal_radiation_column = col
                    break
            if direct_normal_radiation_column is None:
                raise ValueError(
                    "direct_normal_radiation_column not found in DataFrame. You'll need to provide a specific column label rather than relying on a best-guess."
                )
        if diffuse_horizontal_radiation_column is None:
            for col in [
                "diffuse_horizontal_radiation",
                "diffuse_horizontal_irradiance",
                "dhr",
                "dhi",
                "diffuse",
                "Diffuse Horizontal Radiation (Wh/m2)",
                "Diffuse Horizontal Irradiance (W/m2)",
            ]:
                if col in df.columns:
                    diffuse_horizontal_radiation_column = col
                    break
            if diffuse_horizontal_radiation_column is None:
                raise ValueError(
                    "diffuse_horizontal_radiation_column not found in DataFrame. You'll need to provide a specific column label rather than relying on a best-guess."
                )
        if global_horizontal_radiation_column is None:
            for col in [
                "global_horizontal_radiation",
                "global_horizontal_irradiance",
                "ghr",
                "ghi",
                "global",
                "Global Horizontal Radiation (Wh/m2)",
                "Global Horizontal Irradiance (W/m2)",
            ]:
                if col in df.columns:
                    global_horizontal_radiation_column = col
                    break
            if global_horizontal_radiation_column is None:
                raise ValueError(
                    "global_horizontal_radiation_column not found in DataFrame. You'll need to provide a specific column label rather than relying on a best-guess."
                )

        return cls(
            location=location,
            datetimes=df.index.to_pydatetime().tolist(),
            direct_normal_radiation=df[direct_normal_radiation_column].tolist(),
            diffuse_horizontal_radiation=df[
                diffuse_horizontal_radiation_column
            ].tolist(),
            global_horizontal_radiation=df[global_horizontal_radiation_column].tolist(),
        )

    @classmethod
    def from_average(
        cls, objects: list["Solar"], weights: list[int | float] | None = None
    ) -> "Solar":
        # validation
        if not _is_iterable_single_dtype(objects, Solar):
            raise ValueError("objects must be a 1D list of Solar objects.")
        if len(objects) == 0:
            raise ValueError("objects cannot be empty.")
        if len(objects) == 1:
            return objects[0]

        # check datetimes are the same
        for obj in objects:
            if obj.datetimes != objects[0].datetimes:
                raise ValueError("All objects must share the same datetimes.")

        # create the average data's
        dni = np.average(
            [i.direct_normal_radiation for i in objects], weights=weights, axis=0
        )
        dhi = np.average(
            [i.diffuse_horizontal_radiation for i in objects],
            weights=weights,
            axis=0,
        )
        ghi = np.average(
            [i.global_horizontal_radiation for i in objects],
            weights=weights,
            axis=0,
        )
        location = average_location([i.location for i in objects], weights=weights)
        return cls(
            location=location,
            datetimes=objects[0].datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    # endregion: CLASS METHODS

    # region: INSTANCE METHODS

    def to_wea(self, timestep: int = 1) -> Wea:
        return Wea(
            location=self.location,
            direct_normal_irradiance=self.direct_normal_radiation_collection.interpolate_to_timestep(
                timestep=timestep
            ),
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection.interpolate_to_timestep(
                timestep=timestep
            ),
        )

    def apply_shade_objects(
        self,
        shade_objects: list[Any] = (),
    ) -> "Solar":
        """Apply shade objects to the solar data.

        Args:
            shade_objects (list[Any], optional):
                A list of ladybug Shade objects. The shade objects will be used
                to calculate the shading effect on the solar data.

        Returns:
            Solar:
                A new Solar object with the shading applied.
        """
        # TODO - implement this method to calculate the impact of shades at varying locations surrounding the "sensor"
        raise NotImplementedError("Hourly shade impact not yet implemented.")

    def _sky_matrix(
        self,
        north: int = 0,
        ground_reflectance: float = 0.2,
        temperature: HourlyContinuousCollection = None,
        balance_temperature: float = 15,
        balance_offset: float = 2,
        timestep: int = 1,
    ) -> SkyMatrix:
        """Create a ladybug sky matrix from the solar data.

        Args:
            north (int, optional):
                The north direction in degrees.
                Default is 0.
            ground_reflectance (float, optional):
                The ground reflectance value.
                Default is 0.2.
            temperature (HourlyContinuousCollection | list[float | int], optional):
                An iterable of temperature values, or a ladybug HourlyContinuousCollection
                for temperature, which will be used to establish whether radiation
                is desired or not for each time step. The collection must be aligned
                with the irradiance inputs.
            balance_temperature (float, optional):
                The temperature in Celsius between which radiation
                switches from being a benefit to a harm. Typical residential buildings
                have balance temperatures as high as 18C and commercial buildings tend
                to have lower values around 12C.
                Default is 15.
            balance_offset (float, optional):
                The temperature offset from the balance temperature
                in Celsius where radiation is neither harmful nor helpful.
                Default is 2.
            timestep (int, optional):
                The timestep (per hour) in minutes for the sky matrix.
                Default is 1.

        Returns:
            SkyMatrix:
                A ladybug SkyMatrix object.
        """
        if temperature is None:
            return SkyMatrix(
                wea=self.to_wea(timestep=timestep),
                north=north,
                high_density=True,
                ground_reflectance=ground_reflectance,
            )

        # check that temperature is a valid type
        dni = self.direct_normal_radiation_collection.interpolate_to_timestep(
            timestep=timestep
        )
        dhi = self.diffuse_horizontal_radiation_collection.interpolate_to_timestep(
            timestep=timestep
        )
        if not isinstance(temperature, HourlyContinuousCollection):
            raise ValueError(
                "temperature must be a ladybug HourlyContinuousCollection object."
            )
        if not _are_iterables_same_length(self, temperature):
            raise ValueError(
                f"temperature must be the same length as the solar data (n={len(self)})."
            )

        return SkyMatrix.from_components_benefit(
            location=self.location,
            direct_normal_irradiance=dni,
            diffuse_horizontal_irradiance=dhi,
            north=north,
            high_density=True,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )

    def _radiation_rose(
        self,
        sky_matrix: SkyMatrix = None,
        intersection_matrix: Any | None = None,
        direction_count: int = 36,
        tilt_angle: int = 0,
    ) -> RadiationRose:
        """Convert this object to a ladybug RadiationRose object.

        Args:
            sky_matrix (SkyMatrix, optional):
                A SkyMatrix object, which describes the radiation coming
                from the various patches of the sky.
                Default is None, which uses the default sky matrix from the solar data.
            intersection_matrix (Any | None, optional):
                An optional lists of lists, which can be used to account
                for context shade surrounding the radiation rose. The matrix
                should have a length equal to the direction_count and begin
                from north moving clockwise. Each sub-list should consist of
                booleans and have a length equal to the number of sky patches
                times 2 (indicating sky patches and ground patches). True
                indicates that a certain patch is seen and False indicates
                that the match is blocked. If None, the radiation rose will be
                computed assuming no obstructions.
                Default is None.
            direction_count (int, optional):
                An integer greater than or equal to 3, which notes the number
                of arrows to be generated for the radiation rose.
                Default is 36.
            tilt_angle (float, optional):
                A number between 0 and 90 that sets the vertical tilt angle
                (aka. the altitude) for all of the directions. By default,
                the Radiation Rose depicts the amount of solar energy
                received by a vertical wall (tilt_angle=0). The tilt_angle
                be changed to a specific value to assess the solar energy
                falling on geometries that are not perfectly vertical, such
                as a tilted photovoltaic panel.
                Default is 0.

        Returns:
            RadiationRose:
                A ladybug RadiationRose object.
        """

        if sky_matrix is None:
            sky_matrix = self._sky_matrix()

        return RadiationRose(
            sky_matrix=sky_matrix,
            intersection_matrix=intersection_matrix,
            direction_count=direction_count,
            tilt_angle=tilt_angle,
        )

    def _radiation_benefit_data(
        self,
        temperature: HourlyContinuousCollection,
        north: int = 0,
        ground_reflectance: float = 0.2,
        balance_temperature: float = 15,
        balance_offset: float = 2,
    ) -> pd.Series:
        """Return the radiation benefit data from the sky matrix.

        See documentation for self.lb_sky_matrix for more information.
        """

        # create the sky matrix
        smx = self._sky_matrix(
            north=north,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )
        # replace None values with NaN
        d = []
        for i in smx.benefit_matrix:
            if i is None:
                d.append(0.0)
            elif i:
                d.append(1.0)
            else:
                d.append(-1.0)

        return pd.Series(d, index=self.datetimeindex, name="Radiation Benefit")

    def _radiation_rose_data(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 0,
        north: int = 0,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = (),
    ) -> pd.DataFrame:
        """Get directional cumulative radiation in kWh/m2 for a given
        tilt_angle, within the analysis_period and subject to shade_objects.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            directions (int, optional):
                The number of directions to bin data into.
                Defaults to 36.
            tilt_angle (float, optional):
                The tilt (from 0 at horizon, to 90 facing the sky) to assess.
                Defaults to 89.999.
            north (int, optional):
                The north direction in degrees.
                Defaults to 0.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.

        Returns:
            pd.DataFrame:
                A pandas DataFrame containing the radiation data.
        """

        if tilt_angle == 90:
            tilt_angle = 89.99999
        if (tilt_angle > 90) or (tilt_angle < 0):
            raise ValueError("Tilt angle must be between 0 and 90.")

        # create time-filtered sky-matrix
        smx = SkyMatrix.from_components(
            location=self.location,
            direct_normal_irradiance=self.direct_normal_radiation_collection,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection,
            hoys=analysis_period.hoys,
            north=north,
            high_density=True,
            ground_reflectance=ground_reflectance,
        )

        # FixMe - tyhe dreationof an intersection matrix means that values do not match up with raw ladybug

        if shade_objects:
            # create a mesh with the same dumber of faces as the number of
            sensor_mesh = _create_azimuth_mesh(directions, tilt_angle)

            # create a radiation study and intersection matrix from given mesh/objects
            rd = RadiationStudy(
                sky_matrix=smx,
                study_mesh=sensor_mesh,
                context_geometry=shade_objects,
                use_radiance_mesh=True,
            )
            intersection_matrix = rd.intersection_matrix
        else:
            intersection_matrix = None

        # create rad rose
        lb_radrose = RadiationRose(
            sky_matrix=smx,
            intersection_matrix=intersection_matrix,
            direction_count=directions,
            tilt_angle=tilt_angle,
        )

        # get angles
        vectors = lb_radrose.direction_vectors
        angles = [angle_clockwise_from_north([i.x, i.y]) for i in vectors]

        # get the radiation data
        return pd.concat(
            [
                pd.Series(
                    lb_radrose.total_values,
                    index=angles,
                    name=RadiationType.TOTAL.name,
                ),
                pd.Series(
                    lb_radrose.direct_values,
                    index=angles,
                    name=RadiationType.DIRECT.name,
                ),
                pd.Series(
                    lb_radrose.diffuse_values,
                    index=angles,
                    name=RadiationType.DIFFUSE.name,
                ),
            ],
            axis=1,
        )

    def _tilt_orientation_factor_data(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        azimuth_count: int = 36,
        altitude_count: int = 9,
        shade_objects: list[Any] = (),
    ) -> pd.DataFrame:
        """Get tilt-orientation-factor data for the given solar data. This is
        a set of values per tilt and orientation representing the kWh/m2
        received by a surface with that tilt and orientation.

        Args:
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            azimuth_count (int, optional):
                The number of azimuth angles to use.
                Defaults to 36.
            altitude_count (int, optional):
                The number of altitude angles to use.
                Defaults to 9.
            shade_objects (list, optional):
                A list of shade objects to apply to the plot.
                Defaults to an empty list.

        Returns:
            pd.DataFrame:
                A pandas DataFrame containing the tilt-orientation-factor data.
        """
        # check that a full years worth of hourly data is available
        if not _datetimes_span_at_least_1_year(self.datetimes):
            raise ValueError(
                "The Solar object must contain at least 1 year of hourly or sub hourly data."
            )
        # warn if azimuth count is less than 12
        if azimuth_count < 12:
            warnings.warn(
                "The azimuth count is less than 12. This may result in inaccurate results."
            )
        # warn if altitude count is less than 6
        if altitude_count < 6:
            warnings.warn(
                "The altitude count is less than 6. This may result in inaccurate results."
            )

        loc = self.location.duplicate()

        # create time-filtered sky-matrix
        smx = SkyMatrix.from_components(
            location=loc,
            direct_normal_irradiance=self.direct_normal_radiation_collection,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection,
            hoys=analysis_period.hoys,
            high_density=True,
        )

        if shade_objects:
            dome_vectors = RadiationDome.dome_vectors(
                azimuth_count=azimuth_count, altitude_count=altitude_count
            )
            faces = []
            for v in dome_vectors:
                faces.append(
                    Face3D.from_regular_polygon(
                        side_count=3,
                        radius=0.001,
                        base_plane=Plane(n=v, o=Point3D().move(v * 0.001)),
                    )
                )
            sensor_mesh = Mesh3D.from_face_vertices(faces=faces)

            # create a radiation study and intersection matrix from given mesh/objects
            rs = RadiationStudy(
                sky_matrix=smx,
                study_mesh=sensor_mesh,
                context_geometry=shade_objects,
                use_radiance_mesh=True,
            )
            intersection_matrix = rs.intersection_matrix
        else:
            intersection_matrix = None

        # create a radiation dome
        rd = RadiationDome(
            smx,
            intersection_matrix=intersection_matrix,
            azimuth_count=azimuth_count,
            altitude_count=altitude_count,
        )

        # get the raw data
        azimuths, altitudes = np.array(
            [vector3d_to_azimuth_altitude(i) for i in rd.direction_vectors]
        ).T

        # create a dataframe containing the results
        df = pd.DataFrame(
            {
                "azimuth": azimuths,
                "altitude": altitudes,
                RadiationType.TOTAL.name: rd.total_values,
                RadiationType.DIRECT.name: rd.direct_values,
                RadiationType.DIFFUSE.name: rd.diffuse_values,
            }
        ).sort_values(by=["azimuth", "altitude"])

        # add missing extremity values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # FIXME - this is a hack to avoid the warning message from not slicing a dataframe correctly
            temp = df[df["altitude"] == 0]
            temp["altitude"] = 90
            temp[RadiationType.TOTAL.name] = df[df["altitude"] == 90][
                RadiationType.TOTAL.name
            ].values[0]
            temp[RadiationType.DIRECT.name] = df[df["altitude"] == 90][
                RadiationType.DIRECT.name
            ].values[0]
            temp[RadiationType.DIFFUSE.name] = df[df["altitude"] == 90][
                RadiationType.DIFFUSE.name
            ].values[0]

            temp2 = df[df["azimuth"] == 0]
            temp2["azimuth"] = 360

            temp3 = temp[(temp["azimuth"] == 0) & (temp["altitude"] == 90)]
            temp3["azimuth"] = 360

        df = (
            pd.concat([df, temp, temp2, temp3], axis=0)
            .reset_index(drop=True)
            .sort_values(by=["azimuth", "altitude"])
        )
        return df

    # endregion: INSTANCE METHODS

    # region: FILTERING METHODS

    def filter_by_boolean_mask(self, mask: list[bool] = None) -> "Solar":
        """Filter the current object by a boolean mask.

        Args:
            mask (list[bool]):
                A boolean mask to filter the current object.

        Returns:
            Solar:
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

        return Solar(
            location=loc,
            datetimes=[i for i, j in zip(*[self.datetimes, mask]) if j],
            direct_normal_radiation=[
                i for i, j in zip(*[self.direct_normal_radiation, mask]) if j
            ],
            diffuse_horizontal_radiation=[
                i for i, j in zip(*[self.diffuse_horizontal_radiation, mask]) if j
            ],
            global_horizontal_radiation=[
                i for i, j in zip(*[self.global_horizontal_radiation, mask]) if j
            ],
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> "Solar":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Solar:
                A dataset describing solar radiation.
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
        dni = [i for i, j in zip(*[self.direct_normal_radiation, mask]) if j]
        dhi = [i for i, j in zip(*[self.diffuse_horizontal_radiation, mask]) if j]
        ghi = [i for i, j in zip(*[self.global_horizontal_radiation, mask]) if j]

        return Solar(
            location=loc,
            datetimes=datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    def filter_by_time(
        self,
        years: list[int] = None,
        months: list[float] = None,
        days: list[float] = None,
        hours: list[int] = None,
    ) -> "Solar":
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
            Solar:
                A dataset describing historic solar data.
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
        dni = [i for i, j in zip(*[self.direct_normal_radiation, mask]) if j]
        dhi = [i for i, j in zip(*[self.diffuse_horizontal_radiation, mask]) if j]
        ghi = [i for i, j in zip(*[self.global_horizontal_radiation, mask]) if j]

        return Solar(
            location=loc,
            datetimes=datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    # endregion: FILTERING METHODS

    # region: PLOTTING METHODS

    def plot_radiation_rose(
        self,
        ax: Axes | None = None,
        radiation_type: RadiationType = RadiationType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 0,
        north: int = 0,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = (),
        **kwargs,
    ) -> Axes:
        """Plot a radiation rose for the given solar data.

        Args:
            ax (Axes, optional):
                The matplotlib Axes to plot the radiation rose on.
            radiation_type (RadiationType, optional):
                The type of irradiance to plot. Defaults to RadiationType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            directions (int, optional):
                The number of directions to bin data into.
                Defaults to 36.
            tilt_angle (float, optional):
                The tilt (from 0 at horizon, to 90 facing the sky) to assess.
                Defaults to 0.
            north (int, optional):
                The north direction in degrees.
                Defaults to 0.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.
        """

        if radiation_type == RadiationType.REFLECTED:
            raise ValueError(
                "The REFLECTED irradiance type is not supported for plotting a radiation rose."
            )

        # create radiation results
        rad_df = self._radiation_rose_data(
            analysis_period=analysis_period,
            directions=directions,
            tilt_angle=tilt_angle,
            north=north,
            ground_reflectance=ground_reflectance,
            shade_objects=shade_objects,
        )

        # get the radiation data
        match radiation_type:
            case RadiationType.TOTAL:
                data = rad_df[RadiationType.TOTAL.name]
            case RadiationType.DIRECT:
                data = rad_df[RadiationType.DIRECT.name]
            case RadiationType.DIFFUSE:
                data = rad_df[RadiationType.DIFFUSE.name]
            case _:
                raise ValueError("How did you get here?")

        # plot the radiation rose

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        if ax.name != "polar":
            raise ValueError("ax must be a polar axis.")

        # kwargish vars
        # todo - sort out kwargs
        ylim = kwargs.pop("ylim", (0, max(data) * 1.1))
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple of length 2.")
        bar_width = 1
        colors = plt.get_cmap("YlOrRd")(
            np.interp(data.values, (data.min(), data.max() * 1.05), (0, 1))
        )
        title = f"{self.location.source} at {tilt_angle}°\n{_analysis_period_to_string(analysis_period)}"

        rects = ax.bar(
            x=np.deg2rad(data.index),
            height=data.values,
            width=((np.pi / directions) * 2) * bar_width,
            color=colors,
        )

        # add a text label to the peak value bar
        peak_value = max(data.values)
        peak_angle_deg = data.idxmax()
        peak_angle_rad = np.deg2rad(data.idxmax())
        peak_index = np.argmax(data.values)
        peak_bar = rects[peak_index]
        peak_bar.set_edgecolor("black")
        peak_bar.set_zorder(5)
        ax.text(
            peak_angle_rad,
            peak_value * 0.95,
            f"{peak_value:.0f}kWh/m$^2$",
            fontsize="xx-small",
            ha="right" if peak_angle_deg < 180 else "left",
            va="center",
            rotation=(90 - peak_angle_deg)
            if peak_angle_deg < 180
            else (90 - peak_angle_deg + 180),
            rotation_mode="anchor",
            color=contrasting_color(peak_bar.get_facecolor()),
            zorder=5,
        )

        # format the plot
        ax.set_title(title)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(ylim)
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

    def plot_tilt_orientation_factor(
        self,
        ax: Axes | None = None,
        radiation_type: RadiationType = RadiationType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        azimuth_count: int = 36,
        altitude_count: int = 9,
        shade_objects: list[Any] = (),
        show_max: bool = True,
        quantiles: list[float] | None = None,
        show_colorbar: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a tilt-orientation-factor diagram for the given solar data.

        Args:
            ax (Axes, optional):
                The matplotlib Axes to plot the tilt-orientation-factor diagram on.
            radiation_type (RadiationType, optional):
                The type of irradiance to plot. Defaults to RadiationType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            azimuth_count (int, optional):
                The number of azimuth angles to use.
                Defaults to 36.
            altitude_count (int, optional):
                The number of altitude angles to use.
                Defaults to 9.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.
            show_max (bool, optional):
                If True, show the maximum value on the plot.
                Defaults to True.
            quantiles: (list[float], optional):
                A list of quantiles to use for the color levels.
                Defaults to None.
            show_colorbar (bool, optional):
                If True, show the colorbar on the plot.
                Defaults to True.
            **kwargs:
                Additional keyword arguments to pass to the plotting function.

        Return:
            ax: Axes:
                The matplotlib Axes object containing the plot.
        """
        if radiation_type == RadiationType.REFLECTED:
            raise ValueError(
                "The REFLECTED irradiance type is not supported for plotting a tilt-orientation-factor diagram."
            )

        # get the data
        azimuths, altitudes, rads = self._tilt_orientation_factor_data(
            analysis_period=analysis_period,
            azimuth_count=azimuth_count,
            altitude_count=altitude_count,
            shade_objects=shade_objects,
        )[["azimuth", "altitude", radiation_type.name]].values.T

        # split kwargs by endpoint
        tricontourf_kwargs = _filter_kwargs_by_allowable(
            kwargs,
            [
                "levels",
                "colors",
                "alpha",
                "cmap",
                "norm",
                "vmin",
                "vmax",
                "extend",
            ],
        )

        if ax is None:
            ax = plt.gca()

        title = f"{self.location.source}\n{_analysis_period_to_string(analysis_period)}"
        ax.set_title(title)

        tcf = ax.tricontourf(
            azimuths,
            altitudes,
            rads,
            **tricontourf_kwargs,
        )

        if quantiles:
            q_values = np.quantile(rads, quantiles)
            tcl = ax.tricontour(
                azimuths,
                altitudes,
                rads,
                levels=q_values,
                colors="black",
                linewidths=0.5,
            )

            def cl_fmt(x):
                return f"{x:,.0f}kW/m$^2$"

            _ = ax.clabel(tcl, fontsize="small", fmt=cl_fmt)

        if show_max:
            # get max value and location
            max_value = max(rads)
            max_indices = np.where(rads == max_value)
            max_azimuth = np.mean(azimuths[max_indices])
            max_altitude = np.mean(altitudes[max_indices])
            ax.scatter(
                max_azimuth,
                max_altitude,
                marker="o",
                color="black",
                s=50,
                zorder=10,
            )
            ax.text(
                max_azimuth,
                max_altitude + 1,
                f"{max_value:,.0f}kW/m$^2$\n{max_azimuth:.0f}°,{max_altitude:.0f}°",
                fontsize="small",
                ha="left" if max_azimuth < 300 else "right",
                va="bottom" if max_altitude < 80 else "top",
                color="black",
            )
            ax.axvline(max_azimuth, ymax=max_altitude / 90, color="black", ls="--")
            ax.axhline(max_altitude, xmax=max_azimuth / 360, color="black", ls="--")

        if shade_objects:
            ax.text(
                1,
                1,
                "*includes context shading",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
            )

        if show_colorbar:
            # add colorbar
            cb = plt.colorbar(
                tcf,
                ax=ax,
                orientation="vertical",
                drawedges=False,
                fraction=0.05,
                aspect=25,
                pad=0.02,
                label="Cumulative irradiance (kWh/m$^2$)",
            )
            cb.outline.set_visible(False)
            if quantiles:
                qvals = np.quantile(rads, quantiles)
                for quantile_val in qvals:
                    cb.ax.plot(
                        [0, 1],
                        [quantile_val, quantile_val],
                        scalex=False,
                        scaley=True,
                        color="k",
                        ls="-",
                        alpha=0.5,
                    )

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 90)
        ax.xaxis.set_major_locator(MultipleLocator(base=30))
        ax.yaxis.set_major_locator(MultipleLocator(base=10))
        ax.set_xlabel("Orientation (clockwise from North at 0°)")
        ax.set_ylabel("Tilt (0° facing the horizon, 90° facing the sky)")

        return ax

    def plot_radiation_benefit_heatmap(
        self,
        temperature: HourlyContinuousCollection,
        ax: Axes = None,
        north: int = 0,
        ground_reflectance: float = 0.2,
        balance_temperature: float = 15,
        balance_offset: float = 2,
        **kwargs,
    ) -> Axes:
        """Plot the radiation benefit for the given solar data."""
        warnings.warn(
            "The radiation benefit plot is only implemented temporarily here until a proper heatmap method is linked to it."
        )

        # split kwargs by endpoint
        tricontourf_kwargs = _filter_kwargs_by_allowable(
            kwargs,
            [
                "levels",
                "colors",
                "alpha",
                "cmap",
                "norm",
                "vmin",
                "vmax",
                "extend",
            ],
        )

        data = self._radiation_benefit_data(
            temperature=temperature,
            north=north,
            ground_reflectance=ground_reflectance,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )

        if ax is None:
            ax = plt.gca()

        tcf = ax.tricontourf(
            data.index.dayofyear, data.index.hour, data.values, **tricontourf_kwargs
        )
        cb = plt.colorbar(
            tcf,
            ax=ax,
            orientation="vertical",
            drawedges=False,
            fraction=0.05,
            aspect=25,
            pad=0.02,
        )
        cb.outline.set_visible(False)

        return ax

    def plot_sunpath(
        self,
        ax: plt.Axes = None,
        other_data: list[float] = None,
        other_datatype: DataTypeBase = None,
        cmap: Colormap | str = "viridis",
        norm: BoundaryNorm = None,
        sun_size: float = 10,
        show_grid: bool = True,
        show_legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a sun-path for the given Location and analysis period.
        Args:
            location (Location):
                A ladybug Location object.
            ax (plt.Axes, optional):
                A matplotlib Axes object. Defaults to None.
            analysis_period (AnalysisPeriod, optional):
                _description_. Defaults to None.
            data_collection (HourlyContinuousCollection, optional):
                An aligned data collection. Defaults to None.
            cmap (str, optional):
                The colormap to apply to the aligned data_collection. Defaults to None.
            norm (BoundaryNorm, optional):
                A matplotlib BoundaryNorm object containing colormap boundary mapping information.
                Defaults to None.
            sun_size (float, optional):
                The size of each sun in the plot. Defaults to 0.2.
            show_grid (bool, optional):
                Set to True to show the grid. Defaults to True.
            show_legend (bool, optional):
                Set to True to include a legend in the plot if data_collection passed. Defaults to True.
        Returns:
            plt.Axes:
                A matplotlib Axes object.
        """

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})

        title = kwargs.pop("title", self.location.source)

        def to_spherical(point3d):
            """
            Convert a 3D point to spherical coordinates (r, theta, phi).
            r is the distance from the origin to the point,
            theta is the angle in the x-y plane from the x-axis, with y-north at 0 degrees,
            and phi is the angle from the z-axis.
            """
            r = np.sqrt(point3d.x**2 + point3d.y**2 + point3d.z**2)
            theta = np.arctan2(point3d.y, point3d.x)
            phi = np.arccos(point3d.z / r)
            # rotate theta -90 degrees to make 0 degrees north
            theta = theta - np.pi / 2
            return r, theta, phi

        radius = 1

        if other_data is not None:
            raise NotImplementedError("other_data is not implemented yet")
            # todo - implement colormap to other dat, and otehr data type (other_datatype: DataTypeBase = None)
            if len(other_data) != len(self):
                raise ValueError("other_data must be the same length")

        sunpath: Sunpath = self.sunpath

        # plot analemmae
        analemma_polylines_3d = sunpath.hourly_analemma_polyline3d(
            steps_per_month=2, radius=radius
        ) + [
            Polyline3D(i.subdivide_evenly(24))
            for i in sunpath.monthly_day_arc3d(radius=radius)
        ]
        for polyline in analemma_polylines_3d:
            _, theta, phi = np.array([to_spherical(i) for i in polyline]).T
            ax.plot(theta, phi, linewidth=1, color="black", zorder=1)

        # plot suns
        suns = [sun for sun in self.suns if sun.altitude > 0]
        _, theta, phi = np.array(
            [to_spherical(i.position_3d(radius=1)) for i in suns]
        ).T
        ax.scatter(theta, phi, s=1, c="orange", zorder=1)

        # format plot
        ax.spines["polar"].set_visible(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        ax.set_title(title)

        return ax

    def plot_skymatrix(
        self,
        ax: Axes = None,
        radiation_type: RadiationType = RadiationType.TOTAL,
        density: int = 1,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        **kwargs,
    ) -> Axes:
        # split kwargs by endpoint
        plot_kwargs = _filter_kwargs_by_allowable(
            kwargs,
            [
                "levels",
                "alpha",
                "cmap",
                "norm",
            ],
        )

        # create wea
        wea = self.to_wea(timestep=analysis_period.timestep).filter_by_analysis_period(
            analysis_period
        )
        wea_duration = len(wea) / wea.timestep
        wea_folder = Path(tempfile.gettempdir())
        wea_path = wea_folder / "skymatrix.wea"
        wea_file = wea.write(wea_path.as_posix())

        # run gendaymtx
        gendaymtx_exe = (Path(lbr_folders.radbin_path) / "gendaymtx.exe").as_posix()
        cmds = [gendaymtx_exe, "-m", str(density), "-d", "-O1", "-A", wea_file]
        with subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True) as process:
            stdout = process.communicate()
        dir_data_str = stdout[0].decode("ascii")
        cmds = [gendaymtx_exe, "-m", str(density), "-s", "-O1", "-A", wea_file]
        with subprocess.Popen(cmds, stdout=subprocess.PIPE, shell=True) as process:
            stdout = process.communicate()
        diff_data_str = stdout[0].decode("ascii")

        def _broadband_rad(data_str: str) -> list[float]:
            _ = data_str.split("\r\n")[:8]
            data = np.array(
                [[float(j) for j in i.split()] for i in data_str.split("\r\n")[8:]][
                    1:-1
                ]
            )
            patch_values = (
                np.array([0.265074126, 0.670114631, 0.064811243]) * data
            ).sum(axis=1)
            patch_steradians = np.array(ViewSphere().dome_patch_weights(density))
            broadband_radiation = patch_values * patch_steradians * wea_duration / 1000
            return broadband_radiation

        dir_vals = _broadband_rad(dir_data_str)
        diff_vals = _broadband_rad(diff_data_str)

        # create the ,mesh to assign data to
        msh = ViewSphere().dome_patches(density)[0]
        # reshape the data to align with mesh faces
        direct_values = np.concatenate(
            [dir_vals[:-1], np.repeat(dir_vals[0], len(msh.faces) - len(dir_vals) + 1)]
        )
        diffuse_values = np.concatenate(
            [
                diff_vals[:-1],
                np.repeat(diff_vals[0], len(msh.faces) - len(diff_vals) + 1),
            ]
        )
        # create geodataframe with data-linked geometry
        shps = to_shapely(msh)
        df = pd.DataFrame(
            data=[direct_values, diffuse_values, direct_values + diffuse_values],
            index=[
                RadiationType.DIRECT.name,
                RadiationType.DIFFUSE.name,
                RadiationType.TOTAL.name,
            ],
        ).T
        gdf = gpd.GeoDataFrame(df, geometry=list(shps.geoms))

        if ax is None:
            ax = plt.gca()

        # todo - additional plot formatting in here ...

        return gdf.plot(ax=ax, column=radiation_type.name, **plot_kwargs)

    def plot_hours_sunlight(self, ax: Axes = None) -> Axes:
        ax = plt.gca()

        df = self.sunrise_sunset
        adf = df.filter(regex="hours")[
            ["actual", "apparent", "astronomical", "civil", "nautical", "night"]
        ].droplevel(1, axis=1)
        solsices_equinoxes = self.solstices_equinoxes
        renamer = {
            "actual": "Daytime",
            "apparent": "Apparent daytime",
            "astronomical": "Astronomical twilight",
            "civil": "Civil twilight",
            "nautical": "Nautical twilight",
            "night": "Night-time",
        }
        ax = plt.gca()
        colors = ["#FCE49D", "#dbc892ff", "#B9AC86", "#908A7A", "#817F76", "#717171"]
        base = np.zeros_like(adf.index, dtype=float)
        for n, (col_name, col_values) in enumerate(adf.items()):
            vals = np.array(col_values, dtype=float)
            ax.fill_between(
                x=adf.index,
                y1=base,
                y2=base + vals,
                color=colors[n],
                label=renamer[col_name],
            )
            base += vals

        # add solstice and equinox lines
        for col_name, col_values in solsices_equinoxes.items():
            dt = col_values.values[0]
            ax.axvline(x=dt, color="black", ls="--", alpha=0.5)
            # add sunrise/set times for key dates too

            try:
                srise = pd.to_datetime(
                    df[(df.index.month == dt.month) & (df.index.day == dt.day)][
                        ("actual", "sunrise")
                    ].values[0]
                ).strftime("%H:%M")
            except AttributeError:
                srise = np.nan
            try:
                sset = pd.to_datetime(
                    df[(df.index.month == dt.month) & (df.index.day == dt.day)][
                        ("actual", "sunset")
                    ].values[0]
                ).strftime("%H:%M")
            except AttributeError:
                sset = np.nan
            ax.text(
                (dt + timedelta(days=1)) if dt.month < 6 else (dt - timedelta(days=1)),
                23.75,
                f"{col_name.title()}\n{dt.strftime('%d %b')}\nSunrise: {srise}\nSunset: {sset}",
                rotation=0,
                ha="left" if dt.month < 6 else "right",
                va="top",
                fontsize="small",
                alpha=0.5,
            )

        ax.set_xlim(adf.index[0], adf.index[-1])
        ax.set_ylim(0, 24)
        ax.set_yticks(np.arange(0, 25, 3))
        ax.set_title(f"Hours of daylight and twilight\n{self}")
        ax.set_ylabel("Hours")
        ax.legend(
            bbox_to_anchor=(0.5, -0.05),
            loc="upper center",
            ncol=6,
            title="Day period",
        )
        ax.grid(
            which="both",
            ls="--",
            alpha=0.5,
        )
        return ax

    def plot_solar_elevation_azimuth(self, ax: Axes = None) -> Axes:
        """Plot the solar elevation and azimuth for a location.

        Args:
            ax (plt.Axes, optional):
                A matploltib axes to plot on. Defaults to None.

        Returns:
            Axes:
                The matplotlib axes.
        """
        # create suns
        sp = self.sunpath
        idx = pd.date_range("2017-01-01 00:00:00", "2018-01-01 00:00:00", freq="10min")
        suns = [sp.calculate_sun_from_date_time(i) for i in idx]
        a = pd.DataFrame(index=idx)
        a["altitude"] = [i.altitude for i in suns]
        a["azimuth"] = [i.azimuth for i in suns]

        # create cmap
        cmap = ListedColormap(
            colors=(
                "#809FB4",
                "#90ACBE",
                "#9FC7A2",
                "#90BF94",
                "#9FC7A2",
                "#CF807A",
                "#C86C65",
                "#CF807A",
                "#C6ACA0",
                "#BD9F92",
                "#C6ACA0",
                "#90ACBE",
                "#809FB4",
            ),
            name="noname",
        )
        cmap.set_over("#809FB4")
        cmap.set_under("#809FB4")
        norm = BoundaryNorm(
            boundaries=[
                0,
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
                360,
            ],
            ncolors=cmap.N,
        )

        # create plot
        if ax is None:
            ax = plt.gca()

        series = a["azimuth"]
        day_time_matrix = (
            series.dropna()
            .to_frame()
            .pivot_table(
                columns=series.dropna().index.date, index=series.dropna().index.time
            )
        )
        x = mdates.date2num(day_time_matrix.columns.get_level_values(1))
        y = mdates.date2num(
            pd.to_datetime([f"2017-01-01 {i}" for i in day_time_matrix.index])
        )
        z = day_time_matrix.values
        pcm = ax.pcolormesh(
            x,
            y,
            z[:-1, :-1],
            cmap=cmap,
            norm=norm,
        )
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        ax.yaxis_date()
        ax.yaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.tick_params(labelleft=True, labelbottom=True)

        # hide all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in ax.get_xticks():
            ax.axvline(i, color="w", ls=":", lw=0.5, alpha=0.5)
        for i in ax.get_yticks():
            ax.axhline(i, color="w", ls=":", lw=0.5, alpha=0.5)
        cb = plt.colorbar(
            pcm,
            ax=ax,
            orientation="horizontal",
            drawedges=False,
            fraction=0.05,
            aspect=100,
            pad=0.075,
            # extend=extend,
            label=series.name.title(),
        )
        cb.outline.set_visible(False)
        cb.set_ticks(
            [0, 45, 90, 135, 180, 225, 270, 315, 360],
            labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"],
        )
        ax.set_title(f"Sun Altitude and Azimuth\n{self}")
        ylim = ax.get_ylim()

        # create matrix of monthday/hour for pcolormesh
        pvt = a.pivot_table(columns=a.index.date, index=a.index.time)

        # plot the contours for sun positions
        x = mdates.date2num(pvt["altitude"].columns)
        y = mdates.date2num(pd.to_datetime([f"2017-01-01 {i}" for i in pvt.index]))
        z = pvt["altitude"].values
        # z = np.ma.masked_array(z, mask=z < 0)
        ct = ax.contour(x, y, z, colors="k", levels=np.arange(0, 91, 10))
        ax.clabel(ct, inline=1, fontsize="small")
        ax.set_ylim(ylim)

    # endregion: PLOTTING METHODS
