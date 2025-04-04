"""Methods for handling solar data. This module relies heavily on numpy, pandas, and ladybug."""

# TODO - Shade benefit calc (on window) - https://github.com/ladybug-tools/ladybug-grasshopper/blob/master/ladybug_grasshopper/src/LB%20Shade%20Benefit.py
# TODO - Thermal shade benefit calc - https://github.com/ladybug-tools/ladybug-grasshopper/blob/master/ladybug_grasshopper/src//LB%20Thermal%20Shade%20Benefit.py
# TODO - PV calc from pvlib
# TODO - PV with shade objects (from sky matrix, or get total incident radiation on surface using Radiance and then feed into PVLib)
# TODO - Use DirectSun/RadiationStudy to calculate shadedness of a point given context meshes
# TODO - Add location to wind object
# todo - glare risk for aperture in direction
# todo - inherit from common base object for both wind and solar

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
from honeybee.model import Model
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import HourlyContinuousCollection
from ladybug.datatype.energyintensity import (
    DiffuseHorizontalRadiation,
    DirectNormalRadiation,
    GlobalHorizontalRadiation,
)
from ladybug.dt import Date, DateTime
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.skymodel import dirint, disc
from ladybug.sunpath import Location, Sun, Sunpath
from ladybug.wea import Wea
from ladybug_geometry.geometry3d import (
    Face3D,
    Mesh3D,
    Plane,
    Point3D,
)
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.study.radiation import RadiationStudy
from ladybug_radiance.visualize.raddome import RadiationDome
from ladybug_radiance.visualize.radrose import RadiationRose
from lbt_recipes.recipe import Recipe
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from pvlib.irradiance import campbell_norman, get_extra_radiation
from pvlib.location import Location as PVLocation

from kvrrj.geometry.util import (
    _create_azimuth_mesh,
    angle_clockwise_from_north,
    vector3d_to_azimuth_altitude,
)
from kvrrj.honeybee_radiance import (
    create_origin_sensor_grid,
    geo_to_shade,
    radiance_parameters,
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
from kvrrj.logging import CONSOLE_LOGGER
from kvrrj.util import (
    _are_iterables_same_length,
    _datetimes_span_at_least_1_year,
    _filter_kwargs_by_allowable,
    _is_iterable_1d,
    _is_iterable_single_dtype,
    _is_leap_year,
)
from kvrrj.viz.color import contrasting_color


class RadiationType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()
    REFLECTED = auto()


class SunriseSunsetType(Enum):
    """An enumeration of different types of daylight."""

    ACTUAL = auto()
    APPARENT = auto()
    CIVIL = auto()
    NAUTICAL = auto()
    ASTRONOMICAL = auto()

    @property
    def depression_angle(self) -> float:
        """Get the depression angle (in degrees) for the daylight type."""
        return {
            SunriseSunsetType.ACTUAL.name: 0.5334,
            SunriseSunsetType.APPARENT.name: 0.833,
            SunriseSunsetType.CIVIL.name: 6,
            SunriseSunsetType.NAUTICAL.name: 12,
            SunriseSunsetType.ASTRONOMICAL.name: 18,
        }[self.name]


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


def _split_ghi_into_dni_dhi(
    ghi: list[float],
    datetimes: list[datetime],
    location: Location,
    pressures: list[float] | None = None,
    use_disc: bool = False,
) -> tuple:
    """Split global horizontal irradiance into direct normal and diffuse horizontal.

    Args:
        ghi (list[float]):
            An iterable of global horizontal irradiance values, in Wh/m2.
        datetimes (list[datetime]):
            An iterable of datetime-like objects.
        location (Location):
            A ladybug Location object.
        pressures (list[float], optional):
            An iterable of atmospheric pressures, in Pa. If None, a default value of 101325 Pa is used.
        use_disc (bool, optional):
            If True, use the disc method to calculate direct normal irradiance. If False, use the dirint method.

    Returns:
        tuple:
            A tuple of two lists: direct normal irradiance and diffuse horizontal irradiance.

    """

    # validation checks
    if not _is_iterable_1d(ghi):
        raise ValueError("ghi must be a list of floats.")
    if not _is_iterable_single_dtype(ghi, (int, float, np.float64)):
        raise ValueError("ghi must be a list of numeric values.")
    if not isinstance(location, Location):
        raise ValueError("location must be a ladybug Location object.")
    if not _is_iterable_1d(datetimes):
        raise ValueError("datetimes must be a 1D iterable.")
    if not _is_iterable_single_dtype(datetimes, datetime):
        raise ValueError("datetimes must be a list of datetime-like objects.")
    if len(ghi) != len(datetimes):
        raise ValueError(
            f"ghi must be the same length as datetimes. {len(ghi)} != {len(datetimes)}."
        )

    if pressures is not None:
        if not _is_iterable_1d(pressures):
            raise ValueError("pressures must be a 1D iterable.")
        if not _is_iterable_single_dtype(pressures, (int, float, np.float64)):
            raise ValueError("pressures must be a list of numeric values.")
        if len(pressures) != len(datetimes):
            raise ValueError(
                f"pressures must be the same length as datetimes. {len(pressures)} != {len(datetimes)}."
            )
    else:
        # create a list of pressures for each datetime
        pressures = [101325] * len(datetimes)

    # create sun altitudes for each dateitme for the given location
    sunpath = Sunpath.from_location(location)
    sun_altitudes = [
        sunpath.calculate_sun_from_date_time(dt).altitude for dt in datetimes
    ]

    # get the days of the year
    doys = [dt.timetuple().tm_yday for dt in datetimes]

    # calculate the direct normal irradiance (dni)
    if use_disc:
        dni, _, _ = np.array(
            [
                disc(
                    ghi=ghi[i],
                    altitude=sun_altitudes[i],
                    doy=doys[i],
                    pressure=pressures[i],
                )
                for i in range(len(datetimes))
            ]
        ).T.tolist()
    else:
        dni = dirint(
            ghi=ghi,
            altitudes=sun_altitudes,
            doys=doys,
            pressures=pressures,
        )

    # get dhi
    dhi = [
        ghi[i] - (dni[i] * np.sin(np.deg2rad(sun_altitudes[i])))
        for i in range(len(ghi))
    ]

    return dni, dhi


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
    direct_normal_radiation: list[float]
    diffuse_horizontal_radiation: list[float]
    global_horizontal_radiation: list[float]

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

        # irradiance validation
        array_names = [
            "direct_normal_radiation",
            "diffuse_horizontal_radiation",
            "global_horizontal_radiation",
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
            if any([i < 0 for i in getattr(self, name)]):
                raise ValueError(f"{name} must be >= 0")

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

    # endregion: DUNDER METHODS

    # region: STATIC METHODS

    @staticmethod
    def _sunrise_sunset(dates: list[date], location: Location) -> pd.DataFrame:
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
        if not _is_iterable_1d(dates):
            raise ValueError("dates must be a 1D iterable.")
        if not _is_iterable_single_dtype(dates, date):
            raise ValueError("dates must be a list of date objects.")

        loc = location.duplicate()
        # set location timezone to 0
        if loc.time_zone != 0:
            warnings.warn(
                'The location\'s time zone is not "0". This will be set to UTC for the calculation to give local sunrise/set time.'
            )
            loc.time_zone = 0

        # get the sunpath
        sunpath = Sunpath.from_location(loc)

        # get sunrise times
        kk = {}
        for dt in dates:
            if dt.month == 2 and dt.day == 29:
                # skip leap day
                warnings.warn(
                    "Leap day (February 29) is not supported ... for some reason. Blame sunpath.calculate_sunrise_sunset_from_datetime"
                )
                continue
            kk[dt] = {}
            for s_type in SunriseSunsetType:
                d = sunpath.calculate_sunrise_sunset_from_datetime(
                    datetime=datetime(dt.year, month=dt.month, day=dt.day),
                    depression=s_type.depression_angle,
                )
                for tod in ["sunrise", "sunset"]:
                    kk[dt][f"{s_type.name.lower()} {tod}"] = d[tod]
            kk[dt]["noon"] = d["noon"]

        return pd.DataFrame(kk).T.sort_index()

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
    def wea(self) -> Wea:
        return Wea(
            location=self.location,
            direct_normal_irradiance=self.direct_normal_radiation_collection,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection,
        )

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
        return self._sunrise_sunset(
            dates=self.dates,
            location=self.location,
        )

    # endregion: PROPERTIES

    # region: CLASS METHODS
    @classmethod
    def from_wea(cls, wea: Wea) -> "Solar":
        if not isinstance(wea, Wea):
            raise ValueError("wea must be a ladybug Wea object.")

        location = wea.location.duplicate()
        # modify location to state the Wea object in the source field
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
    def from_openmeteo(
        cls,
        location: Location,
        start_date: str | date,
        end_date: str | date,
    ) -> "Solar":
        """Query Open Meteo for solar data."""

        warnings.warn(
            "Radiation data queried from OpenMeteo uses the global horizontal irradiance split into the direct and diffuse components. This can result in peak radiation times and directions not matching corresponding epw files. Caution should be used when constructing solar objects using this method"
        )
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
                "openmeteo_name": "shortwave_radiation",
                "openmeteo_unit": "W/m²",
                "target_name": "Global Horizontal Radiation",
                "target_unit": "Wh/m2",
                "target_multiplier": 1,
            },
            {
                "openmeteo_name": "surface_pressure",
                "openmeteo_unit": "hPa",
                "target_name": "Atmospheric Station Pressure",
                "target_unit": "Pa",
                "target_multiplier": 100,
            },
        ]

        # create the savepath for the returned data
        sp = (
            _dir
            / f"solar_{location.latitude}_{location.longitude}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.json"
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
        # get the global horizontal radiation
        ghi = data["hourly"]["shortwave_radiation"]
        # get the pressures
        pressures = [i * 100 for i in data["hourly"]["surface_pressure"]]
        # Split global rad into direct + diffuse using dirint method (aka. Perez split)
        dni, dhi = _split_ghi_into_dni_dhi(
            ghi=ghi,
            datetimes=datetimes,
            location=location,
            pressures=pressures,
            use_disc=False,
        )
        return cls(
            location=location,
            datetimes=datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
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
                "direct",
                "Direct Normal Radiation (Wh/m2)",
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
                "diffuse_horizontal_radiation_column",
                "diffuse",
                "Diffuse Horizontal Radiation (Wh/m2)",
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
                "global_horizontal_radiation_column",
                "global",
                "Global Horizontal Radiation (Wh/m2)",
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
        # check objects is a list of Solar objects

        # validation
        if not _is_iterable_1d(objects):
            raise ValueError("objects must be a 1D list of Solar objects.")
        if not _is_iterable_single_dtype(objects, Solar):
            raise ValueError("objects must be a list of Solar objects.")
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
        raise NotImplementedError()
        if len(shade_objects) == 0:
            return self

        # create shade objects from the passed shade_objects
        shades = []
        for shd in shade_objects:
            shades.extend(geo_to_shade(shd))
        # construct the model
        model = Model(identifier="solar", orphaned_shades=shades)
        # add sensor grid (single sensor at ground point uopwards)
        model.properties.radiance.sensor_grids = [
            create_origin_sensor_grid(identifier="xyz")
        ]
        # simulate annual irradiance
        recipe = Recipe("annual-irradiance")
        params = radiance_parameters(
            model=model,
            detail_dim=1,
            recipe_type="annual",
            detail_level=0,
        )

        recipe.input_value_by_name("model", model)
        recipe.input_value_by_name("wea", self.wea)
        recipe.input_value_by_name("output-type", "solar")
        recipe.input_value_by_name("radiance-parameters", params)
        results_folder = recipe.run()

        return results_folder

    def _sky_matrix(
        self,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        temperature: HourlyContinuousCollection = None,
        balance_temperature: float = 15,
        balance_offset: float = 2,
    ) -> SkyMatrix:
        """Create a ladybug sky matrix from the solar data.

        Args:
            north (int, optional):
                The north direction in degrees.
                Default is 0.
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Default is True.
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

        Returns:
            SkyMatrix:
                A ladybug SkyMatrix object.
        """
        if temperature is None:
            return SkyMatrix(
                wea=self.wea,
                north=north,
                high_density=high_density,
                ground_reflectance=ground_reflectance,
            )

        # check that temperature is a valid type
        if not isinstance(temperature, HourlyContinuousCollection):
            raise ValueError(
                "temperature must be a ladybug HourlyContinuousCollection object."
            )
        if not _are_iterables_same_length(self, temperature):
            raise ValueError("temperature must be the same length as the solar data.")

        return SkyMatrix.from_components_benefit(
            location=self.location,
            direct_normal_irradiance=self.direct_normal_radiation_collection,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_radiation_collection,
            north=north,
            high_density=high_density,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )

    def _radiation_rose(
        self,
        sky_matrix: SkyMatrix,
        intersection_matrix: Any | None = None,
        direction_count: int = 36,
        tilt_angle: int = 0,
    ) -> RadiationRose:
        """Convert this object to a ladybug RadiationRose object.

        Args:
            sky_matrix (SkyMatrix):
                A SkyMatrix object, which describes the radiation coming
                from the various patches of the sky.
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
        high_density: bool = True,
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
            high_density=high_density,
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
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = (),
    ) -> pd.DataFrame:
        """Get directional cumulative radiation in kWh/m2 for a given
        tilt_angle, within the analysis_period and subject to shade_objects.

        Args:
            irradiance_type (RadiationType, optional):
                The type of irradiance to plot. Defaults to RadiationType.TOTAL.
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
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Defaults to True.
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
            high_density=high_density,
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

    # region: FILTERING

    def filter_by_boolean_mask(self, mask: tuple[bool]) -> "Solar":
        """Filter the current object by a boolean mask.

        Args:
            mask (tuple[bool]):
                A boolean mask to filter the current object.

        Returns:
            Solar:
                A dataset describing solar radiation.
        """

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

        possible_datetimes = pd.to_datetime(analysis_period.datetimes)
        lookup = pd.DataFrame(
            {
                "month": possible_datetimes.month,
                "day": possible_datetimes.day,
                "time": possible_datetimes.time,
            }
        )
        idx = self.datetimeindex
        reference = pd.DataFrame(
            {
                "month": idx.month,
                "day": idx.day,
                "time": idx.time,
            }
        )
        mask = reference.isin(lookup).all(axis=1)

        loc = self.location.duplicate()
        loc.source = (
            f"{self.location.source} (filtered to {_analysis_period_to_string(analysis_period)})",
        )

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

    def filter_by_time(
        self,
        months: tuple[float] = None,
        days: tuple[float] = None,
        hours: tuple[int] = None,
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

        if months is None:
            months = range(1, 13)
        if days is None:
            days = range(1, 32)
        if hours is None:
            hours = range(0, 24)

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

    # region: PLOTTING METHODS

    def plot_radiation_rose(
        self,
        ax: Axes | None = None,
        irradiance_type: RadiationType = RadiationType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 0,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = (),
        **kwargs,
    ) -> Axes:
        """Plot a radiation rose for the given solar data.

        Args:
            ax (Axes, optional):
                The matplotlib Axes to plot the radiation rose on.
            irradiance_type (RadiationType, optional):
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
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Defaults to True.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.
        """

        if irradiance_type == RadiationType.REFLECTED:
            raise ValueError(
                "The REFLECTED irradiance type is not supported for plotting a radiation rose."
            )

        # create radiation results
        rad_df = self._radiation_rose_data(
            analysis_period=analysis_period,
            directions=directions,
            tilt_angle=tilt_angle,
            north=north,
            high_density=high_density,
            ground_reflectance=ground_reflectance,
            shade_objects=shade_objects,
        )

        # get the radiation data
        match irradiance_type:
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
        irradiance_type: RadiationType = RadiationType.TOTAL,
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
            irradiance_type (RadiationType, optional):
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
        if irradiance_type == RadiationType.REFLECTED:
            raise ValueError(
                "The REFLECTED irradiance type is not supported for plotting a tilt-orientation-factor diagram."
            )

        # get the data
        azimuths, altitudes, rads = self._tilt_orientation_factor_data(
            analysis_period=analysis_period,
            azimuth_count=azimuth_count,
            altitude_count=altitude_count,
            shade_objects=shade_objects,
        )[["azimuth", "altitude", irradiance_type.name]].values.T

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

        plt.tight_layout()

        return ax

    def plot_radiation_benefit_heatmap(
        self,
        temperature: HourlyContinuousCollection,
        ax: Axes = None,
        north: int = 0,
        high_density: bool = True,
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
            high_density=high_density,
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

    # endregion: PLOTTING METHODS
