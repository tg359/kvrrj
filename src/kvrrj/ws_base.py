import warnings
from abc import ABC
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
from ladybug.dt import Date, DateTime
from ladybug.sunpath import AnalysisPeriod, Location

from kvrrj.ladybug.analysisperiod import (
    _iterable_datetimes_to_lb_analysis_period,
)
from kvrrj.ladybug.location import (
    _all_timezones_same,
    _is_datetime_location_aligned,
    _is_elevation_valid_for_location,
    _is_location_time_zone_valid_for_location,
    get_tzinfo_from_location,
)
from kvrrj.util import (
    _is_iterable_1d,
    _is_iterable_single_dtype,
    _is_leap_year,
)


@dataclass(eq=True, unsafe_hash=True, repr=False)
class SpaceTimeDataBase(ABC):
    """An object containing geo-located, historic data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (tuple[datetime]):
            An iterable of datetime-like objects.

    """

    # NOTE - BE STRICT WITH THE TYPING!
    # NOTE - Conversions happen in class methods.
    # NOTE - Validation happens at instantiation.

    location: Location
    datetimes: tuple[datetime]

    # region: DUNDER METHODS

    def __post_init__(self):
        """Check for validation of the inputs."""

        # location validation
        if not isinstance(self.location, Location):
            raise ValueError("location must be a ladybug Location object.")

        # create a copy of the input location to avoid modifying the original
        self.location = self.location.duplicate()
        if self.location.source is None:
            warnings.warn(
                'The source field of the Location input is None. This means that things are a bit ambiguous! A default value of "UnknownSource" has been added.'
            )
            self.location.source = "UnknownSource"

        # check time-zone is valid for location
        if not _is_location_time_zone_valid_for_location(self.location):
            raise ValueError(
                f"The time zone of the location ({self.location.time_zone}) does not match the time zone of the lat/lon ({self.location.latitude}, {self.location.longitude})."
            )

        # check elevation is valid for location
        if not _is_elevation_valid_for_location(self.location):
            raise ValueError(
                f"The elevation of the location ({self.location.elevation}) is not valid for the location ({self.location.latitude}, {self.location.longitude})."
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
                # fixme: uncomment warning for prod
                # warnings.warn(
                #     "Timezone information is missing from datetimes. This will be obtained from the Location and added as a UTC offset to the datetimes."
                # )
                dts.append(dt.replace(tzinfo=location_tzinfo))
            else:
                if not _is_datetime_location_aligned(dt, self.location):
                    raise ValueError(
                        "The datetimes' timezone must match the location's time_zone."
                    )
                dts.append(dt)

        # check timezones are the same
        if not _all_timezones_same([d.tzinfo for d in dts]):
            raise ValueError(
                "All datetimes must have the same timezone. This is not the case."
            )
        self.datetimes = dts

    def __len__(self) -> int:
        return len(self.datetimes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} data from {self.location.source} (n={len(self)})"

    def __repr__(self) -> str:
        return str(self)

    # endregion: DUNDER METHODS

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

    # endregion: PROPERTIES

    # region: FILTER METHODS

    def filter_by_boolean_mask(self) -> "SpaceTimeDataBase":
        raise NotImplementedError("Must be implemented by the child class.")

    def filter_by_analysis_period(self) -> "SpaceTimeDataBase":
        raise NotImplementedError("Must be implemented by the child class.")

    def filter_by_time(self) -> "SpaceTimeDataBase":
        raise NotImplementedError("Must be implemented by the child class.")

    # endregion: FILTER METHODS


@dataclass(eq=True, unsafe_hash=True, repr=False)
class Wind(SpaceTimeDataBase):
    """An object containing wind data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (tuple[datetime]):
            An iterable of datetime-like objects.
        wind_speed (tuple[float]):
            An iterable of wind speeds.
        wind_direction (tuple[float]):
            An iterable of wind directions.
    """

    wind_speed: tuple[float]
    wind_direction: tuple[float]

    def __post_init__(self):
        super().__post_init__()

        # validate individual attributes
        if any([i < 0 for i in getattr(self, "wind_speed")]):
            raise ValueError("Wind speed cannot be negative.")
        if any([(i < 0) or (i > 360) for i in getattr(self, "wind_direction")]):
            raise ValueError("Wind direction must be between 0 and 360 degrees.")
        # validate common attributes
        for var in ["wind_speed", "wind_direction"]:
            if not hasattr(self, var):
                raise ValueError(f"{var} is a required attribute.")
            _data = getattr(self, var)
            if not _is_iterable_1d(_data):
                raise ValueError(f"{var} must be a 1D iterable.")
            if not _is_iterable_single_dtype(_data, (int, float, np.float64)):
                raise ValueError(f"{var} must be a list of numbers.")
            if len(_data) != len(self):
                raise ValueError(
                    f"{var} (n={len(_data)}) must be the same length as datetimes (n={len(self.datetimes)})."
                )
            # convert to tuple
            setattr(self, var, tuple(_data))
            # add field as a pandas Series
            setattr(
                self,
                f"{var}_series",
                pd.Series(data=_data, index=self.datetimes, name=var),
            )

    # region: PROPERTIES

    # endregion: PROPERTIES

    # region: FILTER METHODS

    def filter_by_boolean_mask(self, mask: list[bool]) -> "Wind":
        """Filter the data by a boolean mask."""
        if len(mask) != len(self):
            raise ValueError(
                f"Mask (n={len(mask)}) must be the same length as datetimes (n={len(self)})."
            )
        if not _is_iterable_1d(mask):
            raise ValueError("mask must be a 1D iterable.")
        if not _is_iterable_single_dtype(mask, bool):
            raise ValueError("mask must be a list of booleans.")

        # modify the source of the location to indicate that it has been filtered
        loc = self.location.duplicate()
        loc.source = f"{self.location.source} - Filtered by boolean mask"

        return Wind(
            location=loc,
            datetimes=tuple(np.array(self.datetimes)[mask]),
            wind_speed=tuple(np.array(self.wind_speed)[mask]),
            wind_direction=tuple(np.array(self.wind_direction)[mask]),
        )

    # endregion: FILTER METHODS


@dataclass(eq=True, unsafe_hash=True, repr=False)
class Solar(SpaceTimeDataBase):
    """An object containing wind data.

    Args:
        location (Location):
            A ladybug Location object.
        datetimes (tuple[datetime]):
            An iterable of datetime-like objects.
        global_horizontal_radiation (tuple[float]):
            An iterable of global horizontal irradiance values, in Wh/m2.
        direct_normal_radiation (tuple[float]):
            An iterable of direct normal irradiance values, in Wh/m2.
        diffuse_horizontal_radiation (tuple[float]):
            An iterable of diffuse horizontal irradiance values, in Wh/m2.
    """

    global_horizontal_radiation: tuple[float]
    direct_normal_radiation: tuple[float]
    diffuse_horizontal_radiation: tuple[float]

    def __post_init__(self):
        super().__post_init__()

        # validate individual attributes

        # validate common attributes
        for var in [
            "global_horizontal_radiation",
            "direct_normal_radiation",
            "diffuse_horizontal_radiation",
        ]:
            if not hasattr(self, var):
                raise ValueError(f"{var} is a required attribute.")
            _data = getattr(self, var)
            if not _is_iterable_1d(_data):
                raise ValueError(f"{var} must be a 1D iterable.")
            if not _is_iterable_single_dtype(_data, (int, float, np.float64)):
                raise ValueError(f"{var} must be a list of numbers.")
            if any([i < 0 for i in _data]):
                raise ValueError(f"{var} cannot be negative.")
            if len(_data) != len(self):
                raise ValueError(
                    f"{var} (n={len(_data)}) must be the same length as datetimes (n={len(self.datetimes)})."
                )
            # convert to tuple
            setattr(self, var, tuple(_data))
            # add field as a pandas Series
            setattr(
                self,
                f"{var}_series",
                pd.Series(data=_data, index=self.datetimes, name=var),
            )
