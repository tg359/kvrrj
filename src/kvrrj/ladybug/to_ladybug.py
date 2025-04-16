from datetime import date, datetime, time
from functools import singledispatch
from typing import Any

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import BaseCollection
from ladybug.dt import Date, DateTime, Time
from ladybug.location import Location
from pvlib.location import Location as pvlib_location

from kvrrj.ladybug.util import (
    _datetimes_to_analysis_period,
    _series_to_hourly_collection,
    _series_to_monthly_collection,
    _series_to_monthly_per_hour_collection,
    _string_to_analysis_period,
)

# region: LOCATION


@singledispatch
def to_lb_location(obj: Any) -> Location:
    """Convert an object to a ladybug location object."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to ladybug location.")


@to_lb_location.register(pvlib_location)
def _(location: pvlib_location) -> Location:
    return Location(
        latitude=location.latitude,
        longitude=location.longitude,
        elevation=location.altitude,
        time_zone=location.pytz.utcoffset(datetime(2017, 1, 1)).seconds / 3600,
        source=location.name,
    )


# endregion: LOCATION

# region DATETIME


@singledispatch
def to_ladybug_dt(obj: Any) -> DateTime | Time | Date:
    """Convert an object to an equivalent Ladybug."""
    if isinstance(obj, (Date, Time, DateTime)):
        return obj
    raise NotImplementedError(f"Cannot convert {type(obj)} to ladybug object.")


@to_ladybug_dt.register(date)
def _(obj: date) -> Date:
    return Date(month=obj.month, day=obj.day, leap_year=_is_leap_year(obj.year))


@to_ladybug_dt.register(time)
def _(obj: time) -> Time:
    return Time(hour=obj.hour, minute=obj.minute)


@to_ladybug_dt.register(datetime)
def _(obj: datetime) -> DateTime:
    return DateTime(
        month=obj.month,
        day=obj.day,
        hour=obj.hour,
        minute=obj.minute,
        leap_year=_is_leap_year(obj.year),
    )


# endregion: DATETIME

# region: ANALYSIS_PERIOD


@singledispatch
def to_lb_analysis_period(obj: Any) -> AnalysisPeriod:
    """Convert an object to an AnalysisPeriod."""
    # TODO - check that kwargs can be used in some but not all dispatched funcs
    raise NotImplementedError(f"Cannot convert {type(obj)} to ladybug object.")


@to_lb_analysis_period.register(str)
def _(obj: str, **kwargs) -> AnalysisPeriod:
    return _string_to_analysis_period(obj, **kwargs)


@to_lb_analysis_period.register(pd.DatetimeIndex)
def _(obj: pd.DatetimeIndex) -> AnalysisPeriod:
    return _datetimes_to_analysis_period(obj)


# endregion: ANALYSIS_PERIOD


# region: DATACOLLECTION


@singledispatch
def to_ladybug_collection(obj: Any) -> Any:
    """Convert an object to a ladybug collection."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to ladybug object.")


@to_ladybug_collection.register(pd.Series)
def _(obj: pd.Series) -> BaseCollection:
    try:
        return _series_to_hourly_collection(obj)
    except Exception:
        try:
            return _series_to_monthly_collection(obj)
        except Exception:
            return _series_to_monthly_per_hour_collection(obj)


# endregion: DATACOLLECTION


@singledispatch
def to_ladybug(obj: Any) -> Any:
    """Convert an object to a Ladybug object."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to Ladybug object.")


# TODO - tests for all methods in here, once completed ... might need to follow the path back a fair bit!
# TODO - tidy up all teh code!!!!!
# TODO - dispatch methods for different forms of Series to construct differnt forms of datacollection


# @to_ladybug.register(pd.Series)
# def _(obj: pd.Series) -> BaseCollection:

#     #
