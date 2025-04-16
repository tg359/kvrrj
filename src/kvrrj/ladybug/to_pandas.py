from datetime import date, datetime, time, timedelta
from functools import singledispatch
from typing import Any

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    DailyCollection,
    HourlyContinuousCollection,
    HourlyDiscontinuousCollection,
    MonthlyCollection,
    MonthlyPerHourCollection,
)
from ladybug.dt import Date, DateTime, Time
from ladybug.epw import EPW
from ladybug.header import Header

from .util import (
    _analysis_period_to_datetimes,
    _daily_collection_to_series,
    _epw_to_dataframe,
    _header_to_tuple,
    _hourly_collection_to_series,
    _monthly_collection_to_series,
    _monthly_per_hour_collection_to_series,
)


@singledispatch
def to_pandas(
    obj: Any, **kwargs
) -> pd.Series | pd.DataFrame | pd.MultiIndex | pd.DatetimeIndex:
    """Convert an object to a pandas object."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to a pandas object.")


# region: DATETIME


@to_pandas.register(AnalysisPeriod)
def _(obj: AnalysisPeriod) -> pd.DatetimeIndex:
    return _analysis_period_to_datetimes(obj)


@to_pandas.register(Date)
@to_pandas.register(Time)
@to_pandas.register(DateTime)
@to_pandas.register(date)
@to_pandas.register(time)
@to_pandas.register(datetime)
def _(obj: Date) -> pd.Timestamp:
    return pd.to_datetime(obj)


@to_pandas.register(timedelta)
def _(obj: timedelta) -> pd.Timedelta:
    return pd.to_timedelta(obj)


# endregion: DATETIME

# region: HEADER


@to_pandas.register(Header)
def _(obj: Header) -> tuple[str, str]:
    return _header_to_tuple(obj)


# endregion: HEADER

# region: COLLECTION


@to_pandas.register(HourlyContinuousCollection)
@to_pandas.register(HourlyDiscontinuousCollection)
def _(obj: HourlyContinuousCollection | HourlyDiscontinuousCollection) -> pd.Series:
    return _hourly_collection_to_series(obj)


@to_pandas.register(MonthlyCollection)
def _(obj: MonthlyCollection) -> pd.Series:
    return _monthly_collection_to_series(obj)


@to_pandas.register(DailyCollection)
def _(obj: DailyCollection) -> pd.Series:
    return _daily_collection_to_series(obj)


@to_pandas.register(MonthlyPerHourCollection)
def _(obj: MonthlyPerHourCollection) -> pd.DataFrame:
    return _monthly_per_hour_collection_to_series(obj)


# endregion: COLLECTION


# region: EPW
@to_pandas.register(EPW)
def _(obj: EPW) -> pd.DataFrame:
    return _epw_to_dataframe(obj)


# endregion: EPW
