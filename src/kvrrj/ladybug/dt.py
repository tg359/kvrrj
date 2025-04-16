from datetime import date, datetime, time
from functools import singledispatch
from typing import Any

from ladybug.dt import Date, DateTime, Time

from ..util import _is_leap_year


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


@singledispatch
def from_ladybug_dt(obj: Any) -> datetime | date | time:
    """Convert a ladybug object to an equivalent native python object."""
    if isinstance(obj, (date, time, datetime)):
        return obj
    raise NotImplementedError(f"Cannot convert {type(obj)} to datetime-like object.")


@from_ladybug_dt.register(Date)
def _(obj: Date) -> date:
    return date(year=obj.year, month=obj.month, day=obj.day)


@from_ladybug_dt.register(Time)
def _(obj: Time) -> time:
    return time(hour=obj.hour, minute=obj.minute, tzinfo=obj.tzinfo)


@from_ladybug_dt.register(DateTime)
def _(obj: DateTime) -> datetime:
    return datetime(
        year=obj.year,
        month=obj.month,
        day=obj.day,
        hour=obj.hour,
        minute=obj.minute,
        tzinfo=obj.tzinfo,
    )
