"""Methods for manipulating Ladybug analysis periods."""

import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.dt import DateTime as lbdatetime

_TIMESTEP = {
    1: "hour",
    2: "half-hour",
    3: "20-minutes",
    4: "15-minutes",
    5: "12-minutes",
    6: "10-minutes",
    10: "6-minutes",
    12: "5-minutes",
    15: "4-minutes",
    20: "3-minutes",
    30: "2-minutes",
    60: "minute",
}


def lbdatetime_to_datetime(dt: lbdatetime) -> datetime:
    """Convert a Ladybug DateTime object into a native Python datetime object.

    Args:
        lb_datetime (DateTime):
            A Ladybug DateTime object.

    Returns:
        datetime:
            A Python datetime object.
    """

    if not isinstance(dt, lbdatetime):
        raise ValueError("dt must be a Ladybug DateTime object.")

    return datetime(
        year=dt.year,
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        second=dt.second,
    )


def lbdatetime_from_datetime(dt: datetime) -> lbdatetime:
    """Convert a Python datetime object into a Ladybug DateTime object.

    Args:
        date_time (datetime):
            A Python datetime object.

    Returns:
        DateTime:
            A Ladybug DateTime object.
    """

    if not isinstance(dt, datetime):
        raise ValueError("dt must be a Python datetime object.")

    leap_year = (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0)

    return lbdatetime(
        month=dt.month,
        day=dt.day,
        hour=dt.hour,
        minute=dt.minute,
        leap_year=leap_year,
    )


def lbdatetime_to_dict(dt: lbdatetime) -> dict[str : bool | int]:
    """Convert a Ladybug DateTime object into a dictionary."""
    return {
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
        "leap_year": dt.leap_year,
    }


def lbdatetime_from_dict(dt_dict: dict[str : bool | int]) -> lbdatetime:
    """Convert a dictionary into a Ladybug DateTime object."""
    return lbdatetime(
        month=dt_dict["month"],
        day=dt_dict["day"],
        hour=dt_dict["hour"],
        minute=dt_dict["minute"],
        leap_year=dt_dict["leap_year"],
    )


def analysis_period_to_datetimes(analysis_period: AnalysisPeriod) -> list[datetime]:
    """Convert an AnalysisPeriod object into a list of datetimes.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug AnalysisPeriod object.

    Returns:
        list[datetime]:
            A list of datetime objects.
    """
    # check inputs
    if not isinstance(analysis_period, AnalysisPeriod):
        raise ValueError("analysis_period must be a Ladybug AnalysisPeriod object.")

    return [lbdatetime_to_datetime(dt) for dt in analysis_period.datetimes]


def analysis_period_from_datetimes(
    datetimes: list[datetime],
) -> AnalysisPeriod:
    """Convert a list of datetimes into an AnalysisPeriod object.

    Args:
        datetimes (list[datetime]):
            A list of datetimes.

    Returns:
        AnalysisPeriod:
            An AnalysisPeriod object.
    """

    # check inputs
    if not isinstance(datetimes, list):
        raise ValueError("Datetimes must be a list of datetime objects.")
    if not all(isinstance(d, datetime) for d in datetimes):
        raise ValueError("All items in datetimes must be datetime objects.")
    if min(datetimes).year != max(datetimes).year:
        raise ValueError("Datetimes must all be in the same year.")

    # ensure datetime objects are in order and are contiguous
    datetimes = sorted(datetimes)
    if len(set(np.diff(np.array(datetimes)))) != 1:
        raise ValueError("Datetimes must be contiguous.")

    # infer the timestep
    inferred_timestep = (60 * 60) / (datetimes[1] - datetimes[0]).seconds

    analysis_period = AnalysisPeriod.from_start_end_datetime(
        lbdatetime_from_datetime(min(datetimes)),
        lbdatetime_from_datetime(max(datetimes)),
        inferred_timestep,
    )

    if len(analysis_period.datetimes) != len(datetimes):
        raise ValueError(
            f"The number of datetimes ({len(datetimes)}) does not match the number of datetimes in "
            "the AnalysisPeriod ({len(analysis_period.datetimes)}), which probably means your "
            "datetime-list has an irregular time-step and cannot be coerced into an AnalysisPeriod."
        )
    return analysis_period


def analysis_period_from_timedelta(
    td: timedelta | pd.Timedelta, is_leap_year: bool
) -> AnalysisPeriod:
    """Create an AnalysisPeriod object from a timedelta object.

    Args:
        td (timedelta):
            A Python timedelta object.
        is_leap_year (bool):
            A boolean to indicate whether the year is a leap year.

    Returns:
        AnalysisPeriod:
            An AnalysisPeriod object.
    """
    if not isinstance(td, (timedelta, pd.Timedelta)):
        raise ValueError("td must be a timedelta object.")

    return AnalysisPeriod(timestep=int(td.seconds / 3600), is_leap_year=is_leap_year)


def analysis_period_from_pd_freq(freq: str, is_leap_year: bool) -> AnalysisPeriod:
    """Given a pandas frequency string, create an AnalysisPeriod object."""

    td = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
    return analysis_period_from_timedelta(td, is_leap_year)


def analysis_period_to_string(
    analysis_period: AnalysisPeriod,
    save_path: bool = False,
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.
        save_path (bool, optional):
            If True, create a path-safe string from the analysis period.
        include_timestep (bool, optional):
            If True, include the timestep in the description.

    Returns:
        str:
            A description of the analysis period.
    """
    # check inputs
    if not isinstance(analysis_period, AnalysisPeriod):
        raise ValueError("analysis_period must be a Ladybug AnalysisPeriod object.")

    if save_path:
        base_str = (
            f"{analysis_period.st_month:02}{analysis_period.st_day:02}"
            f"_{analysis_period.end_month:02}{analysis_period.end_day:02}"
            f"_{analysis_period.st_hour:02}_{analysis_period.end_hour:02}"
            f"_{analysis_period.timestep:02}"
        )
        return base_str

    base_str = (
        f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to "
        f"{calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between "
        f"{analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:59"
        f" every {_TIMESTEP[analysis_period.timestep]}"
    )

    return base_str


def analysis_period_from_string(text: str) -> AnalysisPeriod:
    """Create an AnalysisPeriod from a string.

    Args:
        text (str):
            A string representation of an AnalysisPeriod.

    Returns:
        AnalysisPeriod:
            An AnalysisPeriod object.
    """
    # check inputs
    if not isinstance(text, str):
        raise ValueError("Text must be a string.")

    # determine whethr the text is a save_path format
    if "_" in text:
        # the text is likely in save_path format
        parts = text.split("_")
        if len(parts) < 4:
            raise ValueError(
                "Text does not contain enough information to create an AnalysisPeriod."
            )

        st_month = int(parts[0][:2])
        st_day = int(parts[0][2:])
        end_month = int(parts[1][:2])
        end_day = int(parts[1][2:])
        st_hour = int(parts[2])
        end_hour = int(parts[3])
        timestep = int(parts[4]) if len(parts) > 4 else 1

        return AnalysisPeriod(
            st_month=st_month,
            st_day=st_day,
            end_month=end_month,
            end_day=end_day,
            st_hour=st_hour,
            end_hour=end_hour,
            timestep=timestep,
        )

    # the text is likely in human-readable format
    parts = (
        text.replace(" to ", " ")
        .replace(" between ", " ")
        .replace(":", " ")
        .replace(" and ", " ")
        .replace(" every", "")
        .split(" ")
    )
    if len(parts) < 8:
        raise ValueError(
            "Text does not contain enough information to create an AnalysisPeriod."
        )

    st_month = list(calendar.month_abbr).index(parts[0])
    st_day = int(parts[1])
    end_month = list(calendar.month_abbr).index(parts[2])
    end_day = int(parts[3])
    st_hour = int(parts[4])
    end_hour = int(parts[6])
    timestep = (
        int({v: k for k, v in _TIMESTEP.items()}[parts[8]]) if len(parts) == 9 else 1
    )

    return AnalysisPeriod(
        st_month=st_month,
        st_day=st_day,
        end_month=end_month,
        end_day=end_day,
        st_hour=st_hour,
        end_hour=end_hour,
        timestep=timestep,
    )
