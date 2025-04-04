"""Methods for manipulating Ladybug analysis periods."""

import calendar
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod

from kvrrj.util import _datetimes_span_at_least_1_year, _is_iterable_1d

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


def _iterable_datetimes_to_lb_analysis_period(dts: list[datetime]) -> AnalysisPeriod:
    """
    Convert a 1D iterable of datetime objects into a Ladybug AnalysisPeriod.

    Args:
        dts (list[datetime]): A 1D list of datetime objects representing the time period.

    Returns:
        AnalysisPeriod: A Ladybug AnalysisPeriod object representing the input datetimes.

    Raises:
        ValueError: If the input is not a 1D iterable of datetime objects.
        ValueError: If the datetimes are not in chronological order.
        ValueError: If the timestep between datetimes is not one of the allowed values.
        ValueError: If the datetimes span more than two consecutive years.
        ValueError: If the number of datetimes does not match the expected count for the AnalysisPeriod.

    Notes:
        - The function ensures that the datetimes are in order and calculates the timestep
          between them. The timestep must be one of the following values:
          [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60].
        - The function determines the start and end months, days, and hours based on the
          largest contiguous range of datetimes.
        - Leap years are identified if any datetime falls on February 29.
    """
    # check that arg passed is a 1d iterable of datetime
    if not _is_iterable_1d(dts):
        raise ValueError("The datetimes must be 1D when passed as an iterable.")
    # check that all elements of the iterable are datetime
    if not all(isinstance(dt, datetime) for dt in dts):
        raise ValueError("All elements of the iterable must be datetime objects.")
    # check that datetimes are in order
    if not np.all(np.diff(dts) >= pd.Timedelta(0)):
        raise ValueError(
            "datetimes must be in order. This error may have occured if an AnalysisPeriod was used to generate the datetimes being passed as only a single year would be assigned to all datetimes in that set."
        )
    # check that the datetimes span at least 1 year
    if not _datetimes_span_at_least_1_year(dts):
        raise ValueError("The datetimes must span an entire year.")
    # get timestep between two adjacent datetimes
    tds = []
    for st, end in np.lib.stride_tricks.sliding_window_view(dts, 2):
        if (st.month == end.month) and (st.day == end.day):
            tds.append(end - st)
    timestep = 3600 / min(tds).seconds
    if timestep not in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
        raise ValueError(
            "Timestep must be one of the following: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30 or 60. It is currently set to "
            f"{timestep}."
        )

    # get the start & end months
    unique_years = np.unique([i.year for i in dts])
    if len(unique_years) > 2:
        raise ValueError(
            "The datetimes must be from the same year or consecutive years when the span the year-end."
        )
    unique_months = np.unique([i.month for i in dts])
    if min(unique_years) < max(unique_years):
        # cross year threshold, meaning the st_month would be greater than the end_month
        st_month = max(unique_months)
        end_month = min(unique_months)
    else:
        # no cross year threshold, so we can use the min and max months directly
        st_month = min(unique_months)
        end_month = max(unique_months)

    # get the start and end hours
    unique_hours = np.unique([i.hour for i in dts])
    if len(unique_hours) == 24:
        # no missing hours, so we can use the min and max hours directly
        st_hour = min(unique_hours)
        end_hour = max(unique_hours)
    else:
        # identify gaps in the datetimes
        gaps = []
        for i in range(1, len(dts)):
            if dts[i] - dts[i - 1] > timedelta(hours=1):
                gaps.append(i)
        # otherwise, find the largest contiguous range
        ranges = []
        start_idx = 0
        for gap in gaps:
            ranges.append(dts[start_idx:gap])
            start_idx = gap
        ranges.append(dts[start_idx:])  # Add the final range
        # find the range with the most datetimes
        largest_range = max(ranges, key=len)
        # determine the start and end hours of the largest contiguous range
        st_hour = largest_range[0].hour
        end_hour = largest_range[-1].hour

    # Identify gaps in the datetimes
    gaps = []
    for i in range(1, len(dts)):
        if dts[i] - dts[i - 1] > timedelta(days=1):
            gaps.append(i)

    # If there are no gaps, the entire range is contiguous
    if not gaps:
        st_day = dts[0].day
        end_day = dts[-1].day
    else:
        # Otherwise, find the largest contiguous range
        ranges = []
        start_idx = 0
        for gap in gaps:
            ranges.append(dts[start_idx:gap])
            start_idx = gap
        ranges.append(dts[start_idx:])  # Add the final range

        # Find the range with the most datetimes
        largest_range = max(ranges, key=len)

        # Determine the start and end days of the largest contiguous range
        st_day = largest_range[0].day
        end_day = largest_range[-1].day

    # determine if the year is a leap year
    is_leap_year = False
    for dt in dts:
        if dt.month == 2 and dt.day == 29:
            is_leap_year = True
            break
    ap = AnalysisPeriod(
        st_month=st_month,
        st_day=st_day,
        st_hour=st_hour,
        end_month=end_month,
        end_day=end_day,
        end_hour=end_hour,
        timestep=int(timestep),
        is_leap_year=is_leap_year,
    )
    # try:
    #     ap.datetimes
    # except TypeError as e:
    #     raise ValueError(
    #         "The number of datetimes does not match the number of datetimes in the AnalysisPeriod. This is most likely due to the arrangement of datetimes not being possible to recreate with an AnalysisPeriod object."
    #     ) from e
    return ap


def _string_to_lb_analysis_period(
    string: str, is_leap_year: bool = False
) -> AnalysisPeriod:
    """
    Convert a string representation of an analysis period into an AnalysisPeriod object.
    This function supports two formats for the input string:
    1. Save path format: A string with underscore-separated values representing the
       start and end dates, hours, and optional timestep.
       Example: "0101_1231_0_23_1" (January 1 to December 31, 0:00 to 23:00, timestep 1).
    2. Human-readable format: A string describing the analysis period in a more natural
       language format.
       Example: "Jan 1 to Dec 31 between 0:00 and 23:00 every 1 hour".

    Args:
        string (str): The string representation of the analysis period.
        is_leap_year (bool, optional): Whether the year is a leap year. Defaults to False.

    Returns:
        AnalysisPeriod: An object representing the analysis period.

    Raises:
        ValueError: If the input string does not contain enough information to create
                    an AnalysisPeriod or if the format is invalid.
    """

    # determine whether the text is a save_path format
    if "_" in string:
        # the text is likely in save_path format
        parts = string.split("_")
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
            is_leap_year=is_leap_year,
        )

    # the text is likely in human-readable format
    parts = (
        string.replace(" to ", " ")
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
        is_leap_year=is_leap_year,
    )


def _analysis_period_to_string(
    analysis_period: AnalysisPeriod,
    save_path: bool = False,
) -> str:
    """Create a description of the given analysis period.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.
        save_path (bool, optional):
            If True, create a path-safe string from the analysis period.
            Defaults to False.
            This will be in the format of
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
