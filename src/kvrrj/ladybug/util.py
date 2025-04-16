import calendar
import re
import warnings
from datetime import date, datetime, timedelta, tzinfo
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytz
import timezonefinder
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    DailyCollection,
    HourlyContinuousCollection,
    HourlyDiscontinuousCollection,
    MonthlyCollection,
    MonthlyPerHourCollection,
)
from ladybug.datatype import TYPESDICT
from ladybug.epw import EPW
from ladybug.header import Header
from ladybug.location import Location

from ..geometry.util import great_circle_distance

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

# region: LOCATION


def _location_to_string(location: Location) -> str:
    """Convert a Ladybug Location object to a custom string representation.

    Args:
        location (Location): A Ladybug Location object.

    Returns:
        str: A custom string representation of the location.
    """

    # NOTE - for now this just uses the default LB representation of a location,
    # but is here in case we want to change that
    return str(location)


def _string_to_location(string: str) -> Location:
    """Convert a string to a Ladybug Location object.

    Args:
        string (str):
            A string representation of the location.

    Returns:
        Location:
            A Ladybug Location object.
    """
    pattern = r"^([\w\.\s]+),\s*lat:([\d\.\-]+),\s*lon:([\d\.\-]+),\s*tz:([\d\.\-]+),\s*elev:([\d\.\-]+)$"
    match = re.match(pattern, string)
    if not match:
        raise ValueError("String does not match the required format.")

    city, latitude, longitude, time_zone, elevation = match.groups()

    return Location(
        latitude=float(latitude),
        longitude=float(longitude),
        elevation=float(elevation),
        city=city,
        time_zone=float(time_zone),
    )


def average_location(
    locations: list[Location], weights: Sequence[int | float] | None = None
) -> Location:
    """Create an average location from a list of locations.
    This will use weighting if provided to adjust latitude/longitude values.

    Args:
        locations (list[Location]):
            A set of ladybug Location objects.
        weights (list[float], optional):
            A list of weights for each location.
            Defaults to None which evenly weights each location.

    Returns:
        Location: A synthetic location that is the average of all locations.
    """

    # check inputs

    if not isinstance(locations, (list, tuple)):
        raise ValueError("Locations must be a list or tuple of Location objects.")

    if len(locations) == 1:
        return locations[0]

    if len(locations) == 0:
        raise ValueError("No locations provided.")

    if weights is None:
        weights = [1] * len(locations)

    if len(weights) != len(locations):
        raise ValueError("The number of weights must match the number of locations.")

    if sum(weights) == 0:
        raise ValueError("The sum of weights cannot be zero.")

    # raise a warning is the locations are quite far away
    distances = []
    for loc1 in locations:
        for loc2 in locations:
            distances.append(great_circle_distance(loc1, loc2))
    if max(distances) > 10000:
        warnings.warn(
            f"The maximum distance between the locations passed is {max(distances)} km. That's quite far!"
        )

    # calculate average latitude, longitude, and elevation
    lat = (
        np.average(
            np.array([loc.latitude for loc in locations]) + 1000, weights=weights
        )
        - 1000
    )
    lon = (
        np.average(
            np.array([loc.longitude for loc in locations]) + 1000, weights=weights
        )
        - 1000
    )
    elv = np.average(np.array([loc.elevation for loc in locations]), weights=weights)

    # create the location descriptors
    state = "|".join(
        [
            loc.state if loc.state not in ["", "-", None] else "NoState"
            for loc in locations
        ]
    )
    city = "|".join(
        [loc.city if loc.city not in ["", "-", None] else "NoCity" for loc in locations]
    )
    country = "|".join(
        [
            str(loc.country) if loc.country not in ["", "-", None] else "NoCountry"
            for loc in locations
        ]
    )
    station_id = "|".join(
        [
            str(loc.station_id)
            if loc.station_id not in ["", "-", None]
            else "NoStationId"
            for loc in locations
        ]
    )
    source = "|".join(
        [
            str(loc.source) if loc.source not in ["", "-", None] else "NoSource"
            for loc in locations
        ]
    )
    return Location(
        city=city,
        state=state,
        country=country,
        latitude=lat,
        longitude=lon,
        elevation=elv,
        station_id=station_id,
        source=source,
    )


# endregion: LOCATION

# region: ANALYSISPERIOD


def _analysis_period_to_datetimes(
    analysis_period: AnalysisPeriod,
) -> list[datetime]:
    """Convert a Ladybug AnalysisPeriod to a pandas DatetimeIndex."""
    return pd.to_datetime(analysis_period.datetimes).to_pydatetime().tolist()


def _datetimes_to_analysis_period(datetimes: list[datetime]) -> AnalysisPeriod:
    """
    Convert a 1D iterable of datetime objects into a Ladybug AnalysisPeriod.

    Args:
        datetimes (list[datetime]):
            A 1D list of datetime objects representing the time period.

    Returns:
        AnalysisPeriod:
            A Ladybug AnalysisPeriod object corresponding with the input datetimes.

    """

    if not all(isinstance(dt, datetime) for dt in datetimes):
        raise ValueError("All elements of the iterable must be datetime objects.")

    for st, end in np.lib.stride_tricks.sliding_window_view(datetimes, 2):
        if end <= st:
            raise ValueError(
                f"datetimes must be in order, from earliest to latest ({st} > {end})."
            )

    # get timestep between two adjacent datetimes
    tds = []
    for st, end in np.lib.stride_tricks.sliding_window_view(datetimes, 2):
        if (st.month == end.month) and (st.day == end.day):
            tds.append(end - st)
    timestep = 3600 / min(tds).seconds
    if timestep not in [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
        raise ValueError(
            "Timestep must be one of the following: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30 or 60. It is currently set to "
            f"{timestep}."
        )

    # get the start & end months
    unique_years = np.unique([i.year for i in datetimes])
    if len(unique_years) > 2:
        raise ValueError(
            "The datetimes must be from the same year or consecutive years when the span the year-end."
        )
    unique_months = np.unique([i.month for i in datetimes])
    if min(unique_years) < max(unique_years):
        # cross year threshold, meaning the st_month would be greater than the end_month
        st_month = max(unique_months)
        end_month = min(unique_months)
    else:
        # no cross year threshold, so we can use the min and max months directly
        st_month = min(unique_months)
        end_month = max(unique_months)

    # get the start and end hours
    unique_hours = np.unique([i.hour for i in datetimes])
    if len(unique_hours) == 24:
        # no missing hours, so we can use the min and max hours directly
        st_hour = min(unique_hours)
        end_hour = max(unique_hours)
    else:
        # identify gaps in the datetimes
        gaps = []
        for i in range(1, len(datetimes)):
            if datetimes[i] - datetimes[i - 1] > timedelta(hours=1):
                gaps.append(i)
        # otherwise, find the largest contiguous range
        ranges = []
        start_idx = 0
        for gap in gaps:
            ranges.append(datetimes[start_idx:gap])
            start_idx = gap
        ranges.append(datetimes[start_idx:])  # Add the final range
        # find the range with the most datetimes
        largest_range = max(ranges, key=len)
        # determine the start and end hours of the largest contiguous range
        st_hour = largest_range[0].hour
        end_hour = largest_range[-1].hour

    # Identify gaps in the datetimes
    gaps = []
    for i in range(1, len(datetimes)):
        if datetimes[i] - datetimes[i - 1] > timedelta(days=1):
            gaps.append(i)

    # If there are no gaps, the entire range is contiguous
    if not gaps:
        st_day = datetimes[0].day
        end_day = datetimes[-1].day
    else:
        # Otherwise, find the largest contiguous range
        ranges = []
        start_idx = 0
        for gap in gaps:
            ranges.append(datetimes[start_idx:gap])
            start_idx = gap
        ranges.append(datetimes[start_idx:])  # Add the final range

        # Find the range with the most datetimes
        largest_range = max(ranges, key=len)

        # Determine the start and end days of the largest contiguous range
        st_day = largest_range[0].day
        end_day = largest_range[-1].day

    # determine if the year is a leap year
    is_leap_year = False
    for dt in datetimes:
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


def _analysis_period_to_string(
    analysis_period: AnalysisPeriod,
    save_path: bool = False,
) -> str:
    """Convert a Ladybug Analysis Period into a custom string representation.

    The resulting string may be converted back into an AnalysisPeriod object.

    Args:
        analysis_period (AnalysisPeriod):
            A Ladybug analysis period.
        save_path (bool, optional):
            If True, create a path-safe string from the analysis period.
            Defaults to False.
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
            f"_{analysis_period.timestep:02}_{'L' if analysis_period.is_leap_year else 'C'}"
        )
        return base_str

    base_str = (
        f"{calendar.month_abbr[analysis_period.st_month]} {analysis_period.st_day:02} to "
        f"{calendar.month_abbr[analysis_period.end_month]} {analysis_period.end_day:02} between "
        f"{analysis_period.st_hour:02}:00 and {analysis_period.end_hour:02}:59"
        f" every {_TIMESTEP[analysis_period.timestep]} {'(L)' if analysis_period.is_leap_year else '(C)'}"
    )

    return base_str


def _string_to_analysis_period(string: str) -> AnalysisPeriod:
    """Convert a custom string representation of an analysis period into a Ladybug AnalysisPeriod.

    Examples:

        >>> string = "0101_1231_0_23_1_L"
        >>> analysis_period = _string_to_analysis_period(string)
        >>> print(analysis_period)
        AnalysisPeriod(1, 1, 0, 12, 31, 23, 1, True)

        >>> string = "Mar 2 to Dec 31 between 04:00 and 22:59 every half-hour (C)"
        >>> analysis_period = _string_to_analysis_period(string)
        >>> print(analysis_period)
        AnalysisPeriod(3, 2, 4, 12, 31, 22, 2, False)

    Args:
        string (str):
            The string representation of the analysis period.

    Returns:
        AnalysisPeriod: An object representing the analysis period.
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
        timestep = int(parts[4]) if len(parts) > 5 else 1
        is_leap_year = True if parts[5] == "L" else False

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
        .replace("(", "")
        .replace(")", "")
        .split(" ")
    )
    if len(parts) < 9:
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
        int({v: k for k, v in _TIMESTEP.items()}[parts[8]]) if len(parts) == 10 else 1
    )
    is_leap_year = True if parts[9] == "L" else False

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


# endregion: ANALYSISPERIOD

# region: HEADER


def _header_to_tuple(header: Header) -> tuple[str, str]:
    """Convert a Ladybug Header object to a tuple of (data_type, unit)."""
    return (header.data_type.name, header.unit)


def _tuple_to_header(
    obj: tuple[str, str],
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    metadata: dict[str, Any] = None,
) -> Header:
    """Convert a tuple of (data_type, unit) to a Ladybug Header object.

    Args:
        obj (tuple[str, str]):
            A tuple of (data_type, unit).
        analysis_period (AnalysisPeriod, optional):
            An optional Ladybug AnalysisPeriod object. Defaults to a full year.
        metadata (dict[str, Any], optional):
            An optional dictionary of metadata. Defaults to None.

    Returns:
        Header:
            A Ladybug Header object.
    """
    if not isinstance(obj, tuple):
        raise ValueError("obj must be a tuple.")
    if len(obj) != 2:
        raise ValueError("obj must be a tuple of length 2.")
    if not isinstance(obj[0], str):
        raise ValueError("obj[0] must be a string.")
    if not isinstance(obj[1], str):
        raise ValueError("obj[1] must be a string.")

    if not isinstance(analysis_period, AnalysisPeriod):
        raise ValueError("analysis_period must be a Ladybug AnalysisPeriod object.")

    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        if not all(isinstance(k, str) for k in metadata.keys()):
            raise ValueError("All keys in metadata must be strings.")
        if not all(isinstance(v, (str, int, float)) for v in metadata.values()):
            raise ValueError("All values in metadata must be strings, ints or floats.")

    return Header(
        data_type=TYPESDICT[obj[0].replace(" ", "")](),
        unit=obj[1],
        analysis_period=analysis_period,
        metadata=None,
    )


# endregion: HEADER

# region: QUERIES


def _is_leap_year(year: int) -> bool:
    """Check if a year is a leap year."""
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                return True
            return False
        return True
    return False


def _timezone_valid_for_lat_long(latitude: float, longitude: float, tz: tzinfo) -> bool:
    """Check if a timezone is valid for a given latitude and longitude.

    Args:
        latitude (float):
            Latitude of the location.
        longitude (float):
            Longitude of the location.
        tz (tzinfo):
            Timezone information.

    Returns:
        bool: True if the timezone is valid for the given latitude and longitude, False otherwise.
    """

    if not isinstance(latitude, (int, float)):
        raise ValueError("Latitude must be a number.")
    if not isinstance(longitude, (int, float)):
        raise ValueError("Longitude must be a number.")
    if isinstance(tz, (float, int)):
        tz = pytz.FixedOffset(tz * 60)
    if not isinstance(tz, tzinfo):
        raise ValueError("tz must be a tzinfo object.")

    # using
    tf = timezonefinder.TimezoneFinder()

    # From the lat/long, get the tz-database-style time zone name (e.g. 'America/Vancouver') or None
    timezone_str = tf.certain_timezone_at(lat=latitude, lng=longitude)
    expected_tzinfo = pytz.timezone(timezone_str)

    dt = datetime.now()
    a = expected_tzinfo.utcoffset(dt=dt)
    b = tz.utcoffset(dt=dt)
    if a == b:
        return True
    return False


def _datetimes_contain_all_months(datetimes: list[datetime]) -> bool:
    """Check if there is at least 1 datetime per month of the year."""

    if not all(isinstance(i, datetime) for i in datetimes):
        raise ValueError("All elements of the iterable must be datetime objects.")

    if len(set([i.month for i in datetimes])) < 12:
        return False
    return True


def _datetimes_contain_all_days(datetimes: list[datetime]) -> bool:
    """Check if there is at least 1 datetime per day of month."""
    if not all(isinstance(i, datetime) for i in datetimes):
        raise ValueError("All elements of the iterable must be datetime objects.")
    if len(set([i.day for i in datetimes])) < 31:
        return False
    return True


def _datetimes_contain_all_hours(datetimes: list[datetime]) -> bool:
    """Check if there is at least 1 datetime per hour of day."""
    if not all(isinstance(i, datetime) for i in datetimes):
        raise ValueError("All elements of the iterable must be datetime objects.")
    if len(set([i.hour for i in datetimes])) < 24:
        return False
    return True


# endregion: QUERIES

# region: COLLECTIONS


def _hourly_collection_to_series(
    collection: HourlyContinuousCollection | HourlyDiscontinuousCollection,
) -> pd.Series:
    """Convert a Ladybug HourlyContinuousCollection or HourlyDiscontinuousCollection to a pandas Series.

    Args:
        collection (HourlyContinuousCollection | HourlyDiscontinuousCollection):
            A Ladybug HourlyContinuousCollection or HourlyDiscontinuousCollection object.

    Returns:
        pd.Series:
            A pandas Series object with the datetimes as the index and the values as the data.
    """

    if not isinstance(
        collection, (HourlyContinuousCollection, HourlyDiscontinuousCollection)
    ):
        raise ValueError(
            "Collection must be a HourlyContinuousCollection or HourlyDiscontinuousCollection."
        )

    # convert analysis period to datetimeindex
    idx = pd.DatetimeIndex(collection.analysis_period.datetimes)

    # attempt to make index tz-aware
    if collection.header.metadata.get("time-zone") is not None:
        try:
            idx = idx.tz_localize(
                pytz.FixedOffset(collection.header.metadata["time-zone"] * 60)
            )
        except KeyError:
            pass

    return pd.Series(
        index=idx,
        name=_header_to_tuple(collection.header),
        data=collection.values,
    )


def _series_to_hourly_collection(
    series: pd.Series,
) -> HourlyContinuousCollection | HourlyDiscontinuousCollection:
    """Convert a pandas Series to a Ladybug HourlyContinuousCollection or HourlyDiscontinuousCollection.

    Args:
        series (pd.Series):
            A pandas Series object with a DatetimeIndex and a tuple as the name.

    Returns:
        HourlyContinuousCollection | HourlyDiscontinuousCollection:
            A Ladybug HourlyContinuousCollection or HourlyDiscontinuousCollection object.
    """

    if not isinstance(series.name, tuple):
        raise ValueError("Series name must be a tuple.")
    if len(series.name) != 2:
        raise ValueError("Series name must be a tuple of length 2.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    # convert index to AnalysisPeriod
    ap = _datetimes_to_analysis_period(series.index)

    # create metadata
    metadata = {"source": "pandas"}
    if pd.DatetimeIndex.tzinfo is not None:
        metadata["time-zone"] = (
            series.index.tz.utcoffset(series.index[0]).total_seconds() / 3600
        )

    # construct header
    header = _tuple_to_header(series.name, analysis_period=ap, metadata=metadata)

    # check that the datetimes span at least 1 year
    if not all(
        [
            _datetimes_contain_all_months(series.index),
            _datetimes_contain_all_days(series.index),
            _datetimes_contain_all_hours(series.index),
        ]
    ):
        return HourlyContinuousCollection(
            header=header,
            values=series.values.tolist(),
        )
    datetimes = [
        (dt.month, dt.day, dt.hour, dt.minute, _is_leap_year(dt)) for dt in series.index
    ]
    return HourlyDiscontinuousCollection(
        header=header,
        values=series.values.tolist(),
        datetimes=datetimes,
    )


def _monthly_collection_to_series(collection: MonthlyCollection) -> pd.Series:
    """Convert a Ladybug MonthlyCollection to a pandas Series.

    Args:
        collection (MonthlyCollection):
            A Ladybug MonthlyCollection object.

    Returns:
        pd.Series:
            A pandas Series object with the datetimes as the index and the values as the data.
    """
    if not isinstance(collection, MonthlyCollection):
        raise ValueError("Collection must be a MonthlyCollection.")

    # convert analysis period to datetimeindex
    year = 2016 if collection.header.analysis_period.is_leap_year else 2017
    idx = pd.DatetimeIndex(
        [
            datetime(year, month, 1, 0, 0)
            + timedelta(days=calendar.monthrange(year, month)[1] - 1)
            for month in collection.datetimes
        ]
    )

    # attempt to make index tz-aware
    if collection.header.metadata.get("time-zone") is not None:
        try:
            idx = idx.tz_localize(
                pytz.FixedOffset(collection.header.metadata["time-zone"] * 60)
            )
        except KeyError:
            pass

    return pd.Series(
        index=idx,
        name=_header_to_tuple(collection.header),
        data=collection.values,
    )


def _series_to_monthly_collection(
    series: pd.Series,
) -> MonthlyCollection:
    if not isinstance(series.name, tuple):
        raise ValueError("Series name must be a tuple.")
    if len(series.name) != 2:
        raise ValueError("Series name must be a tuple of length 2.")

    # check for only one value per month
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")
    if series.index.month.value_counts().max() > 1:
        raise ValueError("Series index must contain only one value per month.")

    # create metadata
    metadata = {"source": "pandas"}
    if pd.DatetimeIndex.tzinfo is not None:
        metadata["time-zone"] = (
            series.index.tz.utcoffset(series.index[0]).total_seconds() / 3600
        )

    # construct header
    header = _tuple_to_header(series.name, metadata=metadata)

    datetimes = [month for month in series.index.month]
    return MonthlyCollection(header=header, values=series.values, datetimes=datetimes)


def _daily_collection_to_series(collection: DailyCollection) -> pd.Series:
    """Convert a Ladybug DailyCollection to a pandas Series.

    Args:
        collection (DailyCollection):
            A Ladybug DailyCollection object.

    Returns:
        pd.Series:
            A pandas Series object with the datetimes as the index and the values as the data.
    """

    base_date = date(
        2016 if collection.header.analysis_period.is_leap_year else 2017, 1, 1
    ) - timedelta(days=1)

    idx = pd.DatetimeIndex(
        [base_date + timedelta(days=i) for i in collection.datetimes]
    )
    try:
        idx = idx.tz_localize(
            pytz.FixedOffset(collection.header.metadata["time-zone"] * 60)
        )
    except KeyError:
        pass

    return pd.Series(
        index=idx,
        name=_header_to_tuple(collection.header),
        data=collection.values,
    )


def _series_to_daily_collection(
    series: pd.Series,
) -> DailyCollection:
    """Convert a pandas Series to a Ladybug DailyCollection.

    Args:
        series (pd.Series):
            A pandas Series object with a DatetimeIndex and a tuple as the name.

    Returns:
        DailyCollection:
            A Ladybug DailyCollection object.
    """
    if not isinstance(series.name, tuple):
        raise ValueError("Series name must be a tuple.")
    if len(series.name) != 2:
        raise ValueError("Series name must be a tuple of length 2.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")
    if series.index.day_of_year.value_counts().max() > 1:
        raise ValueError("Series index must contain only one value per day of month.")

    # create datetimes, which are a list of intergers denoting day of the year
    datetimes = series.index.day_of_year.tolist()
    ap = AnalysisPeriod(is_leap_year=_is_leap_year(series.index.year[0]))

    # TODO - modify AP, perhaps. LB doesnt seem to care about this much.

    # create metadata and header
    metadata = {"source": "pandas"}
    if pd.DatetimeIndex.tzinfo is not None:
        metadata["time-zone"] = (
            series.index.tz.utcoffset(series.index[0]).total_seconds() / 3600
        )
    header = _tuple_to_header(series.name, metadata=metadata, analysis_period=ap)

    return DailyCollection(
        header=header, values=series.values.tolist(), datetimes=datetimes
    )


def _monthly_per_hour_collection_to_series(
    collection: MonthlyPerHourCollection,
) -> pd.Series:
    """Convert a Ladybug MonthlyPerHourCollection to a pandas Series.

    Args:
        collection (MonthlyPerHourCollection):
            A Ladybug MonthlyPerHourCollection object.

    Returns:
        pd.Series:
            A pandas Series object with the datetimes as the index and the values as the data.
    """

    if not isinstance(collection, MonthlyPerHourCollection):
        raise ValueError("Collection must be a MonthlyPerHourCollection.")

    # construct index
    year = 2016 if collection.header.analysis_period.is_leap_year else 2017
    idx = pd.DatetimeIndex(
        [
            datetime(year, month, 1, hour, minute)
            for month, hour, minute in collection.datetimes
        ]
    )

    # attempt to make index tz-aware
    if collection.header.metadata.get("time-zone") is not None:
        try:
            idx = idx.tz_localize(
                pytz.FixedOffset(collection.header.metadata["time-zone"] * 60)
            )
        except KeyError:
            pass

    return pd.Series(
        name=_header_to_tuple(collection.header), data=collection.values, index=idx
    )


def _series_to_monthly_per_hour_collection(
    series: pd.Series,
) -> MonthlyPerHourCollection:
    if not isinstance(series.name, tuple):
        raise ValueError("Series name must be a tuple.")
    if len(series.name) != 2:
        raise ValueError("Series name must be a tuple of length 2.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    # check that there are only 24-values per month
    if len(series.values) != 24 * 12:
        raise ValueError("Series index must contain 24 values per month.")
    if len(series.index.month.value_counts()) != 12:
        raise ValueError("Series index must contain 12 months.")
    if series.index.month.value_counts().max() != 24:
        raise ValueError("Series index must contain 24 values per month.")

    # create metadata
    metadata = {"source": "pandas"}
    if pd.DatetimeIndex.tzinfo is not None:
        metadata["time-zone"] = (
            series.index.tz.utcoffset(series.index[0]).total_seconds() / 3600
        )

    # construct header
    header = Header(
        data_type=TYPESDICT[series.name[0].replace(" ", "")](),
        unit=series.name[1],
        analysis_period=AnalysisPeriod(),
        metadata=metadata,
    )

    datetimes = [(dt.month, dt.hour, dt.minute) for dt in series.index]
    return MonthlyPerHourCollection(
        header=header, values=series.values, datetimes=datetimes
    )


def _epw_to_dataframe(epw: EPW) -> pd.DataFrame:
    """Convert a Ladybug EPW object to a pandas DataFrame.

    Args:
        epw (EPW): A Ladybug EPW object.

    Returns:
        pd.DataFrame: A pandas DataFrame object with the datetimes as the index and the values as the data.
    """
    serieses = [_hourly_collection_to_series(i) for i in epw._data]
    return pd.concat(serieses, axis=1)


def _dataframe_to_epw(df: pd.DataFrame) -> EPW:
    """Convert a pandas DataFrame to a Ladybug EPW object.

    Args:
        df (pd.DataFrame): A pandas DataFrame object with the datetimes as the index and the values as the data.

    Returns:
        EPW: A Ladybug EPW object.
    """
    raise NotImplementedError("Not yet implemented.")


# endregion: COLLECTIONS
