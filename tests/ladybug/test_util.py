from datetime import datetime

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    DailyCollection,
    HourlyContinuousCollection,
    HourlyDiscontinuousCollection,
    MonthlyCollection,
    MonthlyPerHourCollection,
)
from ladybug.epw import EPW
from ladybug.header import Header

from kvrrj.ladybug.util import (
    _analysis_period_to_string,
    _daily_collection_to_series,
    _dataframe_to_epw,
    _datetimes_contain_all_days,
    _datetimes_contain_all_hours,
    _datetimes_contain_all_months,
    _datetimes_to_analysis_period,
    _epw_to_dataframe,
    _header_to_tuple,
    _hourly_collection_to_series,
    _is_leap_year,
    _location_to_string,
    _monthly_collection_to_series,
    _monthly_per_hour_collection_to_series,
    _series_to_daily_collection,
    _series_to_hourly_collection,
    _series_to_monthly_collection,
    _series_to_monthly_per_hour_collection,
    _string_to_analysis_period,
    _string_to_location,
    _tuple_to_header,
)

from .. import EPW_OBJ

# region: LOCATION


def test_location_to_string():
    obj = EPW_OBJ.location
    result = _location_to_string(obj)
    assert isinstance(result, str)
    assert result == "London, UK"


def test_string_to_location():
    obj = "City, Country (10°, 12°, 33m, UTC+06)"
    result = _string_to_location(obj)
    assert isinstance(result, str)


# endregion: LOCATION

# region: ANALYSISPERIOD


def test_analysis_period_to_string():
    ap = AnalysisPeriod()
    result = _analysis_period_to_string(ap, save_path=True)
    assert result == "0101_1231_00_23_01_C"
    result = _analysis_period_to_string(ap, save_path=False)
    assert result == "Jan 01 to Dec 31 between 00:00 and 23:59 every hour (C)"


def test_string_to_analysis_period():
    string = "0101_1231_00_23_01_C"
    result = _string_to_analysis_period(string)
    assert isinstance(result, AnalysisPeriod)

    string = "0301_1231_04_22_02_L"
    result = _string_to_analysis_period(string)
    assert isinstance(result, AnalysisPeriod)
    assert result.is_leap_year
    assert result.st_month == 3
    assert result.end_month == 12
    assert result.st_hour == 4
    assert result.end_hour == 22
    assert result.timestep == 2

    string = "Jan 01 to Dec 31 between 00:00 and 23:59 every hour (C)"
    result = _string_to_analysis_period(string)
    assert isinstance(result, AnalysisPeriod)

    string = "Mar 01 to Dec 31 between 04:00 and 22:59 every 30-minutes (L)"
    result = _string_to_analysis_period(string)
    assert isinstance(result, AnalysisPeriod)
    assert result.is_leap_year
    assert result.st_month == 3
    assert result.end_month == 12
    assert result.st_hour == 4
    assert result.end_hour == 22
    assert result.timestep == 2


def test_datetimes_to_analysis_period():
    datetimes = pd.date_range(start="2017-01-01 00:00:00", periods=8760, freq="h")
    result = _datetimes_to_analysis_period(datetimes)
    assert isinstance(result, AnalysisPeriod)
    assert result.st_month == 1
    assert result.end_month == 12
    assert result.st_day == 1
    assert result.end_day == 31
    assert result.st_hour == 0
    assert result.end_hour == 23
    assert result.timestep == 1


def test_analysis_period_to_datetimes():
    ap = AnalysisPeriod()
    result = _datetimes_to_analysis_period(ap)
    assert isinstance(result, list)
    assert all(isinstance(dt, datetime) for dt in result)


# endregion: ANALYSISPERIOD

# region: HEADER


def test_header_to_tuple():
    obj = EPW_OBJ.dry_bulb_temperature.header
    result = _header_to_tuple(obj)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result == ("Dry Bulb Temperature", "C")


def test_tuple_to_header():
    obj = ("Dry Bulb Temperature", "C")
    result = _tuple_to_header(obj)
    assert isinstance(result, Header)
    assert result.name == "Dry Bulb Temperature"
    assert result.units == "C"
    assert result.analysis_period == AnalysisPeriod()


# endregion: HEADER


# region: QUERIES


def test_is_leap_year():
    assert _is_leap_year(2016) is True
    assert _is_leap_year(2017) is False
    assert _is_leap_year(2020) is True
    assert _is_leap_year(2021) is False
    assert _is_leap_year(1600) is True
    assert _is_leap_year(1800) is False


def test_datetimes_contain_all_months():
    all_months = (
        pd.date_range("2017-01-01 00:00:00", freq="MS", periods=12)
        .to_pydatetime()
        .tolist()
    )
    some_months = (
        pd.date_range("2017-01-01 00:00:00", freq="MS", periods=8)
        .to_pydatetime()
        .tolist()
    )
    assert _datetimes_contain_all_months(all_months) is True
    assert _datetimes_contain_all_months(some_months) is False


def test_datetimes_contain_all_days():
    all_days = (
        pd.date_range("2017-01-01 00:00:00", freq="D", periods=365)
        .to_pydatetime()
        .tolist()
    )
    some_days = (
        pd.date_range("2017-01-01 00:00:00", freq="D", periods=15)
        .to_pydatetime()
        .tolist()
    )
    assert _datetimes_contain_all_days(all_days) is True
    assert _datetimes_contain_all_days(some_days) is False


def test_datetimes_contain_all_hours():
    all_hours = (
        pd.date_range("2017-01-01 00:00:00", freq="H", periods=24)
        .to_pydatetime()
        .tolist()
    )
    some_hours = (
        pd.date_range("2017-01-01 00:00:00", freq="H", periods=12)
        .to_pydatetime()
        .tolist()
    )
    assert _datetimes_contain_all_hours(all_hours) is True
    assert _datetimes_contain_all_hours(some_hours) is False


# endregion: QUERIES

# region: COLLECTIONS


def test_hourly_collection_to_series():
    obj = EPW_OBJ.dry_bulb_temperature
    result = _hourly_collection_to_series(obj)
    assert isinstance(result, pd.Series)


def test_series_to_hourly_collection():
    obj = _hourly_collection_to_series(EPW_OBJ.dry_bulb_temperature)
    result = _series_to_hourly_collection(obj)
    assert isinstance(result, HourlyContinuousCollection)

    obj = _hourly_collection_to_series(EPW_OBJ.dry_bulb_temperature).between_time(
        "00:00", "22:59"
    )
    result = _series_to_hourly_collection(obj)
    assert isinstance(result, HourlyDiscontinuousCollection)


def test_monthly_collection_to_series():
    obj = EPW_OBJ.monthly_ground_temperature[0.5]
    result = _monthly_collection_to_series(obj)
    assert isinstance(result, pd.Series)


def test_series_to_monthly_collection():
    obj = _monthly_collection_to_series(EPW_OBJ.monthly_ground_temperature[0.5])
    result = _series_to_monthly_collection(obj)
    assert isinstance(result, MonthlyCollection)


def test_daily_collection_to_series():
    obj = EPW_OBJ.dry_bulb_temperature.total_daily()
    result = _daily_collection_to_series(obj)
    assert isinstance(result, DailyCollection)


def test_series_to_daily_collection():
    obj = _daily_collection_to_series(EPW_OBJ.dry_bulb_temperature.total_daily())
    result = _series_to_daily_collection(obj)
    assert isinstance(result, pd.Series)


def test_monthly_per_hour_collection_to_series():
    obj = EPW_OBJ.dry_bulb_temperature.average_monthly_per_hour()
    result = _monthly_per_hour_collection_to_series(obj)
    assert isinstance(result, MonthlyPerHourCollection)


def test_series_to_monthly_per_hour_collection():
    obj = _monthly_per_hour_collection_to_series(
        EPW_OBJ.dry_bulb_temperature.average_monthly_per_hour()
    )
    result = _series_to_monthly_per_hour_collection(obj)
    assert isinstance(result, MonthlyPerHourCollection)


def test_epw_to_dataframe():
    obj = EPW_OBJ
    result = _epw_to_dataframe(obj)
    assert isinstance(result, pd.DataFrame)


def test_dataframe_to_epw():
    obj = _epw_to_dataframe(EPW_OBJ)
    result = _dataframe_to_epw(obj)
    assert isinstance(result, EPW)


# endregion: COLLECTIONS
