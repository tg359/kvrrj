# from datetime import datetime

# import pandas as pd
# import pytest

# from kvrrj.ladybug.analysisperiod import (
#     AnalysisPeriod,
#     _analysis_period_to_string,
#     timedelta,
#     to_lb_analysis_period,
# )


# def test_to_lb_analysis_period_from_analysis_period():
#     ap = AnalysisPeriod()
#     result = to_lb_analysis_period(ap)
#     assert isinstance(result, AnalysisPeriod)
#     assert result.st_month == 1
#     assert result.end_month == 12
#     assert result.st_day == 1
#     assert result.end_day == 31
#     assert result.st_hour == 0
#     assert result.end_hour == 23
#     assert result.timestep == 1


# def test_to_lb_analysis_period_from_timedelta():
#     td = timedelta(hours=1)
#     result = to_lb_analysis_period(td, is_leap_year=False)
#     assert isinstance(result, AnalysisPeriod)
#     assert result.timestep == 1
#     assert not result.is_leap_year


# def test_to_lb_analysis_period_from_string_save_path_format():
#     ap_str = "0101_0131_00_23_1"
#     result = to_lb_analysis_period(ap_str)
#     assert isinstance(result, AnalysisPeriod)
#     assert result.st_month == 1
#     assert result.end_month == 1
#     assert result.st_day == 1
#     assert result.end_day == 31
#     assert result.st_hour == 0
#     assert result.end_hour == 23
#     assert result.timestep == 1


# def test_to_lb_analysis_period_from_string_human_readable_format():
#     ap_str = "Jan 1 to Jan 31 between 00:00 and 23:59 every hour"
#     result = to_lb_analysis_period(ap_str)
#     assert isinstance(result, AnalysisPeriod)
#     assert result.st_month == 1
#     assert result.end_month == 1
#     assert result.st_day == 1
#     assert result.end_day == 31
#     assert result.st_hour == 0
#     assert result.end_hour == 23
#     assert result.timestep == 1


# def test_to_lb_analysis_period_from_list():
#     datetimes = [
#         datetime(2025, 1, 1, 0, 0),
#         datetime(2025, 1, 1, 1, 0),
#         datetime(2025, 1, 1, 2, 0),
#     ]
#     result = to_lb_analysis_period(datetimes)
#     assert isinstance(result, AnalysisPeriod)
#     assert result.st_month == 1
#     assert result.end_month == 1
#     assert result.st_day == 1
#     assert result.end_day == 1
#     assert result.st_hour == 0
#     assert result.end_hour == 2
#     assert result.timestep == 1


# def test_to_lb_analysis_period_from_pandas_datetimeindex():
#     datetimes = pd.date_range(start="2017-01-01 00:00:00", periods=8760, freq="h")
#     result = to_lb_analysis_period(datetimes)
#     assert isinstance(result, AnalysisPeriod)
#     assert result.st_month == 1
#     assert result.end_month == 12
#     assert result.st_day == 1
#     assert result.end_day == 31
#     assert result.st_hour == 0
#     assert result.end_hour == 23
#     assert result.timestep == 1


# def test_to_lb_analysis_period_invalid_type():
#     with pytest.raises(NotImplementedError):
#         to_lb_analysis_period(12345)


# def test_to_string_save_path_format():
#     ap = AnalysisPeriod()
#     result = _analysis_period_to_string(ap, save_path=True)
#     assert result == "0101_1231_00_23_01"


# def test_to_string_human_readable_format():
#     ap = AnalysisPeriod()
#     result = _analysis_period_to_string(ap, save_path=False)
#     assert result == "Jan 01 to Dec 31 between 00:00 and 23:59 every hour"


# def test_to_string_invalid_input():
#     with pytest.raises(ValueError):
#         _analysis_period_to_string("invalid_input")
