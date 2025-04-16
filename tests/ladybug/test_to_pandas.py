from datetime import date, datetime, time, timedelta

import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datacollection import (
    MonthlyPerHourCollection,
)
from ladybug.dt import Date, DateTime, Time
from ladybug.epw import EPW
from ladybug.header import Header

from kvrrj.ladybug.to_pandas import to_pandas

from .. import EPW_OBJ

# region: DATETIME


def test_lb_analysis_period_to_pandas():
    ap = AnalysisPeriod()
    res = to_pandas(ap)
    assert isinstance(res, pd.DatetimeIndex)


def test_lb_date_to_pandas():
    dt = Date(month=3, day=31, leap_year=True)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)

    dt = Date(month=3, day=31, leap_year=False)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)


def test_lb_time_to_pandas():
    dt = Time(hour=12, minute=30)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)


def test_lb_datetime_to_pandas():
    dt = DateTime(month=3, day=31, hour=12, minute=30, leap_year=True)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)


def test_py_date_to_pandas():
    dt = date(2025, 3, 31)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)


def test_py_time_to_pandas():
    dt = time(12, 30)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)


def test_py_datetime_to_pandas():
    dt = datetime(2025, 3, 31, 12, 30)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timestamp)


def test_py_timedelta_to_pandas():
    dt = timedelta(days=1, hours=12, minutes=30)
    res = to_pandas(dt)
    assert isinstance(res, pd.Timedelta)


# region: DATETIME


def test_header_to_pandas():
    obj = EPW.dry_bulb_temperature.header
    res = to_pandas(obj)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res == ("Dry Bulb Temperature", "C")


def test_hourlycontinuouscollection_to_pandas():
    obj = EPW_OBJ.dry_bulb_temperature
    res = to_pandas(obj)
    assert isinstance(res, pd.Series)


def test_hourlydiscontinuouscollection_to_pandas():
    obj = EPW_OBJ.dry_bulb_temperature.filter_by_analysis_period(
        AnalysisPeriod(st_hour=3)
    )
    res = to_pandas(obj)
    assert isinstance(res, pd.Series)


def test_monthlycollection_to_pandas():
    obj = EPW_OBJ.monthly_ground_temperature[0.5]
    res = to_pandas(obj)
    assert isinstance(res, pd.Series)
    assert len(res) == 12


def test_dailycollection_to_pandas():
    obj = EPW_OBJ.dry_bulb_temperature.total_daily()
    res = to_pandas(obj)
    assert isinstance(res, pd.Series)
    assert len(res) == 365


def test_monthlyperhourcollection_to_pandas():
    a_per = AnalysisPeriod(6, 1, 0, 7, 31, 23)
    vals = [20] * 24 + [25] * 24
    obj = MonthlyPerHourCollection(
        header=Header(
            data_type=type(EPW_OBJ.dry_bulb_temperature.header.data_type)(),
            unit="C",
            analysis_period=a_per,
        ),
        values=vals,
        datetimes=a_per.months_per_hour,
    )
    res = to_pandas(obj)
    assert isinstance(res, pd.Series)


def test_epw_to_pandas():
    obj = EPW_OBJ
    res = to_pandas(obj)
    assert isinstance(res, pd.DataFrame)
