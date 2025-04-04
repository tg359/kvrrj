from __future__ import annotations  # necessary for str type hinting of numpy arrays

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from ladybug.dt import DateTime as lbdatetime

from src.kvrrj.ladybug.dt import to_lb_datetime, to_py_datetime


def test_to_lb_datetime_from_datetime():
    dt = datetime(2025, 3, 31, 12, 30)
    lb_dt = to_lb_datetime(dt)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.month == 3
    assert lb_dt.day == 31
    assert lb_dt.hour == 12
    assert lb_dt.minute == 30


def test_to_lb_datetime_from_date():
    dt = date(2025, 3, 31)
    lb_dt = to_lb_datetime(dt)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.month == 3
    assert lb_dt.day == 31
    assert lb_dt.hour == 0
    assert lb_dt.minute == 0
    assert not lb_dt.leap_year


def test_to_lb_datetime_from_string():
    dt_str = "2025-03-31 12:30:00"
    lb_dt = to_lb_datetime(dt_str)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.month == 3
    assert lb_dt.day == 31
    assert lb_dt.hour == 12
    assert lb_dt.minute == 30


def test_to_lb_datetime_from_dict():
    dt_dict = {"month": 3, "day": 31, "hour": 12, "minute": 30, "leap_year": False}
    lb_dt = to_lb_datetime(dt_dict)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.month == 3
    assert lb_dt.day == 31
    assert lb_dt.hour == 12
    assert lb_dt.minute == 30
    assert not lb_dt.leap_year


def test_to_lb_datetime_from_pandas_timestamp():
    pd_ts = pd.Timestamp("2025-03-31 12:30:00")
    lb_dt = to_lb_datetime(pd_ts)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.month == 3
    assert lb_dt.day == 31
    assert lb_dt.hour == 12
    assert lb_dt.minute == 30


def test_to_lb_datetime_from_numpy_datetime64():
    np_dt = np.datetime64("2025-03-31T12:30:00")
    lb_dt = to_lb_datetime(np_dt)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.month == 3
    assert lb_dt.day == 31
    assert lb_dt.hour == 12
    assert lb_dt.minute == 30


def test_to_lb_datetime_from_list():
    dt_list = [
        datetime(2025, 3, 31, 12, 30),
        datetime(2025, 3, 31, 13, 30),
    ]
    lb_dts = to_lb_datetime(dt_list)
    assert isinstance(lb_dts, np.ndarray)
    assert len(lb_dts) == 2
    assert all(isinstance(dt, lbdatetime) for dt in lb_dts)


def test_from_lb_datetime_to_datetime():
    lb_dt = lbdatetime(3, 31, 12, 30, leap_year=False)
    dt = to_py_datetime(lb_dt)
    assert isinstance(dt, datetime)
    assert dt.month == 3
    assert dt.day == 31
    assert dt.hour == 12
    assert dt.minute == 30


def test_from_lb_datetime_from_list():
    lb_dt_list = [
        lbdatetime(3, 31, 12, 30, leap_year=False),
        lbdatetime(3, 31, 13, 30, leap_year=False),
    ]
    dts = to_py_datetime(lb_dt_list)
    assert isinstance(dts, np.ndarray)
    assert len(dts) == 2
    assert all(isinstance(dt, datetime) for dt in dts)


def test_to_lb_datetime_invalid_type():
    with pytest.raises(NotImplementedError):
        to_lb_datetime(12345)


def test_from_lb_datetime_invalid_type():
    with pytest.raises(NotImplementedError):
        to_py_datetime(12345)
