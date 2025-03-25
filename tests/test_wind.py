from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kvrrj.wind import (
    AnalysisPeriod,
    HourlyContinuousCollection,
    Wind,
    WindRose,
    analysis_period_to_datetimes,
    analysis_period_to_string,
)

from . import EPW_OBJ


def test_wind_good():
    w = Wind(
        wind_speeds=list(EPW_OBJ.wind_speed.values),
        wind_directions=list(EPW_OBJ.wind_direction.values),
        datetimes=analysis_period_to_datetimes(
            EPW_OBJ.wind_direction.header.analysis_period
        ),
        height_above_ground=10,
    )
    assert isinstance(w, Wind)


def test_wind_bad():
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    with pytest.raises(ValueError):
        # height at 0
        Wind(
            wind_speeds=ws,
            wind_directions=wd,
            datetimes=datetimes,
            height_above_ground=0,
        )
    with pytest.raises(ValueError):
        # speed and direction not same length
        Wind(
            wind_speeds=ws[0:10],
            wind_directions=wd,
            datetimes=datetimes,
            height_above_ground=10,
        )
    with pytest.raises(ValueError):
        # index not same length as speed and direction
        Wind(
            wind_speeds=ws[0:1],
            wind_directions=wd[0:1],
            datetimes=datetimes,
            height_above_ground=10,
        )
    with pytest.raises(ValueError):
        # index not datetime
        Wind(
            wind_speeds=[1, 1, 1, 1, 1],
            wind_directions=[1, 1, 1, 1, 1],
            datetimes=[datetimes[0]] * 5,
            height_above_ground=10,
        )
    with pytest.raises(ValueError):
        # speed contains NaN
        Wind(
            wind_speeds=np.where(np.array(ws) < 5, np.nan, ws).tolist(),
            wind_directions=wd,
            datetimes=datetimes,
            height_above_ground=0,
        )
    with pytest.raises(ValueError):
        # direction contains NaN
        Wind(
            wind_speeds=ws,
            wind_directions=np.where(np.array(wd) < 180, np.nan, wd).tolist(),
            datetimes=datetimes,
            height_above_ground=0,
        )
    with pytest.raises(ValueError):
        # direction contains negative values
        Wind(
            wind_speeds=ws,
            wind_directions=np.where(np.array(wd) < 180, -1, wd).tolist(),
            datetimes=datetimes,
            height_above_ground=0,
        )


def test_wind_from_epw():
    assert isinstance(Wind.from_epw(EPW_OBJ), Wind)
    assert isinstance(Wind.from_epw(EPW_OBJ.file_path), Wind)
    assert Wind.from_epw(EPW_OBJ.file_path).source == Path(EPW_OBJ.file_path).name


def test_wind_to_from_dict():
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create dict
    d = {
        "wind_speeds": ws,
        "wind_directions": wd,
        "datetimes": datetimes,
        "height_above_ground": 10,
    }
    assert isinstance(Wind.from_dict(d), Wind)
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    assert isinstance(w.to_dict(), dict)


def test_wind_to_from_json():
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    json_str = w.to_json()
    assert isinstance(json_str, str)
    jw = Wind.from_json(json_str)
    assert isinstance(jw, Wind)
    assert jw == w


def test_wind_to_from_json_file(tmp_path):
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    tempfile = tmp_path / "pytest_wind.json"
    assert isinstance(w.to_json_file(tempfile), Path)
    assert isinstance(Wind.from_json_file(tempfile), Wind)


def test_wind_to_from_dataframe():
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    df = w.to_dataframe()

    assert isinstance(
        Wind.from_dataframe(
            df=df,
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=10,
        ),
        Wind,
    )

    with pytest.raises(ValueError):
        Wind.from_dataframe(
            df="not_a_dataframe",
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=10,
        )
    with pytest.raises(ValueError):
        Wind.from_dataframe(
            df=df.reset_index(drop=True),
            wind_speed_column="speed",
            wind_direction_column="direction",
            height_above_ground=10,
        )


def test_wind_to_from_csv_file(tmp_path):
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    tempfile = tmp_path / "pytest_wind.csv"
    assert isinstance(w.to_csv_file(tempfile), Path)
    assert isinstance(Wind.from_csv_file(tempfile), Wind)


def test_wind_from_uv():
    TEST_WIND = Wind.from_epw(EPW_OBJ)
    u, v = TEST_WIND.uv.values.T
    with pytest.warns(UserWarning):
        assert isinstance(Wind.from_uv(u=u, v=v, datetimes=TEST_WIND.datetimes), Wind)


def test__is_single_year_hourly():
    # Test with a non-leap year (8760 hours)
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8760)]
    assert Wind._is_single_year_hourly(datetimes) is True

    # Test with a leap year (8784 hours)
    datetimes = [datetime(2020, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8784)]
    assert Wind._is_single_year_hourly(datetimes) is True

    # Test with less than 8760 hours
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8759)]
    assert Wind._is_single_year_hourly(datetimes) is False

    # Test with more than 8784 hours
    datetimes = [datetime(2020, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8785)]
    assert Wind._is_single_year_hourly(datetimes) is False


def test_ws_datacollection():
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    # convert to datacollection
    assert isinstance(w.ws_datacollection, HourlyContinuousCollection)


def test_wd_datacollection():
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    # convert to datacollection
    assert isinstance(w.wd_datacollection, HourlyContinuousCollection)


def test_to_lb_windrose():
    # create data
    ws = list(EPW_OBJ.wind_speed.values)
    wd = list(EPW_OBJ.wind_direction.values)
    datetimes = analysis_period_to_datetimes(
        EPW_OBJ.wind_direction.header.analysis_period
    )
    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    # convert to WindRose
    wind_rose = w.to_lb_windrose(direction_count=16)
    assert isinstance(wind_rose, WindRose)
    assert wind_rose._direction_count == 16


def test_calm():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(5)]
    ws = [0, 0.5, 0.1, 0.15, 0.2]
    wd = [0, 45, 90, 180, 270]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )
    # test calm proportion
    assert w.calm() == 2 / 5  # Two timesteps have wind speed ≤ 1e-10
    assert w.calm(threshold=0.15) == 3 / 5  # Three timesteps have wind speed ≤ 0.15


def test_percentile():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(5)]
    ws = [0, 1, 2, 3, 4]
    wd = [0, 45, 90, 180, 270]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # test percentiles
    assert w.percentile(0.0) == 0  # Minimum wind speed
    assert w.percentile(0.25) == 1  # 25th percentile
    assert w.percentile(0.5) == 2  # Median wind speed
    assert w.percentile(0.75) == 3  # 75th percentile
    assert w.percentile(1.0) == 4  # Maximum wind speed


def test_to_height():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(5)]
    ws = [1, 2, 3, 4, 5]
    wd = [0, 45, 90, 180, 270]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # translate to a different height using log function
    translated_wind_log = w.to_height(target_height=20, log_function=True)
    assert isinstance(translated_wind_log, Wind)
    assert translated_wind_log.height_above_ground == 20
    assert translated_wind_log.source == f"{w.source} translated to 20m"

    # translate to a different height using power function
    translated_wind_pow = w.to_height(target_height=20, log_function=False)
    assert isinstance(translated_wind_pow, Wind)
    assert translated_wind_pow.height_above_ground == 20
    assert translated_wind_pow.source == f"{w.source} translated to 20m"

    # ensure wind directions remain unchanged
    assert translated_wind_log.wind_directions == w.wind_directions
    assert translated_wind_pow.wind_directions == w.wind_directions

    # ensure wind speeds are adjusted
    assert translated_wind_log.wind_speeds != w.wind_speeds
    assert translated_wind_pow.wind_speeds != w.wind_speeds

    # test invalid target height
    with pytest.raises(ValueError):
        w.to_height(target_height=0.05)  # target height below 0.1m


def test_filter_by_boolean_mask():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(5)]
    ws = [1, 2, 3, 4, 5]
    wd = [0, 45, 90, 180, 270]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # valid mask
    mask = [True, False, True, False, True]
    filtered_wind = w.filter_by_boolean_mask(mask)
    assert isinstance(filtered_wind, Wind)
    assert filtered_wind.wind_speeds == [1, 3, 5]
    assert filtered_wind.wind_directions == [0, 90, 270]
    assert filtered_wind.datetimes == [datetimes[0], datetimes[2], datetimes[4]]
    assert filtered_wind.source == f"{w.source} (filtered)"

    # mask with all True
    mask = [True, True, True, True, True]
    filtered_wind = w.filter_by_boolean_mask(mask)
    assert filtered_wind == w  # Should return the same object

    # mask with no True values
    mask = [False, False, False, False, False]
    with pytest.raises(
        ValueError, match="No data remains within the given boolean filters."
    ):
        w.filter_by_boolean_mask(mask)

    # mask with incorrect length
    mask = [True, False]
    with pytest.raises(
        ValueError,
        match="The length of the boolean mask must match the length of the current object.",
    ):
        w.filter_by_boolean_mask(mask)


def test_filter_by_analysis_period():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(24)]
    ws = [1] * 24
    wd = [90] * 24

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # create an analysis period for a specific time range
    analysis_period = AnalysisPeriod(
        st_month=1, st_day=1, st_hour=0, end_month=1, end_day=1, end_hour=12
    )

    # filter by analysis period
    filtered_wind = w.filter_by_analysis_period(analysis_period)

    # check the filtered results
    assert isinstance(filtered_wind, Wind)
    assert len(filtered_wind.datetimes) == 13  # 0:00 to 12:00 inclusive
    assert filtered_wind.wind_speeds == ws[:13]
    assert filtered_wind.wind_directions == wd[:13]
    assert (
        filtered_wind.source
        == f"{w.source} (filtered to {analysis_period_to_string(analysis_period)})"
    )

    # test with an analysis period that doesn't match any data
    analysis_period_empty = AnalysisPeriod(
        st_month=2, st_day=1, st_hour=0, end_month=2, end_day=1, end_hour=12
    )
    with pytest.raises(
        ValueError, match="No data remains within the given boolean filters."
    ):
        w.filter_by_analysis_period(analysis_period_empty)

    # test with the full analysis period
    full_analysis_period = AnalysisPeriod()
    filtered_wind_full = w.filter_by_analysis_period(full_analysis_period)
    assert filtered_wind_full == w  # Should return the same object


def test_filter_by_time():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(48)]
    ws = [1] * 48
    wd = [90] * 48

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # filter by specific years
    filtered_wind = w.filter_by_time(years=(2021,))
    assert isinstance(filtered_wind, Wind)
    assert all(dt.year == 2021 for dt in filtered_wind.datetimes)

    # filter by specific months
    filtered_wind = w.filter_by_time(months=(1,))
    assert isinstance(filtered_wind, Wind)
    assert all(dt.month == 1 for dt in filtered_wind.datetimes)

    # filter by specific days
    filtered_wind = w.filter_by_time(days=(1,))
    assert isinstance(filtered_wind, Wind)
    assert all(dt.day == 1 for dt in filtered_wind.datetimes)

    # filter by specific hours
    filtered_wind = w.filter_by_time(hours=(0, 1, 2))
    assert isinstance(filtered_wind, Wind)
    assert all(dt.hour in (0, 1, 2) for dt in filtered_wind.datetimes)

    # filter by multiple criteria
    filtered_wind = w.filter_by_time(months=(1,), days=(1,), hours=(0, 1, 2))
    assert isinstance(filtered_wind, Wind)
    assert all(
        dt.month == 1 and dt.day == 1 and dt.hour in (0, 1, 2)
        for dt in filtered_wind.datetimes
    )

    # filter with no matching data
    with pytest.raises(
        ValueError, match="No data remains within the given boolean filters."
    ):
        w.filter_by_time(months=(2,))

    # filter with all default values (should return the same object)
    filtered_wind = w.filter_by_time()
    assert filtered_wind == w


def test_filter_by_direction():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(5)]
    ws = [1, 2, 3, 4, 5]
    wd = [0, 90, 180, 270, 360]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # filter within a range (0 to 180 degrees)
    filtered_wind = w.filter_by_direction(left_angle=0, right_angle=180)
    assert isinstance(filtered_wind, Wind)
    assert filtered_wind.wind_directions == [90, 180]
    assert filtered_wind.wind_speeds == [2, 3]
    assert filtered_wind.datetimes == [datetimes[1], datetimes[2]]

    # filter with wrap-around range (270 to 90 degrees)
    filtered_wind = w.filter_by_direction(left_angle=270, right_angle=90)
    assert isinstance(filtered_wind, Wind)
    assert filtered_wind.wind_directions == [0, 90, 0]
    assert filtered_wind.wind_speeds == [1, 2, 5]
    assert filtered_wind.datetimes == [datetimes[0], datetimes[1], datetimes[4]]

    # filter with exclusive right edge
    filtered_wind = w.filter_by_direction(left_angle=0, right_angle=180, right=False)
    assert isinstance(filtered_wind, Wind)
    assert filtered_wind.wind_directions == [0, 90, 0]
    assert filtered_wind.wind_speeds == [1, 2, 5]
    assert filtered_wind.datetimes == [datetimes[0], datetimes[1], datetimes[4]]

    # filter with identical left and right angles (should raise ValueError)
    with pytest.raises(ValueError, match="Angle limits cannot be identical."):
        w.filter_by_direction(left_angle=90, right_angle=90)

    # filter with invalid angle range (left_angle < 0 or right_angle > 360)
    with pytest.raises(
        ValueError, match="Angle limits must be between 0 and 360 degrees."
    ):
        w.filter_by_direction(left_angle=-10, right_angle=90)
    with pytest.raises(
        ValueError, match="Angle limits must be between 0 and 360 degrees."
    ):
        w.filter_by_direction(left_angle=0, right_angle=370)

    # filter with full range (0 to 360 degrees, should return the same object)
    filtered_wind = w.filter_by_direction(left_angle=0, right_angle=360)
    assert filtered_wind == w


def test_filter_by_speed():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(5)]
    ws = [1, 2, 3, 4, 5]
    wd = [0, 45, 90, 180, 270]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # filter with valid range
    filtered_wind = w.filter_by_speed(min_speed=2, max_speed=4)
    assert isinstance(filtered_wind, Wind)
    assert filtered_wind.wind_speeds == [3, 4]
    assert filtered_wind.wind_directions == [90, 180]
    assert filtered_wind.datetimes == [datetimes[2], datetimes[3]]

    # filter with exclusive right edge
    filtered_wind = w.filter_by_speed(min_speed=2, max_speed=4, right=False)
    assert isinstance(filtered_wind, Wind)
    assert filtered_wind.wind_speeds == [2, 3]

    # filter with min_speed = 0 and max_speed = np.inf (should return the same object)
    filtered_wind = w.filter_by_speed(min_speed=0, max_speed=np.inf)
    assert filtered_wind == w

    # filter with no matching data
    with pytest.raises(
        ValueError, match="No data remains within the given boolean filters."
    ):
        w.filter_by_speed(min_speed=6, max_speed=10)

    # filter with invalid min_speed (negative value)
    with pytest.raises(ValueError, match="min_speed cannot be negative."):
        w.filter_by_speed(min_speed=-1, max_speed=4)

    # filter with invalid range (max_speed <= min_speed)
    with pytest.raises(ValueError, match="min_speed must be less than max_speed."):
        w.filter_by_speed(min_speed=4, max_speed=4)
    with pytest.raises(ValueError, match="min_speed must be less than max_speed."):
        w.filter_by_speed(min_speed=5, max_speed=4)


def test_resample():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(24)]
    ws = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]
    wd = [
        0,
        15,
        30,
        45,
        60,
        75,
        90,
        105,
        120,
        135,
        150,
        165,
        180,
        195,
        210,
        225,
        240,
        255,
        270,
        285,
        300,
        315,
        330,
        345,
    ]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # resample to 6-hour intervals
    with pytest.warns(UserWarning):
        resampled_wind = w.resample("6h")
    assert isinstance(resampled_wind, Wind)
    assert len(resampled_wind.datetimes) == 4  # 24 hours resampled to 6-hour intervals
    assert resampled_wind.height_above_ground == w.height_above_ground
    assert resampled_wind.source == f"{w.source} (resampled to 6h)"

    # ensure wind speeds and directions are resampled correctly
    assert resampled_wind.wind_speeds == [
        3.5,
        9.5,
        15.5,
        21.5,
    ]  # Mean of each 6-hour block
    assert len(resampled_wind.wind_directions) == 4  # Circular mean applied

    # test with invalid resampling rule (up-sampling)
    with pytest.warns(UserWarning):
        with pytest.raises(
            ValueError, match="Resampling can only be used to downsample."
        ):
            w.resample("10min")  # Attempting to up-sample to 10-min intervals


def test_direction_bins():
    # Test with valid number of directions
    bins = Wind._direction_bins(4)
    expected_bins = [
        [315.0, 45.0],
        [45.0, 135.0],
        [135.0, 225.0],
        [225.0, 315.0],
    ]
    assert bins == expected_bins

    bins = Wind._direction_bins(8)
    assert len(bins) == 8  # Ensure correct number of bins
    assert all(len(bin) == 2 for bin in bins)  # Each bin should have two edges

    # Test with invalid number of directions (<= 2)
    with pytest.raises(ValueError, match="directions must be > 2."):
        Wind._direction_bins(2)

    with pytest.raises(ValueError, match="directions must be > 2."):
        Wind._direction_bins(1)

    # Test with a large number of directions
    bins = Wind._direction_bins(360)
    assert len(bins) == 360  # Ensure correct number of bins
    assert all(len(bin) == 2 for bin in bins)  # Each bin should have two edges
    assert bins[0] == [359.5, 0.5]  # Check the first bin
    assert bins[-1] == [358.5, 359.5]  # Check the last bin


def test_bin_direction_data():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8)]
    ws = [1, 2, 3, 4, 5, 6, 7, 8]
    wd = [0, 45, 90, 135, 180, 225, 270, 315]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # test with default parameters
    binned_data = w._bin_direction_data(directions=4, right=True)
    assert len(binned_data) == len(ws)
    assert binned_data == [
        (315.0, 45.0),
        (315.0, 45.0),
        (45.0, 135.0),
        (45.0, 135.0),
        (135.0, 225.0),
        (135.0, 225.0),
        (225.0, 315.0),
        (225.0, 315.0),
    ]

    # test with `right=False`
    binned_data = w._bin_direction_data(directions=4, right=False)
    assert binned_data == [
        (315.0, 45.0),
        (45.0, 135.0),
        (45.0, 135.0),
        (135.0, 225.0),
        (135.0, 225.0),
        (225.0, 315.0),
        (225.0, 315.0),
        (315.0, 45.0),
    ]

    # test with invalid number of directions
    with pytest.raises(ValueError, match="directions must be > 2."):
        w._bin_direction_data(directions=2)


def test_bin_other_data():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8)]
    ws = [1, 2, 3, 4, 5, 6, 7, 8]
    wd = [0, 45, 90, 135, 180, 225, 270, 315]
    other_data = [1, 2, 3, 4, 5, 6, 7, 8]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # test with defaults
    binned_data = w._bin_other_data()
    assert len(binned_data) == len(ws)
    assert all(isinstance(bin, tuple) and len(bin) == 2 for bin in binned_data)

    # test with custom bin edges
    other_bins = [0, 2, 4, 6, 8]
    binned_data = w._bin_other_data(other_bins=other_bins)
    expected_bins = [
        (1, 2),
        (1, 2),
        (2, 4),
        (2, 4),
        (4, 6),
        (4, 6),
        (6, 8),
        (6, 8),
    ]
    assert binned_data == expected_bins

    # test with invalid bin edges (max edge < max data)
    invalid_bins = [0, 2, 4]
    # match the error message, with a variable ending
    with pytest.raises(
        ValueError,
    ):
        w._bin_other_data(other_data=ws, other_bins=invalid_bins)

    # test with invalid bin edges (min edge > min data)
    invalid_bins = [3.1, 5, 7]
    with pytest.raises(
        ValueError,
    ):
        w._bin_other_data(other_data=ws, other_bins=invalid_bins)

    # test with invalid other_data
    with pytest.raises(
        ValueError, match="other_data must be the same length as the wind data."
    ):
        w._bin_other_data(other_data=other_data[1:2])

    # test with integer number of bins
    binned_data = w._bin_other_data(other_data=ws, other_bins=4)
    assert len(binned_data) == len(ws)
    assert all(isinstance(bin, tuple) and len(bin) == 2 for bin in binned_data)


def test_bin_data():
    # create data
    datetimes = [datetime(2021, 1, 1, 0, 0) + timedelta(hours=i) for i in range(8)]
    ws = [1, 2, 3, 4, 5, 6, 7, 8]
    wd = [0, 45, 90, 135, 180, 225, 270, 315]

    # create wind object
    w = Wind(
        wind_speeds=ws,
        wind_directions=wd,
        datetimes=datetimes,
        height_above_ground=10,
    )

    # test with custom number of directions
    binned_data = w.bin_data(
        directions=4,
    )
    assert isinstance(binned_data, pd.DataFrame)
    assert len(binned_data) == len(ws)
    assert all(
        isinstance(bin, tuple) and len(bin) == 2 for bin in binned_data["direction"]
    )

    # test with custom other_data
    other_data = [10, 20, 30, 40, 50, 60, 70, 80]
    binned_data = w.bin_data(other_data=other_data)
    assert isinstance(binned_data, pd.DataFrame)
    assert len(binned_data) == len(ws)
    assert all(isinstance(bin, tuple) and len(bin) == 2 for bin in binned_data["other"])

    # test with custom other_bins
    custom_bins = [0, 20, 40, 60, 80]
    binned_data = w.bin_data(other_data=other_data, other_bins=custom_bins)
    assert isinstance(binned_data, pd.DataFrame)
    assert len(binned_data) == len(ws)
    assert all(isinstance(bin, tuple) and len(bin) == 2 for bin in binned_data["other"])

    # test with integer number of bins
    binned_data = w.bin_data(other_data=other_data, other_bins=4)
    assert isinstance(binned_data, pd.DataFrame)
    assert len(binned_data) == len(ws)
    assert all(isinstance(bin, tuple) and len(bin) == 2 for bin in binned_data["other"])
