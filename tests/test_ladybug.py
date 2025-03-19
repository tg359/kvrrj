from datetime import datetime, timedelta

from kvrrj.ladybug.analysis_period import (
    _TIMESTEP,
    AnalysisPeriod,
    analysis_period_from_datetimes,
    analysis_period_from_string,
    analysis_period_to_datetimes,
    analysis_period_to_string,
    lbdatetime,
    lbdatetime_from_datetime,
    lbdatetime_to_datetime,
)
from kvrrj.ladybug.location import Location, average_location, location_to_string


def test_location_to_string():
    loc = Location(
        city="Portobello",
        country="Naboombu",
        state="Liquid",
        latitude=51.5074,  # type: ignore
        longitude=0.1278,  # type: ignore
        elevation=-50,
        station_id="123",
        source="metoffice",
        time_zone=1,
    )
    assert location_to_string(loc) == f"{loc.country} - {loc.city}"


def test_average_location():
    vals = [0, 25, 50, 75]
    locs = [
        Location(latitude=i, longitude=i, elevation=i, city="A", country="B")
        for i in vals
    ]

    avg_loc = average_location(locs)

    assert avg_loc.latitude == sum(vals) / len(vals)
    assert avg_loc.longitude == sum(vals) / len(vals)
    assert avg_loc.elevation == sum(vals) / len(vals)
    assert avg_loc.city == "Synthetic (A|A|A|A)"
    assert avg_loc.country == "Synthetic (B|B|B|B)"


def test_lbdatetime_from_datetime():
    # Test valid conversion
    dt = datetime(2017, 10, 5, 14, 30, 0)
    lb_dt = lbdatetime_from_datetime(dt)
    assert isinstance(lb_dt, lbdatetime)
    assert lb_dt.year == dt.year
    assert lb_dt.month == dt.month
    assert lb_dt.day == dt.day
    assert lb_dt.hour == dt.hour
    assert lb_dt.minute == dt.minute
    assert lb_dt.second == dt.second
    assert lb_dt.leap_year == (
        (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0)
    )

    dt = datetime(2024, 2, 29, 14, 30, 0)
    lb_dt = lbdatetime_from_datetime(dt)
    assert lb_dt.leap_year


def test_lbdatetime_to_datetime():
    lb_dt = lbdatetime(month=10, day=5, hour=14, minute=30, leap_year=False)
    dt = lbdatetime_to_datetime(lb_dt)
    assert isinstance(dt, datetime)
    assert dt.year == 2017
    assert dt.month == lb_dt.month
    assert dt.day == lb_dt.day
    assert dt.hour == lb_dt.hour
    assert dt.minute == lb_dt.minute
    assert dt.second == lb_dt.second
    assert not lb_dt.leap_year

    lb_dt = lbdatetime(month=10, day=5, hour=14, minute=30, leap_year=True)
    dt = lbdatetime_to_datetime(lb_dt)
    assert lb_dt.leap_year


def test_analysis_period_to_datetimes():
    ap = AnalysisPeriod()
    dts = analysis_period_to_datetimes(ap)
    assert len(dts) == len(ap)
    assert all(isinstance(dt, datetime) for dt in dts)
    assert dts[0].year == 2017

    ap = AnalysisPeriod(timestep=4, is_leap_year=True)
    dts = analysis_period_to_datetimes(ap)
    assert len(dts) == len(ap)
    assert all(isinstance(dt, datetime) for dt in dts)
    assert dts[0].year == 2016


def test_analysis_period_from_datetimes():
    dts = [datetime(2017, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(8760)]
    ap = analysis_period_from_datetimes(dts)
    assert len(ap) == len(dts)
    assert ap.timestep == 1

    dts = [datetime(2016, 1, 1, 0, 0, 0) + timedelta(hours=i) for i in range(8784)]
    ap = analysis_period_from_datetimes(dts)
    assert len(ap) == len(dts)
    assert ap.timestep == 1
    assert ap.is_leap_year

    dts = [
        datetime(2017, 1, 1, 0, 0, 0) + timedelta(minutes=30 * i)
        for i in range(8760 * 2)
    ]
    ap = analysis_period_from_datetimes(dts)
    assert len(ap) == len(dts)
    assert ap.timestep == 2


def test_analysis_period_to_string():
    for timestep in [1, 4, 30]:
        ap = AnalysisPeriod(timestep=timestep)
        ap_str = analysis_period_to_string(ap, save_path=False)
        assert (
            ap_str
            == f"Jan 01 to Dec 31 between 00:00 and 23:59 every {_TIMESTEP[ap.timestep]}"
        )
        ap_str = analysis_period_to_string(ap, save_path=True)
        assert ap_str == f"0101_1231_00_23_{ap.timestep:02d}"
    ap = AnalysisPeriod(
        st_month=6, end_month=8, st_day=15, end_day=18, st_hour=6, end_hour=18
    )
    ap_str = analysis_period_to_string(ap, save_path=False)
    assert (
        ap_str
        == f"Jun 15 to Aug 18 between 06:00 and 18:59 every {_TIMESTEP[ap.timestep]}"
    )
    ap_str = analysis_period_to_string(ap, save_path=True)
    assert ap_str == f"0615_0818_06_18_{ap.timestep:02d}"


def test_analysis_period_from_string():
    ap_str = "0615_0818_06_18_30"
    ap = analysis_period_from_string(ap_str)
    assert ap == AnalysisPeriod(
        st_month=6,
        end_month=8,
        st_day=15,
        end_day=18,
        st_hour=6,
        end_hour=18,
        timestep=30,
    )

    ap_str = "Jun 15 to Aug 18 between 06:00 and 18:59 every 2-minutes"
    ap = analysis_period_from_string(ap_str)
    assert ap == AnalysisPeriod(
        st_month=6,
        end_month=8,
        st_day=15,
        end_day=18,
        st_hour=6,
        end_hour=18,
        timestep=30,
    )
