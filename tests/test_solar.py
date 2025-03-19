from datetime import date, timedelta

from kvrrj.solar import (
    Location,
    azimuthal_radiation,
    sunrise_sunset,
    sunrise_sunset_from_epw,
)

from . import EPW_OBJ


def test_sunrise_sunset():
    dts = [date(2017, 1, 1) + timedelta(days=i) for i in range(100, 111, 1)]
    sr_df = sunrise_sunset(dates=dts, location=Location())
    assert len(sr_df) == len(dts)
    assert "civil sunrise" in sr_df.columns
    assert "nautical sunrise" in sr_df.columns
    assert "astronomical sunrise" in sr_df.columns
    assert "noon" in sr_df.columns
    assert "apparent sunset" in sr_df.columns


def test_sunrise_sunset_from_epw():
    sr_df = sunrise_sunset_from_epw(EPW_OBJ)
    assert len(sr_df) == 365


def test_azimuthal_radiation():
    s = azimuthal_radiation(epw=EPW_OBJ)
