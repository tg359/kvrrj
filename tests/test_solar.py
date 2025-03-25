from datetime import date, timedelta

from kvrrj.solar import (
    IrradianceType,
    Location,
    azimuthal_radiation,
    sunrise_sunset,
    sunrise_sunset_from_epw,
    tilt_orientation_factor,
)

from . import EPW_OBJ


def test_sunrise_sunset():
    dts = [date(2017, 1, 1) + timedelta(days=i) for i in range(100, 111, 1)]
    df = sunrise_sunset(dates=dts, location=Location())
    assert len(df) == len(dts)
    assert "civil sunrise" in df.columns
    assert "nautical sunrise" in df.columns
    assert "astronomical sunrise" in df.columns
    assert "noon" in df.columns
    assert "apparent sunset" in df.columns


def test_sunrise_sunset_from_epw():
    df = sunrise_sunset_from_epw(EPW_OBJ)
    assert len(df) == 365


def test_azimuthal_radiation():
    df = azimuthal_radiation(epw=EPW_OBJ)
    assert IrradianceType.TOTAL.to_string() in df.columns
    assert IrradianceType.DIRECT.to_string() in df.columns
    assert IrradianceType.DIFFUSE.to_string() in df.columns


def test_tilt_orientation_factor():
    df = tilt_orientation_factor(epw=EPW_OBJ)
    assert IrradianceType.TOTAL.to_string() in df.columns
    assert IrradianceType.DIRECT.to_string() in df.columns
    assert IrradianceType.DIFFUSE.to_string() in df.columns
    assert df.shape == (37, 27)
