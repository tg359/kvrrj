from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.location import Location

from kvrrj.solar import RadiationRose, SkyMatrix, Solar

from . import EPW_OBJ

# region: INSTANTIATION


def test_solar_initialization():
    location = Location()
    analysis_period = AnalysisPeriod()
    dni = [100.0] * 8760
    dhi = [50.0] * 8760
    ghi = [150.0] * 8760

    solar = Solar(
        location=location,
        analysis_period=analysis_period,
        direct_normal_irradiance=dni,
        diffuse_horizontal_irradiance=dhi,
        global_horizontal_irradiance=ghi,
    )

    assert solar.location == location
    assert solar.analysis_period == analysis_period
    assert np.array_equal(solar.direct_normal_irradiance, np.array(dni))
    assert np.array_equal(solar.diffuse_horizontal_irradiance, np.array(dhi))
    assert np.array_equal(solar.global_horizontal_irradiance, np.array(ghi))

    # different analysis period timestep
    analysis_period = AnalysisPeriod(timestep=3)
    dni = [100.0] * 8760 * 3
    dhi = [50.0] * 8760 * 3
    ghi = [150.0] * 8760 * 3

    solar = Solar(
        location=location,
        analysis_period=analysis_period,
        direct_normal_irradiance=dni,
        diffuse_horizontal_irradiance=dhi,
        global_horizontal_irradiance=ghi,
    )

    assert solar.location == location
    assert solar.analysis_period == analysis_period
    assert np.array_equal(solar.direct_normal_irradiance, np.array(dni))
    assert np.array_equal(solar.diffuse_horizontal_irradiance, np.array(dhi))
    assert np.array_equal(solar.global_horizontal_irradiance, np.array(ghi))

    # incorrect length of data
    dni = [100.0] * 8760
    with pytest.raises(ValueError):
        Solar(
            location=location,
            analysis_period=analysis_period,
            direct_normal_irradiance=dni,
            diffuse_horizontal_irradiance=dhi,
            global_horizontal_irradiance=ghi,
        )


def test_solar_from_epw():
    solar = Solar.from_epw(EPW_OBJ)
    assert isinstance(solar, Solar)
    assert solar.location.source == Path(EPW_OBJ.file_path).name


def test_to_from_dict():
    solar = Solar.from_epw(EPW_OBJ)
    solar_dict = solar.to_dict()
    assert isinstance(solar_dict, dict)
    new_solar = Solar.from_dict(solar_dict)
    assert isinstance(new_solar, Solar)


def test_to_from_json():
    solar = Solar.from_epw(EPW_OBJ)
    solar_json = solar.to_json()
    assert isinstance(solar_json, str)
    new_solar = Solar.from_json(solar_json)
    assert isinstance(new_solar, Solar)


def test_solar_to_from_dataframe():
    solar = Solar.from_epw(EPW_OBJ)
    solar_df = solar.to_dataframe()
    assert isinstance(solar_df, pd.DataFrame)
    new_solar = Solar.from_dataframe(solar_df)
    assert isinstance(new_solar, Solar)
    assert solar == new_solar


def test_equality():
    solar1 = Solar.from_epw(EPW_OBJ)
    solar2 = Solar.from_epw(EPW_OBJ)
    assert solar1 == solar2


def test_length():
    solar = Solar.from_epw(EPW_OBJ)
    assert len(solar) == 8760


def test_sunrise_sunset():
    dts = [date(2017, 1, 1) + timedelta(days=i) for i in range(100, 111, 1)]
    df = Solar._sunrise_sunset(dates=dts, location=Location())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(dts)
    assert "civil sunrise" in df.columns
    assert "nautical sunrise" in df.columns
    assert "astronomical sunrise" in df.columns
    assert "noon" in df.columns
    assert "apparent sunset" in df.columns


def test_lb_sky_matrix():
    solar = Solar.from_epw(EPW_OBJ)
    assert isinstance(solar._sky_matrix(), SkyMatrix)


def test_lb_radiation_rose():
    solar = Solar.from_epw(EPW_OBJ)
    assert isinstance(
        solar.lb_radiation_rose(sky_matrix=solar._sky_matrix()), RadiationRose
    )


def test_radiation_benefit_data():
    solar = Solar.from_epw(EPW_OBJ)
    assert isinstance(
        solar._radiation_benefit_data(temperature=EPW_OBJ.dry_bulb_temperature),
        pd.Series,
    )


def test_radiation_rose_data():
    solar = Solar.from_epw(EPW_OBJ)
    assert isinstance(solar._radiation_rose_data(), pd.DataFrame)


def test_tilt_orientation_factor_data():
    solar = Solar.from_epw(EPW_OBJ)
    assert isinstance(solar._tilt_orientation_factor_data(), pd.DataFrame)
