import pytest

from kvrrj.util import (
    air_pressure_at_height,
    temperature_at_height,
    wind_speed_at_height,
)


def test_wind_speed_at_height():
    assert wind_speed_at_height(
        reference_value=2,
        reference_height=10,
        target_height=2,
        terrain_roughness_length=0.5,
        log_function=False,
    ) == pytest.approx(1.5891948094037045, rel=0.0001)


def test_air_pressure_at_height():
    assert air_pressure_at_height(
        reference_value=2,
        reference_height=10,
        target_height=2,
    ) == pytest.approx(2.001897379377387, rel=0.0001)


def test_temperature_at_height():
    assert temperature_at_height(
        reference_value=2,
        reference_height=10,
        target_height=500,
    ) == pytest.approx(-1.185, rel=0.0001)
