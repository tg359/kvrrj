from ladybug.datatype import TYPESDICT
from ladybug.datatype.temperature import DryBulbTemperature

from kvrrj.ladybug.datatype import (
    Colormap,
    datatype_to_color,
    datatype_to_colormap,
    datatype_to_string,
)


def test_datatype_to_string_valid_input():
    dt = DryBulbTemperature()
    unit = "C"
    assert datatype_to_string(datatype=dt, unit=unit) == "Dry Bulb Temperature (C)"


def test_datatype_to_color():
    for dt in TYPESDICT.values():
        assert datatype_to_color(dt).startswith("#")


def test_datatype_to_colormap():
    for dt in TYPESDICT.values():
        assert isinstance(datatype_to_colormap(dt), Colormap)
