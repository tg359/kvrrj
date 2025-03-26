from kvrrj.solar2 import Solar

from . import EPW_OBJ


def test_solar_from_epw():
    assert isinstance(Solar.from_epw(EPW_OBJ), Solar)
    assert isinstance(Solar.from_epw(EPW_OBJ.file_path), Solar)
