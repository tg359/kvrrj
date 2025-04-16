from honeybee.boundarycondition import boundary_conditions
from honeybee.facetype import face_types
from honeybee.model import Aperture, Door, Face, Shade, ShadeMesh
from ladybug.color import Color
from ladybug_geometry.geometry3d import Face3D

from kvrrj.color.to_ladybug_color import to_ladybug_color

from . import (
    HEX_BLACK,
    HEX_BLACK_TRANSPARENT,
    HEX_RED,
    HEX_RED_TRANSPARENT,
    HEX_WHITE,
    HEX_WHITE_TRANSPARENT,
    LB_BLACK,
    LB_BLACK_TRANSPARENT,
    LB_RED,
    LB_RED_TRANSPARENT,
    LB_WHITE,
    LB_WHITE_TRANSPARENT,
    PLOTLY_RGB_BLACK,
    PLOTLY_RGB_RED,
    PLOTLY_RGB_WHITE,
    PLOTLY_RGBA_BLACK,
    PLOTLY_RGBA_BLACK_TRANSPARENT,
    PLOTLY_RGBA_RED,
    PLOTLY_RGBA_RED_TRANSPARENT,
    PLOTLY_RGBA_WHITE,
    PLOTLY_RGBA_WHITE_TRANSPARENT,
    RGB_1_BLACK,
    RGB_1_RED,
    RGB_1_WHITE,
    RGB_255_BLACK,
    RGB_255_RED,
    RGB_255_WHITE,
    RGBA_1_BLACK,
    RGBA_1_BLACK_TRANSPARENT,
    RGBA_1_RED,
    RGBA_1_RED_TRANSPARENT,
    RGBA_1_WHITE,
    RGBA_1_WHITE_TRANSPARENT,
    RGBA_255_BLACK,
    RGBA_255_BLACK_TRANSPARENT,
    RGBA_255_RED,
    RGBA_255_RED_TRANSPARENT,
    RGBA_255_WHITE,
    RGBA_255_WHITE_TRANSPARENT,
)

WALL_EXT = Face(
    identifier="wall_ext",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("Wall"),
    boundary_condition=boundary_conditions.by_name("Outdoors"),
)
WALL_GROUND = Face(
    identifier="wall_ground",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("Wall"),
    boundary_condition=boundary_conditions.by_name("Ground"),
)
WALL_INT = Face(
    identifier="wall_int",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("Wall"),
    boundary_condition=boundary_conditions.by_name("Adiabatic"),
)
WALL_AIR = Face(
    identifier="wall_air",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("AirBoundary"),
    boundary_condition=boundary_conditions.by_name("Adiabatic"),
)
CEILING_EXT = Face(
    identifier="roof_ceiling_ext",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("RoofCeiling"),
    boundary_condition=boundary_conditions.by_name("Outdoors"),
)
CEILING_INT = Face(
    identifier="ceiling_int",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("RoofCeiling"),
    boundary_condition=boundary_conditions.by_name("Adiabatic"),
)
CEILING_GROUND = Face(
    identifier="ceiling_ground",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("RoofCeiling"),
    boundary_condition=boundary_conditions.by_name("Ground"),
)
CEILING_AIR = Face(
    identifier="ceiling_air",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("AirBoundary"),
    boundary_condition=boundary_conditions.by_name("Adiabatic"),
)
FLOOR_EXT = Face(
    identifier="floor_ext",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("Floor"),
    boundary_condition=boundary_conditions.by_name("Outdoors"),
)
FLOOR_GROUND = Face(
    identifier="floor_ground",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("Floor"),
    boundary_condition=boundary_conditions.by_name("Ground"),
)
FLOOR_INT = Face(
    identifier="floor_int",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("Floor"),
    boundary_condition=boundary_conditions.by_name("Adiabatic"),
)
FLOOR_AIR = Face(
    identifier="floor_air",
    geometry=Face3D.from_regular_polygon(side_count=3),
    type=face_types.by_name("AirBoundary"),
    boundary_condition=boundary_conditions.by_name("Adiabatic"),
)
DOOR = Door(
    identifier="door_ext",
    geometry=Face3D.from_regular_polygon(side_count=3),
)
APERTURE = Aperture(
    identifier="aperture_ext",
    geometry=Face3D.from_regular_polygon(side_count=3),
)
SHADE = Shade(identifier="shade", geometry=Face3D.from_regular_polygon(side_count=3))
SHADEMESH = ShadeMesh(
    identifier="shademesh",
    geometry=Face3D.from_regular_polygon(side_count=3).triangulated_mesh3d,
)


def test_wall_to_ladybug_color():
    assert isinstance(to_ladybug_color(WALL_EXT), Color)
    assert isinstance(to_ladybug_color(WALL_GROUND), Color)
    assert isinstance(to_ladybug_color(WALL_INT), Color)
    assert isinstance(to_ladybug_color(WALL_AIR), Color)


def test_ceiling_to_ladybug_color():
    assert isinstance(to_ladybug_color(CEILING_EXT), Color)
    assert isinstance(to_ladybug_color(CEILING_INT), Color)
    assert isinstance(to_ladybug_color(CEILING_GROUND), Color)
    assert isinstance(to_ladybug_color(CEILING_AIR), Color)


def test_floor_to_ladybug_color():
    assert isinstance(to_ladybug_color(FLOOR_EXT), Color)
    assert isinstance(to_ladybug_color(FLOOR_GROUND), Color)
    assert isinstance(to_ladybug_color(FLOOR_INT), Color)
    assert isinstance(to_ladybug_color(FLOOR_AIR), Color)


def test_door_to_ladybug_color():
    assert isinstance(to_ladybug_color(DOOR), Color)


def test_aperture_to_ladybug_color():
    assert isinstance(to_ladybug_color(APERTURE), Color)


def test_shade_to_ladybug_color():
    assert isinstance(to_ladybug_color(SHADE), Color)
    assert isinstance(to_ladybug_color(SHADEMESH), Color)


def test_rgba_1_to_ladybug_color():
    """Test that the RGBA 0-1 colors are converted to RGBA 0-255."""
    assert to_ladybug_color(RGBA_1_BLACK) == LB_BLACK
    assert to_ladybug_color(RGBA_1_BLACK_TRANSPARENT) == LB_BLACK_TRANSPARENT
    assert to_ladybug_color(RGBA_1_WHITE) == LB_WHITE
    assert to_ladybug_color(RGBA_1_WHITE_TRANSPARENT) == LB_WHITE_TRANSPARENT
    assert to_ladybug_color(RGBA_1_RED) == LB_RED
    assert to_ladybug_color(RGBA_1_RED_TRANSPARENT) == LB_RED_TRANSPARENT


def test_rgba_255_to_ladybug_color():
    """Test that the RGBA 0-255 colors are converted to RGBA 0-1."""
    assert to_ladybug_color(RGBA_255_BLACK) == LB_BLACK
    assert to_ladybug_color(RGBA_255_BLACK_TRANSPARENT) == LB_BLACK_TRANSPARENT
    assert to_ladybug_color(RGBA_255_WHITE) == LB_WHITE
    assert to_ladybug_color(RGBA_255_WHITE_TRANSPARENT) == LB_WHITE_TRANSPARENT
    assert to_ladybug_color(RGBA_255_RED) == LB_RED
    assert to_ladybug_color(RGBA_255_RED_TRANSPARENT) == LB_RED_TRANSPARENT


def test_rgb_1_to_ladybug_color():
    """Test that the RGB 0-1 colors are converted to RGBA 0-1."""
    assert to_ladybug_color(RGB_1_BLACK) == LB_BLACK
    assert to_ladybug_color(RGB_1_WHITE) == LB_WHITE
    assert to_ladybug_color(RGB_1_RED) == LB_RED


def test_rgb_255_to_ladybug_color():
    """Test that the RGB 0-255 colors are converted to RGBA 0-1."""
    assert to_ladybug_color(RGB_255_BLACK) == LB_BLACK
    assert to_ladybug_color(RGB_255_WHITE) == LB_WHITE
    assert to_ladybug_color(RGB_255_RED) == LB_RED


def test_hex_to_ladybug_color():
    """Test that the hex colors are converted to RGBA 0-1."""
    assert to_ladybug_color(HEX_BLACK) == LB_BLACK
    assert to_ladybug_color(HEX_BLACK_TRANSPARENT) == LB_BLACK_TRANSPARENT
    assert to_ladybug_color(HEX_WHITE) == LB_WHITE
    assert to_ladybug_color(HEX_WHITE_TRANSPARENT) == LB_WHITE_TRANSPARENT
    assert to_ladybug_color(HEX_RED) == LB_RED
    assert to_ladybug_color(HEX_RED_TRANSPARENT) == LB_RED_TRANSPARENT


def test_ladybug_to_ladybug_color():
    """Test that the ladybug colors are converted to RGBA 0-1."""
    assert to_ladybug_color(LB_BLACK) == LB_BLACK
    assert to_ladybug_color(LB_BLACK_TRANSPARENT) == LB_BLACK_TRANSPARENT
    assert to_ladybug_color(LB_WHITE) == LB_WHITE
    assert to_ladybug_color(LB_WHITE_TRANSPARENT) == LB_WHITE_TRANSPARENT
    assert to_ladybug_color(LB_RED) == LB_RED
    assert to_ladybug_color(LB_RED_TRANSPARENT) == LB_RED_TRANSPARENT


def test_plotly_to_ladybug_color():
    """Test that the plotly colors are converted to RGBA 0-1."""
    assert to_ladybug_color(PLOTLY_RGB_BLACK) == LB_BLACK
    assert to_ladybug_color(PLOTLY_RGB_WHITE) == LB_WHITE
    assert to_ladybug_color(PLOTLY_RGB_RED) == LB_RED
    assert to_ladybug_color(PLOTLY_RGBA_BLACK) == LB_BLACK
    assert to_ladybug_color(PLOTLY_RGBA_BLACK_TRANSPARENT) == LB_BLACK_TRANSPARENT
    assert to_ladybug_color(PLOTLY_RGBA_WHITE) == LB_WHITE
    assert to_ladybug_color(PLOTLY_RGBA_WHITE_TRANSPARENT) == LB_WHITE_TRANSPARENT
    assert to_ladybug_color(PLOTLY_RGBA_RED) == LB_RED
    assert to_ladybug_color(PLOTLY_RGBA_RED_TRANSPARENT) == LB_RED_TRANSPARENT
