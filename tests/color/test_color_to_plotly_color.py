from kvrrj.color.to_plotly_color import to_plotly_color

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


def test_rgba_1_to_plotly_color():
    """Test that the RGBA 0-1 colors are converted to RGBA 0-255."""
    assert to_plotly_color(RGBA_1_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(RGBA_1_BLACK_TRANSPARENT) == PLOTLY_RGBA_BLACK_TRANSPARENT
    assert to_plotly_color(RGBA_1_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(RGBA_1_WHITE_TRANSPARENT) == PLOTLY_RGBA_WHITE_TRANSPARENT
    assert to_plotly_color(RGBA_1_RED) == PLOTLY_RGBA_RED
    assert to_plotly_color(RGBA_1_RED_TRANSPARENT) == PLOTLY_RGBA_RED_TRANSPARENT


def test_rgba_255_to_plotly_color():
    """Test that the RGBA 0-255 colors are converted to RGBA 0-1."""
    assert to_plotly_color(RGBA_255_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(RGBA_255_BLACK_TRANSPARENT) == PLOTLY_RGBA_BLACK_TRANSPARENT
    assert to_plotly_color(RGBA_255_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(RGBA_255_WHITE_TRANSPARENT) == PLOTLY_RGBA_WHITE_TRANSPARENT
    assert to_plotly_color(RGBA_255_RED) == PLOTLY_RGBA_RED
    assert to_plotly_color(RGBA_255_RED_TRANSPARENT) == PLOTLY_RGBA_RED_TRANSPARENT


def test_rgb_1_to_plotly_color():
    """Test that the RGB 0-1 colors are converted to RGBA 0-1."""
    assert to_plotly_color(RGB_1_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(RGB_1_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(RGB_1_RED) == PLOTLY_RGBA_RED


def test_rgb_255_to_plotly_color():
    """Test that the RGB 0-255 colors are converted to RGBA 0-1."""
    assert to_plotly_color(RGB_255_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(RGB_255_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(RGB_255_RED) == PLOTLY_RGBA_RED


def test_hex_to_plotly_color():
    """Test that the hex colors are converted to RGBA 0-1."""
    assert to_plotly_color(HEX_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(HEX_BLACK_TRANSPARENT) == PLOTLY_RGBA_BLACK_TRANSPARENT
    assert to_plotly_color(HEX_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(HEX_WHITE_TRANSPARENT) == PLOTLY_RGBA_WHITE_TRANSPARENT
    assert to_plotly_color(HEX_RED) == PLOTLY_RGBA_RED
    assert to_plotly_color(HEX_RED_TRANSPARENT) == PLOTLY_RGBA_RED_TRANSPARENT


def test_ladybug_to_plotly_color():
    """Test that the ladybug colors are converted to RGBA 0-1."""
    assert to_plotly_color(LB_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(LB_BLACK_TRANSPARENT) == PLOTLY_RGBA_BLACK_TRANSPARENT
    assert to_plotly_color(LB_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(LB_WHITE_TRANSPARENT) == PLOTLY_RGBA_WHITE_TRANSPARENT
    assert to_plotly_color(LB_RED) == PLOTLY_RGBA_RED
    assert to_plotly_color(LB_RED_TRANSPARENT) == PLOTLY_RGBA_RED_TRANSPARENT


def test_plotly_to_plotly_color():
    """Test that the plotly colors are converted to RGBA 0-1."""
    assert to_plotly_color(PLOTLY_RGB_BLACK) == PLOTLY_RGBA_BLACK
    assert to_plotly_color(PLOTLY_RGB_WHITE) == PLOTLY_RGBA_WHITE
    assert to_plotly_color(PLOTLY_RGB_RED) == PLOTLY_RGBA_RED
    assert to_plotly_color(PLOTLY_RGBA_BLACK) == PLOTLY_RGBA_BLACK
    assert (
        to_plotly_color(PLOTLY_RGBA_BLACK_TRANSPARENT) == PLOTLY_RGBA_BLACK_TRANSPARENT
    )
    assert to_plotly_color(PLOTLY_RGBA_WHITE) == PLOTLY_RGBA_WHITE
    assert (
        to_plotly_color(PLOTLY_RGBA_WHITE_TRANSPARENT) == PLOTLY_RGBA_WHITE_TRANSPARENT
    )
    assert to_plotly_color(PLOTLY_RGBA_RED) == PLOTLY_RGBA_RED
    assert to_plotly_color(PLOTLY_RGBA_RED_TRANSPARENT) == PLOTLY_RGBA_RED_TRANSPARENT
