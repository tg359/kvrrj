from kvrrj.color.to_hex import to_hex

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


def test_rgba_1_to_hex():
    """Test that the RGBA 0-1 colors are converted to RGBA 0-255."""
    assert to_hex(RGBA_1_BLACK) == HEX_BLACK
    assert to_hex(RGBA_1_BLACK_TRANSPARENT) == HEX_BLACK_TRANSPARENT
    assert to_hex(RGBA_1_WHITE) == HEX_WHITE
    assert to_hex(RGBA_1_WHITE_TRANSPARENT) == HEX_WHITE_TRANSPARENT
    assert to_hex(RGBA_1_RED) == HEX_RED
    assert to_hex(RGBA_1_RED_TRANSPARENT) == HEX_RED_TRANSPARENT


def test_rgba_255_to_hex():
    """Test that the RGBA 0-255 colors are converted to RGBA 0-1."""
    assert to_hex(RGBA_255_BLACK) == HEX_BLACK
    assert to_hex(RGBA_255_BLACK_TRANSPARENT) == HEX_BLACK_TRANSPARENT
    assert to_hex(RGBA_255_WHITE) == HEX_WHITE
    assert to_hex(RGBA_255_WHITE_TRANSPARENT) == HEX_WHITE_TRANSPARENT
    assert to_hex(RGBA_255_RED) == HEX_RED
    assert to_hex(RGBA_255_RED_TRANSPARENT) == HEX_RED_TRANSPARENT


def test_rgb_1_to_hex():
    """Test that the RGB 0-1 colors are converted to RGBA 0-1."""
    assert to_hex(RGB_1_BLACK) == HEX_BLACK
    assert to_hex(RGB_1_WHITE) == HEX_WHITE
    assert to_hex(RGB_1_RED) == HEX_RED


def test_rgb_255_to_hex():
    """Test that the RGB 0-255 colors are converted to RGBA 0-1."""
    assert to_hex(RGB_255_BLACK) == HEX_BLACK
    assert to_hex(RGB_255_WHITE) == HEX_WHITE
    assert to_hex(RGB_255_RED) == HEX_RED


def test_hex_to_hex():
    """Test that the hex colors are converted to RGBA 0-1."""
    assert to_hex(HEX_BLACK) == HEX_BLACK
    assert to_hex(HEX_BLACK_TRANSPARENT) == HEX_BLACK_TRANSPARENT
    assert to_hex(HEX_WHITE) == HEX_WHITE
    assert to_hex(HEX_WHITE_TRANSPARENT) == HEX_WHITE_TRANSPARENT
    assert to_hex(HEX_RED) == HEX_RED
    assert to_hex(HEX_RED_TRANSPARENT) == HEX_RED_TRANSPARENT


def test_ladybug_to_hex():
    """Test that the ladybug colors are converted to RGBA 0-1."""
    assert to_hex(LB_BLACK) == HEX_BLACK
    assert to_hex(LB_BLACK_TRANSPARENT) == HEX_BLACK_TRANSPARENT
    assert to_hex(LB_WHITE) == HEX_WHITE
    assert to_hex(LB_WHITE_TRANSPARENT) == HEX_WHITE_TRANSPARENT
    assert to_hex(LB_RED) == HEX_RED
    assert to_hex(LB_RED_TRANSPARENT) == HEX_RED_TRANSPARENT


def test_plotly_to_hex():
    """Test that the plotly colors are converted to RGBA 0-1."""
    assert to_hex(PLOTLY_RGB_BLACK) == HEX_BLACK
    assert to_hex(PLOTLY_RGB_WHITE) == HEX_WHITE
    assert to_hex(PLOTLY_RGB_RED) == HEX_RED
    assert to_hex(PLOTLY_RGBA_BLACK) == HEX_BLACK
    assert to_hex(PLOTLY_RGBA_BLACK_TRANSPARENT) == HEX_BLACK_TRANSPARENT
    assert to_hex(PLOTLY_RGBA_WHITE) == HEX_WHITE
    assert to_hex(PLOTLY_RGBA_WHITE_TRANSPARENT) == HEX_WHITE_TRANSPARENT
    assert to_hex(PLOTLY_RGBA_RED) == HEX_RED
    assert to_hex(PLOTLY_RGBA_RED_TRANSPARENT) == HEX_RED_TRANSPARENT
