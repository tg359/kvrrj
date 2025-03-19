import pytest
from ladybug.color import Color

from kvrrj.viz.color import (
    ColorTypeError,
    _ladybug_to_rgba,
    _plotly_to_rgba,
    average_color,
    contrasting_color,
    lighten_color,
    relative_luminance,
    to_hex,
    to_ladybug,
    to_plotly,
    to_rgba,
)
from kvrrj.viz.colormap import colormap_sequential


def test_ladybug_to_rgba():
    # Test with ladybug Color
    lb_color = Color(0, 0, 255, 255)
    assert _ladybug_to_rgba(lb_color) == (0.0, 0.0, 1.0, 1.0)

    # Test invalid input
    with pytest.raises(ColorTypeError):
        _ladybug_to_rgba(123)


def test_plotly_to_rgba():
    # Test with plotly color string
    assert _plotly_to_rgba("rgb(0,0,255)") == (0.0, 0.0, 1.0, 1.0)
    assert _plotly_to_rgba("rgba(0,0,255,255)") == (0.0, 0.0, 1.0, 1.0)

    # Test invalid input
    with pytest.raises(ColorTypeError):
        _plotly_to_rgba("123")


def test_to_rgba():
    colors = [
        "blue",
        "rgb(0,0,255)",
        "rgba(0,0,255,255)",
        Color(0, 0, 255, 255),
        (0, 0, 1, 1),
        "#0000ffff",
        "#0000ff",
    ]
    for c in colors:
        assert to_rgba(c) == (0.0, 0.0, 1.0, 1.0)

    # Test invalid input
    with pytest.raises(ColorTypeError):
        to_rgba(123)


def test_to_hex():
    colors = [
        "blue",
        "rgb(0,0,255)",
        "rgba(0,0,255,255)",
        Color(0, 0, 255, 255),
        (0, 0, 1, 1),
        "#0000ffff",
        "#0000ff",
    ]
    for c in colors:
        assert to_hex(c) == "#0000ffff"

    # Test invalid input
    with pytest.raises(ColorTypeError):
        to_hex(123)


def test_to_ladybug():
    colors = [
        "blue",
        "rgb(0,0,255)",
        "rgba(0,0,255,255)",
        Color(0, 0, 255, 255),
        (0, 0, 1, 1),
        "#0000ffff",
        "#0000ff",
    ]
    for c in colors:
        assert to_ladybug(c) == Color(0, 0, 255, 255)

    # Test invalid input
    with pytest.raises(ColorTypeError):
        to_ladybug(123)


def test_to_plotly():
    colors = [
        "blue",
        "rgb(0,0,255)",
        "rgba(0,0,255,255)",
        Color(0, 0, 255, 255),
        (0, 0, 1, 1),
        "#0000ffff",
        "#0000ff",
    ]
    for c in colors:
        assert to_plotly(c) == "rgba(0,0,255,255)"

    # Test invalid input
    with pytest.raises(ColorTypeError):
        to_plotly(123)


def test_relative_luminance():
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0, rel=1e-7)
    assert relative_luminance("#000000") == pytest.approx(0.0, rel=1e-7)
    assert relative_luminance("#808080") == pytest.approx(0.215860500965604, rel=1e-7)


def test_contrasting_color():
    assert contrasting_color("#FFFFFF") == ".15"
    assert contrasting_color("#000000") == "w"
    assert contrasting_color("#808080") == "w"


def test_lighten_color():
    # Test lightening a named color
    assert lighten_color("g", 0.3) == (
        0.5500000000000002,
        0.9999999999999999,
        0.5500000000000002,
    )

    # Test lightening a hex color
    assert lighten_color("#F034A3", 0.6) == (
        0.9647058823529411,
        0.5223529411764707,
        0.783529411764706,
    )

    # Test lightening an RGB color
    assert lighten_color((0.3, 0.55, 0.1), 0.5) == (
        0.6365384615384615,
        0.8961538461538462,
        0.42884615384615377,
    )

    # Test lightening a color by 0
    assert lighten_color("g", 0) == (1.0, 1.0, 1.0)

    # Test lightening a color by 1
    assert lighten_color("g", 1) == (0.0, 0.5, 0.0)


def test_average_color():
    assert average_color(["#FF0000", "#00FF00"]) == (0.5, 0.5, 0.0, 1.0)
    assert average_color(["#FF00007F", "#00FF003E"], keep_alpha=True) == (
        0.5,
        0.5,
        0.0,
        0.37058823529411766,
    )


# IMAGE


# COLORMAP


def test_colormap_sequential():
    """_"""
    assert sum(colormap_sequential("red", "green", "blue")(0.25)) == pytest.approx(
        1.750003844675125, rel=0.01
    )
