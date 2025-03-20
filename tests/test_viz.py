import pytest
from ladybug.color import Color
from ladybug_geometry.geometry2d import (
    LineSegment2D,
    Mesh2D,
    Point2D,
    Polygon2D,
    Polyline2D,
    Ray2D,
    Vector2D,
)
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

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
from kvrrj.viz.geometry import (
    LineSegment2D,
    LineString,
    Point,
    Point2D,
    Vector2D,
    _line_to_shapely,
    _mesh_to_shapely,
    _point2d_to_shapely,
    _polygon_to_shapely,
    _polyline_to_shapely,
    _ray_to_shapely,
    _to_shapely_2d,
    _vector2d_to_shapely,
)


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


# GEOMETRY


def test_point_to_shapely():
    point = Point2D(1, 2)
    shapely_point = _point2d_to_shapely(point)
    assert isinstance(shapely_point, Point)
    assert (shapely_point.x, shapely_point.y) == (1, 2)

    with pytest.raises(TypeError):
        _point2d_to_shapely("not a Point2D")


def test_vector_to_shapely():
    vector = Vector2D(3, 4)
    shapely_point = _vector2d_to_shapely(vector)
    assert isinstance(shapely_point, Point)
    assert (shapely_point.x, shapely_point.y) == (3, 4)

    with pytest.raises(TypeError):
        _vector2d_to_shapely(123)


def test_ray_to_shapely():
    ray = Ray2D(Point2D(0, 0), Vector2D(1, 1))
    shapely_line = _ray_to_shapely(ray)
    assert isinstance(shapely_line, LineString)
    assert list(shapely_line.coords) == [(0, 0), (1, 1)]

    with pytest.raises(TypeError):
        _ray_to_shapely(None)


def test_line_to_shapely():
    line = LineSegment2D(Point2D(0, 0), Point2D(1, 1))
    shapely_line = _line_to_shapely(line)
    assert isinstance(shapely_line, LineString)
    assert list(shapely_line.coords) == [(0, 0), (1, 1)]

    with pytest.raises(TypeError):
        _line_to_shapely([])


def test_polyline_to_shapely():
    polyline = Polyline2D([Point2D(0, 0), Point2D(1, 1), Point2D(2, 0)])
    shapely_line = _polyline_to_shapely(polyline)
    assert isinstance(shapely_line, LineString)
    assert list(shapely_line.coords) == [(0, 0), (1, 1), (2, 0)]

    with pytest.raises(TypeError):
        _polyline_to_shapely(3.14)


def test_polygon_to_shapely():
    polygon = Polygon2D([Point2D(0, 0), Point2D(1, 1), Point2D(2, 0)])
    shapely_polygon = _polygon_to_shapely(polygon)
    assert isinstance(shapely_polygon, Polygon)
    assert list(shapely_polygon.exterior.coords) == [(0, 0), (1, 1), (2, 0), (0, 0)]

    with pytest.raises(TypeError):
        _polygon_to_shapely({"invalid": "data"})


def test_mesh_to_shapely():
    mesh = Mesh2D.from_grid(Point2D(), num_x=2, num_y=2)
    shapely_multipolygon = _mesh_to_shapely(mesh)
    assert isinstance(shapely_multipolygon, MultiPolygon)
    assert len(shapely_multipolygon.geoms) == 4

    with pytest.raises(TypeError):
        _mesh_to_shapely(42)


def test_to_shapely_2d():
    point = Point2D(1, 2)
    vector = Vector2D(3, 4)
    ray = Ray2D(Point2D(0, 0), Vector2D(1, 1))
    line = LineSegment2D(Point2D(0, 0), Point2D(1, 1))
    polyline = Polyline2D([Point2D(0, 0), Point2D(1, 1), Point2D(2, 0)])
    polygon = Polygon2D([Point2D(0, 0), Point2D(1, 1), Point2D(2, 0)])
    mesh = Mesh2D.from_grid(Point2D(), num_x=2, num_y=2, x_dim=3, y_dim=3)

    assert isinstance(_to_shapely_2d(point), Point)
    assert isinstance(_to_shapely_2d(vector), Point)
    assert isinstance(_to_shapely_2d(ray), LineString)
    assert isinstance(_to_shapely_2d(line), LineString)
    assert isinstance(_to_shapely_2d(polyline), LineString)
    assert isinstance(_to_shapely_2d(polygon), Polygon)
    assert isinstance(_to_shapely_2d(mesh), MultiPolygon)

    with pytest.raises(ValueError):
        _to_shapely_2d("unsupported type")
