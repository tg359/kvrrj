import pytest

from kvrrj.geometry.to_plotly import go, to_plotly

from . import (
    ARC2D_CLOSED,
    ARC2D_OPEN,
    ARC3D_CLOSED,
    ARC3D_OPEN,
    FACE3D,
    FACE3D_HOLEY,
    MESH2D,
    MESH3D,
    MODEL,
    POINT2D,
    POINT3D,
    POLYFACE3D,
    POLYGON2D,
    POLYLINE2D,
    POLYLINE3D,
    RAY2D,
    RAY3D,
    ROOM,
    SEGMENT2D,
    SEGMENT3D,
    SENSOR,
    SENSORGRID,
    SHADE,
    SHADEMESH,
    VECTOR2D,
    VECTOR3D,
)

# region: TO_PLOTLY


def test_point2d_to_plotly():
    result = to_plotly(POINT2D)
    assert isinstance(result, go.Scatter)


def test_point2d_iterable_to_plotly():
    result = to_plotly([POINT2D, POINT2D])
    assert isinstance(result, go.Scatter)


def test_vector2d_to_plotly():
    result = to_plotly(VECTOR2D)
    assert isinstance(result, go.Scatter)


def test_vector2d_iterable_to_plotly():
    result = to_plotly([VECTOR2D, VECTOR2D])
    assert isinstance(result, go.Scatter)


def test_point3d_to_plotly():
    result = to_plotly(POINT3D)
    assert isinstance(result, go.Scatter3d)


def test_vector3d_to_plotly():
    result = to_plotly(VECTOR3D)
    assert isinstance(result, go.Scatter3d)


def test_vector3d_iterable_to_plotly():
    result = to_plotly([VECTOR3D, VECTOR3D])
    assert isinstance(result, go.Scatter3d)


def test_arc2d_to_plotly():
    assert isinstance(to_plotly(ARC2D_OPEN), go.Scatter)
    assert isinstance(to_plotly(ARC2D_CLOSED), go.Scatter)


def test_arc2d_iterable_to_plotly():
    result = to_plotly([ARC2D_OPEN, ARC2D_CLOSED])
    assert all(isinstance(i, go.Scatter) for i in result)


def test_arc3d_to_plotly():
    assert isinstance(to_plotly(ARC3D_CLOSED), go.Scatter3d)
    assert isinstance(to_plotly(ARC3D_CLOSED), go.Scatter3d)


def test_arc3d_iterable_to_plotly():
    result = to_plotly([ARC3D_OPEN, ARC3D_CLOSED])
    assert all(isinstance(i, go.Scatter3d) for i in result)


def test_ray2d_to_plotly():
    result = to_plotly(RAY2D)
    assert isinstance(result, go.Scatter)


def test_ray2d_iterable_to_plotly():
    result = to_plotly([RAY2D, RAY2D])
    assert all(isinstance(i, go.Scatter) for i in result)


def test_ray3d_to_plotly():
    result = to_plotly(RAY3D)
    assert isinstance(result, go.Scatter3d)


def test_ray3d_iterable_to_plotly():
    result = to_plotly([RAY3D, RAY3D])
    assert all(isinstance(i, go.Scatter3d) for i in result)


def test_linesegment2d_to_plotly():
    result = to_plotly(SEGMENT2D)
    assert isinstance(result, go.Scatter)


def test_linesegment2d_iterable_to_plotly():
    result = to_plotly([SEGMENT2D, SEGMENT2D])
    assert all(isinstance(i, go.Scatter) for i in result)


def test_linesegment3d_to_plotly():
    result = to_plotly(SEGMENT3D)
    assert isinstance(result, go.Scatter3d)


def test_linesegment3d_iterable_to_plotly():
    result = to_plotly([SEGMENT3D, SEGMENT3D])
    assert all(isinstance(i, go.Scatter3d) for i in result)


def test_polyline2d_to_plotly():
    result = to_plotly(POLYLINE2D)
    assert isinstance(result, go.Scatter)


def test_polyline2d_iterable_to_plotly():
    result = to_plotly([POLYLINE2D, POLYLINE2D])
    assert all(isinstance(i, go.Scatter) for i in result)


def test_polyline3d_to_plotly():
    result = to_plotly(POLYLINE3D)
    assert isinstance(result, go.Scatter3d)


def test_polyline3d_iterable_to_plotly():
    result = to_plotly([POLYLINE3D, POLYLINE3D])
    assert all(isinstance(i, go.Scatter3d) for i in result)


def test_polygon2d_to_plotly():
    result = to_plotly(POLYGON2D)
    assert isinstance(result, go.Scatter)


def test_polygon2d_iterable_to_plotly():
    result = to_plotly([POLYGON2D, POLYGON2D])
    assert all(isinstance(i, go.Scatter) for i in result)


def test_face3d_to_plotly():
    result = to_plotly(FACE3D)
    assert isinstance(result, go.Mesh3d)
    result = to_plotly(FACE3D_HOLEY)
    assert isinstance(result, go.Mesh3d)


def test_face3d_iterable_to_plotly():
    result = to_plotly([FACE3D, FACE3D_HOLEY])
    assert isinstance(result, go.Mesh3d)


def test_polyface3d_to_plotly():
    result = to_plotly(POLYFACE3D)
    assert isinstance(result, go.Mesh3d)


def test_polyface3d_iterable_to_plotly():
    result = to_plotly([POLYFACE3D, POLYFACE3D])
    assert isinstance(result, go.Mesh3d)


def test_mesh2d_to_plotly():
    with pytest.raises(NotImplementedError):
        to_plotly(MESH2D)


def test_mesh3d_to_plotly():
    result = to_plotly(MESH3D)
    assert isinstance(result, go.Mesh3d)


def test_mesh3d_iterable_to_plotly():
    result = to_plotly([MESH3D, MESH3D])
    assert isinstance(result, go.Mesh3d)


def test_sensorgrid_to_plotly():
    result = to_plotly(SENSORGRID)
    assert isinstance(result, go.Scatter3d)


def test_sensor_to_plotly():
    result = to_plotly(SENSOR)
    assert isinstance(result, go.Scatter3d)


def test_room_to_plotly():
    result = to_plotly(ROOM)
    assert all(isinstance(i, go.Mesh3d) for i in result)


def test_shade_to_plotly():
    result = to_plotly(SHADE)
    assert isinstance(result, go.Mesh3d)


def test_shademesh_to_plotly():
    result = to_plotly(SHADEMESH)
    assert isinstance(result, go.Mesh3d)


def test_model_to_plotly():
    result = to_plotly(MODEL)
    assert all(isinstance(i, (go.Mesh3d, go.Scatter3d)) for i in result)


# endregion: TO_PLOTLY
