import math

from ladybug_geometry.geometry2d import (
    Arc2D,
    LineSegment2D,
    Mesh2D,
    Point2D,
    Polygon2D,
    Polyline2D,
    Ray2D,
    Vector2D,
)
from ladybug_geometry.geometry3d import (
    Arc3D,
    Face3D,
    LineSegment3D,
    Mesh3D,
    Plane,
    Point3D,
    Polyface3D,
    Polyline3D,
    Ray3D,
    Vector3D,
)

from kvrrj.geometry.to_shapely import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    to_shapely,
)

# construct all possible geometry types in Ladybug
POINT2D = Point2D(1, 2)
POINT3D = Point3D(1, 2, 3)
VECTOR2D = Vector2D(3, 4)
VECTOR3D = Vector3D(3, 4, 5)
ARC2D_OPEN = Arc2D(Point2D(0, 0), 5, 0, 1 * math.pi)
ARC2D_CLOSED = Arc2D(Point2D(0, 0), 5, 0, 2 * math.pi)
ARC3D_OPEN = Arc3D(Plane(), 5, 0, 1 * math.pi)
ARC3D_CLOSED = Arc3D(Plane(), 5, 0, 2 * math.pi)
RAY2D = Ray2D(Point2D(0, 0), Vector2D(1, 1))
RAY3D = Ray3D(Point3D(0, 0, 0), Vector3D(1, 1, 1))
SEGMENT2D = LineSegment2D(Point2D(0, 0), Point2D(3, 4))
SEGMENT3D = LineSegment3D(Point3D(0, 0, 0), Point3D(4, 5, 6))
POLYLINE2D = Polyline2D([Point2D(0, 0), Point2D(3, 4), Point2D(5, 6)])
POLYLINE3D = Polyline3D([Point3D(0, 0, 0), Point3D(4, 5, 6), Point3D(7, 8, 9)])
POLYGON2D = Polygon2D([Point2D(0, 0), Point2D(3, 4), Point2D(5, 6)])
FACE3D = Face3D.from_regular_polygon(side_count=5)
FACE3D_HOLEY = Face3D.from_punched_geometry(
    base_face=Face3D.from_regular_polygon(side_count=4, radius=5),
    sub_faces=[Face3D.from_regular_polygon(side_count=3, radius=2)],
)
POLYFACE3D = Polyface3D.from_faces(
    [Face3D.from_regular_polygon(side_count=5)], tolerance=0.1
)
MESH2D = Mesh2D.from_face_vertices(
    faces=[
        Polygon2D([Point2D(0, 0), Point2D(0, 1), Point2D(1, 1)]),
        Polygon2D([Point2D(0, 0), Point2D(1, 1), Point2D(1, 0)]),
    ]
)
MESH3D = Mesh3D.from_face_vertices(
    faces=[
        Face3D([Point3D(0, 0, 0), Point3D(0, 1, 0), Point3D(1, 1, 0)]),
        Face3D([Point3D(0, 0, 0), Point3D(1, 1, 0), Point3D(1, 0, 0)]),
    ]
)


# region: TO_SHAPELY


def test_point2d_to_shapely():
    result = to_shapely(POINT2D)
    assert isinstance(result, Point)


def test_point2d_iterable_to_shapely():
    result = to_shapely([POINT2D])
    assert isinstance(result, MultiPoint)


def test_vector2d_to_shapely():
    result = to_shapely(VECTOR2D)
    assert isinstance(result, Point)


def test_vector2d_iterable_to_shapely():
    result = to_shapely([VECTOR2D])
    assert isinstance(result, MultiPoint)


def test_point3d_to_shapely():
    result = to_shapely(POINT3D)
    assert isinstance(result, Point)


def test_point3d_iterable_to_shapely():
    result = to_shapely([POINT3D])
    assert isinstance(result, MultiPoint)


def test_vector3d_to_shapely():
    result = to_shapely(VECTOR3D)
    assert isinstance(result, Point)


def test_vector3d_iterable_to_shapely():
    result = to_shapely([VECTOR3D])
    assert isinstance(result, MultiPoint)


def test_arc2d_to_shapely():
    assert isinstance(to_shapely(ARC2D_OPEN), LineString)
    assert isinstance(to_shapely(ARC2D_CLOSED), LinearRing)


def test_arc2d_iterable_to_shapely():
    result = to_shapely([ARC2D_OPEN, ARC2D_CLOSED])
    assert isinstance(result, MultiLineString)


def test_arc3d_to_shapely():
    assert isinstance(to_shapely(ARC3D_CLOSED), LineString)
    assert isinstance(to_shapely(ARC3D_CLOSED), LinearRing)


def test_arc3d_iterable_to_shapely():
    result = to_shapely([ARC3D_OPEN, ARC3D_CLOSED])
    assert isinstance(result, MultiLineString)


def test_ray2d_to_shapely():
    result = to_shapely(RAY2D)
    assert isinstance(result, LineString)


def test_ray2d_iterable_to_shapely():
    result = to_shapely([RAY2D])
    assert isinstance(result, MultiLineString)


def test_ray3d_to_shapely():
    result = to_shapely(RAY3D)
    assert isinstance(result, LineString)


def test_ray3d_iterable_to_shapely():
    result = to_shapely([RAY3D])
    assert isinstance(result, MultiLineString)


def test_linesegment2d_to_shapely():
    result = to_shapely(SEGMENT2D)
    assert isinstance(result, LineString)


def test_linesegment2d_iterable_to_shapely():
    result = to_shapely([SEGMENT2D])
    assert isinstance(result, MultiLineString)


def test_linesegment3d_to_shapely():
    result = to_shapely(SEGMENT3D)
    assert isinstance(result, LineString)


def test_linesegment3d_iterable_to_shapely():
    result = to_shapely([SEGMENT3D])
    assert isinstance(result, MultiLineString)


def test_polyline2d_to_shapely():
    result = to_shapely(POLYLINE2D)
    assert isinstance(result, LineString)


def test_polyline2d_iterable_to_shapely():
    result = to_shapely([POLYLINE2D])
    assert isinstance(result, MultiLineString)


def test_polyline3d_to_shapely():
    result = to_shapely(POLYLINE3D)
    assert isinstance(result, LineString)


def test_polyline3d_iterable_to_shapely():
    result = to_shapely([POLYLINE3D])
    assert isinstance(result, MultiLineString)


def test_polygon2d_to_shapely():
    result = to_shapely(POLYGON2D)
    assert isinstance(result, Polygon)


def test_polygon2d_iterable_to_shapely():
    result = to_shapely([POLYGON2D])
    assert isinstance(result, MultiPolygon)


def test_face3d_to_shapely():
    result = to_shapely(FACE3D)
    assert isinstance(result, MultiPolygon)
    result = to_shapely(FACE3D_HOLEY)
    assert isinstance(result, MultiPolygon)


def test_face3d_iterable_to_shapely():
    result = to_shapely([FACE3D, FACE3D_HOLEY])
    assert isinstance(result, MultiPolygon)


def test_polyface3d_to_shapely():
    result = to_shapely(POLYFACE3D)
    assert isinstance(result, MultiPolygon)


def test_polyface3d_iterable_to_shapely():
    result = to_shapely([POLYFACE3D])
    assert isinstance(result, MultiPolygon)


def test_mesh2d_to_shapely():
    result = to_shapely(MESH2D)
    assert isinstance(result, MultiPolygon)


def test_mesh2d_iterable_to_shapely():
    result = to_shapely([MESH2D])
    assert isinstance(result, MultiPolygon)


def test_mesh3d_to_shapely():
    result = to_shapely(MESH3D)
    assert isinstance(result, MultiPolygon)


def test_mesh3d_iterable_to_shapely():
    result = to_shapely([MESH3D])
    assert isinstance(result, MultiPolygon)


# endregion: TO_SHAPELY
