"""Methods for converting geomerty objects to Shapely Geometry."""

from functools import singledispatch
from typing import Any

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
    Point3D,
    Polyface3D,
    Polyline3D,
    Ray3D,
    Vector3D,
)
from shapely import Geometry
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


@singledispatch
def to_shapely(geo: Any) -> Geometry:
    """Convert an object to Shapely Geometry."""
    raise NotImplementedError(f"Cannot convert {type(geo)} to Shapely Geometry.")


@to_shapely.register(Point2D)
@to_shapely.register(Vector2D)
def _(geo: Point2D | Vector2D) -> Point:
    return Point(geo.x, geo.y)


@to_shapely.register(Point3D)
@to_shapely.register(Vector3D)
def _(geo: Point3D | Vector3D) -> Point:
    return Point(geo.x, geo.y, geo.z)


@to_shapely.register(Arc2D)
def _(geo: Arc2D) -> LineString | LinearRing:
    divisions = 50
    pts = geo.subdivide_evenly(divisions)
    if geo.is_circle:
        return LinearRing([(pt.x, pt.y) for pt in pts])
    return LineString([(pt.x, pt.y) for pt in pts])


@to_shapely.register(Arc3D)
def _(geo: Arc3D) -> LineString | LinearRing:
    divisions = 50
    pts = geo.subdivide_evenly(divisions)
    if geo.is_circle:
        return LinearRing([(pt.x, pt.y, pt.z) for pt in pts])
    return LineString([(pt.x, pt.y, pt.z) for pt in pts])


@to_shapely.register(Ray2D)
def _(geo: Ray2D) -> LineString:
    end = geo.p + geo.v
    return LineString([(geo.p.x, geo.p.y), (end.x, end.y)])


@to_shapely.register(Ray3D)
def _(geo: Ray3D) -> LineString:
    end = geo.p + geo.v
    return LineString([(geo.p.x, geo.p.y, geo.p.z), (end.x, end.y, end.z)])


@to_shapely.register(LineSegment2D)
def _(geo: LineSegment2D) -> LineString:
    return LineString([(geo.p1.x, geo.p1.y), (geo.p2.x, geo.p2.y)])


@to_shapely.register(LineSegment3D)
def _(geo: LineSegment3D) -> LineString:
    return LineString([(geo.p1.x, geo.p1.y, geo.p1.z), (geo.p2.x, geo.p2.y, geo.p2.z)])


@to_shapely.register(Polyline2D)
def _(geo: Polyline2D) -> LineString:
    return LineString([(pt.x, pt.y) for pt in geo.vertices])


@to_shapely.register(Polyline3D)
def _(geo: Polyline3D) -> LineString:
    return LineString([(pt.x, pt.y, pt.z) for pt in geo.vertices])


@to_shapely.register(Polygon2D)
def _(geo: Polygon2D) -> Polygon:
    exterior = [(pt.x, pt.y) for pt in geo.vertices]
    return Polygon(shell=exterior)


@to_shapely.register(Face3D)
def _(geo: Face3D) -> MultiPolygon:
    faces = geo.split_through_holes()
    polygons: list[Polygon] = []
    for face in faces:
        polygons.append(Polygon(shell=[(pt.x, pt.y, pt.z) for pt in face.vertices]))
    return MultiPolygon(polygons=polygons)


@to_shapely.register(Polyface3D)
def _(geo: Polyface3D) -> MultiPolygon:
    multipolygons = [to_shapely(face) for face in geo.faces]
    polygons = []
    for mp in multipolygons:
        polygons.extend(mp.geoms)
    return MultiPolygon(polygons)


@to_shapely.register(Mesh2D)
def _(obj: Mesh2D) -> MultiPolygon:
    polygons: list[Polygon2D] = []
    for i in range(len(obj.faces)):
        vertices = [obj.vertices[j] for j in obj.faces[i]]
        polygons.append(Polygon2D(vertices))
    return MultiPolygon([to_shapely(poly) for poly in polygons])


@to_shapely.register(Mesh3D)
def _(geo: Mesh3D) -> MultiPolygon:
    mps = []
    for i in range(len(geo.faces)):
        vertices = [geo.vertices[j] for j in geo.faces[i]]
        mps.append(to_shapely(Face3D(vertices)).geoms)
    return MultiPolygon([poly for sublist in mps for poly in sublist])


@to_shapely.register(tuple)
@to_shapely.register(list)
def _(geo: tuple | list) -> Geometry:
    if not all(isinstance(g, type(geo[0])) for g in geo):
        raise ValueError("Cannot convert a list of mixed datatypes.")

    if isinstance(geo[0], (Point2D, Point3D, Vector2D, Vector3D)):
        return MultiPoint([to_shapely(g) for g in geo])

    if isinstance(
        geo[0],
        (
            LineSegment2D,
            LineSegment3D,
            Ray2D,
            Ray3D,
            Polyline2D,
            Polyline3D,
            Arc2D,
            Arc3D,
        ),
    ):
        return MultiLineString([to_shapely(g) for g in geo])

    if isinstance(geo[0], Polygon2D):
        return MultiPolygon([to_shapely(g) for g in geo])

    if isinstance(geo[0], (Face3D, Polyface3D, Mesh2D, Mesh3D)):
        mps = [to_shapely(g).geoms for g in geo]
        return MultiPolygon([poly for sublist in mps for poly in sublist])

    raise ValueError(f"Cannot convert a list of {type(geo[0])}.")
