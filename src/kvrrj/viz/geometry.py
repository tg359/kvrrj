"""Utilities for converting lb geometry to other types of geometry."""

import plotly.graph_objects as go
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

LB_GEOMETRY_2D = (
    Vector2D | Point2D | Ray2D | LineSegment2D | Polyline2D | Polygon2D | Mesh2D
)
SHAPELY_GEOMETRY_2D = Point | Polygon | MultiPolygon | LineString


# def _mesh2d_to_pymesh(obj: Mesh2D) -> BaseMesh:
#     if not isinstance(obj, Mesh2D):
#         raise TypeError(f"Expected Mesh2D, got {type(obj)}")
#     m = BaseMesh()
#     m.data = obj.to_vertices_and_faces()
#     return


def _point2d_to_shapely(obj: Point2D | Vector2D) -> Point:
    if not isinstance(obj, (Point2D, Vector2D)):
        raise TypeError(f"Expected Point2D or Vector2D, got {type(obj)}")
    return Point(obj.x, obj.y)


def _points2d_to_plotly(objs: list[Point2D], **kwargs) -> go.Scatter:
    if not isinstance(objs, (list, tuple)):
        raise TypeError(f"Expected list or tuple, got {type(objs)}")
    for obj in objs:
        if not isinstance(obj, Point2D):
            raise TypeError(f"Expected Point2D, got {type(obj)}")
    return go.Scatter(
        x=[o.x for o in objs],
        y=[o.y for o in objs],
        mode="markers",
        marker=dict(size=10, color="red"),
    )


def _vector2d_to_shapely(obj: Vector2D) -> Point:
    if not isinstance(obj, Vector2D):
        raise TypeError(f"Expected Vector2D, got {type(obj)}")
    return _point2d_to_shapely(obj)


def _ray_to_shapely(obj: Ray2D) -> LineString:
    if not isinstance(obj, Ray2D):
        raise TypeError(f"Expected Ray2D, got {type(obj)}")
    end = obj.p + obj.v
    return LineString([(obj.p.x, obj.p.y), (end.x, end.y)])


def _line_to_shapely(obj: LineSegment2D) -> LineString:
    if not isinstance(obj, LineSegment2D):
        raise TypeError(f"Expected LineSegment2D, got {type(obj)}")
    return LineString([(obj.p1.x, obj.p1.y), (obj.p2.x, obj.p2.y)])


def _polyline_to_shapely(obj: Polyline2D) -> LineString:
    if not isinstance(obj, Polyline2D):
        raise TypeError(f"Expected Polyline2D, got {type(obj)}")
    return LineString([(pt.x, pt.y) for pt in obj.vertices])


def _polygon_to_shapely(obj: Polygon2D) -> Polygon:
    if not isinstance(obj, Polygon2D):
        raise TypeError(f"Expected Polygon2D, got {type(obj)}")
    return Polygon(
        [(pt.x, pt.y) for pt in obj.vertices] + [(obj.vertices[0].x, obj.vertices[0].y)]
    )


def _mesh_to_shapely(obj: Mesh2D) -> MultiPolygon:
    if not isinstance(obj, Mesh2D):
        raise TypeError(f"Expected Mesh2D, got {type(obj)}")
    polylines: list[Polyline2D] = obj.face_edges
    polygons = [Polygon2D(polyline.vertices) for polyline in polylines]
    return MultiPolygon([_polygon_to_shapely(poly) for poly in polygons])


def _to_shapely_2d(obj: LB_GEOMETRY_2D) -> SHAPELY_GEOMETRY_2D:
    if isinstance(obj, Point2D):
        return _point2d_to_shapely(obj)
    if isinstance(obj, Vector2D):
        return _vector2d_to_shapely(obj)
    if isinstance(obj, Ray2D):
        return _ray_to_shapely(obj)
    if isinstance(obj, LineSegment2D):
        return _line_to_shapely(obj)
    if isinstance(obj, Polyline2D):
        return _polyline_to_shapely(obj)
    if isinstance(obj, Polygon2D):
        return _polygon_to_shapely(obj)
    if isinstance(obj, Mesh2D):
        return _mesh_to_shapely(obj)
    raise ValueError(f"Unsupported geometry type: {type(obj)}")
