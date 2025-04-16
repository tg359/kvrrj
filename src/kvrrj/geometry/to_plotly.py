"""Methods for converting geomerty objects to Shapely Geometry."""

from functools import singledispatch
from typing import Any

import plotly.graph_objects as go
from honeybee.model import Aperture, Door, Face, Model, Room, Shade, ShadeMesh
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from ladybug_geometry.geometry2d import (
    Arc2D,
    LineSegment2D,
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
from plotly.basedatatypes import BaseTraceType

from kvrrj.color.to_ladybug_color import to_ladybug_color
from kvrrj.color.to_plotly_color import to_plotly_color


@singledispatch
def to_plotly(geo: Any, **kwargs) -> BaseTraceType | list[BaseTraceType]:
    """Convert an object to a plotly graph object."""
    raise NotImplementedError(f"Cannot convert {type(geo)} to plotly graph object.")


@to_plotly.register(Point2D)
@to_plotly.register(Vector2D)
def _(geo: Point2D | Vector2D, **kwargs) -> go.Scatter:
    return go.Scatter(x=[geo.x], y=[geo.y], mode="markers", **kwargs)


@to_plotly.register(Point3D)
@to_plotly.register(Vector3D)
def _(geo: Point3D | Vector3D, **kwargs) -> go.Scatter3d:
    return go.Scatter3d(x=[geo.x], y=[geo.y], z=[geo.z], mode="markers", **kwargs)


@to_plotly.register(Arc2D)
def _(geo: Arc2D, **kwargs) -> go.Scatter:
    divisions = 50
    pts = geo.subdivide_evenly(divisions)
    return go.Scatter(
        x=[pt.x for pt in pts], y=[pt.y for pt in pts], mode="lines", **kwargs
    )


@to_plotly.register(Arc3D)
def _(geo: Arc3D, **kwargs) -> go.Scatter3d:
    divisions = 50
    pts = geo.subdivide_evenly(divisions)
    return go.Scatter3d(
        x=[pt.x for pt in pts],
        y=[pt.y for pt in pts],
        z=[pt.z for pt in pts],
        mode="lines",
        **kwargs,
    )


@to_plotly.register(Ray2D)
def _(geo: Ray2D, **kwargs) -> go.Scatter:
    end = geo.p + geo.v
    return go.Scatter(x=[geo.p.x, end.x], y=[geo.p.y, end.y], mode="lines", **kwargs)


@to_plotly.register(Ray3D)
def _(geo: Ray3D, **kwargs) -> go.Scatter3d:
    end = geo.p + geo.v
    return go.Scatter3d(
        x=[geo.p.x, end.x],
        y=[geo.p.y, end.y],
        z=[geo.p.z, end.z],
        mode="lines",
        **kwargs,
    )


@to_plotly.register(LineSegment2D)
def _(geo: LineSegment2D, **kwargs) -> go.Scatter:
    return go.Scatter(
        x=(geo.p1.x, geo.p2.x), y=(geo.p1.y, geo.p2.y), mode="lines", **kwargs
    )


@to_plotly.register(LineSegment3D)
def _(geo: LineSegment3D, **kwargs) -> go.Scatter3d:
    return go.Scatter3d(
        x=(geo.p1.x, geo.p2.x),
        y=(geo.p1.y, geo.p2.y),
        z=(geo.p1.z, geo.p2.z),
        mode="lines",
        **kwargs,
    )


@to_plotly.register(Polyline2D)
def _(geo: Polyline2D, **kwargs) -> go.Scatter:
    return go.Scatter(
        x=[p.x for p in geo.vertices],
        y=[p.y for p in geo.vertices],
        mode="lines",
        **kwargs,
    )


@to_plotly.register(Polyline3D)
def _(geo: Polyline3D, **kwargs) -> go.Scatter3d:
    return go.Scatter3d(
        x=[p.x for p in geo.vertices],
        y=[p.y for p in geo.vertices],
        z=[p.z for p in geo.vertices],
        mode="lines",
        **kwargs,
    )


@to_plotly.register(Polygon2D)
def _(geo: Polygon2D, **kwargs) -> go.Scatter:
    return go.Scatter(
        x=[p.x for p in geo.vertices],
        y=[p.y for p in geo.vertices],
        fill="toself",
        **kwargs,
    )


@to_plotly.register(Mesh3D)
def _(geo: Mesh3D, **kwargs) -> go.Mesh3d:
    xyz = [pt.to_array() for pt in geo.vertices]
    x, y, z = [list(i) for i in zip(*xyz)]
    i, j, k = [list(i) for i in zip(*geo.faces)]
    return go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        **kwargs,
    )


@to_plotly.register(Face3D)
def _(geo: Face3D, **kwargs) -> go.Mesh3d:
    meshes = []
    if geo.has_holes:
        geos = geo.split_through_holes()
        for g in geos:
            meshes.append(g.triangulated_mesh3d)
    else:
        meshes.append(geo.triangulated_mesh3d)
    msh = Mesh3D.join_meshes(meshes)
    return to_plotly(msh, **kwargs)


@to_plotly.register(Polyface3D)
def _(geo: Polyface3D, **kwargs) -> go.Mesh3d:
    meshes = []
    for face in geo.faces:
        face: Face3D
        if face.has_holes:
            meshes.extend([i.triangulated_mesh3d for i in face.split_through_holes()])
        else:
            meshes.append(face.triangulated_mesh3d)

    return to_plotly(
        Mesh3D.join_meshes(meshes),
        **kwargs,
    )


@to_plotly.register(Door)
@to_plotly.register(Aperture)
@to_plotly.register(Shade)
@to_plotly.register(ShadeMesh)
def _(geo: Door | Aperture | Shade | ShadeMesh, **kwargs) -> go.Mesh3d:
    if kwargs.get("color") is None:
        kwargs["color"] = to_plotly_color(to_ladybug_color(geo))
    if kwargs.get("name") is None:
        kwargs["name"] = geo.identifier
    return to_plotly(geo.geometry, **kwargs)


@to_plotly.register(Face)
def _(geo: Face, **kwargs) -> go.Mesh3d:
    if kwargs.get("color") is None:
        kwargs["color"] = to_plotly_color(to_ladybug_color(geo))
    if kwargs.get("name") is None:
        kwargs["name"] = geo.identifier
    return to_plotly(geo.punched_geometry, **kwargs)


@to_plotly.register(Room)
def _(geo: Room, **kwargs) -> list[go.Mesh3d]:
    traces = []
    for objects in [geo.faces, geo.apertures, geo.doors, geo.shades]:
        for obj in objects:
            print(obj)
            traces.append(to_plotly(obj, **kwargs))
    return traces


@to_plotly.register(Model)
def _(geo: Model, **kwargs) -> list[BaseTraceType]:
    traces = []
    for obj in [
        geo.faces,
        geo.apertures,
        geo.doors,
        geo.shades,
        geo.properties.radiance.sensor_grids,
    ]:
        for item in obj:
            traces.append(to_plotly(item, **kwargs))
    return traces


@to_plotly.register(Sensor)
def _(geo: Sensor, **kwargs) -> go.Scatter3d:
    if kwargs.get("marker") is None:
        kwargs["marker"] = dict(size=2, color="black")
    if kwargs.get("customdata") is None:
        kwargs["customdata"] = f"u: {geo.dir[0]}\nv: {geo.dir[1]}\nw: {geo.dir[2]}"
    x, y, z = geo.pos
    return go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode="markers",
        **kwargs,
    )


@to_plotly.register(SensorGrid)
def _(geo: SensorGrid, **kwargs) -> go.Scatter3d:
    # set defaults
    if kwargs.get("marker") is None:
        kwargs["marker"] = dict(size=2, color="black")
    if kwargs.get("name") is None:
        kwargs["name"] = geo.identifier
    if kwargs.get("customdata") is None:
        kwargs["customdata"] = [
            f"u: {d[0]}\nv: {d[1]}\nw: {d[2]}" for d in geo.directions
        ]
    return go.Scatter3d(
        x=[i[0] for i in geo.positions],
        y=[i[1] for i in geo.positions],
        z=[i[2] for i in geo.positions],
        mode="markers",
        **kwargs,
    )


@to_plotly.register(tuple)
@to_plotly.register(list)
def _(geo: tuple | list, **kwargs) -> BaseTraceType | list[BaseTraceType]:
    if not all(isinstance(g, type(geo[0])) for g in geo):
        raise ValueError("Cannot convert a list of mixed datatypes.")

    if isinstance(geo[0], (Point2D, Vector2D)):
        return go.Scatter(
            x=[g.x for g in geo], y=[g.y for g in geo], mode="markers", **kwargs
        )

    if isinstance(geo[0], (Point3D, Vector3D)):
        return go.Scatter3d(
            x=[g.x for g in geo],
            y=[g.y for g in geo],
            z=[g.z for g in geo],
            mode="markers",
            **kwargs,
        )

    if isinstance(
        geo[0],
        (
            LineSegment2D,
            LineSegment3D,
            Ray2D,
            Ray3D,
            Arc2D,
            Arc3D,
            Polygon2D,
            Polyline2D,
            Polyline3D,
        ),
    ):
        return [to_plotly(g, **kwargs) for g in geo]

    if isinstance(geo[0], Face3D):
        return to_plotly(Polyface3D.from_faces(geo, tolerance=0.001), **kwargs)

    if isinstance(geo[0], Polyface3D):
        faces = []
        for item in geo:
            faces.extend(item.faces)
        return to_plotly(faces, **kwargs)

    if isinstance(geo[0], Mesh3D):
        return to_plotly(Mesh3D.join_meshes(geo), **kwargs)

    raise ValueError(f"Cannot convert a list of {type(geo[0])}.")
