from functools import singledispatch
from typing import Any

from honeybee.model import Aperture, Door, Face, Face3D
from honeybee.shade import Shade
from honeybee_radiance.modifier.material.glass import Glass
from ladybug_geometry.geometry3d import Face3D, Mesh3D


@singledispatch
def to_shades(geo: Any, **kwargs) -> list[Shade]:
    """Convert a geometry-like object to a list of shades."""
    raise NotImplementedError(f"Cannot convert {type(geo)}.")


@to_shades.register(Mesh3D)
def _(geo: Mesh3D, **kwargs) -> list[Shade]:
    if kwargs.get("transmittance") is None:
        kwargs["transmittance"] = 0
    modifier = Glass.from_single_transmittance(
        identifier=f"glass_{kwargs['transmittance']}",
        rgb_transmittance=kwargs["transmittance"],
    )
    shades = []
    for n, face_vertices in enumerate(geo.face_vertices):
        shd = Shade.from_vertices(identifier=f"shade_{n}", vertices=face_vertices)
        shd.properties.radiance.modifier = modifier
        shades.append(shd)
    return shades


@to_shades.register(Face3D)
def _(geo: Face3D, **kwargs) -> list[Shade]:
    return to_shades(
        Mesh3D.join_meshes(
            [face.triangulated_mesh3d for face in geo.split_through_holes()]
        ),
        **kwargs,
    )


@to_shades.register(Door)
@to_shades.register(Aperture)
def _(geo: Door | Aperture, **kwargs) -> list[Shade]:
    # get the transmittance from the door/aperture
    if kwargs.get("transmittance") is None:
        kwargs["transmittance"] = geo.properties.radiance.modifier.transmittance
    return to_shades(geo.geometry, **kwargs)


@to_shades.register(Face)
def _(geo: Face, **kwargs) -> list[Shade]:
    # get the shades representing the sub-faces
    shades = []
    if geo.has_sub_faces:
        for sub_face in geo.sub_faces:
            shades.extend(to_shades(sub_face, **kwargs))
    # convert to mesh
    shades.extend(to_shades(geo.punched_geometry, **kwargs))
    return shades
