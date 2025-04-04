from typing import Any

from honeybee.model import Face, Model
from honeybee.shade import Shade
from honeybee_radiance.modifier.material.glass import Glass
from honeybee_radiance.sensorgrid import Sensor, SensorGrid
from honeybee_radiance_command.options.rfluxmtx import RfluxmtxOptions
from honeybee_radiance_command.options.rpict import RpictOptions
from honeybee_radiance_command.options.rtrace import RtraceOptions
from ladybug_geometry.geometry3d import Face3D, Mesh3D, Point3D, Polyface3D, Vector3D


def create_origin_sensor_grid(identifier: str = "sensorgrid") -> SensorGrid:
    return SensorGrid.from_position_and_direction(
        identifier=identifier,
        positions=[Point3D(0, 0, 0.001)],
        directions=[Vector3D(0, 0, 1)],
    )


def geo_to_shade(geo: Any) -> list[Shade]:
    shades = []
    if isinstance(geo, Shade):
        shades.append(geo)
    elif isinstance(geo, Mesh3D):
        for n, face_vertices in enumerate(geo.face_vertices):
            shades.append(
                Shade.from_vertices(identifier=f"shade_{n}", vertices=face_vertices)
            )
    elif isinstance(geo, Face3D):
        shades.append(Shade.from_vertices(identifier="shade", vertices=geo.vertices))
    elif isinstance(geo, Polyface3D):
        for n, face in enumerate(geo.faces):
            shades.append(
                Shade.from_vertices(identifier=f"shade_{n}", vertices=face.vertices)
            )
    elif isinstance(geo, Face):
        shd = Shade.from_vertices(identifier="shade", vertices=geo.vertices)
        shd.properties.radiance.modifier = Glass.from_single_transmittance(
            geo.properties.radiance.modifier.transmittance
        )
        shades.append(shd)
    else:
        raise ValueError(f"({type(geo)}) is not supported for conversion into a shade.")
    return shades


def radiance_parameters(
    model: Model,
    detail_dim: float,
    recipe_type: str,
    detail_level: int = 0,
    additional_parameters: str = None,
) -> str:
    """Generate the default "recommended" Radiance parameters for a Honeybee
    Radiance simulation.

    This method also includes the estimation of ambient resolution based on the
    model dimensions.

    Args:
        model: Model
            A Honeybee Model.
        detail_dim: float
            The detail dimension in meters.
        recipe_type: str
            One of the following: 'point-in-time-grid', 'daylight-factor',
            'point-in-time-image', 'annual'.
        detail_level: int
            One of 0 (low), 1 (medium) or 2 (high).
        additional_parameters: str
            Additional parameters to add to the Radiance command. Should be in
            the format of a Radiance command string e.g. '-ab 2 -aa 0.25'.

    Returns:
        str: The Radiance parameters as a string.
    """

    # recommendations for radiance parameters
    rtrace = {
        "ab": [2, 3, 6],
        "ad": [512, 2048, 4096],
        "as_": [128, 2048, 4096],
        "ar": [16, 64, 128],
        "aa": [0.25, 0.2, 0.1],
        "dj": [0, 0.5, 1],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.05, 0.01, 0.005],
        "ss": [0, 0.7, 1],
    }

    rpict = {
        "ab": [2, 3, 6],
        "ad": [512, 2048, 4096],
        "as_": [128, 2048, 4096],
        "ar": [16, 64, 128],
        "aa": [0.25, 0.2, 0.1],
        "ps": [8, 4, 2],
        "pt": [0.15, 0.10, 0.05],
        "pj": [0.6, 0.9, 0.9],
        "dj": [0, 0.5, 1],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.05, 0.01, 0.005],
        "ss": [0, 0.7, 1],
    }

    rfluxmtx = {
        "ab": [3, 5, 6],
        "ad": [5000, 15000, 25000],
        "as_": [128, 2048, 4096],
        "ds": [0.5, 0.25, 0.05],
        "dt": [0.5, 0.25, 0.15],
        "dc": [0.25, 0.5, 0.75],
        "dr": [0, 1, 3],
        "dp": [64, 256, 512],
        "st": [0.85, 0.5, 0.15],
        "lr": [4, 6, 8],
        "lw": [0.000002, 6.67e-07, 4e-07],
        "ss": [0, 0.7, 1],
        "c": [1, 1, 1],
    }

    # VALIDATION
    recipe_types = {
        "point-in-time-grid": [rtrace, RtraceOptions()],
        "daylight-factor": [rtrace, RtraceOptions()],
        "point-in-time-image": [rpict, RpictOptions()],
        "annual": [rfluxmtx, RfluxmtxOptions()],
        "annual-daylight": [rfluxmtx, RfluxmtxOptions()],
        "annual-irradiance": [rfluxmtx, RfluxmtxOptions()],
        "sky-view": [rtrace, RtraceOptions()],
    }
    if recipe_type not in recipe_types:
        raise ValueError(
            f"recipe_type ({recipe_type}) must be one of {recipe_types.keys()}"
        )

    if detail_level not in [0, 1, 2]:
        raise ValueError(
            r"detail_level ({detail_level}) must be one of 0 (low), 1 (medium) or 2 (high)."
        )

    options, obj = recipe_types[recipe_type]
    for opt, vals in options.items():
        setattr(obj, opt, vals[detail_level])

    min_pt, max_pt = model.min, model.max
    x_dim = max_pt.x - min_pt.x
    y_dim = max_pt.y - min_pt.y
    z_dim = max_pt.z - min_pt.z
    longest_dim = max((x_dim, y_dim, z_dim))
    try:
        obj.ar = int((longest_dim * obj.aa) / detail_dim)
    except TypeError as _:
        obj.ar = int((longest_dim * 0.1) / detail_dim)

    if additional_parameters:
        obj.update_from_string(additional_parameters)

    return obj.to_radiance()
