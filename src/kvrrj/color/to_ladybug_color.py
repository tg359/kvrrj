"""Methods for converting objects/classes to corresponding colors."""

from functools import singledispatch
from typing import Any

from honeybee.model import (
    Aperture,
    Door,
    Face,
    Shade,
    ShadeMesh,
)
from ladybug.color import Color
from ladybug.datatype.base import DataTypeBase

from .to_rgba import to_rgba


@singledispatch
def to_ladybug_color(obj: Any) -> Color:
    """Get the color of an object."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to a color.")


@to_ladybug_color.register(str)
@to_ladybug_color.register(tuple)
@to_ladybug_color.register(list)
@to_ladybug_color.register(Color)
@to_ladybug_color.register(DataTypeBase)
def _(obj: str | tuple | list | Color | DataTypeBase) -> Color:
    r, g, b, a = to_rgba(obj)
    return Color(int(r * 255), int(g * 255), int(b * 255), int(a * 255))


@to_ladybug_color.register(Door)
@to_ladybug_color.register(Aperture)
@to_ladybug_color.register(Face)
@to_ladybug_color.register(Shade)
@to_ladybug_color.register(ShadeMesh)
def _(obj: Door | Aperture | Face | Shade | ShadeMesh) -> Color:
    return obj.type_color
