from functools import singledispatch
from typing import Any

from ladybug.color import Color
from ladybug.datatype.base import DataTypeBase

from .to_rgba import to_rgba


@singledispatch
def to_plotly_color(obj: Any) -> str:
    """Convert a color-like object to its plotly representation."""
    raise NotImplementedError(f"Cannot convert {type(obj)}.")


@to_plotly_color.register(Color)
@to_plotly_color.register(str)
@to_plotly_color.register(tuple)
@to_plotly_color.register(list)
@to_plotly_color.register(DataTypeBase)
def _(obj: str) -> str:
    r, g, b, a = to_rgba(obj)
    return f"rgba({(r * 255)}, {(g * 255)}, {(b * 255)}, {float(a)})"
