import warnings
from functools import singledispatch
from typing import Any

from ladybug.color import Color
from ladybug.datatype.base import DataTypeBase
from matplotlib.colors import to_hex as mpl_to_hex
from matplotlib.colors import to_rgba as mpl_to_rgba

from .util import _datatype_to_color, _lb_color_to_rgba, _plotly_color_to_rgba


@singledispatch
def to_rgba(obj: Any) -> tuple[float, float, float, float]:
    """Convert a color-like object to rgba, with values between 0 and 1."""
    raise NotImplementedError(f"Cannot convert {type(obj)}.")


@to_rgba.register(Color)
def _(obj: Color) -> tuple[float, float, float, float]:
    return _lb_color_to_rgba(obj)


@to_rgba.register(str)
def _(obj: str) -> tuple[float, float, float, float]:
    # plotly
    if obj.startswith("rgba(") or obj.startswith("rgb("):
        return _plotly_color_to_rgba(obj)
    # hex
    if obj.startswith("#"):
        hex = mpl_to_hex(obj, keep_alpha=True)
        return mpl_to_rgba(hex)
    # other color-strings, named, ...
    return mpl_to_rgba(obj)


@to_rgba.register(list)
@to_rgba.register(tuple)
def _(obj: list | tuple) -> tuple[float, float, float, float]:
    if (len(obj) not in [3, 4]) or not all(isinstance(i, (float, int)) for i in obj):
        raise ValueError("A color from a list must be a 3 or 4 length list of numbers.")
    if any(i > 1 for i in obj):
        obj = [i / 255 for i in obj]
    if all(i == 1 for i in obj):
        warnings.warn(
            "It is likely that the color-passed is white, represented in RGB values between 0-255. To create a dark color at 1, 1, 1, (1), instead use [1/255, 1/255, 1/255, (1/255)]."
        )
    return mpl_to_rgba(obj)


@to_rgba.register(DataTypeBase)
def _(obj: DataTypeBase) -> tuple[float, float, float, float]:
    return to_rgba(_datatype_to_color(obj))
