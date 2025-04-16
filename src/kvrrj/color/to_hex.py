from functools import singledispatch
from typing import Any

from ladybug.color import Color
from ladybug.datatype.base import DataTypeBase
from matplotlib.colors import to_hex as mpl_to_hex

from .to_rgba import to_rgba


@singledispatch
def to_hex(obj: Any) -> str:
    """Convert a color-like object to hex."""
    raise NotImplementedError(f"Cannot convert {type(obj)}.")


@to_hex.register(Color)
@to_hex.register(str)
@to_hex.register(tuple)
@to_hex.register(list)
@to_hex.register(DataTypeBase)
def _(obj: Color | str | tuple | list | DataTypeBase) -> str:
    rgba = to_rgba(obj)
    return mpl_to_hex(rgba, keep_alpha=True)
