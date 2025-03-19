import colorsys
from typing import Any

import numpy as np
from ladybug.color import Color
from matplotlib.colors import colorConverter
from matplotlib.colors import to_hex as mpl_to_hex
from matplotlib.colors import to_rgba as mpl_to_rgba


class ColorTypeError(TypeError):
    """Error for invalid color types."""


def _ladybug_to_rgba(color: Color) -> tuple[float, float, float, float]:
    """Convert a ladybug color to an RGBA float tuple."""
    if not isinstance(color, Color):
        raise ColorTypeError("Input color must be a ladybug Color object.")
    return color.r / 255, color.g / 255, color.b / 255, color.a / 255


def _plotly_to_rgba(color: str) -> tuple[float, float, float, float]:
    """Convert a plotly color to an RGBA float tuple."""

    if not isinstance(color, str):
        raise ColorTypeError("Input color must be a plotly color string.")

    if color.startswith("rgba"):
        r, g, b, a = color.split("(")[1].split(")")[0].split(",")
        return tuple((np.array([r, g, b, a]).astype(float) / 255).tolist())

    if color.startswith("rgb"):
        r, g, b = color.split("(")[1].split(")")[0].split(",")
        return tuple((np.array([r, g, b, 255]).astype(float) / 255).tolist())

    raise ColorTypeError("Input color must be a plotly color string.")


def to_rgba(c: Any) -> tuple[float, float, float, float]:
    """Convert a color-like object to an RGBA float tuple.
    This is an augmentation of tehe standard matplotlib function, including
    additional color types.

    Args:
        color (Any):
            The color-like object to convert.

    Returns:
        tuple[float, float, float, float]:
            The color as an RGBA float tuple.
    """

    if isinstance(c, Color):
        return _ladybug_to_rgba(c)

    if isinstance(c, str):
        try:
            return _plotly_to_rgba(c)
        except ColorTypeError:
            return mpl_to_rgba(c)

    if isinstance(c, (list, tuple, np.ndarray)):
        return mpl_to_rgba(c)  # type: ignore

    raise ColorTypeError(
        "Input color must be a ladybug Color object, a plotly color string, or matplotlib color-like object."
    )


def to_hex(c: Any) -> str:
    """Convert a color-like object to a hex string. This is an augmentation of
    the standard matplotlib function, including
    additional color types."""

    if isinstance(c, Color):
        return mpl_to_hex((c.r / 255, c.g / 255, c.b / 255, c.a / 255), keep_alpha=True)

    if isinstance(c, str):
        try:
            return mpl_to_hex(_plotly_to_rgba(c), keep_alpha=True)
        except ColorTypeError:
            return mpl_to_hex(c, keep_alpha=True)

    if isinstance(c, (list, tuple, np.ndarray)):
        return mpl_to_hex(c, keep_alpha=True)  # type: ignore

    raise ColorTypeError(
        "Input color must be a ladybug Color object, a plotly color string, or matplotlib color-like object."
    )


def to_ladybug(c: Any) -> Color:
    """Convert a color-like object into a ladybug color."""
    color = to_rgba(c)
    return Color(
        int(color[0] * 255),
        int(color[1] * 255),
        int(color[2] * 255),
        int(color[3] * 255),
    )


def to_plotly(c: Any) -> str:
    """Convert a color-like object to a plotly color."""
    color = to_rgba(c)
    return f"rgba({color[0] * 255:0.0f},{color[1] * 255:0.0f},{color[2] * 255:0.0f},{color[3] * 255:0.0f})"


def relative_luminance(color: Any):
    """Calculate the relative luminance of a color according to W3C standards

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        float:
            Luminance value between 0 and 1.
    """
    rgb = colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = rgb.dot([0.2126, 0.7152, 0.0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def contrasting_color(color: Any):
    """Calculate the contrasting color for a given color.

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        str:
            String code of the contrasting color.
    """
    return ".15" if relative_luminance(color) > 0.408 else "w"


def lighten_color(
    color: str | tuple, amount: float = 0.5
) -> tuple[float, float, float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    Args:
        color (str):
            A color-like string.
        amount (float):
            The amount of lightening to apply.

    Returns:
        tuple[float]:
            An RGB value.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    c = colorsys.rgb_to_hls(*to_rgba(color)[:-1])
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def average_color(
    colors: Any,
    keep_alpha: bool = False,
    weights: list[float] = None,  # type: ignore
) -> tuple[float, float, float, float]:
    """Return the average color from a list of colors.

    Args:
        colors (Any):
            A list of colors.
        keep_alpha (bool, optional):
            If True, the alpha value of the color is kept. Defaults to False.
        weights (list[float], optional):
            A list of weights for each color. Defaults to None.

    Returns:
        color: str
            The average color in hex format.
    """

    if not isinstance(colors, (list, tuple)):
        raise ValueError("colors must be a list")

    if len(colors) == 1:
        return colors[0]

    return to_rgba(
        np.average([to_rgba(c) for c in colors], axis=0, weights=weights),
    )
