import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from kvrrj.geometry.util import point_group
from kvrrj.viz.color import to_rgba


def extract_color_coordinates(
    img: Image.Image,
    c: Any,
    proximity_threshold: float = 0,
    color_threshold: int = 2,
) -> list[list[float]]:
    """Create a file containing pt-pixel location coordinates based on color keys.

    Args:
        img (Image.Image):
            A PIL Image object.
        c (Any):
            A color-like object.
        proximity_threshold (float):
            The maximum distance (in pixels) between points to be considered neighbors.
            Defaults to 0, which doesn't group points.
        color_threshold (int, optional):
            The threshold in terms of RGB distance (0-255) for color matching.
            Defaults to 2.

    Returns:
        coordinates (list[list[float]]):
            A list of points, representing the pixel locations of the target color.
            The x, y coordinates are in the format of the bottom left corner of the image.
    """

    # convert color to RGB[0-255] format
    target_color = (np.array(to_rgba(c)) * 255).tolist()

    # get the image pixels
    coords = []
    for x in range(img.width):
        for y in range(img.height):
            color = img.getpixel((x, y))
            if all(np.isclose(color, target_color, atol=color_threshold)):
                coords.append((x, y))

    # if no points found, return empty list
    if not coords:
        return []

    # cluster the points and group by proximity
    groups = point_group(coords, threshold=proximity_threshold)
    centroids = [np.mean(grp, axis=0).tolist() for grp in groups]

    # remove any empty groups
    centroids = [px for px in centroids if px]

    # convert coords into a more typical x, y, starting from bottom left of the image, cos that's easier to understand!
    return [[px[0], img.height - px[1]] for px in centroids]


def base64_to_image_file(base64_string: str, image_path: Path) -> Path:
    """Convert a base64 encoded image into a file on disk.

    Arguments:
        base64_string (str):
            A base64 string encoding of an image file.
        image_path (Path):
            The location where the image should be stored.

    Returns:
        Path:
            The path to the image file.
    """

    # remove html pre-amble, if necessary
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(";")[-1]

    with open(Path(image_path), "wb") as fp:
        fp.write(base64.decodebytes(base64_string.encode("utf-8")))  # type: ignore

    return image_path


def image_file_to_base64(image_path: Path, html: bool = False) -> str:
    """Load an image file from disk and convert to base64 string.

    Arguments:
        image_path (Path):
            The file path for the image to be converted.
        html (bool, optional):
            Set to True to include the HTML preamble for a base64 encoded image. Default is False.

    Returns:
        str:
            A base64 string encoding of the input image file.
    """

    # convert path string to Path object
    image_path = Path(image_path).absolute()

    # ensure format is supported
    supported_formats = [".png", ".jpg", ".jpeg"]
    if image_path.suffix not in supported_formats:
        raise ValueError(
            f"'{image_path.suffix}' format not supported. Use one of {supported_formats}"
        )

    # load image and convert to base64 string
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")

    if html:
        content_type = f"data:image/{image_path.suffix.replace('.', '')}"
        content_encoding = "utf-8"
        return f"{content_type};charset={content_encoding};base64,{base64_string}"

    return base64_string


def figure_to_base64(
    figure: Figure, html: bool = False, transparent: bool = True
) -> str:
    """Convert a matplotlib figure object into a base64 string.

    Arguments:
        figure (Figure):
            A matplotlib figure object.
        html (bool, optional):
            Set to True to include the HTML preamble for a base64 encoded image. Default is False.

    Returns:
        str:
            A base64 string encoding of the input figure object.
    """

    buffer = io.BytesIO()
    figure.savefig(buffer, transparent=transparent)
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.read()).decode("utf-8")

    if html:
        content_type = "data:image/png"
        content_encoding = "utf-8"
        return f"{content_type};charset={content_encoding};base64,{base64_string}"

    return base64_string


def figure_to_image(fig: Figure) -> Image.Image:
    """Convert a matplotlib Figure object into a PIL Image.

    Args:
        fig (Figure):
            A matplotlib Figure object.

    Returns:
        Image:
            A PIL Image.
    """

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # type: ignore
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)

    return Image.fromarray(buf)


def tile_images(images: list[Image.Image], rows: int, cols: int) -> Image.Image:
    """Tile a set of images into a grid.

    Args:
        images (list[Image.Image]):
            A list of images to tile.
        rows (int):
            The number of rows in the grid.
        cols (int):
            The number of columns in the grid.

    Returns:
        Image.Image:
            A PIL image of the tiled images.
    """

    # check that the number of images matches the grid size
    if len(images) != rows * cols:
        raise ValueError(
            f"The number of images given ({len(images)}) does not equal ({rows}*{cols})"
        )

    # check that each image is an Image object
    for img in images:
        if not isinstance(img, Image.Image):
            raise TypeError("All images must be PIL Image")

    # ensure each image has the same dimensions
    w, h = images[0].size
    for im in images:
        if im.size != (w, h):
            raise ValueError("All images must have the same dimensions")

    # create new image
    grid = Image.new("RGBA", size=(cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
        img.close()

    return grid
