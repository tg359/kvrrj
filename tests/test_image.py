import base64
import io

import pytest
from matplotlib.figure import Figure
from PIL import Image

from kvrrj.viz.image import (
    base64_to_image_file,
    extract_color_coordinates,
    figure_to_base64,
    image_file_to_base64,
    tile_images,
)


def test_base64_to_image_file(tmp_path):
    # Create a base64 string for a red image
    img = Image.new("RGBA", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Convert base64 string to image file
    image_path = tmp_path / "test_image.png"
    result_path = base64_to_image_file(base64_string, image_path)

    # Verify the file exists and is a valid image
    assert result_path == image_path
    assert image_path.exists()
    loaded_img = Image.open(image_path)
    assert loaded_img.size == (100, 100)


def test_image_file_to_base64(tmp_path):
    # Create a temporary image file
    img = Image.new("RGBA", (100, 100), color="blue")
    image_path = tmp_path / "test_image.png"
    img.save(image_path)

    # Convert image file to base64 string
    base64_string = image_file_to_base64(image_path)

    # Verify the base64 string can be decoded back to the original image
    decoded_data = base64.b64decode(base64_string)
    buffer = io.BytesIO(decoded_data)
    loaded_img = Image.open(buffer)
    assert loaded_img.size == (100, 100)


def test_figure_to_base64():
    # Create a simple matplotlib figure
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])

    # Convert figure to base64 string
    base64_string = figure_to_base64(fig)

    # Verify the base64 string can be decoded back to an image
    decoded_data = base64.b64decode(base64_string)
    buffer = io.BytesIO(decoded_data)
    loaded_img = Image.open(buffer)
    assert loaded_img.mode == "RGBA"


def test_tile_images():
    # Create mock images
    img1 = Image.new("RGBA", (100, 100), color="red")
    img2 = Image.new("RGBA", (100, 100), color="green")
    img3 = Image.new("RGBA", (100, 100), color="blue")
    img4 = Image.new("RGBA", (100, 100), color="yellow")

    # Test valid grid
    result = tile_images([img1, img2, img3, img4], rows=2, cols=2)
    assert result.size == (200, 200)  # 2x2 grid of 100x100 images

    # Test mismatched grid size
    with pytest.raises(ValueError, match="does not equal"):
        tile_images([img1, img2, img3], rows=2, cols=2)

    # Test non-PIL image input
    with pytest.raises(TypeError, match="must be PIL Image"):
        tile_images([img1, "not_an_image", img3, img4], rows=2, cols=2)

    # Test mismatched image dimensions
    img5 = Image.new("RGBA", (50, 50), color="purple")
    with pytest.raises(ValueError, match="must have the same dimensions"):
        tile_images([img1, img2, img3, img5], rows=2, cols=2)


def test_extract_color_coordinates():
    # Create a mock image (10x10)
    img = Image.new("RGBA", (10, 10), color="white")
    # add a red pixel
    img.putpixel((5, 3), (255, 0, 0, 255))
    # add a blue pixel
    img.putpixel((8, 2), (0, 0, 255, 255))
    # add a green pixel
    img.putpixel((1, 1), (0, 255, 0, 255))
    # add another green pixel
    img.putpixel((9, 9), (0, 255, 0, 255))
    # add a green-ish pixel
    img.putpixel((7, 7), (0, 245, 0, 255))

    # Test with exact color match
    coords = extract_color_coordinates(
        img, (1, 0, 0, 1), proximity_threshold=0, color_threshold=0
    )
    assert coords == [[5, 7]]

    # Test with no matching color
    coords = extract_color_coordinates(
        img, (1, 1, 0, 1), proximity_threshold=0, color_threshold=0
    )
    assert coords == []

    # Test with proximity threshold (grouping points by distance)
    coords = extract_color_coordinates(
        img, (0, 1, 0, 1), proximity_threshold=4.5, color_threshold=0
    )
    assert len(coords) == 2  # Points should be grouped

    # Test with color threshold (similar colors)
    coords = extract_color_coordinates(
        img, (0, 1, 0, 1), proximity_threshold=0, color_threshold=10
    )
    assert len(coords[0]) == 2  # All similar greens should be included

    # Test with proximity threshold (grouping points by distance), and color threshold
    coords = extract_color_coordinates(
        img, (0, 1, 0, 1), proximity_threshold=3, color_threshold=25
    )
    assert len(coords[0]) == 2  # Points should be grouped
