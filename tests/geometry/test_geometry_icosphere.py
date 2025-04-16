import pytest

from kvrrj.geometry.icosphere import Mesh3D, icosphere


def test_icosphere():
    sphere = icosphere(resolution=1)
    assert isinstance(sphere, Mesh3D)
    assert len(sphere.vertices) == 42
    assert len(sphere.faces) == 80

    sphere = icosphere(resolution=4)
    assert isinstance(sphere, Mesh3D)
    assert len(sphere.vertices) == 2562
    assert len(sphere.faces) == 5120

    with pytest.raises(ValueError):
        icosphere(resolution=-1)

    with pytest.raises(ValueError):
        icosphere(resolution=0)

    with pytest.raises(ValueError):
        icosphere(resolution=2.5)
