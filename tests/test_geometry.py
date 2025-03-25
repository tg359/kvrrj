import numpy as np
import pytest

from kvrrj.geometry.icosphere import Mesh3D, create_icosphere
from kvrrj.geometry.util import (
    Location,
    angle_clockwise_from_north,
    angle_from_cardinal,
    angle_to_vector,
    cardinality,
    circular_weighted_mean,
    great_circle_distance,
    haversine,
    point_group,
)


def test_great_circle_distance():
    # Test when both locations are the same
    loc1 = Location(latitude=0, longitude=0)
    loc2 = Location(latitude=0, longitude=0)
    assert great_circle_distance(loc1, loc2) == pytest.approx(0.0, rel=1e-7)

    # Test with known values
    loc1 = Location(latitude=52.5200, longitude=13.4050)  # Berlin
    loc2 = Location(latitude=48.8566, longitude=2.3522)  # Paris
    expected_distance = 878.84 * 1000  # Approximate distance in m
    assert great_circle_distance(loc1, loc2) == pytest.approx(
        expected_distance, rel=1e-2
    )

    # Test with locations on opposite sides of the Earth
    loc1 = Location(latitude=0, longitude=0)
    loc2 = Location(latitude=0, longitude=180)
    expected_distance = 40075.0 / 2 * 1000  # Half the Earth's circumference in m
    assert great_circle_distance(loc1, loc2) == pytest.approx(
        expected_distance, rel=1e-2
    )

    # Test with one location at the North Pole and another at the equator
    loc1 = Location(latitude=90, longitude=0)
    loc2 = Location(latitude=0, longitude=0)
    expected_distance = 40075.0 / 4 * 1000  # Quarter of Earth's circumference in m
    assert great_circle_distance(loc1, loc2) == pytest.approx(
        expected_distance, rel=1e-2
    )


def test_haversine():
    # Test when both locations are the same
    loc1 = Location(latitude=0, longitude=0)
    loc2 = Location(latitude=0, longitude=0)
    assert haversine(loc1, loc2) == pytest.approx(0.0, rel=1e-7)

    # Test with known values
    loc1 = Location(latitude=52.5200, longitude=13.4050)  # Berlin
    loc2 = Location(latitude=48.8566, longitude=2.3522)  # Paris
    expected_distance = 878.84 * 1000  # Approximate distance in m
    assert haversine(loc1, loc2) == pytest.approx(expected_distance, rel=1e-2)

    # Test with locations on opposite sides of the Earth
    loc1 = Location(latitude=0, longitude=0)
    loc2 = Location(latitude=0, longitude=180)
    expected_distance = 40075.0 / 2 * 1000  # Half the Earth's circumference in m
    assert haversine(loc1, loc2) == pytest.approx(expected_distance, rel=1e-2)

    # Test with one location at the North Pole and another at the equator
    loc1 = Location(latitude=90, longitude=0)
    loc2 = Location(latitude=0, longitude=0)
    expected_distance = 40075.0 / 4 * 1000  # Quarter of Earth's circumference in m
    assert haversine(loc1, loc2) == pytest.approx(expected_distance, rel=1e-2)


def test_angle_from_cardinal():
    assert angle_from_cardinal("N") == 0
    assert angle_from_cardinal("E") == 90
    assert angle_from_cardinal("S") == 180
    assert angle_from_cardinal("W") == 270
    assert angle_from_cardinal("NE") == 45
    assert angle_from_cardinal("SE") == 135
    assert angle_from_cardinal("SW") == 225
    assert angle_from_cardinal("NW") == 315

    with pytest.raises(ValueError):
        angle_from_cardinal("Z")


def test_angle_from_north():
    assert angle_clockwise_from_north([0.5, 0.5], degrees=True) == 45


def test_cardinality():
    assert cardinality(22.5, directions=16) == "NNE"
    with pytest.raises(ValueError):
        cardinality(370, directions=16)


def test_angle_to_vector():
    angle = 0
    vector = angle_to_vector(angle)
    assert vector[0] == 0
    assert vector[1] == 1

    angle = 45
    vector = angle_to_vector(angle)
    assert vector[0] == pytest.approx(0.707106, rel=0.0001)
    assert vector[1] == pytest.approx(0.707106, rel=0.0001)

    angle = 90
    vector = angle_to_vector(angle)
    assert vector[0] == pytest.approx(1, rel=0.01)
    assert vector[1] == pytest.approx(0, rel=0.01)

    angle = 180
    vector = angle_to_vector(angle)
    assert vector[0] == pytest.approx(0, rel=0.01)
    assert vector[1] == pytest.approx(-1, rel=0.01)

    angle = 270
    vector = angle_to_vector(angle)
    assert vector[0] == pytest.approx(-1, rel=0.01)
    assert vector[1] == pytest.approx(0, rel=0.01)

    angle = 360
    vector = angle_to_vector(angle)
    assert vector[0] == pytest.approx(0, rel=0.01)
    assert vector[1] == pytest.approx(1, rel=0.01)

    angle = 405
    vector = angle_to_vector(angle)
    assert vector[0] == pytest.approx(0.707106, rel=0.0001)
    assert vector[1] == pytest.approx(0.707106, rel=0.0001)


def test_circular_weighted_mean():
    # Test with angles outside of expected range
    angles = [0, 90, 180, 270, 361]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        circular_weighted_mean(angles, weights)

    # Test with number of weights not equal to number of angles
    angles = [0, 90, 180, 270]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        circular_weighted_mean(angles, weights)

    # Test with negative angles
    angles = [-90, 0, 90, 180, 270]
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    with pytest.raises(ValueError):
        circular_weighted_mean(angles, weights)

    # Test with equal weights
    angles = [90, 180, 270]
    weights = [1 / 3, 1 / 3, 1 / 3]
    assert np.isclose(circular_weighted_mean(angles, weights), 180)  # type: ignore

    # Test equally distributed, with equal weighting
    angles = [45, 135, 225, 315]
    with pytest.warns(UserWarning):
        circular_weighted_mean(angles)

    # Test without weights specified
    angles = [0, 90, 180, 270]
    with pytest.warns(UserWarning):
        assert isinstance(circular_weighted_mean(angles), float)

    # Test with different weights
    angles = [90, 180, 270]
    weights = [0.3, 0.3, 0.4]
    assert np.isclose(circular_weighted_mean(angles, weights), 198.43, rtol=0.1)

    # Test about 0
    angles = [355, 5]
    weights = [0.5, 0.5]
    assert np.isclose(circular_weighted_mean(angles, weights), 0)

    # Test opposing
    angles = [0, 180]
    weights = [0.5, 0.5]
    with pytest.warns(UserWarning):
        assert isinstance(circular_weighted_mean(angles, weights), float)


def test_point_group():
    # Test with a single point, returns the same point as a cluster
    points = [[0, 0]]
    threshold = 2
    clusters = point_group(points, threshold)
    assert len(clusters) == 1
    assert clusters == [[[0, 0]]]

    # Test with threshold of 0 (should return the same points as input)
    points = [[0, 0], [1, 1]]
    threshold = 0
    clusters = point_group(points, threshold)
    assert len(clusters) == 1
    assert len(clusters[0]) == len(points)

    # Test with points of different dimensions (should raise ValueError)
    points = [[0, 0], [1, 1, 1]]
    threshold = 2
    with pytest.raises(ValueError):
        point_group(points, threshold)

    # test with threshold < 0 (should raise ValueError)
    points = [[0, 0], [1, 1]]
    threshold = -1
    with pytest.raises(ValueError):
        point_group(points, threshold)

    # test with unsupported data type (should raise ValueError)
    points = ["abc123", [1, 1]]
    threshold = 2
    with pytest.raises(ValueError):
        point_group(points, threshold)

    # Test with unsupported dimensions (should raise ValueError)
    points = [[0, 0, 0, 0], [1, 1, 1, 1]]
    threshold = 2
    with pytest.raises(ValueError):
        point_group(points, threshold)

    # Test with 2D points into 2-clusters
    points = [[0, 0], [1, 1], [10, 10], [11, 11]]
    threshold = 2
    clusters = point_group(points, threshold)
    assert len(clusters) == 2
    assert clusters[0] == [[0, 0], [1, 1]]
    assert clusters[1] == [[10, 10], [11, 11]]

    # Test with 3D points
    points = [[0, 0, 0], [1, 1, 1], [10, 10, 10], [11, 11, 11]]
    threshold = 2
    clusters = point_group(points, threshold)
    assert len(clusters) == 2
    assert sorted(clusters[0]) == sorted([[0, 0, 0], [1, 1, 1]])
    assert sorted(clusters[1]) == sorted([[10, 10, 10], [11, 11, 11]])

    # Test with 3D points into 3 clusters
    points = [
        [0, 0, 0],
        [1, 0.9, 1],
        [5, 5, 5],
        [4.5, 5.5, 6],
        [10, 10.1, 10],
        [11, 11, 11],
    ]
    threshold = 2
    clusters = point_group(points, threshold)
    assert len(clusters) == 3
    assert clusters[1] == [[5, 5, 5], [4.5, 5.5, 6]]


def test_create_icosphere():
    sphere = create_icosphere(resolution=1)
    assert isinstance(sphere, Mesh3D)
    assert len(sphere.vertices) == 42
    assert len(sphere.faces) == 80

    sphere = create_icosphere(resolution=4)
    assert isinstance(sphere, Mesh3D)
    assert len(sphere.vertices) == 2562
    assert len(sphere.faces) == 5120

    with pytest.raises(ValueError):
        create_icosphere(resolution=-1)

    with pytest.raises(ValueError):
        create_icosphere(resolution=0)

    with pytest.raises(ValueError):
        create_icosphere(resolution=2.5)
