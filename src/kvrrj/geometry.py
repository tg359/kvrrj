"""Methods for working with angles, vectors and geometry."""

import warnings
from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.neighbors import KDTree


def cardinality(direction_angle: int | float, directions: int = 16):
    """Returns the cardinal orientation of a given angle, where that angle is related to north at
        0 degrees.
    Args:
        direction_angle (float):
            The angle to north in degrees (+Ve is interpreted as clockwise from north at 0.0
            degrees).
        directions (int):
            The number of cardinal directions into which angles shall be binned (This value should
            be one of 4, 8, 16 or 32, and is centred about "north").
    Returns:
        int:
            The cardinal direction the angle represents.
    """

    if direction_angle > 360 or direction_angle < 0:
        raise ValueError(
            "The angle entered is beyond the normally expected range for an orientation in degrees."
        )

    cardinal_directions = {
        4: ["N", "E", "S", "W"],
        8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        16: [
            "N",
            "NNE",
            "NE",
            "ENE",
            "E",
            "ESE",
            "SE",
            "SSE",
            "S",
            "SSW",
            "SW",
            "WSW",
            "W",
            "WNW",
            "NW",
            "NNW",
        ],
        32: [
            "N",
            "NbE",
            "NNE",
            "NEbN",
            "NE",
            "NEbE",
            "ENE",
            "EbN",
            "E",
            "EbS",
            "ESE",
            "SEbE",
            "SE",
            "SEbS",
            "SSE",
            "SbE",
            "S",
            "SbW",
            "SSW",
            "SWbS",
            "SW",
            "SWbW",
            "WSW",
            "WbS",
            "W",
            "WbN",
            "WNW",
            "NWbW",
            "NW",
            "NWbN",
            "NNW",
            "NbW",
        ],
    }

    if directions not in cardinal_directions:
        raise ValueError(
            f'The input "directions" must be one of {list(cardinal_directions.keys())}.'
        )

    val = int((direction_angle / (360 / directions)) + 0.5)

    arr = cardinal_directions[directions]

    return arr[(val % directions)]


def angle_from_cardinal(cardinal_direction: str) -> float:
    """
    For a given cardinal direction, return the corresponding angle in degrees.

    Args:
        cardinal_direction (str):
            The cardinal direction.
    Returns:
        float:
            The angle associated with the cardinal direction.
    """
    cardinal_directions = [
        "N",
        "NbE",
        "NNE",
        "NEbN",
        "NE",
        "NEbE",
        "ENE",
        "EbN",
        "E",
        "EbS",
        "ESE",
        "SEbE",
        "SE",
        "SEbS",
        "SSE",
        "SbE",
        "S",
        "SbW",
        "SSW",
        "SWbS",
        "SW",
        "SWbW",
        "WSW",
        "WbS",
        "W",
        "WbN",
        "WNW",
        "NWbW",
        "NW",
        "NWbN",
        "NNW",
        "NbW",
    ]
    if cardinal_direction not in cardinal_directions:
        raise ValueError(f"{cardinal_direction} is not a known cardinal_direction.")
    angles = np.arange(0, 360, 11.25)

    lookup = dict(zip(cardinal_directions, angles))

    return lookup[cardinal_direction]


def angle_clockwise_from_north(
    vector: list[float | int], degrees: bool = True
) -> float:
    """For a vector, determine the clockwise angle to north at [0, 1].

    Args:
        vector (list[float | int]):
            A 2D vector object.
        degrees (bool, optional):
            Return the angle in degrees.
            Defaults to True.

    Returns:
        float:
            The angle between vector and north clockwise from 0-359.9.
    """
    if len(vector) != 2:
        raise ValueError("The vector must be 2D.")

    north = [0, 1]
    angle1 = np.arctan2(*north[::-1])  # type: ignore
    angle2 = np.arctan2(*vector[::-1])  # type: ignore
    rad = (angle1 - angle2) % (2 * np.pi)
    if degrees:
        return np.rad2deg(rad)
    return rad


def angle_to_vector(angle_clockwise_from_north: int | float) -> tuple[float, float]:
    """Return the X, Y vector from of an angle from north at 0-degrees.

    Args:
        clockwise_angle_from_north (float):
            The angle from north in degrees clockwise from [0, 360], though
            any number can be input here for angles greater than a full circle.

    Returns:
        list[float]:
            A vector of length 2.
    """

    angle_clockwise_from_north = np.radians(angle_clockwise_from_north)

    return (np.sin(angle_clockwise_from_north), np.cos(angle_clockwise_from_north))


def circular_weighted_mean(
    angles: Sequence[int | float],
    weights: Sequence[int | float] | None = None,
):
    """Get the average angle from a set of weighted angles.

    Args:
        angles (list[float]):
            A collection of equally weighted wind directions, in degrees from North (0).
        weights (list[float]):
            A collection of weights, which must sum to 1.
            Defaults to None which will equally weight all angles.

    Returns:
        float:
            An average wind direction.
    """
    # convert angles to 0-360
    angles = np.where(angles == 360, 0, angles).tolist()

    # handle case where weights are not provided
    if weights is None:
        weights = (np.ones_like(angles) / len(angles)).tolist()

    if len(angles) != len(weights):  # type: ignore
        raise ValueError("weights must be the same size as angles.")

    if any(i < 0 for i in angles) or any(i > 360 for i in angles):
        raise ValueError("Input angles exist outside of expected range (0-360).")

    # checks for opposing or equally spaced angles, with equal weighting
    if len(set(weights)) == 1:  # type: ignore
        _sorted = np.sort(angles)
        if len(set(angles)) == 2:
            a, b = np.meshgrid(_sorted, _sorted)
            if np.any(a - b == 180):
                warnings.warn(
                    "Input angles are opposing, meaning determining the mean is impossible. An attempt will be made to determine the mean, but this will be perpendicular to the opposing angles and not accurate."
                )
        if any(np.diff(_sorted) == 360 / len(angles)):
            warnings.warn(
                "Input angles are equally spaced, meaning determining the mean is impossible. An attempt will be made to determine the mean, but this will not be accurate."
            )
    weight_sum = sum(weights)  # type: ignore
    weights = [weight / weight_sum for weight in weights]  # type: ignore

    x = y = 0.0
    for angle, weight in zip(angles, weights):
        x += np.cos(np.radians(angle)) * weight
        y += np.sin(np.radians(angle)) * weight

    mean = np.degrees(np.arctan2(y, x))

    if mean < 0:
        mean = 360 + mean

    if mean in (360.0, -0.0):
        mean = 0.0

    return np.round(mean, 5)


def point_group(points: list[list[float]], threshold: float) -> list[list[float]]:
    """Cluster 2D or 3D points based on proximity.

    Args:
        points (list[list[float]]):
            A list of 2D or 3D points.
        threshold (float):
            The maximum distance between points to be considered neighbors.

    Returns:
        clusters: list[list[float]]
            The points in each cluster.
    """

    # ensure points are in the correct format (list[list[number]])
    if not all(isinstance(point, (list, tuple)) for point in points):
        raise ValueError("All points must be a list or tuple.")
    # ensure each point is numeric
    if not all(isinstance(coord, (int, float)) for point in points for coord in point):
        raise ValueError("All points must be numeric.")

    if threshold < 0:
        raise ValueError("The threshold must be greater than 0.")

    # Ensure points are in the correct format
    point_dim = len(points[0])
    if not all(len(point) == point_dim for point in points):
        raise ValueError("All points must have the same dimensionality (2D or 3D).")

    if point_dim not in (2, 3):
        raise ValueError("Only 2D or 3D points are supported.")

    if len(points) == 1:
        return [points]

    if threshold == 0:
        # return original points
        return [points]

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))

        def find(self, i):
            if self.parent[i] != i:
                self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j):
            root_i = self.find(i)
            root_j = self.find(j)
            if root_i != root_j:
                self.parent[root_i] = root_j

    tree = KDTree(points)

    # Initialize Union-Find
    uf = UnionFind(len(points))

    # Find neighboring points within radius and union them
    for i, point in enumerate(points):
        neighbor_indices = tree.query_radius(X=[point], r=threshold)[0]
        for neighbor_index in neighbor_indices:
            uf.union(i, neighbor_index)

    # Collect fused points and assign labels
    label_groups = defaultdict(list)

    for i in range(len(points)):
        root = uf.find(i)
        label_groups[root].append(i)

    clusters = []
    for _, points_indices in label_groups.items():
        clusters.append([points[i] for i in points_indices])

    return clusters
