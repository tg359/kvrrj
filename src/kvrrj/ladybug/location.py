"""Methods for manipulating Ladybug Location objects."""

from typing import Sequence

import numpy as np
from ladybug.location import Location


def location_to_string(location: Location) -> str:
    """Return a simple representation of the Location object.

    Args:
        location (Location):
            A Ladybug location object.

    Returns:
        str:
            A simple string representation of the Location object.
    """
    return f"{location.country.strip()} - {location.city.strip()}"


def average_location(
    locations: list[Location], weights: Sequence[int | float] | None = None
) -> Location:
    """Create an average location from a list of locations.
    This will use weighting if provided to adjust latitude/longitude values.

    Args:
        locations (list[Location]):
            A set of ladybug Location objects.
        weights (list[float], optional):
            A list of weights for each location.
            Defaults to None which evenly weights each location.

    Returns:
        Location: A synthetic location that is the average of all locations.
    """

    # check inputs

    if not isinstance(locations, (list, tuple)):
        raise ValueError("Locations must be a list or tuple of Location objects.")

    if len(locations) == 1:
        return locations[0]

    if len(locations) == 0:
        raise ValueError("No locations provided.")

    if weights is None:
        weights = [1] * len(locations)

    if len(weights) != len(locations):
        raise ValueError("The number of weights must match the number of locations.")

    if sum(weights) == 0:
        raise ValueError("The sum of weights cannot be zero.")

    # calculate average latitude, longitude, and elevation
    lat = (
        np.average(
            np.array([loc.latitude for loc in locations]) + 1000, weights=weights
        )
        - 1000
    )
    lon = (
        np.average(
            np.array([loc.longitude for loc in locations]) + 1000, weights=weights
        )
        - 1000
    )
    elv = np.average(np.array([loc.elevation for loc in locations]), weights=weights)

    # create the location descriptors
    state = "|".join(
        [
            loc.state if loc.state not in ["", "-", None] else "NoState"
            for loc in locations
        ]
    )
    city = "|".join(
        [loc.city if loc.city not in ["", "-", None] else "NoCity" for loc in locations]
    )
    country = "|".join(
        [
            str(loc.country) if loc.country not in ["", "-", None] else "NoCountry"
            for loc in locations
        ]
    )
    station_id = "|".join(
        [
            str(loc.station_id)
            if loc.station_id not in ["", "-", None]
            else "NoStationId"
            for loc in locations
        ]
    )
    source = "|".join(
        [
            str(loc.source) if loc.source not in ["", "-", None] else "NoSource"
            for loc in locations
        ]
    )
    return Location(
        city=f"Synthetic ({city})",
        state=f"Synthetic ({state})",
        country=f"Synthetic ({country})",
        latitude=lat,  # type: ignore
        longitude=lon,  # type: ignore
        elevation=elv,  # type: ignore
        station_id=f"Synthetic ({station_id})",
        source=f"Synthetic ({source})",
    )
