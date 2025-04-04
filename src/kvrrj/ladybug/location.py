"""Methods for manipulating Ladybug Location objects."""

import warnings
from datetime import datetime, timedelta, tzinfo
from typing import Sequence

import numpy as np
import pytz
import timezonefinder
from ladybug.location import Location
from pvlib.location import Location as pvlib_location
from pvlib.location import lookup_altitude
from pytz import timezone as pytz_tz

from kvrrj.geometry.util import great_circle_distance


def get_elevation_from_lat_lon(lat: float, lon: float) -> float:
    """Get the elevation from a latitude and longitude.

    Args:
        lat (float):
            Latitude in degrees.
        lon (float):
            Longitude in degrees.

    Returns:
        float: Elevation in meters.
    """
    elv = lookup_altitude(latitude=lat, longitude=lon)
    return elv


def pvlib_location_to_lb_location(location: pvlib_location) -> Location:
    """Convert a pvlib location object to a ladybug location object."""

    # dtype check
    if not isinstance(location, pvlib_location):
        raise ValueError("location must be a pvlib Location object.")

    # convert to ladybug location
    loc = Location(
        latitude=location.latitude,
        longitude=location.longitude,
        elevation=location.altitude,
        time_zone=get_tz_hours_from_tzinfo(location.tz),
        source=location.name,
    )
    # check time zone is valid for location
    if not _is_location_time_zone_valid_for_location(loc):
        raise ValueError(
            f"The time zone of the location ({loc.time_zone}) does not match the time zone of the lat/lon ({loc.latitude}, {loc.longitude})."
        )
    return loc


def _lb_location_to_pvlib_location(location: Location) -> pvlib_location:
    """Convert a ladybug Location to a pvlib Location."""

    # dtype check
    if not isinstance(location, Location):
        raise ValueError("location must be a ladybug Location object.")

    # check time zone is valid for location
    if not _is_location_time_zone_valid_for_location(location):
        raise ValueError(
            f"The time zone of the location ({location.time_zone}) does not match the time zone of the lat/lon ({location.latitude}, {location.longitude})."
        )
    return pvlib_location(
        latitude=location.latitude,
        longitude=location.longitude,
        tz=location.time_zone,
        altitude=location.elevation,
        name=location.source,
    )


def get_timezone_str_from_lat_lon(lat: float, lon: float) -> str:
    tf = timezonefinder.TimezoneFinder()
    timezone_str = tf.certain_timezone_at(lat=lat, lng=lon)
    return timezone_str


def get_timezone_str_from_location(location: Location) -> str:
    timezone_str = get_timezone_str_from_lat_lon(location.latitude, location.longitude)
    return timezone_str


def get_utc_offset_from_location(location: Location) -> timedelta:
    return pytz_tz(get_timezone_str_from_location(location)).utcoffset(
        datetime(
            2017,
            1,
            1,
            0,
            0,
            0,
        ),
        # is_dst=True,
    )


def get_tzinfo_from_location(location: Location) -> tzinfo:
    """Get the tzinfo from a ladybug location object."""
    return pytz.FixedOffset(location.time_zone * 60)


def get_tz_hours_from_tzinfo(tz: tzinfo | str) -> float:
    """Get the timezone hours from a tzinfo object."""
    if not isinstance(tz, (tzinfo, str)):
        raise ValueError("tz must be a tzinfo object.")
    if isinstance(tz, str):
        tz = pytz_tz(tz)
    return tz.utcoffset(datetime(2017, 1, 1)).seconds / 3600


def _is_datetime_location_aligned(dt: datetime, location: Location) -> bool:
    """Check if a datetimes are aligned with the location timezone."""
    if dt.tzinfo is None:
        return False
    return dt.tzinfo.utcoffset(dt).seconds / 3600 == location.time_zone


def _all_timezones_same(tzs: list[tzinfo]) -> bool:
    """Check if all locations have the same timezone."""
    return len(set(tzs)) == 1


def _is_location_time_zone_valid_for_location(location: Location) -> bool:
    time_zone_hours = get_utc_offset_from_location(location).seconds / 3600
    if time_zone_hours != location.time_zone:
        return False
    return True


def _is_elevation_valid_for_location(
    location: Location, tolerance: float = 100
) -> bool:
    """Check if the elevation is valid for the location."""
    elevation = lookup_altitude(
        latitude=location.latitude, longitude=location.longitude
    )
    if np.isclose(elevation, location.elevation, rtol=tolerance):
        return True
    return False


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

    # raise a warning is the locations are quite far away
    distances = []
    for loc1 in locations:
        for loc2 in locations:
            distances.append(great_circle_distance(loc1, loc2))
    if max(distances) > 10000:
        warnings.warn(
            f"The maximum distance between the locations passed is {max(distances)} km. That's quite far!"
        )

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
        city=city,
        state=state,
        country=country,
        latitude=lat,
        longitude=lon,
        elevation=elv,
        station_id=station_id,
        source=source,
    )
