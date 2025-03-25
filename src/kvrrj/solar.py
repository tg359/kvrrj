"""Methods for handling sun location/position data."""

import pickle
from datetime import date, datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from honeybee.config import folders as hb_folders
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.epw import EPW
from ladybug.sunpath import Location, Sunpath
from ladybug_geometry.geometry2d import Vector2D
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.visualize.radrose import RadiationRose
from tqdm import tqdm

from .geometry.util import angle_clockwise_from_north
from .ladybug.analysis_period import (
    analysis_period_to_datetimes,
    lbdatetime_from_datetime,
)

# TODO - make this an object, similar to the Wind class
# class Solar:
#     """A class for handling solar data."""


class _SunriseType(Enum):
    """An enumeration of different types of daylight."""

    ACTUAL = auto()
    APPARENT = auto()
    CIVIL = auto()
    NAUTICAL = auto()
    ASTRONOMICAL = auto()

    @property
    def depression_angle(self) -> float:
        """Get the depression angle (in degrees) for the daylight type."""
        return {
            _SunriseType.ACTUAL.name: 0.5334,
            _SunriseType.APPARENT.name: 0.833,
            _SunriseType.CIVIL.name: 6,
            _SunriseType.NAUTICAL.name: 12,
            _SunriseType.ASTRONOMICAL.name: 18,
        }[self.name]


def sunrise_sunset(dates: list[date], location: Location) -> pd.DataFrame:
    """Get sunrise times for a list of dates at a location.

    Args:
        dates (list[date]):
            A list of dates for which to get sunrise times.
        location (Location):
            A ladybug Location object.

    Returns:
        pd.DataFrame:
            A DataFrame with sunrise times for each date.
    """

    # check inputs
    if not isinstance(dates, list):
        raise ValueError("dates must be a list of date objects.")
    if not all(isinstance(d, date) for d in dates):
        raise ValueError("All items in dates must be date objects.")
    if not isinstance(location, Location):
        raise ValueError("Location must be a ladybug Location object.")

    # get the sunpath
    sunpath = Sunpath.from_location(location)

    # sort dates and remove duplicates
    dates = list(set(dates))
    dates = sorted(dates)

    # get sunrise times
    kk = {}
    for dt in dates:
        kk[dt] = {}
        for s_type in _SunriseType:
            d = sunpath.calculate_sunrise_sunset_from_datetime(
                datetime=lbdatetime_from_datetime(datetime(dt.year, dt.month, dt.day)),
                depression=s_type.depression_angle,
            )
            for tod in ["sunrise", "sunset"]:
                kk[dt][f"{s_type.name.lower()} {tod}"] = d[tod]
        kk[dt]["noon"] = d["noon"]  # type: ignore

    return pd.DataFrame(kk).T


def sunrise_sunset_from_epw(epw: EPW, daily: bool = True) -> pd.DataFrame:
    """Get sunrise times for an EPW file.

    Args:
        epw (EPW):
            An EPW object.
        daily (bool):
            If True, return daily sunrise times. If False, return hourly sunrise times (repeated for each hour of the day).

    Returns:
        pd.DataFrame:
            A DataFrame with sunrise times for each date in the EPW file.
    """

    # check inputs
    if not isinstance(epw, EPW):
        raise ValueError("EPW must be an EPW object.")

    # get dates
    dts = [
        i.date()
        for i in analysis_period_to_datetimes(
            epw.dry_bulb_temperature.header.analysis_period
        )
    ]
    if daily:
        dts = list(set(dts))

    # get sunrise times
    return sunrise_sunset(
        dates=dts,
        location=epw.location,
    )


class IrradianceType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()

    def to_string(self) -> str:
        """Get the string representation of the IrradianceType."""
        return f"{self.name.title()} irradiance"


def azimuthal_radiation(
    epw: EPW,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    tilt_angle: float = 0,
    directions: int = 36,
    ground_reflectance: float = 0.2,
) -> pd.Series:
    """Calculate the azimuthal radiation data for a given EPW file.

    Args:
        epw (EPW):
            The EPW representing the weather data/location to be visualised.
        analysis_period (AnalysisPeriod, optional):
            The analysis period over which radiation shall be summarised.
            Defaults to AnalysisPeriod().
        tilt_angle (float, optional):
            The tilt (from 0 at horizon, to 90 facing the sky) to assess.
            Defaults to 0.
        directions (int, optional):
            The number of directions to bin data into.
            Defaults to 36.
        ground_reflectance (float, optional):
            The reflectance of the ground.
            Defaults to 0.2.

    Returns:
        pd.Series:
            A Series with azimuthal radiation data, in degrees.
    """

    # create sky conditions
    smx = SkyMatrix.from_epw(
        epw_file=epw.file_path,
        high_density=True,
        hoys=analysis_period.hoys,
        ground_reflectance=ground_reflectance,
    )
    rr = RadiationRose(
        sky_matrix=smx,
        direction_count=directions,
        tilt_angle=tilt_angle,  # type: ignore
    )

    # get properties to plot
    angles = [
        angle_clockwise_from_north(j, degrees=True)
        for j in [Vector2D(*i[:2]) for i in rr.direction_vectors]
    ]

    total_values = getattr(rr, "total_values")
    direct_values = getattr(rr, "direct_values")
    diffuse_values = getattr(rr, "diffuse_values")

    return pd.concat(
        [
            pd.Series(
                total_values, index=angles, name=IrradianceType.TOTAL.to_string()
            ),
            pd.Series(
                direct_values, index=angles, name=IrradianceType.DIRECT.to_string()
            ),
            pd.Series(
                diffuse_values, index=angles, name=IrradianceType.DIFFUSE.to_string()
            ),
        ],
        axis=1,
    )


def tilt_orientation_factor(
    epw: EPW,
    analysis_period: AnalysisPeriod = AnalysisPeriod(),
    directions: int = 36,
    tilts: int = 9,
) -> pd.DataFrame:
    """Create a tilt-orientation factor matrix.

    Args:
        epw_file (Path):
            The EPW file representing the weather data/location to be calculated.
        analysis_period (AnalysisPeriod, optional):
            The analysis period over which radiation shall be summarised.
            Defaults to AnalysisPeriod().
        directions (int, optional):
            The number of directions to bin data into.
            Defaults to 36.
        tilts (int, optional):
            The number of tilts to calculate.
            Defaults to 9.

    Returns:
        pd.DataFrame:
            A DataFrame with tilt-orientation factors.
    """

    # create dir for cached results
    _dir = Path(hb_folders.default_simulation_folder) / "_lbt_tk_solar"
    _dir.mkdir(exist_ok=True, parents=True)
    ndir = directions

    # create sky matrix
    smx = SkyMatrix.from_epw(
        epw_file=epw.file_path, high_density=True, hoys=analysis_period.hoys
    )

    # create roses per tilt angle
    _directions = np.linspace(0, 360, directions + 1)[:-1].tolist()
    _tilts = np.linspace(0, 90, tilts)[:-1].tolist() + [89.999]
    rrs: list[RadiationRose] = []
    for ta in tqdm(_tilts):
        sp = _dir / f"{Path(epw.file_path).stem}_{ndir}_{ta:0.4f}.pickle"
        if sp.exists():
            rr = pickle.load(open(sp, "rb"))
        else:
            rr = RadiationRose(
                sky_matrix=smx, direction_count=directions, tilt_angle=ta
            )
            pickle.dump(rr, open(sp, "wb"))
        rrs.append(rr)
    _directions.append(360)

    # create matrices of values from results
    total_values = np.array([i.total_values for i in rrs]).T
    total_values = np.concatenate([total_values, total_values[[0], :]], axis=0)

    direct_values = np.array([i.direct_values for i in rrs]).T
    direct_values = np.concatenate([direct_values, direct_values[[0], :]], axis=0)

    diffuse_values = np.array([i.diffuse_values for i in rrs]).T
    diffuse_values = np.concatenate([diffuse_values, diffuse_values[[0], :]], axis=0)

    # create dataframe
    df = pd.concat(
        [
            pd.DataFrame(data=total_values, index=_directions, columns=_tilts),
            pd.DataFrame(data=direct_values, index=_directions, columns=_tilts),
            pd.DataFrame(data=diffuse_values, index=_directions, columns=_tilts),
        ],
        axis=1,
        keys=[
            IrradianceType.TOTAL.to_string(),
            IrradianceType.DIRECT.to_string(),
            IrradianceType.DIFFUSE.to_string(),
        ],
    )
    return df
