"""Methods for handling solar data. This module relies heavily on numpy, pandas, and ladybug."""

import concurrent
import inspect
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ladybug.datatype.energyflux import (
    DiffuseHorizontalIrradiance,
    DirectNormalIrradiance,
    GlobalHorizontalIrradiance,
)
from ladybug.datatype.temperature import Temperature
from ladybug.epw import EPW, Header, HourlyContinuousCollection
from ladybug.sunpath import Location
from ladybug.wea import Wea
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.visualize.radrose import RadiationRose
from tqdm import tqdm

from .ladybug.analysis_period import (
    AnalysisPeriod,
    analysis_period_from_datetimes,
    analysis_period_from_pd_freq,
)
from .logging import CONSOLE_LOGGER  # noqa: F401

_LOCATION_ARGS = inspect.signature(Location).parameters.keys()


@dataclass
class Solar:
    """An object containing a full year of solar data.

    Args:
        location (Location):
            A ladybug Location object.
        analysis_period (AnalysisPeriod):
            A ladybug AnalysisPeriod object. This should cover a full year,
            with timesteps aligned with the solar data.
        direct_normal_irradiance (list[float]):
            An iterable of direct normal irradiance values.
        diffuse_horizontal_irradiance (list[float]):
            An iterable of diffuse horizontal irradiance values.
        global_horizontal_irradiance (list[float]):
            An iterable of global horizontal irradiance values.

    """

    location: Location = field(repr=True)
    analysis_period: AnalysisPeriod = field(repr=True)
    direct_normal_irradiance: list[float] | tuple[float] | npt.NDArray[np.float64] = (
        field(repr=False)
    )
    diffuse_horizontal_irradiance: (
        list[float] | tuple[float] | npt.NDArray[np.float64]
    ) = field(repr=False)
    global_horizontal_irradiance: (
        list[float] | tuple[float] | npt.NDArray[np.float64]
    ) = field(repr=False)

    def __post_init__(self):
        # ensure analysis_period covers a full year
        if (self.analysis_period.is_leap_year and len(self.analysis_period) < 8784) or (
            not self.analysis_period.is_leap_year and len(self.analysis_period) < 8760
        ):
            raise ValueError("The analysis_period must cover a full year.")

        arrays = [
            "direct_normal_irradiance",
            "diffuse_horizontal_irradiance",
            "global_horizontal_irradiance",
        ]

        # convert inputs to np.ndarrays
        for var in arrays:
            setattr(self, var, np.array(getattr(self, var)))

        # make sure each array is 1-dimensional
        for var in arrays:
            t = getattr(self, var)
            if t is not None:
                if t.ndim != 1:
                    raise ValueError(f"{var} must be 1-dimensional.")

        # ensure each array is same length as analysis_period
        for var in arrays:
            if len(self.analysis_period) != len(getattr(self, var)):
                raise ValueError(
                    f"The {var} array is not the same length as the analysis_period."
                )

        # ensure datatypes are correct
        if not isinstance(self.location, Location):
            raise ValueError("location must be a ladybug Location object.")
        if not isinstance(self.analysis_period, AnalysisPeriod):
            raise ValueError("analysis_period must be a ladybug AnalysisPeriod object.")
        for var in arrays:
            if not np.issubdtype(getattr(self, var).dtype, np.number):
                raise ValueError(f"{var} must be a numeric array.")

        # ensure there are no negative values, clipping to 0 where this occurs
        for var in arrays:
            t = getattr(self, var)
            if min(t) < 0:
                warnings.warn("Negative values found in {var} are being clipped to 0.")
                setattr(self, var, np.clip(t, a_min=0, a_max=None))

    # region: STATIC METHODS

    @staticmethod
    def _check_matching_locations(solar1: "Solar", solar2: "Solar") -> None:
        """Raise an error if two objects do not have the same location."""
        if solar1.location != solar2.location:
            raise ValueError("Locations must be the same.")

    @staticmethod
    def _check_matching_analysis_periods(solar1: "Solar", solar2: "Solar") -> None:
        """Raise an error if two objects do not have the same analysis period."""
        if solar1.analysis_period != solar2.analysis_period:
            raise ValueError("Analysis periods must be the same.")

    @staticmethod
    def _is_aligned(solar: "Solar", object: Any) -> None:
        if isinstance(object, Solar):
            Solar._check_matching_analysis_periods(solar, object)
            Solar._check_matching_locations(solar, object)

        if isinstance(object, (list, tuple)):
            if len(object) != len(solar):
                raise ValueError(
                    "The length of the object must match the length of the Solar object."
                )
            if not all(isinstance(i, (float, int)) for i in object):
                raise ValueError("All items in the object must be numeric.")

        if isinstance(object, np.ndarray):
            if not np.issubdtype(object.dtype, np.number):
                raise ValueError("The other object must be numeric.")

    @staticmethod
    def _is_aligned_array(solar: "Solar", object: Any) -> None:
        if isinstance(object, Solar):
            raise ValueError(
                "This operation is not supported between two Solar objects."
            )

        if isinstance(object, (list, tuple)):
            if len(object) != len(solar):
                raise ValueError(
                    "The length of the object must match the length of the Solar object."
                )
            if not all(isinstance(i, (float, int)) for i in object):
                raise ValueError("All items in the object must be numeric.")

        if isinstance(object, np.ndarray):
            if not np.issubdtype(object.dtype, np.number):
                raise ValueError("The other object must be numeric.")

    # endregion: STATIC METHODS

    # region: DUNDER METHODS

    def __len__(self) -> int:
        return len(self.global_horizontal_irradiance)

    def __add__(self, other: "Solar") -> "Solar":
        """Add this and another Solar object, and return a new Solar object."""
        self._is_aligned(self, other)
        return Solar(
            location=self.location,
            analysis_period=self.analysis_period,
            direct_normal_irradiance=(
                self.direct_normal_irradiance + other.direct_normal_irradiance
            ),
            diffuse_horizontal_irradiance=(
                self.diffuse_horizontal_irradiance + other.diffuse_horizontal_irradiance
            ),
            global_horizontal_irradiance=(
                self.global_horizontal_irradiance + other.global_horizontal_irradiance
            ),
        )

    def __radd__(self, other: "Solar") -> "Solar":
        """Add a set of solar objects, and return a new Solar object."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other: "Solar") -> "Solar":
        """Subtract another Solar object from this one, and return a new Solar object."""
        self._is_aligned(self, other)
        return Solar(
            location=self.location,
            analysis_period=self.analysis_period,
            direct_normal_irradiance=(
                self.direct_normal_irradiance - other.direct_normal_irradiance
            ),
            diffuse_horizontal_irradiance=(
                self.diffuse_horizontal_irradiance - other.diffuse_horizontal_irradiance
            ),
            global_horizontal_irradiance=(
                self.global_horizontal_irradiance - other.global_horizontal_irradiance
            ),
        )

    def __mul__(self, other: int | float) -> "Solar":
        """Multiply this Solar object by a number, and return a new Solar object."""
        self._is_aligned_array(self, other)

        if isinstance(other, (float, int)):
            other = np.array([other] * len(self))

        return Solar(
            location=self.location,
            analysis_period=self.analysis_period,
            direct_normal_irradiance=self.direct_normal_irradiance * other,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_irradiance * other,
            global_horizontal_irradiance=self.global_horizontal_irradiance * other,
        )

    def __truediv__(self, other: int | float) -> "Solar":
        """Divide this Solar object by a number or other Solar object, and return a new Solar object."""
        self._is_aligned_array(self, other)

        if isinstance(other, (float, int)):
            other = np.array([other] * len(self))

        return Solar(
            location=self.location,
            analysis_period=self.analysis_period,
            direct_normal_irradiance=self.direct_normal_irradiance / other,
            diffuse_horizontal_irradiance=self.diffuse_horizontal_irradiance / other,
            global_horizontal_irradiance=self.global_horizontal_irradiance / other,
        )

    # endregion: DUNDER METHODS

    # region: PROPERTIES

    @property
    def lb_datetimes(self) -> tuple[datetime]:
        """Return the ladybug datetimes of the object."""
        return self.analysis_period.datetimes

    @property
    def utc_offset(self) -> int:
        """Return the UTC offset of the object."""
        return self.location.time_zone

    @property
    def datetimes(self) -> list[datetime]:
        """Return the datetimes of the object."""
        return [
            datetime(
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.minute,
                dt.second,
                tzinfo=timezone(timedelta(hours=self.utc_offset)),
            )
            for dt in self.lb_datetimes
        ]

    @property
    def index(self) -> pd.DatetimeIndex:
        """Return the index of the object."""
        return pd.DatetimeIndex(self.datetimes)

    @property
    def df(self) -> pd.DataFrame:
        """Return the object as a DataFrame."""
        return self.to_dataframe()

    @property
    def dnr(self) -> pd.Series:
        """Return the direct normal irradiance values as a pandas Series."""
        if self.direct_normal_irradiance is None:
            raise ValueError("Direct normal irradiance values are not available.")
        return pd.Series(
            self.direct_normal_irradiance,
            index=self.index,
            name="Direct Normal Irradiance (W/m2)",
        )

    @property
    def dhi(self) -> pd.Series:
        """Return the diffuse horizontal irradiance values as a pandas Series."""
        if self.diffuse_horizontal_irradiance is None:
            raise ValueError("Diffuse horizontal irradiance values are not available.")
        return pd.Series(
            self.diffuse_horizontal_irradiance,
            index=self.index,
            name="Diffuse Horizontal Irradiance (W/m2)",
        )

    @property
    def ghi(self) -> pd.Series:
        """Return the global horizontal irradiance values as a pandas Series."""
        if self.global_horizontal_irradiance is None:
            raise ValueError("Global horizontal irradiance values are not available.")
        return pd.Series(
            self.global_horizontal_irradiance,
            index=self.index,
            name="Global Horizontal Irradiance (W/m2)",
        )

    @property
    def lb_direct_normal_irradiance(self) -> HourlyContinuousCollection:
        return HourlyContinuousCollection(
            header=Header(
                data_type=DirectNormalIrradiance(),
                unit="W/m2",
                analysis_period=self.analysis_period,
            ),
            values=self.direct_normal_irradiance.tolist(),
        )

    @property
    def lb_diffuse_horizontal_irradiance(self) -> HourlyContinuousCollection:
        return HourlyContinuousCollection(
            header=Header(
                data_type=DiffuseHorizontalIrradiance(),
                unit="W/m2",
                analysis_period=self.analysis_period,
            ),
            values=self.diffuse_horizontal_irradiance.tolist(),
        )

    @property
    def lb_global_horizontal_irradiance(self) -> HourlyContinuousCollection:
        return HourlyContinuousCollection(
            header=Header(
                data_type=GlobalHorizontalIrradiance(),
                unit="W/m2",
                analysis_period=self.analysis_period,
            ),
            values=self.global_horizontal_irradiance.tolist(),
        )

    @property
    def lb_wea(self) -> Wea:
        return Wea(
            location=self.location,
            direct_normal_irradiance=self.lb_direct_normal_irradiance,
            diffuse_horizontal_irradiance=self.lb_diffuse_horizontal_irradiance,
        )

    # endregion: PROPERTIES

    # region: CLASS METHODS

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "Solar":
        """Create a Wind object from an EPW file or object.

        Args:
            epw (Path | EPW):
                The path to the EPW file, or an EPW object.
        """

        if isinstance(epw, (str, Path)):
            epw = EPW(epw)

        return cls(
            location=epw.location,
            analysis_period=epw.dry_bulb_temperature.header.analysis_period,
            direct_normal_irradiance=epw.direct_normal_radiation.values,
            diffuse_horizontal_irradiance=epw.diffuse_horizontal_radiation.values,
            global_horizontal_irradiance=epw.global_horizontal_radiation.values,
        )

    def to_dict(self) -> dict:
        """Return the object as a dictionary."""

        return {
            "type": "Solar",
            "location": self.location.to_dict(),
            "analysis_period": self.analysis_period.to_dict(),
            "direct_normal_irradiance": self.direct_normal_irradiance.tolist(),
            "diffuse_horizontal_irradiance": self.diffuse_horizontal_irradiance.tolist(),
            "global_horizontal_irradiance": self.global_horizontal_irradiance.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Solar":
        """Create this object from a dictionary."""

        if d.get("type", None) != "Solar":
            raise ValueError("The dictionary cannot be converted Solar object.")

        return cls(
            location=Location.from_dict(d["location"]),
            analysis_period=AnalysisPeriod.from_dict(d["analysis_period"]),
            direct_normal_irradiance=d["direct_normal_irradiance"],
            diffuse_horizontal_irradiance=d["diffuse_horizontal_irradiance"],
            global_horizontal_irradiance=d["global_horizontal_irradiance"],
        )

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_string: str) -> "Solar":
        """Create this object from a JSON string."""
        return cls.from_dict(json.loads(json_string))

    def to_dataframe(self) -> pd.DataFrame:
        """Return the object as a DataFrame."""
        df = pd.DataFrame(
            {
                "direct_normal_irradiance": self.direct_normal_irradiance,
                "diffuse_horizontal_irradiance": self.diffuse_horizontal_irradiance,
                "global_horizontal_irradiance": self.global_horizontal_irradiance,
            },
            index=self.index,
        )

        # add location data
        ld = self.location.to_dict()
        ld.pop("type")
        for k, v in ld.items():
            df[k] = v

        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, location: Location = None) -> "Solar":
        """Create this object from a DataFrame.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing the solar data.
            location (Location, optional):
                A ladybug Location object. If not provided, the location data
                will be extracted from the DataFrame if present.
        """

        # check that the index is datetime and timezone-aware
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame's index must be of type pd.DatetimeIndex.")
        if df.index.tzinfo is None:
            raise ValueError("The DataFrame's index must be timezone-aware.")

        # check that all months of the year are present in the datetime index
        if set(df.index.month) != set(range(1, 13)):
            raise ValueError("The DataFrame's index must cover a full year.")

        # check that the index frequency is periodic
        freq = pd.infer_freq(df.index)
        td = pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))
        if freq is None:
            raise ValueError("The DataFrame's index frequency must be consistent.")
        if td.seconds > 3600:
            raise ValueError(
                "The DataFrame's index frequency must be hourly or sub-hourly."
            )

        # check that the dataframe does not contain any NaN values
        if df.isnull().values.any():
            raise ValueError("The DataFrame must not contain any NaN values.")

        # if location is not provided, attempt to extract it from the DataFrame
        if location is None:
            for col in _LOCATION_ARGS:
                if col not in df.columns:
                    raise ValueError(
                        f"When a Location is not passed, the DataFrame must contain columns of {_LOCATION_ARGS}."
                    )
                if len(df[col].unique()) != 1:
                    raise ValueError(
                        f"The DataFrame's {col} column must contain a single unique value."
                    )
            loc_dict = df[_LOCATION_ARGS].iloc[0].to_dict()
            location = Location(**loc_dict)
            # drop these columns from the DataFrame
            df = df.drop(columns=_LOCATION_ARGS)

        # check utc_offset is the same as the location time_zone
        if df.index.tzinfo.utcoffset(None).seconds / 3600 != location.time_zone:
            raise ValueError(
                "The DataFrame's index timezone must match the location's time_zone."
            )

        # if the index spans multiple years, groupby time and take the mean
        if min(df.index.year) != max(df.index.year):
            warnings.warn(
                "The dataframe given contains more than one years worth of solar data. The data will be averaged by time and converted into a single years worth."
            )
            df = df.groupby([df.index.month, df.index.day, df.index.time]).mean()
            # construct a new datetimeindex, making it a leap year if necessary
            dts = analysis_period_from_pd_freq(
                freq, is_leap_year=True if len(df) == 8784 else False
            ).datetimes
            df.index = pd.to_datetime(
                [
                    datetime(
                        dt.year,
                        dt.month,
                        dt.day,
                        dt.hour,
                        dt.minute,
                        dt.second,
                        tzinfo=timezone(timedelta(hours=location.time_zone)),
                    )
                    for dt in dts
                ]
            )

        # construct the analysis_period from the datetime index
        ap = analysis_period_from_datetimes(df.index.tolist())

        return cls(
            location=location,
            analysis_period=ap,
            direct_normal_irradiance=df["direct_normal_irradiance"],
            diffuse_horizontal_irradiance=df["diffuse_horizontal_irradiance"],
            global_horizontal_irradiance=df["global_horizontal_irradiance"],
        )

    @classmethod
    def from_average(
        cls, objects: list["Solar"], weights: list[int | float] | None = None
    ) -> "Solar":
        # check that the objects are aligned
        [Solar._is_aligned(objects[0], i) for i in objects[1:]]

        # create the average data's
        dni = np.average(
            [i.direct_normal_irradiance for i in objects], weights=weights, axis=0
        )
        dhi = np.average(
            [i.diffuse_horizontal_irradiance for i in objects],
            weights=weights,
            axis=0,
        )

        ghi = np.average(
            [i.global_horizontal_irradiance for i in objects],
            weights=weights,
            axis=0,
        )

        return cls(
            location=objects[0].location,
            analysis_period=objects[0].analysis_period,
            direct_normal_irradiance=dni,
            diffuse_horizontal_irradiance=dhi,
            global_horizontal_irradiance=ghi,
        )

    # endregion: CLASS METHODS

    # region: INSTANCE METHODS

    def to_lb_sky_matrix(
        self,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        temperature: HourlyContinuousCollection | list[float | int] = None,
        balance_temperature: float = 15,
        balance_offset: float = 2,
    ) -> SkyMatrix:
        """Create a ladybug sky matrix from the solar data. If temperature is
        provided, the sky matrix will be created with the benefit matrix method.

        Args:
            north (int, optional):
                The north direction in degrees.
                Default is 0.
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Default is True.
            ground_reflectance (float, optional):
                The ground reflectance value.
                Default is 0.2.
            temperature (HourlyContinuousCollection | list[float | int], optional):
                A ladybug HourlyContinuousCollection object or a list of temperature values in C.
                If provided, the sky matrix will be created with an associated benefit matrix.
            balance_temperature (float, optional):
                The temperature at which the sky matrix is balanced.
                Default is 15.
            balance_offset (float, optional):
                The offset from the balance temperature.
                Default is 2.
        """
        if temperature is None:
            return SkyMatrix(
                wea=self.lb_wea,
                north=north,
                high_density=high_density,
                ground_reflectance=ground_reflectance,
            )

        # check that temperature is a valid type
        if not isinstance(temperature, (HourlyContinuousCollection)):
            self._is_aligned_array(self, temperature)
            temperature = HourlyContinuousCollection(
                header=Header(
                    data_type=Temperature(),
                    unit="C",
                    analysis_period=self.analysis_period,
                ),
                values=temperature,
            )
        return SkyMatrix.from_components_benefit(
            location=self.location,
            direct_normal_irradiance=self.lb_direct_normal_irradiance,
            diffuse_horizontal_irradiance=self.lb_diffuse_horizontal_irradiance,
            north=north,
            high_density=high_density,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )

    def to_lb_radiation_rose(
        self,
        directions: int = 36,
        tilt_angle: float = 0,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
    ) -> RadiationRose:
        return RadiationRose(
            sky_matrix=self.to_lb_sky_matrix(
                north=north,
                high_density=high_density,
                ground_reflectance=ground_reflectance,
            ),
            direction_count=directions,
            tilt_angle=tilt_angle,
        )

    def radiation_benefit(
        self,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        temperature: HourlyContinuousCollection | list[float | int] = None,
        balance_temperature: float = 15,
        balance_offset: float = 2,
    ) -> pd.Series:
        # create the sky matrix
        smx = self.to_lb_sky_matrix(
            north=north,
            high_density=high_density,
            ground_reflectance=ground_reflectance,
            temperature=temperature,
            balance_temperature=balance_temperature,
            balance_offset=balance_offset,
        )
        # replace None values with NaN
        d = []
        for i in smx.benefit_matrix:
            if i is None:
                d.append(0.0)
            elif i:
                d.append(1.0)
            else:
                d.append(-1.0)
        return pd.Series(d, index=self.index, name="Radiation Benefit")

    def directional_irradiance(
        self,
        altitude: int = 90,
        azimuth: int = 180,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
        as_dataframe: bool = False,
    ) -> dict[str, HourlyContinuousCollection] | pd.DataFrame:
        """Calculate the directional irradiance values on a surface of a given altitude and azimuth."""

        directional = self.lb_wea.directional_irradiance(
            altitude=altitude,
            azimuth=azimuth,
            ground_reflectance=ground_reflectance,
            isotropic=isotropic,
        )

        if as_dataframe:
            return pd.DataFrame(
                data=list(map(list, zip(*[i.values for i in directional]))),
                columns=[
                    "total_irradiance",
                    "direct_irradiance",
                    "diffuse_irradiance",
                    "reflected_irradiance",
                ],
                index=self.index,
            )
        return directional

    def tilt_orientation_irradiance(
        self,
        n_altitudes: int,
        n_azimuths: int,
        ground_reflectance: float = 0.2,
        isotropic: bool = True,
    ) -> pd.DataFrame:
        """Calculate irradiance values for a number of tilts and orientations."""

        # create list of tilts
        if isinstance(n_altitudes, int):
            altitudes = np.linspace(0, 90, n_altitudes)
        else:
            raise ValueError("tilts must be an integer.")

        # create list of orientations
        if isinstance(n_azimuths, int):
            azimuths = np.linspace(0, 360, n_azimuths)
        else:
            raise ValueError("orientations must be an integer.")

        # create a list of tuples of azimuth and altitude
        az_alts = [(a, o) for a in azimuths for o in altitudes]

        # iterate each and create dataframes, but do it in parallel!
        dfs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for az, alt in tqdm(az_alts):
                futures.append(
                    executor.submit(
                        self.directional_irradiance,
                        altitude=alt,
                        azimuth=az,
                        ground_reflectance=ground_reflectance,
                        isotropic=isotropic,
                        as_dataframe=True,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                dfs.append(future.result().values)

        return np.array(dfs)

    # endregion: INSTANCE METHODS
