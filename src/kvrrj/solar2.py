"""Methods for handling solar data. This module relies heavily on numpy, pandas, and ladybug."""

# TODO - From openmeteo
# TODO - Shade benefit calc (on window)
# TODO - Shade benefit calc on comfort exterbal
# TODO - PV calc
# TODO - PV with shade objects (from sky matrix?)
# TODO - From NOAA?
# TODO - New openmeteo method? Save location id and daily data as separate files
# TODO - Use raddome for creating the TOF data
# TODO - Reimplement down sample to enable finer grained shade studry
# TODO - Use DirectSun/RadiationStudy to calculate shadedness of a point given context meshes
# TODO - Add location to wind object
# TODO - load shade objects from a file/create shades, and compute the benefit matrix?

import concurrent
import inspect
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
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
from ladybug_geometry.geometry3d import (
    Face3D,
    LineSegment3D,
    Mesh3D,
    Plane,
    Point3D,
    Vector3D,
)
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.study.radiation import RadiationStudy
from ladybug_radiance.visualize.raddome import RadiationDome
from ladybug_radiance.visualize.radrose import RadiationRose
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm

from kvrrj.geometry.util import vector3d_to_azimuth_altitude
from kvrrj.viz.color import contrasting_color

from .ladybug.analysis_period import (
    AnalysisPeriod,
    analysis_period_from_datetimes,
    analysis_period_from_pd_freq,
    analysis_period_to_string,
)
from .logging import CONSOLE_LOGGER  # noqa: F401

_LOCATION_ARGS = inspect.signature(Location).parameters.keys()


class IrradianceType(Enum):
    """Irradiance types."""

    TOTAL = auto()
    DIRECT = auto()
    DIFFUSE = auto()
    REFLECTED = auto()

    def to_string(self) -> str:
        """Get the string representation of the IrradianceType."""
        return f"{self.name.title()} irradiance"


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

    @staticmethod
    def _create_single_face_mesh(
        azimuth: float,
        altitude: float,
        height_above_ground: float,
        radius: float = 0.01,
    ) -> Mesh3D:
        """Create a single face mesh for a given azimuth and altitude.

        Args:
            azimuth (float):
                The azimuth angle in degrees clockwise from north.
            altitude (float):
                The altitude angle in degrees.
            height_above_ground (float):
                The height above ground in m.
        """

        if azimuth > 360 or azimuth < 0:
            raise ValueError("azimuth must be between 0 and 360.")
        if altitude > 90 or altitude < -90:
            raise ValueError("altitude must be between -90 and 90.")
        if height_above_ground < 0:
            raise ValueError("height_above_ground must be greater than or equal to 0.")

        vec3d = (
            Vector3D(0, 1, 0)
            .rotate(axis=Vector3D(1, 0, 0), angle=np.deg2rad(altitude))
            .rotate_xy(np.deg2rad(-azimuth))
        )
        plane = Plane(n=vec3d, o=Point3D(0, 0, height_above_ground))
        face = Face3D.from_regular_polygon(
            side_count=4, radius=radius, base_plane=plane
        )
        return Mesh3D.from_face_vertices([face])

    @staticmethod
    def _create_azimuth_mesh(directions: int = 36, tilt_angle: float = 0) -> Mesh3D:
        """CReate a mesh of faces for a given number of directions and tilt angle.

        This is used to creation the radiation rose, with one face per direction.

        Args:
            directions (int, optional):
                The number of directions to divide the rose into.
                Default is 36.
            tilt_angle (float, optional):
                The tilt angle in degrees from horizontal. 0 is horizontal, 90 is upwards.
                Default is 0.

        Returns:
            Mesh3D:
                A ladybug Mesh3D object
        """

        angles = np.linspace(0, 360, directions, endpoint=False)
        base_face = Face3D.from_extrusion(
            line_segment=LineSegment3D(p=Point3D(0.05, 0.1, 0), v=Vector3D(-0.1, 0, 0)),
            extrusion_vector=Vector3D(0, 0, 0.1),
        ).rotate(axis=Vector3D(1, 0, 0), angle=np.deg2rad(tilt_angle), origin=Point3D())
        faces = [
            base_face.rotate_xy(angle=np.deg2rad(-a), origin=Point3D()) for a in angles
        ]
        return Mesh3D.from_face_vertices(faces=faces)

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

        # modify location to state the EPW file in the source field
        loc = epw.location
        loc.source = f"{Path(epw.file_path).name}"

        return cls(
            location=loc,
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

        Returns:
            SkyMatrix:
                A ladybug SkyMatrix object.
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

    # TODO - reimplement this
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
        smx._benefit_matrix
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

    def to_lb_radiation_rose(
        self,
        directions: int = 36,
        tilt_angle: float = 90,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
    ) -> RadiationRose:
        """Convert this object to a ladybug RadiationRose object.

        Args:
            directions (int, optional):
                The number of directions to divide the rose into.
                Default is 36.
            tilt_angle (float, optional):
                The tilt angle in degrees from horizontal. 0 is horizontal, 90 is upwards.
                Default is 90.
            north (int, optional):
                The north direction in degrees.
                Default is 0.
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Default is True.
            ground_reflectance (float, optional):
                The ground reflectance value.
                Default is 0.2.

        Returns:
            RadiationRose:
                A ladybug RadiationRose object.
        """

        return RadiationRose(
            sky_matrix=self.to_lb_sky_matrix(
                north=north,
                high_density=high_density,
                ground_reflectance=ground_reflectance,
            ),
            direction_count=directions,
            tilt_angle=tilt_angle,
        )

    def to_lb_radiation_study(
        self,
        **kwargs,
    ) -> RadiationStudy:
        """Convert this object to a ladybug RadiationStudy object.

        Args:
            **kwargs (dict[str, Any]):
                A set of keyword arguments to pass to the Ladybug SkyMatrix and RadiationStudy constructors.

        Returns:
            RadiationStudy:
                A ladybug RadiationStudy object.
        """

        # get the skymatrix kwargs
        smx_kwargs = list(inspect.signature(SkyMatrix).parameters)
        smx_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in smx_kwargs}
        wea = smx_dict.pop("wea", self.lb_wea)
        smx = SkyMatrix(wea=wea, **smx_dict)

        # create the radiation study
        rs_kwargs = list(inspect.signature(RadiationStudy).parameters)
        rs_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in rs_kwargs}
        sky_matrix = rs_dict.pop("sky_matrix", smx)
        study_mesh = rs_dict.pop(
            "study_mesh",
            self._create_single_face_mesh(
                azimuth=180, altitude=90, height_above_ground=0
            ),
        )
        context_geometry = rs_dict.pop(
            "context_geometry",
            [],
        )
        return RadiationStudy(
            sky_matrix=sky_matrix,
            study_mesh=study_mesh,
            context_geometry=context_geometry,
            **rs_dict,
        )

    def to_lb_radiation_dome(self) -> RadiationDome:
        # TODO - implement this
        RadiationDome()
        raise NotImplementedError("This method is not yet implemented.")

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

        # TODO-  make this function wokr, and transform the output into a table of form [rad_types, [aziumths, altitudes]]

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

    # region: PLOTTING METHODS

    def directional_radiation(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 89.999,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = (),
    ) -> pd.DataFrame:
        """Get directional cumulative radiation in kWh/m2 for a given
        tilt_angle, within the analysis_period and subject to shade_objects.

        Args:
            irradiance_type (IrradianceType, optional):
                The type of irradiance to plot. Defaults to IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            directions (int, optional):
                The number of directions to bin data into.
                Defaults to 36.
            tilt_angle (float, optional):
                The tilt (from 0 at horizon, to 90 facing the sky) to assess.
                Defaults to 89.999.
            north (int, optional):
                The north direction in degrees.
                Defaults to 0.
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Defaults to True.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.

        Returns:
            pd.DataFrame:
                A pandas DataFrame containing the radiation data.
        """

        # create time-filtered sky-matrix
        smx = SkyMatrix.from_components(
            location=self.location,
            direct_normal_irradiance=self.lb_direct_normal_irradiance,
            diffuse_horizontal_irradiance=self.lb_diffuse_horizontal_irradiance,
            hoys=analysis_period.hoys,
            north=north,
            high_density=high_density,
            ground_reflectance=ground_reflectance,
        )

        # create a mesh with the same dumber of faces as the number of
        sensor_mesh = self._create_azimuth_mesh(directions, tilt_angle)

        # create a radiation study and intersection matrix from given mesh/objects
        rd = RadiationStudy(
            sky_matrix=smx,
            study_mesh=sensor_mesh,
            context_geometry=shade_objects,
            use_radiance_mesh=True,
        )

        # create rad rose
        lb_radrose = RadiationRose(
            sky_matrix=smx,
            intersection_matrix=rd.intersection_matrix,
            direction_count=directions,
            tilt_angle=tilt_angle,
        )

        # get angles
        angles = np.linspace(0, 360, directions, endpoint=False)

        # get the radiation data
        return pd.concat(
            [
                pd.Series(
                    lb_radrose.total_values,
                    index=angles,
                    name=IrradianceType.TOTAL.to_string(),
                ),
                pd.Series(
                    lb_radrose.direct_values,
                    index=angles,
                    name=IrradianceType.DIRECT.to_string(),
                ),
                pd.Series(
                    lb_radrose.diffuse_values,
                    index=angles,
                    name=IrradianceType.DIFFUSE.to_string(),
                ),
            ],
            axis=1,
        )

    def plot_radrose(
        self,
        ax: Axes | None = None,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        directions: int = 36,
        tilt_angle: float = 90,
        north: int = 0,
        high_density: bool = True,
        ground_reflectance: float = 0.2,
        shade_objects: list[Any] = (),
        **kwargs,
    ) -> Axes:
        """Plot a radiation rose for the given solar data.

        Args:
            ax (Axes, optional):
                The matplotlib Axes to plot the radiation rose on.
            irradiance_type (IrradianceType, optional):
                The type of irradiance to plot. Defaults to IrradianceType.TOTAL.
            analysis_period (AnalysisPeriod, optional):
                The analysis period over which radiation shall be summarised.
                Defaults to AnalysisPeriod().
            directions (int, optional):
                The number of directions to bin data into.
                Defaults to 36.
            tilt_angle (float, optional):
                The tilt (from 0 at horizon, to 90 facing the sky) to assess.
                Defaults to 90.
            north (int, optional):
                The north direction in degrees.
                Defaults to 0.
            high_density (bool, optional):
                If True, the sky matrix will be created with high density.
                Defaults to True.
            ground_reflectance (float, optional):
                The reflectance of the ground.
                Defaults to 0.2.
            shade_objects (list, optional):
                A list of shades to apply to the plot.
                Defaults to an empty list.
        """

        if irradiance_type == IrradianceType.REFLECTED:
            raise ValueError(
                "The REFLECTED irradiance type is not supported for plotting a radiation rose."
            )

        # create radiation results
        rad_df = self.directional_radiation(
            analysis_period=analysis_period,
            directions=directions,
            tilt_angle=tilt_angle,
            north=north,
            high_density=high_density,
            ground_reflectance=ground_reflectance,
            shade_objects=shade_objects,
        )

        # get the radiation data
        match irradiance_type:
            case IrradianceType.TOTAL:
                data = rad_df[IrradianceType.TOTAL.to_string()]
            case IrradianceType.DIRECT:
                data = rad_df[IrradianceType.DIRECT.to_string()]
            case IrradianceType.DIFFUSE:
                data = rad_df[IrradianceType.DIFFUSE.to_string()]
            case _:
                raise ValueError("How did you get here?")

        # plot the radiation rose

        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": "polar"})
        if ax.name != "polar":
            raise ValueError("ax must be a polar axis.")

        # kwargish vars
        # todo - sort out kwargs
        ylim = kwargs.pop("ylim", (0, max(data) * 1.1))
        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple of length 2.")
        bar_width = 1
        colors = plt.get_cmap("YlOrRd")(
            np.interp(data.values, (data.min(), data.max() * 1.05), (0, 1))
        )
        title = f"{self.location.source}\n{analysis_period_to_string(analysis_period)}"

        rects = ax.bar(
            x=np.deg2rad(data.index),
            height=data.values,
            width=((np.pi / directions) * 2) * bar_width,
            color=colors,
        )

        # add a text label to the peak value bar
        peak_value = max(data.values)
        peak_angle_deg = data.idxmax()
        peak_angle_rad = np.deg2rad(data.idxmax())
        peak_index = np.argmax(data.values)
        peak_bar = rects[peak_index]
        peak_bar.set_edgecolor("black")
        peak_bar.set_zorder(5)
        ax.text(
            peak_angle_rad,
            peak_value * 0.95,
            f"{peak_value:.0f}kWh/m$^2$",
            fontsize="xx-small",
            ha="right" if peak_angle_deg < 180 else "left",
            va="center",
            rotation=(90 - peak_angle_deg)
            if peak_angle_deg < 180
            else (90 - peak_angle_deg + 180),
            rotation_mode="anchor",
            color=contrasting_color(peak_bar.get_facecolor()),
            zorder=5,
        )

        # format the plot
        ax.set_title(title)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(ylim)
        ax.spines["polar"].set_visible(False)
        ax.grid(True, which="both", ls="--", zorder=0, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(6))
        plt.setp(ax.get_yticklabels(), fontsize="small")
        ax.set_xticks(np.radians((0, 90, 180, 270)), minor=False)
        ax.set_xticklabels(("N", "E", "S", "W"), minor=False, **{"fontsize": "medium"})
        ax.set_xticks(
            np.radians(
                (
                    22.5,
                    45,
                    67.5,
                    112.5,
                    135,
                    157.5,
                    202.5,
                    225,
                    247.5,
                    292.5,
                    315,
                    337.5,
                )
            ),
            minor=True,
        )
        ax.set_xticklabels(
            (
                "NNE",
                "NE",
                "ENE",
                "ESE",
                "SE",
                "SSE",
                "SSW",
                "SW",
                "WSW",
                "WNW",
                "NW",
                "NNW",
            ),
            minor=True,
            **{"fontsize": "x-small"},
        )

        return ax

    def plot_tilt_orientation_factor(
        self,
        ax: Axes | None = None,
        irradiance_type: IrradianceType = IrradianceType.TOTAL,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
        azimuth_count: int = 36,
        altitude_count: int = 9,
        shade_objects: list[Any] = (),
    ) -> Axes:
        if irradiance_type == IrradianceType.REFLECTED:
            raise ValueError(
                "The REFLECTED irradiance type is not supported for plotting a tilt-orientation-factor diagram."
            )

        # create time-filtered sky-matrix
        smx = SkyMatrix.from_components(
            location=self.location,
            direct_normal_irradiance=self.lb_direct_normal_irradiance,
            diffuse_horizontal_irradiance=self.lb_diffuse_horizontal_irradiance,
            hoys=analysis_period.hoys,
            high_density=True,
        )

        # if shade objects are passed, create a sensor mesh

        # create a dome mesh, but really small to check for intersections with shade_objects
        dome_vectors = RadiationDome.dome_vectors(
            azimuth_count=azimuth_count, altitude_count=altitude_count
        )
        faces = []
        for v in dome_vectors:
            faces.append(
                Face3D.from_regular_polygon(
                    side_count=3,
                    radius=0.001,
                    base_plane=Plane(n=v, o=Point3D().move(v * 0.001)),
                )
            )
        sensor_mesh = Mesh3D.from_face_vertices(faces=faces)

        # create a radiation study and intersection matrix from given mesh/objects
        rs = RadiationStudy(
            sky_matrix=smx,
            study_mesh=sensor_mesh,
            context_geometry=shade_objects,
            use_radiance_mesh=True,
        )
        intersection_matrix = rs.intersection_matrix

        # create a radiation dome
        rd = RadiationDome(
            smx,
            intersection_matrix=intersection_matrix,
            azimuth_count=azimuth_count,
            altitude_count=altitude_count,
        )

        # get the raw data
        azimuths, altitudes = np.array(
            [vector3d_to_azimuth_altitude(i) for i in rd.direction_vectors]
        ).T
        match irradiance_type:
            case IrradianceType.TOTAL:
                values = np.array(rd.total_values)
            case IrradianceType.DIRECT:
                values = np.array(rd.direct_values)
            case IrradianceType.DIFFUSE:
                values = np.array(rd.diffuse_values)
            case _:
                raise ValueError("How did you get here?")

        # create a dataframe containing the results
        df = pd.DataFrame(
            {
                "azimuth": azimuths,
                "altitude": altitudes,
                "value": values,
            }
        ).sort_values(by=["azimuth", "altitude"])

        # add in  missing extremity values
        # todo - fix
        # todo - test with shade
        temp = df[df["altitude"] == 0]
        temp["altitude"] = 90
        temp["value"] = df[df["altitude"] == 90]["value"].values[0]

        temp2 = df[df["azimuth"] == 0]
        temp2["azimuth"] = 360

        df = pd.concat([df, temp, temp2])

        if ax is None:
            ax = plt.gca()

        title = f"{self.location.source}\n{analysis_period_to_string(analysis_period)}"
        ax.set_title(title)

        ax.tricontourf(df["azimuth"], df["altitude"], df["value"], levels=20)
        plt.tight_layout()

        return ax

    # endregion: PLOTTING METHODS
