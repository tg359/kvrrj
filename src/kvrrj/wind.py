import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ladybug.datatype.angle import WindDirection
from ladybug.datatype.speed import WindSpeed
from ladybug.epw import EPW, AnalysisPeriod, Header, HourlyContinuousCollection
from ladybug.windrose import WindRose

from kvrrj.util import wind_speed_at_height

from .geometry.util import (
    angle_clockwise_from_north,
    angle_to_vector,
    circular_weighted_mean,
)
from .ladybug.analysis_period import (
    analysis_period_to_datetimes,
    analysis_period_to_string,
)

_DEFAULT_DIRECTION_COUNT = 8


@dataclass(init=True, eq=True, repr=True)
class Wind:
    """An object containing historic, time-indexed wind data.

    Args:
        wind_speeds (list[int | float | np.number]):
            An iterable of wind speeds in m/s.
        wind_directions (list[int | float | np.number]):
            An iterable of wind directions in degrees from North (with North at 0-degrees).
        datetimes (Union[pd.DatetimeIndex, list[Union[datetime, np.datetime64, pd.Timestamp]]]):
            An iterable of datetime-like objects.
        height_above_ground (float, optional):
            The height above ground (in m) where the input wind speeds and directions were
            collected. Defaults to 10m.
        source (str, optional):
            A source string to describe where the input data comes from. Defaults to None.
    """

    wind_speeds: list[float]
    wind_directions: list[float]
    datetimes: list[datetime]
    height_above_ground: float
    source: str = None

    # region: DUNDER METHODS

    def __post_init__(self):
        if self.height_above_ground < 0.1:
            raise ValueError("Height above ground must be >= 0.1m.")

        if not isinstance(self.wind_speeds, list):
            raise ValueError("wind_speeds must be a list.")
        if not isinstance(self.wind_directions, list):
            raise ValueError("wind_directions must be a list.")
        if not isinstance(self.datetimes, list):
            raise ValueError("datetimes must be a list.")

        if (
            not len(self.wind_speeds)
            == len(self.wind_directions)
            == len(self.datetimes)
        ):
            raise ValueError(
                "wind_speeds, wind_directions and datetimes must be the same length."
            )

        if len(self.wind_speeds) <= 1:
            raise ValueError(
                "wind_speeds, wind_directions and datetimes must be at least 2 items long."
            )

        if len(set(self.datetimes)) != len(self.datetimes):
            raise ValueError("datetimes contains duplicates.")

        # validate wind speeds and directions
        if np.any(np.isnan(self.wind_speeds)):
            raise ValueError("wind_speeds contains null values.")

        if np.any(np.isnan(self.wind_directions)):
            raise ValueError("wind_directions contains null values.")

        if np.any(np.array(self.wind_speeds) < 0):
            raise ValueError("wind_speeds must be >= 0")
        if np.any(np.array(self.wind_directions) < 0) or np.any(
            np.array(self.wind_directions) > 360
        ):
            raise ValueError("wind_directions must be within 0-360")
        self.wind_directions = [i % 360 for i in self.wind_directions]

    def __len__(self) -> int:
        return len(self.datetimes)

    def __repr__(self) -> str:
        """The printable representation of the given object"""
        if self.source:
            return f"{self.__class__.__name__}(@{self.height_above_ground}m) from {self.source}"

        return (
            f"{self.__class__.__name__}({min(self.datetimes):%Y-%m-%d} to "
            f"{max(self.datetimes):%Y-%m-%d}, n={len(self.datetimes)} @{self.freq}, "
            f"@{self.height_above_ground}m) NO SOURCE"
        )

    def __str__(self) -> str:
        """The string representation of the given object"""
        return self.__repr__()

    # endregion: DUNDER METHODS

    # region: CLASS METHODS

    def to_dict(self) -> dict:
        """Return the object as a dictionary."""
        d = asdict(self)
        # d["_t"] = f"{self.__module__}.{self.__class__.__name__}"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Wind":
        """Create this object from a dictionary."""
        return cls(**d)

    def to_json(self) -> str:
        """Convert this object to a JSON string."""
        d = self.to_dict()
        d["datetimes"] = [dt.isoformat() for dt in self.datetimes]
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_string: str) -> "Wind":
        """Create this object from a JSON string."""
        d = json.loads(json_string)
        d["datetimes"] = [datetime.fromisoformat(dt) for dt in d["datetimes"]]
        return cls.from_dict(d)

    def to_json_file(self, path: Path) -> Path:
        """Convert this object to a JSON file."""

        if Path(path).suffix != ".json":
            raise ValueError("path must be a JSON file.")
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(Path(path), "w", encoding="utf-8") as fp:
            fp.write(self.to_json())

        return Path(path)

    @classmethod
    def from_json_file(cls, path: Path) -> "Wind":
        """Create this object from a JSON file."""
        with open(Path(path), "r", encoding="utf-8") as fp:
            return cls.from_json(fp.read())

    def to_csv_file(self, path: Path) -> Path:
        """Save this object as a csv file.

        Args:
            path (Path):
                The path containing the CSV file.

        Returns:
            Path:
                The resultant CSV file.
        """
        if Path(path).suffix != ".csv":
            raise ValueError("path must be a CSV file.")
        path.parent.mkdir(parents=True, exist_ok=True)

        csv_path = Path(path)
        self.df.to_csv(csv_path)
        return csv_path

    @classmethod
    def from_csv_file(
        cls,
        csv_path: Path,
        wind_speed_column: str = None,
        wind_direction_column: str = None,
        height_above_ground: float = 10,
        **kwargs,
    ) -> "Wind":
        """Create a Wind object from a csv containing wind speed and direction columns.

        Args:
            csv_path (Path):
                The path to the CSV file containing speed and direction columns,
                and a datetime index.
            wind_speed_column (str):
                The name of the column where wind-speed data exists.
            wind_direction_column (str):
                The name of the column where wind-direction data exists.
            height_above_ground (float, optional):
                Defaults to 10m.
            **kwargs:
                Additional keyword arguments passed to pd.read_csv.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0, **kwargs)
        return cls.from_dataframe(
            df,
            wind_speed_column=wind_speed_column,
            wind_direction_column=wind_direction_column,
            height_above_ground=height_above_ground,
            source=csv_path.name,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the object as a DataFrame."""
        return pd.DataFrame(
            {
                "speed": self.wind_speeds,
                "direction": self.wind_directions,
            },
            index=self.datetimes,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        wind_speed_column: Any = None,
        wind_direction_column: Any = None,
        height_above_ground: float = 10,
        source: str = "DataFrame",
    ) -> "Wind":
        """Create a Wind object from a Pandas DataFrame, with speed and direction columns.

        Args:
            df (pd.DataFrame):
                A DataFrame object containing speed and direction columns, and a datetime index.
            wind_speed_column (str):
                The name of the column where wind-speed data exists.
            wind_direction_column (str):
                The name of the column where wind-direction data exists.
            height_above_ground (float, optional):
                Defaults to 10m.
            source (str, optional):
                A source string to describe where the input data comes from.
                Defaults to "DataFrame"".
        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"df must be of type {type(pd.DataFrame)}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"The DataFrame's index must be of type {type(pd.DatetimeIndex)}"
            )

        # remove NaN values
        df.dropna(axis=0, how="any", inplace=True)

        # remove duplicates in input dataframe
        df = df.loc[~df.index.duplicated()]

        # if wind_speed_column and wind_direction_column are not provided, attempt to use some best-guesses
        if wind_speed_column is None:
            for col in ["wind_speed", "speed", "ws", "Wind Speed (m/s)"]:
                if col in df.columns:
                    wind_speed_column = col
                    break
            if wind_speed_column is None:
                raise ValueError(
                    "wind_speed_column not found in DataFrame. You'll need to provide a specific column index rather than relying on a best-guess."
                )
        if wind_direction_column is None:
            for col in [
                "wind_direction",
                "direction",
                "wd",
                "Wind Direction (degrees)",
            ]:
                if col in df.columns:
                    wind_direction_column = col
                    break
            if wind_direction_column is None:
                raise ValueError(
                    "wind_direction_column not found in DataFrame. You'll need to provide a specific column index rather than relying on a best-guess."
                )

        return cls(
            wind_speeds=df[wind_speed_column].tolist(),
            wind_directions=df[wind_direction_column].tolist(),
            datetimes=df.index.tolist(),
            height_above_ground=height_above_ground,
            source=source,
        )

    @classmethod
    def from_epw(cls, epw: Path | EPW) -> "Wind":
        """Create a Wind object from an EPW file or object.

        Args:
            epw (Path | EPW):
                The path to the EPW file, or an EPW object.
        """

        if isinstance(epw, (str, Path)):
            source = Path(epw).name
            epw = EPW(epw)
        else:
            source = Path(epw.file_path).name

        return cls(
            wind_speeds=list(epw.wind_speed.values),
            wind_directions=list(epw.wind_direction.values),
            datetimes=analysis_period_to_datetimes(AnalysisPeriod()),
            height_above_ground=10,
            source=source,
        )

    @classmethod
    def from_average(
        cls, wind_objects: list["Wind"], weights: list[float] = None
    ) -> "Wind":
        """Create an average Wind object from a set of input Wind objects, with optional weighting for each."""

        # create default weightings if None
        if weights is None:
            weights = [1 / len(wind_objects)] * len(wind_objects)
        else:
            if sum(weights) != 1:
                raise ValueError("weights must total 1.")

        # create source string
        source = []
        for src, wgt in list(zip([wind_objects, weights])):
            source.append(f"{src.source}|{wgt}")
        source = "_".join(source)

        # align collections so that intersection only is created
        df_ws = pd.concat([i.ws for i in wind_objects], axis=1).dropna()
        df_wd = pd.concat([i.wd for i in wind_objects], axis=1).dropna()

        # construct the weighted means
        wd_avg = np.array(
            [circular_weighted_mean(i, weights) for _, i in df_wd.iterrows()]
        )
        ws_avg = np.average(df_ws, axis=1, weights=weights)
        dts = df_ws.index

        # return the new averaged object
        return cls(
            wind_speeds=ws_avg.tolist(),
            wind_directions=wd_avg.tolist(),
            datetimes=dts,
            height_above_ground=np.average(
                [i.height_above_ground for i in wind_objects], weights=weights
            ),
            source=source,
        )

    @classmethod
    def from_uv(
        cls,
        u: list[float],
        v: list[float],
        datetimes: list[datetime],
        height_above_ground: float = 10,
        source: str = None,
    ) -> "Wind":
        """Create a Wind object from a set of U, V wind components.

        Args:
            u (list[float]):
                An iterable of U (eastward) wind components in m/s.
            v (list[float]):
                An iterable of V (northward) wind components in m/s.
            datetimes (list[datetime]):
                An iterable of datetime-like objects.
            height_above_ground (float, optional):
                The height above ground (in m) where the input wind speeds and
                directions were collected.
                Defaults to 10m.
            source (str, optional):
                A source string to describe where the input data comes from.
                Defaults to None.

        Returns:
            Wind:
                A Wind object!
        """

        # convert UV into angle and magnitude
        wind_direction = angle_clockwise_from_north(np.stack([u, v]))
        wind_speed = np.sqrt(np.square(u) + np.square(v))

        if any(wind_direction[wind_speed == 0] == 90):
            warning_message = "Some input vectors have velocity of 0. This is not bad, but can mean directions may be misreported."
            warnings.warn(warning_message, UserWarning)

        return cls(
            wind_speeds=wind_speed.tolist(),
            wind_directions=wind_direction.tolist(),
            datetimes=datetimes,
            height_above_ground=height_above_ground,
            source=source,
        )

    def to_lb_windrose(self, direction_count: int = 12) -> WindRose:
        """Convert this object to a ladybug WindRose object.

        Args:
            direction_count (int, optional):
                The number of directions to use in the wind rose. Defaults to 12.

        Returns:
            WindRose:
                A ladybug WindRose object
        """
        # check that theobject covers a full year
        return WindRose(
            analysis_data_collection=self.ws_datacollection,
            direction_data_collection=self.wd_datacollection,
            direction_count=direction_count,
        )

    # endregion: CLASS METHODS

    # region: PROPERTIES

    @property
    def freq(self) -> str:
        """Return the inferred frequency of the datetimes associated with this object."""
        freq = pd.infer_freq(self.datetimes)
        if freq is None:
            return "inconsistent"
        return freq

    @property
    def index(self) -> pd.DatetimeIndex:
        """Get the datetimes as a pandas DateTimeIndex."""
        return pd.to_datetime(self.datetimes)

    @property
    def ws(self) -> pd.Series:
        """Convenience accessor for wind speeds as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_speeds, index=self.index, name="Wind Speed (m/s)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def ws_datacollection(self) -> HourlyContinuousCollection:
        """Convert the wind speeds to a ladybug datacollection."""
        # check if the datetimes are a single year of hourly data
        if not self._is_single_year_hourly(self.datetimes):
            raise ValueError(
                "Wind data must be a single year of hourly data to be converted to a ladybug WindRose object."
            )
        # create the datacolection
        col = HourlyContinuousCollection(
            header=Header(
                analysis_period=AnalysisPeriod(
                    is_leap_year=True if len(self.datetimes) == 8784 else False
                ),
                data_type=WindSpeed(),
                unit="m/s",
                metadata={"source": self.source},
            ),
            values=self.wind_speeds,
        )
        return col

    @property
    def wd_datacollection(self) -> HourlyContinuousCollection:
        """Convert the wind directions to a ladybug datacollection."""
        # check if the datetimes are a single year of hourly data
        if not self._is_single_year_hourly(self.datetimes):
            raise ValueError(
                "Wind data must be a single year of hourly data to be converted to a ladybug WindRose object."
            )
        # create the datacolection
        col = HourlyContinuousCollection(
            header=Header(
                analysis_period=AnalysisPeriod(
                    is_leap_year=True if len(self.datetimes) == 8784 else False
                ),
                data_type=WindDirection(),
                unit="degrees",
                metadata={"source": self.source},
            ),
            values=self.wind_directions,
        )
        return col

    @property
    def wd(self) -> pd.Series:
        """Convenience accessor for wind directions as a time-indexed pd.Series object."""
        return pd.Series(
            self.wind_directions, index=self.index, name="Wind Direction (degrees)"
        ).sort_index(ascending=True, inplace=False)

    @property
    def df(self) -> pd.DataFrame:
        """Convenience accessor for wind direction and speed as a time-indexed
        pd.DataFrame object."""
        return self.to_dataframe()

    @property
    def calm_datetimes(self) -> list[datetime]:
        """Return the datetimes where wind speed is < 0.1.

        Returns:
            list[datetime]:
                "Calm" wind datetimes.
        """
        return self.ws[self.ws <= 0.1].index.tolist()  # pylint: disable=E1136

    @property
    def uv(self) -> pd.DataFrame:
        """Return the U and V wind components in m/s."""
        u, v = angle_to_vector(self.wd)
        return pd.concat([u * self.ws, v * self.ws], axis=1, keys=["u", "v"])

    @property
    def mean_uv(self) -> list[float, float]:
        """Calculate the average U and V wind components in m/s.

        Returns:
            list[float, float]:
                A tuple containing the average U and V wind components.
        """
        return self.uv.mean().tolist()

    def mean_speed(self, remove_calm: bool = False) -> float:
        """Return the mean wind speed for this object.

        Args:
            remove_calm (bool, optional):
                Remove calm wind speeds before calculating the mean. Defaults to False.

        Returns:
            float:
                Mean wind speed.

        """
        return np.linalg.norm(
            self.filter_by_speed(min_speed=1e-10 if remove_calm else 0).mean_uv
        )

    @property
    def mean_direction(self) -> tuple[float, float]:
        """Calculate the average direction for this object."""
        return angle_clockwise_from_north(self.mean_uv)

    @property
    def min_speed(self) -> float:
        """Return the min wind speed for this object."""
        return self.ws.min()

    @property
    def max_speed(self) -> float:
        """Return the max wind speed for this object."""
        return self.ws.max()

    @property
    def median_speed(self) -> float:
        """Return the median wind speed for this object."""
        return self.ws.median()

    # endregion: PROPERTIES

    # region: STATIC METHODS

    @staticmethod
    def _is_single_year_hourly(datetimes: list[datetime]) -> bool:
        """Check if the datetimes are a single year of hourly data.

        Args:
            datetimes (list[datetime]):
                A list of datetime objects.

        Returns:
            bool:
                True if the datetimes are a single year of hourly data.
        """

        # check if the datetimes are hourly
        if len(datetimes) not in [8760, 8784]:
            return False

        # check if the datetimes are a single year
        if len(set([dt.year for dt in datetimes])) != 1:
            return False

        # check if the datetimes are hourly
        if len(set([dt.hour for dt in datetimes])) != 24:
            return False

        return True

    @staticmethod
    def _direction_midpoints(directions: int = _DEFAULT_DIRECTION_COUNT) -> list[float]:
        """Calculate the midpoints for a given number of directions.

        Args:
            directions (int):
                The number of directions to calculate midpoints for.

        Returns:
            list[float]:
                A list of midpoints.
        """
        if directions <= 2:
            raise ValueError("directions must be > 2.")
        return np.linspace(0, 360, directions + 1)[:-1].tolist()

    @staticmethod
    def _direction_bins(
        directions: int = _DEFAULT_DIRECTION_COUNT,
    ) -> list[tuple[float, float]]:
        """Calculate the bin edges for a given number of directions.

        Args:
            directions (int):
                The number of directions to calculate bin edges for.

        Returns:
            list[tuple[float, float]]:
                A list of bin edges.
        """

        edges = [
            i + (360 / directions / 2) for i in Wind._direction_midpoints(directions)
        ]
        edges.insert(0, edges[-1])
        return np.lib.stride_tricks.sliding_window_view(edges, 2).tolist()

    # endregion: STATIC METHODS

    # region: METHODS

    def calm(self, threshold: float = 0.1) -> float:
        """Return the proportion of timesteps "calm" (i.e. less than or equal
        to the threshold).

        Args:
            threshold (float, optional):
                The threshold for calm wind speeds. Defaults to 0.1.

        Returns:
            float:
                The proportion of calm instances.
        """
        s = self.ws
        return float((s <= threshold).sum() / len(s))

    def percentile(self, percentile: float) -> float:
        """Calculate the wind speed at the given percentile.

        Args:
            percentile (float):
                The percentile to calculate.

        Returns:
            float:
                Wind speed at the given percentile.
        """
        return float(self.ws.quantile(percentile))

    def to_height(
        self,
        target_height: float,
        terrain_roughness_length: float = 1,
        log_function: bool = True,
    ) -> "Wind":
        """Translate the object to a different height above ground.

        Args:
            target_height (float):
                Height to translate to (in m).
            terrain_roughness_length (float, optional):
                Terrain roughness (how big objects are to adjust translation).
                Defaults to 1.
            log_function (bool, optional):
                Whether to use log-function or pow-function. Defaults to True.

        Returns:
            Wind:
                A translated Wind object.
        """
        ws = wind_speed_at_height(
            reference_value=self.ws,
            reference_height=self.height_above_ground,
            target_height=target_height,
            terrain_roughness_length=terrain_roughness_length,
            log_function=log_function,
        )
        return Wind(
            wind_speeds=ws.tolist(),
            wind_directions=self.wind_directions,
            datetimes=self.datetimes,
            height_above_ground=target_height,
            source=f"{self.source} translated to {target_height}m",
        )

    def apply_directional_factors(
        self, directions: int, factors: tuple[float]
    ) -> "Wind":
        """Adjust wind speed values by a set of factors per direction.
        Factors start at north, and move clockwise.

        Example:
            >>> wind = Wind.from_epw(epw_path)
            >>> wind.apply_directional_factors(
            ...     directions=4,
            ...     factors=(0.5, 0.75, 1, 0.75)
            ... )

        Where northern winds would be multiplied by 0.5, eastern winds by 0.75,
        southern winds by 1, and western winds by 0.75.

        Args:
            directions (int):
                The number of directions to bin wind-directions into.
            factors (tuple[float], optional):
                Adjustment factors per direction.

        Returns:
            Wind:
                An adjusted Wind object.
        """
        # todo - update to new methods
        factors = np.array(factors).tolist()

        if len(factors) != directions:
            raise ValueError(
                f"number of factors ({len(factors)}) must equal number of directions ({directions})"
            )

        direction_binned = self.bin_data(directions=directions)
        directional_factor_lookup = dict(
            zip(*[np.roll(np.unique(direction_binned.iloc[:, 0]), 1), factors])
        )

        adjusted_wind_speed = self.ws * [
            directional_factor_lookup[i] for i in direction_binned.iloc[:, 0]
        ]

        return Wind(
            wind_speeds=adjusted_wind_speed.tolist(),
            wind_directions=self.wd.tolist(),
            datetimes=self.datetimes,
            height_above_ground=self.height_above_ground,
            source=f"{self.source} adjusted by factors {factors}",
        )

    # region: SAMPLING

    def resample(self, rule: pd.DateOffset | pd.Timedelta | str) -> "Wind":
        """Resample the wind data collection to a different timestep.
        This can only be used to downsample.

        Args:
            rule (Union[pd.DateOffset, pd.Timedelta, str]):
                A rule for resampling. This uses the same inputs as a Pandas
                Series.resample() method.

        Returns:
            Wind:
                A wind data collection object!
        """

        warnings.warn(
            (
                "Resampling wind speeds and direction is generally not advisable. "
                "When input directions are opposing, the average returned will be inaccurate, "
                "and the resultant speed does not include any outliers. USE WITH CAUTION!"
            )
        )

        resampled_speeds = self.ws.resample(rule).mean()
        if np.isnan(resampled_speeds).any():
            raise ValueError("Resampling can only be used to downsample.")

        resampled_directions = self.wd.resample(rule).apply(circular_weighted_mean)
        resampled_datetimes = resampled_speeds.index

        return Wind(
            wind_speeds=resampled_speeds.tolist(),
            wind_directions=resampled_directions.tolist(),
            datetimes=resampled_datetimes.tolist(),
            height_above_ground=self.height_above_ground,
            source=f"{self.source} (resampled to {rule})",
        )

    def _bin_direction_data(
        self, directions: int = _DEFAULT_DIRECTION_COUNT, right: bool = True
    ) -> list[tuple[float, float]]:
        """Bin the wind direction data into a set of directions.

        Args:
            directions (int):
                The number of directions to bin the data into.
            right (bool, optional):
                Whether to include the right edge of the bin.
                Defaults to True.
                If false, the left edge is included.

        Returns:
            list[tuple[float, float]]:
                A list of the bins associated with each wind direction.
        """

        bin_edges = np.concat(
            [[-1], np.unique(self._direction_bins(directions)), [361]]
        ).tolist()

        cat = pd.cut(self.wind_directions, bins=bin_edges, right=right)

        cat_bins = [(float(i.left), float(i.right)) for i in cat.categories]
        cat_cat = [(float(i.left), float(i.right)) for i in cat]

        d = dict(zip(cat_bins, cat_bins))
        d[cat_bins[0]] = (cat_bins[-1][0], cat_bins[0][1])
        d[cat_bins[-1]] = (cat_bins[-1][0], cat_bins[0][1])

        return [d[i] for i in cat_cat]

    def _bin_other_data(
        self,
        other_data: list[float] = None,
        other_bins: list[float] | int = 11,
        right: bool = True,
    ) -> list[tuple[float, float]]:
        """Bin the "other" data into a set of bins.

        Args:
            other_data (list[float]):
                The "other" data to bin.
                Defaults to the wind speeds associated with the wind directions.
            other_bins (list[float] | int):
                The bin edges to use.
                Defaults to 11 bins between the min and max of the "other" data.
            right (bool, optional):
                Whether to include the right edge of the bin.
                Defaults to True.
                If false, the left edge is included instead.

        Returns:
            list[tuple[float, float]]:
                A list of the bins associated with each "other" data point.
        """

        if other_data is None:
            other_data = self.wind_speeds

        if len(other_data) != len(self.wind_directions):
            raise ValueError("other_data must be the same length as the wind data.")

        if not isinstance(other_bins, int):
            if max(other_bins) < max(other_data):
                raise ValueError(
                    f"Bin edges must be greater than the maximum value in the data ({max(other_bins)} < {max(other_data)})."
                )
            if min(other_bins) > min(other_data):
                raise ValueError(
                    f"Bin edges must be less than the minimum value in the data ({min(other_bins)} > {min(other_data)})."
                )

        cat = pd.cut(other_data, bins=other_bins, right=right)
        cat_bins = [(float(i.left), float(i.right)) for i in cat.categories]
        cat_cat = [(float(i.left), float(i.right)) for i in cat]

        d = dict(zip(cat_bins, cat_bins))
        d[cat_bins[0]] = (min(other_data), cat_bins[0][1])

        return [d[i] for i in cat_cat]

    def bin_data(
        self,
        directions: int = _DEFAULT_DIRECTION_COUNT,
        other_data: list[float] = None,
        other_bins: list[float] | int = 11,
        directions_right: bool = True,
        other_right: bool = True,
    ) -> pd.DataFrame:
        """Create categories for wind direction and "other" data. By default "other"
        data is the wind speed associate with the wind direction in this object.

        Args:
            directions (int, optional):
                The number of wind directions to bin values into.
            other_data (list[float], optional):
                An iterable of data to bin.
                Defaults to None, which will use the wind speeds associated
                with the wind directions.
            other_bins (list[float], optional):
                An iterable of bin edges to use for the other data.
                Defaults to None, which will use 11 bins between the min and
                max of the "other" data.
            other_data_name (str, optional):
                A name for the "other" data.
                Defaults to None, which will use the column name of the DataFrame.

        Returns:
            pd.DataFrame:
                A DataFrame containing the wind direction categories and the "other"
                data categories.
        """
        if other_data is None:
            other_data = self.wind_speeds

        idx = self.index
        return pd.concat(
            [
                pd.Series(
                    data=self._bin_direction_data(directions, right=directions_right),
                    name="direction",
                    index=idx,
                ),
                pd.Series(
                    data=self._bin_other_data(
                        other_data, other_bins, right=other_right
                    ),
                    name="other",
                    index=idx,
                ),
            ],
            axis=1,
        )

    # todo - frequency matrix

    # endregion: SAMPLING

    # region: FILTERING

    def filter_by_boolean_mask(self, mask: tuple[bool], source: str = None) -> "Wind":
        """Filter the current object by a boolean mask.

        Args:
            mask (tuple[bool]):
                A boolean mask to filter the current object.
            source_ (str, optional):
                The source of the new object.
                Defaults to None, which will append " (filtered)" to the current source.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        if len(mask) != len(self.ws):
            raise ValueError(
                "The length of the boolean mask must match the length of the current object."
            )

        if sum(mask) == len(self.ws):
            return self

        if len(self.ws.values[mask]) == 0:
            raise ValueError("No data remains within the given boolean filters.")

        if source is None:
            source = f"{self.source} (filtered)"

        return Wind(
            wind_speeds=np.array(self.wind_speeds)[mask].tolist(),
            wind_directions=np.array(self.wind_directions)[mask].tolist(),
            datetimes=np.array(self.datetimes)[mask].tolist(),
            height_above_ground=self.height_above_ground,
            source=source,
        )

    def filter_by_analysis_period(
        self,
        analysis_period: AnalysisPeriod = AnalysisPeriod(),
    ) -> "Wind":
        """Filter the current object by a ladybug AnalysisPeriod object.

        Args:
            analysis_period (AnalysisPeriod):
                An AnalysisPeriod object.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        possible_datetimes = pd.to_datetime(
            analysis_period_to_datetimes(analysis_period)
        )
        lookup = pd.DataFrame(
            {
                "month": possible_datetimes.month,
                "day": possible_datetimes.day,
                "hour": possible_datetimes.hour,
            }
        )
        idx = self.ws.index
        reference = pd.DataFrame(
            {
                "month": idx.month,
                "day": idx.day,
                "hour": idx.hour,
            }
        )
        mask = reference.isin(lookup).all(axis=1)

        return self.filter_by_boolean_mask(
            mask,
            source=f"{self.source} (filtered to {analysis_period_to_string(analysis_period)})",
        )

    def filter_by_time(
        self,
        months: tuple[float] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        days: tuple[float] = (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ),
        hours: tuple[int] = (
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
        ),
        years: tuple[int] = None,
    ) -> "Wind":
        """Filter the current object by month and hour.

        Args:
            months (list[int], optional):
                A list of months.
                Defaults to all possible months.
            days (list[int], optional):
                A list of days.
                Defaults to all possible days.
            hours (list[int], optional):
                A list of hours.
                Defaults to all possible hours.
            years (tuple[int], optional):
                A list of years to include.
                Default to all years.

        Returns:
            Wind:
                A dataset describing historic wind speed and direction relationship.
        """

        # convert the datetimes to a pandas datetime index
        idx = pd.DatetimeIndex(self.datetimes)

        # construct the masks
        year_mask = (
            np.ones_like(idx).astype(bool) if years is None else idx.year.isin(years)
        )
        month_mask = idx.month.isin(months)
        day_mask = idx.day.isin(days)
        hour_mask = idx.hour.isin(hours)
        mask = np.all([year_mask, month_mask, day_mask, hour_mask], axis=0)

        filtered_by = []
        if sum(year_mask) != 0:
            filtered_by.append("year")
        if sum(month_mask) != 0:
            filtered_by.append("month")
        if sum(day_mask) != 0:
            filtered_by.append("day")
        if sum(hour_mask) != 0:
            filtered_by.append("hour")
        filtered_by = ", ".join(filtered_by)
        source = f"{self.source} (filtered {filtered_by})"

        return self.filter_by_boolean_mask(
            mask,
            source=source,
        )

    def filter_by_direction(
        self,
        left_angle: float = 0,
        right_angle: float = 360,
        right: bool = True,
    ) -> "Wind":
        """Filter the current object by wind direction, based on the angle as
        observed from a location.

        Args:
            left_angle (float):
                The left-most angle, to the left of which wind speeds and
                directions will be removed.
                Defaults to 0.
            right_angle (float):
                The right-most angle, to the right of which wind speeds and
                directions will be removed.
                Defaults to 360.
            right (bool, optional):
                Indicates whether the interval includes the rightmost edge or not.
                Defaults to True.

        Return:
            Wind:
                A Wind object!
        """

        if left_angle < 0 or right_angle > 360:
            raise ValueError("Angle limits must be between 0 and 360 degrees.")

        if left_angle == 0 and right_angle == 360:
            return self

        if (left_angle == right_angle) or (left_angle == 360 and right_angle == 0):
            raise ValueError("Angle limits cannot be identical.")

        wd = self.wd.values

        if left_angle > right_angle:
            if right:
                mask = (wd > left_angle) | (wd <= right_angle)
            else:
                mask = (wd >= left_angle) | (wd < right_angle)
        else:
            if right:
                mask = (wd > left_angle) & (wd <= right_angle)
            else:
                mask = (self.wd >= left_angle) & (self.wd < right_angle)

        source = f"{self.source} (filtered by direction {'(' if right else '['}{left_angle}-{right_angle}{']' if right else ')'})"

        return self.filter_by_boolean_mask(mask, source=source)

    def filter_by_speed(
        self, min_speed: float = 0, max_speed: float = np.inf, right: bool = True
    ) -> "Wind":
        """Filter the current object by wind speed, based on given low-high limit values.

        Args:
            min_speed (float):
                The lowest speed to include. Values below this wil be removed.
                Defaults to 0.
            max_speed (float):
                The highest speed to include. Values above this wil be removed.
                Defaults to np.inf.
            right (bool, optional):
                Include values that are exactly the min or max speeds.
                Defaults to True.

        Return:
            Wind:
                A Wind object!
        """

        if min_speed < 0:
            raise ValueError("min_speed cannot be negative.")

        if max_speed <= min_speed:
            raise ValueError("min_speed must be less than max_speed.")

        if min_speed == 0 and np.isinf(max_speed):
            return self

        if right:
            mask = (self.ws > min_speed) & (self.ws <= max_speed)
        else:
            mask = (self.ws >= min_speed) & (self.ws < max_speed)

        source = f"{self.source} (filtered by speed {min_speed}m/s-{max_speed}m/s)"

        return self.filter_by_boolean_mask(mask, source=source)

    # endregion: FILTERING

    # endregion: METHODS
