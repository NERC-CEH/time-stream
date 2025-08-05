from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Tuple, Type, Union

import numpy as np
import polars as pl
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, make_interp_spline

from time_stream import TimeSeries
from time_stream.utils import gap_size_count, get_date_filter, pad_time

# Registry for built-in infill methods
_INFILL_REGISTRY = {}


def register_infill_method(cls: Type["InfillMethod"]) -> Type["InfillMethod"]:
    """Decorator to register infill method classes using their name attribute.

    Args:
        cls: The infill class to register.

    Returns:
        The decorated class.
    """
    _INFILL_REGISTRY[cls.name] = cls
    return cls


class InfillMethod(ABC):
    """Base class for infill methods."""

    _ts = None

    @property
    def ts(self) -> TimeSeries:
        if self._ts is None:
            raise AttributeError("TimeSeries has not been initialised for this infill method.")
        return self._ts

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the infill method."""
        pass

    def _infilled_column_name(self, infill_column: str) -> str:
        """Return the name of the infilled column."""
        return f"{infill_column}_{self.name}"

    @abstractmethod
    def _fill(self, df: pl.DataFrame, infill_column: str) -> pl.DataFrame:
        """Return the Polars dataframe containing infilled data.

        Args:
            df: The DataFrame to infill.
            infill_column: The column to infill.

        Returns:
            pl.DataFrame with infilled values
        """
        pass

    @classmethod
    def get(cls, method: Union[str, Type["InfillMethod"]], **kwargs) -> "InfillMethod":
        """Factory method to get an infill method class instance from string names or class type.

        Args:
            method: The infill method specification, which can be:
                - A string name: e.g. "linear_interpolation"
                - A class type: LinearInterpolation
                - An instance: LinearInterpolation(), or any InfillMethod instance
            **kwargs: Parameters specific to the infill method, used to initialise the class object.
                      Ignored if method is already an instance of the class.

        Returns:
            An instance of the appropriate InfillMethod subclass.

        Raises:
            KeyError: If a string name is not registered as an infill method.
            TypeError: If the input type is not supported or a class doesn't inherit from InfillMethod.
        """
        # If it's already an instance, return it
        if isinstance(method, InfillMethod):
            return method

        # If it's a string, look it up in the registry
        if isinstance(method, str):
            try:
                return _INFILL_REGISTRY[method](**kwargs)
            except KeyError:
                raise KeyError(f"Unknown infill method '{method}'.")

        # If it's a class, check the subclass type and return
        elif isinstance(method, type):
            if issubclass(method, InfillMethod):
                return method(**kwargs)  # type: ignore[misc]
            else:
                raise TypeError(f"Infill method class {method.__name__} must inherit from InfillMethod")

        else:
            raise TypeError(f"Infill method must be a string or an InfillMethod class. Got {type(method).__name__}")

    @classmethod
    def _anything_to_infill(
            cls,
            df: pl.DataFrame,
            time_name: str,
            infill_column: str,
            observation_interval: Optional[datetime | Tuple[datetime, datetime | None]] = None,
            max_gap_size: Optional[int] = None,
    ):
        df = gap_size_count(df, infill_column)

        # Default is that there is no data to infill
        filter_expr = pl.lit(False)

        if max_gap_size:
            # Check if there is any missing data in gaps with: 0 < gap <= max_gap_size
            filter_expr = pl.col("gap_size").is_between(0, max_gap_size, closed="right")
        if observation_interval:
            # Check if these gaps are within the specified observation interval
            filter_expr = filter_expr & get_date_filter(time_name, observation_interval)

        # If anything left in the dataframe using the filter, then these are the data points that need infilling
        df = df.filter(filter_expr)
        return not df.is_empty()

    def apply(
        self,
        ts: TimeSeries,
        infill_column: str,
        observation_interval: Optional[datetime | Tuple[datetime, datetime | None]] = None,
        max_gap_size: Optional[int] = None,
    ) -> "TimeSeries":
        """Apply the infill method to the TimeSeries.

        Args:
            ts: The TimeSeries to check.
            infill_column: The column to infill data within.
            observation_interval: Optional time interval to limit the infilling to.
            max_gap_size: The maximum size of consecutive null gaps that should be filled. Any gap larger than this
                          will not be infilled and will remain as null.
        Returns:
            TimeSeries: The infilled time series
        """
        # Validate column exists
        if infill_column not in ts.columns:
            raise KeyError(f"Infill column '{infill_column}' not found in TimeSeries.")

        # Set the timeseries property
        self._ts = ts

        # We need to make sure the data is padded so that missing time steps are filled with nulls
        df = pad_time(ts.df, ts.time_name, ts.periodicity)

        # Check if there is actually anything to infill
        if not self._anything_to_infill(df, ts.time_name, infill_column, observation_interval, max_gap_size):
            # If not, return the original time series
            return ts

        # Apply the specific infill logic from the child class
        df_infilled = self._fill(df, infill_column)
        infilled_column = self._infilled_column_name(infill_column)

        # Apply gap size limitation if specified
        if max_gap_size:
            # Count the size of gaps in the data
            df_infilled = gap_size_count(df_infilled, infill_column)

            # Limit the infilled data to where the gap size is less than the user specified limit
            df_infilled = df_infilled.with_columns(
                pl.when(pl.col("gap_size") <= max_gap_size)
                .then(pl.col(infilled_column))
                .otherwise(None)
                .alias(infilled_column)
            )

        # Apply observation interval filter if specified
        if observation_interval:
            date_filter = get_date_filter(ts.time_name, observation_interval)

            df_infilled = df_infilled.with_columns(
                pl.when(date_filter).then(pl.col(infilled_column)).otherwise(infill_column).alias(infilled_column)
            )

        # Do some tidying up of columns, leaving only the original column names
        df_infilled = df_infilled.with_columns(
            pl.col(infilled_column).alias(infill_column)  # Rename the infilled column back to the original name
        ).drop([infilled_column, "gap_size"], strict=False)  # Drop the temporary processing columns

        # Create result TimeSeries
        #   Need to do this as the time column might have changed due to the padding/adding of infilled rows.
        return TimeSeries(
            df=df_infilled,
            time_name=self.ts.time_name,
            resolution=ts.resolution,
            periodicity=ts.periodicity,
            column_metadata={name: col.metadata() for name, col in ts.columns.items()},
            metadata=ts._metadata,
            supplementary_columns=list(ts.supplementary_columns.keys()),
            flag_systems=ts.flag_systems,
            flag_columns={name: col.flag_system for name, col in ts.flag_columns.items()},
            pad=True,
        )


class ScipyInterpolation(InfillMethod, ABC):
    """Base class for scipy-based interpolation methods."""

    def __init__(self, **kwargs):
        """Initialize a scipy interpolation method.

        Args:
            **kwargs: Additional parameters passed to scipy interpolator method.
        """
        self.scipy_kwargs = kwargs

    @abstractmethod
    def _create_interpolator(self, x_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        """Create the scipy interpolator object.

        Args:
            x_valid: Array of row indices (0, 1, 2, ...) corresponding to non-null data points.
                    For example, if rows 0, 2, 5 have valid data, x_valid = [0, 2, 5].
            y_valid: Array of actual data values at those row indices.

        Returns:
            Scipy interpolator object.

        Raises:
            ValueError: If insufficient data for this interpolation method.

        Example:
            If original data is [10.5, NaN, 12.3, NaN, NaN, 9.8]:
            - x_valid = [0, 2, 5] (row indices of non-null values)
            - y_valid = [10.5, 12.3, 9.8] (the actual non-null values)
            - The interpolator will estimate values for indices 1, 3, 4
        """
        pass

    @property
    @abstractmethod
    def min_points_required(self) -> int:
        """Minimum number of data points required for this interpolation method."""
        pass

    def _fill(self, df: pl.DataFrame, infill_column: str) -> pl.DataFrame:
        """Apply scipy interpolation to fill missing values in the specified column.

        This method handles the common scipy interpolation workflow:
        1. Converts data to numpy arrays for scipy compatibility
        2. Identifies valid (non-null) data points for interpolation
        3. Validates that sufficient data points exist for interpolation method
        4. Creates and applies the specific scipy interpolator
        5. Handles edge cases like infinite values in the interpolated result
        6. Returns the DataFrame with a new column containing interpolated values

        Args:
            df: The DataFrame to infill.
            infill_column: The column to infill.

        Returns:
            pl.DataFrame with infilled values
        """
        # Convert to numpy
        values = df[infill_column].to_numpy()
        x = np.arange(len(values))

        # Find non-null points
        mask = ~np.isnan(values)
        n_valid = np.sum(mask)

        # Check if we have enough points
        if n_valid < self.min_points_required:
            raise ValueError(
                f"{self.name} requires at least {self.min_points_required} data points, "
                f"but only {n_valid} valid points found."
            )

        x_valid = x[mask]
        y_valid = values[mask]

        # Create the specific interpolator
        interpolator = self._create_interpolator(x_valid, y_valid)

        # Apply interpolation
        interpolated = interpolator(x)

        # Handle any remaining NaNs or infinities
        interpolated = np.where(np.isfinite(interpolated), interpolated, np.nan)

        return df.with_columns(pl.Series(self._infilled_column_name(infill_column), interpolated))


@register_infill_method
class BSplineInterpolation(ScipyInterpolation):
    """B-spline interpolation using scipy make_interp_spline with configurable order.
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "bspline"

    def __init__(self, order: int, **kwargs):
        """Initialize B-spline interpolation.

        Args:
            order: Order of the B-spline (1-5, where 3=cubic, 2=quadratic, 1=linear).
            **kwargs: Additional scipy parameters for the `make_interp_spline` method.
        """
        super().__init__(**kwargs)
        self.order = order

    @property
    def min_points_required(self) -> int:
        """B-spline needs at least order+1 points."""
        return self.order + 1

    def _create_interpolator(self, x_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        """Create scipy B-spline interpolator."""
        return make_interp_spline(x_valid, y_valid, k=self.order, **self.scipy_kwargs)


@register_infill_method
class LinearInterpolation(BSplineInterpolation):
    """Linear spline interpolation (Convenience wrapper around B-spline with order=1).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "linear"

    def __init__(self, **kwargs):
        """Initialize linear interpolation."""
        super().__init__(order=1, **kwargs)


@register_infill_method
class QuadraticInterpolation(BSplineInterpolation):
    """Quadratic spline interpolation (Convenience wrapper around B-spline with order=2).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "quadratic"

    def __init__(self, **kwargs):
        """Initialize quadratic interpolation."""
        super().__init__(order=2, **kwargs)


@register_infill_method
class CubicInterpolation(BSplineInterpolation):
    """Cubic spline interpolation (Convenience wrapper around B-spline with order=3).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "cubic"

    def __init__(self, **kwargs):
        """Initialize cubic interpolation."""
        super().__init__(order=3, **kwargs)


@register_infill_method
class AkimaInterpolation(ScipyInterpolation):
    """Akima interpolation using scipy (good for avoiding oscillations).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.Akima1DInterpolator.html
    """

    name = "akima"
    min_points_required = 5

    def _create_interpolator(self, x_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        """Create scipy Akima interpolator."""
        return Akima1DInterpolator(x_valid, y_valid, **self.scipy_kwargs)


@register_infill_method
class PchipInterpolation(ScipyInterpolation):
    """PCHIP interpolation using scipy (preserves monotonicity).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.PchipInterpolator.html
    """

    name = "pchip"
    min_points_required = 2

    def _create_interpolator(self, x_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        """Create scipy PCHIP interpolator."""
        return PchipInterpolator(x_valid, y_valid, **self.scipy_kwargs)
