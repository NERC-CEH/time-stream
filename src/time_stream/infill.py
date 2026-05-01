"""
Time Series Infill Module

This module provides a flexible framework for filling missing values (infilling) in time series data using
Polars and SciPy. Infill methods are implemented as subclasses of ``InfillMethod`` and can be registered
and instantiated by name, class, or instance.

The infill pipeline handles:

- Padding the time series to ensure consistent timestamps
- Identifying gaps and their sizes
- Applying constraints such as maximum gap size and observation intervals
- Delegating to a specific infill method to fill missing values
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, make_interp_spline

from time_stream import Period
from time_stream.exceptions import InfillError, InfillInsufficientValuesError
from time_stream.operation import Operation
from time_stream.utils import check_columns_in_dataframe, gap_size_count, get_date_filter, pad_time


@dataclass(frozen=True)
class InfillCtx:
    """Immutable context passed to infill methods."""

    df: pl.DataFrame
    time_name: str
    periodicity: Period


class InfillMethod(Operation, ABC):
    """Base class for infill methods."""

    def _infilled_column_name(self, infill_column: str) -> str:
        """Return the name of the infilled column."""
        return f"{infill_column}_{self.name}"

    @abstractmethod
    def _fill(self, df: pl.DataFrame, infill_column: str, ctx: InfillCtx) -> pl.DataFrame:
        """Return the Polars dataframe containing infilled data.

        Args:
            df: The DataFrame to infill.
            infill_column: The column to infill.
            ctx: The infill context.

        Returns:
            pl.DataFrame with infilled values
        """
        pass

    def apply(
        self,
        df: pl.DataFrame,
        time_name: str,
        periodicity: Period,
        infill_column: str,
        observation_interval: datetime | tuple[datetime, datetime | None] | None = None,
        max_gap_size: int | None = None,
    ) -> pl.DataFrame:
        """Apply the infill method to the time series data.

        Args:
            df: The Polars DataFrame containing the time series data to infill
            time_name: Name of the time column in the dataframe
            infill_column: The column to infill data within.
            periodicity: Periodicity of the time series
            observation_interval: Optional time interval to limit the infilling to.
            max_gap_size: The maximum size of consecutive null gaps that should be filled. Any gap large_sider than this
                          will not be infilled and will remain as null.
        Returns:
            The infilled time series
        """
        ctx = InfillCtx(df, time_name, periodicity)
        pipeline = InfillMethodPipeline(self, ctx, infill_column, observation_interval, max_gap_size)
        return pipeline.execute()


class InfillMethodPipeline:
    """Encapsulates the logic for the infill pipeline steps."""

    def __init__(
        self,
        infill_method: InfillMethod,
        ctx: InfillCtx,
        column: str,
        observation_interval: datetime | tuple[datetime, datetime | None] | None = None,
        max_gap_size: int | None = None,
    ):
        self.infill_method = infill_method
        self.ctx = ctx
        self.column = column
        self.observation_interval = observation_interval
        self.max_gap_size = max_gap_size

    def execute(self) -> pl.DataFrame:
        """Execute the infill pipeline"""
        self._validate()

        # We need to make sure the data is padded so that missing time steps are filled with nulls
        df = pad_time(self.ctx.df, self.ctx.time_name, self.ctx.periodicity)

        # Calculate sizes of each gap in the time series
        df = gap_size_count(df, self.column)

        # Create a mask determining which values get infilled
        infill_mask = self._infill_mask()

        # Check if there is actually anything to infill
        if df.filter(infill_mask).is_empty():
            # If not, return the original data
            return self.ctx.df

        # Apply the specific infill logic from the child class
        df_infilled = self.infill_method._fill(df, self.column, self.ctx)
        infilled_column = self.infill_method._infilled_column_name(self.column)

        # Limit the infilled data to where the infill mask is True
        df_infilled = df_infilled.with_columns(
            pl.when(infill_mask).then(pl.col(infilled_column)).otherwise(pl.col(self.column)).alias(infilled_column)
        )

        # Do some tidying up of columns, leaving only the original column names
        df_infilled = df_infilled.with_columns(
            pl.col(infilled_column).alias(self.column)  # Rename the infilled column back to the original name
        ).drop([infilled_column, "gap_size"], strict=False)  # Drop the temporary processing columns

        return df_infilled

    def _validate(self) -> None:
        """Carry out validation that the infill method can actually be carried out."""
        if self.ctx.df.is_empty():
            raise InfillError("Cannot perform infilling on an empty DataFrame.")
        check_columns_in_dataframe(self.ctx.df, [self.column, self.ctx.time_name])

    def _infill_mask(self) -> pl.Expr:
        """Create a mask for determining which values in a time series to infill.

        Take into account:
        - Observation interval - constraining the time series to a specific datetime range
        - Maximum gap size - constraining the infilling to gaps of a maximum size
        - Start and end gaps - constraining so nulls at the beginning and end of the series remain null.

        Returns:
            Polars expression that can be used to determine which values to infill (True) or not (False)
        """
        # Base assumption is that any gap can be infilled
        filter_expr = pl.col("gap_size") > 0

        # Check for any gaps
        if self.max_gap_size:
            # If constrained, change the filter to check if there is any missing data with: 0 < gap <= max_gap_size
            filter_expr = pl.col("gap_size").is_between(0, self.max_gap_size, closed="right")

        # Apply observation interval constraint
        if self.observation_interval:
            # Check if these gaps are within the specified observation interval
            filter_expr = filter_expr & get_date_filter(self.ctx.time_name, self.observation_interval)

        # Make a mask to ensure that Nulls at the beginning and end of the series remain null.
        not_null_mask = pl.col(self.column).is_not_null()
        row_idx = pl.arange(0, pl.len())
        filter_expr = filter_expr & row_idx.is_between(
            (row_idx.filter(not_null_mask).min()),  # first True
            (row_idx.filter(not_null_mask).max()),  # last True
        )

        return filter_expr


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

    def _fill(self, df: pl.DataFrame, infill_column: str, ctx: InfillCtx) -> pl.DataFrame:
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
            ctx: The infill context.

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
            raise InfillInsufficientValuesError(
                f"Infill method '{self.name}' requires at least {self.min_points_required} data points, "
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


@InfillMethod.register
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


@InfillMethod.register
class LinearInterpolation(BSplineInterpolation):
    """Linear spline interpolation (Convenience wrapper around B-spline with order=1).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "linear"

    def __init__(self, **kwargs):
        """Initialize linear interpolation."""
        super().__init__(order=1, **kwargs)


@InfillMethod.register
class QuadraticInterpolation(BSplineInterpolation):
    """Quadratic spline interpolation (Convenience wrapper around B-spline with order=2).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "quadratic"

    def __init__(self, **kwargs):
        """Initialize quadratic interpolation."""
        super().__init__(order=2, **kwargs)


@InfillMethod.register
class CubicInterpolation(BSplineInterpolation):
    """Cubic spline interpolation (Convenience wrapper around B-spline with order=3).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.make_interp_spline.html
    """

    name = "cubic"

    def __init__(self, **kwargs):
        """Initialize cubic interpolation."""
        super().__init__(order=3, **kwargs)


@InfillMethod.register
class AkimaInterpolation(ScipyInterpolation):
    """Akima interpolation using scipy (good for avoiding oscillations).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.Akima1DInterpolator.html
    """

    name = "akima"
    min_points_required = 5  # type: ignore[override]

    def _create_interpolator(self, x_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        """Create scipy Akima interpolator."""
        return Akima1DInterpolator(x_valid, y_valid, **self.scipy_kwargs)


@InfillMethod.register
class PchipInterpolation(ScipyInterpolation):
    """PCHIP interpolation using scipy (preserves monotonicity).
    https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.interpolate.PchipInterpolator.html
    """

    name = "pchip"
    min_points_required = 2  # type: ignore[override]

    def _create_interpolator(self, x_valid: np.ndarray, y_valid: np.ndarray) -> Any:
        """Create scipy PCHIP interpolator."""
        return PchipInterpolator(x_valid, y_valid, **self.scipy_kwargs)


@InfillMethod.register
class AltDataStatic(InfillMethod):
    """Infill from an alternative data source and static correction factor."""

    name = "alt_data_static"

    def __init__(self, alt_data_column: str, correction_factor: float = 1.0, alt_df: pl.DataFrame | None = None):
        """Initialize the alternative data infill method.

        Args:
            alt_data_column: The name of the column providing the alternative data.
            correction_factor: An optional, static correction factor to apply to the alternative data.
            alt_df: The DataFrame containing the alternative data.
        """
        self.alt_data_column = alt_data_column
        self.correction_factor = correction_factor
        self.alt_df = alt_df

    def _fill(self, df: pl.DataFrame, infill_column: str, ctx: InfillCtx) -> pl.DataFrame:
        """Fill missing values using data from the alternative column.

        Args:
            df: The DataFrame to infill.
            infill_column: The column to infill.
            ctx: The infill context.

        Returns:
            pl.DataFrame with infilled values.
        """
        if self.alt_df is None:
            check_columns_in_dataframe(df, [self.alt_data_column])
            alt_data_column_name = self.alt_data_column
        else:
            time_column_name = ctx.time_name
            check_columns_in_dataframe(self.alt_df, [time_column_name, self.alt_data_column])
            alt_data_column_name = f"__ALT_DATA__{self.alt_data_column}"
            alt_df = self.alt_df.select([time_column_name, self.alt_data_column]).rename(
                {self.alt_data_column: alt_data_column_name}
            )

            df = df.join(
                alt_df,
                on=time_column_name,
                how="left",
                suffix="_alt",
            )

        infilled = df.with_columns(
            pl.when(pl.col(infill_column).is_null())
            .then(pl.col(alt_data_column_name) * self.correction_factor)
            .otherwise(pl.col(infill_column))
            .alias(self._infilled_column_name(infill_column))
        )

        if self.alt_df is not None:
            infilled = infilled.drop(alt_data_column_name)

        return infilled


@InfillMethod.register
class AltDataDynamic(InfillMethod):
    """Infill from an alternative data source and dynamic correction factor."""

    name = "alt_data_dynamic"

    def __init__(
        self,
        alt_data_column: str,
        window_size: str | Period,
        alt_df: pl.DataFrame | None = None,
        min_threshold: int = 0,
        max_threshold: int | None = None,
    ):
        """Initialize the alternative data infill method.

        Args:
            alt_data_column: The name of the column providing the alternative data.
            alt_df: The DataFrame containing the alternative data.
        """
        self.alt_data_column = alt_data_column
        self.alt_df = alt_df
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def _fill(
        self,
        df: pl.DataFrame,
        infill_column: str,
        ctx: InfillCtx,
    ) -> pl.DataFrame:
        """Fill missing values using data from the alternative column.

        Args:
            df: The DataFrame to infill.
            infill_column: The column to infill.
            ctx: The infill context.

        Returns:
            pl.DataFrame with infilled values.
        """
        # Define time column name
        time_column_name = ctx.time_name

        # Ensure window_size is a valid Period
        periodicity = ctx.periodicity
        if isinstance(self.window_size, str):
            self.window_size = Period.of_iso_duration(self.window_size)
        if self.window_size < periodicity:
            raise ValueError("Window size must be greater than periodicity")
        window_duration = self.window_size.timedelta
        if window_duration is None:
            raise ValueError(
                "Window size must be given in days, hours, seconds...Cannot resolve month or year to timedelta"
            )

        # Join original and alternative dataframes if the latter exists
        if self.alt_df is None:
            check_columns_in_dataframe(df, [self.alt_data_column])
            alt_data_column_name = self.alt_data_column
        else:
            check_columns_in_dataframe(self.alt_df, [time_column_name, self.alt_data_column])
            alt_data_column_name = f"__ALT_DATA__{self.alt_data_column}"
            alt_df = self.alt_df.select([time_column_name, self.alt_data_column]).rename(
                {self.alt_data_column: alt_data_column_name}
            )

            df = df.join(
                alt_df,
                on=time_column_name,
                how="left",
                suffix="_alt",
            )

        # Identify gaps in original dataset
        null_mask = pl.col(infill_column).is_null()
        gap_id = (null_mask != null_mask.shift(1, fill_value=False)).cum_sum()
        df = df.with_columns(gap_id.alias("gap_id"))

        # Find start and end times of gaps
        gap_bounds = (
            df.filter(null_mask)
            .group_by("gap_id")
            .agg(
                pl.min(time_column_name).alias("gap_start"),
                pl.max(time_column_name).alias("gap_end"),
            )
        )

        # Compute correction factors for each gap
        factors = {}
        for row in gap_bounds.iter_rows(named=True):
            gid = row["gap_id"]
            gap_start = row["gap_start"]
            gap_end = row["gap_end"]

            # Define window either side of gap
            # Filter out any null data from either original or alt datasets within window.
            window_df = df.filter(
                (pl.col("gap_id") == gid)
                & (pl.col(time_column_name) >= gap_start - window_duration)
                & (pl.col(time_column_name) <= gap_end + window_duration)
                & (~pl.col(infill_column).is_null())
                & (~pl.col(alt_data_column_name).is_null())
            ).sort(time_column_name)

            cf = self._compute_correction_factor_for_gap(
                window_df,
                time_column_name=time_column_name,
                infill_column=infill_column,
                alt_data_column_name=alt_data_column_name,
                gap_start=gap_start,
                gap_end=gap_end,
            )

            if cf is not None:
                factors[gid] = cf

        # Attach correction factors
        if factors:
            cf_df = pl.DataFrame({"gap_id": list(factors.keys()), "correction_factor": list(factors.values())})
            df = df.join(cf_df, on="gap_id", how="left")
        else:
            df = df.with_columns(pl.lit(None).alias("correction_factor"))

        # Fill gaps
        infilled = df.with_columns(
            pl.when(pl.col(infill_column).is_null() & pl.col("correction_factor").is_not_null())
            .then(pl.col(alt_data_column_name) * pl.col("correction_factor"))
            .otherwise(pl.col(infill_column))
            .alias(infill_column)
        )

        # Cleanup
        if self.alt_df is not None:
            infilled = infilled.drop(alt_data_column_name, "gap_id", "correction_factor")

        return infilled

    def _compute_correction_factor_for_gap(
        self,
        window_df: pl.DataFrame,
        time_column_name: str,
        infill_column: str,
        alt_data_column_name: str,
        gap_start: pl.Datetime,
        gap_end: pl.Datetime,
    ) -> float | None:
        """Compute correction factor for a single gap using hierarchical rules.
        Default: Use all data within window. If none available, return None.
        If max_threshold is provided, use all data available, up to that threshold with the hierarchy:
        1. Same number of datapoints either side of gap, up to max_threshold/2 each.
        2. Different numbers of data points on either side of gap, up to max_threshold total datapoints.
        3. All datapoints are on one side of gap only, up to max_threshold total datapoints.

        Args:
            window_df: a segment of the combined original df and alt_data.
                       the segment contains a window of data either side of the gap, with any null data removed.
                       window_df is guaranteed to have at least the min_threshold of data points, and not be empty.
            time_column_name: name of time column, from ctx.time_column_name
            infill_column: name of column in original dataset with missing data to be infilled
            alt_data_column_name: name of column in alternative dataset to use to infill missing data
            gap_start: timestamp indicating the start of the gap in the original dataset
            gap_end: timestamp indicating the end of the gap in the original dataset

        Returns:
            correction_factor (float) or None if gap cannot be filled
        """

        # Ensure enough data is available before starting calculation
        if window_df.is_empty() or (window_df.height < self.min_threshold):
            return None

        # If max_threshold is None, use all available data in the window
        if self.max_threshold is None:
            alt_sum = window_df[alt_data_column_name].sum()
            if alt_sum != 0:
                return window_df[infill_column].sum() / alt_sum

        # If max_threshold is not None, use up to max_threshold total number of datapoints within window
        # First preference: same number of datapoints either side of gap, up to max_threshold/2 each.
        # Second preference: different numbers of data points on either side of gap,
        #                    up to max_threshold total datapoints.
        # Third preference: all datapoints are on one side of gap only, up to max_threshold total datapoints.
        elif self.max_threshold is not None:
            before_gap_df = window_df.filter(pl.col(time_column_name) < pl.lit(gap_start))
            after_gap_df = window_df.filter(pl.col(time_column_name) > pl.lit(gap_end))

            # Same number of datapoints either side of gap
            if before_gap_df.height >= math.ceil(self.min_threshold / 2) and after_gap_df.height >= math.ceil(
                self.min_threshold / 2
            ):
                # Use at most round(max_threshold/2) datapoints each side of the gap
                datapoints_on_each_side = min(
                    math.ceil(self.max_threshold / 2), before_gap_df.height, after_gap_df.height
                )
                alt_sum = (
                    # Using head/tail ensures data nearest gap is selected.
                    before_gap_df.tail(datapoints_on_each_side)[alt_data_column_name].sum()
                    + after_gap_df.head(datapoints_on_each_side)[alt_data_column_name].sum()
                )
                if alt_sum != 0:
                    return (
                        # Using head/tail ensures data nearest gap is selected.
                        before_gap_df.tail(datapoints_on_each_side)[infill_column].sum()
                        + after_gap_df.head(datapoints_on_each_side)[infill_column].sum()
                    ) / alt_sum

            # Different number of datapoints either side of the gap
            else:
                # Identify which side is smaller/larger
                if before_gap_df.height < after_gap_df.height:
                    small_side = before_gap_df
                    large_side = after_gap_df
                    large_side_slice = large_side.head
                else:
                    small_side = after_gap_df
                    large_side = before_gap_df
                    large_side_slice = large_side.tail

                # Use at most self.max_threshold datapoints in total
                remaining = self.max_threshold - small_side.height
                take_from_large_side = min(remaining, large_side.height)
                alt_sum = (
                    small_side[alt_data_column_name].sum()
                    + large_side_slice(take_from_large_side)[alt_data_column_name].sum()
                )
                if alt_sum != 0:
                    return (
                        small_side[infill_column].sum() + large_side_slice(take_from_large_side)[infill_column].sum()
                    ) / alt_sum

        ############################################
        # Otherwise use different infill data/method
        ############################################

        return None
