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

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator, make_interp_spline

from time_stream import Period
from time_stream.exceptions import InfillError, InfillInsufficientValuesError
from time_stream.operation import Operation
from time_stream.utils import check_columns_in_dataframe, gap_size_count, get_date_filter, pad_time

logger = logging.getLogger(__name__)


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
            max_gap_size: The maximum size of consecutive null gaps that should be filled. Any gap larger than this
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
class AltData(InfillMethod):
    """
    Infills missing values using an alternative data source and optional correction factor.
    The alternative data corresponding to the missing interval is scaled by the correction
    factor to produce the infilled values.
    """

    name = "alt_data"

    def __init__(self, alt_data_column: str, correction_factor: float = 1.0, alt_df: pl.DataFrame | None = None):
        """Initialize the alternative data infill method.

        Args:
            alt_data_column: The name of the column providing the alternative data.
            correction_factor: An optional correction factor to apply to the alternative data.
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
    """
    Infills missing values using an alternative data source and a dynamic
    correction factor derived from surrounding data.

    For each contiguous gap in the original dataset, a time window is defined
    around the gap. A correction factor is computed as the ratio of the sum of
    the original data to the sum of the alternative data within this window.
    The alternative data corresponding to the missing interval is scaled by the
    correction factor to produce the infilled values.

    The method defaults to using data on both sides of the gap.
    If window_side is specified as "left" or "right", then only data left or right of the gap will be used.
    """

    name = "alt_data_dynamic"

    def __init__(
        self,
        alt_data_column: str,
        window_size: str | Period | timedelta,
        alt_df: pl.DataFrame | None = None,
        min_threshold: int = 0,
        max_threshold: int | None = None,
        window_side: Literal["left", "right", "both"] = "both",
    ):
        """Initialize the alternative data infill method.

        Args:
            alt_data_column: Name of the column providing the alternative data.
            window_size: Time window around each gap used to calculate the correction factor.
                Accepts an ISO duration string, Period, or timedelta.
            alt_df: Optional separate DataFrame containing the alternative data. If None,
                alt_data_column must exist in the DataFrame passed to the infill method.
            min_threshold: Minimum number of data points required in the window to calculate
                a correction factor. Gaps with windows that have fewer points than the min_threshold are not infilled.
            max_threshold: Maximum number of data points to use per window. Points closest
                to the gap are used first.
            window_side: Which side of each gap to use for the window. Defaults to "both".
                "left" uses only data before the gap; "right" uses only data after.
        """
        if max_threshold is not None:
            if min_threshold > max_threshold:
                raise ValueError(f"max_threshold must be greater than min_threshold ({min_threshold}).")
            if max_threshold == 0:
                raise ValueError("max_threshold must be greater than zero.")

        self.alt_data_column = alt_data_column
        self.alt_df = alt_df
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.window_side = window_side

    def _fill(
        self,
        df: pl.DataFrame,
        infill_column: str,
        ctx: InfillCtx,
    ) -> pl.DataFrame:
        """Fill missing values using data from the alternative column.

        Args:
            df: The DataFrame to infill.
            infill_column: Name of the column to infill.
            ctx: The infill context.

        Returns:
            DataFrame with an additional infilled column containing the corrected values.
        """
        time_column_name = ctx.time_name
        window_duration = self._window_duration(ctx)

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
        gap_id_column_name = f"__GAP_ID__{infill_column}"
        df = df.with_columns(gap_id.alias(gap_id_column_name))

        # Find start and end times of gaps
        gap_bounds = (
            df.filter(null_mask)
            .group_by(gap_id_column_name)
            .agg(
                pl.min(time_column_name).alias("__GAP_START__"),
                pl.max(time_column_name).alias("__GAP_END__"),
            )
        )

        # Filter out all null values from both the original and alternative dataset.
        filtered_df = df.filter(pl.col(infill_column).is_not_null() & pl.col(alt_data_column_name).is_not_null())

        # Build windowed source data - must not overwrite df
        windowed_df = filtered_df.drop(gap_id_column_name).join_where(
            gap_bounds,
            pl.col(time_column_name) >= pl.col("__GAP_START__") - window_duration,
            pl.col(time_column_name) <= pl.col("__GAP_END__") + window_duration,
        )

        # Define windows either side of each gap
        windowed_df = self._build_windowed_df(
            windowed_df,
            time_column_name,
            gap_id_column_name,
        )

        # Attach correction factors
        cf_column_name = f"__CF__{infill_column}"
        cf_df = self._build_correction_factors(
            windowed_df,
            gap_id_column_name,
            infill_column,
            alt_data_column_name,
            cf_column_name,
        )

        if cf_df is not None:
            df = df.join(cf_df, on=gap_id_column_name, how="left")
        else:
            df = df.with_columns(pl.lit(None).alias(cf_column_name))

        # Fill gaps
        infilled = df.with_columns(
            pl.when(pl.col(infill_column).is_null() & pl.col(cf_column_name).is_not_null())
            .then(pl.col(alt_data_column_name) * pl.col(cf_column_name))  # null if alt_data is null
            .otherwise(pl.col(infill_column))
            .alias(self._infilled_column_name(infill_column))
        )

        # Cleanup
        if self.alt_df is not None:
            infilled = infilled.drop(alt_data_column_name)
        infilled = infilled.drop([gap_id_column_name, cf_column_name])

        return infilled

    def _window_duration(self, ctx: InfillCtx) -> timedelta:
        """Resolve self.window_size to a timedelta and validate it against the data periodicity.

        Args:
            ctx: The infill context, used to obtain the data periodicity.

        Returns:
            The window duration as a timedelta.

        Raises:
            ValueError: If window_size cannot be resolved to a timedelta (e.g. a month or year
                Period), if the window is smaller than the data periodicity, or if the window
                is too small to satisfy min_threshold.
        """
        window_size = self.window_size
        # If window_size is a string, convert to Period type
        if isinstance(window_size, str):
            window_size = Period.of_iso_duration(window_size)

        # Convert window_size to timedelta if not already
        window_duration = window_size.timedelta if isinstance(window_size, Period) else window_size

        # Check window_size gives a valid timedelta
        if window_duration is None:
            raise ValueError(
                "Window size must be given in days, hours or seconds. Cannot resolve month or year to timedelta."
            )

        periodicity = ctx.periodicity
        if periodicity.timedelta is None:
            return window_duration

        # window_duration must be greater than or equal to the periodicity of the data.
        if window_duration < periodicity.timedelta:
            raise ValueError("Window size must be greater than periodicity")

        # Check the window duration contains min_threshold number of datapoints
        factor = 1 if self.window_side in ["left", "right"] else 2
        if window_duration * factor < periodicity.timedelta * self.min_threshold:
            raise ValueError(
                f"Windows must contain at least min_threshold {self.min_threshold} of data points. "
                "Reduce the min_threshold or increase the window size."
            )

        return window_duration

    def _build_windowed_df(
        self,
        windowed_df: pl.DataFrame,
        time_column_name: str,
        gap_id_column_name: str,
    ) -> pl.DataFrame | None:
        """Builds a filtered DataFrame containing only the window data around each gap.

        Optionally restricts to one side of each gap, and applies
        max and min threshold filtering.

        Args:
            df: DataFrame with gap IDs and gap bounds already joined.
            time_column_name: Name of the time column.
            gap_id_column_name: Name of the gap ID column.

        Returns:
            Filtered DataFrame with window data for each gap, or None if no data remains
            after filtering.
        """
        # Use left or right window only, if window_side is specified.
        windowed_df = self._filter_side(windowed_df, time_column_name)

        # Trim data if a max_threshold is specified.
        if self.max_threshold is not None:
            windowed_df = self._apply_max_threshold(
                self.max_threshold,
                windowed_df,
                gap_id_column_name,
                time_column_name,
            )

        # Check there is enough data to meet the min_threshold if specified.
        if self.min_threshold > 0:
            windowed_df = self._apply_min_threshold(windowed_df, gap_id_column_name)

        # If no data is left after filtering, no windows can be defined.
        if windowed_df.is_empty():
            logger.warning("Windows around each gap are empty. No gaps will be infilled.")
            return None
        else:
            return windowed_df

    def _filter_side(
        self,
        windowed_df: pl.DataFrame,
        time_column_name: str,
    ) -> pl.DataFrame:
        """Restrict the window to one side of each gap based on self.window_side.

        Args:
            windowed_df: Window DataFrame.
            time_column_name: Name of the time column.

        Returns:
            DataFrame with rows from the excluded side of each gap removed.
            Unchanged if self.window_side is "both".
        """
        if self.window_side == "left":
            windowed_df = windowed_df.filter(pl.col(time_column_name) < pl.col("__GAP_START__"))
        elif self.window_side == "right":
            windowed_df = windowed_df.filter(pl.col(time_column_name) > pl.col("__GAP_END__"))
        return windowed_df

    def _apply_max_threshold(
        self,
        max_threshold: int,
        windowed_df: pl.DataFrame,
        gap_id_column_name: str,
        time_column_name: str,
    ) -> pl.DataFrame:
        """Trim each gap's window to at most max_threshold rows, keeping the closest rows to the gap.

        When both sides of a gap have sufficient data, trims symmetrically so each side
        contributes at most floor(max_threshold / 2) rows. When one side is smaller,
        keeps all rows from that side and fills the remainder from the other side.

        Args:
            max_threshold: Maximum number of data points to use per window.
            windowed_df: Window DataFrame with gap ID and time columns.
            gap_id_column_name: Name of the gap ID column.
            time_column_name: Name of the time column.

        Returns:
            DataFrame with each gap's window trimmed to at most max_threshold rows.
        """
        # Label which rows occur before each gap
        windowed_df = windowed_df.with_columns(
            (pl.col(time_column_name) < pl.col("__GAP_START__")).alias("__IS_BEFORE__")
        )
        # Count rows in window on each side of each gap
        windowed_df = windowed_df.with_columns(
            pl.len().over([gap_id_column_name, "__IS_BEFORE__"]).alias("__SIDE_COUNT__")
        )

        # Count total rows before, after, and total in window around each gap
        windowed_df = windowed_df.with_columns(
            [
                pl.when(pl.col("__IS_BEFORE__"))
                .then(pl.col("__SIDE_COUNT__"))
                .otherwise(0)
                .max()
                .over(gap_id_column_name)
                .alias("__BEFORE_COUNT__"),
                pl.when(~pl.col("__IS_BEFORE__"))
                .then(pl.col("__SIDE_COUNT__"))
                .otherwise(0)
                .max()
                .over(gap_id_column_name)
                .alias("__AFTER_COUNT__"),
                pl.len().over(gap_id_column_name).alias("__TOTAL_COUNT__"),
            ]
        )

        # Rank closest
        windowed_df = windowed_df.with_columns(
            pl.when(pl.col("__IS_BEFORE__"))
            .then(pl.col(time_column_name).rank("ordinal", descending=True).over([gap_id_column_name, "__IS_BEFORE__"]))
            .otherwise(
                pl.col(time_column_name).rank("ordinal", descending=False).over([gap_id_column_name, "__IS_BEFORE__"])
            )
            .alias("__RANK__")
        )

        # Track which windows around each gap are symmetric/asymmetric
        windowed_df = windowed_df.with_columns(
            [
                # Track which side is largest/smallest
                # In a tie, before count wins
                (pl.col("__BEFORE_COUNT__") > pl.col("__AFTER_COUNT__")).alias("__AFTER_IS_SMALLER__"),
                # If both sides have at least half the min threshold,
                # then use same number of datapoints on either side of gap
                (
                    (pl.col("__BEFORE_COUNT__") >= math.ceil(self.min_threshold / 2))
                    & (pl.col("__AFTER_COUNT__") >= math.ceil(self.min_threshold / 2))
                    & (max_threshold >= 2)
                ).alias("__SYMMETRIC__"),
            ]
        )

        # Trim rows such that only the closest datapoints to each gap,
        # up to the max_threshold number of datapoints in a window around each gap are used.
        windowed_df = windowed_df.with_columns(
            # No trimming needed
            pl.when(pl.col("__TOTAL_COUNT__") <= max_threshold)
            .then(pl.col("__SIDE_COUNT__"))
            # Symmetric: only use up to half of max_threshold number of datapoints on each side of gap
            .when(pl.col("__SYMMETRIC__"))
            .then(pl.lit(math.floor(max_threshold / 2)))  # Never zero, symmetric filter ensures max_threshold >=2
            # If not enough data on each side of gap for windows to be same size,
            # keep all data in smaller window and trim larger window such that
            # the total number of datapoints across windows is up to the max_threshold.
            .when((pl.col("__IS_BEFORE__") != pl.col("__AFTER_IS_SMALLER__")))
            .then(pl.min_horizontal(pl.col("__SIDE_COUNT__"), max_threshold))
            .otherwise(pl.lit(max_threshold) - pl.min_horizontal("__BEFORE_COUNT__", "__AFTER_COUNT__", max_threshold))
            .alias("__FINAL_COUNT__")
        )

        windowed_df = windowed_df.filter(pl.col("__RANK__") <= pl.col("__FINAL_COUNT__"))

        return windowed_df.drop(
            [
                "__IS_BEFORE__",
                "__SIDE_COUNT__",
                "__BEFORE_COUNT__",
                "__AFTER_COUNT__",
                "__TOTAL_COUNT__",
                "__SYMMETRIC__",
                "__RANK__",
                "__AFTER_IS_SMALLER__",
                "__FINAL_COUNT__",
            ]
        )

    def _apply_min_threshold(
        self,
        windowed_df: pl.DataFrame,
        gap_id_column_name: str,
    ) -> pl.DataFrame:
        """Remove gaps whose window contains fewer than self.min_threshold rows.

        Logs a warning listing the gap IDs that are dropped.

        Args:
            windowed_df: Window DataFrame with gap ID column.
            gap_id_column_name: Name of the gap ID column.

        Returns:
            DataFrame with gaps removed whose windows contain below the minimum threshold datapoints.
        """
        window_sizes = windowed_df.group_by(gap_id_column_name).agg(pl.len().alias("__COUNT__"))
        gaps_with_window_below_threshold = window_sizes.filter(pl.col("__COUNT__") < self.min_threshold)[
            gap_id_column_name
        ].to_list()
        if len(gaps_with_window_below_threshold) > 0:
            logger.warning(
                f"gap(s): {gaps_with_window_below_threshold} cannot be filled, "
                f"window size is below min threshold ({self.min_threshold}).",
            )
        valid_ids = window_sizes.filter(pl.col("__COUNT__") >= self.min_threshold)[gap_id_column_name]
        return windowed_df.filter(pl.col(gap_id_column_name).is_in(valid_ids))

    def _build_correction_factors(
        self,
        windowed_df: pl.DataFrame | None,
        gap_id_column_name: str,
        infill_column: str,
        alt_data_column_name: str,
        cf_column_name: str,
    ) -> pl.DataFrame | None:
        """Compute a correction factor per gap as sum(infill) / sum(alt_data) over the window.

        Logs a warning for any gap where the alternative data sums to zero, as no
        correction factor can be computed for those gaps.

        Args:
            windowed_df: Window DataFrame per gap, or None if no window data is available.
            gap_id_column_name: Name of the gap ID column.
            infill_column: Name of the infill column.
            alt_data_column_name: Name of the alternative data column.
            cf_column_name: Name to give the correction factor column in the output.

        Returns:
            DataFrame with one row per gap containing the correction factor, or None if
            windowed_df is None or no correction factors could be computed.
        """
        if windowed_df is None:
            return None

        infill_sum_column_name = f"__SUM__{infill_column}"
        alt_sum_column_name = f"__ALT_SUM__{alt_data_column_name}"

        cf_df = windowed_df.group_by(gap_id_column_name).agg(
            pl.col(infill_column).sum().alias(infill_sum_column_name),
            pl.col(alt_data_column_name).sum().alias(alt_sum_column_name),
        )

        # list gaps where the alt_data sum is zero.
        zero_alt_sum_gaps = cf_df.filter(pl.col(alt_sum_column_name) == 0)[gap_id_column_name].to_list()
        if len(zero_alt_sum_gaps) > 0:
            logger.warning("alt_sum is zero for gap(s) %s and will not be infilled.", zero_alt_sum_gaps)

        cf_df = cf_df.with_columns(
            pl.when(pl.col(alt_sum_column_name) != 0)
            .then(pl.col(infill_sum_column_name) / pl.col(alt_sum_column_name))
            .otherwise(None)
            .alias(cf_column_name)
        ).drop([infill_sum_column_name, alt_sum_column_name])

        return cf_df if not cf_df.is_empty() else None
