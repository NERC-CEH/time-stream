"""
Time Series Quality Control (QC) Module

This module provides a framework for applying quality control checks to TimeFrame using Polars. QC checks are
implemented as subclasses of ``QCCheck`` and can be registered and instantiated by name, class, or instance.

It supports various QC checks including:

- ComparisonCheck: Compare values against thresholds or sets.
- RangeCheck: Verify values fall within or outside a given range.
- TimeRangeCheck: Apply range checks directly to the time column.
- SpikeCheck: Detect sudden spikes based on differences with neighbors.
- FlatLineCheck: Detect flat lines by checking for consecutive repeated values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time

import polars as pl

from time_stream.exceptions import QcError, QcUnknownOperatorError
from time_stream.operation import Operation
from time_stream.types import ClosedInterval
from time_stream.utils import (
    check_columns_in_dataframe,
    check_literal_value,
    check_naive_time,
    check_time_zone,
    get_date_filter,
)


@dataclass(frozen=True)
class QcCtx:
    """Immutable context passed to QC checks."""

    df: pl.DataFrame
    time_name: str


class QCCheck(Operation, ABC):
    """Base class for quality control checks."""

    @abstractmethod
    def expr(self, _ctx: QcCtx, _column: str) -> pl.Expr:
        """Return the boolean Polars expression for this QC check (result of True == Value failed the QC check)"""
        pass

    def apply(
        self,
        df: pl.DataFrame,
        time_name: str,
        check_column: str,
        observation_interval: datetime | tuple[datetime, datetime | None] | None = None,
    ) -> pl.Series:
        """Apply the QC check to the data.

        Args:
            df: The Polars DataFrame containing the time series data to quality control
            time_name: Name of the time column in the dataframe
            check_column: The column to perform the check on.
            observation_interval: Optional time interval to limit the check to. Datetimes must match the time column:
                without a time zone if it has none, or in UTC if it is in UTC.

        Returns:
            pl.Series: Boolean series of the resolved expression.
        """
        ctx = QcCtx(df, time_name)
        pipeline = QcCheckPipeline(self, ctx, check_column, observation_interval)
        return pipeline.execute()


class QcCheckPipeline:
    """Encapsulates the logic for the QC pipeline steps."""

    def __init__(
        self,
        qc_check: QCCheck,
        ctx: QcCtx,
        column: str,
        observation_interval: datetime | tuple[datetime, datetime | None] | None = None,
    ):
        self.qc_check = qc_check
        self.ctx = ctx
        self.column = column
        self.observation_interval = observation_interval

    def execute(self) -> pl.Series:
        """Execute the quality control check pipeline

        Returns:
            Polars boolean series of the result of the QC check
        """
        self._validate()

        # Get the check expression
        check_expr = self.qc_check.expr(self.ctx, self.column)

        # Apply observation interval filter if specified
        if self.observation_interval:
            time_dtype = self.ctx.df.schema[self.ctx.time_name]
            date_filter = get_date_filter(self.ctx.time_name, self.observation_interval, time_dtype)
            check_expr = check_expr & date_filter

        # Evaluate and return the result of the QC check
        #   Name as empty string to avoid accidental collisions.
        #   Up to user if they want to name it and add on to the dataframe.
        result = self.ctx.df.select(check_expr.alias("")).to_series()

        return result

    def _validate(self) -> None:
        """Carry out validation that the QC check can actually be carried out."""
        if self.ctx.df.is_empty():
            raise QcError("Cannot perform QC check on an empty DataFrame.")
        check_columns_in_dataframe(self.ctx.df, [self.column, self.ctx.time_name])


@QCCheck.register
class ComparisonCheck(QCCheck):
    """Compares values against a given value using a comparison operator."""

    name = "comparison"

    def __init__(self, compare_to: float | datetime | list, operator: str, flag_na: bool = False) -> None:
        """Initialise comparison check.

        Args:
            compare_to: The value for comparison.
            operator: Comparison operator. One of: '>', '>=', '<', '<=', '==', '!=', 'is_in'.
            flag_na: If True, also flag NaN/null values as failing the check. Defaults to False.
        """
        self.compare_to = compare_to
        self.operator = operator
        self.flag_na = flag_na

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for threshold checking."""
        # Datetimes must be in the time zone of the column
        compare_to = self.compare_to
        for value in compare_to if isinstance(compare_to, list) else [compare_to]:
            if isinstance(value, datetime):
                check_time_zone(value, ctx.df.schema[column], "compare_to")

        operator_map = {
            ">": pl.col(column) > compare_to,
            ">=": pl.col(column) >= compare_to,
            "<": pl.col(column) < compare_to,
            "<=": pl.col(column) <= compare_to,
            "==": pl.col(column) == compare_to,
            "!=": pl.col(column) != compare_to,
            "is_in": pl.col(column).is_in(compare_to if isinstance(compare_to, list) else [compare_to]),
        }

        if self.operator not in operator_map:
            raise QcUnknownOperatorError(f"Invalid operator '{self.operator}'. Use: {', '.join(operator_map.keys())}")

        operator_expr = operator_map[self.operator]
        if self.flag_na:
            is_na = pl.col(column).is_null()
            if ctx.df.schema[column].is_float():
                is_na = is_na | pl.col(column).is_nan()
            operator_expr = operator_expr | is_na

        return operator_expr


@QCCheck.register
class RangeCheck(QCCheck):
    """Check that values fall within an acceptable range."""

    name = "range"

    def __init__(
        self,
        min_value: float | time | date | datetime,
        max_value: float | time | date | datetime,
        closed: ClosedInterval = "both",
        within: bool = True,
    ) -> None:
        """Initialise range check.

        Args:
            min_value: Minimum of the range. A datetime must match the checked column: without a time zone if it
                has none, or in UTC if it is in UTC. A time of day must not have a time zone.
            max_value: Maximum of the range.
            closed: Define which sides of the interval are closed (inclusive) {'both', 'left', 'right', 'none'}
                    (default = "both")
            within: Whether values get flagged when within or outside the range (default = True (within)).

        Raises:
            TypeError: If ``min_value`` and ``max_value`` are not of the same type.
        """
        check_literal_value(closed, ClosedInterval, "closed")
        if type(min_value) is not type(max_value):
            raise TypeError("'min_value' and 'max_value' must be of same type")
        self.min_value = min_value
        self.max_value = max_value
        self.closed = closed
        self.within = within

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for range checking."""
        # Make a local copy of these variables, otherwise the code that changes these values below would alter the
        # class attributes
        min_value, max_value, closed, within = self.min_value, self.max_value, self.closed, self.within
        check_type = type(min_value)

        # Datetimes must be in the time zone of the column, and times of day can't have a time zone
        if isinstance(min_value, datetime) and isinstance(max_value, datetime):
            check_time_zone(min_value, ctx.df.schema[column], "min_value")
            check_time_zone(max_value, ctx.df.schema[column], "max_value")
        elif isinstance(min_value, time) and isinstance(max_value, time):
            check_naive_time(min_value, "min_value")
            check_naive_time(max_value, "max_value")

        # Check if we're doing a time-based range check
        if check_type is time:
            col_expr = pl.col(column).dt.time()

            # Consider ranges that cross midnight, e.g. min_value = 11:00, max_value = 01:00
            if min_value > max_value:  # type: ignore[operator] - we know the types are the same
                # Swap the values so the comparison operators work the correct way around
                min_value, max_value = max_value, min_value

                # Reverse the within parameter, as we've swapped the min/max logic
                within = not within

                # We also need to swap the close parameter (if "both" or "none")
                # Don't have to change "left" or "right" as it shakes out the same even when reversing the min/max
                if closed == "both":
                    closed = "none"
                elif closed == "none":
                    closed = "both"

        elif check_type is date:
            # For datetime.date objects (NOT datetime.datetime!), we want to consider the whole date part of the column
            col_expr = pl.col(column).dt.date()

        else:
            # This should handle numeric objects and datetime.datetime objects
            col_expr = pl.col(column)

        in_range = col_expr.is_between(
            min_value,
            max_value,
            closed=closed,  # type: ignore[arg-type] ignore Literal typing as the enum constrains the values
        )
        return in_range if within else ~in_range


@QCCheck.register
class TimeRangeCheck(RangeCheck):
    """Flag rows where the primary time column of the time series fall within an acceptable range.

    This can either be used with min / max values of:
        - datetime.time : Useful for scenarios where there are consistent errors at a certain time of day,
                          e.g., during an automated sensor calibration time.
        - datetime.date : Useful for scenarios where a specific date range is known to be bad,
                              e.g., during a time of sensor errors not picked up elsewhere.
        - datetime.datetime : As above, but where there you need to add a time to the date range as well.

    Note: This is equivalent to using `RangeCheck` with `check_column = ts.time_name`. However, adding this as a
          convenience method as it may not be obvious that the `RangeCheck` can be used for this purpose.
    """

    name = "time_range"

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        return super().expr(ctx, ctx.time_name)


@QCCheck.register
class SpikeCheck(QCCheck):
    """Detect spikes by assessing differences with neighboring values."""

    name = "spike"

    def __init__(self, threshold: float):
        """Initialise spike detection check.

        Args:
            threshold: The spike detection threshold.
        """
        self.threshold = threshold

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for spike detection.

        The algorithm:
        1. Calculate differences between current value and neighbors
        2. Compute total combined difference and skew
        3. Flag where (total_difference - skew) > threshold * 2
        """
        # Calculate differences with temporal neighbors
        prev_val = pl.col(column).shift(1)
        next_val = pl.col(column).shift(-1)

        diff_prev = pl.col(column) - prev_val
        diff_next = next_val - pl.col(column)

        # Calculate total difference and skew
        d = (diff_prev - diff_next).abs()
        skew = (diff_prev.abs() - diff_next.abs()).abs()
        d_no_skew = d - skew

        # Double the threshold since we're summing differences
        return d_no_skew > (self.threshold * 2.0)


@QCCheck.register
class FlatLineCheck(QCCheck):
    """Detect flat lines by checking for consecutive repeated values."""

    name = "flat_line"

    def __init__(
        self,
        min_count: int,
        tolerance: float | None = None,
        ignore_value: int | float | str | list | None = None,
    ) -> None:
        """Initialise flat line detection check.

        Args:
            min_count: Minimum number of consecutive repeated values required for a flat line. Must be at least 2.
            tolerance: Optional tolerance for near-equality comparison. When set, consecutive values differing by
                less than or equal to this amount are considered equal. Only valid for numeric columns.
                Defaults to None (exact equality).
            ignore_value: Optional value or list of values that are allowed to repeat without being flagged.
                For float columns, int values are automatically upcast to float. For all other column types,
                values must match the column's type exactly.
        """
        if min_count < 2:
            raise ValueError("min_count for flat line check must be at least 2")
        self.min_count = min_count
        self.tolerance = tolerance
        self.ignore_values = (
            ignore_value if isinstance(ignore_value, list) else ([ignore_value] if ignore_value is not None else [])
        )

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for flat line detection.

        Repeated nulls do not count as flat lines. Null values break flat line groups.
        """
        prev = pl.col(column).shift(1)

        # Treat "equal to previous" such that two nulls are not considered equal.
        # When tolerance is set, use absolute difference; otherwise use exact equality.
        if self.tolerance is None:
            equal_to_prev = pl.col(column).is_not_null() & prev.is_not_null() & (pl.col(column) == prev)
        else:
            equal_to_prev = (
                pl.col(column).is_not_null() & prev.is_not_null() & ((pl.col(column) - prev).abs() <= self.tolerance)
            )

        # Mark starts of new groups, then create a group id by cumulative sum
        change = (~equal_to_prev).cast(pl.Int32)
        group_id = change.cum_sum()

        # Size of each group
        group_size = pl.len().over(group_id)

        # Only consider non-null groups for flat-line detection
        result = (group_size >= self.min_count) & pl.col(column).is_not_null()

        # Apply ignore-values if provided
        if self.ignore_values:
            dtype = ctx.df.schema[column]
            try:
                ignore_list = pl.Series("", self.ignore_values, dtype=dtype, strict=True).to_list()
            except Exception as e:
                raise TypeError(f"ignore_value is incompatible with column '{column}' of type {dtype}: {e}") from e
            result = result & ~pl.col(column).is_in(ignore_list)

        return result
