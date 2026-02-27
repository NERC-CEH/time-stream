"""
Time Series Aggregation Module

This module provides a flexible framework for aggregating time series data using Polars. Aggregation functions are
implemented as subclasses of ``AggregationFunction`` and can be registered and instantiated by name, class, or instance.

It supports various aggregation functions (mean, sum, min, max, etc.) with configurable missing data handling and
period-based grouping.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Callable

import polars as pl

from time_stream import Period
from time_stream.enums import ClosedInterval, MissingCriteria, TimeAnchor
from time_stream.exceptions import AggregationError, AggregationPeriodError, MissingCriteriaError, TimeWindowError
from time_stream.operation import Operation
from time_stream.utils import check_columns_in_dataframe


@dataclass(frozen=True)
class TimeWindow:
    """A helper class for working with time windows: a set period of time between a start time and end time.

    Can be used, for example, to restrict which observations within an aggregation period are included. For example,
    to compute mean daily albedo from 30-minute values, you might restrict to the window 10:30-14:00.

    Only supported for daily or longer aggregation periods when the data periodicity is sub-daily.
    Midnight-wrapping windows (where ``start >= end``) are not supported.

    Attributes:
        start: The start of the time-of-day window.
        end: The end of the time-of-day window. Must be after ``start``.
        closed: Which boundary times are included. Defaults to both ends being closed (inclusive).
    """

    start: time
    end: time
    closed: ClosedInterval | None = ClosedInterval.BOTH

    def __post_init__(self) -> None:
        """Validate the time window on construction."""
        if not isinstance(self.start, time) or not isinstance(self.end, time):
            raise TimeWindowError("'start' and 'end' must be datetime.time objects.")
        if self.start >= self.end:
            raise TimeWindowError(f"'start' ({self.start}) must be strictly before 'end' ({self.end}).")

    @property
    def duration(self) -> timedelta:
        """The duration of the time window as a timedelta.

        Returns:
            The time difference between end and start.
        """
        return datetime.combine(date.min, self.end) - datetime.combine(date.min, self.start)

    def expected_count(self, periodicity: Period) -> int:
        """Count the number of observations expected within this window for a given periodicity.

        Adjusts the counts for the closed boundaries:
        - ``BOTH`` adds one (both boundary timestamps are included)
        - ``NONE`` subtracts one (both are excluded)
        - ``LEFT`` and ``RIGHT`` leave the count unchanged.

        Args:
            periodicity: The Period object from which the timedelta is used for the calculation.

        Returns:
            The expected number of observations within the window for the given periodicity.
        """
        count = self.duration // periodicity.timedelta
        if self.closed == ClosedInterval.BOTH:
            count += 1
        elif self.closed == ClosedInterval.NONE:
            count = max(0, count - 1)
        return count


@dataclass(frozen=True)
class AggregationCtx:
    """Immutable context passed to aggregations."""

    df: pl.DataFrame
    time_name: str
    time_anchor: TimeAnchor
    periodicity: Period


class AggregationFunction(Operation, ABC):
    """Base class for aggregation functions."""

    @abstractmethod
    def expr(self, _ctx: AggregationCtx, _columns: list[str]) -> list[pl.Expr]:
        """Return the Polars expressions for this aggregation."""
        raise NotImplementedError

    def post_expr(self, _ctx: AggregationCtx, _columns: list[str]) -> list[pl.Expr]:
        """Return additional Polars expressions to be applied after the aggregation."""
        return []

    def apply(
        self,
        df: pl.DataFrame,
        time_name: str,
        time_anchor: TimeAnchor,
        periodicity: Period,
        aggregation_period: Period,
        columns: str | list[str],
        aggregation_time_anchor: TimeAnchor,
        missing_criteria: tuple[str, float | int] | None = None,
        time_window: TimeWindow | None = None,
    ) -> pl.DataFrame:
        """Run the aggregation pipeline.

        Args:
            df: The Polars DataFrame containing the time series data to aggregate
            time_name: Name of the time column in the dataframe
            time_anchor: Time anchor of the time series
            periodicity: Periodicity of the time series
            aggregation_period: Period over which to aggregate the data
            columns: Column(s) containing the data to be aggregated
            aggregation_time_anchor: The time anchor for the aggregation result.
            missing_criteria: How the aggregation handles missing data
            time_window: Optional restriction of which time-of-day observations are included.

        Returns:
            The aggregated dataframe
        """
        ctx = AggregationCtx(df, time_name, time_anchor, periodicity)
        pipeline = AggregationPipeline(
            self,
            ctx,
            aggregation_period,
            columns,
            aggregation_time_anchor,
            missing_criteria,
            time_window,
        )
        return pipeline.execute()


class AggregationPipeline:
    """Encapsulates the logic for the aggregation pipeline steps."""

    def __init__(
        self,
        agg_func: AggregationFunction,
        ctx: AggregationCtx,
        aggregation_period: Period,
        columns: str | list[str],
        aggregation_time_anchor: TimeAnchor,
        missing_criteria: tuple[str, float | int] | None = None,
        time_window: TimeWindow | None = None,
    ):
        self.agg_func = agg_func
        self.ctx = ctx
        self.aggregation_period = aggregation_period
        self.aggregation_time_anchor = aggregation_time_anchor
        self.columns = [columns] if isinstance(columns, str) else columns
        self.missing_criteria = missing_criteria
        self.time_window = time_window

    def execute(self) -> pl.DataFrame:
        """The general `Polars` method used for aggregating data is::

        output_dataframe = input_dataframe
            .group_by_dynamic(...)  # Group the time series into groups of values based on the chosen period
            .agg(...)               # Apply the chosen aggregation function (e.g., mean, min, max)
            .with_columns(...)      # Apply additional logic that requires the results of the aggregation

        This method fills in the various stages using pre-defined expressions.

        Returns:
            The aggregated dataframe
        """
        self._validate()

        # Optionally filter observations to those within the time-of-day window
        df = self._apply_time_window(self.ctx.df) if self.time_window else self.ctx.df

        # Group by the aggregation period - taking into account the time anchor of the time series
        label, closed = self._get_label_closed()
        grouper = df.group_by_dynamic(
            index_column=self.ctx.time_name,
            every=self.aggregation_period.pl_interval,
            offset=self.aggregation_period.pl_offset,
            closed=closed,
            label=label,
        )

        # Build expressions to go in the .agg method
        agg_expressions = []
        agg_expressions.extend(self.agg_func.expr(self.ctx, self.columns))
        agg_expressions.extend(self._actual_count_expr())

        # Do the aggregation function
        df = grouper.agg(agg_expressions)

        # Build expressions to go in the .with_columns method.
        #   Note: - Order is important here. Expressions may have dependencies on the results of earlier expressions.
        #         - Doing separate .with_columns calls to group expressions together and take advantage of the Polars
        #           internal planning where possible.

        # Add expected count
        df = df.with_columns(self._expected_count_expr())

        # Add missing-data flags, and any post-agg columns in one plan. Both may require results of the expected count.
        df = df.with_columns(
            [
                *self._missing_data_expr(),
                *self.agg_func.post_expr(self.ctx, self.columns),
            ]
        )

        return df

    def _validate(self) -> None:
        """Carry out validation that the aggregation can actually be carried out."""
        if self.ctx.df.is_empty():
            raise AggregationError("Cannot aggregate an empty DataFrame.")
        check_columns_in_dataframe(self.ctx.df, self.columns + [self.ctx.time_name])
        self._validate_period_compatibility()
        self._validate_time_window()

    def _validate_period_compatibility(self) -> None:
        """Validate that the aggregation period is compatible with the time series periodicity."""
        if not self.aggregation_period.is_epoch_agnostic():
            raise AggregationPeriodError(
                f"Non-epoch agnostic aggregation periods are not supported: '{self.aggregation_period}'."
            )

        if not self.ctx.periodicity.is_subperiod_of(self.aggregation_period):
            raise AggregationPeriodError(
                f"Incompatible aggregation period '{self.aggregation_period}' with TimeFrame periodicity "
                f"'{self.ctx.periodicity}'. TimeFrame periodicity must be a subperiod of the aggregation period."
            )

    def _validate_time_window(self) -> None:
        """Validate that the time_window, if provided, is compatible with the aggregation."""
        if self.time_window is None:
            return

        # Aggregation period must be daily or longer
        agg_td = self.aggregation_period.timedelta
        if agg_td is not None and agg_td < timedelta(days=1):
            raise TimeWindowError("'time_window' is only supported for daily or longer aggregation periods")

        # Data periodicity must be sub-daily
        periodicity_td = self.ctx.periodicity.timedelta
        if periodicity_td is None or periodicity_td >= timedelta(days=1):
            raise TimeWindowError("'time_window' requires the data periodicity to be sub-daily.")

    def _apply_time_window(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter a DataFrame to only rows whose time-of-day falls within the time window.

        Args:
            df: The DataFrame to filter.

        Returns:
            Filtered DataFrame containing only rows within the time window.
        """
        time_col = pl.col(self.ctx.time_name).dt.time()
        return df.filter(
            time_col.is_between(
                self.time_window.start,
                self.time_window.end,
                closed=self.time_window.closed.value,  # type: ignore - linter complains that the string isn't a Literal
            )
        )

    def _get_label_closed(self) -> tuple[str, str]:
        """Map TimeAnchor to Polars label/closed semantics.

        Returns:
            Tuple of (label, closed) values to use in Polars operations
        """
        label = "right" if self.aggregation_time_anchor == TimeAnchor.END else "left"
        closed = "right" if self.ctx.time_anchor == TimeAnchor.END else "left"
        return label, closed

    def _actual_count_expr(self) -> list[pl.Expr]:
        """A `Polars` expression to generate the actual count of values in a TimeFrame found in each period.

        Returns:
            List of `Polars` expressions that can be used to generate actual counts for each column
        """
        return [pl.col(col).count().alias(f"count_{col}") for col in self.columns]

    def _expected_count_expr(self) -> pl.Expr:
        """A `Polars` expression to generate the expected count of values in a TimeFrame found in each
        period (if there were no missing values).

        When a ``time_window`` is set, the expected count reflects only the observations that fall within
        that window, rather than all observations in the aggregation period.

        Returns:
            pl.Expr: Polars expression that can be used to generate expected count on a DataFrame
        """
        expected_count_name = f"expected_count_{self.ctx.time_name}"

        # When a time window is active, the expected count is derived from the window duration
        # and the periodicity, adjusted for the closed boundaries.
        if self.time_window is not None:
            count = self.time_window.expected_count(self.ctx.periodicity)
            expr = pl.lit(count)

        # For some aggregations, the expected count is a constant so use that if possible.
        # For example, when aggregating 15-minute data over a day, the expected count is always 96.
        elif self.ctx.periodicity.count(self.aggregation_period) > 0:
            expr = pl.lit(self.ctx.periodicity.count(self.aggregation_period))

        else:
            # Variable length periods need dynamic calculation, based on the start and end of a period interval
            label, closed = self._get_label_closed()
            if label == "right":
                start_expr = pl.col(self.ctx.time_name).dt.offset_by("-" + self.aggregation_period.pl_interval)
                end_expr = pl.col(self.ctx.time_name)
            else:
                start_expr = pl.col(self.ctx.time_name)
                end_expr = pl.col(self.ctx.time_name).dt.offset_by(self.aggregation_period.pl_interval)

            # This contains 2 cases:
            if self.ctx.periodicity.timedelta:
                # 1. If the data we are aggregating is not monthly, then each interval we are aggregating has
                #    a constant length, so (end - start) / interval will be the expected count.
                micros = self.ctx.periodicity.timedelta // timedelta(microseconds=1)
                expr = (end_expr - start_expr).dt.total_microseconds() // micros

            else:
                # 2. If the data we are aggregating is month-based, then there is no simple way to do the
                #    calculation, so use Polars to create a Series of date ranges lists and get the length
                #    of each list.
                #
                #    Note: This method will work for above use cases also, but if periodicity is small and the
                #    aggregation period is large, it consumes too much memory and causes performance problems.
                #    For example, aggregating 1 microsecond data over a calendar year involves the creation of
                #    length 1_000_000 * 60 * 60 * 24 * 365 arrays which will probably fail with an
                #    out-of-memory error.
                expr = pl.datetime_ranges(
                    start_expr,
                    end_expr,
                    interval=self.ctx.periodicity.pl_interval,
                    closed=closed,  # type: ignore - linter is complaining that the string isn't a Literal
                ).list.len()

        return expr.cast(pl.UInt32).alias(expected_count_name)

    def _missing_data_expr(self) -> list[pl.Expr]:
        """Convert missing criteria to a Polars expression for validation.

        Returns:
            List of `Polars` expressions that can be used to generate missing data validation columns
        """
        # Do some checks that the missing criteria is valid
        try:
            if self.missing_criteria is None:
                criteria, threshold = MissingCriteria.NA, 0
            else:
                criteria, threshold = self.missing_criteria

            criteria = MissingCriteria(criteria)
        except ValueError:
            raise MissingCriteriaError(
                f"Unknown missing criteria: {self.missing_criteria}. "
                f"Available criteria: {[c.name for c in MissingCriteria]}"
            )

        if criteria == MissingCriteria.PERCENT:
            if not 0 <= threshold <= 100:
                raise MissingCriteriaError(f"Invalid percent threshold '{threshold}'. Must be between 0 and 100.")
        else:
            if not isinstance(threshold, int) or threshold < 0:
                raise MissingCriteriaError(f"Invalid threshold '{threshold}'. Must be a non-negative integer.")

        # Build the expression based on the specified criteria
        expressions = []
        for col in self.columns:
            if criteria == MissingCriteria.PERCENT:
                expr = ((pl.col(f"count_{col}") / pl.col(f"expected_count_{self.ctx.time_name}")) * 100) > threshold

            elif criteria == MissingCriteria.MISSING:
                expr = (pl.col(f"expected_count_{self.ctx.time_name}") - pl.col(f"count_{col}")) <= threshold

            elif criteria == MissingCriteria.AVAILABLE:
                expr = pl.col(f"count_{col}") >= threshold

            else:
                # Default to check that the actual count found in the column is above 0.
                expr = pl.col(f"count_{col}") > 0

            expr = expr.alias(f"valid_{col}")
            expressions.append(expr)

        return expressions


@AggregationFunction.register
class Mean(AggregationFunction):
    """An aggregation class to calculate the mean (average) of values within each aggregation period."""

    name = "mean"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the mean in an aggregation period."""
        return [pl.col(col).mean().alias(f"mean_{col}") for col in columns]


@AggregationFunction.register
class AngularMean(AggregationFunction):
    """An aggregation class to calculate the angular mean (average angle) of values within each aggregation period."""

    name = "angular_mean"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """
        Return the `Polars` expression for calculating the angular mean in an aggregation period.
        Assumptions:
        Measurement units: degrees
        Desired output units: degrees
        """

        angular_mean = [
            pl.arctan2((pl.col(col).radians().sin().sum()), (pl.col(col).radians().cos().sum()))
            .degrees()
            .round(1)
            .mod(360)
            .alias(f"angular_mean_{col}")
            for col in columns
        ]

        return angular_mean


@AggregationFunction.register
class Sum(AggregationFunction):
    """An aggregation class to calculate the sum (total) of values within each aggregation period."""

    name = "sum"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the sum in an aggregation period."""
        return [pl.col(col).sum().alias(f"sum_{col}") for col in columns]


@AggregationFunction.register
class MeanSum(AggregationFunction):
    """An aggregation class to calculate the mean sum (averaged total) of values within each aggregation period.
    This will estimate the sum when values are missing according to how many values are expected in the period."""

    name = "mean_sum"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """To calculate the mean sum the expression must return the mean, and be multiplied by the expected
        counts, which is calculated after in the post_expr method."""
        return [pl.col(col).mean().alias(f"mean_sum_{col}") for col in columns]

    def post_expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Multiply the mean by the expected count to get the mean sum."""
        return [
            (pl.col(f"mean_sum_{col}") * pl.col(f"expected_count_{ctx.time_name}")).alias(f"mean_sum_{col}")
            for col in columns
        ]


@AggregationFunction.register
class Min(AggregationFunction):
    """An aggregation class to find the minimum of values within each aggregation period."""

    name = "min"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the minimum in an aggregation period.
        This expression also returns a column that holds the datetime that the minimum value occurred on.
        """
        expressions = []
        for col in columns:
            expressions.extend(
                [
                    pl.col(col).min().alias(f"{self.name}_{col}"),
                    pl.col(ctx.time_name).get(pl.col(col).arg_min()).alias(f"{ctx.time_name}_of_{self.name}_{col}"),
                ]
            )
        return expressions


@AggregationFunction.register
class Max(AggregationFunction):
    """An aggregation class to find the maximum of values within each aggregation period."""

    name = "max"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the maximum in an aggregation period.
        This expression also returns a column that holds the datetime that the maximum value occurred on.
        """
        expressions = []
        for col in columns:
            expressions.extend(
                [
                    pl.col(col).max().alias(f"{self.name}_{col}"),
                    pl.col(ctx.time_name).get(pl.col(col).arg_max()).alias(f"{ctx.time_name}_of_{self.name}_{col}"),
                ]
            )
        return expressions


@AggregationFunction.register
class Percentile(AggregationFunction):
    """An aggregation class to find the nth percentile of values within each aggregation period."""

    name = "percentile"

    def __init__(self, p: int):
        """Initialise Percentile aggregation.

        Args:
            p: The integer percentile value to apply.
        """
        super().__init__()
        self.p = p

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the 'Polars' expression for calculating the percentile"""

        # If the percentile value is between 0 -100 divide it by 100 to convert it to the quantile equivalent.
        if not self.p.is_integer() or not (0 <= self.p <= 100):
            raise ValueError("The percentile value must be provided as an integer value from 0 to 100")

        quantile = self.p / 100

        expressions = [(pl.col(col).quantile(quantile).alias(f"{self.name}_{col}")) for col in columns]

        return expressions


@AggregationFunction.register
class ConditionalCount(AggregationFunction):
    """An aggregation class to count values that meet a given condition within each aggregation period."""

    name = "conditional_count"

    def __init__(self, condition: Callable[[pl.Expr], pl.Expr]):
        """Initialise the conditional count aggregation.

        Args:
            condition: A function that takes a Polars expression and returns a boolean expression.
                      Examples:
                        - lambda col: col > 100
                        - lambda col: (col > 10) & (col <= 50)
                        - lambda col: col.is_not_null() & (col > 0)
        """
        super().__init__()
        self.condition = condition

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the conditional count in an aggregation period."""
        return [self.condition(pl.col(col)).sum().alias(f"{self.name}_{col}") for col in columns]


@AggregationFunction.register
class PeaksOverThreshold(ConditionalCount):
    name = "pot"

    def __init__(self, threshold: int | float):
        """Initialise Peaks Over Threshold aggregation.

        Args:
            threshold: The threshold to count peaks over.
        """
        super().__init__(lambda col: col > threshold)


@AggregationFunction.register
class StDev(AggregationFunction):
    """An aggregation class to calculate the standard deviation of values within each aggregation period."""

    name = "stdev"

    def expr(self, ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the standard deviation in an aggregation period."""
        return [pl.col(col).std().alias(f"stdev_{col}") for col in columns]
