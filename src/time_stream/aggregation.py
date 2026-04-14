"""
Time Series Aggregation Module

This module provides a flexible framework for aggregating time series data using Polars.

Aggregation functions are implemented as subclasses of :class:`AggregationFunction` and can be registered and
instantiated by name, class, or instance. They provide Polars expressions but do not orchestrate execution.

Execution is handled by two concrete pipeline classes:

- :class:`StandardAggregationPipeline`: groups data by fixed periods using ``group_by_dynamic``.
- :class:`RollingAggregationPipeline`: slides a window over the data using ``rolling``.

Both share a common abstract base class :class:`AggregationPipeline`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable

import polars as pl
from polars.dataframe.group_by import DynamicGroupBy, RollingGroupBy

from time_stream import Period
from time_stream.enums import MissingCriteria, RollingAlignment, TimeAnchor
from time_stream.exceptions import AggregationError, AggregationPeriodError, MissingCriteriaError, TimeWindowError
from time_stream.operation import Operation
from time_stream.utils import TimeWindow, check_columns_in_dataframe


@dataclass(frozen=True)
class AggregationCtx:
    """Immutable context passed to aggregations."""

    df: pl.DataFrame
    time_name: str
    time_anchor: TimeAnchor
    periodicity: Period


class AggregationFunction(Operation, ABC):
    """Base class for aggregation functions.

    Subclasses provide the Polars expressions for a specific aggregation (e.g., mean, sum, max).
    Pipeline orchestration is handled separately by :class:`StandardAggregationPipeline` or
    :class:`RollingAggregationPipeline`.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def expr(self, _ctx: AggregationCtx, _columns: list[str]) -> list[pl.Expr]:
        """Return the Polars expressions for this aggregation."""
        raise NotImplementedError

    def post_expr(self, _ctx: AggregationCtx, _columns: list[str]) -> list[pl.Expr]:
        """Return additional Polars expressions to be applied after the aggregation."""
        return []


class AggregationPipeline(ABC):
    """Abstract base class for aggregation pipelines.

    Subclasses implement the grouping strategy (standard period grouping or rolling window) while
    sharing the common ``execute()`` template and helper expressions.

    Use :class:`StandardAggregationPipeline` for fixed-period aggregation and
    :class:`RollingAggregationPipeline` for sliding-window aggregation.
    """

    def __init__(
        self,
        agg_func: AggregationFunction,
        ctx: AggregationCtx,
        aggregation_period: Period,
        columns: str | list[str],
        missing_criteria: tuple[str, float | int] | None = None,
    ):
        self.agg_func = agg_func
        self.ctx = ctx
        self.aggregation_period = aggregation_period
        self.columns = [columns] if isinstance(columns, str) else columns
        self.missing_criteria = missing_criteria

    def execute(self) -> pl.DataFrame:
        """Run the aggregation pipeline.

        The pipeline carries out four stages::
            - _prepare_df(...)       # Optional modifications to the input dataframe (e.g., time window row filtering)
            - _get_grouper(df)...    # Determined by subclass, e.g. standard vs. rolling aggregation grouper
            - .agg(...)              # Apply aggregation expressions
            - .with_columns(...)     # Add expected count, validity flags, post-agg expressions

        Returns:
            The aggregated DataFrame.
        """
        self._validate()

        df = self._prepare_df(self.ctx.df)
        grouper = self._get_grouper(df)

        # Build expressions to go in the .agg method
        agg_expressions = list(self.agg_func.expr(self.ctx, self.columns))
        agg_expressions.extend(self._actual_count_expr())
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

    def _prepare_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Pre-process the DataFrame before grouping.

         Default is to do nothing - subclasses can override this.

        Args:
            df: The input DataFrame.

        Returns:
            The (optionally modified) DataFrame.
        """
        return df

    @abstractmethod
    def _validate(self) -> None:
        """Validate that the aggregation can be carried out. Subclasses add specific checks."""
        raise NotImplementedError

    @abstractmethod
    def _get_grouper(self, df: pl.DataFrame) -> DynamicGroupBy | RollingGroupBy:
        """Return the Polars grouper (e.g. ``group_by_dynamic`` or ``rolling``).

        Args:
            df: The pre-processed DataFrame.

        Returns:
            A Polars grouper object on which ``.agg()`` can be called.
        """
        raise NotImplementedError

    @abstractmethod
    def _dynamic_expected_count_expr(self) -> pl.Expr:
        """Compute the expected count expression for variable-length aggregation periods.

        Called by ``_expected_count_expr()`` when the constant-count fast path is not applicable.
        Subclasses implement this using their own grouping semantics (label/closed for standard,
        alignment for rolling).

        Returns:
            A Polars expression for the expected count.
        """
        raise NotImplementedError

    def _validate_common(self) -> None:
        """Carry out validation checks common to all pipeline types."""
        if self.ctx.df.is_empty():
            raise AggregationError("Cannot aggregate an empty DataFrame.")
        check_columns_in_dataframe(self.ctx.df, self.columns + [self.ctx.time_name])

    def _actual_count_expr(self) -> list[pl.Expr]:
        """A `Polars` expression to generate the actual count of values in a TimeFrame found in each period.

        Returns:
            List of `Polars` expressions that can be used to generate actual counts for each column
        """
        return [pl.col(col).count().alias(f"count_{col}") for col in self.columns]

    def _expected_count_expr(self) -> pl.Expr:
        """A `Polars` expression to generate the expected count of values found in each period.

        Returns:
            pl.Expr: Polars expression that can be used to generate expected count on a DataFrame
        """
        expected_count_name = f"expected_count_{self.ctx.time_name}"

        # Subclasses can optionally define separate paths to determine the expected count - e.g. using the time window
        #   during standard aggregation. Check this first, and if not provided then fall back to the default paths
        #   for determining the expected count.
        expr = self._static_expected_count_expr()
        if expr is None:
            # For some aggregations, the expected count is a constant so use that if possible.
            # For example, when aggregating 15-minute data over a day, the expected count is always 96.
            if self.ctx.periodicity.count(self.aggregation_period) > 0:
                expr = pl.lit(self.ctx.periodicity.count(self.aggregation_period))
            else:
                expr = self._dynamic_expected_count_expr()

        return expr.cast(pl.UInt32).alias(expected_count_name)

    def _static_expected_count_expr(self) -> pl.Expr | None:
        """Return a fixed-value expected count expression, or ``None`` to use default logic.

        Override in subclasses when the expected count can be determined without inspecting each
        row (e.g., from a time window). The base implementation returns ``None``.

        Returns:
            A Polars literal expression, or ``None``.
        """
        return None

    def _count_between_expr(self, start_expr: pl.Expr, end_expr: pl.Expr, closed: str) -> pl.Expr:
        """Compute the number of observations of the data periodicity between two timestamp expressions.

        Uses arithmetic division for fixed-length periodicities (fast path), and
        ``datetime_ranges`` for calendar-based periodicities such as months.

        Args:
            start_expr: Expression for the start of the window.
            end_expr: Expression for the end of the window.
            closed: Polars ``closed`` parameter (``"left"``, ``"right"``, or ``"both"``).

        Returns:
            A Polars expression for the count of observations.
        """
        if self.ctx.periodicity.timedelta:
            micros = self.ctx.periodicity.timedelta // timedelta(microseconds=1)
            return (end_expr - start_expr).dt.total_microseconds() // micros
        return pl.datetime_ranges(
            start_expr,
            end_expr,
            interval=self.ctx.periodicity.pl_interval,
            closed=closed,  # type: ignore - linter complains string isn't a Literal
        ).list.len()

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


class StandardAggregationPipeline(AggregationPipeline):
    """Aggregation pipeline that groups data into fixed periods using ``group_by_dynamic``.

    This is the concrete pipeline used by :meth:`~time_stream.TimeFrame.aggregate`. It reduces the
    resolution of the input data - each output row represents one aggregation period.

    Args:
        agg_func: The aggregation function to apply.
        ctx: Immutable aggregation context (DataFrame, time column, anchor, periodicity).
        aggregation_period: The period to aggregate into.
        columns: The column(s) to aggregate.
        missing_criteria: Optional completeness requirement as ``(policy, threshold)``.
        aggregation_time_anchor: The time anchor for output timestamps. Defaults to the input anchor.
        time_window: Optional restriction of which time-of-day observations are included.
    """

    def __init__(
        self,
        agg_func: AggregationFunction,
        ctx: AggregationCtx,
        aggregation_period: Period,
        columns: str | list[str],
        missing_criteria: tuple[str, float | int] | None = None,
        aggregation_time_anchor: TimeAnchor | None = None,
        time_window: TimeWindow | None = None,
    ):
        super().__init__(agg_func, ctx, aggregation_period, columns, missing_criteria)
        self.aggregation_time_anchor = (
            aggregation_time_anchor if aggregation_time_anchor is not None else ctx.time_anchor
        )
        self.time_window = time_window

    def _validate(self) -> None:
        """Validate period compatibility and time_window settings."""
        self._validate_common()
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
        agg_td = self.aggregation_period.timedelta
        if agg_td is not None and agg_td < timedelta(days=1):
            raise TimeWindowError("'time_window' is only supported for daily or longer aggregation periods")
        periodicity_td = self.ctx.periodicity.timedelta
        if periodicity_td is None or periodicity_td >= timedelta(days=1):
            raise TimeWindowError("'time_window' requires the data periodicity to be sub-daily.")

    def _prepare_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter rows to the time-of-day window if one is set."""
        if self.time_window:
            return self.time_window.filter_df(df, self.ctx.time_name)
        return df

    def _get_label_closed(self) -> tuple[str, str]:
        """Map TimeAnchor to Polars label/closed semantics.

        Returns:
            Tuple of (label, closed) values to use in Polars ``group_by_dynamic``.
        """
        label = "right" if self.aggregation_time_anchor == TimeAnchor.END else "left"
        closed = "right" if self.ctx.time_anchor == TimeAnchor.END else "left"
        return label, closed

    def _get_grouper(self, df: pl.DataFrame) -> DynamicGroupBy:
        """Return a ``group_by_dynamic`` grouper for fixed-period aggregation."""
        label, closed = self._get_label_closed()
        return df.group_by_dynamic(
            index_column=self.ctx.time_name,
            every=self.aggregation_period.pl_interval,
            offset=self.aggregation_period.pl_offset,
            closed=closed,  # type: ignore[arg-type] - Polars Literal is a string
            label=label,  # type: ignore[arg-type] - Polars Literal is a string
        )

    def _static_expected_count_expr(self) -> pl.Expr | None:
        """Return the time-window expected count when a window is active, otherwise ``None``.

        Returns:
            A Polars literal expression when ``time_window`` is set, or ``None``.
        """
        if self.time_window is not None:
            return pl.lit(self.time_window.expected_count(self.ctx.periodicity))
        return None

    def _dynamic_expected_count_expr(self) -> pl.Expr:
        """Compute expected count dynamically for variable-length periods (e.g., months, years).

        Uses the start and end of each aggregation interval to calculate how many observations
        of the given periodicity fit within it.

        Returns:
            Polars expression for the dynamic expected count.
        """
        label, closed = self._get_label_closed()
        if label == "right":
            start_expr = pl.col(self.ctx.time_name).dt.offset_by("-" + self.aggregation_period.pl_interval)
            end_expr = pl.col(self.ctx.time_name)
        else:
            start_expr = pl.col(self.ctx.time_name)
            end_expr = pl.col(self.ctx.time_name).dt.offset_by(self.aggregation_period.pl_interval)
        # Note: datetime_ranges also works for sub-monthly data, but is avoided there because
        # creating large lists (e.g. aggregating 1us data over a year) would exhaust memory.
        return self._count_between_expr(start_expr, end_expr, closed)


class RollingAggregationPipeline(AggregationPipeline):
    """Aggregation pipeline that slides a window over the data using ``rolling``.

    This is the concrete pipeline used by :meth:`~time_stream.TimeFrame.rolling_aggregate`. Unlike
    :class:`StandardAggregationPipeline`, rolling aggregation **preserves the original timestamps
    and resolution** - the output has the same number of rows as the input.

    Args:
        agg_func: The aggregation function to apply.
        ctx: Immutable aggregation context (DataFrame, time column, anchor, periodicity).
        aggregation_period: The rolling window size.
        columns: The column(s) to aggregate.
        missing_criteria: Optional completeness requirement as ``(policy, threshold)``.
        alignment: Where the window is placed relative to each timestamp (default: TRAILING).
    """

    def __init__(
        self,
        agg_func: AggregationFunction,
        ctx: AggregationCtx,
        aggregation_period: Period,
        columns: str | list[str],
        missing_criteria: tuple[str, float | int] | None = None,
        alignment: RollingAlignment = RollingAlignment.TRAILING,
    ):
        super().__init__(agg_func, ctx, aggregation_period, columns, missing_criteria)
        self.alignment = alignment

    def _validate(self) -> None:
        """Validate window size and alignment compatibility."""
        self._validate_common()
        self._validate_rolling()

    def _validate_rolling(self) -> None:
        """Validate rolling-specific constraints.

        Raises:
            AggregationPeriodError: If the window size is smaller than the data periodicity.
            AggregationError: If CENTER alignment is used with a calendar-based (variable-length) window.
        """
        if not self.ctx.periodicity.is_subperiod_of(self.aggregation_period):
            raise AggregationPeriodError(
                f"Rolling window size '{self.aggregation_period}' must be at least as large as the "
                f"data periodicity '{self.ctx.periodicity}'."
            )
        if self.alignment == RollingAlignment.CENTER and self.aggregation_period.timedelta is None:
            raise AggregationError(
                "CENTER alignment is not supported for calendar-based window sizes (e.g., months or years), "
                "because they have variable length and cannot be halved to a fixed offset."
            )

    def _get_rolling_params(self) -> tuple[str, str | None]:
        """Map the alignment to Polars ``closed`` and ``offset`` parameters.

        Returns:
            A ``(closed, offset)`` tuple. ``offset`` is ``None`` for TRAILING, which lets Polars
            use its default backward-looking offset of ``-period``.
        """
        if self.alignment == RollingAlignment.TRAILING:
            return "right", None
        elif self.alignment == RollingAlignment.LEADING:
            return "left", "0us"
        else:
            td = self.aggregation_period.timedelta
            half_us = int(td.total_seconds() * 1_000_000) // 2
            return "both", f"-{half_us}us"

    def _get_grouper(self, df: pl.DataFrame) -> RollingGroupBy:
        """Return a ``rolling`` grouper for sliding-window aggregation."""
        closed, offset = self._get_rolling_params()
        rolling_kwargs: dict = {
            "index_column": self.ctx.time_name,
            "period": self.aggregation_period.pl_interval,
            "closed": closed,
        }
        if offset is not None:
            rolling_kwargs["offset"] = offset
        return df.rolling(**rolling_kwargs)

    def _dynamic_expected_count_expr(self) -> pl.Expr:
        """Compute dynamic expected count for variable-length rolling windows (e.g., monthly).

        Returns:
            Polars expression for the dynamic expected count.

        Raises:
            AggregationError: If CENTER alignment is used (should have been caught in validation).
        """
        if self.alignment == RollingAlignment.TRAILING:
            start_expr = pl.col(self.ctx.time_name).dt.offset_by("-" + self.aggregation_period.pl_interval)
            end_expr = pl.col(self.ctx.time_name)
            closed = "right"
        elif self.alignment == RollingAlignment.LEADING:
            start_expr = pl.col(self.ctx.time_name)
            end_expr = pl.col(self.ctx.time_name).dt.offset_by(self.aggregation_period.pl_interval)
            closed = "left"
        else:
            raise AggregationError("CENTER alignment is not supported for calendar-based window sizes.")
        return self._count_between_expr(start_expr, end_expr, closed)


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
