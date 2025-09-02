from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import Type

import polars as pl
from polars._typing import ClosedInterval, Label

from time_stream import Period
from time_stream.enums import MissingCriteria, TimeAnchor
from time_stream.exceptions import (
    AggregationTypeError,
    MissingCriteriaError,
    UnknownAggregationError,
)

# Registry for built-in aggregations
_AGGREGATION_REGISTRY = {}


def register_aggregation(cls: Type["AggregationFunction"]) -> Type["AggregationFunction"]:
    """Decorator to register aggregation classes using their name attribute.

    Args:
        cls: The aggregation class to register.

    Returns:
        The decorated class.
    """
    _AGGREGATION_REGISTRY[cls.name] = cls
    return cls


@dataclass(frozen=True)
class AggregationCtx:
    """Immutable context passed to aggregations."""

    df: pl.DataFrame
    time_name: str
    time_anchor: TimeAnchor
    periodicity: Period


class AggregationFunction(ABC):
    """Base class for aggregation functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the aggregation function."""
        pass

    @staticmethod
    def expr(columns: list[str]) -> list[pl.Expr]:
        """Return the Polars expressions for this aggregation."""
        raise NotImplementedError

    @staticmethod
    def post_expr(columns: list[str]) -> list[pl.Expr]:
        """Return additional Polars expressions to be applied after the aggregation."""
        return []

    def expr_with_ctx(self, _ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """An AggregationCtx aware hook on the expr method, for child classes that require information
        from the context object. Delegates to the standard expr method if not implemented."""
        return self.expr(columns)

    def post_expr_with_ctx(self, _ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """An AggregationCtx aware hook on the post_expr method, for child classes that require information
        from the context object. Delegates to the standard post_expr method if not implemented."""
        return self.post_expr(columns)

    @classmethod
    def get(cls, aggregation: "str | Type[AggregationFunction] | AggregationFunction") -> "AggregationFunction":
        """Factory method to get an aggregation function instance from string names, class types, or existing instances.

        Args:
            aggregation: The aggregation specification, which can be:
                - A string name: e.g. "mean", "min", "max"
                - A class type: Mean, Min, Max, or any AggregationFunction subclass
                - An instance: Mean(), Min(), or any AggregationFunction instance

        Returns:
            An instance of the appropriate AggregationFunction subclass.

        Raises:
            UnknownAggregationError: If a string name is not registered as an aggregation function.
            AggregationTypeError: If the input type is not supported or class doesn't inherit from AggregationFunction.

        Examples:
            >>> # From string
            >>> agg = AggregationFunction.get("mean")
            >>>
            >>> # From class
            >>> agg = AggregationFunction.get(Mean)
            >>>
            >>> # From instance
            >>> agg = AggregationFunction.get(Mean())
        """
        # If it's already an instance, return it
        if isinstance(aggregation, AggregationFunction):
            return aggregation

        # If it's a string, look it up in the registry
        elif isinstance(aggregation, str):
            try:
                return _AGGREGATION_REGISTRY[aggregation]()
            except KeyError:
                raise UnknownAggregationError(
                    f"Unknown aggregation '{aggregation}'. Available aggregations: {list(_AGGREGATION_REGISTRY.keys())}"
                )

        # If it's a class, instantiate it
        elif isinstance(aggregation, type):
            if issubclass(aggregation, AggregationFunction):
                return aggregation()
            else:
                raise AggregationTypeError(
                    f"Aggregation class '{aggregation.__name__}' must inherit from AggregationFunction."
                )

        else:
            raise AggregationTypeError(
                f"Aggregation must be a string, AggregationFunction class, or instance. "
                f"Got {type(aggregation).__name__}."
            )

    def apply(
        self,
        ctx: AggregationCtx,
        aggregation_period: Period,
        columns: str | list[str],
        missing_criteria: tuple[str, float | int] | None = None,
    ) -> pl.DataFrame:
        """Run the aggregation.

        The general `Polars` method used for aggregating data is:

        >>> output_dataframe = input_dataframe
        >>>     .group_by_dynamic(...)  # Group the time series into groups of values based on the chosen period
        >>>     .agg(...)               # Apply the chosen aggregation function (e.g., mean, min, max)
        >>>     .with_columns(...)      # Apply additional logic that requires the results of the aggregation

        This method fills in the various stages using pre-defined expressions.

        Args:
            ctx: The context object containing dataframe and time properties required for aggregation.
            aggregation_period: The period over which to aggregate the data
            columns: The column(s) containing the data to be aggregated
            missing_criteria: How the aggregation handles missing data.  Tuple of (criteria name, value).

        Returns:
            The aggregated dataframe
        """
        # Handle multiple columns
        if isinstance(columns, str):
            columns = [columns]

        # Remove NULL rows
        df = ctx.df.drop_nulls(subset=columns)

        # Group by the aggregation period - taking into account the time anchor of the time series
        label, closed = self._get_label_closed(ctx.time_anchor)
        grouper = df.group_by_dynamic(
            index_column=ctx.time_name,
            every=aggregation_period.pl_interval,
            offset=aggregation_period.pl_offset,
            closed=closed,
            label=label,
        )

        # Build expressions to go in the .agg method
        agg_expressions = []
        agg_expressions.extend(self.expr_with_ctx(ctx, columns))
        agg_expressions.extend(self._actual_count_expr(columns))

        # Do the aggregation function
        df = grouper.agg(agg_expressions)

        # Build expressions to go in the .with_columns method.
        #   Note: Order is important here. Expressions may have dependencies on the results of earlier expressions.
        with_columns_expressions = []
        with_columns_expressions.append(self._expected_count_expr(ctx, aggregation_period))
        with_columns_expressions.extend(self._missing_data_expr(ctx, columns, missing_criteria))
        with_columns_expressions.extend(self.post_expr_with_ctx(ctx, columns))

        # Do the with_column methods
        for with_column in with_columns_expressions:
            df = df.with_columns(with_column)

        return df

    @staticmethod
    def _get_label_closed(time_anchor: TimeAnchor) -> tuple[Label, ClosedInterval]:
        """Map TimeAnchor to Polars label/closed semantics.

        Args:
            time_anchor: The time anchor to which the date/times conform to.

        Returns:
            Tuple of (label, closed) values to use in Polars operations
        """
        if time_anchor == TimeAnchor.END:
            return "right", "right"
        else:  # TimeAnchor.START, TimeAnchor.POINT
            return "left", "left"

    @staticmethod
    def _actual_count_expr(columns: list[str]) -> list[pl.Expr]:
        """A `Polars` expression to generate the actual count of values in a TimeSeries found in each period.

        Args:
            columns: The name of the value column(s) to count occurrences of

        Returns:
            List of `Polars` expressions that can be used to generate actual counts for each column
        """
        return [pl.col(col).count().alias(f"count_{col}") for col in columns]

    @staticmethod
    def _expected_count_expr(ctx: AggregationCtx, period: Period) -> pl.Expr:
        """A `Polars` expression to generate the expected count of values in a TimeSeries found in each
        period (if there were no missing values).

        Args:
            ctx: The context object containing dataframe and time properties required for aggregation.
            period: The Period over which to calculate expected counts

        Returns:
            pl.Expr: Polars expression that can be used to generate expected count on a DataFrame
        """
        expected_count_name = f"expected_count_{ctx.time_name}"
        # For some aggregations, the expected count is a constant so use that if possible.
        # For example, when aggregating 15-minute data over a day, the expected count is always 96.
        count = ctx.periodicity.count(period)
        if count > 0:
            return pl.lit(count).alias(expected_count_name)

        # Variable length periods need dynamic calculation, based on the start and end of a period interval
        label, closed = AggregationFunction._get_label_closed(ctx.time_anchor)
        if ctx.time_anchor == TimeAnchor.END:
            start_expr = pl.col(ctx.time_name).dt.offset_by("-" + period.pl_interval)
            end_expr = pl.col(ctx.time_name)
        else:  # TimeAnchor.START, TimeAnchor.POINT
            start_expr = pl.col(ctx.time_name)
            end_expr = pl.col(ctx.time_name).dt.offset_by(period.pl_interval)

        # This contains 2 cases:
        if ctx.periodicity.timedelta:
            # 1. If the data we are aggregating is not monthly, then each interval we are aggregating has
            #    a constant length, so (end - start) / interval will be the expected count.
            micros = ctx.periodicity.timedelta // timedelta(microseconds=1)
            return ((end_expr - start_expr).dt.total_microseconds() // micros).alias(expected_count_name)

        else:
            # 2. If the data we are aggregating is month-based, then there is no simple way to do the calculation,
            #    so use Polars to create a Series of date ranges lists and get the length of each list.
            #
            #    Note: This method will work for above use cases also, but if periodicity is small and the aggregation
            #    period is large, it consumes too much memory and causes performance problems. For example, aggregating
            #    1 microsecond data over a calendar year involves the creation length
            #    1000_000 * 60 * 60 * 24 * 365 arrays which will probably fail with an out-of-memory error.
            return (
                pl.datetime_ranges(start_expr, end_expr, interval=ctx.periodicity.pl_interval, closed=closed)
                .list.len()
                .alias(expected_count_name)
            )

    @staticmethod
    def _missing_data_expr(
        ctx: AggregationCtx, columns: list[str], missing_criteria: tuple[str, float | int] | None = None
    ) -> list[pl.Expr]:
        """Convert missing criteria to a Polars expression for validation.

        Args:
            ctx: The context object containing dataframe and time properties required for aggregation.
            columns: The name of the value column(s) to count occurrences of
            missing_criteria: The missing criteria to use

        Returns:
            List of `Polars` expressions that can be used to generate missing data validation columns
        """
        # Do some checks that the missing criteria is valid
        try:
            if missing_criteria is None:
                criteria, threshold = MissingCriteria.NA, 0
            else:
                criteria, threshold = missing_criteria

            criteria = MissingCriteria(criteria)
        except ValueError:
            raise MissingCriteriaError(
                f"Unknown missing criteria: {missing_criteria}. Available criteria: {[c.name for c in MissingCriteria]}"
            )

        if criteria == MissingCriteria.PERCENT:
            if not 0 <= threshold <= 100:
                raise MissingCriteriaError(f"Invalid percent threshold '{threshold}'. Must be between 0 and 100.")
        else:
            if not isinstance(threshold, int) or threshold < 0:
                raise MissingCriteriaError(f"Invalid threshold '{threshold}'. Must be a non-negative integer.")

        # Build the expression based on the specified criteria
        expressions = []
        for col in columns:
            if criteria == MissingCriteria.PERCENT:
                expr = ((pl.col(f"count_{col}") / pl.col(f"expected_count_{ctx.time_name}")) * 100) > threshold

            elif criteria == MissingCriteria.MISSING:
                expr = (pl.col(f"expected_count_{ctx.time_name}") - pl.col(f"count_{col}")) <= threshold

            elif criteria == MissingCriteria.AVAILABLE:
                expr = pl.col(f"count_{col}") >= threshold

            else:
                expr = pl.lit(True)

            expr = expr.alias(f"valid_{col}")
            expressions.append(expr)

        return expressions


@register_aggregation
class Mean(AggregationFunction):
    """An aggregation class to calculate the mean (average) of values within each aggregation period."""

    name = "mean"

    def expr(self, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the mean in an aggregation period."""
        return [pl.col(col).mean().alias(f"mean_{col}") for col in columns]


@register_aggregation
class Sum(AggregationFunction):
    """An aggregation class to calculate the sum (total) of values within each aggregation period."""

    name = "sum"

    def expr(self, columns: list[str]) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the mean in an aggregation period."""
        return [pl.col(col).sum().alias(f"sum_{col}") for col in columns]


@register_aggregation
class MeanSum(AggregationFunction):
    """An aggregation class to calculate the mean sum (averaged total) of values within each aggregation period.
    This will estimate the sum when values are missing according how many values are expected in the period."""

    name = "mean_sum"

    def expr(self, columns: list[str]) -> list[pl.Expr]:
        """To calculate the mean sum the expression must return the mean, and be multiplied by the expected
        counts, which is calculated after in the post_expr method."""
        return [pl.col(col).mean().alias(f"mean_sum_{col}") for col in columns]

    def post_expr_with_ctx(self, _ctx: AggregationCtx, columns: list[str]) -> list[pl.Expr]:
        """Multiply the mean by the expected count to get the mean sum."""
        return [(pl.col(f"mean_sum_{col}") * pl.col(f"expected_count_{_ctx.time_name}")) for col in columns]


@register_aggregation
class Min(AggregationFunction):
    """An aggregation class to find the minimum of values within each aggregation period."""

    name = "min"

    def expr_with_ctx(self, _ctx: AggregationCtx, columns: str) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the minimum in an aggregation period.
        This expression also returns a column that holds the datetime that the minimum value occurred on.
        """
        expressions = []
        for col in columns:
            expressions.extend(
                [
                    pl.col(col).min().alias(f"{self.name}_{col}"),
                    pl.col(_ctx.time_name).get(pl.col(col).arg_min()).alias(f"{_ctx.time_name}_of_{self.name}_{col}"),
                ]
            )
        return expressions


@register_aggregation
class Max(AggregationFunction):
    """An aggregation class to find the maximum of values within each aggregation period."""

    name = "max"

    def expr_with_ctx(self, _ctx: AggregationCtx, columns: str) -> list[pl.Expr]:
        """Return the `Polars` expression for calculating the maximum in an aggregation period.
        This expression also returns a column that holds the datetime that the maximum value occurred on.
        """
        expressions = []
        for col in columns:
            expressions.extend(
                [
                    pl.col(col).max().alias(f"{self.name}_{col}"),
                    pl.col(_ctx.time_name).get(pl.col(col).arg_max()).alias(f"{_ctx.time_name}_of_{self.name}_{col}"),
                ]
            )
        return expressions
