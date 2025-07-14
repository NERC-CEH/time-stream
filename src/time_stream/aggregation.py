from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Optional, Tuple, Type, Union

import polars as pl

from time_stream import Period, TimeSeries
from time_stream.enums import MissingCriteria

# Registry for built-in aggregations
_AGGREGATION_REGISTRY = {}


def register_aggregation(cls: Type["AggregationFunction"]) -> None:
    """Decorator to register aggregation classes using their name attribute.

    Args:
        cls: The aggregation class to register.
    """
    _AGGREGATION_REGISTRY[cls.name] = cls


class AggregationFunction(ABC):
    """Base class for aggregation functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the aggregation function."""
        pass

    @abstractmethod
    def expr(self, ts: TimeSeries, column: str) -> List[pl.Expr]:
        """Return the Polars expressions for this aggregation."""
        pass

    @classmethod
    def get(cls, aggregation: Union[str, Type["AggregationFunction"], "AggregationFunction"]) -> "AggregationFunction":
        """Factory method to get an aggregation function instance from string names, class types, or existing instances.

        Args:
            aggregation: The aggregation specification, which can be:
                - A string name: e.g. "mean", "min", "max"
                - A class type: Mean, Min, Max, or any AggregationFunction subclass
                - An instance: Mean(), Min(), or any AggregationFunction instance

        Returns:
            An instance of the appropriate AggregationFunction subclass.

        Raises:
            KeyError: If a string name is not registered as an aggregation function.
            TypeError: If the input type is not supported or a class doesn't inherit from AggregationFunction.

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
                raise KeyError(f"Unknown aggregation '{aggregation}'.")

        # If it's a class, instantiate it
        elif isinstance(aggregation, type):
            if issubclass(aggregation, AggregationFunction):
                return aggregation()
            else:
                raise TypeError(f"Aggregation class {aggregation.__name__} must inherit from AggregationFunction")

        else:
            raise TypeError(
                f"Aggregation must be a string, AggregationFunction class, or instance. "
                f"Got {type(aggregation).__name__}"
            )

    def apply(
        self,
        ts: TimeSeries,
        period: Period,
        column: str,
        missing_criteria: Optional[Tuple[str, Union[float, int]]] = None,
    ) -> TimeSeries:
        """Run the aggregation.

        Args:
            ts: The time series to aggregate.
            period: The period over which to aggregate the data
            column: The column containing the data to be aggregated
            missing_criteria: How the aggregation handles missing data.  Tuple of (criteria name, value).

        Returns:
            TimeSeries: The aggregated time series
        """
        # Validate that we can carry out the aggregation
        self._validate_aggregation_period(ts, period)

        # Remove NULL rows
        df = ts.df.drop_nulls(subset=[column])

        # Group by the aggregation period
        grouper = df.group_by_dynamic(
            index_column=ts.time_name, every=period.pl_interval, offset=period.pl_offset, closed="left"
        )

        # Build expressions to go in the .agg method
        agg_expressions = []
        agg_expressions.extend(self.expr(ts, column))
        agg_expressions.append(self._actual_count_expr(column))

        # Do the aggregation function
        df = grouper.agg(agg_expressions)

        # Build expressions to go in the .with_columns method
        with_columns_expressions = []
        with_columns_expressions.append(self._expected_count_expr(ts, period))
        with_columns_expressions.append(self._missing_data_expr(ts, column, missing_criteria))

        # Do the with_column methods
        for with_column in with_columns_expressions:
            df = df.with_columns(with_column)

        # Create result TimeSeries
        return TimeSeries(
            df=df,
            time_name=ts.time_name,
            resolution=period,
            periodicity=period,
            metadata=ts._metadata,
            pad=ts._pad,
        )

    def _validate_aggregation_period(self, ts: TimeSeries, period: Period) -> None:
        """Validate that the aggregation period is suitable based on the period of the TimeSeries.

        Args:
            ts: The TimeSeries object that the aggregation will be done on
            period: The aggregation period to check

        Raises:
            NotImplementedError: If period type is not supported
            UserWarning: If aggregation period incompatible with TimeSeries
        """
        if not period.is_epoch_agnostic():
            raise NotImplementedError(f"Non-epoch agnostic aggregation periods are not supported: {period}")

        if not ts.periodicity.is_subperiod_of(period):
            raise UserWarning(f"TimeSeries periodicity: {ts.periodicity} not subperiod of aggregation period: {period}")

    @staticmethod
    def _actual_count_expr(column: str) -> pl.Expr:
        """A `Polars` expression to generate the actual count of values in a TimeSeries found in each period.

        Args:
            column: The name of the value column to count occurrences of

        Returns:
            pl.Expr: Polars expression that can be used to generate actual count on a DataFrame
        """
        return pl.col(column).len().alias(f"count_{column}")

    @staticmethod
    def _expected_count_expr(ts: TimeSeries, period: Period) -> pl.Expr:
        """A `Polars` expression to generate the expected count of values in a TimeSeries found in each
        period (if there were no missing values).

        Args:
            ts: The TimeSeries object on which to calculate expected counts
            period: The Period over which to calculate expected counts

        Returns:
            pl.Expr: Polars expression that can be used to generate expected count on a DataFrame
        """
        expected_count_name = f"expected_count_{ts.time_name}"
        # For some aggregations, the expected count is a constant so use that if possible.
        # For example, when aggregating 15-minute data over a day, the expected count is always 96.
        count = ts.periodicity.count(period)
        if count > 0:
            return pl.lit(count).alias(expected_count_name)

        # Variable length periods need dynamic calculation, based on the start and end of a period interval
        start_expr = pl.col(ts.time_name)
        end_expr = pl.col(ts.time_name).dt.offset_by(period.pl_interval)

        # This contains 2 cases:
        if ts.periodicity.timedelta:
            # 1. If the data we are aggregating is not monthly, then each interval we are aggregating has
            #    a constant length, so (end - start) / interval will be the expected count.
            micros = ts.periodicity.timedelta // timedelta(microseconds=1)
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
                pl.datetime_ranges(start_expr, end_expr, interval=ts.periodicity.pl_interval, closed="right")
                .list.len()
                .alias(expected_count_name)
            )

    @staticmethod
    def _missing_data_expr(
        ts: TimeSeries, column: str, missing_criteria: Optional[Tuple[str, Union[float, int]]] = None
    ) -> pl.Expr:
        """Convert missing criteria to a Polars expression for validation.

        Args:
            ts: The TimeSeries object on which to calculate missing data
            column: The name of the value column to count occurrences of
            missing_criteria: The missing criteria to use

        Returns:
            pl.Expr: Polars expression that can be used to generate missing data validation column
        """
        # Do some checks that the missing criteria is valid
        try:
            if missing_criteria is None:
                criteria, threshold = MissingCriteria.NA, 0
            else:
                criteria, threshold = missing_criteria

            criteria = MissingCriteria(criteria)
        except ValueError:
            raise ValueError(f"Unknown criterion: {missing_criteria}")

        if criteria == MissingCriteria.PERCENT:
            if not 0 <= threshold <= 100:
                raise ValueError("Percent threshold must be between 0 and 100")

        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # Build the expression based on the specified criteria
        if criteria == MissingCriteria.PERCENT:
            expr = ((pl.col(f"count_{column}") / pl.col(f"expected_count_{ts.time_name}")) * 100) > threshold

        elif criteria == MissingCriteria.MISSING:
            expr = (pl.col(f"expected_count_{ts.time_name}") - pl.col(f"count_{column}")) < threshold

        elif criteria == MissingCriteria.AVAILABLE:
            expr = pl.col(f"count_{column}") > threshold

        else:
            expr = pl.lit(True)

        return expr.alias(f"valid_{column}")


@register_aggregation
class Mean(AggregationFunction):
    """An aggregation class to calculate the mean (average) of values within each aggregation period."""

    name = "mean"

    def expr(self, ts: TimeSeries, column: str) -> List[pl.Expr]:
        """Return the `Polars` expression for calculating the mean in an aggregation period."""
        return [pl.col(column).mean().alias(f"mean_{column}")]


@register_aggregation
class Min(AggregationFunction):
    """An aggregation class to find the minimum of values within each aggregation period."""

    name = "min"

    def expr(self, ts: TimeSeries, column: str) -> List[pl.Expr]:
        """Return the `Polars` expression for calculating the minimum in an aggregation period.
        This expression also returns a column that holds the datetime that the minimum value occurred on.
        """
        return [
            pl.col(column).min().alias(f"{self.name}_{column}"),
            # Define a struct, to be able to extract the datetime on which the min occurred
            pl.struct([ts.time_name, column])
            .sort_by(column)
            .first()
            .struct.field(ts.time_name)
            .alias(f"{ts.time_name}_of_{self.name}"),
        ]


@register_aggregation
class Max(AggregationFunction):
    """An aggregation class to find the maximum of values within each aggregation period."""

    name = "max"

    def expr(self, ts: TimeSeries, column: str) -> List[pl.Expr]:
        """Return the `Polars` expression for calculating the maximum in an aggregation period.
        This expression also returns a column that holds the datetime that the maximum value occurred on.
        """
        return [
            pl.col(column).max().alias(f"{self.name}_{column}"),
            # Define a struct, to be able to extract the datetime on which the max occurred
            pl.struct([ts.time_name, column])
            .sort_by(column)
            .last()
            .struct.field(ts.time_name)
            .alias(f"{ts.time_name}_of_{self.name}"),
        ]
