"""
Time-series aggregation

This module is very much a work-in-progress and the classes
contained within will evolve considerably.
"""

from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
)
from typing import override

import polars as pl

from time_series.period import Period
from time_series.time_series_base import AggregationFunction, TimeSeries
from time_series.time_series_polars import TimeSeriesPolars

# A function that tkes a Polars GroupBy as an argument and
# returns a DataFrame
GroupByToDataFrame = Callable[[pl.dataframe.group_by.GroupBy], pl.DataFrame]


def _gb_first(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    """A GroupByToDataFrame that calls .first()"""
    return group_by.first()


def _gb_last(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    """A GroupByToDataFrame that calls .last()"""
    return group_by.last()


def _gb_mean(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    """A GroupByToDataFrame that calls .mean()"""
    return group_by.mean()


# ----------------------------------------------------------------------
class PolarsAggregator:
    """A class used to assist with aggregating data

    Attributes:
        ts: The TimeSeries being aggregated
        aggregation_period: The Period we are aggregating over
    """

    def __init__(self, ts: TimeSeries, aggregation_period: Period) -> None:
        self.ts = ts
        self.aggregation_period = aggregation_period
        self.datetime_column_name: str = ts.time_name
        #       self.resolution_interval: str = ts.resolution.pl_interval
        #       self.resolution_offset: str = ts.resolution.pl_offset
        self.periodicity_interval: str = ts.periodicity.pl_interval
        #       self.periodicity_offset: str = ts.periodicity.pl_offset
        self.aggregation_interval: str = aggregation_period.pl_interval
        self.aggregation_offset: str = aggregation_period.pl_offset

    def aggregate(self, dfa_list: list["DataFrameAggregator"]) -> pl.DataFrame:
        """Create a new DataFrame containing aggregated data
        that is produced by a list of DataFrameAggregators

        The general Polars method used for aggregating data
        is:

            output_dataframe = ( input_dataframe
                .group_by_dynamic( various stuff goes here )
                .agg( ... )
                .unnest( ... )
                .with_columns( ... )
            )

        The DataFrameAggregators are used to fill in the "..."
        parts.

        Args:
            dfa_list: A list of DataFrameAggregator objects

        Returns:
            A DataFrame containing the aggregated data
        """
        expression_list: list[pl.Expr] = []
        unnest_list: list[str] = []
        with_column_list: list[str] = []
        for dfa in dfa_list:
            expression_list.extend(dfa.expressions())
            unnest_list.extend(dfa.unnest())
            with_column_list.extend(dfa.with_columns())

        group_by_dynamic = self.ts.df.group_by_dynamic(
            index_column=self.datetime_column_name,
            every=self.aggregation_interval,
            offset=self.aggregation_offset,
            closed="left",
        )

        def unnest(df: pl.DataFrame) -> pl.DataFrame:
            if len(unnest_list) == 0:
                return df
            return df.unnest(*unnest_list)

        def with_column(df: pl.DataFrame) -> pl.DataFrame:
            if len(with_column_list) == 0:
                return df
            return df.with_columns(*with_column_list)

        return with_column(unnest(group_by_dynamic.agg(expression_list)))


class DataFrameAggregator(ABC):
    """An abstract class used to assist with aggregating data

    Each subclass will add various things to various stages
    of the aggregation process

    Attributes:
        aggregator: The PolarsAggregator being aggregated
        name: The name of this aggregator
    """

    def __init__(self, aggregator: PolarsAggregator, name: str) -> None:
        self.aggregator = aggregator
        self.name = name

    @property
    def datetime_column_name(self) -> str:
        """The column name of the primary datetime field"""
        return self.aggregator.datetime_column_name

    def expressions(self) -> list[pl.Expr]:
        """Return a list of expressions to be applied during the
        aggregation stage.

        These expressions are passed to the Polars .agg() method.

        Subclasses override this method to provide
        a non-empty return list.

        Returns:
            A list of Polars expressions
        """
        return []

    def unnest(self) -> list[str]:
        """Return a list of structs to be unnested after the
        aggregation stage.

        These struct names are passed to the Polars DataFrame.unnest()
        method.

        Subclasses override this method to provide
        a non-empty return list.

        Returns:
            A list of structs to be unnested
        """
        return []

    def with_columns(self) -> list[pl.Expr]:
        """Return a list of expressions to be included in the final
        aggregated DataFrame.

        These expressions are passed to the Polars
        DataFrame.with_columns() method.

        Subclasses override this method to provide
        a non-empty return list.

        Returns:
            A list of expressions to be included in the final
            aggregated DataFrame
        """
        return []


class DateTimeCount(DataFrameAggregator):
    """A DataFrameAggregator

    Creates a "count_{datetime_column}" column containing
    a count of the expected number of elements in each
    period.
    """

    def __init__(self, aggregator: PolarsAggregator) -> None:
        super().__init__(aggregator, "datetime_count")

    def with_columns(self) -> list[pl.Expr]:
        datetime_column = self.aggregator.datetime_column_name
        start_expr: pl.Expr = pl.col(datetime_column)
        end_expr: pl.Expr = pl.col(datetime_column).dt.offset_by(self.aggregator.aggregation_interval)
        # NEED TO REPLACE THIS CODE WITH SOMETHING ELSE IF POSSIBLE
        # The following instantiates a list and then just gets the length.
        # The list is as long as there are 'resolution' units within the
        # aggregation period, which can lead to some very long lists
        # P1S aggregated over P1Y gives a list length of 365*24*60*60
        # Some aggregation operations are very slow or just crash with
        # out-of-memory errors.
        return [
            pl.datetime_ranges(
                start=start_expr, end=end_expr, interval=self.aggregator.periodicity_interval, closed="right"
            )
            .list.len()
            .alias(f"count_{datetime_column}")
        ]


class ValueCount(DataFrameAggregator):
    """A DataFrameAggregator

    Creates a "count_{value_column}" column containing
    a count of the actual number of elements found in each
    period.

    This class probably needs some work as it does not itself
    refer to value_column.

    Attributes:
        value_column: The name of the value column
    """

    def __init__(self, aggregator: PolarsAggregator, value_column: str) -> None:
        super().__init__(aggregator, "value_count")
        self._value_column = value_column

    def expressions(self) -> list[pl.Expr]:
        value_column: str = self._value_column
        return [pl.len().alias(f"count_{value_column}")]


class GroupByWithDateTime(DataFrameAggregator):
    """A DataFrameAggregator

    Creates a struct called "{name}_struct" containing an aggregated
    value and a datetime value associated with that value.

    Attributes:
        name: A name to be used for the created struct and
              the columns in it
        gb2df: A GroupByToDataFrame function to be applied
               to create the values in the aggregated DataFrame
        value_column: The name of the value column
    """

    def __init__(self, aggregator: PolarsAggregator, name: str, gb2df: GroupByToDataFrame, value_column: str) -> None:
        super().__init__(aggregator, name)
        self._gb2df = gb2df
        self._value_column = value_column
        self._struct_name: str = f"{name}_struct"

    def expressions(self) -> list[pl.Expr]:
        name: str = self.name
        value_column: str = self._value_column
        struct_columns: list[str] = [self.aggregator.datetime_column_name, value_column]
        return [
            self._gb2df(pl.struct(struct_columns).sort_by(value_column))
            .alias(self._struct_name)
            .struct.rename_fields([f"{self.datetime_column_name}_of_{name}", f"{name}_{value_column}"])
        ]

    def unnest(self) -> list[str]:
        return [self._struct_name]


class GroupByBasic(DataFrameAggregator):
    """A DataFrameAggregator

    Creates a "{name}_{value_column}" column containing aggregated
    values

    Attributes:
        name: A name to be used for the created column
        gb2df: A GroupByToDataFrame function to be applied
               to create the values in the aggregated DataFrame
        value_column: The name of the value column
    """

    def __init__(self, aggregator: PolarsAggregator, name: str, gb2df: GroupByToDataFrame, value_column: str) -> None:
        super().__init__(aggregator, name)
        self._gb2df = gb2df
        self._value_column = value_column

    def expressions(self) -> list[pl.Expr]:
        name: str = self.name
        value_column: str = self._value_column
        return [self._gb2df(pl.col(value_column)).alias(f"{name}_{value_column}")]

    def unnest(self) -> list[str]:
        return []

    def with_columns(self) -> list[pl.Expr]:
        return []


# ----------------------------------------------------------------------
class PolarsAggregationFunction(AggregationFunction):
    """An AggregationFunction that can be applied
    to a TimeSeriesPolars

    Attributes:
        name: The name of the aggregation function
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> TimeSeries:
        if not isinstance(ts, TimeSeriesPolars):
            raise NotImplementedError()
        return self.pl_agg(ts, aggregation_period, column_name)

    @abstractmethod
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        """Apply this function to a time-series over a period"""
        raise NotImplementedError()


class Mean(PolarsAggregationFunction):
    """A mean AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{datetime_column}"
            The start of the aggregation period
        "mean_{value_column}"
            The arithmetic mean of all values in each aggregation period
        "count_{value_column}"
            The number of values found in each aggregation period
        count_{datetime_column}"
            The maximum number of possible values in each aggregation period

    """

    def __init__(self) -> None:
        super().__init__("mean")

    @override
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByBasic(aggregator, "mean", _gb_mean, column_name),
                ValueCount(aggregator, column_name),
                DateTimeCount(aggregator),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )


class Min(PolarsAggregationFunction):
    """A min AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{datetime_column}"
            The start of the aggregation period
        "min_{value_column}"
            The minimum value in each aggregation period
        "{datetime_column}_of_min"
            The datetime of the minimum value
        "count_{value_column}"
            The number of values found in each aggregation period
        count_{datetime_column}"
            The maximum number of possible values in each aggregation period

    """

    def __init__(self) -> None:
        super().__init__("min")

    @override
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "min", _gb_first, column_name),
                ValueCount(aggregator, column_name),
                DateTimeCount(aggregator),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=ts.resolution,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )


class Max(PolarsAggregationFunction):
    """A max AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{datetime_column}"
            The start of the aggregation period
        "max_{value_column}"
            The maximum value in each aggregation period
        "{datetime_column}_of_max"
            The datetime of the maximum value
        "count_{value_column}"
            The number of values found in each aggregation period
        count_{datetime_column}"
            The maximum number of possible values in each aggregation period

    """

    def __init__(self) -> None:
        super().__init__("max")

    @override
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "max", _gb_last, column_name),
                ValueCount(aggregator, column_name),
                DateTimeCount(aggregator),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=ts.resolution,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )
