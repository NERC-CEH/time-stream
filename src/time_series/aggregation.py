"""
Time-series aggregation

This module is very much a work-in-progress and the classes
contained within will evolve considerably.
"""

from abc import ABC
from collections.abc import (
    Callable,
)
from typing import override

import polars as pl

from time_series.period import Period
from time_series.time_series_base import AggregationFunction, TimeSeries
from time_series.time_series_polars import TimeSeriesPolars

# A function that takes a Polars GroupBy as an argument and returns a DataFrame
GroupByToDataFrame = Callable[[pl.dataframe.group_by.GroupBy], pl.DataFrame]


def _group_by_first(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    """A GroupByToDataFrame that calls .first()"""
    return group_by.first()


def _group_by_last(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    """A GroupByToDataFrame that calls .last()"""
    return group_by.last()


def _group_by_mean(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    """A GroupByToDataFrame that calls .mean()"""
    return group_by.mean()


class PolarsAggregator:
    """A class used to assist with aggregating data

    Attributes:
        ts: The TimeSeries being aggregated
        aggregation_period: The Period we are aggregating over
    """

    def __init__(self, ts: TimeSeries, aggregation_period: Period) -> None:
        self.ts = ts
        self.aggregation_period = aggregation_period

    def aggregate(self, stage_list: list["AggregationStage"]) -> pl.DataFrame:
        """Create a new DataFrame containing aggregated data that is produced by a list of AggregationStages

        The general Polars method used for aggregating data is:

            output_dataframe = ( input_dataframe
                .group_by_dynamic( various stuff goes here )
                .agg( ... )
                .unnest( ... )
                .with_columns( ... )
            )

        The AggregationStages are used to fill in the "..." parts.

        Args:
            stage_list: A list of AggregationStage objects

        Returns:
            A DataFrame containing the aggregated data
        """
        aggregation_expression_list: list[pl.Expr] = []
        unnest_list: list[str] = []
        with_column_list: list[pl.Expr] = []
        for aggregation_stage in stage_list:
            aggregation_expression_list.extend(aggregation_stage.aggregation_expressions())
            unnest_list.extend(aggregation_stage.unnest())
            with_column_list.extend(aggregation_stage.with_columns())

        group_by_dynamic = self.ts.df.group_by_dynamic(
            index_column=self.ts.time_name,
            every=self.aggregation_period.pl_interval,
            offset=self.aggregation_period.pl_offset,
            closed="left",
        )

        df = group_by_dynamic.agg(aggregation_expression_list)

        if unnest_list:
            df = df.unnest(*unnest_list)
        if with_column_list:
            df = df.with_columns(*with_column_list)

        return df


class AggregationStage(ABC):
    """An abstract class used to assist with aggregating data

    Each subclass will add various things to various stages of the aggregation process

    Attributes:
        aggregator: The PolarsAggregator being aggregated
        name: The name of this aggregator
    """

    def __init__(self, aggregator: PolarsAggregator, name: str) -> None:
        self.aggregator = aggregator
        self.name = name

    def aggregation_expressions(self) -> list[pl.Expr]:
        """Return a list of expressions to be applied during the aggregation stage.

        These expressions are passed to the Polars .agg() method.

        Subclasses override this method to provide a non-empty return list.

        Returns:
            A list of Polars expressions
        """
        return []

    def unnest(self) -> list[str]:
        """Return a list of structs to be unnested after the aggregation stage.

        These struct names are passed to the Polars DataFrame.unnest() method.

        Subclasses override this method to provide a non-empty return list.

        Returns:
            A list of structs to be unnested
        """
        return []

    def with_columns(self) -> list[pl.Expr]:
        """Return a list of expressions to be included in the final aggregated DataFrame.

        These expressions are passed to the Polars DataFrame.with_columns() method.

        Subclasses override this method to provide a non-empty return list.

        Returns:
            A list of expressions to be included in the final aggregated DataFrame
        """
        return []


class ExpectedCount(AggregationStage):
    """An AggregationStage

    Creates a "expected_count_{time_name}" column containing a count of the expected number of elements in each period.
    """

    def __init__(self, aggregator: PolarsAggregator) -> None:
        super().__init__(aggregator, "datetime_count")

    def with_columns(self) -> list[pl.Expr]:
        time_name = self.aggregator.ts.time_name
        start_expr: pl.Expr = pl.col(time_name)
        end_expr: pl.Expr = pl.col(time_name).dt.offset_by(self.aggregator.aggregation_period.pl_interval)
        # NEED TO REPLACE THIS CODE WITH SOMETHING ELSE IF POSSIBLE
        # The following instantiates a list and then just gets the length.
        # The list is as long as there are 'resolution' units within the
        # aggregation period, which can lead to some very long lists
        # P1S aggregated over P1Y gives a list length of 365*24*60*60
        # Some aggregation operations are very slow or just crash with
        # out-of-memory errors.
        return [
            pl.datetime_ranges(
                start=start_expr, end=end_expr, interval=self.aggregator.ts.periodicity.pl_interval, closed="right"
            )
            .list.len()
            .alias(f"expected_count_{time_name}")
        ]


class ActualValueCount(AggregationStage):
    """An AggregationStage

    Creates a "count_{value_column}" column containing a count of the actual number of elements found in each period.

    Attributes:
        value_column: The name of the value column
    """

    def __init__(self, aggregator: PolarsAggregator, value_column: str) -> None:
        super().__init__(aggregator, "value_count")
        self._value_column = value_column

    def aggregation_expressions(self) -> list[pl.Expr]:
        return [pl.col(self._value_column).len().alias(f"count_{self._value_column}")]


class GroupByWithDateTime(AggregationStage):
    """An AggregationStage

    Creates a struct called "{name}_struct" containing an aggregated
    value and a datetime value associated with that value.

    Attributes:
        name: A name to be used for the created struct and the columns in it
        gb2df: A GroupByToDataFrame function to be applied to create the values in the aggregated DataFrame
        value_column: The name of the value column
    """

    def __init__(self, aggregator: PolarsAggregator, name: str, gb2df: GroupByToDataFrame, value_column: str) -> None:
        super().__init__(aggregator, name)
        self._gb2df = gb2df
        self._value_column = value_column
        self._struct_name: str = f"{name}_struct"

    def aggregation_expressions(self) -> list[pl.Expr]:
        struct_columns: list[str] = [self.aggregator.ts.time_name, self._value_column]
        return [
            self._gb2df(pl.struct(struct_columns).sort_by(self._value_column))
            .alias(self._struct_name)
            .struct.rename_fields(
                [f"{self.aggregator.ts.time_name}_of_{self.name}", f"{self.name}_{self._value_column}"]
            )
        ]

    def unnest(self) -> list[str]:
        return [self._struct_name]


class GroupByBasic(AggregationStage):
    """An AggregationStage

    Creates a "{name}_{value_column}" column containing aggregated values

    Attributes:
        name: A name to be used for the created column
        gb2df: A GroupByToDataFrame function to be applied to create the values in the aggregated DataFrame
        value_column: The name of the value column
    """

    def __init__(self, aggregator: PolarsAggregator, name: str, gb2df: GroupByToDataFrame, value_column: str) -> None:
        super().__init__(aggregator, name)
        self._gb2df = gb2df
        self._value_column = value_column

    def aggregation_expressions(self) -> list[pl.Expr]:
        return [self._gb2df(pl.col(self._value_column)).alias(f"{self.name}_{self._value_column}")]


class Mean(AggregationFunction):
    """A mean AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{time_name}"
            The start of the aggregation period
        "mean_{value_column}"
            The arithmetic mean of all values in each aggregation period
        "count_{value_column}"
            The number of values found in each aggregation period
        count_{time_name}"
            The maximum number of possible values in each aggregation period

    """

    def __init__(self) -> None:
        super().__init__("mean")

    @override
    def apply(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByBasic(aggregator, "mean", _group_by_mean, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )


class Min(AggregationFunction):
    """A min AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{time_name}"
            The start of the aggregation period
        "min_{value_column}"
            The minimum value in each aggregation period
        "{time_name}_of_min"
            The datetime of the minimum value
        "count_{value_column}"
            The number of values found in each aggregation period
        count_{time_name}"
            The maximum number of possible values in each aggregation period

    """

    def __init__(self) -> None:
        super().__init__("min")

    @override
    def apply(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "min", _group_by_first, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=ts.resolution,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )


class Max(AggregationFunction):
    """A max AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{time_name}"
            The start of the aggregation period
        "max_{value_column}"
            The maximum value in each aggregation period
        "{time_name}_of_max"
            The datetime of the maximum value
        "count_{value_column}"
            The number of values found in each aggregation period
        count_{time_name}"
            The maximum number of possible values in each aggregation period

    """

    def __init__(self) -> None:
        super().__init__("max")

    @override
    def apply(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "max", _group_by_last, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=ts.resolution,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )
