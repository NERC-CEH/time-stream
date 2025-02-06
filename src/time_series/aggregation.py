"""
Time-series aggregation

This module is very much a work-in-progress and the classes
contained within will evolve considerably.
"""

import datetime
from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
    Optional,
)
from typing import override

import polars as pl

from time_series import Period, TimeSeries

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
        # For some aggregations the expected count is a constant so just use that if possible.
        # For example, when aggregating 15-minute data over a day, the expected count is always 96.
        count: int = self.aggregator.ts.periodicity.count(self.aggregator.aggregation_period)
        if count > 0:
            return [pl.lit(count).alias(f"expected_count_{time_name}")]

        # If the data we are aggregating is not monthly then each interval we are aggregating has
        # a constant length, so (end - start) / interval will be the expected count.
        delta: Optional[datetime.timedelta] = self.aggregator.ts.periodicity.timedelta
        if delta is not None:
            i_micros: int = delta // datetime.timedelta(microseconds=1)
            return [
                ((end_expr - start_expr).dt.total_microseconds().floordiv(i_micros)).alias(
                    f"expected_count_{time_name}"
                )
            ]

        # If the data we are aggregating is monthly then there is no simple way to do the calculation,
        # so use Polars to create a Series of lists of date ranges and just get the length of each list.
        #
        # This method will work for the above two cases also, but if periodicity is small
        # and the aggregation period is large it consumes too much memory and causes performance
        # problems.
        #
        # For example, aggregating 1 microsecond data over a calendar year involves the creation
        # of arrays of length 1000_000 * 60 * 60 * 24 * 365 which will probably fail with an
        # out-of-memory error.
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


class AggregationFunction(ABC):
    """An aggregation function that can be applied to a field
    in a TimeSeries.

    A new aggregated TimeSeries can be created from an existing
    TimeSeries by passing a subclass of AggregationFunction
    into the TimeSeries.aggregate method.

    Attributes:
        name: The name of the aggregation function
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this aggregation function"""
        return self._name

    @abstractmethod
    def apply(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> TimeSeries:
        """Apply this aggregation function to the supplied
        TimeSeries column and return a new TimeSeries containing
        the aggregated data

        Note: This is the first attempt at a mechanism for aggregating
        time-series data.  The signature of this method is likely to
        evolve considerably.

        Args:
            ts: The TimeSeries containing the data to be aggregated
            aggregation_period: The time period over which to aggregate
            column_name: The column containing the data to be aggregated

        Returns:
            A TimeSeries containing the aggregated data
        """
        raise NotImplementedError

    @classmethod
    def create(cls) -> "AggregationFunction":
        raise NotImplementedError


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

    @classmethod
    def create(cls) -> "AggregationFunction":
        return cls("mean")

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByBasic(aggregator, "mean", _group_by_mean, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
            ]
        )

        # Aggregator just returns a dataframe with the selected column in. This might need to change when considering
        #   linked supp/flag columns.  But for now, not sending any lists/dicts for supp or flag columns or flag systems
        return TimeSeries(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
            flag_systems=ts.flag_systems,
            column_metadata={column_name: ts.columns[column_name].metadata()},
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

    @classmethod
    def create(cls) -> "AggregationFunction":
        return cls("min")

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "min", _group_by_first, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
            ]
        )
        # Aggregator just returns a dataframe with the selected column in. This might need to change when considering
        #   linked supp/flag columns.  But for now, not sending any lists/dicts for supp or flag columns or flag systems
        return TimeSeries(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
            flag_systems=ts.flag_systems,
            column_metadata={column_name: ts.columns[column_name].metadata()},
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

    @classmethod
    def create(cls) -> "AggregationFunction":
        return cls("max")

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "max", _group_by_last, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
            ]
        )
        # Aggregator just returns a dataframe with the selected column in. This might need to change when considering
        #   linked supp/flag columns.  But for now, not sending any lists/dicts for supp or flag columns or flag systems
        return TimeSeries(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
            flag_systems=ts.flag_systems,
            column_metadata={column_name: ts.columns[column_name].metadata()},
        )
