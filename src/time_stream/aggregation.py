"""
Time-series aggregation

This module is very much a work-in-progress and the classes
contained within will evolve considerably.
"""

import datetime
from abc import ABC
from collections.abc import Callable
from typing import Any, Mapping, Optional, Union, override

import polars as pl

from time_stream import Period, TimeSeries
from time_stream.aggregation_base import AggregationFunction
from time_stream.enums import MissingCriteriaOptions

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
        column_name: The column we are aggregating
    """

    def __init__(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> None:
        self.ts = ts
        self.aggregation_period = aggregation_period
        self.column_name = column_name

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
        # Removing NULL rows from the DataFrame for the specified value column
        df = self.ts.df.drop_nulls(subset=[self.column_name])

        aggregation_expression_list: list[pl.Expr] = []
        unnest_list: list[str] = []
        with_column_list: list[pl.Expr] = []
        for aggregation_stage in stage_list:
            aggregation_expression_list.extend(aggregation_stage.aggregation_expressions())
            unnest_list.extend(aggregation_stage.unnest())
            with_column_list.extend(aggregation_stage.with_columns())

        group_by_dynamic = df.group_by_dynamic(
            index_column=self.ts.time_name,
            every=self.aggregation_period.pl_interval,
            offset=self.aggregation_period.pl_offset,
            closed="left",
        )

        df = group_by_dynamic.agg(aggregation_expression_list)

        if unnest_list:
            df = df.unnest(*unnest_list)
        if with_column_list:
            for with_column in with_column_list:
                df = df.with_columns(with_column)

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
        """Return a list of expressions to be included in the final aggregated DataFrame.

        Add column "expected_count_{time_name}", which is the number of values that would be aggregated in each interval
        if the input timeseries was complete.
        """
        time_name = self.aggregator.ts.time_name
        column_name: str = f"expected_count_{time_name}"
        # For some aggregations the expected count is a constant so just use that if possible.
        # For example, when aggregating 15-minute data over a day, the expected count is always 96.
        count: int = self.aggregator.ts.periodicity.count(self.aggregator.aggregation_period)
        if count > 0:
            return [pl.lit(count).alias(column_name)]

        start_expr: pl.Expr = pl.col(time_name)
        end_expr: pl.Expr = pl.col(time_name).dt.offset_by(self.aggregator.aggregation_period.pl_interval)
        # If the data we are aggregating is not monthly then each interval we are aggregating has
        # a constant length, so (end - start) / interval will be the expected count.
        delta: Optional[datetime.timedelta] = self.aggregator.ts.periodicity.timedelta
        if delta is not None:
            i_micros: int = delta // datetime.timedelta(microseconds=1)
            return [((end_expr - start_expr).dt.total_microseconds().floordiv(i_micros)).alias(column_name)]

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
            .alias(column_name)
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


class ValidAggregation(AggregationStage):
    """An AggregationStage

    Creates a "valid" column containing a boolean of whether the aggregation meets the missing criteria.

    Set to true by default if no missing_criteria options.

    Attributes:
        value_column: The name of the value column
        missing_criteria: What level of missing data is acceptable
    """

    def __init__(
        self,
        aggregator: PolarsAggregator,
        value_column: str,
        missing_criteria: Union[None, Mapping[str, Union[float, int]]],
    ) -> None:
        super().__init__(aggregator, "valid_boolean")
        self._value_column = value_column
        self._missing_criteria = missing_criteria
        self.ts = aggregator.ts

    def _percent(self, column_name: str, limit: Union[int, float]) -> pl.Expr:
        """Check whether the percent of non-missing data satisfies user criteria.

        Aggregation is valid if the percent of non-missing data is greater than the limit.

        Args:
            column_name: Name of the aggregated column
            limit: the lowest percent of required non-missing data

        Returns:
            A polars expression
        """
        expression = (pl.col(f"count_{column_name}") / pl.col(f"expected_count_{self.ts.time_name}")) * 100

        return pl.when(expression > limit).then(True).otherwise(False).alias("valid")

    def _missing(self, column_name: str, limit: int) -> pl.Expr:
        """Check whether the count of missing data satisfies user criteria.

        Aggregation is valid if the count of missing data is less than the limit.

        Args:
            column_name: Name of the aggregated column
            limit: the highest count of missing data

        Returns:
            A polars expression
        """
        expression = pl.col(f"expected_count_{self.ts.time_name}") - pl.col(f"count_{column_name}")

        return pl.when(expression < limit).then(True).otherwise(False).alias("valid")

    def _available(self, column_name: str, limit: int) -> pl.Expr:
        """Check whether the count of non-missing data satisfies user criteria.

        Aggregation is valid if the count of non-missing data is greater than the limit.

        Args:
            column_name: Name of the aggregated column
            limit: the lowest count of non-missing data

        Returns:
            A polars expression
        """
        expression = pl.col(f"count_{column_name}")

        return pl.when(expression > limit).then(True).otherwise(False).alias("valid")

    def _validate_missing_aggregation_criteria(self, missing_criteria: Any) -> Mapping[str, Union[str | int | float]]:
        """Validate user input on how to handle missing data in the aggregation.

        Should be a single item dictionary with one of the following keys:

        missing: Calculate a value only if there are no more than n values missing in the period.
        available: Calculate a value only if there are at least n input values in the period.
        percent: Calculate a value only if the data in the period is at least n percent complete.

        Args:
            missing_criteria: what level of missing data is acceptable.

        Returns:
            modified dictionary of missing criteria.
        """
        if not isinstance(missing_criteria, dict):
            raise ValueError(f"missing_criteria argument should be a dictionary, not {type(missing_criteria)}")

        if len(missing_criteria) != 1:
            raise ValueError(f"missing_criteria argument should contain only one key, not {len(missing_criteria)}")

        supplied_key = list(missing_criteria.keys())[0]

        if supplied_key not in MissingCriteriaOptions:
            raise KeyError(
                f"missing_criteria option should be one of"
                f"{[options.value for options in list(MissingCriteriaOptions)]} "
                f"not '{supplied_key}'"
            )

        return {"method": f"_{supplied_key}", "limit": missing_criteria[supplied_key]}

    def validate_aggregation(
        self, column_name: str, missing_criteria: Mapping[str, Union[int | float]]
    ) -> pl.DataFrame:
        """Check the aggregated dataframe satisfies missing value criteria.

        Args:
            column_name: Name of the column to aggregate
            missing_criteria: What level of missing data is acceptable

        Returns:
            A dataframe containing the aggregated data.
        """
        missing_criteria = self._validate_missing_aggregation_criteria(missing_criteria)
        return getattr(self, missing_criteria["method"])(column_name, missing_criteria["limit"])

    def with_columns(self) -> list[None | pl.Expr]:
        """Return "valid" column to be included in the final dataframe."""
        expression = []
        if self._missing_criteria is not None:
            expression.append(self.validate_aggregation(self._value_column, self._missing_criteria))
        else:
            expression.append(pl.lit(True).alias("valid"))

        return expression


class Mean(AggregationFunction):
    """A mean AggregationFunction

    Creates an aggregated DataFrame with the following columns:

        "{time_name}"
            The start of the aggregation period
        "mean_{value_column}"
            The arithmetic mean of all values in each aggregation period
        "count_{value_column}"
            The number of values found in each aggregation period
        "expected_count_{time_name}"
            The maximum number of possible values in each aggregation period
        "valid"
            Weather the aggregation is valid against the specified missing_criteria

    """

    @classmethod
    def create(cls) -> "AggregationFunction":
        return cls("mean")

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(
        self,
        ts: TimeSeries,
        aggregation_period: Period,
        column_name: str,
        missing_criteria: Optional[Mapping[str, Union[float, int]]] = None,
    ) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period, column_name)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByBasic(aggregator, "mean", _group_by_mean, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
                ValidAggregation(aggregator, column_name, missing_criteria),
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
            pad=ts._pad,
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
        "expected_count_{time_name}"
            The maximum number of possible values in each aggregation period
        "valid"
            Weather the aggregation is valid against the specified missing_criteria

    """

    @classmethod
    def create(cls) -> "AggregationFunction":
        return cls("min")

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(
        self,
        ts: TimeSeries,
        aggregation_period: Period,
        column_name: str,
        missing_criteria: Optional[Mapping[str, Union[float, int]]] = None,
    ) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period, column_name)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "min", _group_by_first, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
                ValidAggregation(aggregator, column_name, missing_criteria),
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
            pad=ts._pad,
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
        "expected_count_{time_name}"
            The maximum number of possible values in each aggregation period
        "valid"
            Weather the aggregation is valid against the specified missing_criteria

    """

    @classmethod
    def create(cls) -> "AggregationFunction":
        return cls("max")

    def __init__(self, name: str) -> None:
        super().__init__(name)

    @override
    def apply(
        self,
        ts: TimeSeries,
        aggregation_period: Period,
        column_name: str,
        missing_criteria: Optional[Mapping[str, Union[float, int]]] = None,
    ) -> TimeSeries:
        aggregator: PolarsAggregator = PolarsAggregator(ts, aggregation_period, column_name)
        df: pl.DataFrame = aggregator.aggregate(
            [
                GroupByWithDateTime(aggregator, "max", _group_by_last, column_name),
                ActualValueCount(aggregator, column_name),
                ExpectedCount(aggregator),
                ValidAggregation(aggregator, column_name, missing_criteria),
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
            pad=ts._pad,
        )
