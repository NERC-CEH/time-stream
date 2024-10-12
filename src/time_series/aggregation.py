"""
Time-series aggregation
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

GroupByToDataFrame = Callable[[pl.dataframe.group_by.GroupBy], pl.DataFrame]


def _gb_first(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    return group_by.first()


def _gb_last(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    return group_by.last()


def _gb_mean(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    return group_by.mean()


def _gb_min(group_by: pl.dataframe.group_by.GroupBy) -> pl.DataFrame:
    return group_by.min()


# ----------------------------------------------------------------------
class AggregationInfo:
    def __init__(self, ts: TimeSeries, aggregation_period: Period) -> None:
        self.ts = ts
        self.aggregation_period = aggregation_period
        self.datetime_column_name: str = ts.time_name
        self.resolution_interval: str = ts.resolution.pl_interval
        self.resolution_offset: str = ts.resolution.pl_offset
        self.periodicity_interval: str = ts.periodicity.pl_interval
        self.periodicity_offset: str = ts.periodicity.pl_offset
        self.aggregation_interval: str = aggregation_period.pl_interval
        self.aggregation_offset: str = aggregation_period.pl_offset

    def aggregate(self, dfa_list: list["DataFrameAggregator"]) -> pl.DataFrame:
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
        df1 = group_by_dynamic.agg(expression_list)
        if len(unnest_list) == 0:
            df2 = df1
        else:
            df2 = df1.unnest(*unnest_list)
        if len(with_column_list) == 0:
            df3 = df2
        else:
            df3 = df2.with_columns(*with_column_list)
        return df3


class DataFrameAggregator(ABC):
    def __init__(self, aggregation_info: AggregationInfo, name: str) -> None:
        self.info = aggregation_info
        self.name = name

    #       print( f"info: {self.info}" )
    #       print( f"name: {self.name}" )

    @property
    def datetime_column_name(self) -> str:
        return self.info.datetime_column_name

    def expressions(self) -> list[pl.Expr]:
        """Return list of expressions"""
        return []

    def unnest(self) -> list[str]:
        """Return list of structs to unnest"""
        return []

    def with_columns(self) -> list[pl.Expr]:
        """Return list of expressions"""
        return []


class DateTimeCount(DataFrameAggregator):
    def __init__(self, aggregation_info: AggregationInfo) -> None:
        super().__init__(aggregation_info, "datetime_count")

    def with_columns(self) -> list[pl.Expr]:
        datetime_column = self.info.datetime_column_name
        start_expr: pl.Expr = pl.col(datetime_column)
        end_expr: pl.Expr = pl.col(datetime_column).dt.offset_by(self.info.aggregation_interval)
        return [
            pl.datetime_ranges(start=start_expr, end=end_expr, interval=self.info.periodicity_interval, closed="right")
            .list.len()
            .alias(f"count_{datetime_column}")
        ]


class ValueCount(DataFrameAggregator):
    def __init__(self, aggregation_info: AggregationInfo, value_column: str) -> None:
        super().__init__(aggregation_info, "value_count")
        self._value_column = value_column

    def expressions(self) -> list[pl.Expr]:
        value_column: str = self._value_column
        return [pl.len().alias(f"count_{value_column}")]


class GroupByWithDateTime(DataFrameAggregator):
    def __init__(
        self, aggregation_info: AggregationInfo, name: str, gb2df: GroupByToDataFrame, value_column: str
    ) -> None:
        super().__init__(aggregation_info, name)
        self._gb2df = gb2df
        self._value_column = value_column
        self._struct_name: str = f"{name}_struct"

    def expressions(self) -> list[pl.Expr]:
        name: str = self.name
        value_column: str = self._value_column
        struct_columns: list[str] = [self.info.datetime_column_name, value_column]
        return [
            self._gb2df(pl.struct(struct_columns).sort_by(value_column))
            .alias(self._struct_name)
            .struct.rename_fields([f"{self.datetime_column_name}_of_{name}", f"{name}_{value_column}"])
        ]

    def unnest(self) -> list[str]:
        return [self._struct_name]


class GroupByBasic(DataFrameAggregator):
    def __init__(
        self, aggregation_info: AggregationInfo, name: str, gb2df: GroupByToDataFrame, value_column: str
    ) -> None:
        super().__init__(aggregation_info, name)
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
    def __init__(self) -> None:
        super().__init__("mean")

    @override
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        info: AggregationInfo = AggregationInfo(ts, aggregation_period)
        df: pl.DataFrame = info.aggregate(
            [GroupByBasic(info, "mean", _gb_mean, column_name), ValueCount(info, column_name), DateTimeCount(info)]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )


class Min(PolarsAggregationFunction):
    def __init__(self) -> None:
        super().__init__("min")

    @override
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        if not isinstance(ts, TimeSeriesPolars):
            raise NotImplementedError()
        info: AggregationInfo = AggregationInfo(ts, aggregation_period)
        df: pl.DataFrame = info.aggregate(
            [
                GroupByWithDateTime(info, "min", _gb_first, column_name),
                ValueCount(info, column_name),
                DateTimeCount(info),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )


class Max(PolarsAggregationFunction):
    def __init__(self) -> None:
        super().__init__("max")

    @override
    def pl_agg(self, ts: TimeSeriesPolars, aggregation_period: Period, column_name: str) -> TimeSeries:
        info: AggregationInfo = AggregationInfo(ts, aggregation_period)
        df: pl.DataFrame = info.aggregate(
            [
                GroupByWithDateTime(info, "max", _gb_last, column_name),
                ValueCount(info, column_name),
                DateTimeCount(info),
            ]
        )
        return TimeSeries.from_polars(
            df=df,
            time_name=ts.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_zone=ts.time_zone,
        )
