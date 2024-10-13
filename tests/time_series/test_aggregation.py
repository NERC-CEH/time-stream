import unittest
from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
)

import random
from datetime import datetime
from typing import Any, Optional

import polars as pl
from parameterized import parameterized

from time_series.time_series_base import TimeSeries, AggregationFunction
from time_series.time_series_polars import TimeSeriesPolars
from time_series.period import Period
import time_series.aggregation as aggregation

TIME: str = "datetime"
VALUE: str = "value"


def _create_df(datetime_list: list[datetime], value_list: list[float]) -> pl.DataFrame:
    return pl.DataFrame({TIME: datetime_list, VALUE: value_list})


def _create_ts(
    datetime_list: list[datetime],
    value_list: list[float],
    resolution: Period,
    periodicity: Period,
    time_zone: Optional[str],
) -> TimeSeries:
    return TimeSeries.from_polars(
        df=_create_df(datetime_list, value_list),
        time_name=TIME,
        resolution=resolution,
        periodicity=periodicity,
        time_zone=time_zone,
    )


MAX_LENGTH: int = 100_000


@dataclass(frozen=True)
class TsData:
    resolution: Period
    periodicity: Period
    datetime_list: list[datetime]
    value_list: list[float]

    def create_ts(self) -> TimeSeries:
        return _create_ts(
            datetime_list=self.datetime_list,
            value_list=self.value_list,
            resolution=self.resolution,
            periodicity=self.periodicity,
            time_zone=None,
        )

    def create_test_dict(self, aggregation_period: Period) -> dict[int, list[tuple[datetime, float]]]:
        result: dict[int, list[tuple[datetime, float]]] = {}
        for dt, val in zip(self.datetime_list, self.value_list):
            aggregation_ordinal = aggregation_period.ordinal(dt)
            tup = (dt, val)
            if aggregation_ordinal in result:
                result[aggregation_ordinal].append(tup)
            else:
                result[aggregation_ordinal] = [tup]
        return result

    @staticmethod
    def create(
        resolution: Period, periodicity: Period, in_datetime_from: datetime, in_datetime_to: datetime
    ) -> "TsData":
        ordinal_from: int = periodicity.ordinal(in_datetime_from)
        ordinal_to: int = periodicity.ordinal(in_datetime_to)
        datetime_from: datetime = periodicity.datetime(ordinal_from)
        datetime_to: datetime = periodicity.datetime(ordinal_to)

        periodicity_ordinal_start = ordinal_from
        periodicity_ordinal_end = min(ordinal_to, periodicity_ordinal_start + MAX_LENGTH)

        datetime_iter: Iterable[datetime]
        if resolution == periodicity:

            def _gen_datetime_series_flat():
                for ordinal in range(periodicity_ordinal_start, periodicity_ordinal_end):
                    yield resolution.datetime(ordinal)

            datetime_iter = _gen_datetime_series_flat()
        else:

            def _gen_datetime_series_gappy():
                for ordinal in range(periodicity_ordinal_start, periodicity_ordinal_end):
                    p_start = periodicity.datetime(ordinal)
                    p_end = periodicity.datetime(ordinal + 1)
                    r_ordinal_start = resolution.ordinal(p_start)
                    r_ordinal_end = resolution.ordinal(p_end)
                    r_start = resolution.datetime(r_ordinal_start)
                    r_end = resolution.datetime(r_ordinal_end)
                    if r_start < p_start:
                        r_ordinal_start += 1
                    if r_end < p_end:
                        r_ordinal_end += 1
                    r_ordinal = random.randrange(r_ordinal_start, r_ordinal_end)
                    yield resolution.datetime(r_ordinal)

            datetime_iter = _gen_datetime_series_gappy()

        datetime_list: list[datetime] = [dt for dt in datetime_iter]
        value_list: list[float] = [
            float(o - periodicity_ordinal_start) for o in range(periodicity_ordinal_start, periodicity_ordinal_end)
        ]
        random.shuffle(value_list)
        return TsData(
            resolution=resolution, periodicity=periodicity, datetime_list=datetime_list, value_list=value_list
        )


@dataclass(frozen=True)
class Case1:
    resolution: Period
    periodicity: Period
    aggregation_period: Period


PT15M = Period.of_minutes(15)
PT1H = Period.of_hours(1)
P1D = Period.of_days(1)
W_P1D = P1D.with_hour_offset(9)
P1M = Period.of_months(1)
W_P1M = P1M.with_hour_offset(9)
P1Y = Period.of_years(1)
W_P1Y = P1Y.with_month_offset(9).with_hour_offset(9)
PT1S = Period.of_seconds(1)
PT1M = Period.of_minutes(1)
PT1H = Period.of_hours(1)
CASE1_LIST: list[Case1] = [
    Case1(resolution=PT15M, periodicity=PT15M, aggregation_period=PT1H),
    Case1(resolution=PT15M, periodicity=PT15M, aggregation_period=P1D),
    Case1(resolution=PT15M, periodicity=PT15M, aggregation_period=W_P1D),
    Case1(resolution=PT15M, periodicity=PT15M, aggregation_period=W_P1M),
    Case1(resolution=PT15M, periodicity=PT15M, aggregation_period=W_P1Y),
    Case1(resolution=PT1H, periodicity=PT1H, aggregation_period=W_P1D),
    Case1(resolution=PT1H, periodicity=PT1H, aggregation_period=W_P1M),
    Case1(resolution=PT1H, periodicity=PT1H, aggregation_period=W_P1Y),
    Case1(resolution=W_P1D, periodicity=W_P1D, aggregation_period=W_P1M),
    Case1(resolution=W_P1D, periodicity=W_P1D, aggregation_period=W_P1Y),
    Case1(resolution=W_P1M, periodicity=W_P1M, aggregation_period=W_P1Y),
    Case1(resolution=PT1S, periodicity=PT1S, aggregation_period=PT1M),
    Case1(resolution=PT1S, periodicity=PT1S, aggregation_period=PT1H),
    Case1(resolution=PT1S, periodicity=PT1S, aggregation_period=P1D),
    Case1(resolution=PT1S, periodicity=PT1S, aggregation_period=P1M),
    Case1(resolution=PT1S, periodicity=PT1S, aggregation_period=P1Y),
    Case1(resolution=PT1M, periodicity=PT1M, aggregation_period=PT1H),
    Case1(resolution=PT1M, periodicity=PT1M, aggregation_period=P1D),
    Case1(resolution=PT1M, periodicity=PT1M, aggregation_period=P1M),
    Case1(resolution=PT1M, periodicity=PT1M, aggregation_period=P1Y),
    Case1(resolution=PT1H, periodicity=PT1H, aggregation_period=P1D),
    Case1(resolution=PT1H, periodicity=PT1H, aggregation_period=P1M),
    Case1(resolution=PT1H, periodicity=PT1H, aggregation_period=P1Y),
    Case1(resolution=P1D, periodicity=P1D, aggregation_period=P1M),
    Case1(resolution=P1D, periodicity=P1D, aggregation_period=P1Y),
    Case1(resolution=P1M, periodicity=P1M, aggregation_period=P1Y),
    Case1(resolution=PT1S, periodicity=PT1M, aggregation_period=PT1H),
    Case1(resolution=PT1M, periodicity=PT1H, aggregation_period=P1D),
    Case1(resolution=PT1H, periodicity=P1D, aggregation_period=P1M),
    Case1(resolution=P1D, periodicity=P1M, aggregation_period=P1Y),
    Case1(resolution=PT15M, periodicity=W_P1D, aggregation_period=W_P1M),
    Case1(resolution=PT15M, periodicity=W_P1M, aggregation_period=W_P1Y),
    Case1(resolution=W_P1D, periodicity=W_P1M, aggregation_period=W_P1Y),
]

PARAMS_CASE1: list[tuple[str, Case1]] = [
    (f"{case1.resolution.iso_duration}_{case1.aggregation_period.iso_duration}", case1) for case1 in CASE1_LIST
]

DATETIME_FROM: datetime = datetime(1990, 1, 1)
DATETIME_TO: datetime = datetime(2020, 1, 1)


def _create_ts_data(resolution: Period, periodicity: Period) -> TsData:
    return TsData.create(resolution, periodicity, DATETIME_FROM, DATETIME_TO)


def _get_pl_datetime_list(df: pl.DataFrame, column_name: str) -> list[datetime]:
    return [dt.replace(tzinfo=None) for dt in df[column_name]]


def _get_pl_float_list(df: pl.DataFrame, column_name: str) -> list[float]:
    return [float(f) for f in df[column_name]]


def _get_pl_int_list(df: pl.DataFrame, column_name: str) -> list[int]:
    return [int(i) for i in df[column_name]]


def _get_datetime_list(
    test_dict: dict[int, list[tuple[datetime, float]]], aggregation_period: Period
) -> list[datetime]:
    return [aggregation_period.datetime(key) for key in test_dict.keys()]


def _get_datetime_of_min_list(test_dict: dict[int, list[tuple[datetime, float]]]) -> list[datetime]:
    def _sort_by_min_value(tup: tuple[datetime, float]) -> Any:
        return tup[1]

    def _get_min_datetime(values: list[tuple[datetime, float]]):
        sorted_list: list[tuple[datetime, float]] = sorted(values, key=_sort_by_min_value)
        return sorted_list[0][0].replace(tzinfo=None)

    return [_get_min_datetime(value) for value in test_dict.values()]


def _get_datetime_of_max_list(test_dict: dict[int, list[tuple[datetime, float]]]) -> list[datetime]:
    def _sort_by_max_value(tup: tuple[datetime, float]) -> Any:
        return 0.0 - tup[1]

    def _get_max_datetime(values: list[tuple[datetime, float]]):
        sorted_list: list[tuple[datetime, float]] = sorted(values, key=_sort_by_max_value)
        return sorted_list[0][0].replace(tzinfo=None)

    return [_get_max_datetime(value) for value in test_dict.values()]


def _get_mean_list(test_dict: dict[int, list[tuple[datetime, float]]]) -> list[float]:
    def _calc_mean(values: list[tuple[datetime, float]]):
        return sum(t[1] for t in values) / len(values)

    return [_calc_mean(value) for value in test_dict.values()]


def _get_min_list(test_dict: dict[int, list[tuple[datetime, float]]]) -> list[float]:
    def _calc_min(values: list[tuple[datetime, float]]):
        return min(t[1] for t in values)

    return [_calc_min(value) for value in test_dict.values()]


def _get_max_list(test_dict: dict[int, list[tuple[datetime, float]]]) -> list[float]:
    def _calc_max(values: list[tuple[datetime, float]]):
        return max(t[1] for t in values)

    return [_calc_max(value) for value in test_dict.values()]


def _get_count_list(test_dict: dict[int, list[tuple[datetime, float]]]) -> list[int]:
    def _calc_count(values: list[tuple[datetime, float]]):
        return len(values)

    return [_calc_count(value) for value in test_dict.values()]


def _get_count_datetime_list(
    test_dict: dict[int, list[tuple[datetime, float]]], aggregation_period: Period, resolution: Period
) -> list[int]:
    def _calc_count(aggregation_ordinal: int):
        agg_start = aggregation_period.datetime(aggregation_ordinal)
        agg_end = aggregation_period.datetime(aggregation_ordinal + 1)
        res_start_o = resolution.ordinal(agg_start)
        res_end_o = resolution.ordinal(agg_end)
        res_start = resolution.datetime(res_start_o)
        res_end = resolution.datetime(res_end_o)
        res_span = res_end_o - res_start_o
        if res_start < agg_start:
            res_span -= 1
        if res_end < agg_end:
            res_span += 1
        return res_span

    return [_calc_count(key) for key in test_dict.keys()]


class TestFunctions(unittest.TestCase):
    def _test_basic(
        self,
        case1: Case1,
        name: str,
        aggregation_function: AggregationFunction,
        value_fn: Callable[[dict[int, list[tuple[datetime, float]]]], list[float]],
    ) -> None:
        ts_data = _create_ts_data(case1.resolution, case1.periodicity)
        ts: TimeSeries = ts_data.create_ts()
        #       print(f"input: {ts.resolution} {ts.periodicity}" )
        #       print(ts.df)
        result = ts.aggregate(aggregation_function, case1.aggregation_period, VALUE)
        test_dict: dict[int, list[tuple[datetime, float]]] = ts_data.create_test_dict(case1.aggregation_period)
        self.assertListEqual(
            _get_pl_datetime_list(result.df, TIME), _get_datetime_list(test_dict, case1.aggregation_period)
        )
        self.assertListEqual(_get_pl_float_list(result.df, f"{name}_{VALUE}"), value_fn(test_dict))
        self.assertListEqual(_get_pl_int_list(result.df, f"count_{VALUE}"), _get_count_list(test_dict))
        self.assertListEqual(
            _get_pl_int_list(result.df, f"count_{TIME}"),
            _get_count_datetime_list(test_dict, case1.aggregation_period, case1.periodicity),
        )

    #       print(f"result: {name}: {result.resolution} {result.periodicity}" )
    #       print(result.df)
    #       self.assertTrue(False)

    def _test_with_datetime(
        self,
        case1: Case1,
        name: str,
        aggregation_function: AggregationFunction,
        datetime_fn: Callable[[dict[int, list[tuple[datetime, float]]]], list[datetime]],
        value_fn: Callable[[dict[int, list[tuple[datetime, float]]]], list[float]],
    ) -> None:
        ts_data = _create_ts_data(case1.resolution, case1.periodicity)
        ts: TimeSeries = ts_data.create_ts()
        #       print(f"input: {ts.resolution} {ts.periodicity}" )
        #       print(ts.df)
        result = ts.aggregate(aggregation_function, case1.aggregation_period, VALUE)
        test_dict: dict[int, list[tuple[datetime, float]]] = ts_data.create_test_dict(case1.aggregation_period)
        self.assertListEqual(
            _get_pl_datetime_list(result.df, TIME), _get_datetime_list(test_dict, case1.aggregation_period)
        )
        self.assertListEqual(_get_pl_datetime_list(result.df, f"{TIME}_of_{name}"), datetime_fn(test_dict))
        self.assertListEqual(_get_pl_float_list(result.df, f"{name}_{VALUE}"), value_fn(test_dict))
        self.assertListEqual(_get_pl_int_list(result.df, f"count_{VALUE}"), _get_count_list(test_dict))
        self.assertListEqual(
            _get_pl_int_list(result.df, f"count_{TIME}"),
            _get_count_datetime_list(test_dict, case1.aggregation_period, case1.periodicity),
        )

    #       print(f"result: {name}: {result.resolution} {result.periodicity}" )
    #       print(result.df)
    #       self.assertTrue(False)

    @parameterized.expand(PARAMS_CASE1)
    def test_mean(self, _, case1):
        self._test_basic(
            case1=case1, name="mean", aggregation_function=AggregationFunction.mean(), value_fn=_get_mean_list
        )

    @parameterized.expand(PARAMS_CASE1)
    def test_min(self, _, case1):
        self._test_with_datetime(
            case1=case1,
            name="min",
            aggregation_function=AggregationFunction.min(),
            datetime_fn=_get_datetime_of_min_list,
            value_fn=_get_min_list,
        )

    @parameterized.expand(PARAMS_CASE1)
    def test_max(self, _, case1):
        self._test_with_datetime(
            case1=case1,
            name="max",
            aggregation_function=AggregationFunction.max(),
            datetime_fn=_get_datetime_of_max_list,
            value_fn=_get_max_list,
        )
