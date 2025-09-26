import random
from datetime import datetime, timedelta
from typing import Type

import pytest
import polars as pl
from pytest import param
from pytest_benchmark.fixture import BenchmarkFixture

from time_stream import TimeFrame, Period
from time_stream.aggregation import AggregationFunction, Max, Mean, Min


AGG_FUNCS = (Max, Mean, Min)

def generate_data(resolution: Period, length_days: int) -> TimeFrame:
    """ Generate test data for a given resolution, across dates in given range

    Args:
        resolution: Resolution of the data to generate
        length_days: Length of number of days of data to generate

    Returns:
        Time Series object containing the data
    """
    dt_from = datetime(2025, 1, 1)
    dt_to = dt_from + timedelta(days=length_days)

    periodicity = resolution
    ordinal_from = periodicity.ordinal(dt_from)
    ordinal_to = periodicity.ordinal(dt_to)

    periodicity_ordinal_start = ordinal_from
    periodicity_ordinal_end = ordinal_to

    timestamps = [resolution.datetime(ordinal) for ordinal in range(periodicity_ordinal_start, periodicity_ordinal_end)]
    values = [random.uniform(0, 1000) for _ in range(len(timestamps))]

    df = pl.DataFrame({"timestamp": timestamps, "data": values})
    ts = TimeFrame(df, "timestamp", resolution, periodicity)

    return ts


class BaseAggregationBenchmark:
    ts: TimeFrame

    def run_aggregation(self,
                        benchmark: BenchmarkFixture,
                        agg_func: Type[AggregationFunction],
                        target_resolution: Period) -> None:
        @benchmark
        def run():
            agg_ts = self.ts.aggregate(target_resolution, agg_func, "data")
            assert agg_ts.resolution == target_resolution


@pytest.mark.parametrize("agg_func", AGG_FUNCS)
class Test40HzAggregationBenchmarks(BaseAggregationBenchmark):
    @pytest.mark.parametrize("target_resolution", (
            param(Period.of_minutes(1), id="40hz-1min"),
            param(Period.of_minutes(15), id="40hz-15min"),
            param(Period.of_minutes(30), id="40hz-30min"),
            param(Period.of_hours(1), id="40hz-1hour"),
            param(Period.of_days(1), id="40hz-1day"),
    ))
    def test_40Hz_aggregation(self, benchmark, agg_func, target_resolution):
        self.run_aggregation(benchmark, agg_func, target_resolution)

    @classmethod
    def setup_class(cls):
        cls.ts = generate_data(Period.of_microseconds(25000), 1)

    @classmethod
    def teardown_class(cls):
        cls.ts = None

@pytest.mark.parametrize("agg_func", AGG_FUNCS)
class Test1MinuteAggregationBenchmarks(BaseAggregationBenchmark):
    @pytest.mark.parametrize("target_resolution", (
            param(Period.of_minutes(15), id="1min-15min"),
            param(Period.of_minutes(30), id="1min-30min"),
            param(Period.of_hours(1), id="1min-1hour"),
            param(Period.of_days(1), id="1min-1day"),
            param(Period.of_months(1), id="1min-1month"),
            param(Period.of_years(1), id="1min-1year"),
    ))
    def test_1Minute_aggregation(self, benchmark, agg_func, target_resolution):
        self.run_aggregation(benchmark, agg_func, target_resolution)

    @classmethod
    def setup_class(cls):
        cls.ts = generate_data(Period.of_minutes(1), 365)

    @classmethod
    def teardown_class(cls):
        cls.ts = None

@pytest.mark.parametrize("agg_func", AGG_FUNCS)
class Test15MinuteAggregationBenchmarks(BaseAggregationBenchmark):
    @pytest.mark.parametrize("target_resolution", (
            param(Period.of_minutes(30), id="15min-30min"),
            param(Period.of_hours(1), id="15min-1hour"),
            param(Period.of_days(1), id="15min-1day"),
            param(Period.of_months(1), id="15min-1month"),
            param(Period.of_years(1), id="15min-1year"),
    ))
    def test_15min_aggregation(self, benchmark, agg_func, target_resolution):
        self.run_aggregation(benchmark, agg_func, target_resolution)

    @classmethod
    def setup_class(cls):
        cls.ts = generate_data(Period.of_minutes(15), 3650)

    @classmethod
    def teardown_class(cls):
        cls.ts = None


@pytest.mark.parametrize("agg_func", AGG_FUNCS)
class Test30MinuteAggregationBenchmarks(BaseAggregationBenchmark):
    @pytest.mark.parametrize("target_resolution", (
            param(Period.of_hours(1), id="30min-1hour"),
            param(Period.of_days(1), id="30min-1day"),
            param(Period.of_months(1), id="30min-1month"),
            param(Period.of_years(1), id="30min-1year"),
    ))
    def test_30min_aggregation(self, benchmark, agg_func, target_resolution):
        self.run_aggregation(benchmark, agg_func, target_resolution)

    @classmethod
    def setup_class(cls):
        cls.ts = generate_data(Period.of_minutes(30), 3650)

    @classmethod
    def teardown_class(cls):
        cls.ts = None


@pytest.mark.parametrize("agg_func", AGG_FUNCS)
class Test1hourAggregationBenchmarks(BaseAggregationBenchmark):
    @pytest.mark.parametrize("target_resolution", (
            param(Period.of_days(1), id="1hour-1day"),
            param(Period.of_months(1), id="1hour-1month"),
            param(Period.of_years(1), id="1hour-1year"),
    ))
    def test_1hour_aggregation(self, benchmark, agg_func, target_resolution):
        self.run_aggregation(benchmark, agg_func, target_resolution)

    @classmethod
    def setup_class(cls):
        cls.ts = generate_data(Period.of_hours(1), 3650)

    @classmethod
    def teardown_class(cls):
        cls.ts = None


@pytest.mark.parametrize("agg_func", AGG_FUNCS)
class Test1dayAggregationBenchmarks(BaseAggregationBenchmark):
    @pytest.mark.parametrize("target_resolution", (
            param(Period.of_months(1), id="1day-1month"),
            param(Period.of_years(1), id="1day-1year"),
    ))
    def test_1day_aggregation(self, benchmark, agg_func, target_resolution):
        self.run_aggregation(benchmark, agg_func, target_resolution)

    @classmethod
    def setup_class(cls):
        cls.ts = generate_data(Period.of_days(1), 3650)

    @classmethod
    def teardown_class(cls):
        cls.ts = None
