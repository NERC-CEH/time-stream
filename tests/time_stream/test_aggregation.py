"""
Unit tests for the aggregation module
"""

import random
import unittest
from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
)
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

import polars as pl
from parameterized import parameterized

from time_stream.period import Period
from time_stream.base import TimeSeries
from time_stream.aggregation import AggregationFunction, Max, Mean, Min, PolarsAggregator, ValidAggregation

TIME: str = "datetime"
VALUE: str = "value"


def _create_df(datetime_list: list[datetime], value_list: list[float]) -> pl.DataFrame:
    """Create a Polars DataFrame

    Args:
        datetime_list: A list of datetime objects
        value_list: A list of floats

    Returns:
        A Polars DataFrame with two columns, TIME and VALUE
    """
    return pl.DataFrame({TIME: datetime_list, VALUE: value_list})


def _create_ts(
    datetime_list: list[datetime],
    value_list: list[float],
    resolution: Period,
    periodicity: Period,
    time_zone: Optional[str],
) -> TimeSeries:
    """Create a TimeSeries

    Args:
        datetime_list: A list of datetime objects
        value_list: A list of floats
        resolution: The resolution of the time-series
        periodicity: The periodicity of the time-series
        time_zone: The time zone of the time-series

    Returns:
        A TimeSeries object
    """
    """Create a TimeSeries
    """
    return TimeSeries(
        df=_create_df(datetime_list, value_list),
        time_name=TIME,
        resolution=resolution,
        periodicity=periodicity,
        time_zone=time_zone,
    )


# Maximum length of a synthetic TimeSeries
MAX_LENGTH: int = 5_000


@dataclass(frozen=True)
class TsData:
    """Some basic time-series data.

    The datetime_list and value_list properties contain
    the actual data.

    Attributes:
        resolution: The resolution of the time-series
        periodicity: The periodicity of the time-series
        datetime_list: A list of datetime objects
        value_list: A list of floats
    """

    resolution: Period
    periodicity: Period
    datetime_list: list[datetime]
    value_list: list[float]

    def create_ts(self) -> TimeSeries:
        """Create a TimeSeries from this data"""
        return _create_ts(
            datetime_list=self.datetime_list,
            value_list=self.value_list,
            resolution=self.resolution,
            periodicity=self.periodicity,
            time_zone=None,
        )

    def create_aggr_dict(self, aggregation_period: Period) -> dict[int, list[tuple[datetime, float]]]:
        """Create a dict containing some aggregated data

        The key of the dict is the ordinal of the aggregation
        period.

        The value of the dict is a list of tuples. Each tuple
        contains the [datetime, float] values of an element
        of time-series data

        Args:
            aggregation_period: The aggregation period

        Returns:
            A dict containing aggregated data that can be used
            to check the results of the Polars aggregation
        """
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
        """Create a TsData object containing synthetic time-series
        date of the supplied resolution and periodicity and within
        the supplied datetime range

        Args:
            resolution: The resolution of the time-series data
            periodicity: The periodicity of the time-series data
            in_datetime_from: The start datetime of the time-series
            in_datetime_to: The end datetime of the time-series

        Returns:
            A TsData object containing synthetic time-series data
        """
        ordinal_from: int = periodicity.ordinal(in_datetime_from)
        ordinal_to: int = periodicity.ordinal(in_datetime_to)

        periodicity_ordinal_start = ordinal_from
        periodicity_ordinal_end = min(ordinal_to, periodicity_ordinal_start + MAX_LENGTH)

        datetime_iter: Iterable[datetime]
        if resolution == periodicity:
            # if resolution == periodicity just create a sequence of
            # datetimes between start and end
            def _gen_datetime_series_flat() -> Iterable[datetime]:
                for ordinal in range(periodicity_ordinal_start, periodicity_ordinal_end):
                    yield resolution.datetime(ordinal)

            datetime_iter = _gen_datetime_series_flat()
        else:
            # if resolution != periodicity then create one datetime
            # for each period defined by the periodicity
            # The datetime will occur at a random position within
            # the periodicity period
            def _gen_datetime_series_gappy() -> Iterable[datetime]:
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
        # Generate a list of distinct float values sequentially
        # between 0 and n for each element of the time-series.
        value_list: list[float] = [
            float(o - periodicity_ordinal_start) for o in range(periodicity_ordinal_start, periodicity_ordinal_end)
        ]
        # Shuffle the resulting list to give the aggregation
        # functions something a bit less ordered to deal with.
        random.shuffle(value_list)
        return TsData(
            resolution=resolution, periodicity=periodicity, datetime_list=datetime_list, value_list=value_list
        )


@dataclass(frozen=True)
class Case1:
    """Some test case data

    For each aggregation function being tested a time-series
    of the given resolution and periodicity is created, which
    is then aggregated over the given aggregation period.

    The results from Polars are compared against the results
    of performing the aggregation just using Python

    Attributes:
        resolution: The resolution of the time-series
        periodicity: The periodicity of the time-series
        aggregation_period: The aggregation period
    """

    resolution: Period
    periodicity: Period
    aggregation_period: Period


# All the Periods used in the tests
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
PT0_1S = Period.of_microseconds(100_000)
PT0_01S = Period.of_microseconds(10_000)
PT0_001S = Period.of_microseconds(1_000)
PT0_0001S = Period.of_microseconds(100)
PT0_00001S = Period.of_microseconds(10)
PT0_000001S = Period.of_microseconds(1)

# A list of test cases
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
    Case1(resolution=PT0_1S, periodicity=PT0_1S, aggregation_period=PT1S),
    Case1(resolution=PT0_01S, periodicity=PT0_01S, aggregation_period=PT0_1S),
    Case1(resolution=PT0_001S, periodicity=PT0_001S, aggregation_period=PT0_01S),
    Case1(resolution=PT0_0001S, periodicity=PT0_0001S, aggregation_period=PT0_001S),
    Case1(resolution=PT0_00001S, periodicity=PT0_00001S, aggregation_period=PT0_0001S),
    Case1(resolution=PT0_000001S, periodicity=PT0_000001S, aggregation_period=PT0_00001S),
    Case1(resolution=PT0_000001S, periodicity=PT0_000001S, aggregation_period=P1D),
]

# A list that can be supplied to the @parameterized.expand
# annotation to test all the test cases
PARAMS_CASE1: list[tuple[str, Case1]] = [
    (f"{case1.resolution!s}_{case1.aggregation_period!s}", case1) for case1 in CASE1_LIST
]

# The dates within which to create test time-series data
DATETIME_FROM: datetime = datetime(1990, 1, 1)
DATETIME_TO: datetime = datetime(2020, 1, 1)


def _create_ts_data(resolution: Period, periodicity: Period) -> TsData:
    """Create some test time-series data"""
    return TsData.create(resolution, periodicity, DATETIME_FROM, DATETIME_TO)


def _get_pl_datetime_list(df: pl.DataFrame, column_name: str) -> list[datetime]:
    """Get a list of datetimes from a Series in a DataFrame"""
    return [dt.replace(tzinfo=None) for dt in df[column_name]]


def _get_pl_float_list(df: pl.DataFrame, column_name: str) -> list[float]:
    """Get a list of floats from a Series in a DataFrame"""
    return [float(f) for f in df[column_name]]


def _get_pl_int_list(df: pl.DataFrame, column_name: str) -> list[int]:
    """Get a list of ints from a Series in a DataFrame"""
    return [int(i) for i in df[column_name]]


def _get_datetime_list(
    aggr_dict: dict[int, list[tuple[datetime, float]]], aggregation_period: Period
) -> list[datetime]:
    """Get a list of datetimes of the start of each aggregation
    period from a dict of aggregated data
    """
    return [aggregation_period.datetime(key) for key in aggr_dict.keys()]


def _get_datetime_of_min_list(aggr_dict: dict[int, list[tuple[datetime, float]]]) -> list[datetime]:
    """Get a list of datetimes of the minimum value for each
    period from a dict of aggregated data
    """

    def _sort_by_min_value(tup: tuple[datetime, float]) -> float:
        return tup[1]

    def _get_min_datetime(values: list[tuple[datetime, float]]) -> datetime:
        sorted_list: list[tuple[datetime, float]] = sorted(values, key=_sort_by_min_value)
        return sorted_list[0][0].replace(tzinfo=None)

    return [_get_min_datetime(value) for value in aggr_dict.values()]


def _get_datetime_of_max_list(aggr_dict: dict[int, list[tuple[datetime, float]]]) -> list[datetime]:
    """Get a list of datetimes of the maximum value for each
    period from a dict of aggregated data
    """

    def _sort_by_max_value(tup: tuple[datetime, float]) -> float:
        return 0.0 - tup[1]

    def _get_max_datetime(values: list[tuple[datetime, float]]) -> datetime:
        sorted_list: list[tuple[datetime, float]] = sorted(values, key=_sort_by_max_value)
        return sorted_list[0][0].replace(tzinfo=None)

    return [_get_max_datetime(value) for value in aggr_dict.values()]


def _get_mean_list(aggr_dict: dict[int, list[tuple[datetime, float]]]) -> list[float]:
    """Get a list of floats of the means value for each
    period from a dict of aggregated data
    """

    def _calc_mean(values: list[tuple[datetime, float]]) -> float:
        return sum(t[1] for t in values) / len(values)

    return [_calc_mean(value) for value in aggr_dict.values()]


def _get_min_list(aggr_dict: dict[int, list[tuple[datetime, float]]]) -> list[float]:
    """Get a list of floats of the min value for each
    period from a dict of aggregated data
    """

    def _calc_min(values: list[tuple[datetime, float]]) -> float:
        return min(t[1] for t in values)

    return [_calc_min(value) for value in aggr_dict.values()]


def _get_max_list(aggr_dict: dict[int, list[tuple[datetime, float]]]) -> list[float]:
    """Get a list of floats of the max value for each
    period from a dict of aggregated data
    """

    def _calc_max(values: list[tuple[datetime, float]]) -> float:
        return max(t[1] for t in values)

    return [_calc_max(value) for value in aggr_dict.values()]


def _get_count_list(aggr_dict: dict[int, list[tuple[datetime, float]]]) -> list[int]:
    """Get a list of ints containing a count of the number of
    items in each period from a dict of aggregated data
    """

    def _calc_count(values: list[tuple[datetime, float]]) -> int:
        return len(values)

    return [_calc_count(value) for value in aggr_dict.values()]


def _get_expected_count_list(
    aggr_dict: dict[int, list[tuple[datetime, float]]], aggregation_period: Period, periodicity: Period
) -> list[int]:
    """Get a list of ints containing a count of the maximum number of
    items that could appear in each period from a dict of aggregated
    data
    """

    def _calc_count(aggregation_ordinal: int) -> int:
        agg_start = aggregation_period.datetime(aggregation_ordinal)
        agg_end = aggregation_period.datetime(aggregation_ordinal + 1)
        prd_ordinal_start = periodicity.ordinal(agg_start)
        prd_ordinal_end = periodicity.ordinal(agg_end)
        prd_start = periodicity.datetime(prd_ordinal_start)
        prd_end = periodicity.datetime(prd_ordinal_end)
        if prd_start < agg_start:
            prd_ordinal_start += 1
        if prd_end < agg_end:
            prd_ordinal_end += 1
        prd_span = prd_ordinal_end - prd_ordinal_start
        return prd_span

    return [_calc_count(key) for key in aggr_dict.keys()]


class TestFunctions(unittest.TestCase):
    """Test the min, mean, and max functions over a range of
    input time-series, aggregation periods and missing criteria options
    """

    def _test_basic(
        self,
        case1: Case1,
        name: str,
        aggregation_function: AggregationFunction,
        value_fn: Callable[[dict[int, list[tuple[datetime, float]]]], list[float]],
    ) -> None:
        """Test a 'basic' aggregation function, which just produces a
        float for each aggregation period
        """
        ts_data = _create_ts_data(case1.resolution, case1.periodicity)
        ts: TimeSeries = ts_data.create_ts()
        #       print(f"input: {ts.resolution} {ts.periodicity}" )
        #       print(ts.df)
        result = ts.aggregate(case1.aggregation_period, aggregation_function, VALUE)
        aggr_dict: dict[int, list[tuple[datetime, float]]] = ts_data.create_aggr_dict(case1.aggregation_period)
        # Compare datetime columns
        self.assertListEqual(
            _get_pl_datetime_list(result.df, TIME), _get_datetime_list(aggr_dict, case1.aggregation_period)
        )
        # Ensure datetime periodicity and resolution matches aggregation period
        self.assertTrue(TimeSeries.check_periodicity(result.df[TIME], case1.aggregation_period))
        self.assertTrue(TimeSeries.check_resolution(result.df[TIME], case1.aggregation_period))
        # Compare value columns
        # An equality check on floats could fail, but this
        # works, for the mean function at least.
        # Might need a version that takes float underflow/overflow
        # into account.
        self.assertListEqual(_get_pl_float_list(result.df, f"{name}_{VALUE}"), value_fn(aggr_dict))
        # Compare value count columns
        self.assertListEqual(_get_pl_int_list(result.df, f"count_{VALUE}"), _get_count_list(aggr_dict))
        # Compare datetime count columns
        self.assertListEqual(
            _get_pl_int_list(result.df, f"expected_count_{TIME}"),
            _get_expected_count_list(aggr_dict, case1.aggregation_period, case1.periodicity),
        )

    def _test_with_datetime(
        self,
        case1: Case1,
        name: str,
        aggregation_function: AggregationFunction,
        datetime_fn: Callable[[dict[int, list[tuple[datetime, float]]]], list[datetime]],
        value_fn: Callable[[dict[int, list[tuple[datetime, float]]]], list[float]],
    ) -> None:
        """Test a 'datetime' aggregation function, which produces a
        float and a datetime for each aggregation period
        """
        ts_data = _create_ts_data(case1.resolution, case1.periodicity)
        ts: TimeSeries = ts_data.create_ts()
        result = ts.aggregate(case1.aggregation_period, aggregation_function, VALUE)
        aggr_dict: dict[int, list[tuple[datetime, float]]] = ts_data.create_aggr_dict(case1.aggregation_period)
        # Compare datetime columns
        self.assertListEqual(
            _get_pl_datetime_list(result.df, TIME), _get_datetime_list(aggr_dict, case1.aggregation_period)
        )
        # Ensure datetime periodicity and resolution matches aggregation period
        self.assertTrue(TimeSeries.check_periodicity(result.df[TIME], case1.aggregation_period))
        self.assertTrue(TimeSeries.check_resolution(result.df[TIME], case1.aggregation_period))
        # Compare datetime of min/max columns
        self.assertListEqual(_get_pl_datetime_list(result.df, f"{TIME}_of_{name}"), datetime_fn(aggr_dict))
        # Ensure datetime of min/max has periodicity of the aggregation period
        self.assertTrue(TimeSeries.check_periodicity(result.df[f"{TIME}_of_{name}"], case1.aggregation_period))
        # Ensure datetime of min/max has resolution of the input time-series
        self.assertTrue(TimeSeries.check_resolution(result.df[f"{TIME}_of_{name}"], case1.resolution))
        # Compare value columns
        # For min/max comparing the float values should work ok, as
        # data is just being copied, there is no calculation involved.
        # Might need a version that takes float underflow/overflow
        # into account.
        self.assertListEqual(_get_pl_float_list(result.df, f"{name}_{VALUE}"), value_fn(aggr_dict))
        # Compare value count columns
        self.assertListEqual(_get_pl_int_list(result.df, f"count_{VALUE}"), _get_count_list(aggr_dict))
        # Compare datetime count columns
        self.assertListEqual(
            _get_pl_int_list(result.df, f"expected_count_{TIME}"),
            _get_expected_count_list(aggr_dict, case1.aggregation_period, case1.periodicity),
        )

    @parameterized.expand(PARAMS_CASE1)
    def test_mean(self, _: Any, case1: Case1) -> None:
        self._test_basic(
            case1=case1, name="mean", aggregation_function=Mean, value_fn=_get_mean_list
        )

    @parameterized.expand(PARAMS_CASE1)
    def test_min(self, _: Any, case1: Case1) -> None:
        self._test_with_datetime(
            case1=case1,
            name="min",
            aggregation_function=Min,
            datetime_fn=_get_datetime_of_min_list,
            value_fn=_get_min_list,
        )

    @parameterized.expand(PARAMS_CASE1)
    def test_max(self, _: Any, case1: Case1) -> None:
        self._test_with_datetime(
            case1=case1,
            name="max",
            aggregation_function=Max,
            datetime_fn=_get_datetime_of_max_list,
            value_fn=_get_max_list,
        )

class TestSubPeriodCheck(unittest.TestCase):
    """Test the "periodicity is a subperiod of aggregation period" check"""
    df = pl.DataFrame({
        "time": [datetime(2020, 1, 1)] ,
        VALUE: [1.0] })
    ts = TimeSeries(df=df, time_name="time",
        resolution=Period.of_days(1),
        periodicity=Period.of_days(1))

    def test_legal(self):
        # The test is that none of the following raise an error
        self.ts.aggregate(Period.of_days(1), Min, VALUE)
        self.ts.aggregate(Period.of_months(1), Min, VALUE)
        self.ts.aggregate(Period.of_months(1).with_day_offset(1), Min, VALUE)
        self.ts.aggregate(Period.of_months(1).with_day_offset(5), Min, VALUE)
        self.ts.aggregate(Period.of_years(1), Min, VALUE)
        self.ts.aggregate(Period.of_years(1).with_day_offset(1), Min, VALUE)
        self.ts.aggregate(Period.of_years(1).with_month_offset(1), Min, VALUE)
        self.ts.aggregate(Period.of_years(1).with_month_offset(1).with_day_offset(1), Min, VALUE)

    def test_illegal(self):
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_hours(1), Min, VALUE)
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_days(1).with_microsecond_offset(1), Min, VALUE)
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_days(1).with_second_offset(1), Min, VALUE)
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_months(1).with_minute_offset(10), Min, VALUE)
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_years(1).with_day_offset(1).with_hour_offset(1), Min, VALUE)
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_years(1).with_month_offset(1).with_hour_offset(1), Min, VALUE)
        with self.assertRaises(UserWarning):
            self.ts.aggregate(Period.of_years(1).with_month_offset(1).with_day_offset(1).with_hour_offset(1), Min, VALUE)


class TestMissingCriteria(unittest.TestCase):
    """Test the missing criteria functionality"""

    def setUp(self):
        """Set up test aggregated TimeSeries object"""

        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(31)]
        values = [10,12,15,14,13,17,19,21,18,17,5,9,0,1,5,11,12,10,21,16,10,11,8,6,14,17,12,10,10,8,5]

        df = pl.DataFrame({"timestamp": dates, "temperature": values})

        self.ts = TimeSeries(df=df, time_name="timestamp", resolution="PT30M", periodicity="PT30M")
    
    @parameterized.expand(
        [
            ({"percent": 1.9}, True),
            ({"percent": 2.1}, False),
            ({"missing": 1400}, False),
            ({"missing": 1500}, True),
            ({"available": 25}, True),
            ({"available": 35}, False),
            (None, True)
        ]
    )
    def test_valid_missing_criteria(self, missing_criteria, validity):
        """Test an aggregation function with valid missing criteria argument"""

        result = self.ts.aggregate(P1M, Mean, "temperature", missing_criteria)

        self.assertEqual(result.df['valid'][0], validity)


    @parameterized.expand(
        [
            ({"percent": 25}, {"method": "_percent", "limit": 25}),
            ({"missing": 25}, {"method": "_missing", "limit": 25}),
            ({"available": 75}, {"method": "_available", "limit": 75})
        ]
    )
    def test_validate_missing_aggregation_criteria_no_error(self, missing_criteria, expected):
        """Test the validate_missing_aggregation_criteria method with valid criteria."""

        aggregator: PolarsAggregator = PolarsAggregator(self.ts, Period.of_months(1))
        validator: ValidAggregation = ValidAggregation(aggregator, "test_column", missing_criteria)
        result = validator._validate_missing_aggregation_criteria(missing_criteria)

        assert result == expected


    @parameterized.expand(
        [
            (("percent", 25), ValueError),
            ({"missing": 25, "available": 45}, ValueError),
            ({"wrong_key": 75}, KeyError)
        ]
    )
    def test_validate_missing_aggregation_criteria_error(self, missing_criteria, expected):
        """Test the validate_missing_aggregation_criteria method with non-valid criteria."""

        aggregator: PolarsAggregator = PolarsAggregator(self.ts, Period.of_months(1))
        validator: ValidAggregation = ValidAggregation(aggregator, "test_column", missing_criteria)
        with self.assertRaises(expected):
            validator._validate_missing_aggregation_criteria(missing_criteria)
