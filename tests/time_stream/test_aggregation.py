import unittest
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal

from time_stream.aggregation import (
    AggregationCtx,
    AggregationFunction,
    AggregationPipeline,
    Max,
    Mean,
    MeanSum,
    Min,
    Sum,
)
from time_stream.base import TimeSeries
from time_stream.enums import TimeAnchor
from time_stream.exceptions import (
    MissingCriteriaError,
    RegistryKeyTypeError,
    UnknownRegistryKeyError,
)
from time_stream.period import Period


def generate_time_series(
    resolution: Period, periodicity: Period, length: int, missing_data: bool = False
) -> TimeSeries:
    """Helper function to generate a TimeSeries object for test purposes.

    Args:
        resolution: Resolution of the time series
        periodicity: Periodicity of the time series
        length: Length of the time series
        missing_data: If True, add some missing data to the time series

    Returns:
        TimeSeries object
    """
    ordinal_from = periodicity.ordinal(datetime(2025, 1, 1))
    timestamps = [resolution.datetime(ordinal_from + i) for i in range(length)]

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "value": list(range(length)),
            "value_plus1": [i + 1 for i in range(length)],
            "value_times2": [i * 2 for i in range(length)],
        }
    )

    if missing_data:
        df = df.remove(pl.col("value") % 7 == 0)

    ts = TimeSeries(df=df, time_name="timestamp", resolution=resolution, periodicity=periodicity)
    return ts


def generate_expected_df(
    timestamps: list[datetime],
    aggregator: type[AggregationFunction],
    columns: str | list[str],
    values: dict[str, list[float | int]],
    expected_counts: list[int],
    actual_counts: list[int],
    timestamps_of: list[datetime] = None,
    valid: dict[str, list[bool]] = None,
) -> pl.DataFrame:
    """Helper function to create a dataframe of expected results from an aggregation test.

    Args:
        timestamps: List of datetime values for the timestamp column
        aggregator: The aggregation function used in the test
        columns: The name of the column(s) that have been aggregated
        values: The aggregated values of the data columns
        expected_counts: The counts of expected values in each aggregation period, if there were no missing values
        actual_counts: The actual counts of values found in each aggregation period
        timestamps_of: For max and min, the datetimes of when the max/min values were found
        valid: Whether the aggregation value is "valid" based on the missing data criteria

    Returns:
        Expected DataFrame
    """
    if isinstance(columns, str):
        columns = [columns]

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "expected_count_timestamp": expected_counts,
        }
    )

    for column in columns:
        df = df.with_columns(
            pl.Series(f"{aggregator.name}_{column}", values[column]),
            pl.Series(f"count_{column}", actual_counts),
            pl.Series(f"valid_{column}", [True] * len(actual_counts)),
        )

        if valid:
            df = df.with_columns(pl.Series(f"valid_{column}", valid[column]))
        else:
            df = df.with_columns(pl.Series(f"valid_{column}", [True] * len(actual_counts)))

        if timestamps_of:
            df = df.with_columns(pl.Series(f"timestamp_of_{aggregator.name}_{column}", timestamps_of))

    return df


# Period instances used throughout these tests
PT1H = Period.of_hours(1)
P1D = Period.of_days(1)
P1D_OFF = P1D.with_hour_offset(9)  # water day
P1M = Period.of_months(1)
P1M_OFF = P1M.with_hour_offset(9)  # water month
P1Y = Period.of_years(1)
P1Y_OFF = P1Y.with_month_offset(9).with_hour_offset(9)  # water year

# TimeSeries instances used throughout these tests
TS_PT1H_2DAYS = generate_time_series(PT1H, PT1H, 48)  # 2 days of 1-hour data
TS_PT1H_2DAYS_MISSING = generate_time_series(PT1H, PT1H, 48, missing_data=True)  # 2 days of 1-hour data
TS_PT1H_2MONTH = generate_time_series(PT1H, PT1H, 1_416)  # 2 months (Jan, Feb 2025) of 1-hour data
TS_P1M_2YEARS = generate_time_series(P1M, P1M, 24)  # 2 years of month data
TS_P1D_OFF_2MONTH = generate_time_series(P1D_OFF, P1D_OFF, 59)  # 2 months (Jan, Feb 2025) of 1-day-offset
TS_P1M_OFF_2YEARS = generate_time_series(P1M_OFF, P1M_OFF, 24)  # 2 years of 1-month-offset data


class TestAggregationPipeline(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.ctx = Mock(spec=AggregationCtx)
        self.ctx.time_name = "time"
        self.ctx.time_anchor = TimeAnchor.START
        self.columns = ["values"]
        self.agg_func = Mock(spec=AggregationFunction)
        self.period = Mock()

    @parameterized.expand(
        [
            ("percent", 50),
            ("percent", 40.8),
            ("missing", 5),
            ("available", 10),
            ("na", 0),
        ]
    )
    def test_missing_data_expr_validation_pass(self, criteria: str, threshold: int) -> None:
        """Test missing data expression validation that should pass."""
        ap = AggregationPipeline(
            self.agg_func, self.ctx, self.period, self.columns, self.ctx.time_anchor, (criteria, threshold)
        )

        expressions = ap._missing_data_expr()
        self.assertIsInstance(expressions, list)

    @parameterized.expand(
        [
            ("percent", 101),
            ("percent", -1),
            ("missing", -1),
            ("available", -1),
            ("missing", 10.5),
            ("available", 10.5),
        ]
    )
    def test_missing_data_expr_validation_fail(self, criteria: str, threshold: int) -> None:
        """Test missing data expression validations that should fail."""
        ap = AggregationPipeline(
            self.agg_func, self.ctx, self.period, self.columns, self.ctx.time_anchor, (criteria, threshold)
        )

        with self.assertRaises(MissingCriteriaError):
            ap._missing_data_expr()


class TestAggregationFunction(unittest.TestCase):
    """Test the base AggregationFunction class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_ts = Mock()
        self.mock_ts.time_name = "timestamp"

    @parameterized.expand([("mean", Mean), ("min", Min), ("max", Max), ("mean_sum", MeanSum), ("sum", Sum)])
    def test_get_with_string(self, get_input: str, expected: type[AggregationFunction]) -> None:
        """Test AggregationFunction.get() with string input."""
        agg = AggregationFunction.get(get_input)
        self.assertIsInstance(agg, expected)

    @parameterized.expand([(Mean, Mean), (Min, Min), (Max, Max), (MeanSum, MeanSum), (Sum, Sum)])
    def test_get_with_class(self, get_input: type[AggregationFunction], expected: type[AggregationFunction]) -> None:
        """Test AggregationFunction.get() with class input."""
        agg = AggregationFunction.get(get_input)
        self.assertIsInstance(agg, expected)

    @parameterized.expand([(Mean(), Mean), (Min(), Min), (Max(), Max), (MeanSum(), MeanSum), (Sum(), Sum)])
    def test_get_with_instance(self, get_input: AggregationFunction, expected: type[AggregationFunction]) -> None:
        """Test AggregationFunction.get() with instance input."""
        agg = AggregationFunction.get(get_input)
        self.assertIsInstance(agg, expected)

    @parameterized.expand(
        [
            "meen",  # noqa
            "MINIMUM",
            "123",
            "mean-sum",
        ]
    )
    def test_get_with_invalid_string(self, get_input: str) -> None:
        """Test AggregationFunction.get() with invalid string."""
        with self.assertRaises(UnknownRegistryKeyError) as err:
            AggregationFunction.get(get_input)
        self.assertEqual(
            f"Unknown name '{get_input}' for class type 'AggregationFunction'. "
            f"Available: {AggregationFunction.available()}.",
            str(err.exception),
        )

    def test_get_with_invalid_class(self) -> None:
        """Test AggregationFunction.get() with invalid class."""

        class InvalidClass:
            pass

        with self.assertRaises(RegistryKeyTypeError) as err:
            AggregationFunction.get(InvalidClass)  # noqa - expecting type warning
        self.assertEqual("Class 'InvalidClass' must inherit from 'AggregationFunction'.", str(err.exception))

    @parameterized.expand([(123,), ([Mean, Max],), ({Min},)])
    def test_get_with_invalid_type(self, get_input: Any) -> None:
        """Test AggregationFunction.get() with invalid type."""
        with self.assertRaises(RegistryKeyTypeError) as err:
            AggregationFunction.get(get_input)
        self.assertEqual(
            f"Check object must be a string, AggregationFunction class, or instance. Got {type(get_input).__name__}.",
            str(err.exception),
        )


class TestSimpleAggregations(unittest.TestCase):
    """Tests for simple aggregation cases, where the input time series has a simple resolution/periodicity and there is
    no missing data"""

    @parameterized.expand(
        [
            (
                "hourly_to_daily_mean",
                TS_PT1H_2DAYS,
                Mean,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [11.5, 35.5]},
                None,
            ),
            (
                "hourly_to_daily_max",
                TS_PT1H_2DAYS,
                Max,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [23, 47]},
                [datetime(2025, 1, 1, 23), datetime(2025, 1, 2, 23)],
            ),
            (
                "hourly_to_daily_min",
                TS_PT1H_2DAYS,
                Min,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [0, 24]},
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            ),
            (
                "hourly_to_daily_mean_sum",
                TS_PT1H_2DAYS,
                MeanSum,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852]},
                None,
            ),
            (
                "hourly_to_daily_sum",
                TS_PT1H_2DAYS,
                Sum,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852]},
                None,
            ),
        ]
    )
    def test_microsecond_to_microsecond(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1 day or less) resolution data, to another
        microsecond-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "hourly_to_monthly_mean",
                TS_PT1H_2MONTH,
                Mean,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [371.5, 1079.5]},
                None,
            ),
            (
                "hourly_to_monthly_max",
                TS_PT1H_2MONTH,
                Max,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [743, 1415]},
                [datetime(2025, 1, 31, 23), datetime(2025, 2, 28, 23)],
            ),
            (
                "hourly_to_monthly_min",
                TS_PT1H_2MONTH,
                Min,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [0, 744]},
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
            ),
            (
                "hourly_to_monthly_mean_sum",
                TS_PT1H_2MONTH,
                MeanSum,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [276396, 725424]},
                None,
            ),
            (
                "hourly_to_monthly_sum",
                TS_PT1H_2MONTH,
                Sum,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [276396, 725424]},
                None,
            ),
        ]
    )
    def test_microsecond_to_month(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data, to a month-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "monthly_to_yearly_mean",
                TS_P1M_2YEARS,
                Mean,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [5.5, 17.5]},
                None,
            ),
            (
                "monthly_to_yearly_max",
                TS_P1M_2YEARS,
                Max,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [11, 23]},
                [datetime(2025, 12, 1), datetime(2026, 12, 1)],
            ),
            (
                "monthly_to_yearly_min",
                TS_P1M_2YEARS,
                Min,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [0, 12]},
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
            ),
            (
                "monthly_to_yearly_mean_sum",
                TS_P1M_2YEARS,
                MeanSum,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [66, 210]},
                None,
            ),
            (
                "monthly_to_yearly_sum",
                TS_P1M_2YEARS,
                Sum,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [66, 210]},
                None,
            ),
        ]
    )
    def test_month_to_month(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of month-based resolution data, to a month-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "multi_column_mean",
                TS_PT1H_2DAYS,
                Mean,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [11.5, 35.5], "value_plus1": [12.5, 36.5], "value_times2": [23, 71]},
                None,
            ),
            (
                "multi_column_max",
                TS_PT1H_2DAYS,
                Max,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [23, 47], "value_plus1": [24, 48], "value_times2": [46, 94]},
                [datetime(2025, 1, 1, 23), datetime(2025, 1, 2, 23)],
            ),
            (
                "multi_column_min",
                TS_PT1H_2DAYS,
                Min,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [0, 24], "value_plus1": [1, 25], "value_times2": [0, 48]},
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
            ),
            (
                "multi_column_mean_sum",
                TS_PT1H_2DAYS,
                MeanSum,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852], "value_plus1": [300, 876], "value_times2": [552, 1704]},
                None,
            ),
            (
                "multi_column_sum",
                TS_PT1H_2DAYS,
                Sum,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852], "value_plus1": [300, 876], "value_times2": [552, 1704]},
                None,
            ),
        ]
    )
    def test_multi_column(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "multi_column_mean",
                TS_PT1H_2DAYS,
                Mean,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                [23, 24],
                {"value": [12.0, 35.5], "value_plus1": [12.9565, 36.5], "value_times2": [23.8261, 71]},
                None,
            ),
        ]
    )
    def test_multi_column_with_nulls(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test that multi-column aggregations work as expected when each column has different null values."""

        # Set null values for each column
        df = input_ts.df.clone()
        for idx, col in enumerate(column):
            df = df.with_columns(
                [
                    pl.when(pl.arange(0, input_ts.df.height) == idx).then(None).otherwise(pl.col(col)).alias(col),
                ]
            )

        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)


class TestComplexPeriodicityAggregations(unittest.TestCase):
    """Tests for more complex aggregation cases, where the input time series and/or target aggregation period
    has a more complex resolution/periodicity.

    Testing 1 "standard" aggregation (Mean) and 1 "date-based" aggregation (Min).
    """

    @parameterized.expand(
        [
            (
                "hourly_to_day_offset_mean",
                TS_PT1H_2DAYS,
                Mean,
                P1D_OFF,
                "value",
                [datetime(2024, 12, 31, 9), datetime(2025, 1, 1, 9), datetime(2025, 1, 2, 9)],
                [24, 24, 24],
                [9, 24, 15],
                {"value": [4.0, 20.5, 40.0]},
                None,
            ),
            (
                "hourly_to_day_offset_max",
                TS_PT1H_2DAYS,
                Max,
                P1D_OFF,
                "value",
                [datetime(2024, 12, 31, 9), datetime(2025, 1, 1, 9), datetime(2025, 1, 2, 9)],
                [24, 24, 24],
                [9, 24, 15],
                {"value": [8, 32, 47]},
                [datetime(2025, 1, 1, 8), datetime(2025, 1, 2, 8), datetime(2025, 1, 2, 23)],
            ),
        ]
    )
    def test_microsecond_to_microsecond_offset(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1 day or less) resolution data, to another
        microsecond-based resolution with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "hourly_to_month_offset_mean",
                TS_PT1H_2MONTH,
                Mean,
                P1M_OFF,
                "value",
                [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
                [744, 744, 672],
                [9, 744, 663],
                {"value": [4.0, 380.5, 1084.0]},
                None,
            ),
            (
                "hourly_to_month_offset_max",
                TS_PT1H_2MONTH,
                Max,
                P1M_OFF,
                "value",
                [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
                [744, 744, 672],
                [9, 744, 663],
                {"value": [8, 752, 1415]},
                [datetime(2025, 1, 1, 8), datetime(2025, 2, 1, 8), datetime(2025, 2, 28, 23)],
            ),
        ]
    )
    def test_microsecond_to_month_offset(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data, to a month-based resolution
        with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "daily_offset_to_month_offset_mean",
                TS_P1D_OFF_2MONTH,
                Mean,
                P1M_OFF,
                "value",
                [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
                [31, 31, 28],
                [1, 31, 27],
                {"value": [0.0, 16.0, 45.0]},
                None,
            ),
            (
                "daily_offset_to_month_offset_max",
                TS_P1D_OFF_2MONTH,
                Max,
                P1M_OFF,
                "value",
                [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
                [31, 31, 28],
                [1, 31, 27],
                {"value": [0, 31, 58]},
                [datetime(2024, 12, 31, 9), datetime(2025, 1, 31, 9), datetime(2025, 2, 27, 9)],
            ),
        ]
    )
    def test_microsecond_offset_to_month_offset(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data that has an offset,
        to a month-based resolution with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "month_offset_to_month_offset_mean",
                TS_P1M_OFF_2YEARS,
                Mean,
                P1Y_OFF,
                "value",
                [datetime(2024, 10, 1, 9), datetime(2025, 10, 1, 9), datetime(2026, 10, 1, 9)],
                [12, 12, 12],
                [10, 12, 2],
                {"value": [4.5, 15.5, 22.5]},
                None,
            ),
            (
                "month_offset_to_month_offset_max",
                TS_P1M_OFF_2YEARS,
                Max,
                P1Y_OFF,
                "value",
                [datetime(2024, 10, 1, 9), datetime(2025, 10, 1, 9), datetime(2026, 10, 1, 9)],
                [12, 12, 12],
                [10, 12, 2],
                {"value": [9, 21, 23]},
                [datetime(2025, 9, 1, 9), datetime(2026, 9, 1, 9), datetime(2026, 11, 1, 9)],
            ),
        ]
    )
    def test_month_offset_to_month_offset(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test aggregations of month-based resolution data that has an offset,
        to a month-based resolution with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            input_ts.time_anchor,
            input_ts.periodicity,
            target_period,
            column,
            input_ts.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)


class TestEndAnchorAggregations(unittest.TestCase):
    """Tests for aggregation cases where the time series has an end anchor."""

    @parameterized.expand(
        [
            (
                "hourly_to_daily_mean",
                TS_PT1H_2DAYS,
                Mean,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                [24, 24, 24],
                [1, 24, 23],
                {"value": [0.0, 12.5, 36.0]},
                None,
            ),
            (
                "hourly_to_daily_max",
                TS_PT1H_2DAYS,
                Max,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                [24, 24, 24],
                [1, 24, 23],
                {"value": [0, 24, 47]},
                [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 2, 23)],
            ),
            (
                "hourly_to_daily_min",
                TS_PT1H_2DAYS,
                Min,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                [24, 24, 24],
                [1, 24, 23],
                {"value": [0, 1, 25]},
                [datetime(2025, 1, 1), datetime(2025, 1, 1, 1), datetime(2025, 1, 2, 1)],
            ),
            (
                "daily_offset_to_month_offset_max",
                TS_P1D_OFF_2MONTH,
                Max,
                P1M_OFF,
                "value",
                [datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9), datetime(2025, 3, 1, 9)],
                [31, 31, 28],
                [2, 31, 26],
                {"value": [1, 32, 58]},
                [datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9), datetime(2025, 2, 27, 9)],
            ),
            (
                "month_offset_to_month_offset_mean",
                TS_P1M_OFF_2YEARS,
                Mean,
                P1Y_OFF,
                "value",
                [datetime(2025, 10, 1, 9), datetime(2026, 10, 1, 9), datetime(2027, 10, 1, 9)],
                [12, 12, 12],
                [11, 12, 1],
                {"value": [5.0, 16.5, 23.0]},
                None,
            ),
        ]
    )
    def test_end_anchor_aggregations(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            input_ts.df, input_ts.time_name, TimeAnchor.END, input_ts.periodicity, target_period, column, TimeAnchor.END
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand(
        [
            (
                "hourly_to_daily_max",
                TS_PT1H_2DAYS,
                Max,
                P1D,
                "value",
                [datetime(2024, 12, 31), datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24, 24],
                [1, 24, 23],
                {"value": [0, 24, 47]},
                [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 2, 23)],
            ),
            (
                "daily_offset_to_month_offset_max",
                TS_P1D_OFF_2MONTH,
                Max,
                P1M_OFF,
                "value",
                [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
                [31, 31, 28],
                [2, 31, 26],
                {"value": [1, 32, 58]},
                [datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9), datetime(2025, 2, 27, 9)],
            ),
        ]
    )
    def test_end_anchor_aggregations_with_start_anchor_result(
        self,
        _: str,
        input_ts: TimeSeries,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        expected_counts: list,
        actual_counts: list,
        values: dict,
        timestamps_of: list | None,
    ) -> None:
        """Test where the aggregation output has a different anchor to the input."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            input_ts.df,
            input_ts.time_name,
            TimeAnchor.END,
            input_ts.periodicity,
            target_period,
            column,
            TimeAnchor.START,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)


class TestMissingCriteriaAggregations(unittest.TestCase):
    """Tests the missing criteria functionality for aggregations."""

    def setUp(self) -> None:
        self.input_ts = TS_PT1H_2DAYS_MISSING
        self.aggregator = Mean
        self.target_period = P1D
        self.column = "value"
        self.timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        self.expected_counts = [24, 24]
        self.actual_counts = [20, 21]
        self.values = {"value": [11.7, 35.5714]}

    @parameterized.expand(
        [
            ("no_missing_criteria", {"value": [True, True]}),
        ]
    )
    def test_no_missing_criteria(self, _: str, valid: dict) -> None:
        """Test aggregation of time series that has missing data but with no criteria for amount that can be
        missing. Essentially testing that the 'expected' and 'actual' counts of data are correct in the result."""
        expected_df = generate_expected_df(
            self.timestamps,
            self.aggregator,
            self.column,
            self.values,
            self.expected_counts,
            self.actual_counts,
            valid=valid,
        )

        result = self.aggregator().apply(
            self.input_ts.df,
            self.input_ts.time_name,
            self.input_ts.time_anchor,
            self.input_ts.periodicity,
            self.target_period,
            self.column,
            self.input_ts.time_anchor,
        )

        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False, check_exact=False)

    @parameterized.expand(
        [
            ("percent_80", {"value": [True, True]}, ("percent", 80)),
            ("percent_83.3", {"value": [False, True]}, ("percent", (20 / 24) * 100)),
            ("percent_85", {"value": [False, True]}, ("percent", 85)),
            ("percent_87.5", {"value": [False, False]}, ("percent", (21 / 24) * 100)),
            ("percent_90", {"value": [False, False]}, ("percent", 90)),
            ("missing_3", {"value": [True, True]}, ("missing", 5)),
            ("missing_4", {"value": [True, True]}, ("missing", 4)),
            ("missing_5", {"value": [False, True]}, ("missing", 3)),
            ("missing_6", {"value": [False, False]}, ("missing", 2)),
            ("available_20", {"value": [True, True]}, ("available", 20)),
            ("available_21", {"value": [False, True]}, ("available", 21)),
            ("available_22", {"value": [False, False]}, ("available", 22)),
            ("available_23", {"value": [False, False]}, ("available", 23)),
        ]
    )
    def test_missing_criteria(self, _: str, valid: dict, criteria: tuple[str, float | int] | None) -> None:
        """Test aggregation of time series that has missing data with a various missing criteria"""
        expected_df = generate_expected_df(
            self.timestamps,
            self.aggregator,
            self.column,
            self.values,
            self.expected_counts,
            self.actual_counts,
            valid=valid,
        )

        result = self.aggregator().apply(
            self.input_ts.df,
            self.input_ts.time_name,
            self.input_ts.time_anchor,
            self.input_ts.periodicity,
            self.target_period,
            self.column,
            self.input_ts.time_anchor,
            criteria,
        )

        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False, check_exact=False)


class TestMeanSumWithMissingData(unittest.TestCase):
    """Tests the MeanSum aggregation with time series that has missing data."""

    def setUp(self) -> None:
        self.input_ts = TS_PT1H_2DAYS_MISSING
        self.target_period = P1D
        self.column = "value"
        self.timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        self.expected_counts = [24, 24]
        self.actual_counts = [20, 21]
        self.values = {"value": [280.8, 853.71]}

    def test_mean_sum_with_missing_data(self) -> None:
        """Test MeanSum aggregation with time series that has missing data."""
        expected_df = generate_expected_df(
            self.timestamps, MeanSum, self.column, self.values, self.expected_counts, self.actual_counts
        )

        result = MeanSum().apply(
            self.input_ts.df,
            self.input_ts.time_name,
            self.input_ts.time_anchor,
            self.input_ts.periodicity,
            self.target_period,
            self.column,
            self.input_ts.time_anchor,
        )

        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False, check_exact=False)


class TestPaddedAggregations(unittest.TestCase):
    """Tests that aggregations work as expected with padded time series."""

    def setUp(self) -> None:
        self.timestamps = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),  # missing the rest of the month,
            # missing all of February
            datetime(2020, 3, 1),
            datetime(2020, 3, 2),
            datetime(2020, 3, 3),  # missing the rest of the month,
        ]
        self.values = [1, 2, 3, 4, 5, 6]
        self.df = pl.DataFrame({"timestamp": self.timestamps, "value": self.values})

    def test_padded_result(self) -> None:
        """Test that the aggregation result is padded if the original time series was padded"""
        ts = TimeSeries(df=self.df, time_name="timestamp", resolution=Period.of_days(1), periodicity=Period.of_days(1))
        ts.pad()

        expected_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
                "mean_value": [2.0, None, 5.0],
                "count_value": [3, 0, 3],
                "expected_count_timestamp": [31, 29, 31],
                "valid_value": [True, False, True],
            }
        )

        expected_ts = TimeSeries(
            df=expected_df, time_name="timestamp", resolution=Period.of_months(1), periodicity=Period.of_months(1)
        )

        result = ts.aggregate(Period.of_months(1), "mean", "value")
        self.assertEqual(result, expected_ts)

    def test_not_padded_result(self) -> None:
        """Test that the aggregation result isn't padded if the original time series wasn't padded"""
        ts = TimeSeries(df=self.df, time_name="timestamp", resolution=Period.of_days(1), periodicity=Period.of_days(1))

        expected_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1), datetime(2020, 3, 1)],
                "mean_value": [2.0, 5.0],
                "count_value": [3, 3],
                "expected_count_timestamp": [31, 31],
                "valid_value": [True, True],
            }
        )

        expected_ts = TimeSeries(
            df=expected_df, time_name="timestamp", resolution=Period.of_months(1), periodicity=Period.of_months(1)
        )

        result = ts.aggregate(Period.of_months(1), "mean", "value")
        self.assertEqual(result, expected_ts)


class TestAggregationWithMetadata(unittest.TestCase):
    """Tests that aggregations work as expected with time series that has metadata."""

    def setUp(self) -> None:
        self.metadata = {
            "KeyA": 123,
            "KeyB": "ValueB",
        }

    def test_with_metadata(self) -> None:
        """Test that the aggregation result includes metadata from input time series"""
        ts = TimeSeries(
            df=TS_PT1H_2DAYS.df, time_name="timestamp", resolution=PT1H, periodicity=PT1H, metadata=self.metadata
        )

        result = ts.aggregate(Period.of_months(1), "mean", "value")
        self.assertEqual(result.metadata(), self.metadata)

    def test_with_no_metadata(self) -> None:
        """Test that the aggregation result metadata is empty if input time series metadata is empty"""
        ts = TimeSeries(df=TS_PT1H_2DAYS.df, time_name="timestamp", resolution=PT1H, periodicity=PT1H)

        result = ts.aggregate(Period.of_months(1), "mean", "value")
        self.assertEqual(result.metadata(), {})
