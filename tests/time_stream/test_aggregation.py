import unittest
from datetime import datetime
from typing import Type
from unittest.mock import Mock

import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal

from time_stream.aggregation import _AGGREGATION_REGISTRY, AggregationFunction, Max, Mean, Min, MeanSum, Sum
from time_stream.base import TimeSeries
from time_stream.period import Period
from time_stream.exceptions import (
    AggregationPeriodError,
    AggregationTypeError,
    MissingCriteriaError,
    UnknownAggregationError,
)


def generate_time_series(resolution: Period, periodicity: Period, length: int, missing_data: bool=False) -> TimeSeries:
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

    df = pl.DataFrame({
        "timestamp": timestamps,
        "value": list(range(length)),
        "value_plus1": [i + 1 for i in range(length)],
        "value_times2": [i * 2 for i in range(length)]
    })

    if missing_data:
        df = df.remove(pl.col("value") % 7 == 0)

    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=resolution,
        periodicity=periodicity,
    )
    return ts


def generate_expected_df(
        timestamps: list[datetime],
        aggregator: Type[AggregationFunction],
        columns: str | list[str],
        values: dict[str, list[float | int]],
        expected_counts: list[int],
        actual_counts: list[int],
        timestamps_of: list[datetime]=None,
        valid: dict[str, list[bool]]=None
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

    df = pl.DataFrame({
        "timestamp": timestamps,
        "expected_count_timestamp": expected_counts,
    })

    for column in columns:
        df = df.with_columns(
            pl.Series(f"{aggregator.name}_{column}", values[column]),
            pl.Series(f"count_{column}", actual_counts),
            pl.Series(f"valid_{column}", [True] * len(actual_counts))
        )

        if valid:
            df = df.with_columns(
                pl.Series(f"valid_{column}", valid[column])
            )
        else:
            df = df.with_columns(
                pl.Series(f"valid_{column}", [True] * len(actual_counts))
            )

        if timestamps_of:
            df = df.with_columns(
                pl.Series(f"timestamp_of_{aggregator.name}_{column}", timestamps_of)
            )

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
ts_PT1H_2days = generate_time_series(PT1H, PT1H, 48)  # 2 days of 1-hour data
ts_PT1H_2days_missing = generate_time_series(PT1H, PT1H, 48, missing_data=True)  # 2 days of 1-hour data
ts_PT1H_2month = generate_time_series(PT1H, PT1H, 1_416)  # 2 months (Jan, Feb 2025) of 1-hour data
ts_P1M_2years = generate_time_series(P1M, P1M, 24)  # 2 years of month data
ts_P1D_OFF_2month = generate_time_series(P1D_OFF, P1D_OFF, 59)  # 2 months (Jan, Feb 2025) of 1-day-offset
ts_P1M_OFF_2years = generate_time_series(P1M_OFF, P1M_OFF, 24)  # 2 years of 1-month-offset data


class TestAggregationFunction(unittest.TestCase):
    """Test the base AggregationFunction class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_ts = Mock()
        self.mock_ts.time_name = "timestamp"

    @parameterized.expand([
        ("mean", Mean), ("min", Min), ("max", Max), ("mean_sum", MeanSum), ("sum", Sum)
    ])
    def test_get_with_string(self, get_input, expected):
        """Test AggregationFunction.get() with string input."""
        agg = AggregationFunction.get(get_input)
        self.assertIsInstance(agg, expected)

    @parameterized.expand([
        (Mean, Mean), (Min, Min), (Max, Max), (MeanSum, MeanSum), (Sum, Sum)
    ])
    def test_get_with_class(self, get_input, expected):
        """Test AggregationFunction.get() with class input."""
        agg = AggregationFunction.get(get_input)
        self.assertIsInstance(agg, expected)

    @parameterized.expand([
        (Mean(), Mean), (Min(), Min), (Max(), Max), (MeanSum(), MeanSum), (Sum(), Sum)
    ])
    def test_get_with_instance(self, get_input, expected):
        """Test AggregationFunction.get() with instance input."""
        agg = AggregationFunction.get(get_input)
        self.assertIsInstance(agg, expected)

    @parameterized.expand([
        "Mean", "MIN", "mAx", "123", "meansum", "sUm"
    ])
    def test_get_with_invalid_string(self, get_input):
        """Test AggregationFunction.get() with invalid string."""
        with self.assertRaises(UnknownAggregationError) as err:
            AggregationFunction.get(get_input)
        self.assertEqual(
            f"Unknown aggregation '{get_input}'. Available aggregations: {list(_AGGREGATION_REGISTRY.keys())}",
            str(err.exception)
        )

    def test_get_with_invalid_class(self):
        """Test AggregationFunction.get() with invalid class."""

        class InvalidClass:
            pass

        with self.assertRaises(AggregationTypeError) as err:
            AggregationFunction.get(InvalidClass)  # noqa - expecting type warning
        self.assertEqual("Aggregation class 'InvalidClass' must inherit from AggregationFunction.", str(err.exception))

    @parameterized.expand([
        (123,), ([Mean, Max],), ({Min},)
    ])
    def test_get_with_invalid_type(self, get_input):
        """Test AggregationFunction.get() with invalid type."""
        with self.assertRaises(AggregationTypeError) as err:
            AggregationFunction.get(get_input)
        self.assertEqual(
            f"Aggregation must be a string, AggregationFunction class, or instance. Got {type(get_input).__name__}.",
            str(err.exception)
        )

    def test_validate_aggregation_period_non_epoch_agnostic(self):
        """Test validation fails for non-epoch agnostic periods."""
        period = Mock()
        period.is_epoch_agnostic.return_value = False

        with self.assertRaises(AggregationPeriodError) as err:
            Mean()._validate_aggregation_period(self.mock_ts, period)
        self.assertEqual(f"Non-epoch agnostic aggregation periods are not supported: '{period}'.", str(err.exception))

    def test_validate_aggregation_period_not_subperiod(self):
        """Test validation fails when the aggregation period is not a subperiod."""
        period = Mock()
        period.is_epoch_agnostic.return_value = True
        self.mock_ts.periodicity.is_subperiod_of.return_value = False

        with self.assertRaises(AggregationPeriodError) as err:
            Mean()._validate_aggregation_period(self.mock_ts, period)
        self.assertEqual(
            f"Incompatible aggregation period '{period}' with TimeSeries periodicity '{self.mock_ts.periodicity}'."
            f"TimeSeries periodicity must be a subperiod of the aggregation period.",
            str(err.exception)
        )

    @parameterized.expand([
        ("percent", 50),
        ("percent", 40.8),
        ("missing", 5),
        ("available", 10),
        ("na", 0),
    ])
    def test_missing_data_expr_validation_pass(self, criteria, threshold):
        """Test missing data expression validation that should pass."""
        expressions = Mean()._missing_data_expr(self.mock_ts, ["value"], (criteria, threshold))
        self.assertIsInstance(expressions, list)

    @parameterized.expand([
        ("percent", 101),
        ("percent", -1),
        ("missing", -1),
        ("available", -1),
        ("missing", 10.5),
        ("available", 10.5),
    ])
    def test_missing_data_expr_validation_fail(self, criteria, threshold):
        """Test missing data expression validations that should fail."""
        with self.assertRaises(MissingCriteriaError):
            Mean()._missing_data_expr(self.mock_ts, ["value"], (criteria, threshold))


class TestSimpleAggregations(unittest.TestCase):
    """Tests for simple aggregation cases, where the input time series has a simple resolution/periodicity and there is
    no missing data"""

    @parameterized.expand([
        ("hourly_to_daily_mean", ts_PT1H_2days, Mean, P1D, "value", [datetime(2025, 1, 1), datetime(2025, 1, 2)],
         [24, 24], {"value": [11.5, 35.5]}, None),

        ("hourly_to_daily_max", ts_PT1H_2days, Max, P1D, "value", [datetime(2025, 1, 1), datetime(2025, 1, 2)],
         [24, 24], {"value": [23, 47]}, [datetime(2025, 1, 1, 23), datetime(2025, 1, 2, 23)]),

        ("hourly_to_daily_min", ts_PT1H_2days, Min, P1D, "value", [datetime(2025, 1, 1), datetime(2025, 1, 2)],
         [24, 24], {"value": [0, 24]}, [datetime(2025, 1, 1), datetime(2025, 1, 2)]),

        ("hourly_to_daily_mean_sum", ts_PT1H_2days, MeanSum, P1D, "value", [datetime(2025, 1, 1), datetime(2025, 1, 2)],
         [24, 24], {"value": [276, 852]}, None),
        
        ("hourly_to_daily_sum", ts_PT1H_2days, Sum, P1D, "value", [datetime(2025, 1, 1), datetime(2025, 1, 2)],
         [24, 24], {"value": [276, 852]}, None),
    ])

    def test_microsecond_to_microsecond(
            self, _, input_ts, aggregator, target_period, column, timestamps, counts, values, timestamps_of
    ):
        """Test aggregations of microsecond-based (i.e., 1 day or less) resolution data, to another
        microsecond-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        ("hourly_to_monthly_mean", ts_PT1H_2month, Mean, P1M, "value", [datetime(2025, 1, 1), datetime(2025, 2, 1)],
         [744, 672], {"value": [371.5, 1079.5]}, None),

        ("hourly_to_monthly_max", ts_PT1H_2month, Max, P1M, "value", [datetime(2025, 1, 1), datetime(2025, 2, 1)],
         [744, 672], {"value": [743, 1415]}, [datetime(2025, 1, 31, 23), datetime(2025, 2, 28, 23)]),

        ("hourly_to_monthly_min", ts_PT1H_2month, Min, P1M, "value", [datetime(2025, 1, 1), datetime(2025, 2, 1)],
         [744, 672], {"value": [0, 744]}, [datetime(2025, 1, 1), datetime(2025, 2, 1)]),
        
        ("hourly_to_monthly_mean_sum", ts_PT1H_2month, MeanSum, P1M, "value", [datetime(2025, 1, 1), datetime(2025, 2, 1)],
         [744, 672], {"value": [276396, 725424]}, None),
        
        ("hourly_to_monthly_sum", ts_PT1H_2month, Sum, P1M, "value", [datetime(2025, 1, 1), datetime(2025, 2, 1)],
         [744, 672], {"value": [276396, 725424]}, None),
    ])
    def test_microsecond_to_month(
            self, _, input_ts, aggregator, target_period, column, timestamps, counts, values, timestamps_of
    ):
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data, to a month-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        ("monthly_to_yearly_mean", ts_P1M_2years, Mean, P1Y, "value", [datetime(2025, 1, 1), datetime(2026, 1, 1)],
         [12, 12], {"value": [5.5, 17.5]}, None),

        ("monthly_to_yearly_max", ts_P1M_2years, Max, P1Y, "value", [datetime(2025, 1, 1), datetime(2026, 1, 1)],
         [12, 12], {"value": [11, 23]}, [datetime(2025, 12, 1), datetime(2026, 12, 1)]),

        ("monthly_to_yearly_min", ts_P1M_2years, Min, P1Y, "value", [datetime(2025, 1, 1), datetime(2026, 1, 1)],
         [12, 12], {"value": [0, 12]}, [datetime(2025, 1, 1), datetime(2026, 1, 1)]),

        ("monthly_to_yearly_mean_sum", ts_P1M_2years, MeanSum, P1Y, "value", [datetime(2025, 1, 1), datetime(2026, 1, 1)],
         [12, 12], {"value": [66, 210]}, None),
        
        ("monthly_to_yearly_sum", ts_P1M_2years, Sum, P1Y, "value", [datetime(2025, 1, 1), datetime(2026, 1, 1)],
         [12, 12], {"value": [66, 210]}, None),
    ])
    def test_month_to_month(
            self, _, input_ts, aggregator, target_period, column, timestamps, counts, values, timestamps_of
    ):
        """Test aggregations of month-based resolution data, to a month-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        ("multi_column_mean", ts_PT1H_2days, Mean, P1D, ["value", "value_plus1", "value_times2"],
         [datetime(2025, 1, 1), datetime(2025, 1, 2)], [24, 24],
         {"value": [11.5, 35.5], "value_plus1": [12.5, 36.5], "value_times2": [23, 71]}, None),

        ("multi_column_max", ts_PT1H_2days, Max, P1D, ["value", "value_plus1", "value_times2"],
         [datetime(2025, 1, 1), datetime(2025, 1, 2)], [24, 24],
         {"value": [23, 47], "value_plus1": [24, 48], "value_times2": [46, 94]},
         [datetime(2025, 1, 1, 23), datetime(2025, 1, 2, 23)]),

        ("multi_column_min", ts_PT1H_2days, Min, P1D, ["value", "value_plus1", "value_times2"],
         [datetime(2025, 1, 1), datetime(2025, 1, 2)], [24, 24],
         {"value": [0, 24], "value_plus1": [1, 25], "value_times2": [0, 48]},
         [datetime(2025, 1, 1), datetime(2025, 1, 2)]),
        
        ("multi_column_mean_sum", ts_PT1H_2days, MeanSum, P1D, ["value", "value_plus1", "value_times2"],
         [datetime(2025, 1, 1), datetime(2025, 1, 2)], [24, 24],
         {"value": [276, 852], "value_plus1": [300, 876], "value_times2": [552, 1704]}, None),
        
        ("multi_column_sum", ts_PT1H_2days, Sum, P1D, ["value", "value_plus1", "value_times2"],
         [datetime(2025, 1, 1), datetime(2025, 1, 2)], [24, 24],
         {"value": [276, 852], "value_plus1": [300, 876], "value_times2": [552, 1704]}, None),
    ])
    def test_multi_column(
            self, _, input_ts, aggregator, target_period, column, timestamps, counts, values, timestamps_of
    ):
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)


class TestComplexPeriodicityAggregations(unittest.TestCase):
    """Tests for more complex aggregation cases, where the input time series and/or target aggregation period
    has a more complex resolution/periodicity.

    Testing 1 "standard" aggregation (Mean) and 1 "date-based" aggregation (Min).
    """

    @parameterized.expand([
        ("hourly_to_day_offset_mean", ts_PT1H_2days, Mean, P1D_OFF, "value",
         [datetime(2024, 12, 31, 9), datetime(2025, 1, 1, 9), datetime(2025, 1, 2, 9)],
         [24, 24, 24], [9, 24, 15], {"value": [4., 20.5, 40.]}, None),

        ("hourly_to_day_offset_max", ts_PT1H_2days, Max, P1D_OFF, "value",
         [datetime(2024, 12, 31, 9), datetime(2025, 1, 1, 9), datetime(2025, 1, 2, 9)],
         [24, 24, 24], [9, 24, 15], {"value": [8, 32, 47]},
         [datetime(2025, 1, 1, 8), datetime(2025, 1, 2, 8), datetime(2025, 1, 2, 23)]),
    ])
    def test_microsecond_to_microsecond_offset(
            self, _, input_ts, aggregator, target_period, column, timestamps, expected_counts,
            actual_counts, values, timestamps_of
    ):
        """Test aggregations of microsecond-based (i.e., 1 day or less) resolution data, to another
        microsecond-based resolution with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        ("hourly_to_month_offset_mean", ts_PT1H_2month, Mean, P1M_OFF, "value",
         [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
         [744, 744, 672], [9, 744, 663], {"value": [4.0, 380.5, 1084.0]}, None),

        ("hourly_to_month_offset_max", ts_PT1H_2month, Max, P1M_OFF, "value",
         [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
         [744, 744, 672], [9, 744, 663], {"value": [8, 752, 1415]},
         [datetime(2025, 1, 1, 8), datetime(2025, 2, 1, 8), datetime(2025, 2, 28, 23)]),
    ])
    def test_microsecond_to_month_offset(
            self, _, input_ts, aggregator, target_period, column, timestamps, expected_counts, actual_counts,
            values, timestamps_of
    ):
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data, to a month-based resolution
        with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        (ts_P1M_2years, Mean, P1Y_OFF),
        (ts_P1M_2years, Max, P1Y_OFF),
    ])
    def test_month_to_month_offset(self, input_ts, aggregator, target_period):
        """Test aggregations of month-based resolution data, to a month-based resolution with an offset. This should
        raise an error as a month-based period cannot be a subperiod of a month-based-with-offset period"""
        with self.assertRaises(AggregationPeriodError):
            aggregator().apply(input_ts, target_period, "value")

    @parameterized.expand([
        ("daily_offset_to_month_offset_mean", ts_P1D_OFF_2month, Mean, P1M_OFF, "value",
         [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
         [31, 31, 28], [1, 31, 27], {"value": [0., 16., 45.]}, None),

        ("daily_offset_to_month_offset_max", ts_P1D_OFF_2month, Max, P1M_OFF, "value",
         [datetime(2024, 12, 1, 9), datetime(2025, 1, 1, 9), datetime(2025, 2, 1, 9)],
         [31, 31, 28], [1, 31, 27], {"value": [0, 31, 58]},
         [datetime(2024, 12, 31, 9), datetime(2025, 1, 31, 9), datetime(2025, 2, 27, 9)]),
    ])
    def test_microsecond_offset_to_month_offset(
            self, _, input_ts, aggregator, target_period, column, timestamps, expected_counts, actual_counts,
            values, timestamps_of
    ):
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data that has an offset,
        to a month-based resolution with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        ("month_offset_to_month_offset_mean", ts_P1M_OFF_2years, Mean, P1Y_OFF, "value",
         [datetime(2024, 10, 1, 9), datetime(2025, 10, 1, 9), datetime(2026, 10, 1, 9)],
         [12, 12, 12], [10, 12, 2], {"value": [4.5, 15.5, 22.5]}, None),

        ("month_offset_to_month_offset_max", ts_P1M_OFF_2years, Max, P1Y_OFF, "value",
         [datetime(2024, 10, 1, 9), datetime(2025, 10, 1, 9), datetime(2026, 10, 1, 9)],
         [12, 12, 12], [10, 12, 2], {"value": [9, 21, 23]},
         [datetime(2025, 9, 1, 9), datetime(2026, 9, 1, 9), datetime(2026, 11, 1, 9)]),
    ])
    def test_month_offset_to_month_offset(
            self, _, input_ts, aggregator, target_period, column, timestamps, expected_counts, actual_counts,
            values, timestamps_of
    ):
        """Test aggregations of month-based resolution data that has an offset,
        to a month-based resolution with an offset."""
        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(input_ts, target_period, column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False)

    @parameterized.expand([
        (ts_PT1H_2days, Mean, Period.of_hours(5)),
        (ts_PT1H_2days, Max, Period.of_days(7)),
        (ts_PT1H_2days, Min, Period.of_months(9)),
    ])
    def test_non_epoch_agnostic_fails(self, input_ts, aggregator, target_period):
        """Test that the aggregation fails to run if the aggregation period is not an epoch-agnostic period"""
        with self.assertRaises(AggregationPeriodError):
            aggregator().apply(input_ts, target_period, "value")


class TestMissingCriteriaAggregations(unittest.TestCase):
    """Tests the missing criteria functionality for aggregations."""

    def setUp(self):
        self.input_ts = ts_PT1H_2days_missing
        self.aggregator = Mean
        self.target_period = P1D
        self.column = "value"
        self.timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        self.expected_counts= [24, 24]
        self.actual_counts = [20, 21]
        self.values = {"value": [11.7, 35.5714]}

    @parameterized.expand([
        ("no_missing_criteria", {"value": [True, True]}),
    ])
    def test_no_missing_criteria(self, _, valid):
        """Test aggregation of time series that has missing data but with no criteria for amount that can be
        missing. Essentially testing that the 'expected' and 'actual' counts of data are correct in the result."""
        expected_df = generate_expected_df(
            self.timestamps, self.aggregator, self.column, self.values, self.expected_counts,
            self.actual_counts, valid=valid
        )
        result = self.aggregator().apply(self.input_ts, self.target_period, self.column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False, check_exact=False)

    @parameterized.expand([
        ("percent_80", {"value": [True, True]}, ("percent", 80)),
        ("percent_83.3", {"value": [False, True]}, ("percent", (20 / 24) * 100)),
        ("percent_85", {"value": [False, True]}, ("percent", 85)),
        ("percent_87.5", {"value": [False, False]}, ("percent", (21 / 24) * 100)),
        ("percent_90", {"value": [False, False]}, ("percent", 90)),
    ])
    def test_missing_criteria_percent(self, _, valid, criteria):
        """Test aggregation of time series that has missing data with a percent-based criteria"""
        expected_df = generate_expected_df(
            self.timestamps, self.aggregator, self.column, self.values, self.expected_counts,
            self.actual_counts, valid=valid
        )
        result = self.aggregator().apply(self.input_ts, self.target_period, self.column, criteria)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False, check_exact=False)

    @parameterized.expand([
        ("missing_3", {"value": [True, True]}, ("missing", 5)),
        ("missing_4", {"value": [True, True]}, ("missing", 4)),
        ("missing_5", {"value": [False, True]}, ("missing", 3)),
        ("missing_6", {"value": [False, False]}, ("missing", 2)),
    ])
    def test_missing_criteria_missing(self, _, valid, criteria):
        """Test aggregation of time series that has missing data with a missing-based criteria
        (Allow at most X missing values)."""
        expected_df = generate_expected_df(
            self.timestamps, self.aggregator, self.column, self.values, self.expected_counts,
            self.actual_counts, valid=valid
        )
        result = self.aggregator().apply(self.input_ts, self.target_period, self.column, criteria)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False, check_exact=False)

    @parameterized.expand([
        ("available_20", {"value": [True, True]}, ("available", 20)),
        ("available_21", {"value": [False, True]}, ("available", 21)),
        ("available_22", {"value": [False, False]}, ("available", 22)),
        ("available_23", {"value": [False, False]}, ("available", 23)),
    ])
    def test_missing_criteria_available(self, _, valid, criteria):
        """Test aggregation of time series that has missing data with an available-based criteria
        (Require at least X values present)."""
        expected_df = generate_expected_df(
            self.timestamps, self.aggregator, self.column, self.values, self.expected_counts,
            self.actual_counts, valid=valid
        )
        result = self.aggregator().apply(self.input_ts, self.target_period, self.column, criteria)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False, check_exact=False)


class TestMeanSumWithMissingData(unittest.TestCase):
    """Tests the MeanSum aggregation with time series that has missing data."""
    def setUp(self):
        self.input_ts = ts_PT1H_2days_missing
        self.target_period = P1D
        self.column = "value"
        self.timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        self.expected_counts = [24, 24]
        self.actual_counts = [20, 21]
        self.values = {"value": [280.8, 853.71]}

    def test_mean_sum_with_missing_data(self):
        """Test MeanSum aggregation with time series that has missing data."""
        expected_df = generate_expected_df(
            self.timestamps, MeanSum, self.column, self.values, self.expected_counts,
            self.actual_counts
        )
        result = MeanSum().apply(self.input_ts, self.target_period, self.column)
        assert_frame_equal(result.df, expected_df, check_dtype=False, check_column_order=False, check_exact=False)


class TestPaddedAggregations(unittest.TestCase):
    """Tests that aggregations work as expected with padded time series."""

    def setUp(self):
        self.timestamps = [
            datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3),  # missing the rest of the month,
            # missing all of February
            datetime(2020, 3, 1), datetime(2020, 3, 2), datetime(2020, 3, 3),  # missing the rest of the month,
        ]
        self.values = [1, 2, 3, 4, 5, 6]
        self.df = pl.DataFrame({"timestamp": self.timestamps, "values": self.values})

    @parameterized.expand([True, False])
    def test_padded_result(self, pad):
        """ Test that the aggregation result is as expected if the original time series was padded (or not)
        """
        ts = TimeSeries(
            df=self.df,
            time_name="timestamp",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1)
        )
        if pad:
            ts.pad()

        expected_df = pl.DataFrame({
            "timestamp": [datetime(2020, 1, 1), datetime(2020, 3, 1)],
            "mean_values": [2., 5.],
            "count_values": [3, 3],
            "expected_count_timestamp": [31, 31],
            "valid_values": [True, True],
        })

        expected_ts = TimeSeries(
            df=expected_df,
            time_name="timestamp",
            resolution=Period.of_months(1),
            periodicity=Period.of_months(1)
        )

        result = ts.aggregate(Period.of_months(1), "mean", "values")
        self.assertEqual(result, expected_ts)


class TestAggregationWithMetadata(unittest.TestCase):
    """Tests that aggregations work as expected with time series that has metadata."""
    def setUp(self):
        self.metadata = {
            "KeyA": 123,
            "KeyB": "ValueB",
        }

    def test_with_metadata(self):
        """ Test that the aggregation result includes metadata from input time series
        """
        ts = TimeSeries(
            df=ts_PT1H_2days.df,
            time_name="timestamp",
            resolution=PT1H,
            periodicity=PT1H,
            metadata=self.metadata
        )

        result = Mean().apply(ts, P1D, "value")
        self.assertEqual(result.metadata(), self.metadata)


    def test_with_no_metadata(self):
        """ Test that the aggregation result metadata is empty if input time series metadata is empty
        """
        ts = TimeSeries(
            df=ts_PT1H_2days.df,
            time_name="timestamp",
            resolution=PT1H,
            periodicity=PT1H
        )

        result = Mean().apply(ts, P1D, "value")
        self.assertEqual(result.metadata(), {})
