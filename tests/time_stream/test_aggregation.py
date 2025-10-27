import re
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from time_stream.aggregation import (
    AggregationCtx,
    AggregationFunction,
    AggregationPipeline,
    Max,
    Mean,
    MeanSum,
    Min,
    Percentile,
    Sum,
)
from time_stream.base import TimeFrame
from time_stream.enums import TimeAnchor
from time_stream.exceptions import (
    MissingCriteriaError,
    RegistryKeyTypeError,
    UnknownRegistryKeyError,
)
from time_stream.period import Period


def generate_time_series(
    resolution: Period, periodicity: Period, length: int, offset: str | None = None, missing_data: bool = False
) -> TimeFrame:
    """Helper function to generate a TimeSeries object for test purposes.

    Args:
        resolution: Resolution of the time series
        offset: Offset of the resolution
        periodicity: Periodicity of the time series
        length: Length of the time series
        missing_data: If True, add some missing data to the time series

    Returns:
        TimeSeries object
    """
    ordinal_from = periodicity.ordinal(datetime(2025, 1, 1))
    timestamps = [periodicity.datetime(ordinal_from + i) for i in range(length)]

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

    tf = TimeFrame(df=df, time_name="timestamp", resolution=resolution, offset=offset, periodicity=periodicity)
    return tf


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
TS_P1D_OFF_2MONTH = generate_time_series(P1D, P1D_OFF, 59, offset="+9H")  # 2 months (Jan, Feb 2025) of 1-day-offset
TS_P1M_OFF_2YEARS = generate_time_series(P1M, P1M_OFF, 24, offset="+9H")  # 2 years of 1-month-offset data


class TestAggregationPipeline:
    @staticmethod
    def setup_agg_context() -> Mock:
        ctx = Mock(spec=AggregationCtx)
        ctx.time_name = "time"
        ctx.time_anchor = TimeAnchor.START

        return ctx

    @pytest.mark.parametrize(
        "criteria,threshold",
        [
            ("percent", 50),
            ("percent", 40.8),
            ("missing", 5),
            ("available", 10),
            ("na", 0),
        ],
    )
    def test_missing_data_expr_validation_pass(self, criteria: str, threshold: int) -> None:
        """Test missing data expression validation that should pass."""
        ctx = self.setup_agg_context()
        columns = ["values"]
        agg_func = Mock(spec=AggregationFunction)
        period = Mock()

        ap = AggregationPipeline(agg_func, ctx, period, columns, ctx.time_anchor, (criteria, threshold))

        expressions = ap._missing_data_expr()
        assert isinstance(expressions, list)

    @pytest.mark.parametrize(
        "criteria,threshold",
        [
            ("percent", 101),
            ("percent", -1),
            ("missing", -1),
            ("available", -1),
            ("missing", 10.5),
            ("available", 10.5),
        ],
    )
    def test_missing_data_expr_validation_fail(self, criteria: str, threshold: int) -> None:
        """Test missing data expression validations that should fail."""
        ctx = self.setup_agg_context()
        columns = ["values"]
        agg_func = Mock(spec=AggregationFunction)
        period = Mock()

        ap = AggregationPipeline(agg_func, ctx, period, columns, ctx.time_anchor, (criteria, threshold))

        with pytest.raises(MissingCriteriaError):
            ap._missing_data_expr()


class TestAggregationFunction:
    """Test the base AggregationFunction class."""

    @staticmethod
    def setup_mock_ts() -> Mock:
        """Set up the mock ts."""
        mock_ts = Mock()
        mock_ts.time_name = "timestamp"
        return mock_ts

    @pytest.mark.parametrize(
        "get_input,expected", [("mean", Mean), ("min", Min), ("max", Max), ("mean_sum", MeanSum), ("sum", Sum)]
    )
    def test_get_with_string(self, get_input: str, expected: type[AggregationFunction]) -> None:
        """Test AggregationFunction.get() with string input."""
        agg = AggregationFunction.get(get_input)
        assert isinstance(agg, expected)

    @pytest.mark.parametrize(
        "get_input,expected", [(Mean, Mean), (Min, Min), (Max, Max), (MeanSum, MeanSum), (Sum, Sum)]
    )
    def test_get_with_class(self, get_input: type[AggregationFunction], expected: type[AggregationFunction]) -> None:
        """Test AggregationFunction.get() with class input."""
        agg = AggregationFunction.get(get_input)
        assert isinstance(agg, expected)

    @pytest.mark.parametrize(
        "get_input,expected", [(Mean(), Mean), (Min(), Min), (Max(), Max), (MeanSum(), MeanSum), (Sum(), Sum)]
    )
    def test_get_with_instance(self, get_input: AggregationFunction, expected: type[AggregationFunction]) -> None:
        """Test AggregationFunction.get() with instance input."""
        agg = AggregationFunction.get(get_input)
        assert isinstance(agg, expected)

    @pytest.mark.parametrize(
        "get_input",
        [
            "meen",
            "MINIMUM",
            "123",
            "mean-sum",
        ],
    )
    def test_get_with_invalid_string(self, get_input: str) -> None:
        """Test AggregationFunction.get() with invalid string."""
        expected_error = (
            f"Unknown name '{get_input}' for class type 'AggregationFunction'. "
            f"Available: {AggregationFunction.available()}."
        )

        with pytest.raises(UnknownRegistryKeyError, match=re.escape(expected_error)):
            AggregationFunction.get(get_input)

    def test_get_with_invalid_class(self) -> None:
        """Test AggregationFunction.get() with invalid class."""

        class InvalidClass:
            pass

        expected_error = "Class 'InvalidClass' must inherit from 'AggregationFunction'."

        with pytest.raises(RegistryKeyTypeError, match=expected_error):
            AggregationFunction.get(InvalidClass)

    @pytest.mark.parametrize(
        "get_input",
        [(123,), [Mean, Max], {Min}],
        ids=[
            "tuple of integers",
            "list of AggegationFunctions",
            "set of AggegationFunctions",
        ],
    )
    def test_get_with_invalid_type(self, get_input: Any) -> None:
        """Test AggregationFunction.get() with invalid type."""
        expected_error = (
            f"Check object must be a string, AggregationFunction class, or instance. Got {type(get_input).__name__}."
        )
        with pytest.raises(RegistryKeyTypeError, match=expected_error):
            AggregationFunction.get(get_input)


class TestSimpleAggregations:
    """Tests for simple aggregation cases, where the input time series has a simple resolution/periodicity and there is
    no missing data"""

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,counts,values,timestamps_of,kwargs",
        [
            (
                TS_PT1H_2DAYS,
                Mean,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [11.5, 35.5]},
                None,
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Max,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [23, 47]},
                [datetime(2025, 1, 1, 23), datetime(2025, 1, 2, 23)],
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Min,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [0, 24]},
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                {},
            ),
            (
                TS_PT1H_2DAYS,
                MeanSum,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852]},
                None,
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Sum,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852]},
                None,
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Percentile,
                P1D,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [22, 46]},
                None,
                {"p": 95},
            ),
        ],
        ids=[
            "hourly to daily mean",
            "hourly to daily max",
            "hourly to daily min",
            "hourly to daily mean_sum",
            "hourly to daily sum",
            "hourly_to_daily_95_percentile",
        ],
    )
    def test_microsecond_to_microsecond(
        self,
        input_tf: TimeFrame,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
        kwargs: dict[str, Any] | None,
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1 day or less) resolution data, to another
        microsecond-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
            **kwargs,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,counts,values,timestamps_of,kwargs",
        [
            (
                TS_PT1H_2MONTH,
                Mean,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [371.5, 1079.5]},
                None,
                {},
            ),
            (
                TS_PT1H_2MONTH,
                Max,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [743, 1415]},
                [datetime(2025, 1, 31, 23), datetime(2025, 2, 28, 23)],
                {},
            ),
            (
                TS_PT1H_2MONTH,
                Min,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [0, 744]},
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                {},
            ),
            (
                TS_PT1H_2MONTH,
                MeanSum,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [276396, 725424]},
                None,
                {},
            ),
            (
                TS_PT1H_2MONTH,
                Sum,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [276396, 725424]},
                None,
                {},
            ),
            (
                TS_PT1H_2MONTH,
                Percentile,
                P1M,
                "value",
                [datetime(2025, 1, 1), datetime(2025, 2, 1)],
                [744, 672],
                {"value": [557, 1247]},
                None,
                {"p": 75},
            ),
        ],
        ids=[
            "hourly to monthly mean",
            "hourly to monthly max",
            "hourly to monthly min",
            "hourly to monthly mean_sum",
            "hourly to monthly sum",
            "hourly_to_monthly_75_percentile",
        ],
    )
    def test_microsecond_to_month(
        self,
        input_tf: TimeFrame,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
        kwargs: dict[str, Any],
    ) -> None:
        """Test aggregations of microsecond-based (i.e., 1-day or less) resolution data, to a month-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
            **kwargs,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,counts,values,timestamps_of,kwargs",
        [
            (
                TS_P1M_2YEARS,
                Mean,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [5.5, 17.5]},
                None,
                {},
            ),
            (
                TS_P1M_2YEARS,
                Max,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [11, 23]},
                [datetime(2025, 12, 1), datetime(2026, 12, 1)],
                {},
            ),
            (
                TS_P1M_2YEARS,
                Min,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [0, 12]},
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                {},
            ),
            (
                TS_P1M_2YEARS,
                MeanSum,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [66, 210]},
                None,
                {},
            ),
            (
                TS_P1M_2YEARS,
                Sum,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [66, 210]},
                None,
                {},
            ),
            (
                TS_P1M_2YEARS,
                Percentile,
                P1Y,
                "value",
                [datetime(2025, 1, 1), datetime(2026, 1, 1)],
                [12, 12],
                {"value": [3, 15]},
                None,
                {"p": 25},
            ),
        ],
        ids=[
            "monthly to yearly mean",
            "monthly to yearly max",
            "monthly to yearly min",
            "monthly to yearly mean_sum",
            "monthly to yearly sum",
            "monthly_to_yearly_25_percentile",
        ],
    )
    def test_month_to_month(
        self,
        input_tf: TimeFrame,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: str,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
        kwargs: dict[str, Any] | None,
    ) -> None:
        """Test aggregations of month-based resolution data, to a month-based resolution."""
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
            **kwargs,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,counts,values,timestamps_of,kwargs",
        [
            (
                TS_PT1H_2DAYS,
                Mean,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [11.5, 35.5], "value_plus1": [12.5, 36.5], "value_times2": [23, 71]},
                None,
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Max,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [23, 47], "value_plus1": [24, 48], "value_times2": [46, 94]},
                [datetime(2025, 1, 1, 23), datetime(2025, 1, 2, 23)],
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Min,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [0, 24], "value_plus1": [1, 25], "value_times2": [0, 48]},
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                {},
            ),
            (
                TS_PT1H_2DAYS,
                MeanSum,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852], "value_plus1": [300, 876], "value_times2": [552, 1704]},
                None,
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Sum,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [276, 852], "value_plus1": [300, 876], "value_times2": [552, 1704]},
                None,
                {},
            ),
            (
                TS_PT1H_2DAYS,
                Percentile,
                P1D,
                ["value", "value_plus1", "value_times2"],
                [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                [24, 24],
                {"value": [12, 36], "value_plus1": [13, 37], "value_times2": [24, 72]},
                None,
                {"p": 50},
            ),
        ],
        ids=[
            "mult column mean",
            "mult column max",
            "mult column min",
            "mult column mean_sum",
            "mult column sum",
            "multi_column_50_percentile",
        ],
    )
    def test_multi_column(
        self,
        input_tf: TimeFrame,
        aggregator: type[AggregationFunction],
        target_period: Period,
        column: list,
        timestamps: list,
        counts: list,
        values: dict,
        timestamps_of: list | None,
        kwargs: dict[str, Any],
    ) -> None:
        expected_df = generate_expected_df(timestamps, aggregator, column, values, counts, counts, timestamps_of)
        result = aggregator().apply(
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
            **kwargs,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=["multi column mean"],
    )
    def test_multi_column_with_nulls(
        self,
        input_tf: TimeFrame,
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
        df = input_tf.df.clone()
        for idx, col in enumerate(column):
            df = df.with_columns(
                [
                    pl.when(pl.arange(0, input_tf.df.height) == idx).then(None).otherwise(pl.col(col)).alias(col),
                ]
            )

        expected_df = generate_expected_df(
            timestamps, aggregator, column, values, expected_counts, actual_counts, timestamps_of
        )
        result = aggregator().apply(
            df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)


class TestComplexPeriodicityAggregations:
    """Tests for more complex aggregation cases, where the input time series and/or target aggregation period
    has a more complex resolution/periodicity.

    Testing 1 "standard" aggregation (Mean) and 1 "date-based" aggregation (Min).
    """

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=[
            "hourly to day offset mean",
            "hourly to day offset max",
        ],
    )
    def test_microsecond_to_microsecond_offset(
        self,
        input_tf: TimeFrame,
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
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=[
            "hourly to month offset mean",
            "hourly to month offset max",
        ],
    )
    def test_microsecond_to_month_offset(
        self,
        input_tf: TimeFrame,
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
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=[
            "daily_offset_to_month_offset_mean",
            "daily_offset_to_month_offset_max",
        ],
    )
    def test_microsecond_offset_to_month_offset(
        self,
        input_tf: TimeFrame,
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
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=[
            "month_offset_to_month_offset_mean",
            "month_offset_to_month_offset_max",
        ],
    )
    def test_month_offset_to_month_offset(
        self,
        input_tf: TimeFrame,
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
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)


class TestEndAnchorAggregations:
    """Tests for aggregation cases where the time series has an end anchor."""

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=[
            "hourly to daily mean",
            "hourly to daily max",
            "hourly to daily min",
            "hourly to daily offset_max",
            "hourly to daily offset_mean",
        ],
    )
    def test_end_anchor_aggregations(
        self,
        input_tf: TimeFrame,
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
            input_tf.df, input_tf.time_name, TimeAnchor.END, input_tf.periodicity, target_period, column, TimeAnchor.END
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize(
        "input_tf,aggregator,target_period,column,timestamps,expected_counts,actual_counts,values,timestamps_of",
        [
            (
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
        ],
        ids=[
            "hourly to daily max",
            "daily offset to month offset max",
        ],
    )
    def test_end_anchor_aggregations_with_start_anchor_result(
        self,
        input_tf: TimeFrame,
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
            input_tf.df,
            input_tf.time_name,
            TimeAnchor.END,
            input_tf.periodicity,
            target_period,
            column,
            TimeAnchor.START,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)


class TestMissingCriteriaAggregations:
    """Tests the missing criteria functionality for aggregations."""

    input_tf = TS_PT1H_2DAYS_MISSING
    aggregator = Mean
    target_period = P1D
    column = "value"
    timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
    expected_counts = [24, 24]
    actual_counts = [20, 21]
    values = {"value": [11.7, 35.5714]}

    @pytest.mark.parametrize("valid", [{"value": [True, True]}], ids=["no missing criteria"])
    def test_no_missing_criteria(self, valid: dict) -> None:
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
            self.input_tf.df,
            self.input_tf.time_name,
            self.input_tf.time_anchor,
            self.input_tf.periodicity,
            self.target_period,
            self.column,
            self.input_tf.time_anchor,
        )

        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False, check_exact=False)

    @pytest.mark.parametrize(
        "valid,criteria",
        [
            ({"value": [True, True]}, ("percent", 80)),
            ({"value": [False, True]}, ("percent", (20 / 24) * 100)),
            ({"value": [False, True]}, ("percent", 85)),
            ({"value": [False, False]}, ("percent", (21 / 24) * 100)),
            ({"value": [False, False]}, ("percent", 90)),
            ({"value": [True, True]}, ("missing", 5)),
            ({"value": [True, True]}, ("missing", 4)),
            ({"value": [False, True]}, ("missing", 3)),
            ({"value": [False, False]}, ("missing", 2)),
            ({"value": [True, True]}, ("available", 20)),
            ({"value": [False, True]}, ("available", 21)),
            ({"value": [False, False]}, ("available", 22)),
            ({"value": [False, False]}, ("available", 23)),
        ],
        ids=[
            "percent 80",
            "percent 83.3",
            "percent 85",
            "percent 87.5",
            "percent 90",
            "missing 3",
            "missing 4",
            "missing 5",
            "missing 6",
            "available 20",
            "available 21",
            "available 22",
            "available 23",
        ],
    )
    def test_missing_criteria(self, valid: dict, criteria: tuple[str, float | int] | None) -> None:
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
            self.input_tf.df,
            self.input_tf.time_name,
            self.input_tf.time_anchor,
            self.input_tf.periodicity,
            self.target_period,
            self.column,
            self.input_tf.time_anchor,
            criteria,
        )

        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False, check_exact=False)


class TestMeanSumWithMissingData:
    """Tests the MeanSum aggregation with time series that has missing data."""

    def test_mean_sum_with_missing_data(self) -> None:
        """Test MeanSum aggregation with time series that has missing data."""
        input_tf = TS_PT1H_2DAYS_MISSING
        target_period = P1D
        column = "value"
        timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        expected_counts = [24, 24]
        actual_counts = [20, 21]
        values = {"value": [280.8, 853.71]}

        expected_df = generate_expected_df(timestamps, MeanSum, column, values, expected_counts, actual_counts)

        result = MeanSum().apply(
            input_tf.df,
            input_tf.time_name,
            input_tf.time_anchor,
            input_tf.periodicity,
            target_period,
            column,
            input_tf.time_anchor,
        )

        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False, check_exact=False)


class TestPaddedAggregations:
    """Tests that aggregations work as expected with padded time series."""

    @property
    def df(self) -> pl.DataFrame:
        timestamps = [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
            datetime(2020, 1, 3),  # missing the rest of the month,
            # missing all of February
            datetime(2020, 3, 1),
            datetime(2020, 3, 2),
            datetime(2020, 3, 3),  # missing the rest of the month,
        ]
        values = [1, 2, 3, 4, 5, 6]

        df = pl.DataFrame({"timestamp": timestamps, "value": values})
        return df

    def test_padded_result(self) -> None:
        """Test that the aggregation result is padded if the original time series was padded"""
        tf = TimeFrame(df=self.df, time_name="timestamp", resolution=Period.of_days(1), periodicity=Period.of_days(1))
        tf = tf.pad()

        expected_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
                "mean_value": [2.0, None, 5.0],
                "count_value": [3, 0, 3],
                "expected_count_timestamp": [31, 29, 31],
                "valid_value": [True, False, True],
            }
        )

        expected_tf = TimeFrame(
            df=expected_df, time_name="timestamp", resolution=Period.of_months(1), periodicity=Period.of_months(1)
        )

        result = tf.aggregate(Period.of_months(1), "mean", "value")
        assert result == expected_tf

    def test_not_padded_result(self) -> None:
        """Test that the aggregation result isn't padded if the original time series wasn't padded"""
        tf = TimeFrame(df=self.df, time_name="timestamp", resolution=Period.of_days(1), periodicity=Period.of_days(1))

        expected_df = pl.DataFrame(
            {
                "timestamp": [datetime(2020, 1, 1), datetime(2020, 3, 1)],
                "mean_value": [2.0, 5.0],
                "count_value": [3, 3],
                "expected_count_timestamp": [31, 31],
                "valid_value": [True, True],
            }
        )

        expected_tf = TimeFrame(
            df=expected_df, time_name="timestamp", resolution=Period.of_months(1), periodicity=Period.of_months(1)
        )

        result = tf.aggregate(Period.of_months(1), "mean", "value")
        assert result == expected_tf


class TestAggregationWithMetadata:
    """Tests that aggregations work as expected with time series that has metadata."""

    metadata = {
        "KeyA": 123,
        "KeyB": "ValueB",
    }

    def test_with_metadata(self) -> None:
        """Test that the aggregation result includes metadata from input time series"""
        tf = TimeFrame(df=TS_PT1H_2DAYS.df, time_name="timestamp", resolution=PT1H, periodicity=PT1H)
        tf.metadata = self.metadata

        result = tf.aggregate(Period.of_months(1), "mean", "value")
        assert result.metadata == self.metadata

    def test_with_no_metadata(self) -> None:
        """Test that the aggregation result metadata is empty if input time series metadata is empty"""
        tf = TimeFrame(df=TS_PT1H_2DAYS.df, time_name="timestamp", resolution=PT1H, periodicity=PT1H)

        result = tf.aggregate(Period.of_months(1), "mean", "value")
        assert result.metadata == {}


class TestPercentileAggregation:
    @pytest.mark.parametrize(
        "percentile,expected_values",
        [
            (0, {"value": [0, 24]}),
            (1, {"value": [0, 24]}),
            (5, {"value": [1, 25]}),
            (25, {"value": [6, 30]}),
            (50, {"value": [12, 36]}),
            (75, {"value": [17, 41]}),
            (95, {"value": [22, 46]}),
            (100, {"value": [23, 47]}),
        ],
    )
    def test_percentile_aggregation(self, percentile: int, expected_values: list[int]) -> None:
        input_tf = TS_PT1H_2DAYS
        column = "value"
        timestamps = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
        counts = [24, 24]

        expected_df = generate_expected_df(timestamps, Percentile, column, expected_values, counts, counts, None)
        result = Percentile().apply(
            df=input_tf.df,
            time_name=input_tf.time_name,
            time_anchor=input_tf.time_anchor,
            periodicity=input_tf.periodicity,
            aggregation_period=P1D,
            columns="value",
            aggregation_time_anchor=input_tf.time_anchor,
            p=percentile,
        )
        assert_frame_equal(result, expected_df, check_dtype=False, check_column_order=False)

    @pytest.mark.parametrize("percentile", [0.000000001, 0.999999, 101, 10000])
    def test_invalid_percentile(self, percentile: float) -> None:
        input_tf = TS_PT1H_2DAYS

        expected_error = "The percentile value must be provided as an integer value from 0 to 100"

        with pytest.raises(ValueError, match=expected_error):
            Percentile().apply(
                df=input_tf.df,
                time_name=input_tf.time_name,
                time_anchor=input_tf.time_anchor,
                periodicity=input_tf.periodicity,
                aggregation_period=P1D,
                columns="value",
                aggregation_time_anchor=input_tf.time_anchor,
                p=percentile,
            )
