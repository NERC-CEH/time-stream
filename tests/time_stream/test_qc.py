from datetime import date, datetime, time
from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_series_equal

from time_stream.base import TimeFrame
from time_stream.exceptions import QcUnknownOperatorError, RegistryKeyTypeError, UnknownRegistryKeyError
from time_stream.qc import ComparisonCheck, QCCheck, QcCtx, RangeCheck, SpikeCheck


class MockQC(QCCheck):
    name = "Mock"

    def expr(self, _ctx: QcCtx, _column: str) -> pl.Expr:
        return pl.col(_column)


class TestQCCheck:
    """Test the base QCCheck class."""

    def setup_mock_tf(self) -> None:
        """Set up test fixtures."""
        self.mock_tf = Mock()
        self.mock_tf.time_name = "timestamp"
        self.mock_tf.df = pl.DataFrame({"timestamp": [datetime(2025, m, 1) for m in range(1, 8)]})

    @pytest.mark.parametrize(
        "get_input,input_args,expected",
        [
            ("comparison", {"compare_to": 10, "operator": ">"}, ComparisonCheck),
            ("spike", {"threshold": 10}, SpikeCheck),
            ("range", {"min_value": 0, "max_value": 10}, RangeCheck),
        ],
    )
    def test_get_with_string(self, get_input: str, input_args: dict, expected: type) -> None:
        """Test QCCheck.get() with string input."""
        qc = QCCheck.get(get_input, **input_args)
        assert isinstance(qc, expected)
        for arg, val in input_args.items():
            assert getattr(qc, arg) == val

    @pytest.mark.parametrize(
        "get_input,input_args,expected",
        [
            (ComparisonCheck, {"compare_to": 10, "operator": ">"}, ComparisonCheck),
            (SpikeCheck, {"threshold": 10}, SpikeCheck),
            (RangeCheck, {"min_value": 0, "max_value": 10}, RangeCheck),
        ],
    )
    def test_get_with_class(self, get_input: str, input_args: dict, expected: type) -> None:
        """Test QCCheck.get() with class input."""
        qc = QCCheck.get(get_input, **input_args)
        assert isinstance(qc, expected)
        for arg, val in input_args.items():
            assert getattr(qc, arg) == val

    @pytest.mark.parametrize(
        "get_input,expected",
        [(ComparisonCheck(10, ">"), ComparisonCheck), (SpikeCheck(100.0), SpikeCheck), (RangeCheck(1, 50), RangeCheck)],
    )
    def test_get_with_instance(self, get_input: str, expected: type) -> None:
        """Test QCCheck.get() with instance input."""
        qc = QCCheck.get(get_input)
        assert isinstance(qc, expected)

    @pytest.mark.parametrize("get_input", ["dummy", "RAANGE", "123"])
    def test_get_with_invalid_string(self, get_input: str) -> None:
        """Test QCCheck.get() with invalid string."""
        with pytest.raises(UnknownRegistryKeyError):
            QCCheck.get(get_input)

    def test_get_with_invalid_class(self) -> None:
        """Test QCCheck.get() with invalid class."""

        class InvalidClass:
            pass

        with pytest.raises(RegistryKeyTypeError):
            QCCheck.get(InvalidClass)  # noqa - expecting type warning

    @pytest.mark.parametrize("get_input", [123, [SpikeCheck, ComparisonCheck], {RangeCheck}])
    def test_get_with_invalid_type(self, get_input: str) -> None:
        """Test QCCheck.get() with invalid type."""
        with pytest.raises(RegistryKeyTypeError):
            QCCheck.get(get_input)


class TestComparisonCheck:
    data = pl.DataFrame(
        {
            "time": [datetime(2023, 8, 10), datetime(2023, 8, 11), datetime(2023, 8, 12), datetime(2023, 8, 13)],
            "value_a": [5.0, 10.0, 20.0, 30.0],
            "value_b": [1.0, 1.1, 1.2, 1.3],
            "value_c": [None, 50.0, 100.0, None],
        }
    )
    tf = TimeFrame(data, "time")

    def test_greater_than(self) -> None:
        """Test the comparison check function with '>' operator."""
        check = ComparisonCheck(10, ">")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([False, False, True, True])
        assert_series_equal(result, expected)

    def test_greater_than_or_equal(self) -> None:
        """Test the comparison check function with '>=' operator."""
        check = ComparisonCheck(10, ">=")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([False, True, True, True])
        assert_series_equal(result, expected)

    def test_less_than(self) -> None:
        """Test the comparison check function with '<' operator."""
        check = ComparisonCheck(10, "<")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([True, False, False, False])
        assert_series_equal(result, expected)

    def test_less_than_or_equal(self) -> None:
        """Test the comparison check function with '<=' operator."""
        check = ComparisonCheck(10, "<=")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([True, True, False, False])
        assert_series_equal(result, expected)

    def test_equal(self) -> None:
        """Test the comparison check function with '==' operator."""
        check = ComparisonCheck(10, "==")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([False, True, False, False])
        assert_series_equal(result, expected)

    def test_not_equal(self) -> None:
        """Test the comparison check function with '!=' operator."""
        check = ComparisonCheck(10, "!=")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([True, False, True, True])
        assert_series_equal(result, expected)

    def test_flag_na_when_true(self) -> None:
        """Test that setting flag_na to True will mean any NULL values in the check column are treated as failing
        the QC check (so qc flag set in the result).
        """
        check = ComparisonCheck(10, ">", flag_na=True)
        result = check.apply(self.tf.df, self.tf.time_name, "value_c")
        expected = pl.Series([True, True, True, True])
        assert_series_equal(result, expected)

    def test_flag_na_when_false(self) -> None:
        """Test that setting flag_na to False will mean that any NULL values in the check column are ignored in
        the QC check (so qc flag not set in the result).
        """
        check = ComparisonCheck(10, ">", flag_na=False)
        result = check.apply(self.tf.df, self.tf.time_name, "value_c")
        expected = pl.Series([None, True, True, None])
        assert_series_equal(result, expected)

    def test_invalid_operator(self) -> None:
        """Test that invalid operator raises error"""
        with pytest.raises(QcUnknownOperatorError):
            check = ComparisonCheck(10, ">>")
            check.apply(self.tf.df, self.tf.time_name, "value_a")


class TestRangeCheck:
    data = pl.DataFrame(
        {
            "time": [
                datetime(2023, 8, 9, 23, 0),
                datetime(2023, 8, 9, 23, 30),
                datetime(2023, 8, 10, 0, 0),
                datetime(2023, 8, 10, 0, 30),
                datetime(2023, 8, 10, 1, 0),
                datetime(2023, 8, 10, 1, 30),
                datetime(2023, 8, 10, 2, 0),
                datetime(2023, 8, 10, 2, 30),
            ],
            "value_a": range(8),
        }
    )
    tf = TimeFrame(data, "time")
    ctx = QcCtx(tf.df, tf.time_name)

    @pytest.mark.parametrize(
        "min_value,max_value",
        [
            (1.0, 5),
            (1.0, datetime(2023, 8, 10)),
            (datetime(2023, 8, 10), 100),
            (datetime(2023, 8, 9), date(2023, 8, 10)),
            (date(2023, 8, 9), datetime(2023, 8, 10)),
        ],
    )
    def test_type_mismatch(self, min_value: Any, max_value: Any) -> None:
        """Test that error raised if min and max value are not the same type."""
        with pytest.raises(TypeError):
            check = RangeCheck(min_value, max_value)
            check.expr(self.ctx, "value_a")

    @pytest.mark.parametrize(
        "min_value,max_value,closed,within,expected",
        [
            (0.5, 3.5, "both", True, [False, True, True, True, False, False, False, False]),
            (0.5, 3.5, "both", False, [True, False, False, False, True, True, True, True]),
            (-1, 10, "both", True, [True, True, True, True, True, True, True, True]),
            (
                -1,
                10,
                "both",
                False,
                [False, False, False, False, False, False, False, False],
            ),
            (
                90,
                100,
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (90, 100, "both", False, [True, True, True, True, True, True, True, True]),
        ],
        ids=[
            "distinct_flag_within",
            "distinct_flag_outside",
            "all_within_flag_within",
            "all_within_flag_outside",
            "all_outside_flag_within",
            "all_outside_flag_outside",
        ],
    )
    def test_range_check(self, min_value: int, max_value: int, closed: str, within: bool, expected: list) -> None:
        """Test that the range-check returns expected results"""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        assert_series_equal(result, pl.Series(expected))

    @pytest.mark.parametrize(
        "min_value,max_value,closed,within,expected",
        [
            (1, 3, "both", True, [False, True, True, True, False, False, False, False]),
            (1, 3, "both", False, [True, False, False, False, True, True, True, True]),
            (1, 3, "left", True, [False, True, True, False, False, False, False, False]),
            (1, 3, "left", False, [True, False, False, True, True, True, True, True]),
            (1, 3, "right", True, [False, False, True, True, False, False, False, False]),
            (1, 3, "right", False, [True, True, False, False, True, True, True, True]),
            (1, 3, "none", True, [False, False, True, False, False, False, False, False]),
            (1, 3, "none", False, [True, True, False, True, True, True, True, True]),
        ],
        ids=[
            "closed_both_flag_within",
            "closed_both_flag_outside",
            "closed_left_flag_within",
            "closed_left_flag_outside",
            "closed_right_flag_within",
            "closed_right_flag_outside",
            "closed_none_flag_within",
            "closed_none_flag_outside",
        ],
    )
    def test_range_check_at_values(
        self, min_value: int, max_value: int, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range-check returns expected results when the min and max values are exact values
        in the time series.  This is intended to test the "closed" parameter.
        """
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        assert_series_equal(result, pl.Series(expected))

    @pytest.mark.parametrize(
        "min_value,max_value,closed,within,expected",
        [
            (
                time(0, 15),
                time(1, 45),
                "both",
                True,
                [False, False, False, True, True, True, False, False],
            ),
            (
                time(0, 15),
                time(1, 45),
                "both",
                False,
                [True, True, True, False, False, False, True, True],
            ),
            (
                time(3, 0),
                time(12, 0),
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (
                time(3, 0),
                time(12, 0),
                "both",
                False,
                [True, True, True, True, True, True, True, True],
            ),
        ],
        ids=[
            "distinct_flag_within",
            "distinct_flag_outside",
            "all_outside_flag_within",
            "all_outside_flag_outside",
        ],
    )
    def test_range_check_using_time(
        self, min_value: time, max_value: time, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range check accepts time values."""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))

    @pytest.mark.parametrize(
        "min_value,max_value,closed,within,expected",
        [
            (
                time(22, 0),
                time(3, 0),
                "both",
                True,
                [True, True, True, True, True, True, True, True],
            ),
            (
                time(22, 0),
                time(3, 0),
                "both",
                False,
                [False, False, False, False, False, False, False, False],
            ),
            (
                time(23, 30),
                time(1, 30),
                "both",
                True,
                [False, True, True, True, True, True, False, False],
            ),
            (
                time(23, 30),
                time(1, 30),
                "both",
                False,
                [True, False, False, False, False, False, True, True],
            ),
            (
                time(23, 30),
                time(1, 30),
                "left",
                True,
                [False, True, True, True, True, False, False, False],
            ),
            (
                time(23, 30),
                time(1, 30),
                "left",
                False,
                [True, False, False, False, False, True, True, True],
            ),
            (
                time(23, 30),
                time(1, 30),
                "right",
                True,
                [False, False, True, True, True, True, False, False],
            ),
            (
                time(23, 30),
                time(1, 30),
                "right",
                False,
                [True, True, False, False, False, False, True, True],
            ),
            (
                time(23, 30),
                time(1, 30),
                "none",
                True,
                [False, False, True, True, True, False, False, False],
            ),
            (
                time(23, 30),
                time(1, 30),
                "none",
                False,
                [True, True, False, False, False, True, True, True],
            ),
        ],
        ids=[
            "all_within_flag_within",
            "all_within_flag_outside",
            "closed_both_flag_within",
            "closed_both_flag_outside",
            "closed_left_flag_within",
            "closed_left_flag_outside",
            "closed_right_flag_within",
            "closed_right_flag_outside",
            "closed_none_flag_within",
            "closed_none_flag_outside",
        ],
    )
    def test_time_range_check_across_midnight(
        self, min_value: time, max_value: time, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the time range check works as expected when the range is across midnight"""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))

    @pytest.mark.parametrize(
        "min_value,max_value,closed,within,expected",
        [
            (
                datetime(2023, 8, 9, 23, 45),
                datetime(2023, 8, 10, 1, 45),
                "both",
                True,
                [False, False, True, True, True, True, False, False],
            ),
            (
                datetime(2023, 8, 9, 23, 45),
                datetime(2023, 8, 10, 1, 45),
                "both",
                False,
                [True, True, False, False, False, False, True, True],
            ),
            (
                datetime(2023, 8, 1),
                datetime(2023, 8, 9, 12),
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (
                datetime(2023, 8, 1),
                datetime(2023, 8, 9, 12),
                "both",
                False,
                [True, True, True, True, True, True, True, True],
            ),
        ],
        ids=[
            "distinct_flag_within",
            "distinct_flag_outside",
            "all_outside_flag_within",
            "all_outside_flag_outside",
        ],
    )
    def test_range_check_using_datetimes(
        self, min_value: datetime, max_value: datetime, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range check accepts datetime values."""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))

    @pytest.mark.parametrize(
        "min_value,max_value,closed,within,expected",
        [
            (
                date(2023, 8, 9),
                date(2023, 8, 10),
                "both",
                True,
                [True, True, True, True, True, True, True, True],
            ),
            (
                date(2023, 8, 9),
                date(2023, 8, 10),
                "both",
                False,
                [False, False, False, False, False, False, False, False],
            ),
            (
                date(2023, 8, 1),
                date(2023, 8, 8),
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (
                date(2023, 8, 1),
                date(2023, 8, 8),
                "both",
                False,
                [True, True, True, True, True, True, True, True],
            ),
        ],
        ids=[
            "distinct_flag_within",
            "distinct_flag_outside",
            "all_outside_flag_within",
            "all_outside_flag_outside",
        ],
    )
    def test_range_check_using_dates(
        self, min_value: date, max_value: date, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range check accepts date values."""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))


class TestSpikeCheck:
    data = pl.DataFrame(
        {
            "time": [
                datetime(2023, 8, 10),
                datetime(2023, 8, 11),
                datetime(2023, 8, 12),
                datetime(2023, 8, 13),
                datetime(2023, 8, 14),
                datetime(2023, 8, 15),
            ],
            "value_a": range(6),
        }
    )
    tf = TimeFrame(data, "time")

    def test_basic_spike(self) -> None:
        """Test that the spike check returns expected results for a simple spike in the data"""
        df = self.tf.df.with_columns(pl.Series([1.0, 2.0, 3.0, 40.0, 5.0, 6.0]).alias("value_a"))
        tf = self.tf.with_df(df)

        check = SpikeCheck(10)
        result = check.apply(tf.df, tf.time_name, "value_a")
        expected = pl.Series([None, False, False, True, False, None])
        assert_series_equal(result, expected)

    def test_no_spike(self) -> None:
        """Test that no flags are added for data with no spike"""
        check = SpikeCheck(10)
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        expected = pl.Series([None, False, False, False, False, None])
        assert_series_equal(result, expected)

    def test_large_spike(self) -> None:
        """Large spikes can cause the values around the spike to be flagged if method not working correctly.
        Check this is not the case here.
        """
        df = self.tf.df.with_columns(pl.Series([1.0, 999.0, 3.0, 4.0, 999999999.0, 6.0]).alias("value_a"))
        tf = self.tf.with_df(df)

        check = SpikeCheck(10)
        result = check.apply(tf.df, tf.time_name, "value_a")
        expected = pl.Series([None, True, False, False, True, None])
        assert_series_equal(result, expected)


class TestCheckWithDateRange:
    data = pl.DataFrame({"time": [datetime(2023, 8, d + 1) for d in range(10)], "value_a": range(10)})
    tf = TimeFrame(data, "time")

    def test_no_date_range(self) -> None:
        """Test the check is applied to whole time series if no date range provided"""
        check = ComparisonCheck(1, ">")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a", observation_interval=None)
        expected = pl.Series([False, False, True, True, True, True, True, True, True, True])
        assert_series_equal(result, expected)

    def test_with_date_range(self) -> None:
        """Test the check is applied to part of the time series if a date range provided"""
        check = ComparisonCheck(1, ">")
        result = check.apply(
            self.tf.df, self.tf.time_name, "value_a", observation_interval=(datetime(2023, 8, 1), datetime(2023, 8, 5))
        )
        expected = pl.Series([False, False, True, True, True, False, False, False, False, False])
        assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "observation_interval",
        [
            # Different ways that observation_interval can specify an open-ended end date
            (datetime(2023, 8, 8)),
            ((datetime(2023, 8, 8), None)),
        ],
        ids=[
            "start_date_only",
            "end_date_none",
        ],
    )
    def test_with_date_range_open(self, observation_interval: tuple[datetime, datetime]) -> None:
        """Test the check is applied to part of the time series if a date range provided an open-ended end date"""
        check = ComparisonCheck(1, ">")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a", observation_interval=observation_interval)
        expected = pl.Series([False, False, False, False, False, False, False, True, True, True])
        assert_series_equal(result, expected)
