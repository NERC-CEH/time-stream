import unittest
from datetime import date, datetime, time
from typing import Any
from unittest.mock import Mock

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream.base import TimeFrame
from time_stream.exceptions import QcUnknownOperatorError, RegistryKeyTypeError, UnknownRegistryKeyError
from time_stream.qc import ComparisonCheck, QCCheck, QcCtx, RangeCheck, SpikeCheck


class MockQC(QCCheck):
    name = "Mock"

    def expr(self, _ctx: QcCtx, _column: str) -> pl.Expr:
        return pl.col(_column)


class TestQCCheck(unittest.TestCase):
    """Test the base QCCheck class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_tf = Mock()
        self.mock_tf.time_name = "timestamp"
        self.mock_tf.df = pl.DataFrame({"timestamp": [datetime(2025, m, 1) for m in range(1, 8)]})

    @parameterized.expand(
        [
            ("comparison", {"compare_to": 10, "operator": ">"}, ComparisonCheck),
            ("spike", {"threshold": 10}, SpikeCheck),
            ("range", {"min_value": 0, "max_value": 10}, RangeCheck),
        ]
    )
    def test_get_with_string(self, get_input: str, input_args: dict, expected: type) -> None:
        """Test QCCheck.get() with string input."""
        qc = QCCheck.get(get_input, **input_args)
        self.assertIsInstance(qc, expected)
        for arg, val in input_args.items():
            self.assertEqual(getattr(qc, arg), val)

    @parameterized.expand(
        [
            (ComparisonCheck, {"compare_to": 10, "operator": ">"}, ComparisonCheck),
            (SpikeCheck, {"threshold": 10}, SpikeCheck),
            (RangeCheck, {"min_value": 0, "max_value": 10}, RangeCheck),
        ]
    )
    def test_get_with_class(self, get_input: str, input_args: dict, expected: type) -> None:
        """Test QCCheck.get() with class input."""
        qc = QCCheck.get(get_input, **input_args)
        self.assertIsInstance(qc, expected)
        for arg, val in input_args.items():
            self.assertEqual(getattr(qc, arg), val)

    @parameterized.expand(
        [(ComparisonCheck(10, ">"), ComparisonCheck), (SpikeCheck(100.0), SpikeCheck), (RangeCheck(1, 50), RangeCheck)]
    )
    def test_get_with_instance(self, get_input: str, expected: type) -> None:
        """Test QCCheck.get() with instance input."""
        qc = QCCheck.get(get_input)
        self.assertIsInstance(qc, expected)

    @parameterized.expand(["dummy", "RAANGE", "123"])
    def test_get_with_invalid_string(self, get_input: str) -> None:
        """Test QCCheck.get() with invalid string."""
        with self.assertRaises(UnknownRegistryKeyError):
            QCCheck.get(get_input)

    def test_get_with_invalid_class(self) -> None:
        """Test QCCheck.get() with invalid class."""

        class InvalidClass:
            pass

        with self.assertRaises(RegistryKeyTypeError):
            QCCheck.get(InvalidClass)  # noqa - expecting type warning

    @parameterized.expand([(123,), ([SpikeCheck, ComparisonCheck],), ({RangeCheck},)])
    def test_get_with_invalid_type(self, get_input: str) -> None:
        """Test QCCheck.get() with invalid type."""
        with self.assertRaises(RegistryKeyTypeError):
            QCCheck.get(get_input)


class TestComparisonCheck(unittest.TestCase):
    def setUp(self) -> None:
        data = pl.DataFrame(
            {
                "time": [datetime(2023, 8, 10), datetime(2023, 8, 11), datetime(2023, 8, 12), datetime(2023, 8, 13)],
                "value_a": [5.0, 10.0, 20.0, 30.0],
                "value_b": [1.0, 1.1, 1.2, 1.3],
                "value_c": [None, 50.0, 100.0, None],
            }
        )
        self.tf = TimeFrame(data, "time")

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
        with self.assertRaises(QcUnknownOperatorError):
            check = ComparisonCheck(10, ">>")
            check.apply(self.tf.df, self.tf.time_name, "value_a")


class TestRangeCheck(unittest.TestCase):
    def setUp(self) -> None:
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
        self.tf = TimeFrame(data, "time")
        self.ctx = QcCtx(self.tf.df, self.tf.time_name)

    @parameterized.expand(
        [
            (1.0, 5),
            (1.0, datetime(2023, 8, 10)),
            (datetime(2023, 8, 10), 100),
            (datetime(2023, 8, 9), date(2023, 8, 10)),
            (date(2023, 8, 9), datetime(2023, 8, 10)),
        ]
    )
    def test_type_mismatch(self, min_value: Any, max_value: Any) -> None:
        """Test that error raised if min and max value are not the same type."""
        with self.assertRaises(TypeError):
            check = RangeCheck(min_value, max_value)
            check.expr(self.ctx, "value_a")

    @parameterized.expand(
        [
            # min, max, closed, within, expected
            ("distinct_flag_within", 0.5, 3.5, "both", True, [False, True, True, True, False, False, False, False]),
            ("distinct_flag_outside", 0.5, 3.5, "both", False, [True, False, False, False, True, True, True, True]),
            ("all_within_flag_within", -1, 10, "both", True, [True, True, True, True, True, True, True, True]),
            (
                "all_within_flag_outside",
                -1,
                10,
                "both",
                False,
                [False, False, False, False, False, False, False, False],
            ),
            (
                "all_outside_flag_within",
                90,
                100,
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            ("all_outside_flag_outside", 90, 100, "both", False, [True, True, True, True, True, True, True, True]),
        ]
    )
    def test_range_check(
        self, _: str, min_value: int, max_value: int, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range-check returns expected results"""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand(
        [
            # min, max, closed, within, expected
            ("closed_both_flag_within", 1, 3, "both", True, [False, True, True, True, False, False, False, False]),
            ("closed_both_flag_outside", 1, 3, "both", False, [True, False, False, False, True, True, True, True]),
            ("closed_left_flag_within", 1, 3, "left", True, [False, True, True, False, False, False, False, False]),
            ("closed_left_flag_outside", 1, 3, "left", False, [True, False, False, True, True, True, True, True]),
            ("closed_right_flag_within", 1, 3, "right", True, [False, False, True, True, False, False, False, False]),
            ("closed_right_flag_outside", 1, 3, "right", False, [True, True, False, False, True, True, True, True]),
            ("closed_none_flag_within", 1, 3, "none", True, [False, False, True, False, False, False, False, False]),
            ("closed_none_flag_outside", 1, 3, "none", False, [True, True, False, True, True, True, True, True]),
        ]
    )
    def test_range_check_at_values(
        self, _: str, min_value: int, max_value: int, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range-check returns expected results when the min and max values are exact values
        in the time series.  This is intended to test the "closed" parameter.
        """
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "value_a")
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand(
        [
            # min, max, closed, within, expected
            (
                "distinct_flag_within",
                time(0, 15),
                time(1, 45),
                "both",
                True,
                [False, False, False, True, True, True, False, False],
            ),
            (
                "distinct_flag_outside",
                time(0, 15),
                time(1, 45),
                "both",
                False,
                [True, True, True, False, False, False, True, True],
            ),
            (
                "all_outside_flag_within",
                time(3, 0),
                time(12, 0),
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (
                "all_outside_flag_outside",
                time(3, 0),
                time(12, 0),
                "both",
                False,
                [True, True, True, True, True, True, True, True],
            ),
        ]
    )
    def test_range_check_using_time(
        self, _: str, min_value: time, max_value: time, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range check accepts time values."""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand(
        [
            # min_time, max_time, inclusive, within, expected
            (
                "all_within_flag_within",
                time(22, 0),
                time(3, 0),
                "both",
                True,
                [True, True, True, True, True, True, True, True],
            ),
            (
                "all_within_flag_outside",
                time(22, 0),
                time(3, 0),
                "both",
                False,
                [False, False, False, False, False, False, False, False],
            ),
            (
                "closed_both_flag_within",
                time(23, 30),
                time(1, 30),
                "both",
                True,
                [False, True, True, True, True, True, False, False],
            ),
            (
                "closed_both_flag_outside",
                time(23, 30),
                time(1, 30),
                "both",
                False,
                [True, False, False, False, False, False, True, True],
            ),
            (
                "closed_left_flag_within",
                time(23, 30),
                time(1, 30),
                "left",
                True,
                [False, True, True, True, True, False, False, False],
            ),
            (
                "closed_left_flag_outside",
                time(23, 30),
                time(1, 30),
                "left",
                False,
                [True, False, False, False, False, True, True, True],
            ),
            (
                "closed_right_flag_within",
                time(23, 30),
                time(1, 30),
                "right",
                True,
                [False, False, True, True, True, True, False, False],
            ),
            (
                "closed_right_flag_outside",
                time(23, 30),
                time(1, 30),
                "right",
                False,
                [True, True, False, False, False, False, True, True],
            ),
            (
                "closed_none_flag_within",
                time(23, 30),
                time(1, 30),
                "none",
                True,
                [False, False, True, True, True, False, False, False],
            ),
            (
                "closed_none_flag_outside",
                time(23, 30),
                time(1, 30),
                "none",
                False,
                [True, True, False, False, False, True, True, True],
            ),
        ]
    )
    def test_time_range_check_across_midnight(
        self, _: str, min_value: time, max_value: time, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the time range check works as expected when the range is across midnight"""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand(
        [
            # min, max, closed, within, expected
            (
                "distinct_flag_within",
                datetime(2023, 8, 9, 23, 45),
                datetime(2023, 8, 10, 1, 45),
                "both",
                True,
                [False, False, True, True, True, True, False, False],
            ),
            (
                "distinct_flag_outside",
                datetime(2023, 8, 9, 23, 45),
                datetime(2023, 8, 10, 1, 45),
                "both",
                False,
                [True, True, False, False, False, False, True, True],
            ),
            (
                "all_outside_flag_within",
                datetime(2023, 8, 1),
                datetime(2023, 8, 9, 12),
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (
                "all_outside_flag_outside",
                datetime(2023, 8, 1),
                datetime(2023, 8, 9, 12),
                "both",
                False,
                [True, True, True, True, True, True, True, True],
            ),
        ]
    )
    def test_range_check_using_datetimes(
        self, _: str, min_value: datetime, max_value: datetime, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range check accepts datetime values."""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand(
        [
            # min, max, closed, within, expected
            (
                "distinct_flag_within",
                date(2023, 8, 9),
                date(2023, 8, 10),
                "both",
                True,
                [True, True, True, True, True, True, True, True],
            ),
            (
                "distinct_flag_outside",
                date(2023, 8, 9),
                date(2023, 8, 10),
                "both",
                False,
                [False, False, False, False, False, False, False, False],
            ),
            (
                "all_outside_flag_within",
                date(2023, 8, 1),
                date(2023, 8, 8),
                "both",
                True,
                [False, False, False, False, False, False, False, False],
            ),
            (
                "all_outside_flag_outside",
                date(2023, 8, 1),
                date(2023, 8, 8),
                "both",
                False,
                [True, True, True, True, True, True, True, True],
            ),
        ]
    )
    def test_range_check_using_dates(
        self, _: str, min_value: date, max_value: date, closed: str, within: bool, expected: list
    ) -> None:
        """Test that the range check accepts date values."""
        check = RangeCheck(min_value, max_value, closed, within)
        result = check.apply(self.tf.df, self.tf.time_name, "time")
        assert_series_equal(result, pl.Series(expected))


class TestSpikeCheck(unittest.TestCase):
    def setUp(self) -> None:
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
        self.tf = TimeFrame(data, "time")

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


class TestCheckWithDateRange(unittest.TestCase):
    def setUp(self) -> None:
        data = pl.DataFrame({"time": [datetime(2023, 8, d + 1) for d in range(10)], "value_a": range(10)})
        self.tf = TimeFrame(data, "time")

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

    @parameterized.expand(
        [
            # Different ways that observation_interval can specify an open-ended end date
            ("start_date_only", datetime(2023, 8, 8)),
            ("end_date_none", (datetime(2023, 8, 8), None)),
        ]
    )
    def test_with_date_range_open(self, _: str, observation_interval: tuple[datetime, datetime]) -> None:
        """Test the check is applied to part of the time series if a date range provided an open-ended end date"""
        check = ComparisonCheck(1, ">")
        result = check.apply(self.tf.df, self.tf.time_name, "value_a", observation_interval=observation_interval)
        expected = pl.Series([False, False, False, False, False, False, False, True, True, True])
        assert_series_equal(result, expected)
