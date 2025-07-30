import unittest
from datetime import datetime, time
from unittest.mock import Mock

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream.base import TimeSeries
from time_stream.qc import QCCheck, ComparisonCheck, SpikeCheck, RangeCheck


class MockQC(QCCheck):
    name = "Mock"

    def expr(self, check_column):
        return pl.col(check_column)

class TestQCCheck(unittest.TestCase):
    """Test the base QCCheck class."""
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ts = Mock()
        self.mock_ts.time_name = "timestamp"
        self.mock_ts.df = pl.DataFrame({"timestamp": [datetime(2025, m, 1) for m in range(1, 8)]})

    @parameterized.expand([
        ("comparison", {"compare_to": 10, "operator": ">"}, ComparisonCheck),
        ("spike", {"threshold": 10}, SpikeCheck),
        ("range", {"min_value": 0, "max_value": 10}, RangeCheck)
    ])
    def test_get_with_string(self, get_input, input_args, expected):
        """Test QCCheck.get() with string input."""
        qc = QCCheck.get(get_input, **input_args)
        self.assertIsInstance(qc, expected)
        for arg, val in input_args.items():
            self.assertEqual(getattr(qc, arg), val)

    @parameterized.expand([
        (ComparisonCheck, {"compare_to": 10, "operator": ">"}, ComparisonCheck),
        (SpikeCheck, {"threshold": 10}, SpikeCheck),
        (RangeCheck, {"min_value": 0, "max_value": 10}, RangeCheck)
    ])
    def test_get_with_class(self, get_input, input_args, expected):
        """Test QCCheck.get() with class input."""
        qc = QCCheck.get(get_input, **input_args)
        self.assertIsInstance(qc, expected)
        for arg, val in input_args.items():
            self.assertEqual(getattr(qc, arg), val)

    @parameterized.expand([
        (ComparisonCheck(10, ">"), ComparisonCheck), (SpikeCheck(100.), SpikeCheck), (RangeCheck(1, 50), RangeCheck)
    ])
    def test_get_with_instance(self, get_input, expected):
        """Test QCCheck.get() with instance input."""
        qc = QCCheck.get(get_input)
        self.assertIsInstance(qc, expected)

    @parameterized.expand([
        "dummy", "RANGE", "123"
    ])
    def test_get_with_invalid_string(self, get_input):
        """Test QCCheck.get() with invalid string."""
        with self.assertRaises(KeyError):
            QCCheck.get(get_input)

    def test_get_with_invalid_class(self):
        """Test QCCheck.get() with invalid class."""

        class InvalidClass:
            pass

        with self.assertRaises(TypeError):
            QCCheck.get(InvalidClass)  # noqa - expecting type warning

    @parameterized.expand([
        (123,), ([SpikeCheck, ComparisonCheck],), ({RangeCheck},)
    ])
    def test_get_with_invalid_type(self, get_input):
        """Test QCCheck.get() with invalid type."""
        with self.assertRaises(TypeError):
            QCCheck.get(get_input)

    @parameterized.expand([
        ("some_within", (datetime(2025, 1, 1), datetime(2025, 4, 1)), [True, True, True, True, False, False, False]),
        ("all_within", (datetime(2024, 1, 1), datetime(2026, 1, 1)), [True, True, True, True, True, True, True]),
        ("all_out", (datetime(2026, 1, 1), datetime(2027, 1, 1)), [False, False, False, False, False, False, False]),
        ("start_only", datetime(2025, 3, 1), [False, False, True, True, True, True, True]),
        ("start_only_with_none", (datetime(2025, 3, 1), None), [False, False, True, True, True, True, True]),
    ])
    def test_get_date_filter(self, _, observation_interval, expected_eval):
        date_filter = MockQC()._get_date_filter(self.mock_ts, observation_interval)
        result = self.mock_ts.df.select(date_filter).to_series()
        expected = pl.Series("timestamp", expected_eval)
        assert_series_equal(result, expected)


class TestComparisonCheck(unittest.TestCase):
    def setUp(self):
        data = pl.DataFrame({
            "time": [datetime(2023, 8, 10), datetime(2023, 8, 11), datetime(2023, 8, 12), datetime(2023, 8, 13)],
            "value_a": [5., 10., 20., 30.],
            "value_b": [1.0, 1.1, 1.2, 1.3],
            "value_c": [None, 50., 100., None]
        })
        self.ts = TimeSeries(data, "time")

    def test_greater_than(self):
        """ Test the comparison check function with '>' operator.
        """
        check = ComparisonCheck(10, ">")
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([False, False, True, True])
        assert_series_equal(result, expected)

    def test_greater_than_or_equal(self):
        """ Test the comparison check function with '>=' operator.
        """
        check = ComparisonCheck(10, ">=")
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([False, True, True, True])
        assert_series_equal(result, expected)

    def test_less_than(self):
        """ Test the comparison check function with '<' operator.
        """
        check = ComparisonCheck(10, "<")
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([True, False, False, False])
        assert_series_equal(result, expected)

    def test_less_than_or_equal(self):
        """ Test the comparison check function with '<=' operator.
        """
        check = ComparisonCheck(10, "<=")
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([True, True, False, False])
        assert_series_equal(result, expected)

    def test_equal(self):
        """ Test the comparison check function with '==' operator.
        """
        check = ComparisonCheck(10, "==")
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([False, True, False, False])
        assert_series_equal(result, expected)

    def test_not_equal(self):
        """ Test the comparison check function with '!=' operator.
        """
        check = ComparisonCheck(10, "!=")
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([True, False, True, True])
        assert_series_equal(result, expected)

    def test_flag_na_when_true(self):
        """ Test that setting flag_na to True will mean any NULL values in the check column are treated as failing
        the QC check (so qc flag set in the result).
        """
        check = ComparisonCheck(10, ">", flag_na=True)
        result = check.apply(self.ts, "value_c")
        expected = pl.Series([True, True, True, True])
        assert_series_equal(result, expected)

    def test_flag_na_when_false(self):
        """ Test that setting flag_na to False will mean that any NULL values in the check column are ignored in
        the QC check (so qc flag not set in the result).
        """
        check = ComparisonCheck(10, ">", flag_na=False)
        result = check.apply(self.ts, "value_c")
        expected = pl.Series([None, True, True, None])
        assert_series_equal(result, expected)

    def test_invalid_operator(self):
        """ Test that invalid operator raises error
        """
        with self.assertRaises(KeyError):
            check = ComparisonCheck(10, ">>")
            check.apply(self.ts, "value_a")


class TestRangeCheck(unittest.TestCase):
    def setUp(self):
        data = pl.DataFrame({
            "time": [
                datetime(2023, 8, 10, 0, 0),
                datetime(2023, 8, 10, 0, 30),
                datetime(2023, 8, 10, 1, 0),
                datetime(2023, 8, 10, 1, 30),
                datetime(2023, 8, 10, 2, 0),
                datetime(2023, 8, 10, 2, 30),
            ],
            "value_a": range(6)
        })
        self.ts = TimeSeries(data, "time")

    def test_range_check(self):
        """ Test that the range check returns expected results when some values outside of range
        """
        check = RangeCheck(0.5, 3.5)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([True, False, False, False, True, True])
        assert_series_equal(result, expected)

    def test_range_check_within(self):
        """ Test that the range check returns expected results with the within parameter set.
        """
        check = RangeCheck(0.5, 3.5, within=True)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([False, True, True, True, False, False])
        assert_series_equal(result, expected)

    def test_range_check_all_within(self):
        """ Test that the range check returns expected results when all values within range
        """
        check = RangeCheck(-1, 10)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([False, False, False, False, False, False])
        assert_series_equal(result, expected)

    def test_range_check_all_outside(self):
        """ Test that the range check returns expected results when all values within range
        """
        check = RangeCheck(90, 100)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([True, True, True, True, True, True])
        assert_series_equal(result, expected)

    def test_range_check_at_boundaries_inclusive(self):
        """ Test that the range check returns expected results when values at boundaries, with the
        inclusive option set to True
        """
        check = RangeCheck(0., 5., inclusive=True)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([False, False, False, False, False, False])
        assert_series_equal(result, expected)

    def test_range_check_at_boundaries_non_inclusive(self):
        """ Test that the range check returns expected results when values at boundaries, with the
        inclusive option set to False.
        """
        check = RangeCheck(0., 5., inclusive=False)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([True, False, False, False, False, True])
        assert_series_equal(result, expected)

    def test_range_check_using_time(self):
        """ Test that the range check accepts time values.
        """
        check = RangeCheck(time(0, 30), time(1, 30))
        result = check.apply(self.ts, "time")
        expected = pl.Series([True, False, False, False, True, True])
        assert_series_equal(result, expected)

    def test_range_check_using_time_within(self):
        """ Test that the range check works with time values, with the within parameter set to True.
        """
        check = RangeCheck(time(0, 30), time(1, 30), within=True)
        result = check.apply(self.ts, "time")
        expected = pl.Series([False, True, True, True, False, False])
        assert_series_equal(result, expected)

    def test_range_check_using_time_non_inclusive(self):
        """ Test that the range check works with time values, with the within parameter set to True,
        and inclusive set to False.
        """
        check = RangeCheck(time(0, 30), time(1, 30), within=True, inclusive=False)
        result = check.apply(self.ts, "time")
        expected = pl.Series([False, False, True, False, False, False])
        assert_series_equal(result, expected)


class TestSpikeCheck(unittest.TestCase):
    def setUp(self):
        data = pl.DataFrame({
            "time": [
                datetime(2023, 8, 10),
                datetime(2023, 8, 11),
                datetime(2023, 8, 12),
                datetime(2023, 8, 13),
                datetime(2023, 8, 14),
                datetime(2023, 8, 15),
            ],
            "value_a": range(6)
        })
        self.ts = TimeSeries(data, "time")

    def test_basic_spike(self):
        """ Test that the spike check returns expected results for a simple spike in the data
        """
        self.ts.df = self.ts.df.with_columns(pl.Series([1., 2., 3., 40., 5., 6.]).alias("value_a"))

        check = SpikeCheck(10)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([None, False, False, True, False, None])
        assert_series_equal(result, expected)

    def test_no_spike(self):
        """ Test that no flags are added for data with no spike
        """
        check = SpikeCheck(10)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([None, False, False, False, False, None])
        assert_series_equal(result, expected)

    def test_large_spike(self):
        """ Large spikes can cause the values around the spike to be flagged if method not working correctly.
        Check this is not the case here.
        """
        self.ts.df = self.ts.df.with_columns(pl.Series([1., 999., 3., 4., 999999999., 6.]).alias("value_a"))

        check = SpikeCheck(10)
        result = check.apply(self.ts, "value_a")
        expected = pl.Series([None, True, False, False, True, None])
        assert_series_equal(result, expected)

class TestCheckWithDateRange(unittest.TestCase):
    def setUp(self):
        data = pl.DataFrame({
            "time": [datetime(2023, 8, d+1) for d in range(10)],
            "value_a": range(10)
        })
        self.ts = TimeSeries(data, "time")

    def test_no_date_range(self):
        """ Test the check is applied to whole time series if no date range provided
        """
        check = ComparisonCheck(1, ">")
        result = check.apply(self.ts, "value_a", observation_interval=None)
        expected = pl.Series([False, False, True, True, True, True, True, True, True, True])
        assert_series_equal(result, expected)

    def test_with_date_range(self):
        """ Test the check is applied to part of the time series if a date range provided
        """
        check = ComparisonCheck(1, ">")
        result = check.apply(self.ts, "value_a", observation_interval=(datetime(2023, 8, 1), datetime(2023, 8, 5)))
        expected = pl.Series([False, False, True, True, True, False, False, False, False, False])
        assert_series_equal(result, expected)

    @parameterized.expand([
        # Different ways that observation_interval can specify an open-ended end date
        ("start_date_only", datetime(2023, 8, 8)),
        ("end_date_none", (datetime(2023, 8, 8), None))
    ])
    def test_with_date_range_open(self, _, observation_interval):
        """ Test the check is applied to part of the time series if a date range provided an open-ended end date
        """
        check = ComparisonCheck(1, ">")
        result = check.apply(self.ts, "value_a", observation_interval=observation_interval)
        expected = pl.Series([False, False, False, False, False, False, False, True, True, True])
        assert_series_equal(result, expected)
