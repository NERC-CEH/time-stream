import unittest
from unittest.mock import patch
from datetime import date, datetime

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal, assert_frame_equal

from time_stream.base import TimeSeries
from time_stream.enums import DuplicateOption
from time_stream.period import Period
from time_stream.columns import DataColumn, FlagColumn, PrimaryTimeColumn, SupplementaryColumn, TimeSeriesColumn


class TestInitSupplementaryColumn(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "data_column": [1, 2, 3],
        "supp_column1": ["a", "b", "c"],
        "supp_column2": ["x", "y", "z"]
    })

    @parameterized.expand([
        ("int", "new_int_col", 5, pl.Int32),
        ("float", "new_float_col", 3.14, pl.Float64),
        ("str", "new_str_col", "test", pl.String),
        ("none", "new_null_col", None, pl.Null),
    ])
    def test_init_supplementary_column_with_single_value(self, _, new_col_name, new_col_value, new_col_type):
        """Test initialising a supplementary column with a single value (None, int, float or str)."""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, [new_col_value] * len(self.df), new_col_type))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand([
        ("int_int", "new_int_col", 5, pl.Int32),
        ("int_float", "new_float_col", 3, pl.Float64),
        ("float_int", "new_int_col", 3.4, pl.Int32),  # Expect this to convert to 3
        ("none_float", "new_float_col", None, pl.Float64),
        ("none_string", "new_str_col", None, pl.String),
        ("str_int", "new_int_col", "5", pl.Int32),
        ("str_float", "new_float_col", "3.5", pl.Float64),
        ("float_string", "new_null_col", 3.4, pl.String),
    ])
    def test_init_supplementary_column_with_single_value_and_dtype(self, _, new_col_name, new_col_value, dtype):
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, [new_col_value] * len(self.df)).cast(dtype))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand([
        ("str_int", "new_int_col", "test", pl.Int32),
        ("str_float", "new_float_col", "test", pl.Float64),
    ])
    def test_init_supplementary_column_with_single_value_and_bad_dtype(self, _, new_col_name, new_col_value, dtype):
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(pl.exceptions.InvalidOperationError):
            ts.init_supplementary_column(new_col_name, new_col_value, dtype)

    @parameterized.expand([
        ("int_iterable", "new_int_iter_col", [5, 6, 7]),
        ("float_iterable", "new_float_iter_col", [1.1, 1.2, 1.3]),
        ("str_iterable", "new_str_iter_col", ["a", "b", "c"]),
        ("none_iterable", "new_null_iter_col", [None, None, None]),
    ])
    def test_init_supp_column_with_iterable(self, _, new_col_name, new_col_value):
        """Test initialising a supplementary column with an iterable."""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, new_col_value))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand([
        ("too_short", "new_too_short_iter", [5, 6]),
        ("too_long", "new_too_long_iter", [5, 6, 7, 8, 9, 10])
    ])
    def test_init_supp_column_with_wrong_len_iterable_raises_error(self, _, new_col_name, new_col_value):
        """Test initialising a supplementary column with an iterable of the wrong length raises an error."""
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(pl.ShapeError):
            ts.init_supplementary_column(new_col_name, new_col_value)

    @parameterized.expand([
        ("int_int", "new_int_col", [5, 6, 7], pl.Int32),
        ("int_float", "new_float_col", [3, 4, 5], pl.Float64),
        ("none_float", "new_float_col", [None, None, None], pl.Float64),
        ("str_int", "new_int_col", ["5", "6", "7"], pl.Int32),
        ("none_string", "new_str_col", [None, None, None], pl.String),
    ])
    def test_init_supp_column_with_iterable_and_dtype(self, _, new_col_name, new_col_value, dtype):
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, new_col_value).cast(dtype))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand([
        ("str_int", "new_int_col", ["t1", "t2", "t3"], pl.Int32),
        ("str_float", "new_float_col", ["t1", "t2", "t3"], pl.Float64),
        ("mix_float", "new_float_col", [4.5, "3", None], pl.Float64),
    ])
    def test_init_supp_column_with_iterable_and_bad_dtype(self, name, new_col_name, new_col_value, dtype):
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        if name == "mix_float":
            # Special case where mixed types cause TypeError
            with self.assertRaises(TypeError):
                ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        else:
            with self.assertRaises(pl.exceptions.InvalidOperationError):
                ts.init_supplementary_column(new_col_name, new_col_value, dtype)


class TestSetSupplementaryColumns(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "data_column": [1, 2, 3],
        "supp_column1": ["a", "b", "c"],
        "supp_column2": ["x", "y", "z"]
    })

    def test_no_supplementary_columns(self):
        """Test that a TimeSeries object is initialised without any supplementary columns set."""
        ts = TimeSeries(self.df, time_name="time")
        expected_data_columns = {"data_column": DataColumn("data_column", ts),
                                 "supp_column1": DataColumn("supp_column1", ts),
                                 "supp_column2": DataColumn("supp_column2", ts),
                                 }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, {})

    def test_empty_list(self):
        """Test that an empty list behaves the same as no list sent"""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column([])
        expected_data_columns = {"data_column": DataColumn("data_column", ts),
                                 "supp_column1": DataColumn("supp_column1", ts),
                                 "supp_column2": DataColumn("supp_column2", ts),
                                 }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, {})

    def test_single_supplementary_column(self):
        """Test that a single supplementary column is set correctly."""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column(["supp_column1"])
        expected_data_columns = {"data_column": DataColumn("data_column", ts),
                                 "supp_column2": DataColumn("supp_column2", ts),
                                 }
        expected_supp_columns = {"supp_column1": SupplementaryColumn("supp_column1", ts),
                                 }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)

    def test_multiple_supplementary_column(self):
        """Test that multiple supplementary columns are set correctly."""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column(["supp_column1", "supp_column2"])
        expected_data_columns = {"data_column": DataColumn("data_column", ts),
                                 }
        expected_supp_columns = {"supp_column1": SupplementaryColumn("supp_column1", ts),
                                 "supp_column2": SupplementaryColumn("supp_column2", ts),
                                 }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)

    @parameterized.expand([
        ("One bad", ["data_column", "supp_column1", "supp_column2", "non_col"]),
        ("multi bad", ["data_column", "bad_col", "non_col"]),
        ("All bad", ["bad_col", "non_col"]),
    ])
    def test_bad_supplementary_columns(self, _, supplementary_columns):
        """Test that error raised for supplementary columns specified that are not in df"""
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(KeyError):
            ts.set_supplementary_column(supplementary_columns)

    def test_appending_supplementary_column(self):
        """Test that adding supplementary columns maintains existing supplementary columns."""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column(["supp_column1"])
        ts.set_supplementary_column(["supp_column2"])
        expected_data_columns = {"data_column": DataColumn("data_column", ts),
                                 }
        expected_supp_columns = {"supp_column1": SupplementaryColumn("supp_column1", ts),
                                 "supp_column2": SupplementaryColumn("supp_column2", ts),
                                 }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)


class TestCheckResolution(unittest.TestCase):
    """Test the check_resolution method."""

    def _check_success(self, times, resolution):
        """Test that a TimeSeries.check_resolution call returns True"""
        self.assertTrue(TimeSeries.check_resolution(pl.Series("time", times),resolution))

    def _check_failure(self, times, resolution):
        """Test that a TimeSeries.check_resolution call returns False"""
        self.assertFalse(TimeSeries.check_resolution(pl.Series("time", times),resolution))

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("yearly with gaps",
         [datetime(1950, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("long term yearly",
         [datetime(500, 1, 1), datetime(1789, 1, 1), datetime(2099, 1, 1)],
         Period.of_years(1)),

        ("water years",
         [datetime(2006, 10, 1, 9), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_check_year_resolution_success(self, _, times, resolution):
        """ Test that a year based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("simple yearly error",
         [datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("water years error",
         [datetime(2006, 10, 1, 10), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_check_year_resolution_failure(self, _, times, resolution):
        """ Test that a year based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
         Period.of_months(1)),

        ("monthly with gaps",
         [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2024, 1, 1)],
         Period.of_months(1)),

        ("long term monthly",
         [datetime(1200, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)],
         Period.of_months(1)),

        ("6 monthly",
         [datetime(2024, 1, 1), datetime(2024, 7, 1), datetime(2025, 1, 1)],
         Period.of_months(6)),

        ("3 monthly (quarterly)",
         [datetime(2024, 1, 1), datetime(2024, 4, 1), datetime(2024, 7, 1), datetime(2024, 10, 1), datetime(2024, 1, 1)],
         Period.of_months(3)),

        ("monthly with mid-month offset",
         [datetime(2024, 1, 15), datetime(2024, 3, 15), datetime(2024, 3, 15)],
         Period.of_months(1).with_day_offset(14)),

        ("water months",
         [datetime(2024, 1, 1, 9), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_check_month_resolution_success(self, _, times, resolution):
        """ Test that a month based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 1)],
         Period.of_months(1)),

        ("water months error",
         [datetime(2024, 1, 1, 9, 20), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_check_month_resolution_failure(self, _, times, resolution):
        """ Test that a month based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("daily with gaps",
         [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2024, 1, 10)],
         Period.of_days(1)),

        ("long term daily",
         [datetime(1800, 1, 1), datetime(2023, 1, 2), datetime(2024, 1, 10)],
         Period.of_days(1)),

        ("water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),

        ("daily across leap year feb",
         [datetime(2024, 2, 28), datetime(2024, 2, 29), datetime(2024, 3, 1)],
         Period.of_days(1)),
    ])
    def test_check_day_resolution_success(self, _, times, resolution):
        """ Test that a day based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("simple daily error",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3, 0, 0, 0, 1)],
         Period.of_days(1)),

        ("water days error",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 19)],
         Period.of_days(1).with_hour_offset(9)),
    ])
    def test_check_day_resolution_failure(self, _, times, resolution):
        """ Test that a day based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         Period.of_hours(1)),

        ("hourly with gaps",
         [datetime(2023, 1, 1, 6), datetime(2023, 6, 8, 19), datetime(2024, 3, 10, 4)],
         Period.of_hours(1)),

        ("every 12 hours",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 12), datetime(2024, 1, 2), datetime(2024, 1, 2, 12)],
         Period.of_hours(12)),

        ("every 12 hours starting at non-midnight",
         [datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 17), datetime(2024, 1, 2, 5), datetime(2024, 1, 2, 17)],
         Period.of_hours(12).with_hour_offset(5)),

        ("long term hourly",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 18), datetime(2024, 1, 10, 23)],
         Period.of_hours(1)),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 30)],
         Period.of_hours(1).with_minute_offset(30)),

        ("hourly across leap year feb",
         [datetime(2024, 2, 29, 23), datetime(2024, 3, 1), datetime(2024, 3, 1, 1)],
         Period.of_hours(1)),
    ])
    def test_check_hour_resolution_success(self, _, times, resolution):
        """ Test that an hour based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2, 15), datetime(2024, 1, 1, 3)],
         Period.of_hours(1)),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 31)],
         Period.of_hours(1).with_minute_offset(30)),
    ])
    def test_check_hour_resolution_failure(self, _, times, resolution):
        """ Test that an hour based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         Period.of_minutes(1)),

        ("minutes with gaps",
         [datetime(2023, 1, 1, 1, 1), datetime(2023, 12, 1, 19, 5), datetime(2024, 2, 25, 12, 52)],
         Period.of_minutes(1)),

        ("every 15 minutes",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 1, 45), datetime(2024, 1, 1, 2)],
         Period.of_minutes(15)),

        ("every 60 minutes",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         Period.of_minutes(15)),

        ("long term minutes",
         [datetime(1800, 1, 1, 15, 1), datetime(2023, 1, 2, 18, 18), datetime(2024, 1, 10, 23, 55)],
         Period.of_minutes(1)),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 30, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30)),
    ])
    def test_check_minute_resolution_success(self, _, times, resolution):
        """ Test that a minute based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3, 59)],
         Period.of_minutes(1)),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 31, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30)),
    ])
    def test_check_minute_resolution_failure(self, _, times, resolution):
        """ Test that a minute based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),

        ("seconds with gaps",
         [datetime(1823, 1, 1, 1, 1, 1), datetime(2023, 7, 11, 12, 19, 59), datetime(2024, 1, 1, 1, 1, 13)],
         Period.of_seconds(1)),

        ("every 5 seconds",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 5), datetime(2024, 1, 1, 1, 1, 10)],
         Period.of_seconds(5)),

        ("every 86400 seconds",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_seconds(86400)),

        ("long term seconds",
         [datetime(1800, 1, 1, 15, 1, 0), datetime(2023, 1, 2, 18, 18, 42), datetime(2024, 1, 10, 23, 55, 5)],
         Period.of_seconds(1)),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 40), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40)),
    ])
    def test_check_second_resolution_success(self, _, times, resolution):
        """ Test that a second based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2, 9000), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 41), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40)),
    ])
    def test_check_second_resolution_failure(self, _, times, resolution):
        """ Test that a second based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)

    @parameterized.expand([
        ("simple microseconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001), datetime(2024, 1, 1, 1, 1, 1, 1002)],
         Period.of_microseconds(1)),

        ("microseconds with gaps",
         [datetime(2023, 1, 1, 1, 1, 1, 5), datetime(2023, 7, 11, 12, 19, 59, 10), datetime(2024, 1, 1, 1, 1, 13, 9595)],
         Period.of_microseconds(5)),

        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000), datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),

        ("long term microseconds",
         [datetime(1800, 1, 1, 15, 1, 0, 0), datetime(2023, 1, 2, 18, 18, 42, 1), datetime(2024, 1, 10, 23, 55, 5, 10)],
         Period.of_microseconds(1)),

        ("every 1 second",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2)],
         Period.of_microseconds(1_000_000)),
    ])
    def test_check_microsecond_resolution_success(self, _, times, resolution):
        """ Test that a microsecond based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution)

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000), datetime(2024, 1, 1, 1, 1, 1, 55_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_check_microsecond_resolution_failure(self, _, times, resolution):
        """ Test that a microsecond based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution)


class TestCheckPeriodicity(unittest.TestCase):
    """Test the check_periodicity method."""

    def _check_success(self, times, periodicity):
        """Test that a TimeSeries.check_periodicity call returns True"""
        self.assertTrue(TimeSeries.check_periodicity(pl.Series("time", times),periodicity))

    def _check_failure(self, times, periodicity):
        """Test that a TimeSeries.check_periodicity call returns False"""
        self.assertFalse(TimeSeries.check_periodicity(pl.Series("time", times),periodicity))

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("days within yearly",
         [datetime(2021, 1, 1), datetime(2022, 10, 5), datetime(2023, 2, 17)],
         Period.of_years(1)),

        ("long term yearly",
         [datetime(500, 1, 1), datetime(1789, 1, 1), datetime(2099, 1, 1)],
         Period.of_years(1)),

        ("simple water year",
         [datetime(2005, 1, 1), datetime(2006, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),

        ("water year before/after Oct 1 9am",
         [datetime(2006, 10, 1, 8, 59), datetime(2006, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_check_year_periodicity_success(self, _, times, periodicity):
        """ Test that a year based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("simple water year",
         [datetime(2005, 1, 1), datetime(2005, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_check_year_periodicity_failure(self, _, times, periodicity):
        """ Test that a year based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 31)],
         Period.of_months(1)),

        ("long term monthly",
         [datetime(1200, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)],
         Period.of_months(1)),

        ("6 monthly",
         [datetime(2024, 4, 16), datetime(2024, 7, 1), datetime(2025, 2, 25)],
         Period.of_months(6)),

        ("3 monthly (quarterly)",
         [datetime(2024, 1, 1), datetime(2024, 4, 9), datetime(2024, 9, 2), datetime(2024, 12, 31)],
         Period.of_months(3)),

        ("simple water months",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 5), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9)),

        ("water months before/after 9am",
         [datetime(2024, 1, 1, 8, 59), datetime(2024, 1, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_check_month_periodicity_success(self, _, times, periodicity):
        """ Test that a month based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 1, 31), datetime(2024, 2, 1)],
         Period.of_months(1)),

        ("simple water months",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 1, 8, 59), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_check_month_periodicity_failure(self, _, times, periodicity):
        """ Test that a month based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("long term daily",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 19), datetime(2024, 1, 10, 1)],
         Period.of_days(1)),

        ("simple water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),

        ("water days before/after 9am",
         [datetime(2024, 1, 1, 8, 59), datetime(2024, 1, 1, 9)],
         Period.of_days(1).with_hour_offset(9)),

        ("daily across leap year feb",
         [datetime(2024, 2, 28, 15), datetime(2024, 2, 29, 23, 59), datetime(2024, 3, 1)],
         Period.of_days(1)),
    ])
    def test_check_day_periodicity_success(self, _, times, periodicity):
        """ Test that a day based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 23, 59), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("simple water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 8, 59, 59), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),
    ])
    def test_check_day_periodicity_failure(self, _, times, periodicity):
        """ Test that a day based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 2, 59), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1)),

        ("every 12 hours",
         [datetime(2024, 1, 1, 4, 30), datetime(2024, 1, 1, 12, 5), datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)],
         Period.of_hours(12)),

        ("long term hourly",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 18), datetime(2024, 1, 10, 23)],
         Period.of_hours(1)),

        ("hourly across leap year feb",
         [datetime(2024, 2, 29, 23, 59), datetime(2024, 3, 1), datetime(2024, 3, 1, 1)],
         Period.of_hours(1)),
    ])
    def test_check_hour_periodicity_success(self, _, times, periodicity):
        """ Test that an hour based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 16), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1)),
    ])
    def test_check_hour_periodicity_failure(self, _, times, periodicity):
        """ Test that an hour based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         Period.of_minutes(1)),

        ("every 15 minutes",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 35), datetime(2024, 1, 1, 1, 59)],
         Period.of_minutes(15)),

        ("long term minutes",
         [datetime(1800, 1, 1, 15, 1), datetime(2023, 1, 2, 18, 18), datetime(2024, 1, 10, 23, 55)],
         Period.of_minutes(1)),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 30, 29), datetime(2024, 1, 1, 1, 30, 31), datetime(2024, 1, 1, 1, 35, 30)],
         Period.of_minutes(5).with_second_offset(30)),
    ])
    def test_check_minute_periodicity_success(self, _, times, periodicity):
        """ Test that a minute based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 2, 1)],
         Period.of_minutes(1)),
    ])
    def test_check_minute_periodicity_failure(self, _, times, periodicity):
        """ Test that a minute based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3, 40000)],
         Period.of_seconds(1)),

        ("every 5 seconds",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 7), datetime(2024, 1, 1, 1, 1, 14)],
         Period.of_seconds(5)),

        ("long term seconds",
         [datetime(1800, 1, 1, 15, 1, 0), datetime(2023, 1, 2, 18, 18, 42), datetime(2024, 1, 10, 23, 55, 5)],
         Period.of_seconds(1)),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 39), datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 100000)],
         Period.of_seconds(30).with_microsecond_offset(40)),
    ])
    def test_check_second_periodicity_success(self, _, times, periodicity):
        """ Test that a second based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 1, 15001), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),
    ])
    def test_check_second_periodicity_failure(self, _, times, periodicity):
        """ Test that a second based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)

    @parameterized.expand([
        ("simple microseconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001),
          datetime(2024, 1, 1, 1, 1, 1, 1002)],
         Period.of_microseconds(1)),

        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 12_521), datetime(2024, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),

        ("long term microseconds",
         [datetime(1800, 1, 1, 15, 1, 0, 0), datetime(2023, 1, 2, 18, 18, 42, 1), datetime(2024, 1, 10, 23, 55, 5, 10)],
         Period.of_microseconds(1)),
    ])
    def test_check_microsecond_periodicity_success(self, _, times, periodicity):
        """ Test that a microsecond based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity)

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 25_001), datetime(2023, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_check_microsecond_periodicity_failure(self, _, times, periodicity):
        """ Test that a microsecond based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity)


class TestTruncateToPeriod(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2020, 6, 1, 8, 10, 5, 2000),
                 datetime(2021, 4, 30, 15, 30, 10, 250),
                 datetime(2022, 12, 31, 12)]
    })

    @parameterized.expand([
        ("simple yearly", Period.of_years(1),
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)]
         ),
        ("yearly with offset", Period.of_years(1).with_month_offset(2),
         [datetime(2020, 3, 1), datetime(2021, 3, 1), datetime(2022, 3, 1)]
         )
    ])
    def test_truncate_to_year_period(self, _, period, expected):
        """ Test that truncating a time series to a given year period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple monthly", Period.of_months(1),
         [datetime(2020, 6, 1), datetime(2021, 4, 1), datetime(2022, 12, 1)]
         ),
        ("3 monthly (quarterly)", Period.of_months(3),
         [datetime(2020, 4, 1), datetime(2021, 4, 1), datetime(2022, 10, 1)]
         ),
        ("monthly with offset", Period.of_months(1).with_hour_offset(9),
         [datetime(2020, 5, 1, 9), datetime(2021, 4, 1, 9), datetime(2022, 12, 1, 9)]
         )
    ])
    def test_truncate_to_month_period(self, _, period, expected):
        """ Test that truncating a time series to a given month period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple daily", Period.of_days(1),
         [datetime(2020, 6, 1), datetime(2021, 4, 30), datetime(2022, 12, 31)]
         ),
        ("daily with offset", Period.of_days(1).with_hour_offset(9),
         [datetime(2020, 5, 31, 9), datetime(2021, 4, 30, 9), datetime(2022, 12, 31, 9)]
         )
    ])
    def test_truncate_to_day_period(self, _, period, expected):
        """ Test that truncating a time series to a given day period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple hourly", Period.of_hours(1),
         [datetime(2020, 6, 1, 8), datetime(2021, 4, 30, 15), datetime(2022, 12, 31, 12)]
         ),
        ("12 hourly", Period.of_hours(12),
         [datetime(2020, 6, 1), datetime(2021, 4, 30, 12), datetime(2022, 12, 31, 12)]
         ),
        ("hourly with offset", Period.of_hours(1).with_minute_offset(30),
         [datetime(2020, 6, 1, 7, 30), datetime(2021, 4, 30, 15, 30), datetime(2022, 12, 31, 11, 30)]
         )
    ])
    def test_truncate_to_hour_period(self, _, period, expected):
        """ Test that truncating a time series to a given hour period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple minutes", Period.of_minutes(1),
         [datetime(2020, 6, 1, 8, 10), datetime(2021, 4, 30, 15, 30), datetime(2022, 12, 31, 12)]
         ),
        ("15 minutes", Period.of_minutes(15),
         [datetime(2020, 6, 1, 8), datetime(2021, 4, 30, 15, 30), datetime(2022, 12, 31, 12)]
         ),
        ("minutes with offset", Period.of_minutes(1).with_second_offset(45),
         [datetime(2020, 6, 1, 8, 9, 45), datetime(2021, 4, 30, 15, 29, 45), datetime(2022, 12, 31, 11, 59, 45)]
         )
    ])
    def test_truncate_to_minute_period(self, _, period, expected):
        """ Test that truncating a time series to a given minute period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple seconds", Period.of_seconds(1),
         [datetime(2020, 6, 1, 8, 10, 5), datetime(2021, 4, 30, 15, 30, 10), datetime(2022, 12, 31, 12)]
         ),
        ("3 seconds", Period.of_seconds(3),
         [datetime(2020, 6, 1, 8, 10, 3), datetime(2021, 4, 30, 15, 30, 9), datetime(2022, 12, 31, 12)]
         )
    ])
    def test_truncate_to_second_period(self, _, period, expected):
        """ Test that truncating a time series to a given second period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple microseconds", Period.of_microseconds(200),
         [datetime(2020, 6, 1, 8, 10, 5, 2000), datetime(2021, 4, 30, 15, 30, 10, 200), datetime(2022, 12, 31, 12)]
         ),
    ])
    def test_truncate_to_microsecond_period(self, _, period, expected):
        """ Test that truncating a time series to a given microsecond period works as expected.
        """
        result = TimeSeries.truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))


class TestValidateResolution(unittest.TestCase):
    """Test the _validate_resolution method."""

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),
    
        ("simple yearly string",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         "P1Y"),

        ("yearly with gaps",
         [datetime(1950, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("long term yearly",
         [datetime(500, 1, 1), datetime(1789, 1, 1), datetime(2099, 1, 1)],
         Period.of_years(1)),

        ("water years",
         [datetime(2006, 10, 1, 9), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),

        ("water years string",
         [datetime(2006, 10, 1, 9), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         "P1Y+9MT9H"),
    ])
    def test_validate_year_resolution_success(self, _, times, resolution):
        """ Test that a year based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("simple yearly error",
         [datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("simple yearly string error",
         [datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)],
         "P1Y"),

        ("water years error",
         [datetime(2006, 10, 1, 10), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),

        ("water years string error",
         [datetime(2006, 10, 1, 10), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         "P1Y+9MT9H"),
    ])
    def test_validate_year_resolution_error(self, _, times, resolution):
        """ Test that a year based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
         Period.of_months(1)),

        ("simple monthly string",
         [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
         "P1M"),

        ("monthly with gaps",
         [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2024, 1, 1)],
         Period.of_months(1)),

        ("long term monthly",
         [datetime(1200, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)],
         Period.of_months(1)),

        ("6 monthly",
         [datetime(2024, 1, 1), datetime(2024, 7, 1), datetime(2025, 1, 1)],
         Period.of_months(6)),

        ("3 monthly (quarterly)",
         [datetime(2024, 1, 1), datetime(2024, 4, 1), datetime(2024, 7, 1), datetime(2024, 10, 1),
          datetime(2024, 1, 1)],
         Period.of_months(3)),

        ("monthly with mid-month offset",
         [datetime(2024, 1, 15), datetime(2024, 3, 15), datetime(2024, 3, 15)],
         Period.of_months(1).with_day_offset(14)),

        ("water months",
         [datetime(2024, 1, 1, 9), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),

        ("water months string",
         [datetime(2024, 1, 1, 9), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         "P1M+T9H")
    ])
    def test_validate_month_resolution_success(self, _, times, resolution):
        """ Test that a month based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("simple monthly error",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 1)],
         Period.of_months(1)),

        ("simple monthly string error",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 1)],
         "P1M"),

        ("water months error",
         [datetime(2024, 1, 1, 9, 20), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),

        ("water months string error",
         [datetime(2024, 1, 1, 9, 20), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         "P1M+T9H")
    ])
    def test_validate_month_resolution_error(self, _, times, resolution):
        """ Test that a month based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         "P1D"),

        ("daily with gaps",
         [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2024, 1, 10)],
         Period.of_days(1)),

        ("long term daily",
         [datetime(1800, 1, 1), datetime(2023, 1, 2), datetime(2024, 1, 10)],
         Period.of_days(1)),

        ("water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),

        ("water days string",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         "P1D+T9H"),

        ("daily across leap year feb",
         [datetime(2024, 2, 28), datetime(2024, 2, 29), datetime(2024, 3, 1)],
         Period.of_days(1)),
    ])
    def test_validate_day_resolution_success(self, _, times, resolution):
        """ Test that a day based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("simple daily error",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3, 0, 0, 0, 1)],
         Period.of_days(1)),

        ("simple daily string error",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3, 0, 0, 0, 1)],
         "P1D"),

        ("water days error",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 19)],
         Period.of_days(1).with_hour_offset(9)),

        ("water days string error",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 19)],
         "P1D+T9H"),
    ])
    def test_validate_day_resolution_error(self, _, times, resolution):
        """ Test that a day based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         Period.of_hours(1)),

        ("simple hourly string",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         "PT1H"),

        ("hourly with gaps",
         [datetime(2023, 1, 1, 6), datetime(2023, 6, 8, 19), datetime(2024, 3, 10, 4)],
         Period.of_hours(1)),

        ("every 12 hours",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 12), datetime(2024, 1, 2), datetime(2024, 1, 2, 12)],
         Period.of_hours(12)),

        ("every 12 hours starting at non-midnight",
         [datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 17), datetime(2024, 1, 2, 5), datetime(2024, 1, 2, 17)],
         Period.of_hours(12).with_hour_offset(5)),

        ("long term hourly",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 18), datetime(2024, 1, 10, 23)],
         Period.of_hours(1)),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 30)],
         Period.of_hours(1).with_minute_offset(30)),

        ("hourly with minute offset string",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 30)],
         "PT1H+T30M"),

        ("hourly across leap year feb",
         [datetime(2024, 2, 29, 23), datetime(2024, 3, 1), datetime(2024, 3, 1, 1)],
         Period.of_hours(1)),
    ])
    def test_validate_hour_resolution_success(self, _, times, resolution):
        """ Test that an hour based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("simple hourly error",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2, 15), datetime(2024, 1, 1, 3)],
         Period.of_hours(1)),

        ("simple hourly string error",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2, 15), datetime(2024, 1, 1, 3)],
         "PT1H"),

        ("hourly with minute offset error",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 31)],
         Period.of_hours(1).with_minute_offset(30)),

        ("hourly with minute offset string error",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 31)],
         "PT1H+T30M"),
    ])
    def test_validate_hour_resolution_error(self, _, times, resolution):
        """ Test that an hour based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         Period.of_minutes(1)),

        ("simple minute string",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         "PT1M"),

        ("minutes with gaps",
         [datetime(2023, 1, 1, 1, 1), datetime(2023, 12, 1, 19, 5), datetime(2024, 2, 25, 12, 52)],
         Period.of_minutes(1)),

        ("every 15 minutes",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 1, 45),
          datetime(2024, 1, 1, 2)],
         Period.of_minutes(15)),

        ("every 60 minutes",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         Period.of_minutes(15)),

        ("long term minutes",
         [datetime(1800, 1, 1, 15, 1), datetime(2023, 1, 2, 18, 18), datetime(2024, 1, 10, 23, 55)],
         Period.of_minutes(1)),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 30, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30)),

        ("5 minutes with second offset string",
         [datetime(2024, 1, 1, 1, 30, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         "PT5M+T30S")
    ])
    def test_validate_minute_resolution_success(self, _, times, resolution):
        """ Test that a minute based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("simple minute error",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3, 59)],
         Period.of_minutes(1)),

        ("simple minute string error",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3, 59)],
         "PT1M"),

        ("5 minutes with second offset error",
         [datetime(2024, 1, 1, 1, 31, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30)),

        ("5 minutes with second offset string error",
         [datetime(2024, 1, 1, 1, 31, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         "PT5M+T30S")
    ])
    def test_validate_minute_resolution_error(self, _, times, resolution):
        """ Test that a minute based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),

        ("simple seconds string",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3)],
         "PT1S"),

        ("seconds with gaps",
         [datetime(1823, 1, 1, 1, 1, 1), datetime(2023, 7, 11, 12, 19, 59), datetime(2024, 1, 1, 1, 1, 13)],
         Period.of_seconds(1)),

        ("every 5 seconds",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 5), datetime(2024, 1, 1, 1, 1, 10)],
         Period.of_seconds(5)),

        ("every 86400 seconds",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_seconds(86400)),

        ("long term seconds",
         [datetime(1800, 1, 1, 15, 1, 0), datetime(2023, 1, 2, 18, 18, 42), datetime(2024, 1, 10, 23, 55, 5)],
         Period.of_seconds(1)),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 40), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40)),

        ("30 seconds with microsecond offset string",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 40), datetime(2024, 1, 1, 1, 2, 30, 40)],
         "PT30S+T0.00004S")
    ])
    def test_validate_second_resolution_success(self, _, times, resolution):
        """ Test that a second based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("simple seconds error",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2, 9000), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),

        ("simple seconds string error",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2, 9000), datetime(2024, 1, 1, 1, 1, 3)],
         "PT1S"),

        ("30 seconds with microsecond offset error",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 41), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40)),

        ("30 seconds with microsecond offset string error",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 41), datetime(2024, 1, 1, 1, 2, 30, 40)],
         "PT30S+T0.00004S")
    ])
    def test_validate_second_resolution_error(self, _, times, resolution):
        """ Test that a second based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()

    @parameterized.expand([
        ("simple microseconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001),
          datetime(2024, 1, 1, 1, 1, 1, 1002)],
         Period.of_microseconds(1)),

        ("simple microseconds string",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001),
          datetime(2024, 1, 1, 1, 1, 1, 1002)],
         "PT0.000001S"),

        ("microseconds with gaps",
         [datetime(2023, 1, 1, 1, 1, 1, 5), datetime(2023, 7, 11, 12, 19, 59, 10),
          datetime(2024, 1, 1, 1, 1, 13, 9595)],
         Period.of_microseconds(5)),

        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),

        ("long term microseconds",
         [datetime(1800, 1, 1, 15, 1, 0, 0), datetime(2023, 1, 2, 18, 18, 42, 1), datetime(2024, 1, 10, 23, 55, 5, 10)],
         Period.of_microseconds(1)),

        ("every 1 second",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2)],
         Period.of_microseconds(1_000_000)),
    ])
    def test_validate_microsecond_resolution_success(self, _, times, resolution):
        """ Test that a microsecond based time series that does conform to the given resolution passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            ts._validate_resolution()

    @parameterized.expand([
        ("40Hz error",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000),
          datetime(2024, 1, 1, 1, 1, 1, 55_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_validate_microsecond_resolution_error(self, _, times, resolution):
        """ Test that a microsecond based time series that doesn't conform to the given resolution raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", resolution=resolution)
            with self.assertRaises(UserWarning):
                ts._validate_resolution()


class TestValidatePeriodicity(unittest.TestCase):
    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("simple yearly string",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         "P1Y"),

        ("days within yearly",
         [datetime(2021, 1, 1), datetime(2022, 10, 5), datetime(2023, 2, 17)],
         Period.of_years(1)),

        ("long term yearly",
         [datetime(500, 1, 1), datetime(1789, 1, 1), datetime(2099, 1, 1)],
         Period.of_years(1)),

        ("simple water year",
         [datetime(2005, 1, 1), datetime(2006, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),

        ("simple water year string",
         [datetime(2005, 1, 1), datetime(2006, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         "P1Y+9MT9H"),

        ("water year before/after Oct 1 9am",
         [datetime(2006, 10, 1, 8, 59), datetime(2006, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_validate_year_periodicity_success(self, _, times, periodicity):
        """ Test that a year based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("simple yearly error",
         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("simple yearly string error",
         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2022, 1, 1)],
         "P1Y"),

        ("simple water year error",
         [datetime(2005, 1, 1), datetime(2005, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),

        ("simple water year string error",
         [datetime(2005, 1, 1), datetime(2005, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         "P1Y+9MT9H")
    ])
    def test_validate_year_periodicity_error(self, _, times, periodicity):
        """ Test that a year based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 31)],
         Period.of_months(1)),

        ("simple monthly string",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 31)],
         "P1M"),

        ("long term monthly",
         [datetime(1200, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)],
         Period.of_months(1)),

        ("6 monthly",
         [datetime(2024, 4, 16), datetime(2024, 7, 1), datetime(2025, 2, 25)],
         Period.of_months(6)),

        ("3 monthly (quarterly)",
         [datetime(2024, 1, 1), datetime(2024, 4, 9), datetime(2024, 9, 2), datetime(2024, 12, 31)],
         Period.of_months(3)),

        ("simple water months",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 5), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9)),

        ("simple water months string",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 5), datetime(2024, 5, 1, 10, 15)],
         "P1M+T9H"),

        ("water months before/after 9am",
         [datetime(2024, 1, 1, 8, 59), datetime(2024, 1, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_validate_month_periodicity_success(self, _, times, periodicity):
        """ Test that a month based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("simple monthly error",
         [datetime(2024, 1, 1), datetime(2024, 1, 31), datetime(2024, 2, 1)],
         Period.of_months(1)),

        ("simple monthly string error",
         [datetime(2024, 1, 1), datetime(2024, 1, 31), datetime(2024, 2, 1)],
         "P1M"),

        ("simple water months error",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 1, 8, 59), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9)),

        ("simple water months string error",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 1, 8, 59), datetime(2024, 5, 1, 10, 15)],
         "P1M+T9H")
    ])
    def test_validate_month_periodicity_error(self, _, times, periodicity):
        """ Test that a month based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("simple daily error",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         "P1D"),

        ("long term daily",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 19), datetime(2024, 1, 10, 1)],
         Period.of_days(1)),

        ("simple water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),

        ("simple water days string",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         "P1D+T9H"),

        ("water days before/after 9am",
         [datetime(2024, 1, 1, 8, 59), datetime(2024, 1, 1, 9)],
         Period.of_days(1).with_hour_offset(9)),

        ("daily across leap year feb",
         [datetime(2024, 2, 28, 15), datetime(2024, 2, 29, 23, 59), datetime(2024, 3, 1)],
         Period.of_days(1)),
    ])
    def test_validate_day_periodicity_success(self, _, times, periodicity):
        """ Test that a day based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("simple daily error",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 23, 59), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("simple daily string error",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 23, 59), datetime(2024, 1, 3)],
         "P1D"),

        ("simple water days error",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 8, 59, 59), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),
    ])
    def test_validate_day_periodicity_error(self, _, times, periodicity):
        """ Test that a day based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 2, 59), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1)),

        ("simple hourly string",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 2, 59), datetime(2024, 1, 1, 3, 1)],
         "PT1H"),

        ("every 12 hours",
         [datetime(2024, 1, 1, 4, 30), datetime(2024, 1, 1, 12, 5), datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)],
         Period.of_hours(12)),

        ("long term hourly",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 18), datetime(2024, 1, 10, 23)],
         Period.of_hours(1)),

        ("hourly across leap year feb",
         [datetime(2024, 2, 29, 23, 59), datetime(2024, 3, 1), datetime(2024, 3, 1, 1)],
         Period.of_hours(1)),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 30)],
         Period.of_hours(1).with_minute_offset(30)),

        ("hourly with minute offset string",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 30)],
         "PT1H+T30M")
    ])
    def test_validate_hour_periodicity_success(self, _, times, periodicity):
        """ Test that an hour based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("simple hourly error",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 16), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1)),

        ("simple hourly string error",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 16), datetime(2024, 1, 1, 3, 1)],
         "PT1H"),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 2, 31)],
         Period.of_hours(1).with_minute_offset(30)),

        ("hourly with minute offset string",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 2, 31)],
         "PT1H+T30M")
    ])
    def test_validate_hour_periodicity_error(self, _, times, periodicity):
        """ Test that an hour based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         Period.of_minutes(1)),

        ("simple minute string",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         "PT1M"),

        ("every 15 minutes",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 35), datetime(2024, 1, 1, 1, 59)],
         Period.of_minutes(15)),

        ("long term minutes",
         [datetime(1800, 1, 1, 15, 1), datetime(2023, 1, 2, 18, 18), datetime(2024, 1, 10, 23, 55)],
         Period.of_minutes(1)),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 30, 29), datetime(2024, 1, 1, 1, 30, 31), datetime(2024, 1, 1, 1, 35, 30)],
         Period.of_minutes(5).with_second_offset(30)),

        ("5 minutes with second offset string",
         [datetime(2024, 1, 1, 1, 30, 29), datetime(2024, 1, 1, 1, 30, 31), datetime(2024, 1, 1, 1, 35, 30)],
         "PT5M+T30S")
    ])
    def test_validate_minute_periodicity_success(self, _, times, periodicity):
        """ Test that a minute based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("simple minute error",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 2, 1)],
         Period.of_minutes(1)),

        ("simple minute string error",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 2, 1)],
         "PT1M"),

        ("5 minutes with second offset error",
         [datetime(2024, 1, 1, 1, 30, 29), datetime(2024, 1, 1, 1, 30, 31), datetime(2024, 1, 1, 1, 30, 30)],
         Period.of_minutes(5).with_second_offset(30)),

        ("5 minutes with second offset string error",
         [datetime(2024, 1, 1, 1, 30, 29), datetime(2024, 1, 1, 1, 30, 31), datetime(2024, 1, 1, 1, 30, 30)],
         "PT5M+T1S")
    ])
    def test_validate_minute_periodicity_error(self, _, times, periodicity):
        """ Test that a minute based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3, 40000)],
         Period.of_seconds(1)),

        ("simple seconds string",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3, 40000)],
         "PT1S"),

        ("every 5 seconds",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 7), datetime(2024, 1, 1, 1, 1, 14)],
         Period.of_seconds(5)),

        ("long term seconds",
         [datetime(1800, 1, 1, 15, 1, 0), datetime(2023, 1, 2, 18, 18, 42), datetime(2024, 1, 10, 23, 55, 5)],
         Period.of_seconds(1)),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 39), datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 100000)],
         Period.of_seconds(30).with_microsecond_offset(40)),

        ("30 seconds with microsecond offset string",
         [datetime(2024, 1, 1, 1, 1, 0, 39), datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 100000)],
         "PT30S+T0.00004S"),
    ])
    def test_validate_second_periodicity_success(self, _, times, periodicity):
        """ Test that a second based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("simple seconds error",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 1, 15001), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),

        ("simple seconds string error",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 1, 15001), datetime(2024, 1, 1, 1, 1, 3)],
         "PT1S"),

        ("30 seconds with microsecond offset error",
         [datetime(2024, 1, 1, 1, 1, 0, 39), datetime(2024, 1, 1, 1, 1, 30, 40), datetime(2024, 1, 1, 1, 1, 30, 100000)],
         Period.of_seconds(30).with_microsecond_offset(40)),

        ("30 seconds with microsecond offset string error",
         [datetime(2024, 1, 1, 1, 1, 0, 39), datetime(2024, 1, 1, 1, 1, 30, 40), datetime(2024, 1, 1, 1, 1, 30, 100000)],
         "PT30S+T0.00004S")
    ])
    def test_validate_second_periodicity_error(self, _, times, periodicity):
        """ Test that a second based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()

    @parameterized.expand([
        ("simple microseconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001),
          datetime(2024, 1, 1, 1, 1, 1, 1002)],
         Period.of_microseconds(1)),

        ("simple microseconds string",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001),
          datetime(2024, 1, 1, 1, 1, 1, 1002)],
         "PT0.000001S"),

        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 12_521), datetime(2024, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),

        ("long term microseconds",
         [datetime(1800, 1, 1, 15, 1, 0, 0), datetime(2023, 1, 2, 18, 18, 42, 1), datetime(2024, 1, 10, 23, 55, 5, 10)],
         Period.of_microseconds(1)),
    ])
    def test_validate_microsecond_periodicity_success(self, _, times, periodicity):
        """ Test that a microsecond based time series that does conform to the given periodicity passes the validation.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            ts._validate_periodicity()

    @parameterized.expand([
        ("40Hz error",
         [datetime(2023, 1, 1, 1, 1, 1, 25_001), datetime(2023, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_validate_microsecond_periodicity_error(self, _, times, periodicity):
        """ Test that a microsecond based time series that doesn't conform to the given periodicity raises an error.
        """
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time", periodicity=periodicity)
            with self.assertRaises(UserWarning):
                ts._validate_periodicity()


class TestEpochCheck(unittest.TestCase):
    @parameterized.expand([
        Period.of_years(2), Period.of_years(7), Period.of_years(10),
        Period.of_months(5), Period.of_months(7), Period.of_months(9), Period.of_months(10), Period.of_months(11),
        Period.of_months(13),
        Period.of_days(2), Period.of_days(7), Period.of_days(65),
        Period.of_hours(5), Period.of_hours(7), Period.of_hours(9), Period.of_hours(11), Period.of_hours(25),
        Period.of_minutes(7), Period.of_minutes(11), Period.of_minutes(50), Period.of_minutes(61),
    ])
    def test_non_epoch_agnostic_period_fails(self, period):
        """ Test that non epoch agnostic Periods fail the epoch check.
        """
        with self.assertRaises(NotImplementedError):
            TimeSeries._epoch_check(period)

    @parameterized.expand([
        Period.of_years(1),
        Period.of_months(1), Period.of_months(2), Period.of_months(3), Period.of_months(4), Period.of_months(6),
        Period.of_days(1),
        Period.of_hours(1), Period.of_hours(2), Period.of_hours(3), Period.of_hours(4), Period.of_hours(24),
        Period.of_minutes(1), Period.of_minutes(2), Period.of_minutes(15), Period.of_minutes(30), Period.of_minutes(60),
    ])
    def test_epoch_agnostic_period_passes(self, period):
        """ Test that epoch agnostic Periods pass the epoch check.
        """
        TimeSeries._epoch_check(period)


class TestSortTime(unittest.TestCase):
    def test_sort_random_dates(self):
        """Test that random dates are sorted appropriately
        """
        times = [date(1990, 1, 1), date(2019, 5, 8), date(1967, 12, 25), date(2059, 8, 12)]
        expected = pl.Series("time", [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)])

        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
            ts._setup_columns()
            ts.sort_time()
            assert_series_equal(ts.time_column.data, expected)

    def test_sort_sorted_dates(self):
        """Test that already sorted dates are maintained
        """
        times = [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)]
        expected = pl.Series("time", times)

        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
            ts._setup_columns()
            ts.sort_time()
            assert_series_equal(ts.time_column.data, expected)

    def test_sort_times(self):
        """Test that times are sorted appropriately
        """
        times = [datetime(2024, 1, 1, 12, 59), datetime(2024, 1, 1, 12, 55), datetime(2024, 1, 1, 12, 18),
                 datetime(2024, 1, 1, 1, 5)]

        expected = pl.Series("time", [datetime(2024, 1, 1, 1, 5), datetime(2024, 1, 1, 12, 18),
                                      datetime(2024, 1, 1, 12, 55), datetime(2024, 1, 1, 12, 59)])
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
            ts._setup_columns()
            ts.sort_time()
            assert_series_equal(ts.time_column.data, expected)


class TestValidateTimeColumn(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "data_column": [1, 2, 3]
    })

    def test_validate_unchanged_dataframe(self):
        """Test that an unchanged DataFrame passes validation."""
        ts = TimeSeries(self.df, time_name="time")
        ts._validate_time_column(ts._df)

    def test_validate_dataframe_changed_values(self):
        """Test that a DataFrame with different values, but same times, passes validation."""
        ts = TimeSeries(self.df, time_name="time")
        new_df = pl.DataFrame({
            "time": ts.df["time"],
            "data_column": [10, 20, 30]
        })
        ts._validate_time_column(new_df)

    def test_validate_time_column_missing_time_column(self):
        """Test that a DataFrame missing the time column raises a ValueError."""
        invalid_df = pl.DataFrame(self.df["data_column"])
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(ValueError):
            ts._validate_time_column(invalid_df)

    def test_validate_time_column_mutated_timestamps(self):
        """Test that a DataFrame with mutated timestamps raises a ValueError."""
        ts = TimeSeries(self.df, time_name="time")
        mutated_df = pl.DataFrame({
            "time": [datetime(1924, 1, 1), datetime(1924, 1, 2), datetime(1924, 1, 3)]
        })
        with self.assertRaises(ValueError):
            ts._validate_time_column(mutated_df)

    def test_validate_time_column_extra_timestamps(self):
        """Test that a DataFrame with extra timestamps raises a ValueError."""
        ts = TimeSeries(self.df, time_name="time")
        extra_timestamps_df = pl.DataFrame({
            "time": list(ts.df["time"]) + [datetime(2024, 1, 4), datetime(2024, 1, 5)]
        })
        with self.assertRaises(ValueError):
            ts._validate_time_column(extra_timestamps_df)

    def test_validate_time_column_fewer_timestamps(self):
        """Test that a DataFrame with fewer timestamps raises a ValueError."""
        ts = TimeSeries(self.df, time_name="time")
        fewer_timestamps_df = pl.DataFrame({
            "time": list(ts.df["time"])[:-1]
        })
        with self.assertRaises(ValueError):
            ts._validate_time_column(fewer_timestamps_df)


class TestRemoveMissingColumns(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })
    metadata = {
        "col1": {"description": "meta1"},
        "col2": {"description": "meta2"},
        "col3": {"description": "meta3"}
    }

    def test_no_columns_removed(self):
        """Test that no columns are removed when all columns are present in the new DataFrame."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        ts._remove_missing_columns(ts._df, self.df)
        ts._df = self.df

        self.assertEqual(list(ts.columns.keys()), ["col1", "col2", "col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), ["col1"])
        self.assertEqual(list(ts.data_columns.keys()), ["col2", "col3"])
        self.assertEqual(ts.col1.metadata(), self.metadata["col1"])
        self.assertEqual(ts.col2.metadata(), self.metadata["col2"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

    def test_single_data_column_removed(self):
        """Test that single data column is removed correctly."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        new_df = self.df.drop("col2")
        ts._remove_missing_columns(ts._df, new_df)
        ts._df = new_df

        self.assertEqual(list(ts.columns.keys()), ["col1", "col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), ["col1"])
        self.assertEqual(list(ts.data_columns.keys()), ["col3"])
        self.assertEqual(ts.col1.metadata(), self.metadata["col1"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

        # try to access removed columns
        with self.assertRaises(AttributeError):
            _ = ts.col2

    def test_single_supplementary_column_removed(self):
        """Test that single supplementary column is removed correctly."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        new_df = self.df.drop("col1")
        ts._remove_missing_columns(ts._df, new_df)
        ts._df = new_df

        self.assertEqual(list(ts.columns.keys()), ["col2", "col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), [])
        self.assertEqual(list(ts.data_columns.keys()), ["col2", "col3"])
        self.assertEqual(ts.col2.metadata(), self.metadata["col2"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

        # try to access removed columns
        with self.assertRaises(AttributeError):
            _ = ts.col1

    def test_multiple_columns_removed(self):
        """Test that multiple columns are removed correctly."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        new_df = self.df.drop(["col1", "col2"])
        ts._remove_missing_columns(ts._df, new_df)
        ts._df = new_df

        self.assertEqual(list(ts.columns.keys()), ["col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), [])
        self.assertEqual(list(ts.data_columns.keys()), ["col3"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

        # try to access removed columns
        with self.assertRaises(AttributeError):
            _ = ts.col1
        with self.assertRaises(AttributeError):
            _ = ts.col2


class TestAddNewColumns(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })

    def test_add_new_columns(self):
        """Test that new columns are correctly added as DataColumns."""
        ts = TimeSeries(self.df, time_name="time")
        new_df = self.df.with_columns(pl.Series("col4", [1.1, 2.2, 3.3]))

        with patch.object(TimeSeriesColumn, "_validate_name", lambda *args, **kwargs: None):
            ts._add_new_columns(ts._df, new_df)

        self.assertIn("col4", ts.columns)
        self.assertIsInstance(ts.col4, DataColumn)

    def test_no_changes_when_no_new_columns(self):
        """Test that does nothing if there are no new columns."""
        ts = TimeSeries(self.df, time_name="time")
        original_columns = ts._columns.copy()
        ts._add_new_columns(ts._df, self.df)

        self.assertEqual(original_columns, ts._columns)

    def test_columns_added_to_relationship_manager(self):
        """Test that new columns are initialised in the relationship manager, with no relationships"""
        ts = TimeSeries(self.df, time_name="time")
        new_df = self.df.with_columns(pl.Series("col4", [1.1, 2.2, 3.3]))

        with patch.object(TimeSeriesColumn, "_validate_name", lambda *args, **kwargs: None):
            ts._add_new_columns(ts._df, new_df)

        self.assertIn("col4", ts.columns)
        self.assertIn("col4", ts._relationship_manager._relationships)
        self.assertEqual(ts._relationship_manager._relationships["col4"], set())


class TestSelectColumns(unittest.TestCase):
    times = [datetime(2024, 1, 1),
             datetime(2024, 1, 2),
             datetime(2024, 1, 3)]
    df = pl.DataFrame({
        "time": times,
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_select_single_column(self):
        """Test selecting a single of column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time",
                              column_metadata={"col1": self.metadata["col1"]})
        result = ts.select(["col1"])
        self.assertEqual(result, expected)

    def test_select_multiple_columns(self):
        """Test selecting multiple columns."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        expected = TimeSeries(self.df.select(["time", "col1", "col2"]), time_name="time",
                              column_metadata={"col1": self.metadata["col1"], "col2": self.metadata["col2"]})
        result = ts.select(["col1", "col2"])
        self.assertEqual(result, expected)

    def test_select_no_columns_raises_error(self):
        """Test selecting no columns raises error."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(KeyError):
            ts.select([])

    def test_select_nonexistent_column(self):
        """Test selecting a column that does not exist raises error."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(KeyError):
            ts.select(["nonexistent_column"])

    def test_select_existing_and_nonexistent_column(self):
        """Test selecting a column that does not exist, alongside existing columns, still raises error"""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(KeyError):
            ts.select(["col1", "col2", "nonexistent_column"])

    def test_select_column_doesnt_mutate_original_ts(self):
        """When selecting a column, the original ts should be unchanged"""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        original_df = ts.df

        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time",
                              column_metadata={"col1": self.metadata["col1"]})
        col1_ts = ts.select(["col1"])
        self.assertEqual(col1_ts, expected)
        assert_frame_equal(ts.df, original_df)

        expected = TimeSeries(self.df.select(["time", "col2"]), time_name="time",
                              column_metadata={"col2": self.metadata["col2"]})
        col2_ts = ts.select(["col2"])
        self.assertEqual(col2_ts, expected)
        assert_frame_equal(ts.df, original_df)


class TestGetattr(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })
    column_metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}

    def test_access_time_column(self):
        """Test accessing the time column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.column_metadata)
        result = ts.time
        expected = PrimaryTimeColumn("time", ts)
        self.assertEqual(result, expected)

    def test_access_data_column(self):
        """Test accessing a data column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.column_metadata)
        result = ts.col1
        expected = DataColumn("col1", ts, self.column_metadata["col1"])
        self.assertEqual(result, expected)

    @parameterized.expand([
        ("int", "site_id", 1234),
        ("str", "network", "FDRI"),
        ("dict", "some_info", {1: "a", 2: "b", 3: "c"}),
    ])
    def test_access_metadata_key(self, _, key, expected):
        """Test accessing metadata key."""
        ts = TimeSeries(self.df, time_name="time", metadata=self.metadata)
        result = ts.__getattr__(key)
        self.assertEqual(result, expected)

    def test_access_nonexistent_attribute(self):
        """Test accessing metadata key that doesn't exist"""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.column_metadata)
        with self.assertRaises(AttributeError):
            _ = ts.col1.key0


class TestGetItem(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_access_time_column(self):
        """Test accessing the time column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        result = ts["time"]
        expected = PrimaryTimeColumn("time", ts)
        self.assertEqual(result, expected)

    def test_access_data_column(self):
        """Test accessing a data column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        result = ts["col1"]
        expected = TimeSeries(self.df.select(["time", "col1"]),
                              time_name="time", column_metadata={"col1": self.metadata["col1"]})
        self.assertEqual(result, expected)

    def test_access_multiple_data_columns(self):
        """Test accessing multiple data columns."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        result = ts[["col1", "col2"]]
        expected = TimeSeries(self.df.select(["time", "col1", "col2"]),
                              time_name="time",
                              column_metadata={"col1": self.metadata["col1"], "col2": self.metadata["col2"]})
        self.assertEqual(result, expected)

    def test_non_existent_column(self):
        """Test accessing non-existent data column raises error."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(KeyError):
            _ = ts["col0"]


class TestSetupColumns(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "data_col": [10, 20, 30], "supp_col": ["A", "B", "C"], "flag_col": [1, 2, 3]
    })
    flag_systems = {"example_flag_system": {"OK": 1, "WARNING": 2}}

    def test_valid_setup_columns(self):
        """Test that valid columns are correctly classified."""
        ts = TimeSeries(df=self.df,
                        time_name="time",
                        supplementary_columns=["supp_col"],
                        flag_columns={"flag_col": "example_flag_system"},
                        flag_systems=self.flag_systems)

        self.assertIsInstance(ts.time_column, PrimaryTimeColumn)
        self.assertIsInstance(ts.columns["supp_col"], SupplementaryColumn)
        self.assertIsInstance(ts.columns["flag_col"], FlagColumn)
        self.assertIsInstance(ts.columns["data_col"], DataColumn)

    def test_missing_supplementary_column_raises_error(self):
        """Test that an error raised when supplementary columns do not exist."""
        with self.assertRaises(KeyError):
            TimeSeries(df=self.df,
                       time_name="time",
                       supplementary_columns=["missing_col"],
                       flag_columns={"flag_col": "example_flag_system"},
                       flag_systems=self.flag_systems)

    def test_missing_flag_column_raises_error(self):
        """Test that an error raised when flag columns do not exist."""
        with self.assertRaises(KeyError):
            TimeSeries(df=self.df,
                       time_name="time",
                       supplementary_columns=["supp_col"],
                       flag_columns={"missing_col": "example_flag_system"},
                       flag_systems=self.flag_systems)


class TestTimeColumn(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })

    def test_valid_time_column(self):
        """Test that time_column correctly returns the PrimaryTimeColumn instance."""
        ts = TimeSeries(self.df, time_name="time")
        self.assertIsInstance(ts.time_column, PrimaryTimeColumn)
        self.assertEqual(ts.time_column.name, "time")

    def test_no_time_column_raises_error(self):
        """Test that error is raised if no primary time column is found."""
        ts = TimeSeries(self.df, time_name="time")
        with patch.object(ts, "_columns", {}):  # Simulate missing columns
            with self.assertRaises(ValueError):
                _ = ts.time_column

    def test_multiple_time_columns_raises_error(self):
        """Test that error is raised if multiple primary time columns exist."""
        ts = TimeSeries(self.df, time_name="time")
        with patch.object(TimeSeriesColumn, "_validate_name", lambda *args, **kwargs: None):
            with patch.object(ts, "_columns", {
                "time1": PrimaryTimeColumn("time1", ts),
                "time2": PrimaryTimeColumn("time2", ts),
            }):
                with self.assertRaises(ValueError):
                    _ = ts.time_column


class TestTimeSeriesEquality(unittest.TestCase):
    def setUp(self):
        """Set up multiple TimeSeries objects for testing."""
        self.df_original = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
            "value": [10, 20, 30]
        })
        self.df_same = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
            "value": [10, 20, 30]
        })
        self.df_different_values = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
            "value": [100, 200, 300]
        })
        self.df_different_times = pl.DataFrame({
            "time": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)],
            "value": [10, 20, 30]
        })

        self.flag_systems_1 = {"quality_flags": {"OK": 1, "WARNING": 2}}
        self.flag_systems_2 = {"quality_flags": {"NOT_OK": 4, "ERROR": 8}}

        self.ts_original = TimeSeries(
            df=self.df_original, time_name="time", resolution=Period.of_days(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_1
        )

    def test_equal_timeseries(self):
        """Test that two identical TimeSeries objects are considered equal."""
        ts_same = TimeSeries(
            df=self.df_same, time_name="time", resolution=Period.of_days(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_1
        )
        self.assertEqual(self.ts_original, ts_same)

    def test_different_data_values(self):
        """Test that TimeSeries objects with different data values are not equal."""
        ts_different_df = TimeSeries(
            df=self.df_different_values, time_name="time", resolution=Period.of_days(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_1
        )
        self.assertNotEqual(self.ts_original, ts_different_df)

    def test_different_time_values(self):
        """Test that TimeSeries objects with different time values are not equal."""
        ts_different_times = TimeSeries(
            df=self.df_different_times, time_name="time", resolution=Period.of_days(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_1
        )
        self.assertNotEqual(self.ts_original, ts_different_times)

    def test_different_time_name(self):
        """Test that TimeSeries objects with different time column name are not equal."""
        ts_different_time_name = TimeSeries(
            df=self.df_original.rename({"time": "timestamp"}), time_name="timestamp", resolution=Period.of_days(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_1
        )
        self.assertNotEqual(self.ts_original, ts_different_time_name)

    def test_different_periodicity(self):
        """Test that TimeSeries objects with different periodicity are not equal."""
        ts_different_periodicity = TimeSeries(
            df=self.df_original, time_name="time", resolution=Period.of_days(1),
            periodicity=Period.of_months(1), flag_systems=self.flag_systems_1
        )
        self.assertNotEqual(self.ts_original, ts_different_periodicity)

    def test_different_resolution(self):
        """Test that TimeSeries objects with different resolution are not equal."""
        ts_different_resolution = TimeSeries(
            df=self.df_original, time_name="time", resolution=Period.of_seconds(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_1
        )
        self.assertNotEqual(self.ts_original, ts_different_resolution)

    def test_different_flag_systems(self):
        """Test that TimeSeries objects with different flag systems are not equal."""
        ts_different_flags = TimeSeries(
            df=self.df_original, time_name="time", resolution=Period.of_days(1),
            periodicity=Period.of_days(1), flag_systems=self.flag_systems_2
        )
        self.assertNotEqual(self.ts_original, ts_different_flags)


class TestColumnMetadata(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }
    ts = TimeSeries(df, time_name="time", column_metadata=metadata)

    def test_retrieve_all_metadata(self):
        """Test retrieving all metadata for all columns."""
        self.assertEqual(self.ts.column_metadata(), self.metadata)

    def test_retrieve_metadata_for_single_column(self):
        """Test retrieving metadata for a single column."""
        expected = {"col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"}}
        result = self.ts.column_metadata("col1")
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_columns(self):
        """Test retrieving metadata for multiple columns."""
        expected = {"col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"},
                    "col2": {"key1": "2", "key2": "20", "key3": "200"}}
        result = self.ts.column_metadata(["col1", "col2"])
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_specific_key(self):
        """Test retrieving a specific metadata key."""
        expected = {"col1": {"key1": "1"}}
        result = self.ts.column_metadata("col1", "key1")
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self):
        """Test retrieving multiple metadata keys."""
        expected = {"col1": {"key1": "1", "key3": "100"}}
        result = self.ts.column_metadata("col1", ["key1", "key3"])
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_column(self):
        """Test that a KeyError is raised when requesting a non-existent column."""
        with self.assertRaises(KeyError):
            self.ts.column_metadata("nonexistent_column")

    def test_retrieve_metadata_for_nonexistent_key_single_column(self):
        """Test that error raised when requesting a non-existent metadata key for an existing single column"""
        with self.assertRaises(KeyError):
            self.ts.column_metadata("col1", "nonexistent_key")

    def test_retrieve_metadata_for_nonexistent_key_in_one_column(self):
        """Test that dict returned when requesting a metadata key exists in one column, but not another"""
        expected = {"col1": {"key4": "1000"}, "col2": {"key4": None}}
        result = self.ts.column_metadata(["col1", "col2"], "key4")
        self.assertEqual(result, expected)


class TestMetadata(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })
    metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}
    ts = TimeSeries(df, time_name="time", metadata=metadata)

    def test_retrieve_all_metadata(self):
        """Test retrieving all metadata"""
        result = self.ts.metadata()
        self.assertEqual(result, self.metadata)

    @parameterized.expand([
        ("int", "site_id"),
        ("str", "network"),
        ("dict", "some_info"),
    ])
    def test_retrieve_metadata_for_specific_key(self, _, key):
        """Test retrieving a specific metadata key."""
        result = self.ts.metadata(key)
        expected = {key: self.metadata[key]}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self):
        """Test retrieving multiple metadata keys."""
        result = self.ts.metadata(["site_id", "network"])
        expected = {"site_id": 1234, "network": "FDRI"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_key_raises_error(self):
        """Test that an error is raised for a non-existent key."""
        with self.assertRaises(KeyError):
            self.ts.metadata("nonexistent_key")

    def test_retrieve_metadata_for_nonexistent_key_strict_false(self):
        """Test that an empty result is returned when strict is false for non-existent key."""
        expected = {'nonexistent_key': None}
        result = self.ts.metadata("nonexistent_key", strict=False)
        self.assertEqual(result, expected)


class TestSetupMetadata(unittest.TestCase):
    df = pl.DataFrame({
        "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
        "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]
    })

    def test_setup_metadata_success(self):
        """Test that metadata entries with keys not in _columns are added successfully."""
        metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(self.df, time_name="time")
            ts._setup_metadata(metadata)
        self.assertEqual(ts._metadata, metadata)

    def test_setup_metadata_conflict(self):
        """Test that providing a metadata key that conflicts with _columns raises a KeyError."""
        metadata = {"col1": 1234}
        with patch.object(TimeSeries, '_setup'):
            with self.assertRaises(KeyError):
                ts = TimeSeries(self.df, time_name="time")
                ts._setup_metadata(metadata)

    def test_setup_metadata_empty(self):
        """Test that passing an empty metadata dictionary leaves _metadata unchanged."""
        metadata = {}
        with patch.object(TimeSeries, '_setup'):
            ts = TimeSeries(self.df, time_name="time")
            ts._setup_metadata(metadata)
        self.assertEqual(ts._metadata, {})


class TestGetFlagSystemColumn(unittest.TestCase):
    def setUp(self):
        df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "data_col": [10, 20, 30], "supp_col": ["A", "B", "C"],
            "flag_col": [1, 2, 3]
        })
        flag_systems = {"example_flag_system": {"OK": 1, "WARNING": 2}}
        self.ts = TimeSeries(df=df,
                             time_name="time",
                             supplementary_columns=["supp_col"],
                             flag_columns={"flag_col": "example_flag_system"},
                             flag_systems=flag_systems)
        self.ts.data_col.add_relationship(["flag_col"])

    def test_data_column_not_exist(self):
        """Test return when the specified data column doesn't exist"""
        data_column = "data_col_not_exist"
        flag_system = "example_flag_system"
        with self.assertRaises(KeyError):
            self.ts.get_flag_system_column(data_column, flag_system)

    @parameterized.expand([
        "supp_col", "flag_col"
    ])
    def test_data_column_not_a_data_column(self, data_column):
        """Test return when the specified data column isn't actually a data column"""
        flag_system = "example_flag_system"
        with self.assertRaises(TypeError):
            self.ts.get_flag_system_column(data_column, flag_system)

    def test_get_expected_flag_column(self):
        """Test expected flag column returned for valid flag system"""
        data_column = "data_col"
        flag_system = "example_flag_system"
        expected = self.ts.flag_col
        result = self.ts.get_flag_system_column(data_column, flag_system)
        self.assertEqual(result, expected)


class TestSubPeriodCheck(unittest.TestCase):
    """Test the "resolution is a subperiod of periodicity" check"""
    df = pl.DataFrame({ "time": [datetime(2020, 1, 1)] })

    def test_legal(self):
        # The test is that the following does not raise an error
        TimeSeries(df=self.df, time_name="time",
            resolution=Period.of_seconds(1),
            periodicity=Period.of_seconds(10))

    def test_illegal(self):
        with self.assertRaises(UserWarning):
            TimeSeries(df=self.df, time_name="time",
                resolution=Period.of_seconds(10),
                periodicity=Period.of_seconds(1))


class TestHandleDuplicates(unittest.TestCase):
    def setUp(self):
        # A dataframe with some duplicate times
        self.df = pl.DataFrame({
            "time": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 7),
                datetime(2024, 1, 8),
                datetime(2024, 1, 8),
                datetime(2024, 1, 8),
                datetime(2024, 1, 9),
                datetime(2024, 1, 10)
            ],
            "colA": [1, None, 2, 3, 4, 5, 6, 7, None, 8, 8, 9, 10],
            "colB": [None, 1, 2, 3, 4, 5, 6, 7, 9, None, 9, 9, 10],
            "colC": [None, 2, 2, 3, 4, 5, 6, 7, 8, 9, None, 9, 10],
        })

    def test_error(self):
        """ Test that the error strategy works as expected
        """
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=self.df, time_name="time", on_duplicates="error")

        with self.assertRaises(ValueError):
            ts._handle_duplicates()

    def test_keep_first(self):
        """ Test that the keep first strategy works as expected
        """
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=self.df, time_name="time", on_duplicates="keep_first")

        ts._handle_duplicates()

        expected = pl.DataFrame({
            "time": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 7),
                datetime(2024, 1, 8),
                datetime(2024, 1, 9),
                datetime(2024, 1, 10)
            ],
            "colA": [1, 2, 3, 4, 5, 6, 7, None, 9, 10],
            "colB": [None, 2, 3, 4, 5, 6, 7, 9, 9, 10],
            "colC": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        assert_frame_equal(ts.df, expected)

    def test_keep_last(self):
        """ Test that the keep last strategy works as expected
        """
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=self.df, time_name="time", on_duplicates="keep_last")

        ts._handle_duplicates()

        expected = pl.DataFrame({
            "time": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 7),
                datetime(2024, 1, 8),
                datetime(2024, 1, 9),
                datetime(2024, 1, 10)
            ],
            "colA": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "colB": [1, 2, 3, 4, 5, 6, 7, 9, 9, 10],
            "colC": [2, 2, 3, 4, 5, 6, 7, None, 9, 10],
        })

        assert_frame_equal(ts.df, expected)

    def test_drop(self):
        """ Test that the drop strategy works as expected
        """
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=self.df, time_name="time", on_duplicates="drop")

        ts._handle_duplicates()

        expected = pl.DataFrame({
            "time": [
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 7),
                datetime(2024, 1, 9),
                datetime(2024, 1, 10)
            ],
            "colA": [2, 3, 4, 5, 6, 7, 9, 10],
            "colB": [2, 3, 4, 5, 6, 7, 9, 10],
            "colC": [2, 3, 4, 5, 6, 7, 9, 10],
        })

        assert_frame_equal(ts.df, expected)

    def test_merge(self):
        """ Test that the merge strategy works as expected
        """
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=self.df, time_name="time", on_duplicates="merge")
        ts._setup_columns()  # Need to run this setup for this test as we need the column details to be available
        ts._handle_duplicates()

        expected = pl.DataFrame({
            "time": [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
                datetime(2024, 1, 7),
                datetime(2024, 1, 8),
                datetime(2024, 1, 9),
                datetime(2024, 1, 10)
            ],
            "colA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "colB": [1, 2, 3, 4, 5, 6, 7, 9, 9, 10],
            "colC": [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        assert_frame_equal(ts.df, expected)

    def test_enum_parameter(self):
        """ Test that passing an enum rather than a string works as expected
        """
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=self.df, time_name="time", on_duplicates=DuplicateOption.ERROR)

        with self.assertRaises(ValueError):
            ts._handle_duplicates()


class TestPadTime(unittest.TestCase):
    simple_test_cases = (
        (
            "microsecond_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 0, 1), interval="20us", eager=True),
            Period.of_microseconds(20),
            Period.of_microseconds(20),
        ),
        (
            "second_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 1, 2), interval="1s", eager=True),
            Period.of_seconds(1),
            Period.of_seconds(1),
        ),
        (
            "minute_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 2, 1), interval="1m", eager=True),
            Period.of_minutes(1),
            Period.of_minutes(1),
        ),
        (
            "daily_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2030, 1, 1), interval="1d", eager=True),
            Period.of_days(1),
            Period.of_days(1),
        ),
        (
            "monthly_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2035, 1, 1), interval="1mo", eager=True),
            Period.of_months(1),
            Period.of_months(1),
        ),
        (
            "yearly_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2125, 1, 1), interval="1y", eager=True),
            Period.of_years(1),
            Period.of_years(1),
        ),
    )

    @parameterized.expand(simple_test_cases)
    def test_simple_cases(self, _, time_stamps, resolution, periodicity):
        """ Test that the padding of missing time values works for simple periodicity and resolution cases
        """
        df = pl.DataFrame({"time": time_stamps})
        # Remove some rows - but not the first or last!        
        df_to_pad = pl.concat([
            df.head(1),
            df.slice(1, df.height - 2).sample(len(df) - 10, with_replacement=False),
            df.tail(1)
        ])

        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=df_to_pad, time_name="time", resolution=resolution, periodicity=periodicity, pad=True)

        ts._pad_time()
        pl.testing.assert_frame_equal(ts.df, df)

    @parameterized.expand(simple_test_cases)
    def test_simple_cases_no_pad(self, _, time_stamps, resolution, periodicity):
        """ Test that the data with missing time values is maintained if pad = False
        """
        df = pl.DataFrame({"time": time_stamps})
        # Remove some rows - but not the first or last!
        df_to_pad = pl.concat([
            df.head(1),
            df.slice(1, df.height - 2).sample(len(df) - 10, with_replacement=False),
            df.tail(1)
        ])

        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=df_to_pad, time_name="time", resolution=resolution, periodicity=periodicity, pad=False)

        ts._pad_time()
        pl.testing.assert_frame_equal(ts.df, df_to_pad)

    @parameterized.expand(simple_test_cases)
    def test_no_missing_simple_cases(self, _, time_stamps, resolution, periodicity):
        """ Test that the padding of missing time values works for simple periodicity and resolution cases
        when there are no missing rows (shouldn't alter the time series)
        """
        df = pl.DataFrame({"time": time_stamps})

        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=df, time_name="time", resolution=resolution, periodicity=periodicity, pad=True)

        ts._pad_time()
        pl.testing.assert_frame_equal(ts.df, df)

    @parameterized.expand(simple_test_cases)
    def test_no_missing_simple_cases_no_pad(self, _, time_stamps, resolution, periodicity):
        """ Test that data is maintained when no missing rows and pad = False
        """
        df = pl.DataFrame({"time": time_stamps})

        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=df, time_name="time", resolution=resolution, periodicity=periodicity, pad=False)

        ts._pad_time()
        pl.testing.assert_frame_equal(ts.df, df)

    complex_test_cases = (
        (
            "annual_periodicity_15minute_resolution",
            [
                datetime(2020, 5, 1, 17, 15, 0),
                datetime(2022, 1, 30, 2, 45, 0),
                datetime(2024, 9, 1, 10, 30, 0),
                datetime(2025, 12, 4, 14, 20, 0),
            ],
            [
                datetime(2020, 5, 1, 17, 15, 0),
                datetime(2021, 1, 1, 0, 0, 0),
                datetime(2022, 1, 30, 2, 45, 0),
                datetime(2023, 1, 1, 0, 0, 0),
                datetime(2024, 9, 1, 10, 30, 0),
                datetime(2025, 12, 4, 14, 20, 0),
            ],
            Period.of_minutes(15),
            Period.of_years(1),
        ),
        (
            "annual_periodicity_15minute_resolution_with_simple_offset",
            [
                datetime(2020, 5, 1, 17, 15, 30),
                datetime(2022, 1, 30, 2, 45, 30),
                datetime(2024, 9, 1, 10, 30, 30),
                datetime(2025, 12, 4, 14, 20, 30),
            ],
            [
                datetime(2020, 5, 1, 17, 15, 30),
                datetime(2021, 1, 1, 0, 0, 30),
                datetime(2022, 1, 30, 2, 45, 30),
                datetime(2023, 1, 1, 0, 0, 30),
                datetime(2024, 9, 1, 10, 30, 30),
                datetime(2025, 12, 4, 14, 20, 30),
            ],
            Period.of_minutes(15).with_second_offset(30),
            Period.of_years(1).with_second_offset(30),
        ),

        (
            "water_year_periodicity_15minute_resolution",
            [
                datetime(2020, 5, 1, 17, 15, 0),
                datetime(2022, 1, 30, 2, 45, 0),
                datetime(2024, 9, 1, 10, 30, 0),
                datetime(2025, 12, 4, 14, 20, 0),
            ],
            [
                datetime(2020, 5, 1, 17, 15, 0),
                datetime(2020, 10, 1, 9, 0, 0),
                datetime(2022, 1, 30, 2, 45, 0),
                datetime(2022, 10, 1, 9, 0, 0),
                datetime(2024, 9, 1, 10, 30, 0),
                datetime(2024, 10, 1, 9, 0, 0),
                datetime(2025, 12, 4, 14, 20, 0),
            ],
            Period.of_minutes(15),
            Period.of_years(1).with_month_offset(9).with_hour_offset(9),
        ),
    )

    @parameterized.expand(complex_test_cases)
    def test_complex_cases(self, _, time_stamps, expected, resolution, periodicity):
        """ Test that the padding of missing time values works for more complex periodicity and resolution cases
        """
        df = pl.DataFrame({"time": time_stamps})
        df_expected = pl.DataFrame({"time": expected})
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=df, time_name="time", resolution=resolution, periodicity=periodicity, pad=True)

        ts._pad_time()
        pl.testing.assert_frame_equal(ts.df, df_expected)

    @parameterized.expand(complex_test_cases)
    def test_no_missing_complex_cases(self, _, __, expected, resolution, periodicity):
        """ Test that the padding of missing time values works for simple periodicity and resolution cases
        when there are no missing rows (shouldn't alter the time series)
        """
        df = pl.DataFrame({"time": expected})
        with patch.object(TimeSeries, '_setup'):
            # Omit the setup method so we have control over what we're testing
            ts = TimeSeries(df=df, time_name="time", resolution=resolution, periodicity=periodicity, pad=True)

        ts._pad_time()
        pl.testing.assert_frame_equal(ts.df, df)

