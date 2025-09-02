import unittest
from datetime import datetime

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream import Period
from time_stream.enums import TimeAnchor
from time_stream.exceptions import ColumnNotFoundError
from time_stream.utils import (
    check_columns_in_dataframe,
    get_date_filter,
    truncate_to_period,
    pad_time,
    check_periodicity,
    check_resolution
)


class TestCheckColumnsInDataframe(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

    def test_all_columns_exist(self):
        """Test error not raised when all requested columns exist."""
        check_columns_in_dataframe(self.df, ["a", "b", "c"])

    def test_single_missing_column(self):
        """Should raise when one column is missing."""
        with self.assertRaises(ColumnNotFoundError) as err:
            check_columns_in_dataframe(self.df, ["a", "x"])
        self.assertEqual("Columns not found in dataframe: ['x']", str(err.exception))

    def test_multiple_missing_columns(self):
        """Should raise when multiple columns are missing."""
        with self.assertRaises(ColumnNotFoundError) as err:
            check_columns_in_dataframe(self.df, ["a", "x", "y"])
        self.assertEqual("Columns not found in dataframe: ['x', 'y']", str(err.exception))

    def test_empty_columns_list(self):
        """Test that empty list passes without error."""
        check_columns_in_dataframe(self.df, [])


class TestGetDateFilter(unittest.TestCase):
    df = pl.DataFrame({"timestamp": [datetime(2025, m, 1) for m in range(1, 8)]})

    @parameterized.expand([
        ("some_within", (datetime(2025, 1, 1), datetime(2025, 4, 1)), [True, True, True, True, False, False, False]),
        ("all_within", (datetime(2024, 1, 1), datetime(2026, 1, 1)), [True, True, True, True, True, True, True]),
        ("all_out", (datetime(2026, 1, 1), datetime(2027, 1, 1)), [False, False, False, False, False, False, False]),
        ("start_only", datetime(2025, 3, 1), [False, False, True, True, True, True, True]),
        ("start_only_with_none", (datetime(2025, 3, 1), None), [False, False, True, True, True, True, True]),
    ])
    def test_get_date_filter(self, _, observation_interval, expected_eval):
        date_filter = get_date_filter("timestamp", observation_interval)
        result = self.df.select(date_filter).to_series()
        expected = pl.Series("timestamp", expected_eval)
        assert_series_equal(result, expected)


class TestTruncateToPeriod(unittest.TestCase):
    dt = pl.Series([
            datetime(2020, 6, 1, 8, 10, 5, 2001),
            datetime(2021, 4, 30, 15, 30, 10, 250),
            datetime(2022, 12, 31, 12, 0, 0),
            datetime(2023, 1, 1, 0, 0),
            datetime(2023, 3, 1, 0, 15)
    ])

    @parameterized.expand([
        ("simple yearly", Period.of_years(1), TimeAnchor.START,
         pl.Series([
             datetime(2020, 1, 1),
             datetime(2021, 1, 1),
             datetime(2022, 1, 1),
             datetime(2023, 1, 1),
             datetime(2023, 1, 1)
         ])
         ),
        ("yearly with offset", Period.of_years(1).with_month_offset(2), TimeAnchor.START,
         pl.Series([
             datetime(2020, 3, 1),
             datetime(2021, 3, 1),
             datetime(2022, 3, 1),
             datetime(2022, 3, 1),
             datetime(2023, 3, 1)
         ])
         ),
        ("simple yearly end anchor", Period.of_years(1), TimeAnchor.END,
         pl.Series([
             datetime(2021, 1, 1),
             datetime(2022, 1, 1),
             datetime(2023, 1, 1),
             datetime(2023, 1, 1),
             datetime(2024, 1, 1)
         ])
         ),
        ("yearly with offset end anchor", Period.of_years(1).with_month_offset(2), TimeAnchor.END,
         pl.Series([
             datetime(2021, 3, 1),
             datetime(2022, 3, 1),
             datetime(2023, 3, 1),
             datetime(2023, 3, 1),
             datetime(2024, 3, 1)
         ])
         )
    ])
    def test_truncate_to_year_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given year period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, expected)

    @parameterized.expand([
        ("simple monthly", Period.of_months(1), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1),
             datetime(2021, 4, 1),
             datetime(2022, 12, 1),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1)
         ])
         ),
        ("3 monthly (quarterly)", Period.of_months(3), TimeAnchor.START,
         pl.Series([
             datetime(2020, 4, 1),
             datetime(2021, 4, 1),
             datetime(2022, 10, 1),
             datetime(2023, 1, 1),
             datetime(2023, 1, 1)
         ])
         ),
        ("monthly with offset", Period.of_months(1).with_hour_offset(9), TimeAnchor.START,
         pl.Series([
             datetime(2020, 5, 1, 9),
             datetime(2021, 4, 1, 9),
             datetime(2022, 12, 1, 9),
             datetime(2022, 12, 1, 9),
             datetime(2023, 2, 1, 9)
         ])
         ),
        ("simple monthly end anchor", Period.of_months(1), TimeAnchor.END,
         pl.Series([
             datetime(2020, 7, 1),
             datetime(2021, 5, 1),
             datetime(2023, 1, 1),
             datetime(2023, 1, 1),
             datetime(2023, 4, 1)
         ])
         ),
        ("3 monthly (quarterly) end anchor", Period.of_months(3), TimeAnchor.END,
         pl.Series([
             datetime(2020, 7, 1),
             datetime(2021, 7, 1),
             datetime(2023, 1, 1),
             datetime(2023, 1, 1),
             datetime(2023, 4, 1)
         ])
         ),
        ("monthly with offset end anchor", Period.of_months(1).with_hour_offset(9), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 9),
             datetime(2021, 5, 1, 9),
             datetime(2023, 1, 1, 9),
             datetime(2023, 1, 1, 9),
             datetime(2023, 3, 1, 9)
         ])
         )
    ])
    def test_truncate_to_month_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given month period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand([
        ("simple daily", Period.of_days(1), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1),
             datetime(2021, 4, 30),
             datetime(2022, 12, 31),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1)
         ])
         ),
        ("daily with offset", Period.of_days(1).with_hour_offset(9), TimeAnchor.START,
         pl.Series([
             datetime(2020, 5, 31, 9),
             datetime(2021, 4, 30, 9),
             datetime(2022, 12, 31, 9),
             datetime(2022, 12, 31, 9),
             datetime(2023, 2, 28, 9)
         ])
         ),
        ("simple daily end anchor", Period.of_days(1), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 2),
             datetime(2021, 5, 1),
             datetime(2023, 1, 1),
             datetime(2023, 1, 1),
             datetime(2023, 3, 2)
         ])
         ),
        ("daily with offset end anchor", Period.of_days(1).with_hour_offset(9), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 9),
             datetime(2021, 5, 1, 9),
             datetime(2023, 1, 1, 9),
             datetime(2023, 1, 1, 9),
             datetime(2023, 3, 1, 9)
         ])
         )
    ])
    def test_truncate_to_day_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given day period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand([
        ("simple hourly", Period.of_hours(1), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8),
             datetime(2021, 4, 30, 15),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1)
         ])
         ),
        ("12 hourly", Period.of_hours(12), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1),
             datetime(2021, 4, 30, 12),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1)
         ])
         ),
        ("hourly with offset", Period.of_hours(1).with_minute_offset(30), TimeAnchor.START,
        pl.Series([
             datetime(2020, 6, 1, 7, 30),
             datetime(2021, 4, 30, 15, 30),
             datetime(2022, 12, 31, 11, 30),
             datetime(2022, 12, 31, 23, 30),
             datetime(2023, 2, 28, 23, 30)
         ])
         ),
        ("simple hourly end anchor", Period.of_hours(1), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 9),
             datetime(2021, 4, 30, 16),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 1)
         ])
         ),
        ("12 hourly end anchor", Period.of_hours(12), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 12),
             datetime(2021, 5, 1),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 12)
         ])
         ),
        ("hourly with offset end anchor", Period.of_hours(1).with_minute_offset(30), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 30),
             datetime(2021, 4, 30, 16, 30),
             datetime(2022, 12, 31, 12, 30),
             datetime(2023, 1, 1, 0, 30),
             datetime(2023, 3, 1, 0, 30)
         ])
         )
    ])
    def test_truncate_to_hour_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given hour period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand([
        ("simple minutes", Period.of_minutes(1), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8, 10),
             datetime(2021, 4, 30, 15, 30),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("15 minutes", Period.of_minutes(15), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8),
             datetime(2021, 4, 30, 15, 30),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("minutes with offset", Period.of_minutes(1).with_second_offset(45), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8, 9, 45),
             datetime(2021, 4, 30, 15, 29, 45),
             datetime(2022, 12, 31, 11, 59, 45),
             datetime(2022, 12, 31, 23, 59, 45),
             datetime(2023, 3, 1, 0, 14, 45)
         ])
         ),
        ("simple minutes end anchor", Period.of_minutes(1), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 11),
             datetime(2021, 4, 30, 15, 31),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("15 minutes end anchor", Period.of_minutes(15), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 15),
             datetime(2021, 4, 30, 15, 45),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("minutes with offset end anchor", Period.of_minutes(1).with_second_offset(45), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 45),
             datetime(2021, 4, 30, 15, 30, 45),
             datetime(2022, 12, 31, 12, 0, 45),
             datetime(2023, 1, 1, 0, 0, 45),
             datetime(2023, 3, 1, 0, 15, 45)
         ])
         )
    ])
    def test_truncate_to_minute_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given minute period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand([
        ("simple seconds", Period.of_seconds(1), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 5),
             datetime(2021, 4, 30, 15, 30, 10),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("3 seconds", Period.of_seconds(3), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 3),
             datetime(2021, 4, 30, 15, 30, 9),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("seconds with offset", Period.of_seconds(1).with_microsecond_offset(5000), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 4, 5000),
             datetime(2021, 4, 30, 15, 30, 9, 5000),
             datetime(2022, 12, 31, 11, 59, 59, 5000),
             datetime(2022, 12, 31, 23, 59, 59, 5000),
             datetime(2023, 3, 1, 0, 14, 59, 5000)
         ])
         ),
        ("simple seconds end anchor", Period.of_seconds(1), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 6),
             datetime(2021, 4, 30, 15, 30, 11),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("3 seconds end anchor", Period.of_seconds(3), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 6),
             datetime(2021, 4, 30, 15, 30, 12),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
         ("seconds with offset end anchor", Period.of_seconds(1).with_microsecond_offset(5000), TimeAnchor.END,
          pl.Series([
              datetime(2020, 6, 1, 8, 10, 5, 5000),
              datetime(2021, 4, 30, 15, 30, 10, 5000),
              datetime(2022, 12, 31, 12, 0, 0, 5000),
              datetime(2023, 1, 1, 0, 0, 0, 5000),
              datetime(2023, 3, 1, 0, 15, 0, 5000)
          ])
          ),
    ])
    def test_truncate_to_second_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given second period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, pl.Series(expected))

    @parameterized.expand([
        ("simple microseconds", Period.of_microseconds(250), TimeAnchor.START,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 5, 2000),
             datetime(2021, 4, 30, 15, 30, 10, 250),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
        ("simple microseconds end anchor", Period.of_microseconds(250), TimeAnchor.END,
         pl.Series([
             datetime(2020, 6, 1, 8, 10, 5, 2250),
             datetime(2021, 4, 30, 15, 30, 10, 250),
             datetime(2022, 12, 31, 12),
             datetime(2023, 1, 1),
             datetime(2023, 3, 1, 0, 15)
         ])
         ),
    ])
    def test_truncate_to_microsecond_period(self, _, period, anchor, expected):
        """ Test that truncating a time series to a given microsecond period works as expected.
        """
        result = truncate_to_period(self.dt, period, anchor)
        assert_series_equal(result, pl.Series(expected))


class TestPadTime(unittest.TestCase):
    simple_test_cases = (
        (
            "microsecond_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 1, 1, 0, 0, 1), interval="20us", eager=True),
            Period.of_microseconds(20),
        ),
        (
            "second_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 1, 2), interval="1s", eager=True),
            Period.of_seconds(1),
        ),
        (
            "minute_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2025, 2, 1), interval="1m", eager=True),
            Period.of_minutes(1),
        ),
        (
            "daily_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2030, 1, 1), interval="1d", eager=True),
            Period.of_days(1),
        ),
        (
            "monthly_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2035, 1, 1), interval="1mo", eager=True),
            Period.of_months(1),
        ),
        (
            "yearly_data",
            pl.datetime_range(datetime(2025, 1, 1), datetime(2125, 1, 1), interval="1y", eager=True),
            Period.of_years(1),
        ),
    )

    @parameterized.expand(simple_test_cases)
    def test_simple_cases(self, _, time_stamps, periodicity):
        """ Test that the padding of missing time values works for simple periodicity and resolution cases
        """
        df = pl.DataFrame({"time": time_stamps})
        # Remove some rows - but not the first or last!
        df_to_pad = pl.concat([
            df.head(1),
            df.slice(1, df.height - 2).sample(len(df) - 10, with_replacement=False),
            df.tail(1)
        ])

        result = pad_time(df_to_pad, "time", periodicity)
        pl.testing.assert_frame_equal(result, df)

    @parameterized.expand(simple_test_cases)
    def test_no_missing_simple_cases(self, _, time_stamps, periodicity):
        """ Test that the padding of missing time values works for simple periodicity and resolution cases
        when there are no missing rows (shouldn't alter the time series)
        """
        df = pl.DataFrame({"time": time_stamps})

        result = pad_time(df, "time", periodicity)
        pl.testing.assert_frame_equal(result, df)

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
            Period.of_years(1).with_month_offset(9).with_hour_offset(9),
        ),
    )

    @parameterized.expand(complex_test_cases)
    def test_complex_cases(self, _, time_stamps, expected, periodicity):
        """ Test that the padding of missing time values works for more complex periodicity cases
        """
        df = pl.DataFrame({"time": time_stamps})
        df_expected = pl.DataFrame({"time": expected})

        result = pad_time(df, "time", periodicity)
        pl.testing.assert_frame_equal(result, df_expected)

    @parameterized.expand(complex_test_cases)
    def test_no_missing_complex_cases(self, _, __, expected, periodicity):
        """ Test that the padding of missing time values works for complex periodicity cases
        when there are no missing rows (shouldn't alter the time series)
        """
        df = pl.DataFrame({"time": expected})
        result = pad_time(df, "time", periodicity)
        pl.testing.assert_frame_equal(result, df)


class TestCheckResolution(unittest.TestCase):

    def _check_success(self, times, resolution, time_anchor):
        """Test that a check_resolution call returns True"""
        self.assertTrue(check_resolution(pl.Series("time", times), resolution, time_anchor))

    def _check_failure(self, times, resolution, time_anchor):
        """Test that a .check_resolution call returns False"""
        self.assertFalse(check_resolution(pl.Series("time", times), resolution, time_anchor))

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1), TimeAnchor.START
         ),

        ("yearly with gaps",
         [datetime(1950, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1), TimeAnchor.START
         ),

        ("long term yearly",
         [datetime(500, 1, 1), datetime(1789, 1, 1), datetime(2099, 1, 1)],
         Period.of_years(1), TimeAnchor.START
         ),

        ("water years",
         [datetime(2006, 10, 1, 9), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9), TimeAnchor.START
         ),
    ])
    def test_check_year_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that a year based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple yearly error",
         [datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)],
         Period.of_years(1), TimeAnchor.START
         ),

        ("water years error",
         [datetime(2006, 10, 1, 10), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9), TimeAnchor.START
         ),
    ])
    def test_check_year_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that a year based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)],
         Period.of_months(1), TimeAnchor.START),

        ("monthly with gaps",
         [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2024, 1, 1)],
         Period.of_months(1), TimeAnchor.START),

        ("long term monthly",
         [datetime(1200, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)],
         Period.of_months(1), TimeAnchor.START),

        ("6 monthly",
         [datetime(2024, 1, 1), datetime(2024, 7, 1), datetime(2025, 1, 1)],
         Period.of_months(6), TimeAnchor.START),

        ("3 monthly (quarterly)",
         [datetime(2024, 1, 1), datetime(2024, 4, 1), datetime(2024, 7, 1), datetime(2024, 10, 1), datetime(2024, 1, 1)],
         Period.of_months(3), TimeAnchor.START),

        ("monthly with mid-month offset",
         [datetime(2024, 1, 15), datetime(2024, 3, 15), datetime(2024, 3, 15)],
         Period.of_months(1).with_day_offset(14), TimeAnchor.START),

        ("water months",
         [datetime(2024, 1, 1, 9), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_month_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that a month based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 1)],
         Period.of_months(1), TimeAnchor.START),

        ("water months error",
         [datetime(2024, 1, 1, 9, 20), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_month_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that a month based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_days(1), TimeAnchor.START),

        ("daily with gaps",
         [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2024, 1, 10)],
         Period.of_days(1), TimeAnchor.START),

        ("long term daily",
         [datetime(1800, 1, 1), datetime(2023, 1, 2), datetime(2024, 1, 10)],
         Period.of_days(1), TimeAnchor.START),

        ("water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9), TimeAnchor.START),

        ("daily across leap year feb",
         [datetime(2024, 2, 28), datetime(2024, 2, 29), datetime(2024, 3, 1)],
         Period.of_days(1), TimeAnchor.START),
    ])
    def test_check_day_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that a day based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple daily error",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3, 0, 0, 0, 1)],
         Period.of_days(1), TimeAnchor.START),

        ("water days error",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 19)],
         Period.of_days(1).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_day_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that a day based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         Period.of_hours(1), TimeAnchor.START),

        ("hourly with gaps",
         [datetime(2023, 1, 1, 6), datetime(2023, 6, 8, 19), datetime(2024, 3, 10, 4)],
         Period.of_hours(1), TimeAnchor.START),

        ("every 12 hours",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 12), datetime(2024, 1, 2), datetime(2024, 1, 2, 12)],
         Period.of_hours(12), TimeAnchor.START),

        ("every 12 hours starting at non-midnight",
         [datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 17), datetime(2024, 1, 2, 5), datetime(2024, 1, 2, 17)],
         Period.of_hours(12).with_hour_offset(5), TimeAnchor.START),

        ("long term hourly",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 18), datetime(2024, 1, 10, 23)],
         Period.of_hours(1), TimeAnchor.START),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 30)],
         Period.of_hours(1).with_minute_offset(30), TimeAnchor.START),

        ("hourly across leap year feb",
         [datetime(2024, 2, 29, 23), datetime(2024, 3, 1), datetime(2024, 3, 1, 1)],
         Period.of_hours(1), TimeAnchor.START),
    ])
    def test_check_hour_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that an hour based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2, 15), datetime(2024, 1, 1, 3)],
         Period.of_hours(1), TimeAnchor.START),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 31)],
         Period.of_hours(1).with_minute_offset(30), TimeAnchor.START),
    ])
    def test_check_hour_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that an hour based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         Period.of_minutes(1), TimeAnchor.START),

        ("minutes with gaps",
         [datetime(2023, 1, 1, 1, 1), datetime(2023, 12, 1, 19, 5), datetime(2024, 2, 25, 12, 52)],
         Period.of_minutes(1), TimeAnchor.START),

        ("every 15 minutes",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 1, 45), datetime(2024, 1, 1, 2)],
         Period.of_minutes(15), TimeAnchor.START),

        ("every 60 minutes",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2), datetime(2024, 1, 1, 3)],
         Period.of_minutes(15), TimeAnchor.START),

        ("long term minutes",
         [datetime(1800, 1, 1, 15, 1), datetime(2023, 1, 2, 18, 18), datetime(2024, 1, 10, 23, 55)],
         Period.of_minutes(1), TimeAnchor.START),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 30, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30), TimeAnchor.START),
    ])
    def test_check_minute_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that a minute based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3, 59)],
         Period.of_minutes(1), TimeAnchor.START),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 31, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30), TimeAnchor.START),
    ])
    def test_check_minute_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that a minute based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1), TimeAnchor.START),

        ("seconds with gaps",
         [datetime(1823, 1, 1, 1, 1, 1), datetime(2023, 7, 11, 12, 19, 59), datetime(2024, 1, 1, 1, 1, 13)],
         Period.of_seconds(1), TimeAnchor.START),

        ("every 5 seconds",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 5), datetime(2024, 1, 1, 1, 1, 10)],
         Period.of_seconds(5), TimeAnchor.START),

        ("every 86400 seconds",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_seconds(86400), TimeAnchor.START),

        ("long term seconds",
         [datetime(1800, 1, 1, 15, 1, 0), datetime(2023, 1, 2, 18, 18, 42), datetime(2024, 1, 10, 23, 55, 5)],
         Period.of_seconds(1), TimeAnchor.START),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 40), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40), TimeAnchor.START),
    ])
    def test_check_second_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that a second based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2, 9000), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1), TimeAnchor.START),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 41), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40), TimeAnchor.START),
    ])
    def test_check_second_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that a second based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)

    @parameterized.expand([
        ("simple microseconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001), datetime(2024, 1, 1, 1, 1, 1, 1002)],
         Period.of_microseconds(1), TimeAnchor.START),

        ("microseconds with gaps",
         [datetime(2023, 1, 1, 1, 1, 1, 5), datetime(2023, 7, 11, 12, 19, 59, 10), datetime(2024, 1, 1, 1, 1, 13, 9595)],
         Period.of_microseconds(5), TimeAnchor.START),

        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000), datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000), TimeAnchor.START),

        ("long term microseconds",
         [datetime(1800, 1, 1, 15, 1, 0, 0), datetime(2023, 1, 2, 18, 18, 42, 1), datetime(2024, 1, 10, 23, 55, 5, 10)],
         Period.of_microseconds(1), TimeAnchor.START),

        ("every 1 second",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2)],
         Period.of_microseconds(1_000_000), TimeAnchor.START),
    ])
    def test_check_microsecond_resolution_success(self, _, times, resolution, time_anchor):
        """ Test that a microsecond based time series that does conform to the given resolution passes the check.
        """
        self._check_success(times, resolution, time_anchor)

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000), datetime(2024, 1, 1, 1, 1, 1, 55_000)],
         Period.of_microseconds(25_000), TimeAnchor.START),
    ])
    def test_check_microsecond_resolution_failure(self, _, times, resolution, time_anchor):
        """ Test that a microsecond based time series that doesn't conform to the given resolution fails the check.
        """
        self._check_failure(times, resolution, time_anchor)


class TestCheckPeriodicity(unittest.TestCase):

    def _check_success(self, times, periodicity, time_anchor):
        """Test that a check_periodicity call returns True"""
        self.assertTrue(check_periodicity(pl.Series("time", times), periodicity, time_anchor))

    def _check_failure(self, times, periodicity, time_anchor):
        """Test that a check_periodicity call returns False"""
        self.assertFalse(check_periodicity(pl.Series("time", times), periodicity, time_anchor))

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
         Period.of_years(1), TimeAnchor.START),

        ("days within yearly",
         [datetime(2021, 1, 1), datetime(2022, 10, 5), datetime(2023, 2, 17)],
         Period.of_years(1), TimeAnchor.START),

        ("long term yearly",
         [datetime(500, 1, 1), datetime(1789, 1, 1), datetime(2099, 1, 1)],
         Period.of_years(1), TimeAnchor.START),

        ("simple water year",
         [datetime(2005, 1, 1), datetime(2006, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9), TimeAnchor.START),

        ("water year before/after Oct 1 9am",
         [datetime(2006, 10, 1, 8, 59), datetime(2006, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_year_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that a year based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2022, 1, 1)],
         Period.of_years(1), TimeAnchor.START),

        ("simple water year",
         [datetime(2005, 1, 1), datetime(2005, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_year_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that a year based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 31)],
         Period.of_months(1), TimeAnchor.START),

        ("long term monthly",
         [datetime(1200, 1, 1), datetime(2023, 1, 1), datetime(2024, 1, 1)],
         Period.of_months(1), TimeAnchor.START),

        ("6 monthly",
         [datetime(2024, 4, 16), datetime(2024, 7, 1), datetime(2025, 2, 25)],
         Period.of_months(6), TimeAnchor.START),

        ("3 monthly (quarterly)",
         [datetime(2024, 1, 1), datetime(2024, 4, 9), datetime(2024, 9, 2), datetime(2024, 12, 31)],
         Period.of_months(3), TimeAnchor.START),

        ("simple water months",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 5), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9), TimeAnchor.START),

        ("water months before/after 9am",
         [datetime(2024, 1, 1, 8, 59), datetime(2024, 1, 1, 9)],
         Period.of_months(1).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_month_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that a month based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 1, 31), datetime(2024, 2, 1)],
         Period.of_months(1), TimeAnchor.START),

        ("simple water months",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 1, 8, 59), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_month_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that a month based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
         Period.of_days(1), TimeAnchor.START),

        ("long term daily",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 19), datetime(2024, 1, 10, 1)],
         Period.of_days(1), TimeAnchor.START),

        ("simple water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9), TimeAnchor.START),

        ("water days before/after 9am",
         [datetime(2024, 1, 1, 8, 59), datetime(2024, 1, 1, 9)],
         Period.of_days(1).with_hour_offset(9), TimeAnchor.START),

        ("daily across leap year feb",
         [datetime(2024, 2, 28, 15), datetime(2024, 2, 29, 23, 59), datetime(2024, 3, 1)],
         Period.of_days(1), TimeAnchor.START),
    ])
    def test_check_day_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that a day based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 23, 59), datetime(2024, 1, 3)],
         Period.of_days(1), TimeAnchor.START),

        ("simple water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 8, 59, 59), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9), TimeAnchor.START),
    ])
    def test_check_day_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that a day based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 2, 59), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1), TimeAnchor.START),

        ("every 12 hours",
         [datetime(2024, 1, 1, 4, 30), datetime(2024, 1, 1, 12, 5), datetime(2024, 1, 2), datetime(2024, 1, 2, 23, 59)],
         Period.of_hours(12), TimeAnchor.START),

        ("long term hourly",
         [datetime(1800, 1, 1, 15), datetime(2023, 1, 2, 18), datetime(2024, 1, 10, 23)],
         Period.of_hours(1), TimeAnchor.START),

        ("hourly across leap year feb",
         [datetime(2024, 2, 29, 23, 59), datetime(2024, 3, 1), datetime(2024, 3, 1, 1)],
         Period.of_hours(1), TimeAnchor.START),
    ])
    def test_check_hour_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that an hour based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 16), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1), TimeAnchor.START),
    ])
    def test_check_hour_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that an hour based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3)],
         Period.of_minutes(1), TimeAnchor.START),

        ("every 15 minutes",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 35), datetime(2024, 1, 1, 1, 59)],
         Period.of_minutes(15), TimeAnchor.START),

        ("long term minutes",
         [datetime(1800, 1, 1, 15, 1), datetime(2023, 1, 2, 18, 18), datetime(2024, 1, 10, 23, 55)],
         Period.of_minutes(1), TimeAnchor.START),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 30, 29), datetime(2024, 1, 1, 1, 30, 31), datetime(2024, 1, 1, 1, 35, 30)],
         Period.of_minutes(5).with_second_offset(30), TimeAnchor.START),
    ])
    def test_check_minute_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that a minute based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 2, 1)],
         Period.of_minutes(1), TimeAnchor.START),
    ])
    def test_check_minute_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that a minute based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 1, 3, 40000)],
         Period.of_seconds(1), TimeAnchor.START),

        ("every 5 seconds",
         [datetime(2024, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 7), datetime(2024, 1, 1, 1, 1, 14)],
         Period.of_seconds(5), TimeAnchor.START),

        ("long term seconds",
         [datetime(1800, 1, 1, 15, 1, 0), datetime(2023, 1, 2, 18, 18, 42), datetime(2024, 1, 10, 23, 55, 5)],
         Period.of_seconds(1), TimeAnchor.START),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 39), datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 100000)],
         Period.of_seconds(30).with_microsecond_offset(40), TimeAnchor.START),
    ])
    def test_check_second_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that a second based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 1, 15001), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1), TimeAnchor.START),
    ])
    def test_check_second_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that a second based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)

    @parameterized.expand([
        ("simple microseconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1000), datetime(2024, 1, 1, 1, 1, 1, 1001),
          datetime(2024, 1, 1, 1, 1, 1, 1002)],
         Period.of_microseconds(1), TimeAnchor.START),

        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 12_521), datetime(2024, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000), TimeAnchor.START),

        ("long term microseconds",
         [datetime(1800, 1, 1, 15, 1, 0, 0), datetime(2023, 1, 2, 18, 18, 42, 1), datetime(2024, 1, 10, 23, 55, 5, 10)],
         Period.of_microseconds(1), TimeAnchor.START),
    ])
    def test_check_microsecond_periodicity_success(self, _, times, periodicity, time_anchor):
        """ Test that a microsecond based time series that does conform to the given periodicity passes the check.
        """
        self._check_success(times, periodicity, time_anchor)

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 25_001), datetime(2023, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000), TimeAnchor.START),
    ])
    def test_check_microsecond_periodicity_failure(self, _, times, periodicity, time_anchor):
        """ Test that a microsecond based time series that doesn't conform to the given periodicity fails the check.
        """
        self._check_failure(times, periodicity, time_anchor)
