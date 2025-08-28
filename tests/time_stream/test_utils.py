import unittest
from datetime import datetime

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream import Period
from time_stream.exceptions import ColumnNotFoundError
from time_stream.utils import check_columns_in_dataframe, get_date_filter, truncate_to_period, pad_time


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
        result = truncate_to_period(self.df["time"],period)
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
        result = truncate_to_period(self.df["time"],period)
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
        result = truncate_to_period(self.df["time"],period)
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
        result = truncate_to_period(self.df["time"],period)
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
        result = truncate_to_period(self.df["time"],period)
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
        result = truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple microseconds", Period.of_microseconds(200),
         [datetime(2020, 6, 1, 8, 10, 5, 2000), datetime(2021, 4, 30, 15, 30, 10, 200), datetime(2022, 12, 31, 12)]
         ),
    ])
    def test_truncate_to_microsecond_period(self, _, period, expected):
        """ Test that truncating a time series to a given microsecond period works as expected.
        """
        result = truncate_to_period(self.df["time"],period)
        assert_series_equal(result, pl.Series("time", expected))


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
