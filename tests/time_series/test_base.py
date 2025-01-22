import unittest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timezone
from typing import Optional, Union

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal, assert_frame_equal

from time_series.base import TimeSeries
from time_series.period import Period
from time_series.columns import DataColumn, FlagColumn, PrimaryTimeColumn, SupplementaryColumn

TZ_UTC = timezone.utc


def init_timeseries(
    times: Optional[list[Union[date, datetime]]] = None,
    values: Optional[dict] = None,
    resolution: Optional[Period] = None,
    periodicity: Optional[Period] = None,
    time_zone: Optional[str] = None,
    supplementary_columns: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
    df_time_zone: Optional[str] = None
):
    """ Initialises a `TimeSeries` object with the given parameters to use within unit tests.

    Args:
        times: List of datetime objects to be used as the time data.
        values: Dictionary of {"column name": values} to build up the dataframe
        resolution: The resolution of the times data.
        periodicity: The periodicity of the times data.
        time_zone: The time zone to set for the `TimeSeries` object.
        supplementary_columns: Columns to set as  supplementary.
        metadata: Dictionary of column metadata
        df_time_zone: The time zone to assign to the DataFrame's 'time' column.

    Returns:
        An instance of `TimeSeries` class.

     Notes:
        - If `times` is None, a `MagicMock` of a Polars DataFrame is used.
        - The `_setup` method of `TimeSeries` is patched to prevent side effects during initialization.
    """
    if times is None:
        df = MagicMock(pl.DataFrame)
    else:
        df_dict = {"time": times}
        if values:
            df_dict |= values
        df = pl.DataFrame(df_dict)
        if df_time_zone:
            df = df.with_columns(pl.col("time").dt.replace_time_zone(df_time_zone))

    with patch.object(TimeSeries, '_setup'):
        ts = TimeSeries(
            df=df,
            time_name="time",
            resolution=resolution,
            periodicity=periodicity,
            time_zone=time_zone,
            supplementary_columns=supplementary_columns,
            column_metadata=metadata
        )

    if supplementary_columns:
        ts.set_supplementary_columns(supplementary_columns)

    return ts


class TestInitSupplementaryColumn(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    values = {"data_column": [1, 2, 3], "supp_column1": ["a", "b", "c"], "supp_column2": ["x", "y", "z"]}

    @parameterized.expand([
        ("int", "new_int_col", 5, pl.Int32),
        ("float", "new_float_col", 3.14, pl.Float64),
        ("str", "new_str_col", "test", pl.String),
        ("none", "new_null_col", None, pl.Null),
    ])
    def test_init_supplementary_column_with_single_value(self, _, new_col_name, new_col_value, new_col_type):
        """Test initialising a supplementary column with a single value (None, int, float or str)."""
        ts = init_timeseries(self.times, self.values)
        ts.init_supplementary_column(new_col_name, new_col_value)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, [new_col_value] * len(self.times), new_col_type))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand([
        ("int_int", "new_int_col", 5, pl.Int32),
        ("int_float", "new_float_col", 3, pl.Float64),
        ("float_int", "new_int_col", 3.4, pl.Int32), # Expect this to convert to 3
        ("none_float", "new_float_col", None, pl.Float64),
        ("none_string", "new_str_col", None, pl.String),
        ("str_int", "new_int_col", "5", pl.Int32),
        ("str_float", "new_float_col", "3.5", pl.Float64),
        ("float_string", "new_null_col", 3.4, pl.String),
    ])
    def test_init_supplementary_column_with_single_value_and_dtype(self, _, new_col_name, new_col_value, dtype):
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = init_timeseries(self.times, self.values)
        ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, [new_col_value] * len(self.times)).cast(dtype))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand([
        ("str_int", "new_int_col", "test", pl.Int32),
        ("str_float", "new_float_col", "test", pl.Float64),
    ])
    def test_init_supplementary_column_with_single_value_and_bad_dtype(self, _, new_col_name, new_col_value, dtype):
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = init_timeseries(self.times, self.values)
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
        ts = init_timeseries(self.times, self.values)
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
        ts = init_timeseries(self.times, self.values)
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
        ts = init_timeseries(self.times, self.values)
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
        ts = init_timeseries(self.times, self.values)
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
                                 "supp_column2": DataColumn("supp_column2", ts),
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
                                 "supp_column2": DataColumn("supp_column2", ts),
                                 }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)


class TestRoundToPeriod(unittest.TestCase):
    times = [datetime(2020, 6, 1, 8, 10, 5, 2000), datetime(2021, 4, 30, 15, 30, 10, 250), datetime(2022, 12, 31, 12)]

    @parameterized.expand([
        ("simple yearly", Period.of_years(1),
         [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)]
         ),
        ("yearly with offset", Period.of_years(1).with_month_offset(2),
         [datetime(2020, 3, 1), datetime(2021, 3, 1), datetime(2022, 3, 1)]
         )
    ])
    def test_round_to_year_period(self, _, period, expected):
        """ Test that rounding a time series to a given year period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
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
    def test_round_to_month_period(self, _, period, expected):
        """ Test that rounding a time series to a given month period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple daily", Period.of_days(1),
         [datetime(2020, 6, 1), datetime(2021, 4, 30), datetime(2022, 12, 31)]
         ),
        ("daily with offset", Period.of_days(1).with_hour_offset(9),
         [datetime(2020, 5, 31, 9), datetime(2021, 4, 30, 9), datetime(2022, 12, 31, 9)]
         )
    ])
    def test_round_to_day_period(self, _, period, expected):
        """ Test that rounding a time series to a given day period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
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
    def test_round_to_hour_period(self, _, period, expected):
        """ Test that rounding a time series to a given hour period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
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
    def test_round_to_minute_period(self, _, period, expected):
        """ Test that rounding a time series to a given minute period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple seconds", Period.of_seconds(1),
         [datetime(2020, 6, 1, 8, 10, 5), datetime(2021, 4, 30, 15, 30, 10), datetime(2022, 12, 31, 12)]
         ),
        ("3 seconds", Period.of_seconds(3),
         [datetime(2020, 6, 1, 8, 10, 3), datetime(2021, 4, 30, 15, 30, 9), datetime(2022, 12, 31, 12)]
         )
    ])
    def test_round_to_second_period(self, _, period, expected):
        """ Test that rounding a time series to a given second period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple microseconds", Period.of_microseconds(200),
         [datetime(2020, 6, 1, 8, 10, 5, 2000), datetime(2021, 4, 30, 15, 30, 10, 200), datetime(2022, 12, 31, 12)]
         ),
    ])
    def test_round_to_microsecond_period(self, _, period, expected):
        """ Test that rounding a time series to a given microsecond period works as expected.
        """
        ts = init_timeseries(self.times)
        result = ts._round_time_to_period(period)
        assert_series_equal(result, pl.Series("time", expected))


class TestValidateResolution(unittest.TestCase):
    """Test the _validate_resolution method."""

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
    def test_validate_year_resolution_success(self, _, times, resolution):
        """ Test that a year based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("simple yearly error",
         [datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("water years error",
         [datetime(2006, 10, 1, 10), datetime(2007, 10, 1, 9), datetime(2008, 10, 1, 9)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_validate_year_resolution_error(self, _, times, resolution):
        """ Test that a year based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()

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
    def test_validate_month_resolution_success(self, _, times, resolution):
        """ Test that a month based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 2, 15), datetime(2024, 3, 1)],
         Period.of_months(1)),

        ("water months error",
         [datetime(2024, 1, 1, 9, 20), datetime(2024, 2, 1, 9), datetime(2024, 3, 1, 9)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_validate_month_resolution_error(self, _, times, resolution):
        """ Test that a month based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()

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
    def test_validate_day_resolution_success(self, _, times, resolution):
        """ Test that a day based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("simple daily error",
         [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3, 0, 0, 0, 1)],
         Period.of_days(1)),

        ("water days error",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 9), datetime(2024, 1, 3, 19)],
         Period.of_days(1).with_hour_offset(9)),
    ])
    def test_validate_day_resolution_error(self, _, times, resolution):
        """ Test that a day based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()

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
    def test_validate_hour_resolution_success(self, _, times, resolution):
        """ Test that an hour based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 2, 15), datetime(2024, 1, 1, 3)],
         Period.of_hours(1)),

        ("hourly with minute offset",
         [datetime(2024, 1, 1, 1, 30), datetime(2024, 1, 1, 2, 30), datetime(2024, 1, 1, 3, 31)],
         Period.of_hours(1).with_minute_offset(30)),
    ])
    def test_validate_hour_resolution_error(self, _, times, resolution):
        """ Test that an hour based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()

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
    def test_validate_minute_resolution_success(self, _, times, resolution):
        """ Test that a minute based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 3, 59)],
         Period.of_minutes(1)),

        ("5 minutes with second offset",
         [datetime(2024, 1, 1, 1, 31, 30), datetime(2024, 1, 1, 1, 35, 30), datetime(2024, 1, 1, 1, 40, 30)],
         Period.of_minutes(5).with_second_offset(30)),
    ])
    def test_validate_minute_resolution_error(self, _, times, resolution):
        """ Test that a minute based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()

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
    def test_validate_second_resolution_success(self, _, times, resolution):
        """ Test that a second based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 1, 2, 9000), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),

        ("30 seconds with microsecond offset",
         [datetime(2024, 1, 1, 1, 1, 0, 40), datetime(2024, 1, 1, 1, 1, 30, 41), datetime(2024, 1, 1, 1, 2, 30, 40)],
         Period.of_seconds(30).with_microsecond_offset(40)),
    ])
    def test_validate_second_resolution_error(self, _, times, resolution):
        """ Test that a second based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()

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
    def test_validate_microsecond_resolution_success(self, _, times, resolution):
        """ Test that a microsecond based time series that does conform to the given resolution passes the validation.
        """
        ts = init_timeseries(times, resolution=resolution)
        ts._validate_resolution()

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000), datetime(2024, 1, 1, 1, 1, 1, 55_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_validate_microsecond_resolution_error(self, _, times, resolution):
        """ Test that a microsecond based time series that doesn't conform to the given resolution raises an error.
        """
        ts = init_timeseries(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts._validate_resolution()


class TestValidatePeriodicity(unittest.TestCase):
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
    def test_validate_year_periodicity_success(self, _, times, periodicity):
        """ Test that a year based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("simple yearly",
         [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2022, 1, 1)],
         Period.of_years(1)),

        ("simple water year",
         [datetime(2005, 1, 1), datetime(2005, 5, 1), datetime(2006, 12, 1), datetime(2007, 10, 10)],
         Period.of_years(1).with_month_offset(9).with_hour_offset(9)),
    ])
    def test_validate_year_periodicity_error(self, _, times, periodicity):
        """ Test that a year based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts._validate_periodicity()

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
    def test_validate_month_periodicity_success(self, _, times, periodicity):
        """ Test that a month based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("simple monthly",
         [datetime(2024, 1, 1), datetime(2024, 1, 31), datetime(2024, 2, 1)],
         Period.of_months(1)),

        ("simple water months",
         [datetime(2024, 1, 1, 10), datetime(2024, 2, 1, 8, 59), datetime(2024, 5, 1, 10, 15)],
         Period.of_months(1).with_hour_offset(9)),
    ])
    def test_validate_month_periodicity_error(self, _, times, periodicity):
        """ Test that a month based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts._validate_periodicity()

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
    def test_validate_day_periodicity_success(self, _, times, periodicity):
        """ Test that a day based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("simple daily",
         [datetime(2024, 1, 1), datetime(2024, 1, 1, 23, 59), datetime(2024, 1, 3)],
         Period.of_days(1)),

        ("simple water days",
         [datetime(2024, 1, 1, 9), datetime(2024, 1, 2, 8, 59, 59), datetime(2024, 1, 3, 9)],
         Period.of_days(1).with_hour_offset(9)),
    ])
    def test_validate_day_periodicity_error(self, _, times, periodicity):
        """ Test that a day based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts._validate_periodicity()

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
    def test_validate_hour_periodicity_success(self, _, times, periodicity):
        """ Test that an hour based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 16), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1)),
    ])
    def test_validate_hour_periodicity_error(self, _, times, periodicity):
        """ Test that an hour based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts._validate_periodicity()

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
    def test_validate_minute_periodicity_success(self, _, times, periodicity):
        """ Test that a minute based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 2, 1)],
         Period.of_minutes(1)),
    ])
    def test_validate_minute_periodicity_error(self, _, times, periodicity):
        """ Test that a minute based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts._validate_periodicity()

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
    def test_validate_second_periodicity_success(self, _, times, periodicity):
        """ Test that a second based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 1, 15001), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),
    ])
    def test_validate_second_periodicity_error(self, _, times, periodicity):
        """ Test that a second based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts._validate_periodicity()

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
    def test_validate_microsecond_periodicity_success(self, _, times, periodicity):
        """ Test that a microsecond based time series that does conform to the given periodicity passes the validation.
        """
        ts = init_timeseries(times, periodicity=periodicity)
        ts._validate_periodicity()

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 25_001), datetime(2023, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_validate_microsecond_periodicity_error(self, _, times, periodicity):
        """ Test that a microsecond based time series that doesn't conform to the given periodicity raises an error.
        """
        ts = init_timeseries(times, periodicity=periodicity)
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
        ts = init_timeseries()
        with self.assertRaises(NotImplementedError):
            ts._epoch_check(period)

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
        ts = init_timeseries()
        ts._epoch_check(period)


class TestSetTimeZone(unittest.TestCase):
    times = [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 15), datetime(2020, 1, 1, 18)]

    def test_time_zone_provided(self):
        """Test that the time zone is set to time_zone if it is provided.
        """
        ts = init_timeseries(self.times, time_zone="America/New_York")
        ts._set_time_zone()

        self.assertEqual(ts._time_zone, "America/New_York")
        self.assertEqual(ts.df.schema["time"].time_zone, "America/New_York")

    def test_time_zone_not_provided_but_df_has_tz(self):
        """Test that the time zone is set to the DataFrame's time zone if time_zone is None.
        """
        ts = init_timeseries(self.times, time_zone=None, df_time_zone="Europe/London")
        ts._set_time_zone()

        self.assertEqual(ts._time_zone, "Europe/London")
        self.assertEqual(ts.df.schema["time"].time_zone, "Europe/London")

    def test_time_zone_default(self):
        """Test that the default time zone is used if both time_zone and the dataframe time zone are None.
        """
        ts = init_timeseries(self.times)
        ts._set_time_zone()

        self.assertEqual(ts._time_zone, "UTC")
        self.assertEqual(ts.df.schema["time"].time_zone, "UTC")

    def test_time_zone_provided_but_df_has_different_tz(self):
        """Test that the time zone is changed if the provided time zone is different to that of the df
        """
        ts = init_timeseries(self.times, time_zone="America/New_York", df_time_zone="Europe/London")
        ts._set_time_zone()

        self.assertEqual(ts._time_zone, "America/New_York")
        self.assertEqual(ts.df.schema["time"].time_zone, "America/New_York")


class TestSelectTime(unittest.TestCase):
    times = [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 15), datetime(2020, 1, 1, 18)]

    def test_select_time(self):
        """Test that we select just the primary datetime column as a series
        """
        ts = init_timeseries(self.times)
        result = ts.select_time()
        expected = pl.Series("time", self.times)
        assert_series_equal(result, expected)


class TestSortTime(unittest.TestCase):

    def test_sort_random_dates(self):
        """Test that random dates are sorted appropriately
        """
        times = [date(1990, 1, 1), date(2019, 5, 8), date(1967, 12, 25), date(2059, 8, 12)]
        ts = init_timeseries(times)
        ts.sort_time()

        expected = pl.Series("time", [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)])
        assert_series_equal(ts.select_time(), expected)

    def test_sort_sorted_dates(self):
        """Test that already sorted dates are maintained
        """
        times = [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)]
        ts = init_timeseries(times)
        ts.sort_time()

        expected = pl.Series("time", times)
        assert_series_equal(ts.select_time(), expected)

    def test_sort_times(self):
        """Test that times are sorted appropriately
        """
        times = [datetime(2024, 1, 1, 12, 59), datetime(2024, 1, 1, 12, 55), datetime(2024, 1, 1, 12, 18),
                 datetime(2024, 1, 1, 1, 5)]
        ts = init_timeseries(times)
        ts.sort_time()

        expected = pl.Series("time", [datetime(2024, 1, 1, 1, 5), datetime(2024, 1, 1, 12, 18),
                                      datetime(2024, 1, 1, 12, 55), datetime(2024, 1, 1, 12, 59)])
        assert_series_equal(ts.select_time(), expected)


class TestValidateTimeColumn(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    values = {"data_column": [1, 2, 3]}

    def test_validate_unchanged_dataframe(self):
        """Test that an unchanged DataFrame passes validation."""
        ts = init_timeseries(self.times, self.values)
        ts._validate_time_column(ts._df)

    def test_validate_dataframe_changed_values(self):
        """Test that a DataFrame with different values, but same times, passes validation."""
        ts = init_timeseries(self.times, self.values)
        new_df = pl.DataFrame({
            "time": self.times,
            "data_column": [10, 20, 30]
        })
        ts._validate_time_column(ts._df)

    def test_validate_time_column_missing_time_column(self):
        """Test that a DataFrame missing the time column raises a ValueError."""
        invalid_df = pl.DataFrame(self.values)
        ts = init_timeseries()
        with self.assertRaises(ValueError):
            ts._validate_time_column(invalid_df)

    def test_validate_time_column_mutated_timestamps(self):
        """Test that a DataFrame with mutated timestamps raises a ValueError."""
        ts = init_timeseries(self.times, self.values)
        mutated_df = pl.DataFrame({
            "time": [datetime(1924, 1, 1), datetime(1924, 1, 2), datetime(1924, 1, 3)]
        })
        with self.assertRaises(ValueError):
            ts._validate_time_column(mutated_df)

    def test_validate_time_column_extra_timestamps(self):
        """Test that a DataFrame with extra timestamps raises a ValueError."""
        ts = init_timeseries(self.times, self.values)
        extra_timestamps_df = pl.DataFrame({
            "time": self.times + [datetime(2024, 1, 4), datetime(2024, 1, 5)]
        })
        with self.assertRaises(ValueError):
            ts._validate_time_column(extra_timestamps_df)

    def test_validate_time_column_fewer_timestamps(self):
        """Test that a DataFrame with fewer timestamps raises a ValueError."""
        ts = init_timeseries(self.times, self.values)
        fewer_timestamps_df = pl.DataFrame({
            "time": self.times[:-1]
        })
        with self.assertRaises(ValueError):
            ts._validate_time_column(fewer_timestamps_df)


class TestRemoveMissingColumns(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"description": "meta1"},
        "col2": {"description": "meta2"},
        "col3": {"description": "meta3"}
    }

    def test_no_columns_removed(self):
        """Test that no columns are removed when all columns are present in the new DataFrame."""
        ts = init_timeseries(self.times, self.values, supplementary_columns=["col1"], metadata=self.metadata)

        new_df = pl.DataFrame({
            "time": self.times,
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        ts._remove_missing_columns(new_df)
        ts._df = new_df

        self.assertIn("col1", ts.supplementary_columns)
        self.assertIn("col2", ts.data_columns)
        self.assertIn("col3", ts.data_columns)
        self.assertEqual(ts.metadata(), self.metadata)

    def test_single_data_column_removed(self):
        """Test that single data column is removed correctly."""
        ts = init_timeseries(self.times, self.values, supplementary_columns=["col1"], metadata=self.metadata)

        new_df = pl.DataFrame({
            "time": self.times,
            "col1": [1, 2, 3],
            "col3": [7, 8, 9]
        })
        ts._remove_missing_columns(new_df)
        ts._df = new_df

        self.assertEqual(ts.columns, ["col1", "col3"])
        self.assertEqual(ts.supplementary_columns, ["col1"])
        self.assertEqual(ts.data_columns, ["col3"])
        self.assertEqual(ts.metadata(), {"col1": {"description": "meta1"}, "col3": {"description": "meta3"}})

    def test_single_supplementary_column_removed(self):
        """Test that single supplementary column is removed correctly."""
        ts = init_timeseries(self.times, self.values, supplementary_columns=["col1"], metadata=self.metadata)

        new_df = pl.DataFrame({
            "time": self.times,
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        ts._remove_missing_columns(new_df)
        ts._df = new_df

        self.assertEqual(ts.columns, ["col2", "col3"])
        self.assertEqual(ts.supplementary_columns, [])
        self.assertEqual(ts.data_columns, ["col2", "col3"])
        self.assertEqual(ts.metadata(), {"col2": {"description": "meta2"}, "col3": {"description": "meta3"}})

    def test_multiple_columns_removed(self):
        """Test that multiple columns are removed correctly."""
        ts = init_timeseries(self.times, self.values, supplementary_columns=["col1"], metadata=self.metadata)

        new_df = pl.DataFrame({
            "time": self.times,
            "col3": [7, 8, 9]
        })
        ts._remove_missing_columns(new_df)
        ts._df = new_df

        self.assertEqual(ts.columns, ["col3"])
        self.assertEqual(ts.supplementary_columns, [])
        self.assertEqual(ts.data_columns, ["col3"])
        self.assertEqual(ts.metadata(), {"col3": {"description": "meta3"}})

    def test_added_column(self):
        """Test that adding a new column works"""
        ts = init_timeseries(self.times, self.values, supplementary_columns=["col1"], metadata=self.metadata)

        new_df = pl.DataFrame({
            "time": self.times,
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
            "col4": [10, 11, 12],
        })
        ts._remove_missing_columns(new_df)
        ts._df = new_df

        self.assertEqual(ts.columns, ["col1", "col2", "col3", "col4"])
        self.assertEqual(ts.supplementary_columns, ["col1"])
        self.assertEqual(ts.data_columns, ["col2", "col3", "col4"])
        self.assertEqual(ts.metadata(), {"col1": {"description": "meta1"}, "col2": {"description": "meta2"},
                                         "col3": {"description": "meta3"}, "col4": {}})


class TestSelectColumns(unittest.TestCase):
    times = [datetime(2024, 1, 1, tzinfo=TZ_UTC),
             datetime(2024, 1, 2, tzinfo=TZ_UTC),
             datetime(2024, 1, 3, tzinfo=TZ_UTC)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_select_single_column(self):
        """Test selecting a single of column."""
        ts = init_timeseries(self.times, self.values)
        expected_df = pl.DataFrame({"time": self.times} | {"col1": self.values["col1"]})

        ts = ts.select(["col1"])
        assert_frame_equal(ts.df, expected_df)

    def test_select_multiple_columns(self):
        """Test selecting a single of column."""
        ts = init_timeseries(self.times, self.values)
        expected_df = pl.DataFrame({"time": self.times} | {"col1": self.values["col1"], "col2": self.values["col2"]})

        ts = ts.select(["col1", "col2"])
        assert_frame_equal(ts.df, expected_df)

    def test_select_no_columns_raises_error(self):
        """Test selecting no columns raises error."""
        ts = init_timeseries(self.times, self.values)
        with self.assertRaises(ValueError):
            ts.select([])

    def test_select_nonexistent_column(self):
        """Test selecting a column that does not exist raises error."""
        ts = init_timeseries(self.times, self.values)
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            ts.select(["nonexistent_column"])

    def test_select_existing_and_nonexistent_column(self):
        """Test selecting a column that does not exist, alongside existing columns, still raises error"""
        ts = init_timeseries(self.times, self.values)
        with self.assertRaises(pl.exceptions.ColumnNotFoundError):
            ts.select(["col1", "col2", "nonexistent_column"])

    def test_select_column_doesnt_mutate_original_ts(self):
        """When selecting a column, the original ts should be unchanged"""
        ts = init_timeseries(self.times, self.values)
        original_df = ts.df

        col1_ts = ts.select(["col1"])
        expected_df = pl.DataFrame({"time": self.times} | {"col1": self.values["col1"]})
        assert_frame_equal(col1_ts.df, expected_df)
        assert_frame_equal(ts.df, original_df)

        col2_ts = ts.select(["col2"])
        expected_df = pl.DataFrame({"time": self.times} | {"col2": self.values["col2"]})
        assert_frame_equal(col2_ts.df, expected_df)
        assert_frame_equal(ts.df, original_df)

    def test_select_column_trims_metadata(self):
        """When selecting a column, the original ts metadata should be unchanged"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)

        col1_ts = ts.select(["col1"])
        expected_metadata = {"col1": self.metadata["col1"]}
        self.assertEqual(col1_ts.metadata(), expected_metadata)
        # Ensure original ts metadata unchanged
        self.assertEqual(ts.metadata(), self.metadata)

    def test_select_column_trims_supplementary_columns(self):
        """When selecting a column, the original ts supplementary columns should be unchanged"""
        ts = init_timeseries(self.times, self.values, supplementary_columns=["col2"])

        col1_ts = ts.select(["col1"])
        self.assertEqual(col1_ts.columns, ["col1"])
        self.assertEqual(col1_ts.supplementary_columns, [])
        self.assertEqual(col1_ts.data_columns, ["col1"])

        # Ensure original ts supplementary columns unchanged
        self.assertEqual(ts.columns, ["col1", "col2", "col3"])
        self.assertEqual(ts.supplementary_columns, ["col2"])
        self.assertEqual(ts.data_columns, ["col1", "col3"])


class TestMetadata(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }
    
    def test_metadata_no_key(self):
        """Test retrieving all metadata when no key is provided."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.metadata()
        self.assertEqual(result, self.metadata)

    def test_metadata_single_key(self):
        """Test relevant metadata is returned when a single key is provided."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.metadata("key1")
        expected = {
            "col1": {"key1": "1"},
            "col2": {"key1": "2"},
            "col3": {"key1": "3"},
        }
        self.assertEqual(result, expected)

    def test_metadata_multiple_keys(self):
        """Test relevant metadata is returned when multiple keys are provided."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.metadata(["key1", "key2"])
        expected = {
            "col1": {"key1": "1", "key2": "10"},
            "col2": {"key1": "2", "key2": "20"},
            "col3": {"key1": "3", "key2": "30"},
        }
        self.assertEqual(result, expected)

    def test_metadata_non_existent_key_returns_none(self):
        """Test that a non-existent key returns no metadata"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.metadata("nonexistent_key")
        expected = {
            "col1": {"nonexistent_key": None},
            "col2": {"nonexistent_key": None},
            "col3": {"nonexistent_key": None},
        }
        self.assertEqual(result, expected)

    def test_metadata_mix_existent_and_non_existent_key(self):
        """Test that a mix of existent and non-existent keys returns appropriate metadata"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.metadata(["key1", "nonexistent_key"])
        expected = {
            "col1": {"key1": "1", "nonexistent_key": None},
            "col2": {"key1": "2", "nonexistent_key": None},
            "col3": {"key1": "3", "nonexistent_key": None},
        }
        self.assertEqual(result, expected)


class TestGetMetadata(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"}
    }

    def test_get_all_metadata_for_column(self):
        """Test retrieving all metadata for a single column"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("col1")
        expected = {"key1": "1", "key2": "10", "key3": "100"}
        self.assertEqual(result, expected)

    def test_get_single_metadata_key_for_column(self):
        """Test retrieving  a single metadata item for a single column"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("col1", "key1")
        expected = {"key1": "1"}
        self.assertEqual(result, expected)
        
    def test_get_multiple_metadata_keys_for_column(self):
        """Test retrieving multiple metadata items for a single column"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("col1", ["key1", "key2"])
        expected = {"key1": "1", "key2": "10"}
        self.assertEqual(result, expected)

    def test_get_non_existent_key_for_column(self):
        """Test a non-existent key returns None"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("col1", "nonexistent_key")
        expected = {"nonexistent_key": None}
        self.assertEqual(result, expected)

    def test_get_mix_existing_and_non_existent_key_for_column(self):
        """Test a mix of existing and non-existent keys returns appropriate metadata"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("col1", ["key1", "key2", "nonexistent_key"])
        expected = {"key1": "1", "key2": "10", "nonexistent_key": None}
        self.assertEqual(result, expected)

    def test_get_metadata_for_existing_column_with_no_metadata(self):
        """Test retrieving metadata for a column with no metadata."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("col3")
        expected = {}
        self.assertEqual(result, expected)

    def test_get_metadata_for_non_existent_column(self):
        """Test retrieving metadata for a non-existing column returns empty"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts._get_metadata("nonexistent_column")
        expected = {}
        self.assertEqual(result, expected)


class TestRemoveMetadata(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_remove_all_metadata_for_column(self):
        """Test removing all metadata for a single column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        ts.remove_metadata("col1")
        expected = {
            "col1": {},
            "col2": {"key1": "2", "key2": "20", "key3": "200"},
            "col3": {"key1": "3", "key2": "30", "key3": "300"},
        }
        self.assertEqual(ts.metadata(), expected)

    def test_remove_all_metadata_by_keys_for_column(self):
        """Test removing all the metadata keys by name for a single column keeps that column in the metadata dict"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        ts.remove_metadata("col1", ["key1", "key2", "key3"])
        expected = {
            "col1": {},
            "col2": {"key1": "2", "key2": "20", "key3": "200"},
            "col3": {"key1": "3", "key2": "30", "key3": "300"},
        }
        self.assertEqual(ts._metadata, expected)

    def test_remove_single_metadata_key_for_column(self):
        """Test removing a single metadata key for a single column, ensuring other keys are maintained."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        ts.remove_metadata("col1", "key1")
        expected = {
            "col1": {"key2": "10", "key3": "100"},
            "col2": {"key1": "2", "key2": "20", "key3": "200"},
            "col3": {"key1": "3", "key2": "30", "key3": "300"},
        }
        self.assertEqual(ts._metadata, expected)

    def test_remove_multiple_metadata_key_for_column(self):
        """Test removing multiple metadata keys for a single column, ensuring other keys are maintained."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        ts.remove_metadata("col1", ["key1", "key2"])
        expected = {
            "col1": {"key3": "100"},
            "col2": {"key1": "2", "key2": "20", "key3": "200"},
            "col3": {"key1": "3", "key2": "30", "key3": "300"},
        }
        self.assertEqual(ts._metadata, expected)

    def test_remove_metadata_for_non_existent_column(self):
        """Test removing metadata for a non-existent column doesn't change the existing metadata"""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        ts.remove_metadata("nonexistent_column")        
        self.assertEqual(ts._metadata, self.metadata)
        
    def test_remove_metadata_for_non_existent_key(self):
        """Test removing metadata for an existing column, but the metadata key doesn't exist."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        ts.remove_metadata("col1", "nonexistent_key")
        self.assertEqual(ts._metadata, self.metadata)


class TestGetattr(unittest.TestCase):
    times = [datetime(2024, 1, 1, tzinfo=TZ_UTC),
             datetime(2024, 1, 2, tzinfo=TZ_UTC),
             datetime(2024, 1, 3, tzinfo=TZ_UTC)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_access_time_column(self):
        """Test accessing the time column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.time
        expected = pl.Series("time", self.times)
        assert_series_equal(result, expected)

    def test_access_data_column(self):
        """Test accessing a data column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.col1
        expected = pl.DataFrame({"time": self.times, "col1": self.values["col1"]})
        assert_frame_equal(result.df, expected)

    def test_access_metadata_key(self):
        """Test accessing metadata key for a single column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts.col1.key1
        expected = "1"
        self.assertEqual(result, expected)

    def test_access_metadata_key_without_single_column(self):
        """Test accessing metadata key without filtering to a single column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        with self.assertRaises(AttributeError):
            result = ts.key1

    def test_access_nonexistent_attribute(self):
        """Test accessing metadata key without filtering to a single column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        with self.assertRaises(AttributeError):
            result = ts.col1.key0


class TestGetItem(unittest.TestCase):
    times = [datetime(2024, 1, 1, tzinfo=TZ_UTC),
             datetime(2024, 1, 2, tzinfo=TZ_UTC),
             datetime(2024, 1, 3, tzinfo=TZ_UTC)]
    values = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_access_time_column(self):
        """Test accessing the time column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts["time"]
        expected = pl.Series("time", self.times)
        assert_series_equal(result, expected)

    def test_access_data_column(self):
        """Test accessing a data column."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts["col1"]
        expected = pl.DataFrame({"time": self.times, "col1": self.values["col1"]})
        assert_frame_equal(result.df, expected)

    def test_access_multiple_data_columns(self):
        """Test accessing multiple data columns."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        result = ts[["col1", "col2"]]
        expected = pl.DataFrame({"time": self.times, "col1": self.values["col1"], "col2": self.values["col2"]})
        assert_frame_equal(result.df, expected)

    def test_non_existent_column(self):
        """Test accessing non-existent data column raises error."""
        ts = init_timeseries(self.times, self.values, metadata=self.metadata)
        with self.assertRaises(pl.ColumnNotFoundError):
            result = ts["col0"]

if __name__ == '__main__':
    unittest.main()