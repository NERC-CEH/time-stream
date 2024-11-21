import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from typing import Optional

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal, assert_frame_equal

from time_series.time_series_polars import TimeSeriesPolars
from time_series.period import Period


def init_ts_polars(
    times: Optional[list[datetime]] = None,
    resolution: Optional[Period] = None,
    periodicity: Optional[Period] = None,
    time_zone: Optional[str] = None,
    df_time_zone: Optional[str] = None
):
    """ Initializes a `TimeSeriesPolars` object with the given parameters to use within unit tests.

    Args:
        times: List of datetime objects to be used as the time data.
        resolution: The resolution of the times data.
        periodicity: The periodicity of the times data.
        time_zone: The time zone to set for the `TimeSeriesPolars` object.
        df_time_zone: The time zone to assign to the DataFrame's 'time' column.

    Returns:
        An instance of `TimeSeriesPolars` class.

     Notes:
        - If `times` is None, a `MagicMock` of a Polars DataFrame is used.
        - The `_setup` method of `TimeSeriesPolars` is patched to prevent side effects during initialization.
    """
    if times is None:
        df = MagicMock(pl.DataFrame)
    else:
        df = pl.DataFrame({"time": times})
        if df_time_zone:
            df = df.with_columns(pl.col("time").dt.replace_time_zone(df_time_zone))

    with patch.object(TimeSeriesPolars, '_setup'):
        ts_polars = TimeSeriesPolars(df=df,
                                     time_name="time",
                                     resolution=resolution,
                                     periodicity=periodicity,
                                     time_zone=time_zone)
    return ts_polars


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
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
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
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
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
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
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
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
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
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
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
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
        assert_series_equal(result, pl.Series("time", expected))

    @parameterized.expand([
        ("simple microseconds", Period.of_microseconds(200),
         [datetime(2020, 6, 1, 8, 10, 5, 2000), datetime(2021, 4, 30, 15, 30, 10, 200), datetime(2022, 12, 31, 12)]
         ),
    ])
    def test_round_to_microsecond_period(self, _, period, expected):
        """ Test that rounding a time series to a given microsecond period works as expected.
        """
        ts_polars = init_ts_polars(self.times)
        result = ts_polars._round_time_to_period(period)
        assert_series_equal(result, pl.Series("time", expected))


class TestInitialization(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "value": [1, 2, 3]
        })

    def test_initialization(self):
        """Test that the object initializes correctly."""
        ts_polars = TimeSeriesPolars(
            df=self.df,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            time_zone="UTC"
        )
        self.assertIsInstance(ts_polars, TimeSeriesPolars)

    def test_initialization_no_supp_cols(self):
        """Test that the object initializes correctly with no supplematery columns set."""
        ts_polars = TimeSeriesPolars(
            df=self.df,
            time_name="time",
        )
        self.assertEqual(ts_polars.data_col_names, ("value",))
        self.assertEqual(ts_polars.supp_col_names, ())

    def test_initialization_with_supp_cols(self):
        """Test that the object initializes correctly with supplementary columns set."""
        ts_polars = TimeSeriesPolars(
            df=self.df,
            time_name="time",
            supp_col_names=["value"]
        )
        self.assertEqual(ts_polars.data_col_names, ())
        self.assertEqual(ts_polars.supp_col_names, ("value",))

    @parameterized.expand([
        ("One bad", ["value", "non_col"]),
        ("multi bad", ["value", "bad_col", "non_col"]),
        ("All bad", ["bad_col", "non_col"]),
    ])
    def test_initialization_with_bad_supp_cols(self, name, supp_col_names):
        """Test that the object raise error for supplementary columns specified that are not in df"""
        df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "value": [1, 2, 3]
        })

        with self.assertRaises(ValueError):
            ts_polars = TimeSeriesPolars(
                df=df,
                time_name="time",
                supp_col_names=supp_col_names
            )


class TestDFOperation(unittest.TestCase):
    """Test the TimeSeries.df_operation method"""

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 2, tzinfo=timezone.utc),
                     datetime(2024, 1, 3, tzinfo=timezone.utc)],
            "value": [1, 2, 3],
            "flags": ["E", None, "S"]
        })
        self.ts = TimeSeriesPolars(
            self.df,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            time_zone="UTC",
            supp_col_names=("flags",))

    @parameterized.expand([
        ("value_change", pl.DataFrame({"time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                                                datetime(2024, 1, 2, tzinfo=timezone.utc),
                                                datetime(2024, 1, 3, tzinfo=timezone.utc)],
                                      "value": [3, 4, 5],
                                      "flags": ["E", None, "S"]})),
        ("time_change", pl.DataFrame({"time": [datetime(2024, 1, 4, tzinfo=timezone.utc),
                                               datetime(2024, 1, 5, tzinfo=timezone.utc),
                                               datetime(2024, 1, 6, tzinfo=timezone.utc)],
                                      "value": [1, 2, 3],
                                      "flags": ["E", None, "S"]})),
        ("col_added", pl.DataFrame({"time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                                             datetime(2024, 1, 2, tzinfo=timezone.utc),
                                             datetime(2024, 1, 3, tzinfo=timezone.utc)],
                                   "value": [1, 2, 3],
                                   "new_col": [3, 4, 5],
                                   "flags": ["E", None, "S"]})),
        ("row_added", pl.DataFrame({"time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                                             datetime(2024, 1, 2, tzinfo=timezone.utc),
                                             datetime(2024, 1, 3, tzinfo=timezone.utc),
                                             datetime(2024, 1, 4, tzinfo=timezone.utc)],
                                      "value": [1, 2, 3, 4],
                                      "flags": ["E", None, "S", None]})),
    ])
    def test_valid_df_operation(self, name, return_df):
        """Test function that returns a valid dataframe."""
        # Mock a function that returns a DataFrame
        mock_func = MagicMock(return_value=return_df)
        result = self.ts.df_operation(mock_func)

        self.assertIsInstance(result, TimeSeriesPolars)
        assert_frame_equal(result.df, return_df)

    def test_invalid_df_operation_return_type(self):
        """Test function that returns an invalid dataframe."""
        # Mock a function that returns an invalid type
        mock_func = MagicMock(return_value="not a dataframe")
        with self.assertRaises(ValueError):
            self.ts.df_operation(mock_func)

    def test_time_name_change_not_specified(self):
        """Test function that returns a dataframe with changed time name."""
        # Mock a function that changes the time name
        mock_func = MagicMock(return_value=pl.DataFrame({
            "new_time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                         datetime(2024, 1, 2, tzinfo=timezone.utc),
                         datetime(2024, 1, 3, tzinfo=timezone.utc)],
            "value": [10, 20, 30]}))
        with self.assertRaises(ValueError):
            self.ts.df_operation(mock_func)

    def test_time_name_change_specified(self):
        """Test function with optional parameters."""
        # Mock a function that changes the time name
        return_df = pl.DataFrame({
            "new_time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                         datetime(2024, 1, 2, tzinfo=timezone.utc),
                         datetime(2024, 1, 3, tzinfo=timezone.utc)],
             "value": [15, 25, 35]})
        mock_func = MagicMock(return_value=return_df)
        result = self.ts.df_operation(mock_func, time_name="new_time")

        assert_frame_equal(result.df, return_df)

    def test_resolution_change_not_specified(self):
        """Test function with  resolution change not specified."""
        # Mock a function that changes the time name
        return_df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                     datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 1, 2, tzinfo=timezone.utc)],
             "value": [15, 25, 35]})
        mock_func = MagicMock(return_value=return_df)

        with self.assertRaises(UserWarning):
            self.ts.df_operation(mock_func)

    def test_resolution_change_specified(self):
        """Test function with optional period parameters."""
        # Mock a function that changes the time name
        return_df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                     datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 1, 2, tzinfo=timezone.utc)],
             "value": [15, 25, 35]})
        mock_func = MagicMock(return_value=return_df)
        result = self.ts.df_operation(mock_func, resolution=Period.of_hours(1), periodicity=Period.of_hours(1))

        assert_frame_equal(result.df, return_df)

    def test_supp_cols_changed_not_specified(self):
        """Test function which adds new supp col."""
        # Mock a function that changes the time name
        return_df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 2, tzinfo=timezone.utc),
                     datetime(2024, 1, 3, tzinfo=timezone.utc)],
             "value": [1, 2, 3],
             "flags_new": ["E", None, "S"]})
        mock_func = MagicMock(return_value=return_df)
        result = self.ts.df_operation(mock_func)

        # As new supp col name not specified and old one removed, there shouldn't be any
        self.assertEqual(result.supp_col_names, ())
        self.assertEqual(result.data_col_names, ("value", "flags_new"))

    def test_supp_cols_changed_specified(self):
        """Test function which adds new supp col and specifies which."""
        # Mock a function that changes the time name
        return_df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 2, tzinfo=timezone.utc),
                     datetime(2024, 1, 3, tzinfo=timezone.utc)],
             "value": [1, 2, 3],
             "flags_new": ["E", None, "S"]})
        mock_func = MagicMock(return_value=return_df)
        result = self.ts.df_operation(mock_func, supp_col_names=("flags_new",))

        # As new supp col name not specified and old one removed, there shouldn't be any
        self.assertEqual(result.supp_col_names, ("flags_new",))
        self.assertEqual(result.data_col_names, ("value",))

    def test_supp_cols_not_changed(self):
        """Test function which adds new supp col and specifies which."""
        # Mock a function that changes the time name
        return_df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 2, tzinfo=timezone.utc),
                     datetime(2024, 1, 3, tzinfo=timezone.utc)],
             "value": [1, 2, 3],
             "flags": ["E", None, "R"]})
        mock_func = MagicMock(return_value=return_df)
        result = self.ts.df_operation(mock_func)

        # As new supp col name not specified and old one removed, there shouldn't be any
        self.assertEqual(result.supp_col_names, ("flags",))
        self.assertEqual(result.data_col_names, ("value",))


class TestSetSupplementalColumns(unittest.TestCase):
    """Test the TimeSeries.set_columns_supplemental method."""

    def setUp(self):
        """Set up a sample DataFrame for testing."""
        self.df = pl.DataFrame({
            "time": [datetime(2024, 1, 1, tzinfo=timezone.utc),
                     datetime(2024, 1, 2, tzinfo=timezone.utc),
                     datetime(2024, 1, 3, tzinfo=timezone.utc)],
            "value": [1, 2, 3],
            "supp1": [4, 5, 6],
            "supp2": [7, 8, 9],
        })
        self.ts = TimeSeriesPolars(self.df, time_name="time", supp_col_names=("supp1",))

    def test_set_new_supp_column(self):
        """Test setting an addtional column as supplemental."""
        result = self.ts.set_columns_supplemental(("supp2",))
        self.assertEqual(set(result.supp_col_names), set(("supp2", "supp1")))

    def test_set_existing_supp_column(self):
        """Test setting a current supplemental column makes no difference."""
        result = self.ts.set_columns_supplemental(("supp1",))
        self.assertEqual(result.supp_col_names, ("supp1",))


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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()

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
        ts_polars = init_ts_polars(times, resolution=resolution)
        ts_polars._validate_resolution()

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 0), datetime(2024, 1, 1, 1, 1, 1, 25_000), datetime(2024, 1, 1, 1, 1, 1, 55_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_validate_microsecond_resolution_error(self, _, times, resolution):
        """ Test that a microsecond based time series that doesn't conform to the given resolution raises an error.
        """
        ts_polars = init_ts_polars(times, resolution=resolution)
        with self.assertRaises(UserWarning):
            ts_polars._validate_resolution()


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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

    @parameterized.expand([
        ("simple hourly",
         [datetime(2024, 1, 1, 1, 15), datetime(2024, 1, 1, 1, 16), datetime(2024, 1, 1, 3, 1)],
         Period.of_hours(1)),
    ])
    def test_validate_hour_periodicity_error(self, _, times, periodicity):
        """ Test that an hour based time series that doesn't conform to the given periodicity raises an error.
        """
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

    @parameterized.expand([
        ("simple minute",
         [datetime(2024, 1, 1, 1, 1), datetime(2024, 1, 1, 1, 2), datetime(2024, 1, 1, 1, 2, 1)],
         Period.of_minutes(1)),
    ])
    def test_validate_minute_periodicity_error(self, _, times, periodicity):
        """ Test that a minute based time series that doesn't conform to the given periodicity raises an error.
        """
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

    @parameterized.expand([
        ("simple seconds",
         [datetime(2024, 1, 1, 1, 1, 1, 1500), datetime(2024, 1, 1, 1, 1, 1, 15001), datetime(2024, 1, 1, 1, 1, 3)],
         Period.of_seconds(1)),
    ])
    def test_validate_second_periodicity_error(self, _, times, periodicity):
        """ Test that a second based time series that doesn't conform to the given periodicity raises an error.
        """
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()

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
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        ts_polars._validate_periodicity()

    @parameterized.expand([
        ("40Hz",
         [datetime(2023, 1, 1, 1, 1, 1, 25_001), datetime(2023, 1, 1, 1, 1, 1, 49_999),
          datetime(2024, 1, 1, 1, 1, 1, 50_000)],
         Period.of_microseconds(25_000)),
    ])
    def test_validate_microsecond_periodicity_error(self, _, times, periodicity):
        """ Test that a microsecond based time series that doesn't conform to the given periodicity raises an error.
        """
        ts_polars = init_ts_polars(times, periodicity=periodicity)
        with self.assertRaises(UserWarning):
            ts_polars._validate_periodicity()


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
        ts_polars = init_ts_polars()
        with self.assertRaises(NotImplementedError):
            ts_polars._epoch_check(period)

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
        ts_polars = init_ts_polars()
        ts_polars._epoch_check(period)


class TestSetTimeZone(unittest.TestCase):
    times = [datetime(2020, 1, 1, 12), datetime(2020, 1, 1, 15), datetime(2020, 1, 1, 18)]

    def test_time_zone_provided(self):
        """Test that the time zone is set to time_zone if it is provided.
        """
        ts_polars = init_ts_polars(self.times, time_zone="America/New_York")
        ts_polars._set_time_zone()

        self.assertEqual(ts_polars._time_zone, "America/New_York")
        self.assertEqual(ts_polars.df.schema["time"].time_zone, "America/New_York")

    def test_time_zone_not_provided_but_df_has_tz(self):
        """Test that the time zone is set to the DataFrame's time zone if time_zone is None.
        """
        ts_polars = init_ts_polars(self.times, time_zone=None, df_time_zone="Europe/London")
        ts_polars._set_time_zone()

        self.assertEqual(ts_polars._time_zone, "Europe/London")
        self.assertEqual(ts_polars.df.schema["time"].time_zone, "Europe/London")

    def test_time_zone_default(self):
        """Test that the default time zone is used if both time_zone and the dataframe time zone are None.
        """
        ts_polars = init_ts_polars(self.times)
        ts_polars._set_time_zone()

        self.assertEqual(ts_polars._time_zone, "UTC")
        self.assertEqual(ts_polars.df.schema["time"].time_zone, "UTC")

    def test_time_zone_provided_but_df_has_different_tz(self):
        """Test that the time zone is changed if the provided time zone is different to that of the df
        """
        ts_polars = init_ts_polars(self.times, time_zone="America/New_York", df_time_zone="Europe/London")
        ts_polars._set_time_zone()

        self.assertEqual(ts_polars._time_zone, "America/New_York")
        self.assertEqual(ts_polars.df.schema["time"].time_zone, "America/New_York")

if __name__ == "__main__":
    unittest.main()