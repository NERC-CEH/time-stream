import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime

import polars as pl
from parameterized import parameterized

from time_series.time_series_polars import TimeSeriesPolars
from time_series.period import Period


class TestInitialization(unittest.TestCase):
    def test_initialization(self):
        """Test that the object initializes correctly."""
        df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "value": [1, 2, 3]
        })

        ts_polars = TimeSeriesPolars(
            df=df,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            time_zone="UTC"
        )
        self.assertIsInstance(ts_polars, TimeSeriesPolars)


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
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_year_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
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
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_month_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
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
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_day_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
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
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_hour_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
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
         Period.of_minutes(1).with_second_offset(30)),
    ])
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_minute_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
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
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_second_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
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
    @patch.object(TimeSeriesPolars, '_setup')
    def test_validate_microsecond_resolution_success(self, name, times, resolution, mock_setup):
        df = pl.DataFrame({"time": times})
        ts_polars = TimeSeriesPolars(df=df, time_name="time", resolution=resolution, periodicity=MagicMock(Period))
        ts_polars._validate_resolution()

    @parameterized.expand([
        Period.of_years(2), Period.of_years(7), Period.of_years(10)
    ])
    @patch.object(TimeSeriesPolars, '_setup')
    def test_non_epoch_agnostic_year_resolution_fails(self, resolution, mock_setup):
        ts_polars = TimeSeriesPolars(df=MagicMock(pl.DataFrame), time_name="time", resolution=resolution,
                                     periodicity=MagicMock(Period))
        with self.assertRaises(NotImplementedError):
            ts_polars._validate_resolution()

    @parameterized.expand([
        Period.of_years(2), Period.of_years(7), Period.of_years(10),
        Period.of_months(5), Period.of_months(7), Period.of_months(9), Period.of_months(10), Period.of_months(11), Period.of_months(13),
        Period.of_days(2), Period.of_days(7), Period.of_days(65),
        Period.of_hours(5), Period.of_hours(7), Period.of_hours(9), Period.of_hours(11), Period.of_hours(25),
        Period.of_minutes(7), Period.of_minutes(11), Period.of_minutes(50), Period.of_minutes(61),
    ])
    @patch.object(TimeSeriesPolars, '_setup')
    def test_non_epoch_agnostic_resolution_fails(self, resolution, mock_setup):
        ts_polars = TimeSeriesPolars(df=MagicMock(pl.DataFrame), time_name="time", resolution=resolution,
                                     periodicity=MagicMock(Period))
        with self.assertRaises(NotImplementedError):
            ts_polars._validate_resolution()