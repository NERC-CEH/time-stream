from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from time_stream.base import TimeFrame
from time_stream.calculations import calculate_min_max_envelope
from time_stream.period import Period


class TestCalculateMinMaxEnvelopes:
    @staticmethod
    def create_timeframe(period: Period, df: pl.DataFrame) -> TimeFrame:
        tf = TimeFrame(df=df, time_name="timestamp", resolution=period, periodicity=period)
        return tf

    def test_calculate_min_max_envelope_daily(self) -> None:
        """Check the min max envelope is calculated correctly for daily resolution data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2022, 1, 5),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2022, 1, 5),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
                "min": [1, 2, 3, 1, 2, 3, 7, 8],
                "max": [4, 5, 6, 4, 5, 6, 7, 8],
            }
        )

        tf = self.create_timeframe(period=Period.of_days(1), df=df)

        actual_df = calculate_min_max_envelope(tf)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)

    def test_calculate_min_max_envelope_hourly(self) -> None:
        """Check the min max envelope is calculated correctly for hourly resolution data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 0, 0),
                    datetime(2020, 1, 1, 1, 0, 0),
                    datetime(2020, 1, 2, 0, 0, 0),
                    datetime(2020, 1, 2, 1, 0, 0),
                    datetime(2021, 1, 1, 0, 0, 0),
                    datetime(2021, 1, 1, 1, 0, 0),
                    datetime(2021, 1, 2, 0, 0, 0),
                    datetime(2021, 1, 2, 1, 0, 0),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 0, 0),
                    datetime(2020, 1, 1, 1, 0, 0),
                    datetime(2020, 1, 2, 0, 0, 0),
                    datetime(2020, 1, 2, 1, 0, 0),
                    datetime(2021, 1, 1, 0, 0, 0),
                    datetime(2021, 1, 1, 1, 0, 0),
                    datetime(2021, 1, 2, 0, 0, 0),
                    datetime(2021, 1, 2, 1, 0, 0),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
                "min": [1, 2, 3, 4, 1, 2, 3, 4],
                "max": [5, 6, 7, 8, 5, 6, 7, 8],
            }
        )

        tf = self.create_timeframe(period=Period.of_hours(1), df=df)

        actual_df = calculate_min_max_envelope(tf)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)

    def test_calculate_min_max_envelope_minute_resolution(self) -> None:
        """Check the min max envelope is calculated correctly for minute resolution data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 1, 0),
                    datetime(2020, 1, 1, 0, 2, 0),
                    datetime(2020, 1, 1, 1, 1, 0),
                    datetime(2020, 1, 1, 1, 2, 0),
                    datetime(2020, 1, 2, 0, 1, 0),
                    datetime(2020, 1, 2, 0, 2, 0),
                    datetime(2020, 1, 2, 1, 1, 0),
                    datetime(2020, 1, 2, 1, 2, 0),
                    datetime(2021, 1, 1, 0, 1, 0),
                    datetime(2021, 1, 1, 0, 2, 0),
                    datetime(2021, 1, 1, 1, 1, 0),
                    datetime(2021, 1, 1, 1, 2, 0),
                    datetime(2021, 1, 2, 0, 1, 0),
                    datetime(2021, 1, 2, 0, 2, 0),
                    datetime(2021, 1, 2, 1, 1, 0),
                    datetime(2021, 1, 2, 1, 2, 0),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 1, 0),
                    datetime(2020, 1, 1, 0, 2, 0),
                    datetime(2020, 1, 1, 1, 1, 0),
                    datetime(2020, 1, 1, 1, 2, 0),
                    datetime(2020, 1, 2, 0, 1, 0),
                    datetime(2020, 1, 2, 0, 2, 0),
                    datetime(2020, 1, 2, 1, 1, 0),
                    datetime(2020, 1, 2, 1, 2, 0),
                    datetime(2021, 1, 1, 0, 1, 0),
                    datetime(2021, 1, 1, 0, 2, 0),
                    datetime(2021, 1, 1, 1, 1, 0),
                    datetime(2021, 1, 1, 1, 2, 0),
                    datetime(2021, 1, 2, 0, 1, 0),
                    datetime(2021, 1, 2, 0, 2, 0),
                    datetime(2021, 1, 2, 1, 1, 0),
                    datetime(2021, 1, 2, 1, 2, 0),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                "min": [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
                "max": [9, 10, 11, 12, 13, 14, 15, 16, 9, 10, 11, 12, 13, 14, 15, 16],
            }
        )

        tf = self.create_timeframe(period=Period.of_minutes(1), df=df)

        actual_df = calculate_min_max_envelope(tf)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)

    def test_calculate_min_max_envelope_second_resolution(self) -> None:
        """Check the min max envelope is calculated correctly for second resolution data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 1, 0),
                    datetime(2020, 1, 1, 0, 1, 1),
                    datetime(2020, 1, 1, 1, 1, 0),
                    datetime(2020, 1, 1, 1, 1, 1),
                    datetime(2020, 1, 1, 1, 2, 0),
                    datetime(2020, 1, 1, 1, 2, 1),
                    datetime(2021, 1, 1, 0, 1, 0),
                    datetime(2021, 1, 1, 0, 1, 1),
                    datetime(2021, 1, 1, 1, 1, 0),
                    datetime(2021, 1, 1, 1, 1, 1),
                    datetime(2021, 1, 1, 1, 2, 0),
                    datetime(2021, 1, 1, 1, 2, 1),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 1, 0),
                    datetime(2020, 1, 1, 0, 1, 1),
                    datetime(2020, 1, 1, 1, 1, 0),
                    datetime(2020, 1, 1, 1, 1, 1),
                    datetime(2020, 1, 1, 1, 2, 0),
                    datetime(2020, 1, 1, 1, 2, 1),
                    datetime(2021, 1, 1, 0, 1, 0),
                    datetime(2021, 1, 1, 0, 1, 1),
                    datetime(2021, 1, 1, 1, 1, 0),
                    datetime(2021, 1, 1, 1, 1, 1),
                    datetime(2021, 1, 1, 1, 2, 0),
                    datetime(2021, 1, 1, 1, 2, 1),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "min": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
                "max": [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12],
            }
        )

        tf = self.create_timeframe(period=Period.of_seconds(1), df=df)

        actual_df = calculate_min_max_envelope(tf)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)

    def test_calculate_min_max_envelope_microsecond_resolution(self) -> None:
        """Check the min max envelope is calculated correctly for microsecond resolution data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 1, 0, 0),
                    datetime(2020, 1, 1, 0, 1, 0, 1),
                    datetime(2020, 1, 1, 0, 1, 1, 0),
                    datetime(2020, 1, 1, 0, 1, 1, 1),
                    datetime(2021, 1, 1, 0, 1, 0, 0),
                    datetime(2021, 1, 1, 0, 1, 0, 1),
                    datetime(2022, 1, 1, 0, 1, 1, 0),
                    datetime(2022, 1, 1, 0, 1, 1, 1),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1, 0, 1, 0, 0),
                    datetime(2020, 1, 1, 0, 1, 0, 1),
                    datetime(2020, 1, 1, 0, 1, 1, 0),
                    datetime(2020, 1, 1, 0, 1, 1, 1),
                    datetime(2021, 1, 1, 0, 1, 0, 0),
                    datetime(2021, 1, 1, 0, 1, 0, 1),
                    datetime(2022, 1, 1, 0, 1, 1, 0),
                    datetime(2022, 1, 1, 0, 1, 1, 1),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
                "min": [1, 2, 3, 4, 1, 2, 3, 4],
                "max": [5, 6, 7, 8, 5, 6, 7, 8],
            }
        )

        tf = self.create_timeframe(period=Period.of_microseconds(1), df=df)

        actual_df = calculate_min_max_envelope(tf)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)

    def test_calculate_min_max_envelope_daily_observation_interval(self) -> None:
        """Check observation_interval logic works correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2022, 1, 5),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                ],
                "value": [1, 2, 3],
                "min": [1, 2, 3],
                "max": [1, 2, 3],
            }
        )

        tf = self.create_timeframe(period=Period.of_days(1), df=df)
        observation_interval = [datetime(2020, 1, 1), datetime(2020, 4, 1)]

        actual_df = calculate_min_max_envelope(tf, observation_interval=observation_interval)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)

    def test_observation_interval_outside_bounds(self) -> None:
        """Check an error is raised when the observation_interval is outside the timeseries range."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1),
                    datetime(2020, 1, 2),
                    datetime(2020, 1, 3),
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2022, 1, 5),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        tf = self.create_timeframe(period=Period.of_days(1), df=df)
        observation_interval = [datetime(2025, 1, 1), datetime(2025, 4, 1)]

        expected_error = "The observation interval is outside the bounds of the TimeFrame."
        with pytest.raises(ValueError, match=expected_error):
            calculate_min_max_envelope(tf, observation_interval=observation_interval)

    def test_daily_min_max_feb_29(self) -> None:
        """Check leap years are handled correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 2, 27),
                    datetime(2020, 2, 28),
                    datetime(2020, 2, 29),
                    datetime(2021, 2, 27),
                    datetime(2021, 2, 28),
                    datetime(2022, 2, 27),
                    datetime(2022, 2, 28),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7],
            }
        )

        expected_df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 2, 27),
                    datetime(2020, 2, 28),
                    datetime(2020, 2, 29),
                    datetime(2021, 2, 27),
                    datetime(2021, 2, 28),
                    datetime(2022, 2, 27),
                    datetime(2022, 2, 28),
                ],
                "value": [1, 2, 3, 4, 5, 6, 7],
                "min": [1, 2, 3, 1, 2, 1, 2],
                "max": [6, 7, 3, 6, 7, 6, 7],
            }
        )

        tf = self.create_timeframe(period=Period.of_days(1), df=df)

        actual_df = calculate_min_max_envelope(tf)

        assert_frame_equal(expected_df, actual_df, check_column_order=False)
