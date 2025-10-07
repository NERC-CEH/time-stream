import unittest
from datetime import datetime

import polars as pl
from polars.testing import assert_frame_equal

from time_stream import Period
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import (
    ColumnNotFoundError,
    ColumnTypeError,
    DuplicateTimeError,
    PeriodicityError,
    ResolutionError,
    TimeMutatedError,
)
from time_stream.time_manager import TimeManager


class TestValidateAlignment(unittest.TestCase):
    """Test the _validate_alignment method. Note the main functionality is more thoroughly tested in the
    utils function `check_alignment`.
    """

    def setUp(self) -> None:
        self.tm = object.__new__(TimeManager)  # skips __init__
        self.tm._resolution = Period.of_years(1)
        self.tm._offset = None
        self.tm._periodicity = None
        self.tm._time_anchor = TimeAnchor.START
        self.tm._configure_period_properties()

    def test_validate_alignment_success(self) -> None:
        """Test that a correct alignment to time series passes the validation."""
        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)])
        self.tm._validate_alignment(times)

    def test_validate_alignment_fails(self) -> None:
        """Test that an incorrect alignment to time series fails the validation."""
        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)])

        with self.assertRaises(ResolutionError) as err:
            self.tm._validate_alignment(times)
        self.assertEqual("Time values are not aligned to resolution[+offset]: P1Y", str(err.exception))


class TestValidatePeriodicity(unittest.TestCase):
    """Test the _validate_periodicity method. Note the main functionality is more thoroughly tested in the
    utils function `check_periodicity`.
    """

    def test_validate_periodicity_success(self) -> None:
        """Test that a correct periodicity to time series passes the validation."""
        tm = object.__new__(TimeManager)  # skips __init__
        tm._periodicity = Period.of_years(1)
        tm._time_anchor = TimeAnchor.START

        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)])
        tm._validate_periodicity(times)

    def test_validate_periodicity_fails(self) -> None:
        """Test that an incorrect periodicity to time series fails the validation."""
        tm = object.__new__(TimeManager)  # skips __init__
        tm._periodicity = Period.of_years(1)
        tm._time_anchor = TimeAnchor.START

        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 6, 1), datetime(2022, 1, 1)])

        with self.assertRaises(PeriodicityError) as err:
            tm._validate_periodicity(times)
        self.assertEqual(f"Time values do not conform to periodicity: {tm._periodicity}", str(err.exception))


class TestValidateTimeColumn(unittest.TestCase):
    df = pl.DataFrame(
        {"time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], "data_column": [1, 2, 3]}
    )

    def test_validate_missing_time_column(self) -> None:
        """Test error raised if DataFrame is missing the time column"""
        invalid_df = pl.DataFrame(self.df["data_column"])

        tm = object.__new__(TimeManager)  # skips __init__
        tm._time_name = "time"

        with self.assertRaises(ColumnNotFoundError) as err:
            tm._validate_time_column(invalid_df)
        self.assertEqual(
            f"Time column 'time' not found in DataFrame. Available columns: {list(invalid_df.columns)}",
            str(err.exception),
        )

    def test_validate_time_column_not_temporal(self) -> None:
        """Test error raised if time column is not temporal."""
        invalid_df = pl.DataFrame(self.df["data_column"])

        tm = object.__new__(TimeManager)  # skips __init__
        tm._time_name = "data_column"

        with self.assertRaises(ColumnTypeError) as err:
            tm._validate_time_column(invalid_df)
        self.assertEqual("Time column 'data_column' must be datetime type, got 'Int64'", str(err.exception))


class TestHandleTimeDuplicates(unittest.TestCase):
    def setUp(self) -> None:
        # A dataframe with some duplicate times
        self.df = pl.DataFrame(
            {
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
                    datetime(2024, 1, 10),
                ],
                "colA": [1, None, 2, 3, 4, 5, 6, 7, None, 8, 8, 9, 10],
                "colB": [None, 1, 2, 3, 4, 5, 6, 7, 9, None, 9, 9, 10],
                "colC": [None, 2, 2, 3, 4, 5, 6, 7, 8, 9, None, 9, 10],
            }
        )
        self.new_df = pl.DataFrame()

        self.tm = object.__new__(TimeManager)  # skips __init__
        self.tm._time_name = "time"

    def test_error(self) -> None:
        """Test that the error strategy works as expected"""
        with self.assertRaises(DuplicateTimeError):
            self.tm._on_duplicates = DuplicateOption.ERROR
            self.tm._handle_time_duplicates(self.df)

    def test_keep_first(self) -> None:
        """Test that the keep first strategy works as expected"""
        self.tm._on_duplicates = DuplicateOption.KEEP_FIRST
        result = self.tm._handle_time_duplicates(self.df)

        expected = pl.DataFrame(
            {
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
                    datetime(2024, 1, 10),
                ],
                "colA": [1, 2, 3, 4, 5, 6, 7, None, 9, 10],
                "colB": [None, 2, 3, 4, 5, 6, 7, 9, 9, 10],
                "colC": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        assert_frame_equal(result, expected)

    def test_keep_last(self) -> None:
        """Test that the keep last strategy works as expected"""
        self.tm._on_duplicates = DuplicateOption.KEEP_LAST
        result = self.tm._handle_time_duplicates(self.df)

        expected = pl.DataFrame(
            {
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
                    datetime(2024, 1, 10),
                ],
                "colA": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "colB": [1, 2, 3, 4, 5, 6, 7, 9, 9, 10],
                "colC": [2, 2, 3, 4, 5, 6, 7, None, 9, 10],
            }
        )

        assert_frame_equal(result, expected)

    def test_drop(self) -> None:
        """Test that the drop strategy works as expected"""
        self.tm._on_duplicates = DuplicateOption.DROP
        result = self.tm._handle_time_duplicates(self.df)

        expected = pl.DataFrame(
            {
                "time": [
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 4),
                    datetime(2024, 1, 5),
                    datetime(2024, 1, 6),
                    datetime(2024, 1, 7),
                    datetime(2024, 1, 9),
                    datetime(2024, 1, 10),
                ],
                "colA": [2, 3, 4, 5, 6, 7, 9, 10],
                "colB": [2, 3, 4, 5, 6, 7, 9, 10],
                "colC": [2, 3, 4, 5, 6, 7, 9, 10],
            }
        )

        assert_frame_equal(result, expected)

    def test_merge(self) -> None:
        """Test that the merge strategy works as expected"""
        self.tm._on_duplicates = DuplicateOption.MERGE
        result = self.tm._handle_time_duplicates(self.df)

        expected = pl.DataFrame(
            {
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
                    datetime(2024, 1, 10),
                ],
                "colA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "colB": [1, 2, 3, 4, 5, 6, 7, 9, 9, 10],
                "colC": [2, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )

        assert_frame_equal(result, expected)


class TestCheckTimeIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {"time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 4)]}
        )
        self.tm = TimeManager("time", "P1D", "P1D")

    def test_same_time_values(self) -> None:
        """Test that no changes to the time column is valid"""
        self.tm._check_time_integrity(self.df, self.df.clone())

    def test_different_time_values(self) -> None:
        """Test that no changes to the time column is valid"""
        new_df = pl.DataFrame(
            {"time": [datetime(1990, 1, 1), datetime(1990, 1, 2), datetime(1990, 1, 3), datetime(1990, 1, 4)]}
        )
        with self.assertRaises(TimeMutatedError):
            self.tm._check_time_integrity(self.df, new_df)
