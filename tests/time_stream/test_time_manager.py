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
    PeriodParsingError,
    PeriodValidationError,
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
        self.tm._resolution = None
        self.tm._offset = None
        self.tm._periodicity = None
        self.tm._time_anchor = TimeAnchor.START

    def test_validate_alignment_success(self) -> None:
        """Test that a correct alignment to time series passes the validation."""
        self.tm._resolution = Period.of_years(1)
        self.tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)])
        self.tm._validate_alignment(times)

    def test_validate_alignment_fails(self) -> None:
        """Test that an incorrect alignment to time series fails the validation."""
        self.tm._resolution = Period.of_years(1)
        self.tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)])

        with self.assertRaises(ResolutionError) as err:
            self.tm._validate_alignment(times)
        self.assertEqual("Time values are not aligned to resolution[+offset]: P1Y", str(err.exception))

    def test_validate_alignment_with_offset_success(self) -> None:
        """Test that a correct alignment to time series passes the validation, when an offset has been given."""
        self.tm._resolution = Period.of_years(1)
        self.tm._offset = "+T9H"
        self.tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1, 9), datetime(2021, 1, 1, 9), datetime(2022, 1, 1, 9)])

        self.tm._validate_alignment(times)

    def test_validate_alignment_with_offset_fails(self) -> None:
        """Test that an incorrect alignment to time series fails the validation, when an offset has been given."""
        self.tm._resolution = Period.of_years(1)
        self.tm._offset = "+T9H"
        self.tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1, 9, 1), datetime(2021, 1, 1, 9), datetime(2022, 1, 1, 9)])

        with self.assertRaises(ResolutionError) as err:
            self.tm._validate_alignment(times)
        self.assertEqual("Time values are not aligned to resolution[+offset]: P1Y+T9H", str(err.exception))


class TestValidatePeriodicity(unittest.TestCase):
    """Test the _validate_periodicity method. Note the main functionality is more thoroughly tested in the
    utils function `check_periodicity`.
    """

    def setUp(self) -> None:
        self.tm = object.__new__(TimeManager)  # skips __init__
        self.tm._resolution = None
        self.tm._offset = None
        self.tm._periodicity = None
        self.tm._time_anchor = TimeAnchor.START

    def test_validate_periodicity_success(self) -> None:
        """Test that a correct periodicity to time series passes the validation."""
        self.tm._periodicity = Period.of_years(1)
        self.tm._configure_period_properties()

        # 1 value per year allowed
        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 7, 19, 5), datetime(2022, 12, 25, 9, 30)])
        self.tm._validate_periodicity(times)

    def test_validate_periodicity_fails(self) -> None:
        """Test that an incorrect periodicity to time series fails the validation."""
        self.tm._periodicity = Period.of_years(1)
        self.tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 6, 1), datetime(2022, 1, 1)])

        with self.assertRaises(PeriodicityError) as err:
            self.tm._validate_periodicity(times)
        self.assertEqual("Time values do not conform to periodicity: P1Y", str(err.exception))

    def test_validate_periodicity_with_offset_success(self) -> None:
        """Test that a correct periodicity to time series passes the validation, when an offset has been given."""
        self.tm._periodicity = Period.of_years(1).with_month_offset(3)
        self.tm._configure_period_properties()

        # 1 value per year, from April to April, allowed
        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 4, 1, 1), datetime(2021, 12, 9, 23)])

        self.tm._validate_periodicity(times)

    def test_validate_periodicity_with_offset_fails(self) -> None:
        """Test that an incorrect periodicity to time series fails the validation, when an offset has been given."""
        self.tm._periodicity = Period.of_years(1).with_month_offset(3)
        self.tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 3, 1, 1), datetime(2021, 12, 9, 23)])

        with self.assertRaises(PeriodicityError) as err:
            self.tm._validate_periodicity(times)
        self.assertEqual("Time values do not conform to periodicity: P1Y+3M", str(err.exception))


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
        self.tm = TimeManager("time", "P1D")

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


class TestConfigurePeriodProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.tm = object.__new__(TimeManager)  # skips __init__
        self.tm._resolution = None
        self.tm._offset = None
        self.tm._alignment = None
        self.tm._periodicity = None

    def test_resolution_str_no_offset_defaults_periodicity_to_alignment(self) -> None:
        self.tm._resolution = "PT15M"
        self.tm._configure_period_properties()

        self.assertIsInstance(self.tm._resolution, Period)
        self.assertIsInstance(self.tm._alignment, Period)
        self.assertIsInstance(self.tm._periodicity, Period)

        self.assertEqual(self.tm._resolution, Period.of_minutes(15))
        self.assertIsNone(self.tm._offset)
        self.assertEqual(self.tm._alignment, Period.of_minutes(15))
        self.assertEqual(self.tm._periodicity, Period.of_minutes(15))

    def test_offset_and_default_periodicity(self) -> None:
        self.tm._resolution = "P1D"
        self.tm._offset = "+T9H"
        self.tm._configure_period_properties()

        self.assertIsInstance(self.tm._resolution, Period)
        self.assertIsInstance(self.tm._offset, str)
        self.assertIsInstance(self.tm._alignment, Period)
        self.assertIsInstance(self.tm._periodicity, Period)

        self.assertEqual(self.tm._resolution, Period.of_days(1))
        self.assertEqual(self.tm._offset, "+T9H")
        self.assertEqual(self.tm._alignment, Period.of_days(1).with_hour_offset(9))
        self.assertEqual(self.tm._periodicity, Period.of_days(1).with_hour_offset(9))

    def test_override_periodicity(self) -> None:
        self.tm._resolution = "P1D"
        self.tm._periodicity = "P1Y+9MT9H"
        self.tm._configure_period_properties()

        self.assertIsInstance(self.tm._resolution, Period)
        self.assertIsInstance(self.tm._alignment, Period)
        self.assertIsInstance(self.tm._periodicity, Period)

        self.assertEqual(self.tm._resolution, Period.of_days(1))
        self.assertIsNone(self.tm._offset)
        self.assertEqual(self.tm._alignment, Period.of_days(1))
        self.assertEqual(self.tm._periodicity, Period.of_years(1).with_month_offset(9).with_hour_offset(9))

    def test_defaults(self) -> None:
        self.tm._configure_period_properties()

        self.assertIsInstance(self.tm._resolution, Period)
        self.assertIsNone(self.tm._offset)
        self.assertIsInstance(self.tm._alignment, Period)
        self.assertIsInstance(self.tm._periodicity, Period)

        self.assertEqual(self.tm._resolution, Period.of_microseconds(1))
        self.assertEqual(self.tm._alignment, Period.of_microseconds(1))
        self.assertEqual(self.tm._periodicity, Period.of_microseconds(1))

    def test_non_iso_standard_resolution_string_raises(self) -> None:
        """Periods can be created with a modified iso string specified an offset. We want the resolution parameter to
        be a "non-offset" period (which we can apply a specified offset too later).
        """
        self.tm._resolution = "P1D+9H"
        with self.assertRaises(PeriodParsingError):
            self.tm._configure_period_properties()

    def test_explicit_resolution_with_offset_raises(self) -> None:
        """Periods can have an offset. We want the resolution parameter to be a "non-offset" period
        (which we can apply a specified offset too later).
        """
        self.tm._resolution = Period.of_days(1).with_hour_offset(9)
        with self.assertRaises(PeriodValidationError):
            self.tm._configure_period_properties()

    def test_non_offset_string_raises(self) -> None:
        """The offset parameter should be provided as an offset string (e.g. +1D, +1DT9H, etc.)"""
        self.tm._offset = "P1D+9H"
        with self.assertRaises(PeriodParsingError):
            self.tm._configure_period_properties()

    def test_invalid_resolution_type_raises(self) -> None:
        self.tm._resolution = 123  # type: ignore
        with self.assertRaises(TypeError):
            self.tm._configure_period_properties()

    def test_invalid_offset_type_raises(self) -> None:
        self.tm._offset = Period.of_years(1)  # type: ignore
        with self.assertRaises(TypeError):
            self.tm._configure_period_properties()

    def test_invalid_periodicity_type_raises(self) -> None:
        self.tm._periodicity = 123  # type: ignore
        with self.assertRaises(TypeError):
            self.tm._configure_period_properties()

    def test_invalid_resolution_string_raises(self) -> None:
        self.tm._resolution = "NOT_A_PERIOD"
        with self.assertRaises(PeriodParsingError):
            self.tm._configure_period_properties()

    def test_invalid_offset_string_raises(self) -> None:
        self.tm._offset = "NOT_A_PERIOD"
        with self.assertRaises(PeriodParsingError):
            self.tm._configure_period_properties()

    def test_invalid_periodicity_string_raises(self) -> None:
        self.tm._periodicity = "NOT_A_PERIOD"
        with self.assertRaises(PeriodParsingError):
            self.tm._configure_period_properties()
