import re
from datetime import datetime

import polars as pl
import pytest
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



@pytest.fixture
def tm() -> TimeManager:
    tm = object.__new__(TimeManager)  # skips __init__
    tm._resolution = None
    tm._offset = None
    tm._periodicity = None
    tm._time_anchor = TimeAnchor.START
    tm._time_name = "time"

    return tm


class TestValidateAlignment:
    """Test the _validate_alignment method. Note the main functionality is more thoroughly tested in the
    utils function `check_alignment`.
    """

    def test_validate_alignment_success(self, tm: TimeManager) -> None:
        """Test that a correct alignment to time series passes the validation."""
        tm._resolution = Period.of_years(1)
        tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)])
        tm._validate_alignment(times)

    def test_validate_alignment_fails(self, tm: TimeManager) -> None:
        """Test that an incorrect alignment to time series fails the validation."""
        tm._resolution = Period.of_years(1)
        tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 6, 1), datetime(2022, 1, 1)])

        expected_error = "Time values are not aligned to resolution[+offset]: P1Y"

        with pytest.raises(ResolutionError, match=re.escape(expected_error)):
            tm._validate_alignment(times)

    def test_validate_alignment_with_offset_success(self, tm: TimeManager) -> None:
        """Test that a correct alignment to time series passes the validation, when an offset has been given."""
        tm._resolution = Period.of_years(1)
        tm._offset = "+T9H"
        tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1, 9), datetime(2021, 1, 1, 9), datetime(2022, 1, 1, 9)])

        tm._validate_alignment(times)

    def test_validate_alignment_with_offset_fails(self, tm: TimeManager) -> None:
        """Test that an incorrect alignment to time series fails the validation, when an offset has been given."""
        tm._resolution = Period.of_years(1)
        tm._offset = "+T9H"
        tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1, 9, 1), datetime(2021, 1, 1, 9), datetime(2022, 1, 1, 9)])

        expected_error = "Time values are not aligned to resolution[+offset]: P1Y+T9H"

        with pytest.raises(ResolutionError, match=re.escape(expected_error)):
            tm._validate_alignment(times)


class TestValidatePeriodicity:
    """Test the _validate_periodicity method. Note the main functionality is more thoroughly tested in the
    utils function `check_periodicity`.
    """

    def test_validate_periodicity_success(self, tm: TimeManager) -> None:
        """Test that a correct periodicity to time series passes the validation."""
        tm._periodicity = Period.of_years(1)
        tm._configure_period_properties()

        # 1 value per year allowed
        times = pl.Series([datetime(2020, 1, 1), datetime(2021, 7, 19, 5), datetime(2022, 12, 25, 9, 30)])
        tm._validate_periodicity(times)

    def test_validate_periodicity_fails(self, tm: TimeManager) -> None:
        """Test that an incorrect periodicity to time series fails the validation."""
        tm._periodicity = Period.of_years(1)
        tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 6, 1), datetime(2022, 1, 1)])

        expected_error = "Time values do not conform to periodicity: P1Y"

        with pytest.raises(PeriodicityError, match=expected_error):
            tm._validate_periodicity(times)

    def test_validate_periodicity_with_offset_success(self, tm: TimeManager) -> None:
        """Test that a correct periodicity to time series passes the validation, when an offset has been given."""
        tm._periodicity = Period.of_years(1).with_month_offset(3)
        tm._configure_period_properties()

        # 1 value per year, from April to April, allowed
        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 4, 1, 1), datetime(2021, 12, 9, 23)])

        tm._validate_periodicity(times)

    def test_validate_periodicity_with_offset_fails(self, tm: TimeManager) -> None:
        """Test that an incorrect periodicity to time series fails the validation, when an offset has been given."""
        tm._periodicity = Period.of_years(1).with_month_offset(3)
        tm._configure_period_properties()

        times = pl.Series([datetime(2020, 1, 1), datetime(2020, 3, 1, 1), datetime(2021, 12, 9, 23)])

        expected_error = "Time values do not conform to periodicity: P1Y+3M"
        with pytest.raises(PeriodicityError, match=re.escape(expected_error)):
            tm._validate_periodicity(times)


class TestValidateTimeColumn:
    df = pl.DataFrame(
        {"time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], "data_column": [1, 2, 3]}
    )

    def test_validate_missing_time_column(self) -> None:
        """Test error raised if DataFrame is missing the time column"""
        invalid_df = pl.DataFrame(self.df["data_column"])

        tm = object.__new__(TimeManager)  # skips __init__
        tm._time_name = "time"

        expected_error = f"Time column 'time' not found in DataFrame. Available columns: {list(invalid_df.columns)}"
        with pytest.raises(ColumnNotFoundError, match=re.escape(expected_error)):
            tm._validate_time_column(invalid_df)

    def test_validate_time_column_not_temporal(self) -> None:
        """Test error raised if time column is not temporal."""
        invalid_df = pl.DataFrame(self.df["data_column"])

        tm = object.__new__(TimeManager)  # skips __init__
        tm._time_name = "data_column"

        expected_error = "Time column 'data_column' must be datetime type, got 'Int64'"
        with pytest.raises(ColumnTypeError, match=expected_error):
            tm._validate_time_column(invalid_df)


class TestHandleTimeDuplicates:
    # A dataframe with some duplicate times
    df = pl.DataFrame(
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

    # self.tm = object.__new__(TimeManager)  # skips __init__
    # self.tm._time_name = "time"

    def test_error(self, tm: TimeManager) -> None:
        """Test that the error strategy works as expected"""
        with pytest.raises(DuplicateTimeError):
            tm._on_duplicates = DuplicateOption.ERROR
            tm._handle_time_duplicates(self.df)

    def test_keep_first(self, tm: TimeManager) -> None:
        """Test that the keep first strategy works as expected"""
        tm._on_duplicates = DuplicateOption.KEEP_FIRST
        result = tm._handle_time_duplicates(self.df)

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

    def test_keep_last(self, tm: TimeManager) -> None:
        """Test that the keep last strategy works as expected"""
        tm._on_duplicates = DuplicateOption.KEEP_LAST
        result = tm._handle_time_duplicates(self.df)

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

    def test_drop(self, tm: TimeManager) -> None:
        """Test that the drop strategy works as expected"""
        tm._on_duplicates = DuplicateOption.DROP
        result = tm._handle_time_duplicates(self.df)

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

    def test_merge(self, tm: TimeManager) -> None:
        """Test that the merge strategy works as expected"""
        tm._on_duplicates = DuplicateOption.MERGE
        result = tm._handle_time_duplicates(self.df)

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


class TestCheckTimeIntegrity:
    df = pl.DataFrame(
        {"time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 4)]}
    )
    tm = TimeManager("time", "P1D")

    def test_same_time_values(self) -> None:
        """Test that no changes to the time column is valid"""
        self.tm._check_time_integrity(self.df, self.df.clone())

    def test_different_time_values(self) -> None:
        """Test that no changes to the time column is valid"""
        new_df = pl.DataFrame(
            {"time": [datetime(1990, 1, 1), datetime(1990, 1, 2), datetime(1990, 1, 3), datetime(1990, 1, 4)]}
        )
        with pytest.raises(TimeMutatedError):
            self.tm._check_time_integrity(self.df, new_df)


class TestConfigurePeriodProperties:
    def test_resolution_str_no_offset_defaults_periodicity_to_alignment(self, tm: TimeManager) -> None:
        tm._resolution = "PT15M"
        tm._configure_period_properties()

        assert isinstance(tm._resolution, Period)
        assert isinstance(tm._alignment, Period)
        assert isinstance(tm._periodicity, Period)

        assert tm._resolution == Period.of_minutes(15)
        assert tm._offset is None
        assert tm._alignment == Period.of_minutes(15)
        assert tm._periodicity == Period.of_minutes(15)

    def test_offset_and_default_periodicity(self, tm: TimeManager) -> None:
        tm._resolution = "P1D"
        tm._offset = "+T9H"
        tm._configure_period_properties()

        assert isinstance(tm._resolution, Period)
        assert isinstance(tm._offset, str)
        assert isinstance(tm._alignment, Period)
        assert isinstance(tm._periodicity, Period)

        assert tm._resolution == Period.of_days(1)
        assert tm._offset == "+T9H"
        assert tm._alignment == Period.of_days(1).with_hour_offset(9)
        assert tm._periodicity == Period.of_days(1).with_hour_offset(9)

    def test_override_periodicity(self, tm: TimeManager) -> None:
        tm._resolution = "P1D"
        tm._periodicity = "P1Y+9MT9H"
        tm._configure_period_properties()

        assert isinstance(tm._resolution, Period)
        assert isinstance(tm._alignment, Period)
        assert isinstance(tm._periodicity, Period)

        assert tm._resolution == Period.of_days(1)
        assert tm._offset is None
        assert tm._alignment == Period.of_days(1)
        assert tm._periodicity == Period.of_years(1).with_month_offset(9).with_hour_offset(9)

    def test_defaults(self, tm: TimeManager) -> None:
        tm._configure_period_properties()

        assert isinstance(tm._resolution, Period)
        assert tm._offset is None
        assert isinstance(tm._alignment, Period)
        assert isinstance(tm._periodicity, Period)

        assert tm._resolution == Period.of_microseconds(1)
        assert tm._alignment == Period.of_microseconds(1)
        assert tm._periodicity == Period.of_microseconds(1)

    def test_non_iso_standard_resolution_string_raises(self, tm: TimeManager) -> None:
        """Periods can be created with a modified iso string specified an offset. We want the resolution parameter to
        be a "non-offset" period (which we can apply a specified offset too later).
        """
        tm._resolution = "P1D+9H"
        with pytest.raises(PeriodParsingError):
            tm._configure_period_properties()

    def test_explicit_resolution_with_offset_raises(self, tm: TimeManager) -> None:
        """Periods can have an offset. We want the resolution parameter to be a "non-offset" period
        (which we can apply a specified offset too later).
        """
        tm._resolution = Period.of_days(1).with_hour_offset(9)
        with pytest.raises(PeriodValidationError):
            tm._configure_period_properties()

    def test_non_offset_string_raises(self, tm: TimeManager) -> None:
        """The offset parameter should be provided as an offset string (e.g. +1D, +1DT9H, etc.)"""
        tm._offset = "P1D+9H"
        with pytest.raises(PeriodParsingError):
            tm._configure_period_properties()

    def test_invalid_resolution_type_raises(self, tm: TimeManager) -> None:
        tm._resolution = 123  # type: ignore
        with pytest.raises(TypeError):
            tm._configure_period_properties()

    def test_invalid_offset_type_raises(self, tm: TimeManager) -> None:
        tm._offset = Period.of_years(1)  # type: ignore
        with pytest.raises(TypeError):
            tm._configure_period_properties()

    def test_invalid_periodicity_type_raises(self, tm: TimeManager) -> None:
        tm._periodicity = 123  # type: ignore
        with pytest.raises(TypeError):
            tm._configure_period_properties()

    def test_invalid_resolution_string_raises(self, tm: TimeManager) -> None:
        tm._resolution = "NOT_A_PERIOD"
        with pytest.raises(PeriodParsingError):
            tm._configure_period_properties()

    def test_invalid_offset_string_raises(self, tm: TimeManager) -> None:
        tm._offset = "NOT_A_PERIOD"
        with pytest.raises(PeriodParsingError):
            tm._configure_period_properties()

    def test_invalid_periodicity_string_raises(self, tm: TimeManager) -> None:
        tm._periodicity = "NOT_A_PERIOD"
        with pytest.raises(PeriodParsingError):
            tm._configure_period_properties()

class TestTimeManagerResolutionContiguity:
    def test_contiguous_time_series(self)-> None:
        """Test validation passes when the time series is continuously the same resolution."""
        pass

    def test_multiple_time_resolutions(self) -> None:
        # PT30M with PT1M section within it

        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020,1,1,0,0,0),
                    datetime(2020,1,1,0,30,0),
                    datetime(2020,1,1,1,0,0),
                    datetime(2020,1,1,1,30,0),
                    datetime(2020,1,1,1,31,0),
                    datetime(2020,1,1,1,32,0),
                    datetime(2020,1,1,1,33,0),
                    datetime(2020,1,1,2,0,0),
                    datetime(2020,1,1,2,30,0),
                ]
            }
        )
        period = Period.of_minutes(30)
        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period )
        time_manager._check_time_resolution_contiguity(df)

    