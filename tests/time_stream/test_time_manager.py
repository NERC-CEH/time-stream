import logging
import re
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from time_stream import Period
from time_stream.enums import DuplicateOption, TimeAnchor, ValidationErrorOptions
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


invalid_resolution_test_cases = (
    pytest.param(
        [
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 1, 0, 30, 0),
            datetime(2020, 1, 1, 1, 0, 0),
            datetime(2021, 1, 1, 0, 0, 0),  # Gap of 1 year
            datetime(2021, 1, 1, 0, 1, 0),  # 1 min
            datetime(2021, 1, 1, 0, 2, 0),  # 1 min
            datetime(2021, 1, 1, 0, 3, 0),  # 1 min
            datetime(2021, 1, 1, 0, 30, 0),
            datetime(2021, 1, 1, 1, 0, 0),
            datetime(2021, 1, 2, 0, 0, 0),  # Gap of 1 day
            datetime(2021, 1, 2, 0, 30, 0),
            datetime(2021, 1, 2, 1, 0, 0),
        ],
        Period.of_minutes(30),
        ["2021-01-01 00:01:00", "2021-01-01 00:02:00", "2021-01-01 00:03:00"],
        id="complex_with_gaps",
    ),
    pytest.param(
        [
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 1, 0, 30, 0),
            datetime(2020, 1, 1, 1, 0, 0),
            datetime(2020, 1, 1, 1, 30, 0),
            datetime(2020, 1, 1, 1, 31, 0),  # 1min
            datetime(2020, 1, 1, 1, 32, 0),  # 1min
            datetime(2020, 1, 1, 1, 33, 0),  # 1min
            datetime(2020, 1, 1, 2, 0, 0),
            datetime(2020, 1, 1, 2, 30, 0),
        ],
        Period.of_minutes(30),
        ["2020-01-01 01:31:00", "2020-01-01 01:32:00", "2020-01-01 01:33:00"],
        id="multiple_time_resolutions",
    ),
    pytest.param(
        [
            datetime(2020, 1, 1, 0, 0, 0),
            datetime(2020, 1, 1, 0, 30, 0),
            datetime(2020, 1, 1, 1, 0, 0),
            datetime(2020, 1, 1, 1, 30, 0),
            datetime(2020, 1, 1, 1, 31, 0),  # 1min
            datetime(2020, 1, 1, 2, 0, 0),
            datetime(2020, 1, 1, 2, 30, 0),
        ],
        Period.of_minutes(30),
        ["2020-01-01 01:31:00"],
        id="single_invalid_row",
    ),
    pytest.param(
        [
            datetime(2020, 1, 1),  # Daily
            datetime(2020, 1, 2),  # Daily
            datetime(2020, 1, 3),  # Daily
            datetime(2020, 2, 1),  # Monthly
            datetime(2020, 3, 1),  # Monthly
        ],
        Period.of_days(1),
        ["2020-01-01 01:31:00"],
        id="P1M present in P1D",
    ),
    pytest.param(
        [
            datetime(2020, 1, 1),  # Daily
            datetime(2020, 1, 2),  # Daily
            datetime(2020, 1, 3),  # Daily
            datetime(2020, 2, 1),  # Monthly
            datetime(2020, 3, 1),  # Monthly
        ],
        Period.of_months(1),
        ["2020-01-01 01:31:00"],
        id="P1D present in P1M",
    ),
)


class TestTimeManagerResolutionContiguity:
    def make_df_from_period(self, period: Period, length: int) -> pl.DataFrame:
        df = pl.DataFrame(
            {
                "timestamp": [period.datetime(period.ordinal(datetime(2025, 1, 1)) + i) for i in range(length)],
                "value": list(range(length)),
            }
        )
        return df

    @pytest.mark.parametrize(
        "period, length",
        [
            (Period.of_years(1), 10),
            (Period.of_months(1), 48),
            (Period.of_days(1), 366),
            (Period.of_hours(1), 72),
            (Period.of_minutes(30), 2880),
        ],
        ids=["P1Y", "P1M", "P1D", "PT1H", "PT30M"],
    )
    def test_contiguous_time_series_no_gaps(self, period: Period, length: int) -> None:
        """Test validation passes when the time series is continuously the same resolution with no gaps."""
        df = self.make_df_from_period(period=period, length=length)

        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period)

        try:
            time_manager._check_time_resolution_contiguity(df)
        except ResolutionError as err:
            pytest.fail(f"No ResolutionError was expected to be raised. Error:{str(err)}")

    def test_valid_PT30M_resolution_with_yearly_gaps(self) -> None:
        """No errors should be raised for gaps within valid PT30M data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2021, 1, 1, 0, 0, 0),
                    datetime(2021, 1, 1, 0, 30, 0),
                    datetime(2021, 1, 1, 1, 0, 0),  # Year gap
                    datetime(2022, 1, 1, 0, 0, 0),
                    datetime(2022, 1, 1, 0, 30, 0),
                    datetime(2022, 1, 1, 1, 0, 0),
                    datetime(2023, 1, 1, 0, 0, 0),  # Year gap
                    datetime(2023, 1, 1, 0, 30, 0),
                    datetime(2023, 1, 1, 1, 0, 0),
                ]
            }
        )
        period = Period.of_minutes(30)
        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period)
        try:
            time_manager._check_time_resolution_contiguity(df)
        except ResolutionError as err:
            pytest.fail(f"No ResolutionError was expected to be raised. Error:{str(err)}")

    def test_valid_monthly_resolution_with_yearly_gaps(self) -> None:
        """No error should be raised for gaps within valid monthly resolution data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2021, 1, 1),
                    datetime(2021, 2, 1),
                    datetime(2021, 3, 1),  # Year gap
                    datetime(2022, 1, 1),
                    datetime(2022, 2, 1),
                    datetime(2022, 3, 1),
                    datetime(2023, 1, 1),  # Year gap
                    datetime(2023, 2, 1),
                    datetime(2023, 3, 1),
                ]
            }
        )
        period = Period.of_months(1)
        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period)
        try:
            time_manager._check_time_resolution_contiguity(df)
        except ResolutionError as err:
            pytest.fail(f"No ResolutionError was expected to be raised. Error:{str(err)}")

    @pytest.mark.parametrize("input_timestamps, period, error_dates", invalid_resolution_test_cases)
    def test_invalid_with_error(self, input_timestamps: list[datetime], period: Period, error_dates: list[str]) -> None:
        """An error should be raised for the invalid data."""
        df = pl.DataFrame({"timestamp": input_timestamps})
        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period)

        expected_error = f"The following timestamps do not conform to the expected resolution of PT30M: `{error_dates}`"
        with pytest.raises(ResolutionError, match=re.escape(expected_error)):
            time_manager._check_time_resolution_contiguity(df)

    @pytest.mark.parametrize("input_timestamps, period, error_dates", invalid_resolution_test_cases)
    def test_invalid_with_warn(
        self, input_timestamps: list[datetime], period: Period, error_dates: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning should be logged for the invalid data."""
        df = pl.DataFrame({"timestamp": input_timestamps})
        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period)

        expected_log_message = (
            f"The following timestamps do not conform to the expected resolution of PT30M: `{error_dates}`"
        )
        with caplog.at_level(logging.INFO):
            time_manager._check_time_resolution_contiguity(df, error_method=ValidationErrorOptions.WARN)

            assert caplog.messages[0] == expected_log_message

    @pytest.mark.parametrize("input_timestamps, period, error_dates", invalid_resolution_test_cases)
    def test_invalid_with_resolve(
        self, input_timestamps: list[datetime], period: Period, error_dates: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        """The invalid data should be removed with a log message to indicate which rows are considered invalid."""
        df = pl.DataFrame({"timestamp": input_timestamps})
        time_manager = TimeManager(time_name="timestamp", resolution=period, offset=None, periodicity=period)

        expected_log_message = (
            f"Removing the following timestamps which were found to not conform to the expected resolution of "
            f"PT30M: `{error_dates}`"
        )

        timestamps_to_remove = [datetime.strptime(item, "%Y-%m-%d %H:%M:%S") for item in error_dates]
        expected_df = df.filter(~pl.col("timestamp").is_in(timestamps_to_remove))
        with caplog.at_level(logging.INFO):
            actual_df = time_manager._check_time_resolution_contiguity(df, error_method=ValidationErrorOptions.RESOLVE)

            assert caplog.messages[0] == expected_log_message

        assert_frame_equal(expected_df, actual_df)

    def test_no_initial_resolution_valid(self) -> None:
        """Check a TimeManager instance with no initial resolution is handled correctly.

        If no resolution period is provided then the TimeManager will be an initialised with a resolution of 1
        microsecond causing the validation to pass by default.

        """
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "value": [1, 2, 3],
            }
        )

        time_manager = TimeManager(time_name="timestamp", resolution=None, offset=None, periodicity=None)
        try:
            time_manager._check_time_resolution_contiguity(df)
        except ResolutionError as err:
            pytest.fail(f"No ResolutionError was expected to be raised. Error:{str(err)}")

    def test_no_initial_resolution_invalid(self) -> None:
        """Check a TimeManager instance with no initial resolution is handled correctly.

        If no resolution period is provided then the TimeManager will be an initialised with a resolution of 1
        microsecond causing the validation to pass by default even if invalid rows are present.

        """
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2020, 1, 1),  # Daily
                    datetime(2020, 1, 2),  # Daily
                    datetime(2020, 1, 3),  # Daily
                    datetime(2020, 2, 1),  # Monthly
                    datetime(2020, 3, 1),  # Monthly
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

        time_manager = TimeManager(time_name="timestamp", resolution=None, offset=None, periodicity=None)
        try:
            time_manager._check_time_resolution_contiguity(df)
        except ResolutionError as err:
            pytest.fail(f"No ResolutionError was expected to be raised. Error:{str(err)}")
