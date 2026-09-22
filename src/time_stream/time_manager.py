"""
Time Management Module.

This module defines and enforces integrity rules for the temporal aspects of a TimeFrame object.  This includes:

- Validates the presence and type of the time column.
- Ensures datetimes align to the specified resolution (precision).
- Ensures datetimes conform to the specified periodicity (frequency).
- Handles duplicate timestamps according to a chosen strategy.
- Prevents mutation of time values between DataFrame operations.
"""

import logging
from copy import copy

import polars as pl
from isoperiod import Period, PeriodValidationError

from time_stream.exceptions import (
    ColumnNotFoundError,
    ColumnTypeError,
    DuplicateTimeError,
    DuplicateValueError,
    NullTimeValueError,
    PeriodicityError,
    ResolutionError,
    TimeMutatedError,
)
from time_stream.types import DuplicateOption, TimeAnchor, ValidationErrorOptions
from time_stream.utils import (
    check_alignment,
    check_literal_value,
    check_periodicity,
    configure_period_object,
    handle_duplicates,
    truncate_to_period,
)

logger = logging.getLogger(__name__)


class TimeManager:
    """Enforces integrity of the temporal aspects of the TimeFrame"""

    _time_name: str
    _resolution: Period
    _offset: str | None
    _alignment: Period
    _periodicity: Period
    _on_duplicates: DuplicateOption
    _on_misaligned_rows: ValidationErrorOptions
    _time_anchor: TimeAnchor

    def __init__(
        self,
        time_name: str,
        resolution: str | Period | None = None,
        offset: str | None = None,
        periodicity: str | Period | None = None,
        on_duplicates: DuplicateOption = "error",
        on_misaligned_rows: ValidationErrorOptions = "error",
        time_anchor: TimeAnchor = "start",
    ):
        """Initialise the time manager.

        Args:
            time_name: The name of the time column of the parent TimeFrame.
            resolution: Sampling interval for the timeseries.
            offset: Offset applied from the natural boundary of ``resolution`` to position the datetime values along the
                    timeline.
            periodicity: Defines the allowed "frequency" of datetimes in your timeseries, i.e., how many datetime
                         entries are allowed within a given period of time.
            on_duplicates: What to do if duplicate rows are found in the data.
            on_misaligned_rows: What to do if misaligned rows are found in the data.
            time_anchor: The time anchor to which the date/times conform to.
        """
        check_literal_value(time_anchor, TimeAnchor, "time_anchor")
        check_literal_value(on_duplicates, DuplicateOption, "on_duplicates")
        check_literal_value(on_misaligned_rows, ValidationErrorOptions, "on_misaligned_rows")

        self._time_name = time_name
        self._resolution = self._configure_resolution_property(resolution)
        self._offset = self._configure_offset_property(offset)
        self._alignment = self._configure_alignment_property(self._resolution, self._offset)
        self._periodicity = self._configure_periodicity_property(periodicity, self._alignment)
        self._on_duplicates = on_duplicates
        self._on_misaligned_rows = on_misaligned_rows
        self._time_anchor = time_anchor

    @property
    def time_name(self) -> str:
        return self._time_name

    @property
    def resolution(self) -> Period:
        return self._resolution

    @property
    def offset(self) -> str | None:
        return self._offset

    @property
    def alignment(self) -> Period:
        return self._alignment

    @property
    def periodicity(self) -> Period:
        return self._periodicity

    @property
    def time_anchor(self) -> TimeAnchor:
        return self._time_anchor

    @staticmethod
    def _configure_resolution_property(resolution: str | Period | None) -> Period:
        """Normalise and derive the period properties of ``resolution``

        Parses string into Period object. If none, default to 1 microsecond to allow any resolution.

        Args:
            resolution: Sampling interval for the timeseries.

        Returns:
            Resolution period object
        """
        if resolution is None:
            return Period.of_microseconds(1)
        elif isinstance(resolution, str):
            return Period.of_iso_duration(resolution)
        elif isinstance(resolution, Period):
            # Check that this is a non-offset period
            if resolution.has_offset():
                raise PeriodValidationError(f"Resolution must be a non-offset period. Got: '{resolution}'")
            return resolution
        else:
            raise TypeError(f"Resolution must be str | Period | None. Got: '{type(resolution)}'")

    @staticmethod
    def _configure_offset_property(offset: str | None) -> str | None:
        """Normalise and derive the period properties of ``offset``

        Checks if valid offset string. If none, no action - no offset used.

        Args:
            offset: Offset applied from the natural boundary of ``resolution`` to position the datetime values along the
                    timeline.

        Returns:
            Offset object
        """
        if not isinstance(offset, (str, type(None))):
            raise TypeError(f"Offset must be str | None. Got: '{type(offset)}'")
        return offset

    @staticmethod
    def _configure_alignment_property(resolution: Period, offset: str | None) -> Period:
        """Normalise and derive the period properties of ``alignment``

        Represents resolution+offset. Uses string form of those objects to create a new Period object, e.g. P1D+PT9

        Args:
            resolution: Sampling interval for the timeseries.
            offset: Offset applied from the natural boundary of ``resolution`` to position the datetime values along the
                    timeline.

        Returns:
            Alignment object
        """
        # Configure alignment parameter (resolution + offset)
        offset_str = offset or ""
        alignment = Period.of_duration(str(resolution) + offset_str)
        return alignment

    @staticmethod
    def _configure_periodicity_property(periodicity: str | Period | None, alignment: Period) -> Period:
        """Normalise and derive the period properties of ``periodicity``

        Parses string into Period object. If none, default to same Period as the alignment object
        (i.e., to represent one value per alignment bucket).

        Args:
            periodicity: Defines the allowed "frequency" of datetimes in your timeseries, i.e., how many datetime
                         entries are allowed within a given period of time.
            alignment: A Period object representing resolution+offset
        Returns:
            Resolution period object
        """
        # Configure periodicity parameter
        if periodicity is None:
            # Default to the alignment Period
            return Period.of_duration(str(alignment))
        elif isinstance(periodicity, str):
            return Period.of_duration(periodicity)
        elif isinstance(periodicity, Period):
            return periodicity
        else:
            raise TypeError(f"Periodicity must be str | Period | None. Got: '{type(periodicity)}'")

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return a sorted DataFrame whose time column satisfies this manager's temporal rules.

        Duplicate and misaligned rows are resolved according to the configured strategies, and what remains
        is validated.

        Args:
            df: Dataframe to prepare.

        Returns:
            The prepared DataFrame, sorted by the time column.
        """
        self._validate_time_column(df)

        df = self._handle_time_duplicates(df)
        df = self._handle_misaligned_rows(df)
        df = df.sort(self._time_name)

        self._validate_time_values(df[self.time_name])
        return df

    def validate(self, df: pl.DataFrame) -> None:
        """Carry out a series of validations on the temporal aspects of the TimeFrame.

        Args:
            df: Dataframe to validate against.
        """
        self._validate_time_column(df)
        self._validate_time_values(df[self.time_name])

    def _validate_time_values(self, dt: pl.Series) -> None:
        """Validate the alignment and periodicity of the time values.

        Args:
            dt: The datetime series to validate.
        """
        self._validate_alignment(dt)
        self._validate_periodicity(dt)

    def _validate_alignment(self, dt: pl.Series) -> None:
        """Validate that the time values of the time series align to the steps along the timeline
        defined by the resolution and offset parameters.

        Args:
            dt: The datetime series to validate.

        Raises:
            ResolutionError: If the resolution isn't epoch agnostic, or the datetimes are not aligned to the defined
                temporal lattice.
        """
        if not self.alignment.is_epoch_agnostic():
            raise ResolutionError(f"Non-epoch agnostic resolution is not supported: '{self.alignment}'")
        if not self.alignment.is_subperiod_of(self.periodicity):
            raise ResolutionError(
                f"Alignment '{self.alignment}' must be a subperiod of periodicity '{self.periodicity}'"
            )
        if not check_alignment(dt, self.alignment, self.time_anchor):
            raise ResolutionError(f"Time values are not aligned to resolution[+offset]: {self.alignment}")

    def _validate_periodicity(self, dt: pl.Series) -> None:
        """Validate the periodicity of the time series.

        Args:
            dt: The datetime series to validate the periodicity of.

        Raises:
            PeriodicityError: If the periodicity isn't epoch agnostic, or the datetimes do not conform to it.
        """
        if not self.periodicity.is_epoch_agnostic():
            raise PeriodicityError(f"Non-epoch agnostic periodicity is not supported: '{self.periodicity}'")
        if not check_periodicity(dt, self.periodicity, self.time_anchor):
            raise PeriodicityError(f"Time values do not conform to periodicity: {self.periodicity}")

    def _validate_time_column(self, df: pl.DataFrame) -> None:
        """Validate that the DataFrame contains the required time column with time data.

        Args:
            df: The DataFrame to validate against.

        Raises:
            ColumnNotFoundError: If the time column is missing.
            ColumnTypeError: If the time column is not Date or Datetime, or has a time zone other than UTC.
            NullTimeValueError: If the time column contains null values.
        """
        # Validate that the time column actually exists
        if self.time_name not in df.columns:
            raise ColumnNotFoundError(
                f"Time column '{self.time_name}' not found in DataFrame. Available columns: {list(df.columns)}"
            )

        # Validate time column type
        dtype = df[self.time_name].dtype
        if not isinstance(dtype, (pl.Date, pl.Datetime)):
            raise ColumnTypeError(f"Time column '{self.time_name}' must be Date or Datetime type, got '{dtype}'")

        # Time zones with daylight saving have days that aren't 24 hours long, which aren't supported
        if isinstance(dtype, pl.Datetime) and dtype.time_zone not in (None, "UTC"):
            raise ColumnTypeError(
                f"Time column '{self.time_name}' has time zone '{dtype.time_zone}'. Only UTC is supported, as time "
                f"zones with daylight saving don't have 24-hour days. Convert the column to UTC with "
                f"`.dt.convert_time_zone('UTC')`, or remove the time zone with `.dt.replace_time_zone(None)`."
            )

        # Validate that every row has a time value
        null_count = df[self.time_name].null_count()
        if null_count:
            raise NullTimeValueError(
                f"Time column '{self.time_name}' contains {null_count} null value(s). A TimeFrame must have a "
                f"time value on every row."
            )

    def with_periodicity(self, periodicity: str | Period) -> "TimeManager":
        """Return a copy of this manager with a new periodicity.

        Args:
            periodicity: The new periodicity.

        Returns:
            A new TimeManager.
        """
        new = copy(self)
        new._periodicity = configure_period_object(periodicity)
        return new

    def with_time_name(self, time_name: str) -> "TimeManager":
        """Return a copy of this manager with a new time column name.

        Args:
            time_name: The new time column name.

        Returns:
            A new TimeManager.
        """
        new = copy(self)
        new._time_name = time_name
        return new

    def check_integrity(self, old_df: pl.DataFrame, new_df: pl.DataFrame) -> None:
        """Raise an error if the time values change between old and new DataFrames.

        Args:
            old_df: The old `Polars` DataFrame to validate against.
            new_df: The new `Polars` DataFrame to validate from.

        Raises:
            TimeMutatedError: If the new time values differ from the old.
        """
        if new_df is old_df:
            return

        new_ts = new_df[self._time_name]
        old_ts = old_df[self._time_name]

        # Compare sorted series
        if not old_ts.sort().equals(new_ts.sort()):
            raise TimeMutatedError(old_timestamps=old_ts, new_timestamps=new_ts)

    def _handle_time_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle duplicate values in the time column based on a specified strategy.

        Args:
            df: Dataframe to handle duplicates from.

        Returns:
            Dataframe with duplicate values handled based on specified strategy.

        Raises:
            DuplicateTimeError: If there are duplicate timestamps and the "error" strategy is being used.
        """
        try:
            return handle_duplicates(df, self._time_name, self._on_duplicates)
        except DuplicateValueError:
            raise DuplicateTimeError()

    def _handle_misaligned_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle misaligned rows.

        If the _on_misaligned_rows property is set to "RESOLVE" then any rows found to have an unexpected
        resolution are removed. Otherwise the rows are left alone, for alignment validation to reject.

        Args:
            df: DataFrame to check for, and potentially remove, misaligned rows.

        Returns:
            DataFrame with any misaligned rows removed.

        """
        if self._on_misaligned_rows == "resolve":
            df = self._remove_misaligned_rows(df)

        return df

    def _remove_misaligned_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove misaligned rows

        Identify rows within the time series which have a different resolution to that expected (e.g. rows of PT1M
        data within a PT30M dataset) and remove them.

        Args:
            df: DataFrame containing the timeseries to check the resolution contiguity for.

        Returns:
            DataFrame with invalid rows removed.

        """
        mask = df[self.time_name] != truncate_to_period(
            date_times=df[self.time_name], period=self.alignment, time_anchor=self.time_anchor
        )
        invalid_timestamps = df[self.time_name].filter(mask)

        # If no invalid timestamps have been found, exit early as there is nothing else to do.
        if invalid_timestamps.is_empty():
            return df

        formatted_timestamps = [item.strftime("%Y-%m-%d %H:%M:%S") for item in invalid_timestamps.to_list()]
        logger.info(
            f"Removing the following timestamps which were found to not conform to the expected resolution of "
            f"{self.resolution.iso_duration}: `{formatted_timestamps}`"
        )
        valid_data = df.filter(~mask)
        return valid_data
