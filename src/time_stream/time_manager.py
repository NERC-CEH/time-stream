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

import polars as pl

from time_stream import Period
from time_stream.enums import DuplicateOption, TimeAnchor, ValidationErrorOptions
from time_stream.exceptions import (
    ColumnNotFoundError,
    ColumnTypeError,
    DuplicateTimeError,
    DuplicateValueError,
    PeriodicityError,
    PeriodValidationError,
    ResolutionError,
    TimeMutatedError,
)
from time_stream.utils import check_alignment, check_periodicity, epoch_check, handle_duplicates, truncate_to_period

logger = logging.getLogger(__name__)


class TimeManager:
    """Enforces integrity of the temporal aspects of the TimeFrame"""

    def __init__(
        self,
        time_name: str,
        resolution: str | Period | None = None,
        offset: str | None = None,
        periodicity: str | Period | None = None,
        on_duplicates: str | DuplicateOption = DuplicateOption.ERROR,
        on_misaligned_rows: str | ValidationErrorOptions = ValidationErrorOptions.ERROR,
        time_anchor: str | TimeAnchor = TimeAnchor.START,
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
            time_anchor: The time anchor to which the date/times conform to.
        """
        self._time_name = time_name
        self._resolution = resolution
        self._offset = offset
        self._alignment = None
        self._periodicity = periodicity
        self._on_duplicates = DuplicateOption(on_duplicates)
        self._on_misaligned_rows = ValidationErrorOptions(on_misaligned_rows)
        self._time_anchor = TimeAnchor(time_anchor)

        self._configure_period_properties()

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

    def _configure_period_properties(self) -> None:
        """Normalise and derive the period properties: ``resolution``, ``offset``, ``alignment`` (resolution+offset),
        and ``periodicity``.

        ``resolution``: Parses string into Period object. If none, default to 1 microsecond to allow any resolution.
        ``offset``: Checks if valid offset string. If none, no action - no offset used.
        ``alignment``: Represents resolution+offset. Uses string form of those objects to create a new Period object
                       e.g. P1D+PT9H.
        ``periodicity``: Parses string into Period object. If none, default to same Period as the alignment object
                         (i.e., to represent one value per alignment bucket).

        Raises:
            TypeError:  If any of ``_resolution``, ``_offset``, or ``_periodicity`` is not ``str``,
                        ``Period``, or ``None``.
        """
        # Configure resolution parameter
        if self._resolution is None:
            self._resolution = Period.of_microseconds(1)
        elif isinstance(self._resolution, str):
            self._resolution = Period.of_iso_duration(self._resolution)
        elif isinstance(self._resolution, Period):
            # Check that this is a non-offset period
            if self._resolution.has_offset():
                raise PeriodValidationError(f"Resolution must be a non-offset period. Got: '{self._resolution}'")
        else:
            raise TypeError(f"Resolution must be str | Period | None. Got: '{type(self._resolution)}'")

        # Configure offset parameter
        if isinstance(self._offset, str):
            pass
        elif self._offset is not None:
            raise TypeError(f"Offset must be str | None. Got: '{type(self._offset)}'")

        # Configure alignment parameter (resolution + offset)
        offset_str = self._offset or ""
        self._alignment = Period.of_duration(str(self._resolution) + offset_str)

        # Configure periodicity parameter
        if self._periodicity is None:
            # Default to the alignment Period
            self._periodicity = Period.of_duration(str(self._alignment))
        elif isinstance(self._periodicity, str):
            self._periodicity = Period.of_duration(self._periodicity)
        elif not isinstance(self._periodicity, Period):
            raise TypeError(f"Periodicity must be str | Period | None. Got: '{type(self._periodicity)}'")

    def validate(self, df: pl.DataFrame) -> None:
        """Carry out a series of validations on the temporal aspects of the TimeFrame.

        Args:
            df: Dataframe to validate against.
        """
        self._validate_time_column(df)

        dt = df[self.time_name]
        self._validate_alignment(dt)
        self._validate_periodicity(dt)

    def _validate_alignment(self, dt: pl.Series) -> None:
        """Validate that the time values of the time series align to the steps along the timeline
        defined by the resolution and offset parameters.

        Args:
            dt: The datetime series to validate.

        Raises:
            ResolutionError: If the datetimes are not aligned to the defined temporal lattice.
        """
        epoch_check(self.alignment)
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
            PeriodicityError: If the datetimes do not conform to the periodicity.
        """
        epoch_check(self.periodicity)
        if not check_periodicity(dt, self.periodicity, self.time_anchor):
            raise PeriodicityError(f"Time values do not conform to periodicity: {self.periodicity}")

    def _validate_time_column(self, df: pl.DataFrame) -> None:
        """Validate that the DataFrame contains the required time column with time data.

        Args:
            df: The DataFrame to validate against.

        Raises:
            ColumnNotFoundError: If the time column is missing.
            ColumnTypeError: If the time column does not contain temporal data.
        """
        # Validate that the time column actually exists
        if self.time_name not in df.columns:
            raise ColumnNotFoundError(
                f"Time column '{self.time_name}' not found in DataFrame. Available columns: {list(df.columns)}"
            )

        # Validate time column type
        dtype = df[self.time_name].dtype
        if not dtype.is_temporal():
            raise ColumnTypeError(f"Time column '{self.time_name}' must be datetime type, got '{dtype}'")

    def _check_time_integrity(self, old_df: pl.DataFrame, new_df: pl.DataFrame) -> None:
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
            new_df = handle_duplicates(df, self._time_name, DuplicateOption(self._on_duplicates))
        except DuplicateValueError:
            raise DuplicateTimeError()

        # Polars aggregate methods can change the order due to how it optimises the functionality, so sort times after
        new_df = new_df.sort(self._time_name)
        return new_df

    def _handle_misaligned_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._on_misaligned_rows == ValidationErrorOptions.ERROR:
            self._validate_alignment(df[self.time_name])
        elif self._on_misaligned_rows == ValidationErrorOptions.RESOLVE:
            # No need to run _validate_alignment as the row removal logic will cover the same checks and any
            # invalid rows will be logged before being removed
            df = self._remove_rows_with_invalid_resolution(df)

        return df

    def _check_time_resolution_contiguity(
        self, df: pl.DataFrame, error_method: ValidationErrorOptions = ValidationErrorOptions.ERROR
    ) -> pl.DataFrame | None:
        """Check the time resolution contiguity

        Identify rows within the time series which have a different resolution to that expected (e.g. rows of PT1M
        data within a PT30M dataset).

        It also searches for gaps within the data, and will log these as warnings. For example where the dataset
        stops, and is restarted again later in the year, but with no change in resolution.

        Args:
            df: DataFrame containing the timeseries to check the resolution contiguity for.
            error_method: Strategy for handling duplicates:
                - ERROR: Raise an error.
                - WARN: Log a warining.
                - RESOLVE: Remove the erroneous rows.

        Raises:
            ResolutionError: Rows have been found with an unexpected resolution and the error_method is set to `ERROR`.

        Returns:
            DataFrame with invalid rows removed if the error_method is RESOLVE

        """
        mask = df[self.time_name] != truncate_to_period(
            date_times=df[self.time_name], period=self.alignment, time_anchor=self.time_anchor
        )
        invalid_timestamps = df[self.time_name].filter(mask)

        # If no invalid timestamps have been found, exit early as there is nothing else to do.
        if invalid_timestamps.is_empty():
            return

        formatted_timestamps = [item.strftime("%Y-%m-%d %H:%M:%S") for item in invalid_timestamps.to_list()]
        error_message = (
            f"The following timestamps do not conform to the expected resolution of "
            f"{self.resolution.iso_duration}: `{formatted_timestamps}`"
        )

        if error_method == ValidationErrorOptions.ERROR:
            raise ResolutionError(error_message)

        if error_method == ValidationErrorOptions.WARN:
            logger.warning(error_message)
            return

        if error_method == ValidationErrorOptions.RESOLVE:
            logger.info(
                f"Removing the following timestamps which were found to not conform to the expected resolution of "
                f"{self.resolution.iso_duration}: `{formatted_timestamps}`"
            )
            valid_data = df.filter(~mask)
            return valid_data

    def _remove_rows_with_invalid_resolution(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Check the time resolution contiguity

        Identify rows within the time series which have a different resolution to that expected (e.g. rows of PT1M
        data within a PT30M dataset).

        It also searches for gaps within the data, and will log these as warnings. For example where the dataset
        stops, and is restarted again later in the year, but with no change in resolution.

        Args:
            df: DataFrame containing the timeseries to check the resolution contiguity for.
            error_method: Strategy for handling duplicates:
                - ERROR: Raise an error.
                - WARN: Log a warining.
                - RESOLVE: Remove the erroneous rows.

        Raises:
            ResolutionError: Rows have been found with an unexpected resolution and the error_method is set to `ERROR`.

        Returns:
            DataFrame with invalid rows removed if the error_method is RESOLVE

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
