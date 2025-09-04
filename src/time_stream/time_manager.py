from collections.abc import Callable

import polars as pl

from time_stream import Period
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import (
    ColumnNotFoundError,
    ColumnTypeError,
    DuplicateTimeError,
    DuplicateValueError,
    PeriodicityError,
    ResolutionError,
    TimeMutatedError,
)
from time_stream.utils import check_periodicity, check_resolution, configure_period_object, handle_duplicates


class TimeManager:
    """Enforces integrity of the temporal aspect of the TimeSeries"""

    def __init__(
        self,
        get_df: Callable[[], pl.DataFrame],
        time_name: str,
        resolution: str | Period | None = None,
        periodicity: str | Period | None = None,
        on_duplicates: str | DuplicateOption = DuplicateOption.ERROR,
        time_anchor: str | TimeAnchor = TimeAnchor.START,
    ):
        """Initialise the time manager.

        Args:
            get_df: A callback to access the current dataframe, so it never has to hold a direct
                    reference to the parent TimeSeries to avoid circular dependencies.
            time_name: The name of the time column of the parent TimeSeries.
            resolution: Resolution defines how "precise" the datetimes are, i.e., to what precision of time unit
                        should each datetime in the time series match to.

                        Some examples:
                        PT0.000001S  Allow all datetime values, including microseconds.
                        PT1S         Allow datetimes with a whole number of seconds. Microseconds must be "0".
                        PT1M         Allow datetimes to be specified to the minute.
                                        Seconds and Microseconds must be "0".
                        PT15M	     Allow datetimes to be specified to a multiple of 15 minutes.
                                        Seconds and Microseconds must be "0" and Minutes be one of (0, 15, 30, 45)
                        P1D	         Allow all dates, but the time must be "00:00:00"
                        P1M	         Allow all years and months, but the day must be "1" and time "00:00:00"
                        P3M	         Allow quarterly dates.
                                        Month must be one of (1, 4, 7, 10), day must be "1" and time "00:00:00"
                        P1Y+9M9H	 Only dates at 09:00 am on the 1st of October are allowed.
            periodicity: Periodicity defines the allowed "frequency" of the datetimes, i.e., how many datetimes
                         entries are allowed within a given period of time.

                         Some examples:
                         PT0.000001S Effectively there is no "periodicity".
                         PT1S	     At most 1 datetime can occur within any given second.
                         PT1M	     At most 1 datetime can occur within any given minute.
                         PT15M	     At most 1 datetime can occur within any 15-minute duration.
                                        Each 15-minute duration starts at (0, 15, 30, 45) minutes past the hour.
                         P1D	     At most 1 datetime can occur within any given calendar day
                                        (from midnight of the first day up to, but not including, midnight of
                                        the next day).
                         P1M	     At most 1 datetime can occur within any given calendar month
                                        (from midnight on the 1st of the month up to, but not including,
                                        midnight on the 1st of the following month).
                         P3M	     At most 1 datetime can occur within any given quarterly period.
                         P1Y+9M9H	 At most 1 datetime can occur within any given water year
                                        (from 09:00 am on the 1st of October up to, but including, 09:00 am on
                                        the 1st of the following year).
            on_duplicates: What to do if duplicate rows are found in the data. Default to ERROR.
                           DROP: Raise an error if duplicate rows are found.
                           KEEP_FIRST: Keep the first row of any duplicate groups.
                           KEEP_LAST: Keep the last row of any duplicate groups.
                           DROP: Drop all duplicate rows.
                           MERGE: Merge duplicate rows using "coalesce"
                                  (the first non-null value for each column takes precedence).
            time_anchor: The time anchor to which the date/times conform to.
                         In the descriptions below, "x" is the time value, "r" stands for a single unit of the
                         resolution of the data (15-minute, 1-hour, 1-day, etc.)::
                         POINT: The time stamp is anchored for the instant of time "x".
                                A value at "x" is considered valid only for the instant of time "x".
                         START: The time stamp is anchored starting at "x". A value at "x" is considered valid
                                starting at "x" (inclusive) and ending at "x+r" (exclusive).
                         END: The time stamp is anchored ending at "x". A value at "x" is considered valid
                              starting at "x-r" (exclusive) and ending at "x" (inclusive).
        """
        self._get_df = get_df
        self._time_name = time_name
        self._resolution = configure_period_object(resolution)
        self._periodicity = configure_period_object(periodicity)
        self._on_duplicates = DuplicateOption(on_duplicates)
        self._time_anchor = TimeAnchor(time_anchor)

    @property
    def time_name(self) -> str:
        return self._time_name

    @property
    def resolution(self) -> Period:
        return self._resolution

    @property
    def periodicity(self) -> Period:
        return self._periodicity

    @property
    def time_anchor(self) -> TimeAnchor:
        return self._time_anchor

    def _time_data(self) -> pl.Series:
        return self._get_df()[self.time_name]

    def validate(self) -> None:
        """Carry out a series of validations on the temporal aspects of the TimeSeries.

        Raises:
            ResolutionError: If the resolution is not a subperiod of the periodicity
        """
        df = self._get_df()
        self._validate_time_column(df)

        if not self._resolution.is_subperiod_of(self._periodicity):
            raise ResolutionError(
                f"Resolution {self._resolution} must be a subperiod of periodicity {self._periodicity}"
            )

        dt = self._time_data()
        self._validate_resolution(dt)
        self._validate_periodicity(dt)

    def _validate_resolution(self, dt: pl.Series) -> None:
        """Validate the resolution of the time series.

        Args:
            dt: The datetime series to validate the resolution of.

        Raises:
            ResolutionError: If the datetimes are not aligned to the resolution.
        """
        resolution_check = check_resolution(dt, self.resolution, self.time_anchor)
        if not resolution_check:
            raise ResolutionError(f"Time values are not aligned to resolution: {self.resolution}")

    def _validate_periodicity(self, dt: pl.Series) -> None:
        """Validate the periodicity of the time series.

        Args:
            dt: The datetime series to validate the periodicity of.

        Raises:
            PeriodicityError: If the datetimes do not conform to the periodicity.
        """
        periodicity_check = check_periodicity(dt, self.periodicity, self.time_anchor)
        if not periodicity_check:
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

    def _handle_time_duplicates(self) -> pl.DataFrame:
        """Handle duplicate values in the time column based on a specified strategy.

        Returns:
            Dataframe with duplicate values handled based on specified strategy.

        Raises:
            DuplicateTimeError: If there are duplicate timestamps and the "error" strategy is being used.
        """
        try:
            new_df = handle_duplicates(self._get_df(), self._time_name, DuplicateOption(self._on_duplicates))
        except DuplicateValueError:
            raise DuplicateTimeError()

        # Polars aggregate methods can change the order due to how it optimises the functionality, so sort times after
        new_df = new_df.sort(self._time_name)
        return new_df
