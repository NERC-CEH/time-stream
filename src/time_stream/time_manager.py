from collections.abc import Callable

import polars as pl

from time_stream import Period
from time_stream.enums import DuplicateOption
from time_stream.exceptions import DuplicateTimeError, DuplicateValueError, TimeMutatedError
from time_stream.utils import handle_duplicates, pad_time


class TimeManager:
    """Handles operations on the time data and enforces integrity of the temporal aspect of the TimeSeries

    Responsibilities:
      - Ensure time values have not been mutated when setting new DataFrames.
      - Handle duplicate timestamps according to a DuplicateOption policy.
      - Optionally pad missing timestamps using a given periodicity.
      - Keep the frame sorted by the time column.
    """

    def __init__(
            self,
            get_df: Callable[[], pl.Dataframe],
            set_df: Callable[pl.Dataframe, []],
            time_name: str,
            resolution: Period,
            periodicity: Period,
            on_duplicates: DuplicateOption
    ):
        """Initialise time manager.

        Args:
            get_df: A callback to access the current dataframe, so it never has to hold a direct
                    reference to the parent TimeSeries to avoid circular dependencies.
            set_df: A callback to set a new dataframe object in the parent TimeSeries.
            time_name: The name of the time column of the parent TimeSeries.
            resolution: The resolution of the time series.
            periodicity: The periodicity of the time series.
            on_duplicates: What to do if duplicate rows are found in the data. Default to ERROR.
        """
        self._get_df = get_df
        self._set_df = set_df
        self._time_name = time_name
        self._resolution = resolution
        self._periodicity = periodicity
        self._on_duplicates = on_duplicates

    def check_time_integrity(self, old_df: pl.DataFrame, new_df: pl.DataFrame) -> None:
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

    def handle_time_duplicates(self, on_duplicates: DuplicateOption | str | None = None) -> None:
        """Handle duplicate values in the time column based on a specified strategy.

        Args:
            on_duplicates: The strategy to use to handle duplicate timestamps. Current options:
                            - "error": Raise an error if duplicate rows are found.
                            - "keep_first": Keep the first row of any duplicate groups.
                            - "keep_last": Keep the last row of any duplicate groups.
                            - "drop": Drop all duplicate rows.
                            - "merge": Merge duplicate rows using "coalesce" (the first non-null value for each column takes precedence).

        Raises:
            DuplicateTimeError: If there are duplicate timestamps and the "error" strategy is being used.
        """
        if on_duplicates is None:
            on_duplicates = DuplicateOption.ERROR

        try:
            new_df = handle_duplicates(self._get_df(), self._time_name, DuplicateOption(on_duplicates))
        except DuplicateValueError:
            raise DuplicateTimeError()

        self._set_df(new_df)
        # Polars aggregate methods can change the order due to how it optimises the functionality, so sort times after
        self.sort_time()

    def pad_time(self) -> None:
        """Pad the time series with missing datetime rows, filling in NULLs for missing values.
        """
        df = self._get_df()
        padded_df = pad_time(df, self._time_name, self._periodicity)
        self._set_df(padded_df)

    def sort_time(self) -> None:
        """Sort the TimeSeries DataFrame by the time column."""
        self._set_df(self._get_df().sort(self._time_name))
