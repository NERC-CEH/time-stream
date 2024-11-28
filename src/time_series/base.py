import copy
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Type, Union

import polars as pl

from time_series.period import Period

if TYPE_CHECKING:
    # Import is for type hinting only.  Make sure there is no runtime import, to avoid recursion.
    from time_series.aggregation import AggregationFunction


class TimeSeries:
    """A class representing a time series data model, with data held in a Polars DataFrame."""

    def __init__(
        self,
        df: pl.DataFrame,
        time_name: str,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        time_zone: Optional[str] = None,
        supplementary_columns: Optional[list] = None,
        metadata: Optional[dict[str, dict[str, Any]]] = None,
    ) -> None:
        """Initialise a TimeSeries instance.

        Args:
            df: The Polars DataFrame containing the time series data.
            time_name: The name of the time column in the DataFrame.
            resolution: The resolution of the time series. Defaults to None.
            periodicity: The periodicity of the time series. Defaults to None.
            time_zone: The time zone of the time series. Defaults to None.
            supplementary_columns: Columns within the dataframe that are considered supplementary. Defaults to None.
            metadata: The metadata of the variables within the time series. Defaults to None.
        """
        self._time_name = time_name
        self._resolution = resolution
        self._periodicity = periodicity
        self._time_zone = time_zone
        self._supplementary_columns = set(supplementary_columns) if supplementary_columns else set()

        #  NOTE: Doing a deep copy of this mutable object, otherwise the original object will refer to the same
        #   object in memory and will be changed by class methods.
        self._metadata = copy.deepcopy(metadata) or {}

        self._df = df

        self._setup()

    def _setup(self) -> None:
        """Perform initial setup for the TimeSeries instance."""
        self._set_time_zone()
        self.sort_time()
        self._validate_resolution()
        self._validate_periodicity()
        self.set_supplementary_columns(self.supplementary_columns)

    @property
    def time_name(self) -> str:
        """The name of the primary datetime column in the underlying TimeSeries DataFrame."""
        return self._time_name

    @property
    def time_zone(self) -> str:
        """The time zone of the primary datetime column in the underlying TimeSeries DataFrame."""
        return self._time_zone

    def _set_time_zone(self) -> None:
        """Set the time zone for the TimeSeries.

        If a time zone is provided in class initialisation, this will overwrite any time zone set on the dataframe.
        If no time zone provided, defaults to either the dataframe time zone, or if this is not set either, then UTC.
        """
        default_time_zone = "UTC"
        df_time_zone = self.df.schema[self.time_name].time_zone

        if self.time_zone is not None:
            time_zone = self.time_zone
        elif self.time_zone is None and df_time_zone is not None:
            time_zone = df_time_zone
        else:
            time_zone = default_time_zone

        # Set _df directly, otherwise the df setter considers the time column is mutated due to time zone added
        self._df = self.df.with_columns(pl.col(self.time_name).dt.replace_time_zone(time_zone))
        self._time_zone = time_zone

    def select_time(self) -> pl.Series:
        """Return just the data series of the primary datetime field."""
        return self.df[self.time_name]

    def sort_time(self) -> None:
        """Sort the TimeSeries DataFrame by the time column."""
        self.df = self.df.sort(self.time_name)

    @property
    def resolution(self) -> Period:
        """Resolution defines how "precise" the datetimes are, i.e. to what precision of time unit should each
        datetime in the time series match to.

        Some examples:
        P0.000001S  Allow all datetime values, including microseconds.
        P1S	        Allow datetimes with a whole number of seconds. Microseconds must be "0".
        PT1M	    Allow datetimes to be specified to the minute. Seconds and Microseconds must be "0".
        PT15M	    Allow datetimes to be specified to a multiple of 15 minutes.
                    Seconds and Microseconds must be "0", and Minutes be one of ("00", "15", "30", "45")
        P1D	        Allow all dates, but the time must be "00:00:00"
        P1M	        Allow all years and months, but the day must be "1" and time "00:00:00"
        P3M	        Quarterly dates; month must be one of ("1", "4", "7", "10"), day must be "1" and time "00:00:00"
        P1Y+9M9H	Only dates at 09:00 am on the 1st of October are allowed.
        """
        return self._resolution

    def _validate_resolution(self) -> None:
        """Validate the resolution of the time series.

        Raises:
            UserWarning: If the datetimes are not aligned to the resolution.
        """
        if self.resolution is None:
            # Default to a resolution that accepts all datetimes
            self._resolution = Period.of_microseconds(1)

        self._epoch_check(self.resolution)

        # Compare the original series to the rounded series.  If no match, it is not aligned to the resolution.
        rounded_times = self._round_time_to_period(self.resolution)
        aligned = self.df[self.time_name].equals(rounded_times)
        if not aligned:
            raise UserWarning(
                f'Values in time field: "{self.time_name}" are not aligned to ' f"resolution: {self.resolution}"
            )

    @property
    def periodicity(self) -> Period:
        """Periodicity defines the allowed "frequency" of the datetimes, i.e. how many datetimes entries are allowed
        within a given period of time.

        Some examples:
        P0.000001S	Effectively there is no "periodicity".
        P1S	        At most 1 datetime can occur within any given second.
        PT1M	    At most 1 datetime can occur within any given minute.
        PT15M	    At most 1 datetime can occur within any 15-minute duration.
                    Each 15-minute durations starts at ("00", "15", "30", "45") minutes past the hour.
        P1D	        At most 1 datetime can occur within any given calendar day
                    (from midnight of first day up to, but not including, midnight of the next day)
        P1M	        At most 1 datetime can occur within any given calendar month
                    (from midnight on the 1st of the month up to, but not including, midnight on the 1st of the
                    following month).
        P3M	        At most 1 datetime can occur within any given quarterly period.
        P1Y+9M9H	At most 1 datetime can occur within any given water year
                    (from 09:00 am on the 1st of October up to, but including, 09:00 am on the 1st of the
                    following year).
        """
        return self._periodicity

    def _validate_periodicity(self) -> None:
        """Validate the periodicity of the time series.

        Raises:
            UserWarning: If the datetimes do not conform to the periodicity.
        """
        if self.periodicity is None:
            # Default to a periodicity that accepts all datetimes
            self._periodicity = Period.of_microseconds(1)

        self._epoch_check(self.periodicity)

        # Check how many unique values are in the rounded times. It should equal the length of the original time series
        # if all time values map to a single periodicity
        rounded_times = self._round_time_to_period(self.periodicity)
        all_unique = rounded_times.n_unique() == len(self.df[self.time_name])
        if not all_unique:
            raise UserWarning(
                f'Values in time field: "{self.time_name}" do not conform to ' f"periodicity: {self.periodicity}"
            )

    def _round_time_to_period(self, period: Period) -> pl.Series:
        """Round the time column to the given period.

        Args:
           period: The period to which the time column should be rounded.

        Returns:
           A Polars Series with the rounded time values.
        """
        # Remove any offset from the time series
        time_series_no_offset = self.df[self.time_name].dt.offset_by("-" + period.pl_offset)

        # Round the (non offset) time series to the given resolution interval and add the offset back on
        rounded_times = time_series_no_offset.dt.truncate(period.pl_interval)
        rounded_times_with_offset = rounded_times.dt.offset_by(period.pl_offset)

        return rounded_times_with_offset

    @staticmethod
    def _epoch_check(period: Period) -> None:
        """Check if the period is epoch-agnostic.

        A period is considered "epoch agnostic" if it divides the timeline into consistent intervals regardless of the
        epoch (starting point) used for calculations. This ensures that the intervals are aligned with natural
        calendar or clock units (e.g., days, months, years), rather than being influenced by the specific epoch used
        in arithmetic.

        For example:
            - Epoch-agnostic periods include:
                - `P1Y` (1 year): Intervals are aligned to calendar years.
                - `P1M` (1 month): Intervals are aligned to calendar months.
                - `P1D` (1 day): Intervals are aligned to whole days.
                - `PT15M` (15 minutes): Intervals are aligned to clock minutes.

            - Non-epoch-agnostic periods include:
                - `P7D` (7 days): Intervals depend on the epoch. For example,
                  starting from 2023-01-01 vs. 2023-01-03 would result in
                  different alignments of 7-day periods.

        Args:
            period: The period to check.

        Raises:
            NotImplementedError: If the period is not epoch-agnostic.
        """
        if not period.is_epoch_agnostic():
            # E.g. 5 hours, 7 days, 9 months, etc.
            raise NotImplementedError("Not available for non-epoch agnostic periodicity")

    @property
    def df(self) -> pl.DataFrame:
        """The underlying DataFrame of the TimeSeries object."""
        return self._df

    @df.setter
    def df(self, new_df: pl.DataFrame) -> None:
        """Update the underlying DataFrame while preserving the integrity of primary datetime field, metadata and
        supplementary column settings.

        Checks that the time column has not changed, and removes any metadata and or supplementary column
        settings for any columns missing in the new DataFrame.

        Args:
            new_df: The new Polars DataFrame to set as the time series data.
        """
        self._validate_time_column(new_df)
        self._remove_missing_columns(new_df)
        self._df = new_df

    def _validate_time_column(self, df: pl.DataFrame) -> None:
        """Validate that the DataFrame contains the required time column with unchanged timestamps.

        This ensures that the `time_name` column exists and its values match the current time series.
        Adding or removing rows (therefore modifying the time aspect of the timeseries) is considered a
        significant change and should result in a new TimeSeries object.

        Args:
            df: The Polars DataFrame to validate.

        Raises:
            ValueError: If the time column is missing or if its timestamps differ from the current DataFrame.
        """
        if self.time_name not in df.columns:
            raise ValueError(f"Time column {self.time_name} not found.")

        new_timestamps = df[self.time_name]
        old_timestamps = self.df[self.time_name]
        if Counter(new_timestamps.to_list()) != Counter(old_timestamps.to_list()):
            raise ValueError("Time column has mutated.")

    def _remove_missing_columns(self, new_df: pl.DataFrame) -> None:
        """Remove metadata and supplementary column settings for columns that are missing in the new DataFrame.

        Args:
            new_df: The new Polars DataFrame to compare against the current DataFrame.
        """
        removed_columns = list(set(self.df.columns) - set(new_df.columns))
        self.unset_supplementary_columns(removed_columns)
        for column in removed_columns:
            self.remove_metadata(column)

    @property
    def data_columns(self) -> list:
        """Sorted list of data column names in the TimeSeries."""
        return sorted([col for col in self.columns if col not in self.supplementary_columns])

    @property
    def supplementary_columns(self) -> list:
        """Sorted list of supplementary column names in the TimeSeries."""
        return sorted(list(self._supplementary_columns))

    @property
    def columns(self) -> list:
        """Sorted list of all columns (data columns and supplementary columns) in the TimeSeries,
        excluding the time column.
        """
        return sorted([col for col in self.df.columns if col != self.time_name])

    def select_columns(self, columns: list[str]) -> "TimeSeries":
        """Filter TimeSeries instance to include only the specified columns.

        Args:
            columns: A list of column names to retain in the updated TimeSeries.

        Returns:
            New TimeSeries instance with only selected columns.
        """
        if not columns:
            raise ValueError("No columns specified.")

        new_df = self.df.select([self.time_name] + columns)
        new_supplementary_columns = [col for col in self.supplementary_columns if col in columns]
        new_metadata = {col: self._get_metadata(col) for col in columns}

        ts = TimeSeries(
            new_df,
            self.time_name,
            self.resolution,
            self.periodicity,
            self.time_zone,
            new_supplementary_columns,
            new_metadata,
        )

        return ts

    def init_supplementary_column(self, column: str, data: Optional[Union[int, float, str, Iterable]] = None) -> None:
        """Initialises a supplementary column, adding it to the TimeSeries DataFrame.

        Supplementary columns are additional columns that are not considered part of the
        primary data but provide supporting information.

        Args:
            column: The name of the new supplementary column.
            data: The data to populate the column. Can be a scalar, an iterable, or None.
                  If iterable, must have the same length as the current TimeSeries.
                  If None, the column will be filled with `None`.

        Raises:
            ValueError: If the column name already exists in the DataFrame.
        """
        if column in self.df.columns:
            raise ValueError(f"Column '{column}' already exists in the DataFrame.")

        if data is None or isinstance(data, (float, int, str)):
            data = pl.lit(data)
        else:
            data = pl.Series(column, data)

        self.df = self.df.with_columns(data.alias(column))
        self.set_supplementary_columns(column)

    def set_supplementary_columns(self, columns: Union[str, list]) -> None:
        """Mark the specified columns as supplementary columns.

        Supplementary columns must exist in the DataFrame. Attempting to mark the defined TimeSeries time column
        as supplementary will raise an error.

        Args:
            columns: A column name (string) or a list of column names to mark as supplementary.

        Raises:
            ValueError: If any of the specified columns do not exist in the DataFrame.
        """
        # NOTE: self.columns does not include the time column, so this handles erroring if trying to set the
        #   time column as a supplementary column
        if isinstance(columns, str):
            columns = [columns]

        invalid = [col for col in columns if col not in self.columns]
        if invalid:
            raise ValueError(f"Invalid supplementary columns: {invalid}")

        for col in columns:
            self._supplementary_columns.add(col)

    def unset_supplementary_columns(self, columns: Optional[list] = None) -> None:
        """Remove the specified columns from the supplementary columns list.

        Args:
            columns: A list of column names to remove from the supplementary columns list.
                     If None, all supplementary columns will be removed.
        """
        if columns is None:
            # If None provided, remove all current supplementary columns
            columns = self.supplementary_columns

        for col in columns:
            self._supplementary_columns.discard(col)

    def metadata(self, key: Optional[Union[str, list[str], tuple[str, ...]]] = None) -> dict:
        """Retrieve metadata for all or specific keys across columns.

        Args:
            key: A specific key or list/tuple of keys to filter the metadata. If None, all metadata for the
                 relevant columns is returned.

        Returns:
            A dictionary of the requested metadata.

        Raises:
            KeyError: If the requested key(s) are not found in the metadata.
        """
        try:
            result = defaultdict(dict)
            for column in self.columns:
                result[column] |= self._get_metadata(column, key)
            return dict(result)

        except (KeyError, TypeError):
            raise KeyError(key)

    def _get_metadata(self, column: str, key: Optional[Union[str, list[str], tuple[str, ...]]] = None) -> dict:
        """Retrieve metadata for a specific column and key(s).

        This is an internal method that retrieves metadata for a single column, optionally filtered by specific key(s).

        Args:
            column: The column name for which metadata should be retrieved.
            key: A specific key or list/tuple of keys to filter the metadata. If None, all metadata for the column
                 is returned.

        Returns:
            A dictionary of the requested metadata.
        """
        column_metadata = self._metadata.get(column, {})
        if isinstance(key, str):
            key = [key]

        if key is None:
            return column_metadata
        else:
            return {k: column_metadata.get(k) for k in key}

    def remove_metadata(self, column: str, key: Optional[Union[str, list[str], tuple[str, ...]]] = None) -> None:
        """Removes metadata associated with a column, either completely or for specific keys.

        Args:
            column: The name of the column for which metadata should be removed.
            key: A specific key or list/tuple of keys to remove. If None, all metadata for the column is removed.
        """
        if isinstance(key, str):
            key = [key]

        if key is None:
            self._metadata = {col: col_metadata for col, col_metadata in self._metadata.items() if col != column}
        else:
            column_metadata = self._metadata.get(column, {})
            self._metadata[column] = {k: v for k, v in column_metadata.items() if k not in key}

    def aggregate(
        self, aggregation_period: Period, aggregation_function: Type["AggregationFunction"], column_name: str
    ) -> "TimeSeries":
        """Apply an aggregation function to a column in this TimeSeries and return a new derived TimeSeries containing
        the aggregated data.

        The AggregationFunction class provides static methods that return aggregation function objects that can be used
        with this method.

        NOTE: This is the first attempt at a mechanism for aggregating time-series data.  The signature of this method
            is likely to evolve considerably.

        Args:
            aggregation_period: The period over which to aggregate the data
            aggregation_function: The aggregation function to apply
            column_name: The column containing the data to be aggregated

        Returns:
            A TimeSeries containing the aggregated data.
        """
        return aggregation_function.create().apply(self, aggregation_period, column_name)

    def __getattr__(self, name: str) -> Any:
        """Dynamically handle attribute access for the TimeSeries object.

        This method provides convenience for accessing data columns, metadata, or the time column of the DataFrame.
        It supports the following behaviors:

        - If the attribute name matches the time column, it returns the time column as a Polars Series.
        - If the attribute name matches a column in the DataFrame (excluding the time column), it selects that column
            and returns a new TimeSeries instance.
        - If the attribute name does not match a column, it assumes this is a Metadata key. In this case,
            The TimeSeries should only contain one data column (i.e. has already been filtered to a selected
            column of interest), and then it checks for metadata with the requested attribute name for that column.

        Args:
            name: The attribute name being accessed.

        Returns:
            If `name` is:
              - The time column: A Polars Series containing the time data.
              - A data column: The TimeSeries instance with that column selected.
              - Metadata key: The metadata value(s) for the single data column.

        Raises:
            AttributeError: If attribute does not match a column, metadata key for a single column, or the time column.

        Examples:
            >>> ts.timestamp  # Access the time column (assumed time_name set to "timestamp")
            <Polars Series: "timestamp">

            >>> ts.temperature  # Access a column named "temperature"
            <TimeSeries object, filtered to only contain the "temperature" data column>

            >>> ts.temperature.metadata_key
            <Metadata value for the given key on the given column>
        """
        try:
            if name == self.time_name:
                # If the attribute name matches the time column, return the time column as a Polars Series.
                return self.select_time()
            elif name in self.columns:
                # If the attribute name matches a column in the DataFrame (excluding the time column), select that
                #  column and return the TimeSeries instance.
                return self.select_columns([name])
            elif name not in self.columns and len(self.columns) == 1:
                # If the attribute name does not match a column, it assumes this is a Metadata key. In this case,
                #   the TimeSeries can only contain one data column. Check for metadata for that column.
                metadata_value = self._metadata[self.columns[0]][name]
                return metadata_value
            else:
                raise AttributeError

        except (KeyError, AttributeError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Union[str, list[str], tuple[str]]) -> Union["TimeSeries", pl.Series]:
        """Access columns or the time column using indexing syntax.

        This method enables convenient access to DataFrame columns or the time column by using indexing. It supports:

        - A single column name: Selects and returns a TimeSeries instance with the specified column.
        - A list/tuple of column names: Selects and returns a TimeSeries instance with the specified columns.
        - The time column name: Returns the time column as a Polars Series.

        Args:
            key: A single column name, a list/tuple of column names, or the time column name.

        Returns:
            - If `key` is a column name: A TimeSeries instance with the specified column(s) selected.
            - If `key` is the time column name: A Polars Series containing the time data.

        Examples:
            >>> ts["timestamp"]  # Access the time column (assumed time_name set to "timestamp")
            <Polars Series: "timestamp">

            >>> ts["temperature"]  # Access a single column
            <TimeSeries object, filtered to only contain the "temperature" data column>

            >>> ts[["temperature", "pressure"]]  # Access multiple columns
            <TimeSeries object, filtered to only contain the "temperature" and "pressure" data columns>
        """
        if key == self.time_name:
            return self.select_time()
        if isinstance(key, str):
            key = [key]
        return self.select_columns(key)

    def __len__(self) -> int:
        """Get the number of rows in the time series."""
        return self.df.height

    def __iter__(self) -> Iterator:
        """Return an iterator over the rows of the DataFrame."""
        return self.df.iter_rows()

    def __str__(self) -> str:
        """Return the string representation of the TimeSeries class."""
        return self.df.__str__()

    def __dir__(self) -> list[str]:
        """Return a list of attributes associated with the TimeSeries class.

        This method extends the default attributes of the TimeSeries class by including the column names of the
        underlying DataFrame and the time column name. This allows for dynamic attribute access to DataFrame columns
        using dot notation or introspection tools like `dir()`.

        Returns:
            A sorted list of attributes, combining the Default attributes of the class along with the names of the
            DataFrame's columns.
        """
        default_attrs = list(super().__dir__())
        custom_attrs = default_attrs + self.columns + [self.time_name]
        return sorted(set(custom_attrs))
