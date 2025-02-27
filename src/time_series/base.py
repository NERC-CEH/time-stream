from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Sequence, Type, Union

import polars as pl

from time_series.columns import DataColumn, FlagColumn, PrimaryTimeColumn, SupplementaryColumn, TimeSeriesColumn
from time_series.flag_manager import TimeSeriesFlagManager
from time_series.period import Period
from time_series.relationships import RelationshipManager

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
        flag_systems: Optional[Union[Dict[str, dict], Dict[str, Type[Enum]]]] = None,
        flag_columns: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        column_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise a TimeSeries instance.

        Args:
            df: The Polars DataFrame containing the time series data.
            time_name: The name of the time column in the DataFrame.
            resolution: The resolution of the time series.
            periodicity: The periodicity of the time series.
            time_zone: The time zone of the time series.
            supplementary_columns: Columns within the dataframe that are considered supplementary.
            flag_systems: A dictionary defining the flagging systems that can be used for flag columns.
            flag_columns: Columns within the dataframe that are considered flag columns.
                          Mapping of {column name: flag system name}.
            metadata: Metadata relevant to the overall time series, e.g. network, site ID, license, etc.
            column_metadata: The metadata of the variables within the time series.
        """
        self._time_name = time_name
        self._resolution = resolution
        self._periodicity = periodicity
        self._time_zone = time_zone

        self._flag_manager = TimeSeriesFlagManager(self, flag_systems)
        self._columns: dict[str, TimeSeriesColumn] = {}
        self._metadata = {}
        self._df = df

        self._setup(supplementary_columns or [], flag_columns or {}, column_metadata or {}, metadata or {})
        self._relationship_manager = RelationshipManager(self)

    def _setup(
        self,
        supplementary_columns: list[str],
        flag_columns: dict[str, str],
        column_metadata: dict[str, dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """
        Performs the initial setup for the TimeSeries instance.

        This method:
        - Sets the time zone of the time column.
        - Sorts the DataFrame by the time column.
        - Validates the time resolution.
        - Validates the periodicity of the time column.
        - Initializes supplementary and flag columns.

        Args:
            supplementary_columns: A list of column names that are considered supplementary.
            flag_columns: A dictionary mapping flag column names to their corresponding flag systems.
            column_metadata: A dictionary containing metadata for columns in the DataFrame.
            metadata: The user defined metadata for this time series instance.
        """
        self._set_time_zone()
        self.sort_time()
        self._validate_resolution()
        self._validate_periodicity()
        self._setup_columns(supplementary_columns, flag_columns, column_metadata)
        self._setup_metadata(metadata)

    def _setup_columns(
        self,
        supplementary_columns: list[str] = None,
        flag_columns: dict[str, str] = None,
        column_metadata: dict[str, dict[str, Any]] = None,
    ) -> None:
        """
        Initializes column classifications for the TimeSeries instance.

        This method:
        - Validates that all specified supplementary columns exist in the DataFrame.
        - Validates that all specified flag columns exist in the DataFrame.
        - Classifies each column into its appropriate type: primary time column, supplementary, flag, or data column.

        Args:
            supplementary_columns: A list of column names that are considered supplementary.
            flag_columns: A dictionary mapping flag column names to their corresponding flag systems.
            column_metadata: A dictionary containing metadata for columns in the DataFrame.

        Raises:
            KeyError: If any specified supplementary or flag column does not exist in the DataFrame.
        """
        # Validate that all supplementary columns exist in the DataFrame
        if supplementary_columns is None:
            supplementary_columns = []
        if flag_columns is None:
            flag_columns = {}
        if column_metadata is None:
            column_metadata = {}

        invalid_supplementary_columns = set(supplementary_columns) - set(self.df.columns)
        if invalid_supplementary_columns:
            raise KeyError(f"Invalid supplementary columns: {invalid_supplementary_columns}")

        # Validate that all flag columns exist in the DataFrame
        invalid_flag_columns = set(flag_columns) - set(self.df.columns)
        if invalid_flag_columns:
            raise KeyError(f"Invalid flag columns: {invalid_flag_columns}")

        # Classify each column in the DataFrame
        for col_name in self.df.columns:
            col_metadata = column_metadata.get(col_name)

            if col_name == self.time_name:
                time_col = PrimaryTimeColumn(col_name, self, col_metadata)
                self._columns[col_name] = time_col

            elif col_name in supplementary_columns:
                supplementary_col = SupplementaryColumn(col_name, self, col_metadata)
                self._columns[col_name] = supplementary_col

            elif col_name in flag_columns:
                flag_system = flag_columns[col_name]
                flag_col = FlagColumn(col_name, self, flag_system, col_metadata)
                self._columns[col_name] = flag_col

            else:
                data_col = DataColumn(col_name, self, col_metadata)
                self._columns[col_name] = data_col

    def _setup_metadata(self, metadata: dict[str, Any]) -> None:
        """Configure metadata for the Time Series instance.

        Processes a dictionary of metadata entries and adds them to the instance's internal metadata storage
        (`_metadata`). Verifies that the key does not conflict with any existing column names stored in `_columns`.
        KeyError is raised to prevent naming collisions between metadata and column identifiers.

        Args:
            metadata: The user defined metadata for this time series instance.

        Raises:
            KeyError: If a metadata key is already present in the instance's `_columns`.
        """
        for k, v in metadata.items():
            if k in self._df.columns:
                raise KeyError(f"Metadata key {k} exists as a Column in the Time Series.")
            self._metadata[k] = v

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
                f'Values in time field: "{self.time_name}" are not aligned to resolution: {self.resolution}'
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
                f'Values in time field: "{self.time_name}" do not conform to periodicity: {self.periodicity}'
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
        self._add_new_columns(new_df)

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
        """Remove reference to columns that are missing in the new DataFrame.

        Args:
            new_df: The new Polars DataFrame to compare against the current DataFrame.
        """
        removed_columns = list(set(self.df.columns) - set(new_df.columns))
        for col_name in removed_columns:
            self._relationship_manager._handle_deletion(self._columns[col_name])
            self._columns.pop(col_name, None)

    def _add_new_columns(self, new_df: pl.DataFrame) -> None:
        """Add reference to any new columns that are present in the new DataFrame.

        Assumption is all new columns are data columns.

        Args:
            new_df: The new Polars DataFrame to compare against the current DataFrame.
        """
        new_columns = list(set(new_df.columns) - set(self.df.columns))
        for col_name in new_columns:
            # Add a placeholder for the new column, so that the Column init knows there is a new column expected.
            self._columns[col_name] = None

            # Now add the actual column.
            data_col = DataColumn(col_name, self, {})
            self._columns[col_name] = data_col

    @property
    def time_column(self) -> Union[PrimaryTimeColumn, None]:
        """The primary time column of the TimeSeries."""
        time_column = [col for col in self._columns.values() if isinstance(col, PrimaryTimeColumn)]
        num_cols = len(time_column)
        if num_cols == 1:
            return time_column[0]
        elif num_cols == 0:
            raise ValueError("No primary time column found.")
        elif num_cols > 1:
            raise ValueError(f"Multiple primary time columns found: {time_column}")

    @property
    def data_columns(self) -> dict[str, TimeSeriesColumn]:
        columns = {col.name: col for col in self._columns.values() if type(col) is DataColumn}
        return columns

    @property
    def supplementary_columns(self) -> dict[str, TimeSeriesColumn]:
        columns = {col.name: col for col in self._columns.values() if type(col) is SupplementaryColumn}
        return columns

    @property
    def flag_columns(self) -> dict[str, TimeSeriesColumn]:
        columns = {col.name: col for col in self._columns.values() if type(col) is FlagColumn}
        return columns

    @property
    def columns(self) -> dict[str, TimeSeriesColumn]:
        columns = {col.name: col for col in self._columns.values() if type(col) is not PrimaryTimeColumn}
        return columns

    def select(self, col_names: list[str]) -> "TimeSeries":
        """Filter TimeSeries instance to include only the specified columns.

        Args:
            col_names: A list of column names to retain in the updated TimeSeries.

        Returns:
            New TimeSeries instance with only selected columns.
        """
        if not col_names:
            raise KeyError("No columns specified.")

        invalid_columns = set(col_names) - set(self.df.columns)
        if invalid_columns:
            raise KeyError(f"Invalid columns found: {invalid_columns}")

        new_df = self.df.select([self.time_name] + col_names)
        new_columns = [self.columns[col] for col in new_df.columns if col in self.columns]

        # Construct new time series object based on updated columns.  Need to build new column mappings etc.
        new_supplementary_columns = [col.name for col in new_columns if col.name in self.supplementary_columns]
        new_flag_columns = {col.name: col.flag_system for col in new_columns if col.name in self.flag_columns}
        new_flag_systems = {k: v for k, v in self.flag_systems.items() if k in new_flag_columns.values()}
        new_metadata = {col.name: col.metadata() for col in new_columns}

        ts = TimeSeries(
            df=new_df,
            time_name=self.time_name,
            resolution=self.resolution,
            periodicity=self.periodicity,
            time_zone=self.time_zone,
            supplementary_columns=new_supplementary_columns,
            flag_columns=new_flag_columns,
            flag_systems=new_flag_systems,
            column_metadata=new_metadata,
        )

        return ts

    def init_supplementary_column(
        self,
        col_name: str,
        data: Optional[Union[int, float, str, Iterable]] = None,
        dtype: Optional[pl.DataType] = None,
    ) -> None:
        """Add a new column to the TimeSeries DataFrame, marking it as a supplementary column.

        Args:
            col_name: The name of the new supplementary column.
            data: The data to populate the column. Can be a scalar, an iterable, or None.
                  If iterable, must have the same length as the current TimeSeries.
                  If None, the column will be filled with `None`.
            dtype: The data type to use. If set to None (default), the data type is inferred from the values input.

        Raises:
            KeyError: If the column name already exists in the DataFrame.
        """
        if col_name in self.columns:
            raise KeyError(f"Column '{col_name}' already exists in the DataFrame.")

        if data is None or isinstance(data, (float, int, str)):
            data = pl.lit(data, dtype=dtype)
        else:
            if dtype is None:
                data = pl.Series(col_name, data)
            else:
                data = pl.Series(col_name, data).cast(dtype)

        self.df = self.df.with_columns(data.alias(col_name))
        supplementary_col = SupplementaryColumn(col_name, self)
        self._columns[col_name] = supplementary_col

    def set_supplementary_column(self, col_name: Union[str, list[str]]) -> None:
        """Mark the specified existing column(s) as a supplementary column.

        Args:
            col_name: A column name (or list of column names) to mark as a supplementary column.
        """
        if isinstance(col_name, str):
            col_name = [col_name]

        for col in col_name:
            self.columns[col].set_as_supplementary()

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

    def metadata(self, key: Optional[Sequence[str]] = None, strict: bool = True) -> Dict[str, Any]:
        """Retrieve metadata for all or specific keys.

        Args:
            key: A specific key or list/tuple of keys to filter the metadata. If None, all metadata is returned.
            strict: If True, raises a KeyError when a key is missing.  Otherwise, missing keys return None.
        Returns:
            A dictionary of the requested metadata.

        Raises:
            KeyError: If the requested key(s) are not found in the metadata.
        """
        if not key:
            return self._metadata

        if isinstance(key, str):
            key = [key]

        result = {}
        for k in key:
            value = self._metadata.get(k)
            if strict and value is None:
                raise KeyError(f"Metadata key '{k}' not found")
            result[k] = value
        return result

    def column_metadata(
        self, column: Optional[Union[str, Sequence[str]]] = None, key: Optional[Union[str, Sequence[str]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve metadata for a given column(s), for all or specific keys.

        Args:
            column: A specific column or list of columns to filter the metadata. If None, all columns are returned.
            key: A specific key or list/tuple of keys to filter the metadata. If None, all metadata is returned.

        Returns:
            A dictionary of the requested metadata.

        Raises:
            KeyError: If the requested key(s) are not found in the metadata of any column.
        """
        if isinstance(column, str):
            column = [column]
        elif column is None:
            column = self.columns.keys()

        if isinstance(key, str):
            key = [key]

        result = {col: self.columns[col].metadata(key, strict=False) for col in column}

        missing_keys = {
            k
            for col, metadata in result.items()
            for k, v in metadata.items()
            if all(metadata.get(k) is None for metadata in result.values())
        }

        if missing_keys:
            raise KeyError(f"Metadata key(s) '{missing_keys}' not found in any column.")

        return result

    def add_column_relationship(self, primary: str, other: Union[str, list[str]]) -> None:
        """Adds a relationship between the primary column and other column(s).

        Args:
            primary: The primary column in the relationship.
            other: Other column(s) to associate.
        """
        if isinstance(other, str):
            other = [other]

        primary = self.columns[primary]
        primary.add_relationship(other)

    def remove_column_relationship(self, primary: str, other: Union[str, list[str]]) -> None:
        """Removes a relationship between the primary column and other column(s).

        Args:
            primary: The primary column in the relationship.
            other: Other column(s) to remove association.
        """
        if isinstance(other, str):
            other = [other]

        primary = self.columns[primary]
        for other_column in other:
            primary.remove_relationship(other_column)

    def get_flag_system_column(
        self, data_column: Union[str, TimeSeriesColumn], flag_system: str
    ) -> Optional["TimeSeriesColumn"]:
        """Retrieves the flag system column corresponding to the given data column and flag system.

        Args:
            data_column: The data column identifier. This can either be the name of the column or an instance
                            of TimeSeriesColumn.
            flag_system: The name of the flag system.

        Returns:
            The matching flag column if exactly one match is found, or None if no matching column is found.
        """
        if isinstance(data_column, str):
            data_column = self.columns.get(data_column, None)

        if data_column is None:
            # raise KeyError(f"Column '{data_column}' not found.")
            # TODO: decide whether to raise error or just return None
            return None

        if type(data_column) is not DataColumn:
            # raise TypeError(f"Column '{data_column}' not a data column.")
            # TODO: decide whether to raise error or just return None
            return None

        return data_column.get_flag_system_column(flag_system)

    def __getattr__(self, name: str) -> Any:
        """Dynamically handle attribute access for the TimeSeries object.

        This method provides convenience for accessing data columns or the time column of the DataFrame.
        It supports the following behaviors:

        - If the attribute name matches the time column, it returns the time column as a Polars Series.
        - If the attribute name matches a column in the DataFrame (excluding the time column), it selects that column
            and returns a new TimeSeries instance.
        - If the attribute name does not match a column, it assumes this is a Metadata key. Return that.

        Args:
            name: The attribute name being accessed.

        Returns:
            If `name` is:
              - The time column: A Polars Series containing the time data.
              - A non-time column: The TimeSeries instance with that column selected.
              - Metadata key: The metadata value for that key.

        Raises:
            AttributeError: If attribute does not match a column or the time column or a metadata key.

        Examples:
            >>> ts.timestamp  # Access the time column (assumed time_name set to "timestamp")
            <Polars Series: "timestamp">

            >>> ts.temperature  # Access a column named "temperature"
            <TimeSeries object, filtered to only contain the "temperature" data column>

            >>> ts.site_id
            <Metadata value for the given key>
        """
        try:
            # Delegate flag-related (non-private) calls to the flag manager
            if hasattr(self._flag_manager, name) and not name.startswith("_"):
                return getattr(self._flag_manager, name)

            # Otherwise, look for name within columns
            if name == self.time_name:
                # If the attribute name matches the time column, return the PrimaryTimeColumn
                return self.time_column
            elif name in self.columns:
                # If the attribute name matches a column in the DataFrame (excluding the time column),
                # select that Column
                return self.columns[name]
            elif name not in self.columns:
                # If the attribute name does not match a column, it assumes this is a Metadata key.
                return self.metadata(name, strict=True)[name]
            else:
                raise AttributeError

        except (KeyError, AttributeError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: Union[str, list[str], tuple[str]]) -> Union["TimeSeries", PrimaryTimeColumn]:
        """Access columns or the time column using indexing syntax.

        This method enables convenient access to DataFrame columns or the time column by using indexing. It supports:

        - A single column name: Selects and returns a TimeSeries instance with the specified column.
        - A list/tuple of column names: Selects and returns a TimeSeries instance with the specified columns.
        - The time column name: Returns the time column as a Polars Series.

        Args:
            key: A single column name, a list/tuple of column names, or the time column name.

        Returns:
            - If `key` is a column name: A TimeSeries instance with the specified column(s) selected.
            - If `key` is the time column name: The PrimaryTimeColumn object.

        Examples:
            >>> ts["timestamp"]  # Access the time column (assumed time_name set to "timestamp")
            <Polars Series: "timestamp">

            >>> ts["temperature"]  # Access a single column
            <TimeSeries object, filtered to only contain the "temperature" data column>

            >>> ts[["temperature", "pressure"]]  # Access multiple columns
            <TimeSeries object, filtered to only contain the "temperature" and "pressure" data columns>
        """
        if key == self.time_name:
            return self.time_column
        if isinstance(key, str):
            key = [key]
        return self.select(key)

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def __len__(self) -> int:
        """Get the number of rows in the time series."""
        return self.df.height

    def __iter__(self) -> Iterator:
        """Return an iterator over the rows of the DataFrame."""
        return self.df.iter_rows()

    def __str__(self) -> str:
        """Return the string representation of the TimeSeries dataframe."""
        return self.df.__str__()

    def __repr__(self) -> str:
        """Returns a string representation of the TimeSeries instance, summarising key properties."""
        return (
            f"TimeSeries("
            f"time_name={self.time_name}, "
            f"resolution={self.resolution}, "
            f"periodicity={self.periodicity}, "
            f"time_zone={self.time_zone}, "
            f"data_columns={list(self.data_columns.keys())}, "
            f"supplementary_columns={list(self.supplementary_columns.keys())}, "
            f"flag_columns={list(self.flag_columns.keys())}, "
        )

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
        custom_attrs = default_attrs + list(self.columns.keys()) + [self.time_name] + list(self._metadata.keys())
        return sorted(set(custom_attrs))

    def __eq__(self, other: object) -> bool:
        """Checks if two TimeSeries instances are equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the TimeSeries instances are equal, False otherwise.
        """
        if not isinstance(other, TimeSeries):
            return False

        # Compare DataFrames
        if not self.df.equals(other.df):
            return False

        # Compare core metadata attributes
        if (
            self.time_name != other.time_name
            or self.resolution != other.resolution
            or self.periodicity != other.periodicity
            or self._metadata != other._metadata
        ):
            return False

        # Compare flag systems
        if self.flag_systems.keys() != other.flag_systems.keys():
            return False
        for name, flag_system in self.flag_systems.items():
            other_flag_system = other.flag_systems[name]
            if str(flag_system) != str(other_flag_system) or flag_system.__members__ != other_flag_system.__members__:
                return False

        # Compare column mappings
        if (
            self.data_columns.keys() != other.data_columns.keys()
            or self.supplementary_columns.keys() != other.supplementary_columns.keys()
            or self.flag_columns.keys() != other.flag_columns.keys()
        ):
            return False

        return True

    def __ne__(self, other: object) -> bool:
        """Checks if two TimeSeries instances are not equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the TimeSeries instances are not equal, False otherwise.
        """
        return not self.__eq__(other)
