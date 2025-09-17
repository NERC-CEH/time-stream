from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Iterator, Self, Sequence, Type

import polars as pl

from time_stream.aggregation import AggregationFunction
from time_stream.columns import DataColumn, FlagColumn, PrimaryTimeColumn, SupplementaryColumn, TimeSeriesColumn
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import ColumnNotFoundError, DuplicateColumnError, MetadataError
from time_stream.flag_manager import TimeSeriesFlagManager
from time_stream.infill import InfillMethod
from time_stream.period import Period
from time_stream.qc import QCCheck
from time_stream.time_manager import TimeManager
from time_stream.utils import check_columns_in_dataframe, configure_period_object, pad_time


class TimeSeries:  # noqa: PLW1641 ignore hash warning
    """A class representing a time series data model, with data held in a Polars DataFrame."""

    def __init__(
        self,
        df: pl.DataFrame,
        time_name: str,
        resolution: Period | str | None = None,
        periodicity: Period | str | None = None,
        time_anchor: TimeAnchor | str = TimeAnchor.START,
        supplementary_columns: list | None = None,
        flag_systems: dict[str, dict] | dict[str, Type[Enum]] | None = None,
        flag_columns: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        column_metadata: dict[str, dict[str, Any]] | None = None,
        on_duplicates: DuplicateOption | str = DuplicateOption.ERROR,
    ) -> None:
        """Initialise a TimeSeries instance.

        Args:
            df: The `Polars` DataFrame containing the time series data.
            time_name: The name of the time column in the DataFrame.
            resolution: The resolution of the time series.
            periodicity: The periodicity of the time series.
            time_anchor: The time anchor to which the date/times of the time series conform to.
            supplementary_columns: Columns within the dataframe that are considered supplementary.
            flag_systems: A dictionary defining the flagging systems that can be used for flag columns.
            flag_columns: Columns within the dataframe that are considered flag columns.
                          Mapping of {column name: flag system name}.
            metadata: Metadata relevant to the overall time series, e.g., network, site ID, license, etc.
            column_metadata: The metadata of the variables within the time series.
            on_duplicates: What to do if duplicate rows are found in the data. Default to ERROR.
        """
        self._df = df

        self._time_manager = TimeManager(
            get_df=lambda: self.df,
            time_name=time_name,
            resolution=resolution,
            periodicity=periodicity,
            on_duplicates=on_duplicates,
            time_anchor=time_anchor,
        )

        self._flag_manager = TimeSeriesFlagManager(self, flag_systems)
        self._columns: dict[str, TimeSeriesColumn] = {}
        self._metadata = {}

        self._setup(supplementary_columns or [], flag_columns or {}, column_metadata or {}, metadata or {})

    def _setup(
        self,
        supplementary_columns: list[str],
        flag_columns: dict[str, str],
        column_metadata: dict[str, dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        """Performs the initial setup for the TimeSeries instance.

        This method:
        - Initializes supplementary and flag columns.
        - Initializes metadata.
        - Handles any potentially duplicated rows (based on a user option)
        - Validates the resolution of the time column.
        - Validates the periodicity of the time column.
        - Sorts the DataFrame by the time column.

        Args:
            supplementary_columns: A list of column names that are considered supplementary.
            flag_columns: A dictionary mapping flag column names to their corresponding flag systems.
            column_metadata: A dictionary containing metadata for columns in the DataFrame.
            metadata: The user defined metadata for this time series instance.
        """
        # Doing this first as we need info on columns before doing certain things in the validation steps below
        self._setup_columns(supplementary_columns, flag_columns, column_metadata)
        self._setup_metadata(metadata)

        # NOTE: Set _df directly, otherwise the df setter will complain that the time column is mutated.
        #       In this instance, we are happy as we know we are mutating the time values to handle duplicate values.
        self._df = self._time_manager._handle_time_duplicates()

        self._time_manager.validate()
        self.sort_time()

    def _setup_columns(
        self,
        supplementary_columns: list[str] = None,
        flag_columns: dict[str, str] = None,
        column_metadata: dict[str, dict[str, Any]] = None,
    ) -> None:
        """Initializes column classifications for the TimeSeries instance.

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
        if supplementary_columns is None:
            supplementary_columns = []
        if flag_columns is None:
            flag_columns = {}
        if column_metadata is None:
            column_metadata = {}

        # Validate that all supplementary and flag columns exist in the DataFrame
        check_columns_in_dataframe(self.df, supplementary_columns)
        check_columns_in_dataframe(self.df, flag_columns.keys())

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
                raise MetadataError(f"Metadata key '{k}' exists as a Column in the Time Series.")
            self._metadata[k] = v

    @property
    def time_name(self) -> str:
        """The name of the primary datetime column in the underlying TimeSeries DataFrame."""
        return self._time_manager.time_name

    @property
    def resolution(self) -> Period:
        return self._time_manager.resolution

    @property
    def periodicity(self) -> Period:
        return self._time_manager.periodicity

    @property
    def time_anchor(self) -> TimeAnchor:
        return self._time_manager.time_anchor

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @df.setter
    def df(self, new_df: pl.DataFrame) -> None:
        self._update_df(new_df)

    def _update_df(self, new_df: pl.DataFrame) -> None:
        """Update the underlying DataFrame while preserving the integrity of primary datetime field, metadata, and
        supplementary column settings.

        Checks that the time column has not changed, and removes any metadata and or supplementary column
        settings for any columns missing in the new DataFrame.

        Args:
            new_df: The new Polars DataFrame to set as the time series data.
        """
        old_df = self._df.clone()
        self._time_manager._check_time_integrity(old_df, new_df)
        self._df = new_df
        self._remove_missing_columns(old_df, new_df)
        self._add_new_columns(old_df, new_df)

    def _remove_missing_columns(self, old_df: pl.DataFrame, new_df: pl.DataFrame) -> None:
        """Remove reference to columns that are missing in the new DataFrame.

        Args:
            old_df: The old Polars DataFrame to compare against the new DataFrame.
            new_df: The new Polars DataFrame to compare against the old DataFrame.
        """
        removed_columns = list(set(old_df.columns) - set(new_df.columns))
        for col_name in removed_columns:
            self._columns.pop(col_name, None)

    def _add_new_columns(self, old_df: pl.DataFrame, new_df: pl.DataFrame) -> None:
        """Add reference to any new columns that are present in the new DataFrame.

        Assumption is all new columns are data columns.

        Args:
            old_df: The old Polars DataFrame to compare against the new DataFrame.
            new_df: The new Polars DataFrame to compare against the old DataFrame.
        """
        new_columns = list(set(new_df.columns) - set(old_df.columns))
        for col_name in new_columns:
            # Add a placeholder for the new column, so that the Column init knows there is a new column expected.
            self._columns[col_name] = None

            # Now add the actual column.
            data_col = DataColumn(col_name, self, {})
            self._columns[col_name] = data_col

    @property
    def time_column(self) -> PrimaryTimeColumn:
        """Return the primary time column of the TimeSeries."""
        time_column = [col for col in self._columns.values() if isinstance(col, PrimaryTimeColumn)]
        num_cols = len(time_column)
        if num_cols != 1:
            raise ColumnNotFoundError("No single primary time column found.")
        return time_column[0]

    @property
    def data_columns(self) -> dict[str, TimeSeriesColumn]:
        """Return the data columns of the TimeSeries."""
        columns = {col.name: col for col in self._columns.values() if type(col) is DataColumn}
        return columns

    @property
    def supplementary_columns(self) -> dict[str, TimeSeriesColumn]:
        """Return the supplementary columns of the TimeSeries."""
        columns = {col.name: col for col in self._columns.values() if type(col) is SupplementaryColumn}
        return columns

    @property
    def flag_columns(self) -> dict[str, TimeSeriesColumn]:
        """Return the flag columns of the TimeSeries."""
        columns = {col.name: col for col in self._columns.values() if type(col) is FlagColumn}
        return columns

    @property
    def columns(self) -> dict[str, TimeSeriesColumn]:
        """Return all the columns of the TimeSeries."""
        columns = {col.name: col for col in self._columns.values() if type(col) is not PrimaryTimeColumn}
        return columns

    def sort_time(self) -> None:
        """Sort the TimeSeries DataFrame by the time column."""
        self._df = self.df.sort(self.time_name)

    def pad(self) -> None:
        """Pad the time series with missing datetime rows, filling in NULLs for missing values."""
        self._df = pad_time(self.df, self.time_name, self.periodicity, self.time_anchor)
        self.sort_time()

    def select(self, col_names: list[str]) -> Self:
        """Filter TimeSeries instance to include only the specified columns.

        Args:
            col_names: A list of column names to retain in the updated TimeSeries.

        Returns:
            New TimeSeries instance with only selected columns.
        """
        if not col_names:
            raise ColumnNotFoundError("No columns specified.")

        check_columns_in_dataframe(self.df, col_names)

        new_df = self.df.select([self.time_name] + col_names)
        new_columns = [self.columns[col] for col in new_df.columns if col in self.columns]

        # Construct a new time series object based on updated columns.  Need to build new column mappings etc.
        new_supplementary_columns = [col.name for col in new_columns if col.name in self.supplementary_columns]
        new_flag_columns = {col.name: col.flag_system for col in new_columns if col.name in self.flag_columns}
        new_flag_systems = {k: v for k, v in self.flag_systems.items() if k in new_flag_columns.values()}
        new_metadata = {col.name: col.metadata() for col in new_columns}

        return TimeSeries(
            df=new_df,
            time_name=self.time_name,
            resolution=self.resolution,
            periodicity=self.periodicity,
            time_anchor=self.time_anchor,
            supplementary_columns=new_supplementary_columns,
            flag_columns=new_flag_columns,
            flag_systems=new_flag_systems,
            column_metadata=new_metadata,
            metadata=self._metadata,
            on_duplicates=self._time_manager._on_duplicates,
        )

    def init_supplementary_column(
        self,
        col_name: str,
        data: int | float | str | Iterable | None = None,
        dtype: pl.DataType | None = None,
    ) -> None:
        """Add a new column to the TimeSeries DataFrame, marking it as a supplementary column.

        Args:
            col_name: The name of the new supplementary column.
            data: The data to populate the column. It can be a scalar, an iterable, or None.
                  If iterable, it must have the same length as the current TimeSeries.
                  If None, the column will be filled with `None`.
            dtype: The data type to use. If set to None (default), the data type is inferred from the `values` input.

        Raises:
            KeyError: If the column name already exists in the DataFrame.
        """
        if col_name in self.columns:
            raise DuplicateColumnError(f"Column '{col_name}' already exists in the DataFrame.")

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

    def set_supplementary_column(self, col_name: str | list[str]) -> None:
        """Mark the specified existing column(s) as a supplementary column.

        Args:
            col_name: A column name (or list of column names) to mark as a supplementary column.
        """
        if isinstance(col_name, str):
            col_name = [col_name]

        for col in col_name:
            self.columns[col].set_as_supplementary()

    def aggregate(
        self,
        aggregation_period: Period | str,
        aggregation_function: str | Type[AggregationFunction] | AggregationFunction,
        columns: str | list[str],
        missing_criteria: tuple[str, float | int] | None = None,
        aggregation_time_anchor: TimeAnchor | None = None,
    ) -> Self:
        """Apply an aggregation function to a column in this TimeSeries, check the aggregation satisfies user
        requirements and return a new derived TimeSeries containing the aggregated data.

        Args:
            aggregation_period: The period over which to aggregate the data
            aggregation_function: The aggregation function to apply
            columns: The column(s) containing the data to be aggregated
            missing_criteria: How the aggregation handles missing data
            aggregation_time_anchor: The time anchor for the aggregation result.

        Returns:
            A TimeSeries containing the aggregated data.
        """
        # Get the aggregation function instance and run the apply method
        agg_func = AggregationFunction.get(aggregation_function)
        aggregation_period = configure_period_object(aggregation_period)
        aggregation_time_anchor = TimeAnchor(aggregation_time_anchor) if aggregation_time_anchor else self.time_anchor

        agg_df = agg_func.apply(
            self.df,
            self.time_name,
            self.time_anchor,
            self.periodicity,
            aggregation_period,
            columns,
            missing_criteria=missing_criteria,
            aggregation_time_anchor=aggregation_time_anchor,
        )

        return TimeSeries(
            df=agg_df,
            time_name=self.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_anchor=aggregation_time_anchor,
            metadata=self._metadata.copy(),
        )

    def qc_check(
        self,
        check: str | Type[QCCheck] | QCCheck,
        column: str,
        observation_interval: tuple[datetime, datetime | None] | None = None,
        into: str | bool = False,
        **kwargs,
    ) -> "TimeSeries | pl.Series":
        """Apply a quality control check to the TimeSeries.

        Args:
            check: The QC check to apply.
            column: The column to perform the check on.
            observation_interval: Optional time interval to limit the check to.
            into: Whether to add the result of the QC to the TimeSeries dataframe (True | string name of column to add),
                  or just return a boolean Series of the QC result (False).
            **kwargs: Parameters specific to the check type.

        Returns:
            Result of the QC check, either as a boolean Series or added to the TimeSeries dataframe

        Examples:
            # Threshold check
            ts_flagged = ts.qc_check("comparison", "battery_voltage", compare_to=12.0, operator="<")

            # Range check
            ts_flagged = ts.qc_check("range", "temperature", min_value=-50, max_value=50)

            # Spike check
            ts_flagged = ts.qc_check("spike", "wind_speed", threshold=5.0)

            # Value check for error codes
            ts_flagged = ts.qc_check("comparison", "status_code", compare_to=[99, -999, "ERROR"])
        """
        # Get the QC check instance and run the apply method
        check_instance = QCCheck.get(check, **kwargs)
        qc_result = check_instance.apply(self.df, self.time_name, column, observation_interval)

        # Return the boolean series, if requested
        if not into:
            return qc_result

        # Determine the name of the column for the QC result
        if isinstance(into, str):
            qc_result_col_name = into
        else:
            qc_result_col_name = f"__qc__{column}__{check_instance.name}"

        if qc_result_col_name in self.df.columns:
            # Auto-suffix the column name to avoid accidental overwrite
            col_suffix = 1
            while f"{qc_result_col_name}__{col_suffix}" in self.df.columns:
                col_suffix += 1
            qc_result_col_name = f"{qc_result_col_name}__{col_suffix}"

        # Create a copy of the current TimeSeries, and update the dataframe with the QC result
        new_ts = self.select([*self.data_columns.keys(), *self.supplementary_columns.keys(), *self.flag_columns.keys()])
        new_ts.df = new_ts.df.with_columns(pl.Series(qc_result_col_name, qc_result))
        return new_ts

    def infill(
        self,
        infill_method: str | Type[InfillMethod] | InfillMethod,
        column: str,
        observation_interval: tuple[datetime, datetime | None] | None = None,
        max_gap_size: int | None = None,
        **kwargs,
    ) -> Self:
        """Apply an infilling method to a column in the TimeSeries to fill in missing data.

        Args:
            infill_method: The method to use for infilling
            column: The column to infill
            observation_interval: Optional time interval to limit the check to.
            max_gap_size: The maximum size of consecutive null gaps that should be filled. Any gap larger than this
                          will not be infilled and will remain as null.
            **kwargs: Parameters specific to the infill method.

        Returns:
            A TimeSeries containing the aggregated data.
        """
        # Get the infill method instance and run the apply method
        infill_instance = InfillMethod.get(infill_method, **kwargs)
        infill_result = infill_instance.apply(
            self.df, self.time_name, self.periodicity, column, observation_interval, max_gap_size
        )

        # Create result TimeSeries
        # Create a copy of the current TimeSeries, and update the dataframe with the infilled data
        new_ts = self.select([*self.data_columns.keys(), *self.supplementary_columns.keys(), *self.flag_columns.keys()])
        new_ts.df = infill_result
        return new_ts

    def metadata(self, key: str | Sequence[str] | None = None, strict: bool = True) -> dict[str, Any]:
        """Retrieve metadata for all or specific keys.

        Args:
            key: A specific key or sequence of keys to filter the metadata. If None, all metadata is returned.
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
                raise MetadataError(f"Metadata key '{k}' not found")
            result[k] = value
        return result

    def column_metadata(
        self, column: str | Sequence[str] | None = None, key: str | Sequence[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Retrieve metadata for a given column(s), for all or specific keys.

        Args:
            column: A specific column or sequence of columns to filter the metadata. If None, all columns are returned.
            key: A specific key or sequence of keys to filter the metadata. If None, all metadata is returned.

        Returns:
            A dictionary of the requested metadata.

        Raises:
            KeyError: If the requested key(s) is not found in the metadata of any column.
        """
        if isinstance(column, str):
            column = [column]
        elif column is None:
            column = self.columns.keys()

        if isinstance(key, str):
            key = [key]

        result = {col: self.columns[col].metadata(key, strict=False) for col in column}

        missing_keys = sorted(
            {
                k
                for col, metadata in result.items()
                for k, v in metadata.items()
                if all(metadata.get(k) is None for metadata in result.values())
            }
        )

        if missing_keys:
            raise MetadataError(f"Metadata key(s) {missing_keys} not found in any column.")

        return result

    def __getattr__(self, name: str) -> Any:
        """Dynamically handle attribute access for the TimeSeries object.

        This method provides convenience for accessing data columns or the time column of the DataFrame.
        It supports the following behaviors:

        - If the attribute name matches the time column, it returns the time column as a `Polars` Series.
        - If the attribute name matches a column in the DataFrame (excluding the time column), it selects that column
            and returns a new TimeSeries instance.
        - If the attribute name does not match a column, it assumes this is a Metadata key. Return that.

        Args:
            name: The attribute name being accessed.

        Returns:
            If `name` is:
              - The time column: A `Polars` Series containing the time data.
              - A non-time column: The TimeSeries instance with that column selected.
              - Metadata key: The metadata value for that key.

        Raises:
            AttributeError: If attribute does not match a column or the time column or a metadata key.

        Examples:
            ts.timestamp  # Access the time column (assumed time_name set to "timestamp")
            <Polars Series: "timestamp">

            ts.temperature  # Access a column named "temperature"
            <TimeSeries object, filtered to only contain the "temperature" data column>

            ts.site_id
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

        except (KeyError, AttributeError, MetadataError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str | Iterable[str]) -> "TimeSeries | PrimaryTimeColumn":
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
            ts["timestamp"]  # Access the time column (assumed time_name set to "timestamp")
            <Polars Series: "timestamp">

            ts["temperature"]  # Access a single column
            <TimeSeries object, filtered to only contain the "temperature" data column>

            ts[["temperature", "pressure"]]  # Access multiple columns
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
        """Return the number of rows in the time series."""
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
            f"time_anchor={self.time_anchor}, "
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
        """Check if two TimeSeries instances are equal.

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
