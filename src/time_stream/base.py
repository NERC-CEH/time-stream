from datetime import datetime
from typing import Any, Iterator, Self, Sequence, Type

import polars as pl

from time_stream.aggregation import AggregationFunction
from time_stream.bitwise import BitwiseFlag
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import ColumnNotFoundError, MetadataError
from time_stream.flag_manager import FlagColumn, FlagManager
from time_stream.infill import InfillMethod
from time_stream.metadata import MetadataStore
from time_stream.period import Period
from time_stream.qc import QCCheck
from time_stream.time_manager import TimeManager
from time_stream.utils import check_columns_in_dataframe, configure_period_object, pad_time


class TimeSeries:  # noqa: PLW1641 ignore hash warning
    """A class representing a time series data model, with data held in a Polars DataFrame."""

    _df: pl.DataFrame
    _time_manager: TimeManager
    _flag_manager: FlagManager
    _metadata_store: MetadataStore

    def __init__(
        self,
        df: pl.DataFrame,
        time_name: str,
        resolution: Period | str | None = None,
        periodicity: Period | str | None = None,
        time_anchor: TimeAnchor | str = TimeAnchor.START,
        on_duplicates: DuplicateOption | str = DuplicateOption.ERROR,
    ) -> None:
        """Initialise a TimeSeries instance.

        Args:
            df: The `Polars` DataFrame containing the time series data.
            time_name: The name of the time column in the DataFrame.
            resolution: The resolution of the time series.
            periodicity: The periodicity of the time series.
            time_anchor: The time anchor to which the date/times of the time series conform to.
            on_duplicates: What to do if duplicate rows are found in the data. Default to ERROR.
        """
        self._time_manager = TimeManager(
            time_name=time_name,
            resolution=resolution,
            periodicity=periodicity,
            on_duplicates=on_duplicates,
            time_anchor=time_anchor,
        )

        self._metadata_store = MetadataStore()
        self._flag_manager = FlagManager()

        self._df = self._time_manager._handle_time_duplicates(df)
        self._time_manager.validate(self.df)
        self.sort_time()

    def copy(self, share_df: bool = True) -> Self:
        """Return a shallow copy of this ``TimeSeries``, either sharing or cloning the underlying DataFrame.

        Args:
            share_df: If True, the copy references the same DataFrame object. If False, a cloned DataFrame is used.

        Returns:
            A copy of this TimeSeries
        """
        df = self.df if share_df else self.df.clone()
        out = TimeSeries(
            df,
            time_name=self.time_name,
            resolution=self.resolution,
            periodicity=self.periodicity,
            time_anchor=self.time_anchor,
        )
        out._metadata_store = self._metadata_store.copy()
        out._flag_manager = self._flag_manager.copy()

        return out

    def with_series_metadata(self, metadata: dict[str, Any]) -> Self:
        """Return a new ``TimeSeries`` with timeseries-level metadata.

        Args:
            metadata: Mapping of arbitrary keys/values describing the time series as a whole.

        Returns:
            A new TimeSeries with timeseries-level metadata has set to the provided metadata.
        """
        ts = self.copy()
        ts._metadata_store.set_series_metadata(metadata)
        return ts

    def with_column_metadata(self, metadata: dict[str, dict[str, Any]]) -> Self:
        """Return a new ``TimeSeries`` with column-level metadata.

        Args:
            metadata: Mapping of column names to a dict of arbitrary keys/values describing the column.

        Returns:
            A new TimeSeries with column-level metadata has set to the provided metadata.
        """
        check_columns_in_dataframe(self.df, metadata.keys())
        ts = self.copy()
        for column_name, meta in metadata.items():
            ts._metadata_store.set_column_metadata(column_name, meta)
        return ts

    def with_df(self, new_df: pl.DataFrame) -> Self:
        """Return a new TimeSeries with a new DataFrame, checking the integrity of the time values hasn't
        been compromised between the old and new TimeSeries.

        Args:
            new_df: The new Polars DataFrame to set as the new time series data.
        """
        old_df = self._df.clone()
        self._time_manager._check_time_integrity(old_df, new_df)
        ts = self.copy()
        ts._df = new_df
        return ts

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

    @property
    def columns(self) -> list[str]:
        """Return all the columns of the TimeSeries."""
        return [c for c in self.df.columns if c != self.time_name]

    @property
    def flag_columns(self) -> list[str]:
        """Return all the flag columns of the TimeSeries."""
        return list(self._flag_manager.flag_columns.keys())

    @property
    def data_columns(self) -> list[str]:
        """Return all the data columns of the TimeSeries (essentially any column that isn't the time column
        or a flag column).
        """
        return [c for c in self.columns if c not in self.flag_columns]

    def sort_time(self) -> None:
        """Sort the TimeSeries DataFrame by the time column."""
        self._df = self.df.sort(self.time_name)

    def pad(self) -> None:
        """Pad the time series with missing datetime rows, filling in NULLs for missing values."""
        self._df = pad_time(self.df, self.time_name, self.periodicity, self.time_anchor)
        self.sort_time()

    def set_series_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        """Alias for `time_stream.metadata.MetadataStore.set_series_metadata`."""
        self._metadata_store.set_series_metadata(metadata)

    def update_series_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        """Alias for `time_stream.metadata.MetadataStore.update_series_metadata`."""
        self._metadata_store.update_series_metadata(metadata)

    def set_column_metadata(self, column_name: str, metadata: dict[str, Any]) -> None:
        """Alias for `time_stream.metadata.MetadataStore.set_column_metadata`."""
        check_columns_in_dataframe(self.df, column_name)
        self._metadata_store.set_column_metadata(column_name, metadata)

    def update_column_metadata(self, column_name: str, metadata: dict[str, Any]) -> None:
        """Alias for `time_stream.metadata.MetadataStore.update_column_metadata`."""
        check_columns_in_dataframe(self.df, column_name)
        self._metadata_store.update_column_metadata(column_name, metadata)

    def metadata(self, keys: str | Sequence[str] | None = None, strict: bool = False) -> dict[str, Any]:
        """Retrieve TimeSeries-level metadata for all or specific keys.

        Usage:
            ts.metadata()                          -> full series-level metadata dict
            ts.metadata("title")                   -> single value
            ts.metadata(["title", "source"])       -> multiple values

        Args:
            keys: A specific key or sequence of keys to filter the metadata. If None, all metadata is returned.
            strict: If True, raises a KeyError when a key is missing.  Otherwise, missing keys return None.

        Returns:
            A dictionary of the requested metadata.
        """
        series_metadata = self._metadata_store.get_series_metadata()
        if keys is None:
            return series_metadata

        if isinstance(keys, str):
            keys = [keys]

        result = {}
        for k in keys:
            value = series_metadata.get(k)
            if strict and value is None:
                raise MetadataError(f"Metadata key '{k}' not found")
            result[k] = value
        return result

    def column_metadata(
        self, column_name: str, keys: str | Sequence[str] | None = None, strict: bool = False
    ) -> dict[str, Any]:
        """Retrieve metadata for a given column, for all or specific keys.

        Usage:
            ts.column_metadata("flow")                     -> full dict for the column 'flow'
            ts.column_metadata("flow", "unit")             -> single value
            ts.column_metadata("flow", ["unit","sensor"])  -> multiple values

        Args:
            column_name: A specific column or sequence of columns to filter the metadata.
                            If None, all columns are returned.
            keys: A specific key or sequence of keys to filter the metadata. If None, all metadata is returned.
            strict: If True, raises a KeyError when a key is missing.  Otherwise, missing keys return None.

        Returns:
            A dictionary of the requested metadata.
        """
        check_columns_in_dataframe(self.df, column_name)
        column_metadata = self._metadata_store.get_column_metadata(column_name)
        if keys is None:
            return column_metadata

        if isinstance(keys, str):
            keys = [keys]

        result = {}
        for k in keys:
            value = column_metadata.get(k)
            if strict and value is None:
                raise MetadataError(f"Metadata key '{k}' not found for column '{column_name}'")
            result[k] = value
        return result

    def register_flag_system(self, name: str, flag_system: dict[str, int] | type[BitwiseFlag]) -> None:
        """Register a named flag system with the internal flag manager.

        Args:
            name: Short name for the flag system
            flag_system: The flag system to register, provided either as:
                            - a dict mapping of flag names to single-bit integer values, or;
                            - a ``BitwiseFlag`` enum class, whose members are single-bit integers.
        """
        self._flag_manager.register_flag_system(name, flag_system)

    def get_flag_system(self, name: str) -> type[BitwiseFlag]:
        """Return a registered flag system.

        Args:
            name: The registered flag system name

        Returns:
            The ``BitwiseFlag`` enum that defines the flag system.
        """
        return self._flag_manager.get_flag_system(name)

    def register_flag_column(self, column_name: str, base: str, flag_system: str) -> None:
        """Mark the specified existing column as a flag column.

        This does not modify the DataFrame; it only records that ``name`` is a flag column associated with the
        value column ``base``, with values handled by the flag system ``flag_system``.

        Args:
            column_name: A column name to mark as a flag column.
            base: Name of the value/data column this flag column refers to.
            flag_system: The name of the flag system.
        """
        check_columns_in_dataframe(self.df, [column_name, base])
        self._flag_manager.register_flag_column(column_name, base, flag_system)

    def init_flag_column(
        self, base: str, flag_system: str, column_name: str | None = None, data: int | Sequence[int] = 0
    ) -> None:
        """ "Add a new column to the TimeSeries DataFrame, setting it as a Flag Column.

        Args:
            base: Name of the value/data column this flag column will refer to.
            flag_system: The name of the flag system.
            column_name: Optional name for the new flag column. If omitted, a name of the
                    form ``f"{base}__flag__{flag_system}"`` is used.
            data: The default value to populate the flag column with. Can be a scalar or list-like. Defaults to 0.
        """
        check_columns_in_dataframe(self.df, base)

        # Validate that the flag system exists
        self.get_flag_system(flag_system)

        # Set up data that will go into the new column
        if isinstance(data, int):
            data = pl.lit(data, dtype=pl.Int64)
        else:
            data = pl.Series(data, dtype=pl.Int64)

        # Determine name of flag column
        if not column_name:
            column_name = f"{base}__flag__{flag_system}"

        # Add and register as a flag column
        self._df = self.df.with_columns(data.alias(column_name))
        self.register_flag_column(column_name, base, flag_system)

    def get_flag_column(self, column_name: str) -> FlagColumn:
        """Look up a registered flag column by name.

        Args:
            column_name: Flag column name.

        Returns:
            The corresponding ``FlagColumn`` object.
        """
        return self._flag_manager.get_flag_column(column_name)

    def add_flag(self, column_name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """Add flag value (if not there) to flag column, where expression is True.

        Args:
            column_name: The name of the flag column
            flag_value: The flag value to add
            expr: Polars expression for which rows to add flag to
        """
        flag_column = self.get_flag_column(column_name)
        self._df = flag_column.add_flag(self.df, flag_value, expr)

    def remove_flag(self, column_name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """
        Remove flag value (if there) from flag column.

        Args:
            column_name: The name of the flag column
            flag_value: The flag value to remove
            expr: Polars expression for which rows to remove flag from
        """
        flag_column = self.get_flag_column(column_name)
        self._df = flag_column.remove_flag(self.df, flag_value, expr)

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

        new_ts = TimeSeries(
            df=agg_df,
            time_name=self.time_name,
            resolution=aggregation_period,
            periodicity=aggregation_period,
            time_anchor=aggregation_time_anchor,
        )
        new_ts.set_series_metadata(self.metadata())
        return new_ts

    def qc_check(
        self,
        check: str | Type[QCCheck] | QCCheck,
        column_name: str,
        observation_interval: tuple[datetime, datetime | None] | None = None,
        into: str | bool = False,
        **kwargs,
    ) -> "TimeSeries | pl.Series":
        """Apply a quality control check to the TimeSeries.

        Args:
            check: The QC check to apply.
            column_name: The column to perform the check on.
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
        qc_result = check_instance.apply(self.df, self.time_name, column_name, observation_interval)

        # Return the boolean series, if requested
        if not into:
            return qc_result

        # Determine the name of the column for the QC result
        if isinstance(into, str):
            qc_result_col_name = into
        else:
            qc_result_col_name = f"__qc__{column_name}__{check_instance.name}"

        if qc_result_col_name in self.df.columns:
            # Auto-suffix the column name to avoid accidental overwrite
            col_suffix = 1
            while f"{qc_result_col_name}__{col_suffix}" in self.df.columns:
                col_suffix += 1
            qc_result_col_name = f"{qc_result_col_name}__{col_suffix}"

        # Create a copy of the current TimeSeries, and update the dataframe with the QC result
        new_df = self.df.with_columns(pl.Series(qc_result_col_name, qc_result))
        new_ts = self.with_df(new_df)
        return new_ts

    def infill(
        self,
        infill_method: str | Type[InfillMethod] | InfillMethod,
        column_name: str,
        observation_interval: tuple[datetime, datetime | None] | None = None,
        max_gap_size: int | None = None,
        **kwargs,
    ) -> Self:
        """Apply an infilling method to a column in the TimeSeries to fill in missing data.

        Args:
            infill_method: The method to use for infilling
            column_name: The column to infill
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
            self.df, self.time_name, self.periodicity, column_name, observation_interval, max_gap_size
        )

        # Create result TimeSeries
        # Create a copy of the current TimeSeries, and update the dataframe with the infilled data
        new_ts = self.with_df(infill_result)
        return new_ts

    def select(
        self,
        column_names: str | Sequence[str],
        include_flag_columns: bool = True,
    ) -> Self:
        """Return a new TimeSeries instance to include only the specified columns.

        By default, this:
          - carries over **series-level metadata** unchanged,
          - prunes **column-level metadata** to the kept columns,
          - rebuilds the **flag manager** to include only kept flag columns.

        Args:
            column_names:  Column name(s) to retain in the updated TimeSeries.
            include_flag_columns: If True, include any registered flag columns whose ``base`` is among the
                                  kept value columns. If a specific flag column name is explicitly requested
                                  in ``columns`` it is always kept.

        Returns:
             New TimeSeries instance with only selected columns.
        """
        if not column_names:
            raise ColumnNotFoundError("No columns specified.")

        if isinstance(column_names, str):
            column_names = [column_names]
        check_columns_in_dataframe(self.df, column_names)

        # Include primary time column (if not already included)
        if self.time_name not in column_names:
            column_names.insert(0, self.time_name)

        # Optionally include associated flag columns
        if include_flag_columns:
            for flag_name, flag_column in self._flag_manager.flag_columns.items():
                # include if its base (value col) is being kept
                if flag_column.base in column_names:
                    column_names.append(flag_name)

        # Build new frame
        new_df = self.df.select(column_names)

        # New time series
        new_ts = self.with_df(new_df)

        # Prune column level metadata to kept columns
        kept_metadata = {col: self.column_metadata(col) for col in column_names}
        new_ts._metadata_store.reset_column_metadata()
        new_ts = new_ts.with_column_metadata(kept_metadata)

        # Rebuild the flag registry for kept columns
        new_flag_manager = FlagManager()
        # re-register flag systems
        for sys_name, enum_cls in self._flag_manager.flag_systems.items():
            new_flag_manager.register_flag_system(sys_name, enum_cls)

        # keep only flag columns that survived
        for flag_name, flag_column in self._flag_manager.flag_columns.items():
            if flag_name in column_names:
                new_flag_manager.register_flag_column(flag_name, flag_column.base, flag_column.flag_system_name)

        new_ts._flag_manager = new_flag_manager

        return new_ts

    def __getitem__(self, key: str | Sequence[str]) -> Self:
        """Access columns using indexing syntax.

        Args:
            key: Column name(s) to access

        Returns:
            A new TimeSeries instance with the specified column(s) selected.

        Notes:
            This is equivalent to ``ts.select([...])``.
        """
        if isinstance(key, str):
            key = [key]
        return self.select(key, include_flag_columns=False)

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
        """Returns the representation of the TimeSeries"""
        return self.df.__repr__()

    def __copy__(self) -> Self:
        return self.copy(share_df=True)

    def __deepcopy__(self, memo: dict) -> Self:
        return self.copy(share_df=False)

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
            or self.time_anchor != other.time_anchor
        ):
            return False

        return True
