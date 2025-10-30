"""
Core TimeFrame Data Model.

This module defines the `TimeFrame` class, the central abstraction of the time_stream package.
A `TimeFrame` wraps a Polars DataFrame and adds functionality for handling temporal data, quality flags, metadata, and
derived operations.

Main responsibilities
---------------------
1. **Time management** (via `TimeManager`):
   - Validates that the time column exists and has a temporal dtype.
   - Enforces resolution and periodicity of the time series.
   - Uses time anchoring to define the period over which values are valid (`POINT`, `START`, `END`).
   - Detects duplicates and resolves them according to a strategy (`ERROR`, `KEEP_FIRST`, `KEEP_LAST`, `DROP`, `MERGE`).
   - Prevents mutation of time values between operations.

2. **Metadata management**:
   - TimeFrame-level metadata (`.metadata`).
   - Per-column metadata (`.column_metadata`) that stays in sync with the DataFrame.

3. **Flag management** (via `FlagManager`):
   - Register reusable flag systems (based on `BitwiseFlag` enums).
   - Initialise flag columns linked to data columns.
   - Add/remove flags with Polars expressions.

4. **Data operations**:
   - Aggregation: run `AggregationFunction` pipelines with support for missing-data criteria and time anchoring.
   - Quality control: apply `QCCheck` classes to detect anomalies or enforce validation rules.
   - Infilling: apply `InfillMethod` classes to fill missing values according to defined strategies.
   - Column selection: return reduced TimeFrame with metadata and flags synced consistently.
"""

from copy import deepcopy
from datetime import datetime
from typing import Any, Self, Sequence, Type

import polars as pl

from time_stream.aggregation import AggregationFunction
from time_stream.bitwise import BitwiseFlag
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import ColumnNotFoundError, MetadataError
from time_stream.flag_manager import FlagColumn, FlagManager, FlagSystemType
from time_stream.formatting import timeframe_repr
from time_stream.infill import InfillMethod
from time_stream.metadata import ColumnMetadataDict
from time_stream.period import Period
from time_stream.qc import QCCheck
from time_stream.time_manager import TimeManager
from time_stream.utils import check_columns_in_dataframe, configure_period_object, pad_time


class TimeFrame:
    """A class representing a time series data model, with data held in a Polars DataFrame.

    Args:
        df: The :class:`polars.DataFrame` containing the time-series data.
        time_name: The name of the time column in ``df``.
        resolution: Sampling interval for the timeseries; the unit of time step allowable between consecutive data
            points. Accepts a :class:`Period` or ISO-8601 duration string (e.g. ``"PT15M"``, ``"P1D"``, ``"P1Y"``).
            If ``None``, defaults to microsecond step (PT0.000001S) (effectively allows any set of datetime values).
        offset: Offset applied from the natural boundary of ``resolution`` to position the datetime values along the
            timeline. For example, you may have daily data (``resolution="P1D"``), but all the values are measured
            at 9:00am, an offset of 9 hours (``"+T9H"``) from the natural boundary of midnight 00:00.
            Accepts an offset string, following the principles of ISO-8601 but replacing the "P" with a "+"
            (e.g. ``"+T9H"``, `"+9MT9H"`). If ``None``, no offset is applied.
        periodicity: Defines the allowed "frequency" of datetimes in your timeseries, i.e., how many datetime
            entries are allowed within a given period of time. For example, you may have an annual maximum
            timeseries, where the individual data points are considered to be at daily resolution
            (``resolution="P1D"``), but are limited to only one data point per year (``periodicity="P1Y"``).
            Accepts a :class:`Period` or ISO-8601 duration string (e.g. ``"PT15M"``, ``"P1D"``, ``"P1Y"``) with an
            optional offset syntax (e.g. ``"P1D+T9H"``, ``"P1Y+9MT9H"``). If ``None``, it defaults to the period
            defined by ``resolution + offset``.
        time_anchor: Defines the window of time over which a given timestamp refers to. In the descriptions below,
            "t" is the time value, "r" stands for a single unit of the resolution of the data:

            - ``POINT``: The time stamp is anchored for the instant of time "t".
              A value at "t" is considered valid only for the instant of time "t".
            - ``START``: The time stamp is anchored starting at "t". A value at "t" is considered valid
              starting at "t" (inclusive) and ending at "t+r" (exclusive).
            - ``END``: The time stamp is anchored ending at "t". A value at "t" is considered valid
              starting at "t-r" (exclusive) and ending at "t" (inclusive)
        on_duplicates: What to do if duplicate rows are found in the data:

            - ``ERROR`` (default): Raise error
            - ``KEEP_FIRST``: Keep the first row of any duplicate groups.
            - ``KEEP_LAST``: Keep the last row of any duplicate groups.
            - ``DROP``: Drop all duplicate rows.
            - ``MERGE``: Merge duplicate rows using coalesce (the first non-null value for each column takes precedence)

    Examples:
        >>> # Simple 15 minute timeseries:
        >>> tf = TimeFrame(
        >>>     df, "timestamp", resolution="PT15M"
        >>> )
        >>> print(
        >>>     "resolution=", tf.resolution,
        >>>     " alignment=", tf.alignment,
        >>>     " periodicity=", tf.periodicity
        >>> )
        resoution=PT15M alignment=PT15M periodicity=PT15M

        >>> # Daily water day (09:00 to 09:00) with default uniqueness per water day:
        >>>
        >>> tf = TimeFrame(
        >>>     df, "timestamp", resolution="P1D", offset="+T9H"
        >>> )
        >>> print(
        >>>     "resolution=", tf.resolution,
        >>>     " alignment=", tf.alignment,
        >>>     " periodicity=", tf.periodicity
        >>> )
        resoution=P1D alignment=P1D+T9H periodicity=P1D+T9H

        >>> # Daily timestamps but uniqueness per water-year:
        >>>
        >>> tf = TimeFrame(
        >>>     df, "timestamp", resolution="P1D", offset="+T9H", periodicity="P1Y+P9MT9H"
        >>> )
        >>> print(
        >>>     "resolution=", tf.resolution,
        >>>     " alignment=", tf.alignment,
        >>>     " periodicity=", tf.periodicity
        >>> )
        resoution=P1D alignment=P1D+T9H periodicity=P1Y+P9MT9H

        >>> # Annual series stored directly on water-year boundary:
        >>>
        >>> tf = TimeFrame(
        >>>     df, "timestamp", resolution="P1Y", offset="+9MT9H"
        >>> )
        >>> print(
        >>>     "resolution=", tf.resolution,
        >>>     " alignment=", tf.alignment,
        >>>     " periodicity=", tf.periodicity
        >>> )
        resoution=P1Y alignment=P1D+9MT9H periodicity=P1Y+P9MT9H
    """

    _df: pl.DataFrame
    _time_manager: TimeManager
    _flag_manager: FlagManager
    _metadata: dict[str, Any]
    _column_metadata: ColumnMetadataDict

    def __init__(
        self,
        df: pl.DataFrame,
        time_name: str,
        resolution: Period | str | None = None,
        offset: str | None = None,
        periodicity: Period | str | None = None,
        time_anchor: TimeAnchor | str = TimeAnchor.START,
        on_duplicates: DuplicateOption | str = DuplicateOption.ERROR,
    ) -> None:
        self._time_manager = TimeManager(
            time_name=time_name,
            resolution=resolution,
            offset=offset,
            periodicity=periodicity,
            on_duplicates=on_duplicates,
            time_anchor=time_anchor,
        )

        self._df = self._time_manager._handle_time_duplicates(df)
        self._time_manager.validate(self.df)
        self.sort_time()

        self._metadata = {}
        self._column_metadata = ColumnMetadataDict(lambda: self.df.columns)
        self._flag_manager = FlagManager()

    def copy(self, share_df: bool = True) -> Self:
        """Return a shallow copy of this ``TimeFrame``, either sharing or cloning the underlying DataFrame.

        Args:
            share_df: If True, the copy references the same DataFrame object. If False, a cloned DataFrame is used.

        Returns:
            A copy of this TimeFrame
        """
        df = self.df if share_df else self.df.clone()
        out = TimeFrame(
            df,
            time_name=self.time_name,
            resolution=self.resolution,
            offset=self.offset,
            periodicity=self.periodicity,
            time_anchor=self.time_anchor,
        )

        out.metadata = deepcopy(self._metadata)
        out.column_metadata.update(deepcopy(self._column_metadata))

        out._flag_manager = self._flag_manager.copy()

        return out

    def with_df(self, new_df: pl.DataFrame) -> Self:
        """Return a new TimeFrame with a new DataFrame, checking the integrity of the time values hasn't
        been compromised between the old and new TimeFrame.

        Args:
            new_df: The new Polars DataFrame to set as the new time series data.
        """
        old_df = self._df.clone()
        self._time_manager._check_time_integrity(old_df, new_df)
        tf = self.copy()
        tf._df = new_df
        tf._column_metadata.sync()
        return tf

    def with_metadata(self, metadata: dict[str, Any]) -> Self:
        """Return a new TimeFrame with TimeFrame-level metadata.

        Args:
            metadata: Mapping of arbitrary keys/values describing the time series as a whole.

        Returns:
            A new TimeFrame with timeFrame-level metadata has set to the provided metadata.
        """
        tf = self.copy()
        tf.metadata = metadata
        return tf

    def with_column_metadata(self, metadata: dict[str, dict[str, Any]]) -> Self:
        """Return a new TimeFrame with column-level metadata.

        Args:
            metadata: Mapping of column names to a dict of arbitrary keys/values describing the column.

        Returns:
            A new TimeFrame with column-level metadata has set to the provided metadata.
        """
        tf = self.copy()
        tf.column_metadata.update(metadata)
        tf._column_metadata.sync()
        return tf

    def with_flag_system(self, name: str, flag_system: FlagSystemType) -> Self:
        """Return a new TimeFrame, with a flag system registered.

        Args:
            name: Short name for the flag system
            flag_system: The flag system to register

        Returns:
            A new TimeFrame with flag system set.
        """
        tf = self.copy()
        tf.register_flag_system(name, flag_system)
        return tf

    def with_periodicity(self, periodicity: str | Period) -> Self:
        """Return a new TimeFrame, with a new periodicity registered.

        Args:
            periodicity: The new periodicity

        Returns:
            A new TimeFrame with a new periodicity set.
        """
        tf = self.copy()
        tf._periodicity = periodicity
        tf._time_manager.validate(tf.df)

    @property
    def metadata(self) -> dict[str, Any]:
        """TimeFrame-level metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        """Set the TimeFrame-level metadata.

        This method checks type of object being set to ensure we continue to work with expected dicts.

        Args:
            value: The new metadata to set.
        """
        if value is None:
            self._metadata = {}
        elif isinstance(value, dict):
            self._metadata = value
        else:
            raise MetadataError(f"TimeFrame-level metadata must be a dict object. Got: '{type(value)}'")

    @metadata.deleter
    def metadata(self) -> None:
        """Clear TimeFrame-level metadata."""
        self._metadata.clear()

    @property
    def column_metadata(self) -> dict[str, dict[str, Any]]:
        """Per-column metadata."""
        return self._column_metadata

    @column_metadata.setter
    def column_metadata(self, value: dict[str, dict[str, Any]] | None) -> None:
        """Set the column-level metadata.

        This method checks type of object being set to ensure we continue to work with expected dicts.

        Args:
            value: The new metadata to set.
        """
        if value is None:
            # Reset all the columns metadata to empty dicts
            self._column_metadata = ColumnMetadataDict(lambda: self.df.columns)

        elif isinstance(value, dict):
            self.column_metadata.update(value)

        else:
            raise MetadataError(f"Column-level metadata must be a dict object. Got: '{type(value)}'")

    @column_metadata.deleter
    def column_metadata(self) -> None:
        """Clear all per-column metadata."""
        self._column_metadata.clear()

    @property
    def time_name(self) -> str:
        """The name of the primary datetime column in the underlying TimeFrame DataFrame."""
        return self._time_manager.time_name

    @property
    def resolution(self) -> Period:
        """The resolution of the timeseries data within the TimeFrame"""
        return self._time_manager.resolution

    @property
    def offset(self) -> str:
        """The offset of the time steps within the TimeFrame"""
        return self._time_manager.offset

    @property
    def alignment(self) -> Period:
        """The alignment of the time steps within the TimeFrame"""
        return self._time_manager.alignment

    @property
    def periodicity(self) -> Period:
        """The periodicity of the timeseries data within the TimeFrame"""
        return self._time_manager.periodicity

    @property
    def time_anchor(self) -> TimeAnchor:
        """The time anchor of the timeseries data within the TimeFrame"""
        return self._time_manager.time_anchor

    @property
    def df(self) -> pl.DataFrame:
        """The underlying ``Polars`` DataFrame containing the timeseries data."""
        return self._df

    @property
    def columns(self) -> list[str]:
        """All column labels of the DataFrame within the TimeFrame."""
        return [c for c in self.df.columns if c != self.time_name]

    @property
    def flag_columns(self) -> list[str]:
        """Only the labels for any flag columns within the TimeFrame."""
        return list(self._flag_manager.flag_columns.keys())

    @property
    def flag_systems(self) -> dict[str, type[BitwiseFlag]]:
        """The registered flag systems of this TimeFrame."""
        return self._flag_manager.flag_systems

    @property
    def data_columns(self) -> list[str]:
        """Only the labels for the data columns within the TimeFrame."""
        return [c for c in self.columns if c not in self.flag_columns]

    def sort_time(self) -> None:
        """Sort the TimeFrame DataFrame by the time column."""
        self._df = self.df.sort(self.time_name)

    def pad(self, start: datetime | None = None, end: datetime | None = None) -> Self:
        """Pad the time series with missing datetime rows, filling in NULLs for missing values.

        Args:
            start: The starting datetime value to pad time values from (inclusive). If not provided then the beginning
                of the dataframe will be used.
            end: The final datetime value to pad time values to (inclusive). If not provided then the beginning of the
                dataframe will be used.

        Returns:
            Padded TimeFrame
        """
        tf = self.copy()
        tf._df = pad_time(
            df=self.df,
            time_name=self.time_name,
            periodicity=self.periodicity,
            time_anchor=self.time_anchor,
            start=start,
            end=end,
        )
        tf.sort_time()
        return tf

    def register_flag_system(self, name: str, flag_system: FlagSystemType) -> None:
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
        """Add a new column to the TimeFrame DataFrame, setting it as a Flag Column.

        Args:
            base: Name of the value/data column this flag column will refer to.
            flag_system: The name of the flag system.
            column_name: Optional name for the new flag column. If omitted, a name of the
                    form "{base}__flag__{flag_system}" is used.
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
        self._column_metadata.sync()

    def get_flag_column(self, flag_column_name: str) -> FlagColumn:
        """Look up a registered flag column by name.

        Args:
            flag_column_name: Flag column name.

        Returns:
            The corresponding ``FlagColumn`` object.
        """
        return self._flag_manager.get_flag_column(flag_column_name)

    def add_flag(self, flag_column_name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """Add flag value (if not there) to flag column, where expression is True.

        Args:
            flag_column_name: The name of the flag column
            flag_value: The flag value to add
            expr: Polars expression for which rows to add flag to
        """
        flag_column = self.get_flag_column(flag_column_name)
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
        columns: str | list[str] | None = None,
        missing_criteria: tuple[str, float | int] | None = None,
        aggregation_time_anchor: TimeAnchor | None = None,
        **kwargs,
    ) -> Self:
        """Apply an aggregation function to a column in this TimeFrame, check the aggregation satisfies user
        requirements and return a new derived TimeFrame containing the aggregated data.

        Args:
            aggregation_period: The period over which to aggregate the data
            aggregation_function: The aggregation function to apply
            columns: The column(s) containing the data to be aggregated. If omitted, will use all data columns.
            missing_criteria: How the aggregation handles missing data
            aggregation_time_anchor: The time anchor for the aggregation result.
            **kwargs: Parameters specific to the aggregation function.

        Returns:
            A TimeFrame containing the aggregated data.
        """
        # Get the aggregation function instance and run the apply method
        agg_func = AggregationFunction.get(aggregation_function, **kwargs)
        aggregation_period = configure_period_object(aggregation_period)
        aggregation_time_anchor = TimeAnchor(aggregation_time_anchor) if aggregation_time_anchor else self.time_anchor

        if not columns:
            columns = self.data_columns

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

        # The resulting resolution and offset needs to be extracted from the aggregation period
        new_resolution = aggregation_period.without_offset()
        new_offset = aggregation_period.offset

        tf = TimeFrame(
            df=agg_df,
            time_name=self.time_name,
            resolution=new_resolution,
            offset=new_offset,
            periodicity=aggregation_period,
            time_anchor=aggregation_time_anchor,
        )
        tf.metadata = deepcopy(self.metadata)
        return tf

    def qc_check(
        self,
        check: str | Type[QCCheck] | QCCheck,
        column_name: str,
        observation_interval: tuple[datetime, datetime | None] | None = None,
        into: str | bool = False,
        **kwargs,
    ) -> "TimeFrame | pl.Series":
        """Apply a quality control check to the TimeFrame.

        Args:
            check: The QC check to apply.
            column_name: The column to perform the check on.
            observation_interval: Optional time interval to limit the check to.
            into: Whether to add the result of the QC to the TimeFrame dataframe (True | string name of column to add),
                  or just return a boolean Series of the QC result (False).
            **kwargs: Parameters specific to the check type.

        Returns:
            Result of the QC check, either as a boolean Series or added to the TimeFrame dataframe

        Examples:
            # Threshold check
            ts_flagged = tf.qc_check("comparison", "battery_voltage", compare_to=12.0, operator="<")

            # Range check
            ts_flagged = tf.qc_check("range", "temperature", min_value=-50, max_value=50)

            # Spike check
            ts_flagged = tf.qc_check("spike", "wind_speed", threshold=5.0)

            # Value check for error codes
            ts_flagged = tf.qc_check("comparison", "status_code", compare_to=[99, -999, "ERROR"])
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

        # Create a copy of the current TimeFrame, and update the dataframe with the QC result
        new_df = self.df.with_columns(pl.Series(qc_result_col_name, qc_result))
        return self.with_df(new_df)

    def infill(
        self,
        infill_method: str | Type[InfillMethod] | InfillMethod,
        column_name: str,
        observation_interval: tuple[datetime, datetime | None] | None = None,
        max_gap_size: int | None = None,
        **kwargs,
    ) -> Self:
        """Apply an infilling method to a column in the TimeFrame to fill in missing data.

        Args:
            infill_method: The method to use for infilling
            column_name: The column to infill
            observation_interval: Optional time interval to limit the check to.
            max_gap_size: The maximum size of consecutive null gaps that should be filled. Any gap larger than this
                          will not be infilled and will remain as null.
            **kwargs: Parameters specific to the infill method.

        Returns:
            A TimeFrame containing the aggregated data.
        """
        # Get the infill method instance and run the apply method
        infill_instance = InfillMethod.get(infill_method, **kwargs)
        infill_result = infill_instance.apply(
            self.df, self.time_name, self.periodicity, column_name, observation_interval, max_gap_size
        )

        # Create result TimeFrame
        # Create a copy of the current TimeFrame, and update the dataframe with the infilled data
        return self.with_df(infill_result)

    def select(
        self,
        column_names: str | Sequence[str],
        include_flag_columns: bool = True,
    ) -> Self:
        """Return a new TimeFrame instance to include only the specified columns.

        By default, this:
          - carries over TimeFrame-level metadata,
          - prunes column-level metadata to the kept columns,
          - rebuilds the flag manager to include only kept flag columns.

        Args:
            column_names:  Column name(s) to retain in the updated TimeFrame.
            include_flag_columns: If True, include any registered flag columns whose base is among the
                                  kept value columns.

        Returns:
             New TimeFrame instance with only selected columns.
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

        # New TimeFrame
        tf = self.with_df(new_df)

        # Prune column level metadata to kept columns
        kept_metadata = {col: self.column_metadata[col] for col in column_names}
        tf.column_metadata.clear()
        tf.column_metadata.update(kept_metadata)

        # Rebuild the flag registry for kept columns
        new_flag_manager = FlagManager()
        # re-register flag systems
        for name, flag_system in self._flag_manager.flag_systems.items():
            new_flag_manager.register_flag_system(name, flag_system.to_dict())

        # keep only flag columns that survived
        for flag_name, flag_column in self._flag_manager.flag_columns.items():
            if flag_name in column_names:
                new_flag_manager.register_flag_column(
                    flag_name, flag_column.base, flag_column.flag_system.system_name()
                )

        tf._flag_manager = new_flag_manager
        tf._column_metadata.sync()
        return tf

    def __getitem__(self, key: str | Sequence[str]) -> Self:
        """Access columns using indexing syntax.

        Args:
            key: Column name(s) to access

        Returns:
            A new TimeFrame instance with the specified column(s) selected.

        Notes:
            This is equivalent to ``tf.select([...])``.
        """
        if isinstance(key, str):
            key = [key]
        return self.select(key, include_flag_columns=False)

    def __str__(self) -> str:
        """Return the string representation of the TimeFrame dataframe."""
        return timeframe_repr(self)

    def __repr__(self) -> str:
        """Returns the representation of the TimeFrame"""
        return timeframe_repr(self)

    def __copy__(self) -> Self:
        return self.copy(share_df=True)

    def __deepcopy__(self, memo: dict) -> Self:
        return self.copy(share_df=False)

    def __eq__(self, other: object) -> bool:
        """Check if two TimeFrame instances are equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the TimeFrame instances are equal, False otherwise.
        """
        if not isinstance(other, TimeFrame):
            return False

        return (
            self.df.equals(other.df)
            and self.time_name == other.time_name
            and self.resolution == other.resolution
            and self.periodicity == other.periodicity
            and self.time_anchor == other.time_anchor
            and self._flag_manager == other._flag_manager
            and self.metadata == other.metadata
            and self.column_metadata == other.column_metadata
        )

    # Make class instances unhashable
    __hash__ = None
