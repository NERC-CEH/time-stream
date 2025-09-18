from datetime import datetime
from typing import Any, Iterator, Self, Sequence, Type

import polars as pl

from time_stream.aggregation import AggregationFunction
from time_stream.bitwise import BitwiseFlag
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import MetadataError
from time_stream.flag_manager import FlagColumn, FlagManager
from time_stream.infill import InfillMethod
from time_stream.period import Period
from time_stream.qc import QCCheck
from time_stream.time_manager import TimeManager
from time_stream.utils import configure_period_object, pad_time


class TimeSeries:  # noqa: PLW1641 ignore hash warning
    """A class representing a time series data model, with data held in a Polars DataFrame."""

    _df: pl.DataFrame
    _time_manager: TimeManager
    _metadata: dict
    _flag_manager: FlagManager = FlagManager()

    def __init__(
        self,
        df: pl.DataFrame,
        time_name: str,
        resolution: Period | str | None = None,
        periodicity: Period | str | None = None,
        time_anchor: TimeAnchor | str = TimeAnchor.START,
        metadata: dict[str, Any] | None = None,
        on_duplicates: DuplicateOption | str = DuplicateOption.ERROR,
    ) -> None:
        """Initialise a TimeSeries instance.

        Args:
            df: The `Polars` DataFrame containing the time series data.
            time_name: The name of the time column in the DataFrame.
            resolution: The resolution of the time series.
            periodicity: The periodicity of the time series.
            time_anchor: The time anchor to which the date/times of the time series conform to.
            metadata: Metadata relevant to the overall time series, e.g., network, site ID, license, etc.
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

        self._metadata = metadata
        self._setup()

    def _setup(self) -> None:
        """Performs the initial setup for the TimeSeries instance."""
        # NOTE: Set _df directly, otherwise the df setter will complain that the time column is mutated.
        #       In this instance, we are happy as we know we are mutating the time values to handle duplicate values.
        self._df = self._time_manager._handle_time_duplicates()
        self._time_manager.validate()
        self.sort_time()

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
        """Update the underlying DataFrame of the TimeSeries, checking the integrity of the time values hasn't
        been compromised

        Args:
            new_df: The new Polars DataFrame to set as the time series data.
        """
        old_df = self._df.clone()
        self._time_manager._check_time_integrity(old_df, new_df)
        self._df = new_df

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
        """Return all the data columns of the TimeSeries."""
        return [c for c in self.columns if c not in self.flag_columns]

    def sort_time(self) -> None:
        """Sort the TimeSeries DataFrame by the time column."""
        self._df = self.df.sort(self.time_name)

    def pad(self) -> None:
        """Pad the time series with missing datetime rows, filling in NULLs for missing values."""
        self._df = pad_time(self.df, self.time_name, self.periodicity, self.time_anchor)
        self.sort_time()

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

    def register_flag_column(self, name: str, base: str, flag_system: str) -> None:
        """Mark the specified existing column as a flag column.

        This does not modify the DataFrame; it only records that ``name`` is a flag column associated with the
        value column ``base``, with values handled by the flag system ``flag_system``.

        Args:
            name: A column name to mark as a flag column.
            base: Name of the value/data column this flag column refers to.
            flag_system: The name of the flag system.
        """
        self._flag_manager.register_flag_column(name, base, flag_system)

    def init_flag_column(
        self, base: str, flag_system: str, name: str | None = None, data: int | Sequence[int] = 0
    ) -> None:
        """ "Add a new column to the TimeSeries DataFrame, setting it as a Flag Column.

        Args:
            base: Name of the value/data column this flag column will refer to.
            flag_system: The name of the flag system.
            name: Optional name for the new flag column. If omitted, a name of the
                    form ``f"{base}__flag__{flag_system}"`` is used.
            data: The default value to populate the flag column with. Can be a scalar or list-like. Defaults to 0.
        """
        # Validate that the flag system exists
        self.get_flag_system(flag_system)

        # Set up data that will go into the new column
        if isinstance(data, int):
            data = pl.lit(data, dtype=pl.Int64)
        else:
            data = pl.Series(data, dtype=pl.Int64)

        # Determine name of flag column
        if not name:
            name = f"{base}__flag__{flag_system}"

        # Add and register as a flag column
        self.df = self.df.with_columns(data.alias(name))
        self.register_flag_column(name, base, flag_system)

    def get_flag_column(self, name: str) -> FlagColumn:
        """Look up a registered flag column by name.

        Args:
            name: Flag column name.

        Returns:
            The corresponding ``FlagColumn`` object.
        """
        return self._flag_manager.get_flag_column(name)

    def add_flag(self, name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """Add flag value (if not there) to flag column, where expression is True.

        Args:
            name: The name of the flag column
            flag_value: The flag value to add
            expr: Polars expression for which rows to add flag to
        """
        flag_column = self.get_flag_column(name)
        self.df = flag_column.add_flag(self.df, flag_value, expr)

    def remove_flag(self, name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """
        Remove flag value (if there) from flag column.

        Args:
            name: The name of the flag column
            flag_value: The flag value to remove
            expr: Polars expression for which rows to remove flag from
        """
        flag_column = self.get_flag_column(name)
        self.df = flag_column.remove_flag(self.df, flag_value, expr)

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

        return True
