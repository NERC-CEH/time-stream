import copy
from abc import ABC
from typing import TYPE_CHECKING, Any, Self, Sequence, Type

import polars as pl

from time_stream.exceptions import ColumnNotFoundError, ColumnTypeError, MetadataError

if TYPE_CHECKING:
    # Import is for type hinting only.  Make sure there is no runtime import, to avoid recursion.
    from time_stream import TimeSeries


class TimeSeriesColumn(ABC):  # noqa: PLW1641 ignore hash warning
    """Base class for all column types in a TimeSeries."""

    def __init__(self, name: str, ts: "TimeSeries", metadata: dict[str, Any] | None = None) -> None:
        """Initializes a TimeSeriesColumn instance.

        Args:
            name: The name of the column.
            ts: The TimeSeries instance this column belongs to.
            metadata: Optional metadata for the column.
        """
        self._name = name
        self._ts = ts
        self._validate_name()

        #  NOTE: Doing a deep copy of this mutable object, otherwise the original object will refer to the same
        #   object in memory and will be changed by class methods.
        self._metadata = copy.deepcopy(metadata) or {}

    @property
    def name(self) -> str:
        return self._name

    def _validate_name(self) -> None:
        """Check if column name is within the TimeSeries.

        Raises:
            ColumnNotFoundError: If the column name is not within the TimeSeries.
        """
        if self.name not in self._ts.df.columns:
            raise ColumnNotFoundError(f"Column '{self.name}' not found in TimeSeries.")

    def metadata(self, key: str | Sequence[str] | None = None, strict: bool = True) -> dict[str, Any]:
        """Retrieve metadata for all or specific keys.

        Args:
            key: A specific key or sequence of keys to filter the metadata. If None, all metadata is returned.
            strict: If True, raises a MetadataError when a key is missing.  Otherwise, missing keys return None.
        Returns:
            A dictionary of the requested metadata.

        Raises:
            MetadataError: If the requested key(s) are not found in the metadata.
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

    def remove_metadata(self, key: str | Sequence[str] | None = None) -> None:
        """Removes metadata associated with a column, either completely or for specific keys.

        Args:
            key: A specific key or sequence of keys to remove. If None, all metadata for the column is removed.
        """
        if isinstance(key, str):
            key = [key]

        if key is None:
            self._metadata = {}
        else:
            self._metadata = {k: v for k, v in self.metadata().items() if k not in key}

    def _set_as(self, column_type: Type[Self], *args, **kwargs) -> Self:
        """Converts the column to a specified column type while preserving metadata.

        Args:
            column_type: Type of TimeSeriesColumn to convert to.
            args: Arguments passed to column_type init.
            kwargs: Keyword arguments passed to column_type init.

        Returns:
            The converted column object.
        """
        if type(self) is column_type:
            return self

        column = column_type(*args, **kwargs)
        self._ts._columns[self.name] = column

        return column

    def set_as_supplementary(self) -> Self:
        """Converts the column to a SupplementaryColumn while preserving metadata.

        If the column is already a SupplementaryColumn, no changes are made.

        Returns:
            The supplementary column object.
        """
        column = self._set_as(SupplementaryColumn, self.name, self._ts, self.metadata())
        return column

    def set_as_flag(self, flag_system: str) -> Self:
        """Converts the column to a FlagColumn while preserving metadata.

        If the column is already a FlagColumn, no changes are made.

        Args:
            flag_system: The name of the flag system used for flagging.

        Returns:
            The flag column object.
        """
        column = self._set_as(FlagColumn, self.name, self._ts, flag_system, self.metadata())
        return column

    def unset(self) -> Self:
        """Converts the column back into a normal DataColumn while preserving metadata.

        If the column is already a DataColumn, no changes are made.

        Returns:
            The data column object.
        """
        column = self._set_as(DataColumn, self.name, self._ts, self.metadata())
        return column

    def remove(self) -> None:
        """Removes this column from the TimeSeries.

        - Drops the column from the DataFrame (if it exists).
        - Clear reference to time series to prevent further modifications
        """
        if self.name in self._ts.columns:
            self._ts.df = self._ts.df.drop(self.name)
        self._ts = None

    def add_flag(self, *args, **kwargs) -> None:
        """Placeholder method for FlagColumn add_flag. Raises an error if attempting to add a flag to a column that
        is not a FlagColumn.

        Raises:
            ColumnTypeError: If the column is not a FlagColumn.
        """
        raise ColumnTypeError(
            f"Column '{self.name}' is not a FlagColumn. Use `set_as_flag(flag_system_name)` first to enable flagging."
        )

    def remove_flag(self, *args, **kwargs) -> None:
        """Placeholder method for FlagColumn remove_flag. Raises an error if attempting to remove a flag from a
        column that is not a FlagColumn.

        Raises:
            ColumnTypeError: If the column is not a FlagColumn.
        """
        raise ColumnTypeError(
            f"Column '{self.name}' is not a FlagColumn. Use `set_as_flag(flag_system_name)` first to enable flagging."
        )

    @property
    def data(self) -> pl.DataFrame:
        """Returns a DataFrame containing the time column and the current column.

        This ensures that data is always retrieved alongside the primary time column.

        Returns:
            pl.DataFrame: A DataFrame with the time column and the current column.

        Raises:
            ColumnNotFoundError: If the column does not exist in the DataFrame.
        """
        if self.name not in self._ts.df.columns:
            raise ColumnNotFoundError(f"Column '{self.name}' not found in TimeSeries.")

        return self._ts.df.select([self._ts.time_name, self.name])

    def as_timeseries(self) -> "TimeSeries":
        """Returns a new TimeSeries instance containing only this column.

        This method filters the original TimeSeries to include only the time column and this column.

        Returns:
            TimeSeries: A new TimeSeries object with just the selected column.

        Raises:
            ColumnNotFoundError: If the column does not exist in the TimeSeries.
        """
        if self.name not in self._ts.df.columns:
            raise ColumnNotFoundError(f"Column '{self.name}' not found in TimeSeries.")

        return self._ts.select([self.name])

    def __str__(self) -> str:
        """Return the string representation of the Column."""
        return str(self.data)

    def __repr__(self) -> str:
        """Returns a string representation of the Colum instance, summarising key properties."""
        return f"{type(self).__name__}('{self.name}')"

    def __getattr__(self, name: str) -> Any:
        """Dynamically handle metadata attribute access for the Column object.

        Args:
            name: The attribute name being accessed.

        Returns:
            Metadata key: The metadata value for the column.

        Raises:
            AttributeError: If attribute not found.
        """
        try:
            return self.metadata(name, strict=True)[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __eq__(self, other: object) -> bool:
        """Check equality between two TimeSeriesColumn instances.

        Two instances are considered equal if they have the same name, reference the same TimeSeries object,
        and have identical data and metadata.

        Args:
            other: The object to compare against.

        Returns:
            True if both objects are equal, otherwise False.
        """
        if not isinstance(other, TimeSeriesColumn):
            return NotImplemented

        if type(self) is not type(other):
            return False

        if (
            self.name != other.name
            or self._ts is not other._ts
            or self._metadata != other._metadata
            or not self.data.equals(other.data)
        ):
            return False

        if type(self) is FlagColumn and (self.flag_system != other.flag_system):
            return False

        return True


class PrimaryTimeColumn(TimeSeriesColumn):
    """Represents the primary datetime column that controls the Time Series.
    This column is immutable and cannot be converted to other column types.
    """

    def set_as_supplementary(self, *args, **kwargs) -> Self:
        """Raises an error because the primary time column cannot be converted to a supplementary column.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot be converted to a supplementary column.")

    def set_as_flag(self, *args, **kwargs) -> Self:
        """Raises an error because the primary time column cannot be converted to a flag column.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot be converted to a flag column.")

    def unset(self, *args, **kwargs) -> Self:
        """Raises an error because the primary time column cannot be unset.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot be unset.")

    @property
    def data(self) -> pl.Series:
        """Returns the time column as a Polars Series.

        Returns:
            pl.Series: The Polars Series containing the time column.
        """
        return self._ts.df[self.name]


class DataColumn(TimeSeriesColumn):
    """Represents primary data columns."""


class SupplementaryColumn(TimeSeriesColumn):
    """Represents supplementary columns (e.g., metadata, extra information)."""


class FlagColumn(SupplementaryColumn):
    """Represents a flag column within a TimeSeries.

    A flag column uses a predefined flagging system to store quality control indicators
    or other categorical flags. Flags are stored using bitwise operations for efficiency.
    """

    def __init__(self, name: str, ts: "TimeSeries", flag_system: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Initializes a FlagColumn.

        Args:
            name: The name of the flag column.
            ts: The TimeSeries instance this column belongs to.
            flag_system: The name of the flag system used for flagging.
            metadata: Optional metadata associated with this column.
        """
        super().__init__(name, ts, metadata)
        self.flag_system = self._ts.flag_systems[flag_system]

    def add_flag(self, flag: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """Adds a flag value to this FlagColumn using a bitwise OR operation.

        Args:
            flag: The flag value to add.
            expr: A Polars expression defining the condition for applying the flag.
                  Defaults to `pl.lit(True)`, meaning all rows are flagged.
        """
        flag = self.flag_system.get_single_flag(flag)
        self._ts.df = self._ts.df.with_columns(
            pl.when(expr).then(pl.col(self.name) | flag.value).otherwise(pl.col(self.name))
        )

    def remove_flag(self, flag: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """Remove a flag value from this FlagColumn using a bitwise AND operation.

        Args:
            flag: The flag value to remove.
            expr: A Polars expression defining the condition for removing the flag.
                  Defaults to `pl.lit(True)`, meaning flag remove from all rows.
        """
        flag = self.flag_system.get_single_flag(flag)
        self._ts.df = self._ts.df.with_columns(
            pl.when(expr).then(pl.col(self.name) & ~flag.value).otherwise(pl.col(self.name))
        )
