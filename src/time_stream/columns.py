import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence, Type

import polars as pl

from time_stream.exceptions import ColumnNotFoundError, ColumnTypeError, MetadataError
from time_stream.relationships import DeletionPolicy, Relationship, RelationshipType

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

    def metadata(self, key: Sequence[str] = None, strict: bool = True) -> dict[str, Any]:
        """Retrieve metadata for all or specific keys.

        Args:
            key: A specific key or list/tuple of keys to filter the metadata. If None, all metadata is returned.
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

    def remove_metadata(self, key: str | list[str] | tuple[str, ...] | None = None) -> None:
        """Removes metadata associated with a column, either completely or for specific keys.

        Args:
            key: A specific key or list/tuple of keys to remove. If None, all metadata for the column is removed.
        """
        if isinstance(key, str):
            key = [key]

        if key is None:
            self._metadata = {}
        else:
            self._metadata = {k: v for k, v in self.metadata().items() if k not in key}

    def _set_as(
        self,
        column_type: Type["TimeSeriesColumn"],
        related_columns: "str | TimeSeriesColumn | list[TimeSeriesColumn | str] | None" = None,
        *args,
        **kwargs,
    ) -> "TimeSeriesColumn":
        """Converts the column to a specified column type while preserving metadata.

        Args:
            column_type: Type of TimeSeriesColumn to convert to.
            related_columns: Optional column(s) to relate to this converted column
            args: Arguments passed to column_type init.
            kwargs: Keyword arguments passed to column_type init.

        Returns:
            The converted column object.
        """
        if type(self) is column_type:
            return self

        column = column_type(*args, **kwargs)
        self._ts._columns[self.name] = column

        # Remove existing relationships from related columns
        for relationship in self.get_relationships():
            self.remove_relationship(relationship.other_column)

        # Add new relationships as specified by the user
        if related_columns:
            column.add_relationship(related_columns)

        return column

    def set_as_supplementary(
        self, related_columns: "str | TimeSeriesColumn | list[TimeSeriesColumn | str] | None" = None
    ) -> "TimeSeriesColumn":
        """Converts the column to a SupplementaryColumn while preserving metadata.

        If the column is already a SupplementaryColumn, no changes are made.

        Args:
            related_columns: Column(s) to be given a relationship with the new supplementary column.

        Returns:
            The supplementary column object.
        """
        column = self._set_as(SupplementaryColumn, related_columns, self.name, self._ts, self.metadata())
        return column

    def set_as_flag(
        self,
        flag_system: str,
        related_columns: "str | TimeSeriesColumn | list[TimeSeriesColumn | str] | None" = None,
    ) -> "TimeSeriesColumn":
        """Converts the column to a FlagColumn while preserving metadata.

        If the column is already a FlagColumn, no changes are made.

        Args:
            flag_system: The name of the flag system used for flagging.
            related_columns: Column(s) to be given a relationship with the new flag column.

        Returns:
            The flag column object.
        """
        column = self._set_as(FlagColumn, related_columns, self.name, self._ts, flag_system, self.metadata())
        return column

    def unset(self) -> "TimeSeriesColumn":
        """Converts the column back into a normal DataColumn while preserving metadata.

        If the column is already a DataColumn, no changes are made.

        Returns:
            The data column object.
        """
        column = self._set_as(DataColumn, None, self.name, self._ts, self.metadata())
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

    @abstractmethod
    def add_relationship(self, other: "str | TimeSeriesColumn | list[TimeSeriesColumn | str]") -> None:
        """Defines relationships between columns.

        Args:
            other: Column(s) to establish a relationship with.
        """
        pass

    def remove_relationship(self, other: "TimeSeriesColumn | str") -> None:
        """Remove relationships between columns.

        Args:
            other: Column(s) to remove relationship from.
        """
        if isinstance(other, str):
            other = self._ts.columns[other]

        for relationship in self.get_relationships():
            if relationship.other_column == other:
                self._ts._relationship_manager._remove(relationship)

    def get_relationships(self) -> list["Relationship"]:
        """Retrieves all relationships associated with this column.

        Returns:
            The list of Relationship objects for this column.
        """
        return self._ts._relationship_manager._get_relationships(self)

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

    def __dir__(self) -> list[str]:
        """Return a list of attributes associated with the TimeSeries class.

        This method extends the default attributes of the TimeSeries class by including the metadata keys of this
        Column. This allows for dynamic attribute access using dot notation or introspection tools like `dir()`.

        Returns:
            A sorted list of attributes, combining the Default attributes of the class along with the names of the
            Column's metadata keys.
        """
        default_attrs = list(super().__dir__())
        custom_attrs = default_attrs + list(self._metadata.keys())
        return sorted(set(custom_attrs))

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

    def __ne__(self, other: object) -> bool:
        """Checks if two TimeSeriesColumn instances are not equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the TimeSeriesColumn instances are not equal, False otherwise.
        """
        return not self.__eq__(other)


class PrimaryTimeColumn(TimeSeriesColumn):
    """Represents the primary datetime column that controls the Time Series.
    This column is immutable and cannot be converted to other column types.
    """

    def set_as_supplementary(self, *args, **kwargs) -> None:
        """Raises an error because the primary time column cannot be converted to a supplementary column.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot be converted to a supplementary column.")

    def set_as_flag(self, *args, **kwargs) -> None:
        """Raises an error because the primary time column cannot be converted to a flag column.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot be converted to a flag column.")

    def unset(self, *args, **kwargs) -> None:
        """Raises an error because the primary time column cannot be unset.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot be unset.")

    def add_relationship(self, *args, **kwargs) -> None:
        """Raises an error because the primary time column cannot set related columns.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot set related columns.")

    def remove_relationship(self, *args, **kwargs) -> None:
        """Raises an error because the primary time column cannot have other column relationships.

        Raises:
            NotImplementedError: Always raised because time columns are immutable.
        """
        raise NotImplementedError("Primary time column cannot unset related columns.")

    @property
    def data(self) -> pl.Series:
        """Returns the time column as a Polars Series.

        Returns:
            pl.Series: The Polars Series containing the time column.
        """
        return self._ts.df[self.name]


class DataColumn(TimeSeriesColumn):
    """Represents primary data columns."""

    def add_relationship(self, other: "str | TimeSeriesColumn | list[TimeSeriesColumn | str]") -> None:
        """Adds a relationship between this data column and supplementary or flag column(s).

        Args:
            other: Supplementary or flag column(s) to associate.

        Raises:
            ColumnTypeError: If any other column is not supplementary or flag.
        """
        if not isinstance(other, list):
            other = [other]

        other = [self._ts.columns[col] if isinstance(col, str) else col for col in other]

        for col in other:
            if type(col) is SupplementaryColumn:
                relationship = Relationship(self, col, RelationshipType.MANY_TO_MANY, DeletionPolicy.UNLINK)
            elif type(col) is FlagColumn:
                relationship = Relationship(self, col, RelationshipType.ONE_TO_MANY, DeletionPolicy.CASCADE)
            else:
                raise ColumnTypeError(f"Related column must be supplementary or flag: {col.name}:{type(col)}")

            self._ts._relationship_manager._add(relationship)

    def get_flag_system_column(self, flag_system: str) -> "TimeSeriesColumn | None":
        """Retrieves the flag column linked to this data column that corresponds to the specified flag system.

        Args:
            flag_system: The name of the flag system.

        Raises:
            ColumnNotFoundError: If more than one flag column is found matching the given flag system.

        Returns:
            The matching flag column if exactly one match is found, or None if no matching column is found.
        """
        relationships = self.get_relationships()
        flag_system = self._ts._flag_manager.flag_systems.get(flag_system, None)
        matches = []
        for relationship in relationships:
            if type(relationship.other_column) is FlagColumn and relationship.other_column.flag_system == flag_system:
                matches.append(relationship.other_column)

        if len(matches) > 1:
            # Should only be one (or no) matches.  Something gone wrong further upstream if this has happened!
            raise ColumnNotFoundError(f"More than one column matches found for flag system {flag_system}: {matches}")
        elif len(matches) == 1:
            return matches[0]
        else:
            return None


class SupplementaryColumn(TimeSeriesColumn):
    """Represents supplementary columns (e.g., metadata, extra information)."""

    def add_relationship(self, other: "str | TimeSeriesColumn | list[TimeSeriesColumn | str]") -> None:
        """Adds a relationship between this supplementary column and data column(s).

        Args:
            other: Data column(s) to associate.

        Raises:
            ColumnTypeError: If any other column is not data.
        """
        if not isinstance(other, list):
            other = [other]

        other = [self._ts.columns[col] if isinstance(col, str) else col for col in other]

        for col in other:
            if type(col) is DataColumn:
                relationship = Relationship(col, self, RelationshipType.MANY_TO_MANY, DeletionPolicy.UNLINK)
            else:
                raise ColumnTypeError(f"Related column must be data: {col.name}:{type(col)}")

            self._ts._relationship_manager._add(relationship)


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

    def add_relationship(self, other: "str | TimeSeriesColumn | list[TimeSeriesColumn | str]") -> None:
        """Adds a relationship between this flag column and data column(s).

        Args:
            other: Data column(s) to associate.

        Raises:
            ColumnTypeError: If any other column is not data.
        """
        if not isinstance(other, list):
            other = [other]

        other = [self._ts.columns[col] if isinstance(col, str) else col for col in other]

        for col in other:
            if type(col) is DataColumn:
                relationship = Relationship(col, self, RelationshipType.ONE_TO_MANY, DeletionPolicy.CASCADE)
            else:
                raise ColumnTypeError(f"Related column must be data: {col.name}:{type(col)}")

            self._ts._relationship_manager._add(relationship)

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
