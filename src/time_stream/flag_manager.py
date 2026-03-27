"""
Flag System Management Module.

This module provides a small registry for use by the TimeFrame class for bitwise flag systems and for flag columns
bound to data columns in the TimeFrame's DataFrame.

Typical use from within the TimeFrame class:
    1) Register a flag system by name (from a dict or an existing BitwiseFlag subclass).
    2) Register a DataFrame column as a flag column linked to that system.
    3) Use FlagColumn.add_flag / remove_flag with Polars expressions to set/clear the flag bits.
"""

from dataclasses import dataclass
from typing import Self

import polars as pl

from time_stream.bitwise import BitwiseFlag, BitwiseMeta
from time_stream.exceptions import (
    BitwiseFlagUnknownError,
    ColumnNotFoundError,
    DuplicateFlagSystemError,
    FlagSystemError,
    FlagSystemNotFoundError,
    FlagSystemTypeError,
)

FlagSystemType = dict[str, int] | type[BitwiseFlag] | list[str] | None


@dataclass
class FlagColumn:
    """Represents a flag column in a TimeFrame.

    A flag column stores bitwise flags governed by a specific flag system.

    Attributes:
        name: Name of the flag column in the DataFrame.
        flag_system: The enum class that defines the available flags and their bit values.
        is_decoded: Whether the column is currently in decoded List(String) form rather than integer form.
    """

    name: str
    flag_system: type[BitwiseFlag]
    is_decoded: bool = False

    def decode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace the integer flag column with a List(String) column of active flag names.

        Each row contains the names of flags that are set, sorted by ascending bit value. A value of 0 produces an
        empty list.

        Args:
            df: The DataFrame containing the integer flag column.

        Returns:
            A new DataFrame with the flag column replaced by a List(String) column.
        """

        # Sort the flag system mapping into ascending bit value order
        flag_map = sorted(self.flag_system.to_dict().items(), key=lambda kv: kv[1])

        # Build expressions for decoding each flag value
        exprs = [
            pl.when((pl.col(self.name) & pl.lit(val)) != 0).then(pl.lit(name)).otherwise(pl.lit(None))
            for name, val in flag_map
        ]
        return df.with_columns(pl.concat_list(exprs).list.drop_nulls().alias(self.name))

    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace a List(String) flag column with a bitwise integer column.

        Each flag name in the list contributes its bit value. An empty list produces 0.

        Args:
            df: The DataFrame containing the decoded List(String) flag column.

        Returns:
            A new DataFrame with the flag column replaced by an integer column.

        Raises:
            BitwiseFlagUnknownError: If any flag name in the column is not in the flag system.
        """
        flag_map = self.flag_system.to_dict()
        present = set(df[self.name].explode().drop_nulls().unique().to_list())
        unknown = present - flag_map.keys()
        if unknown:
            raise BitwiseFlagUnknownError(f"Unknown flag names in column '{self.name}': {sorted(unknown)}.")

        exprs = [
            pl.when(pl.col(self.name).list.contains(pl.lit(name)))
            .then(pl.lit(val, dtype=pl.Int64))
            .otherwise(pl.lit(0, dtype=pl.Int64))
            for name, val in flag_map.items()
        ]
        return df.with_columns(pl.sum_horizontal(exprs).alias(self.name))

    def add_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Adds a flag value to this FlagColumn using a bitwise OR operation.

        If the column is currently decoded (List(String)), it is encoded first, the flag is applied,
        and the result is decoded again.

        Args:
            df: The dataframe to add the flag value to.
            flag: The flag value to add.
            expr: A Polars expression defining the condition for applying the flag.
                  Defaults to `pl.lit(True)`, meaning all rows are flagged.

        Returns:
            A new DataFrame with the flag column updated.
        """
        flag = self.flag_system.get_single_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        df = df.with_columns(
            pl.when(expr).then(pl.col(self.name) | pl.lit(flag.value)).otherwise(pl.col(self.name)).alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def remove_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Remove a flag value from this FlagColumn using a bitwise AND operation.

        If the column is currently decoded (List(String)), it is encoded first, the flag is removed,
        and the result is decoded again.

        Args:
            df: The dataframe to remove the flag value from.
            flag: The flag value to remove.
            expr: A Polars expression defining the condition for removing the flag.
                  Defaults to `pl.lit(True)`, meaning flag remove from all rows.

        Returns:
            A new DataFrame with the flag column updated.
        """
        flag = self.flag_system.get_single_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        df = df.with_columns(
            pl.when(expr).then(pl.col(self.name) & ~pl.lit(flag.value)).otherwise(pl.col(self.name)).alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def __eq__(self, other: object) -> bool:
        """Check if two FlagColumn instances are equal.

        Compares name and flag_system only. is_decoded is runtime state and is excluded
        from the comparison.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the FlagColumn instances are equal, False otherwise.
        """
        if not isinstance(other, FlagColumn):
            return False

        return self.name == other.name and self.flag_system == other.flag_system

    # Make class instances unhashable
    __hash__ = None


class FlagManager:
    """Registry for flag systems and flag columns.

    This class:
      * registers **flag systems** (bit registries) under a string name;
      * registers **flag columns** linked to a specific flag system.
    """

    def __init__(self) -> None:
        self._flag_systems: dict[str, type[BitwiseFlag]] = {}
        self._flag_columns: dict[str, FlagColumn] = {}

    @property
    def flag_systems(self) -> dict[str, type[BitwiseFlag]]:
        """Registered flag systems."""
        return self._flag_systems

    @property
    def flag_columns(self) -> dict[str, FlagColumn]:
        """Registered flag columns."""
        return self._flag_columns

    def register_flag_system(self, flag_system_name: str, flag_system: FlagSystemType = None) -> None:
        """Registers a flag system with the flag manager.  A flag system will contain flag values and their meanings.

        A flag system can be used to create flag columns that are specific to a particular type of flag.
        The flag system must contain valid bitwise values and their name. Using bitwise values
        allows for multiple flags to be set on a single value.

        For example, if we had a quality_control flag system, we could define it as follows using a dict:

        > flag_system_name = "quality_control"
        > flag_dict = {
        >     "OUT_OF_RANGE": 1,
        >     "SPIKE": 2,
        >     "LOW_BATTERY": 4,
        > }

        The flag_dict itself can be passed to this method, or a BitwiseFlag object can be created from the dict:

        > flag_system = BitwiseFlag(flag_dict, name=flag_system_name)

        Alternatively, a list of category names can be passed and bit values will be assigned automatically.
        The list is sorted before assigning values, so the same set of names always produces the same mapping:

        > flag_system = ["OUT_OF_RANGE", "SPIKE", "LOW_BATTERY"]

        would produce a dict equivalent to:

        > flag_dict = {
        >     "LOW_BATTERY": 1,
        >     "OUT_OF_RANGE": 2,
        >     "SPIKE": 4,
        > }

        An empty list or ``None`` produces a default flag system with a single member at value 1:

        > flag_dict = {
        >     "FLAGGED": 1
        > }

        Args:
            flag_system_name: The name of the new flag system.
            flag_system: The flag system containing flag values and their meanings. Defaults to ``None``.

        Raises:
            DuplicateFlagSystemError: If a flag system with the same name is already registered.
            FlagSystemTypeError: If the flag system is not a valid type, or the list contains duplicate names.
        """
        if flag_system_name in self._flag_systems:
            raise DuplicateFlagSystemError(f"Flag system '{flag_system_name}' already exists.")

        default_flag_dict = {"FLAGGED": 1}

        if flag_system is None:
            self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, default_flag_dict)

        elif isinstance(flag_system, BitwiseMeta):
            self._flag_systems[flag_system_name] = flag_system

        elif isinstance(flag_system, list):
            if len(set(flag_system)) != len(flag_system):
                raise FlagSystemTypeError("Flag system list contains duplicate category names.")
            if len(flag_system) == 0:
                flag_dict = default_flag_dict
            else:
                sorted_categories = sorted(flag_system)
                flag_dict = {name: 2**i for i, name in enumerate(sorted_categories)}
            self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, flag_dict)

        elif isinstance(flag_system, dict):
            self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, flag_system)

        else:
            raise FlagSystemTypeError(
                f"Unknown type of flag system: {type(flag_system)}. "
                f"Expected dict[str, int], list[str], or a ``BitwiseFlag`` subclass."
            )

    def get_flag_system(self, flag_system_name: str) -> type[BitwiseFlag]:
        """Return a registered flag system.

        Args:
            flag_system_name: The registered flag system name

        Returns:
            The ``BitwiseFlag`` enum that defines the flag system.
        """
        try:
            return self._flag_systems[flag_system_name]
        except KeyError:
            raise FlagSystemNotFoundError(f"No such flag system: '{flag_system_name}'")

    def register_flag_column(self, name: str, flag_system_name: str) -> None:
        """Mark the specified existing column as a flag column.

        Args:
            name: A column name to mark as a flag column.
            flag_system_name: The name of the flag system.
        """
        flag_column = self._flag_columns.get(name)
        if flag_column:
            raise FlagSystemError(f"Flag column '{name}' already registered. System: '{flag_system_name}'.")
        else:
            flag_system = self.get_flag_system(flag_system_name)
            flag_column = FlagColumn(name, flag_system)
            self._flag_columns[name] = flag_column

    def get_flag_column(self, name: str) -> FlagColumn:
        """Look up a registered flag column by name.

        Args:
            name: Flag column name.

        Returns:
            The corresponding ``FlagColumn`` object.
        """
        try:
            return self._flag_columns[name]
        except KeyError:
            raise ColumnNotFoundError(f"No such flag column: '{name}'.")

    def copy(self) -> Self:
        """Create a copy of this flag manager object."""
        out = FlagManager()

        # register flag systems in the new copy under the same names as previous
        for name, flag_system in self._flag_systems.items():
            out.register_flag_system(name, flag_system.to_dict())

        # register flag columns in the new copy with their associated system names
        for name, flag_column in self._flag_columns.items():
            out.register_flag_column(name, flag_column.flag_system.system_name())
            out.flag_columns[name].is_decoded = flag_column.is_decoded

        return out

    def __copy__(self) -> Self:
        return self.copy()

    def __eq__(self, other: object) -> bool:
        """Check if two FlagManager instances are equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the FlagManager instances are equal, False otherwise.
        """
        if not isinstance(other, FlagManager):
            return False

        return self._flag_systems == other._flag_systems and self._flag_columns == other._flag_columns

    # Make class instances unhashable
    __hash__ = None
