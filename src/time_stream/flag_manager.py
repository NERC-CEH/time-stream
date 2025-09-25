"""
Flag System Management Module.

This module provides a small registry for use by the TimeSeries class for bitwise flag systems and for flag columns
bound to data columns in the TimeSeries' DataFrame.

Typical use from within the TimeSeries class:
    1) Register a flag system by name (from a dict or an existing BitwiseFlag subclass).
    2) Register a DataFrame column as a flag column associated with that system.
    3) Use FlagColumn.add_flag / remove_flag with Polars expressions to set/clear the flag bits.
"""

from dataclasses import dataclass
from typing import Self

import polars as pl

from time_stream.bitwise import BitwiseFlag, BitwiseMeta
from time_stream.exceptions import (
    ColumnNotFoundError,
    DuplicateFlagSystemError,
    FlagSystemError,
    FlagSystemNotFoundError,
    FlagSystemTypeError,
)

FlagSystemType = dict[str, int] | type[BitwiseFlag]


@dataclass
class FlagColumn:
    """Represents a flag column in a TimeSeries.

    A flag column stores bitwise flags governed by a specific flag system. Each flag column is associated with a base
    data column.

    Attributes:
        name: Name of the flag column in the DataFrame.
        base: Name of the associated value/data column.
        flag_system: The enum class that defines the available flags and their bit values.
    """

    name: str
    base: str
    flag_system: type[BitwiseFlag]

    def add_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Adds a flag value to this FlagColumn using a bitwise OR operation.

        Args:
            df: The dataframe to add the flag value to.
            flag: The flag value to add.
            expr: A Polars expression defining the condition for applying the flag.
                  Defaults to `pl.lit(True)`, meaning all rows are flagged.

        Returns:
            A new DataFrame with the flag column updated.
        """
        flag = self.flag_system.get_single_flag(flag)
        return df.with_columns(
            pl.when(expr).then(pl.col(self.name) | pl.lit(flag.value)).otherwise(pl.col(self.name)).alias(self.name)
        )

    def remove_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Remove a flag value from this FlagColumn using a bitwise AND operation.

        Args:
            df: The dataframe to remove the flag value from.
            flag: The flag value to remove.
            expr: A Polars expression defining the condition for removing the flag.
                  Defaults to `pl.lit(True)`, meaning flag remove from all rows.

        Returns:
            A new DataFrame with the flag column updated.
        """
        flag = self.flag_system.get_single_flag(flag)
        return df.with_columns(
            pl.when(expr).then(pl.col(self.name) & ~pl.lit(flag.value)).otherwise(pl.col(self.name)).alias(self.name)
        )

    def __eq__(self, other: object) -> bool:
        """Check if two FlagColumn instances are equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the FlagColumn instances are equal, False otherwise.
        """
        if not isinstance(other, FlagColumn):
            return False

        return self.name == other.name and self.base == other.base and self.flag_system == other.flag_system

    # Make class instances unhashable
    __hash__ = None


class FlagManager:
    """Registry for flag systems and flag columns.

    This class:
      * registers **flag systems** (bit registries) under a string name;
      * registers **flag columns** that reference a base data column and a specific flag system.
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

    def register_flag_system(self, flag_system_name: str, flag_system: FlagSystemType) -> None:
        """Registers a flag system with the flag manager.  A flag system will contain flag values and their meanings.

        A flag system can be used to create flag columns that are specific to a particular type of flag.
        The flag system must contain valid bitwise values and their name. Using bitwise values
        allows for multiple flags to be set on a single value.

        For example, if we had a quality_control flag system, we could define it as follows:

        > flag_system_name = "quality_control"
        > flag_dict = {
        >     "OUT_OF_RANGE": 1,
        >     "SPIKE": 2,
        >     "LOW_BATTERY": 4,
        > }

        The flag_dict itself can be passed to this method, or a BitwiseFlag object can be created from the dict:
        > flag_system = BitwiseFlag(flag_dict, name=flag_system_name)

        Args:
            flag_system_name: The name of the new flag system.
            flag_system: The flag system containing flag values and their meanings.

        Raises:
            FlagSystemTypeError: If the flag system is not a valid type.
        """
        if flag_system_name in self._flag_systems:
            raise DuplicateFlagSystemError(f"Flag system '{flag_system_name}' already exists.")

        if isinstance(flag_system, BitwiseMeta):
            self._flag_systems[flag_system_name] = flag_system

        elif isinstance(flag_system, dict):
            self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, flag_system)

        else:
            raise FlagSystemTypeError(
                f"Unknown type of flag system: {type(flag_system)}. "
                f"Expected dict[str, int] or a ``BitwiseFlag`` subclass."
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

    def register_flag_column(self, name: str, base: str, flag_system_name: str) -> None:
        """Mark the specified existing column as a flag column.

        Args:
            name: A column name to mark as a flag column.
            base: Name of the value/data column this flag column refers to.
            flag_system_name: The name of the flag system.
        """
        flag_column = self._flag_columns.get(name)
        if flag_column:
            raise FlagSystemError(
                f"Flag column '{name}' already registered. Base: '{flag_column.base}'; System: '{flag_system_name}'."
            )
        else:
            flag_system = self.get_flag_system(flag_system_name)
            flag_column = FlagColumn(name, base, flag_system)
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
            out.register_flag_column(name, flag_column.base, flag_column.flag_system.system_name())

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
