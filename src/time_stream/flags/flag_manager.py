"""
Flag System Management Module.

This module provides a small registry for use by the ``TimeFrame`` class for flag systems and for flag columns
in the TimeFrame's DataFrame.

Two flag system types are supported:
  - ``BitwiseFlag`` - power-of-two integer values; multiple flags can be combined per row.
  - ``CategoricalFlag`` - arbitrary int or str values; each row holds one value (or a list of values).

Typical use from within the ``TimeFrame`` class:
    1) Register a flag system by name
    2) Register a DataFrame column as a flag column linked to that system.
    3) Use ``add_flag`` / ``remove_flag`` on the flag column, with Polars expressions to modify the flag values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import polars as pl

from time_stream.exceptions import (
    BitwiseFlagUnknownError,
    CategoricalFlagUnknownError,
    ColumnNotFoundError,
    DuplicateFlagSystemError,
    FlagSystemError,
    FlagSystemNotFoundError,
    FlagSystemTypeError,
)
from time_stream.flags.bitwise_flag_system import BitwiseFlag
from time_stream.flags.categorical_flag_system import CategoricalFlag
from time_stream.flags.flag_system import FlagSystemBase

FlagSystemType = dict[str, int | str] | list[str] | None
FlagSystemLiteral = Literal["bitwise", "categorical"]


class FlagColumn(ABC):
    """Abstract base class for flag columns in a TimeFrame.

    A flag column binds a DataFrame column to a flag system and provides ``add_flag`` and
    ``remove_flag`` operations. Concrete subclasses implement the appropriate semantics:

    - ``BitwiseFlagColumn`` - bitwise OR/AND operations on integer columns.
    - ``CategoricalFlagColumn`` - set value or list append/filter operations.

    Attributes:
        name: Name of the flag column in the DataFrame.
        flag_system: The flag system enum class governing this column.
        is_decoded: Whether the column is currently in decoded (human-readable) form.
    """

    name: str
    flag_system: type[FlagSystemBase]
    is_decoded: bool

    @abstractmethod
    def decode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace raw flag values with their human-readable names.

        Args:
            df: The DataFrame containing the flag column.

        Returns:
            A new DataFrame with the flag column replaced by its decoded form.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace human-readable flag names back to their raw values.

        Args:
            df: The DataFrame containing the decoded flag column.

        Returns:
            A new DataFrame with the flag column replaced by its encoded form.
        """
        raise NotImplementedError

    @abstractmethod
    def add_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Add a flag value to rows where ``expr`` is true.

        Args:
            df: The DataFrame containing the flag column.
            flag: The flag name or value to add.
            expr: A Polars expression defining which rows to update.

        Returns:
            A new DataFrame with the flag column updated.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Remove a flag value from rows where ``expr`` is true.

        Args:
            df: The DataFrame containing the flag column.
            flag: The flag name or value to remove.
            expr: A Polars expression defining which rows to update.

        Returns:
            A new DataFrame with the flag column updated.
        """
        raise NotImplementedError


@dataclass
class BitwiseFlagColumn(FlagColumn):
    """Represents a bitwise flag column in a TimeFrame.

    A bitwise flag column stores bitwise flags governed by a specific ``BitwiseFlag`` system.

    Attributes:
        name: Name of the flag column in the DataFrame.
        flag_system: The ``BitwiseFlag`` enum class that defines the available flags and their bit values.
        is_decoded: Whether the column is currently in decoded ``List(String)`` form rather than integer form.
    """

    name: str
    flag_system: type[BitwiseFlag]
    is_decoded: bool = False

    def decode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace the integer flag column with a ``List(String)`` column of active flag names.

        Each row contains the names of flags that are set, sorted by ascending 'bit' value. A value of 0 produces an
        empty list.

        Args:
            df: The DataFrame containing the integer flag column.

        Returns:
            A new DataFrame with the flag column replaced by a ``List(String)`` column.
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
        """Replace a ``List(String)`` flag column with a bitwise integer column.

        Each flag name in the list contributes its bit value. An empty list produces 0.

        Args:
            df: The DataFrame containing the decoded ``List(String)`` flag column.

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
        """Add a flag value to this ``BitwiseFlagColumn`` using a bitwise OR operation.

        If the column is currently decoded, it is encoded first, the flag is applied, and the result is decoded again.

        Args:
            df: The DataFrame to add the flag value to.
            flag: The flag name or value to add.
            expr: A Polars expression defining the condition for applying the flag.
                  Defaults to ``pl.lit(True)``, meaning all rows are flagged.

        Returns:
            A new DataFrame with the flag column updated.
        """
        flag_value = self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        df = df.with_columns(
            pl.when(expr).then(pl.col(self.name) | pl.lit(flag_value)).otherwise(pl.col(self.name)).alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def remove_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Remove a flag value from this ``BitwiseFlagColumn`` using a bitwise AND NOT operation.

        If the column is currently decoded, it is encoded first, the flag is removed, and the result is decoded again.

        Args:
            df: The DataFrame to remove the flag value from.
            flag: The flag name or value to remove.
            expr: A Polars expression defining which rows to update.
                  Defaults to ``pl.lit(True)``, meaning the flag is removed from all rows.

        Returns:
            A new DataFrame with the flag column updated.
        """
        flag_value = self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        df = df.with_columns(
            pl.when(expr).then(pl.col(self.name) & ~pl.lit(flag_value)).otherwise(pl.col(self.name)).alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def __eq__(self, other: object) -> bool:
        """Check if two ``BitwiseFlagColumn`` instances are equal.

        Compares ``name`` and ``flag_system`` only. ``is_decoded`` is runtime state and is excluded
        from the comparison.

        Args:
            other: The object to compare.

        Returns:
            True if both instances have the same name and flag system, False otherwise.
        """
        if not isinstance(other, BitwiseFlagColumn):
            return False

        return self.name == other.name and self.flag_system == other.flag_system

    # Make class instances unhashable
    __hash__ = None


@dataclass
class CategoricalFlagColumn(FlagColumn):
    """Represents a categorical flag column in a TimeFrame.

    A categorical flag column stores arbitrary int or str flag values governed by a ``CategoricalFlag``
    system. Each row holds a single value (scalar mode) or a list of values (list mode).

    Attributes:
        name: Name of the flag column in the DataFrame.
        flag_system: The ``CategoricalFlag`` enum class that defines the available flag values.
        list_mode: If ``True``, the column stores ``List`` values and flags are appended/removed from
            the list. If ``False``, the column stores a single nullable value per row.
        is_decoded: Whether the column is currently in decoded (flag-name) form rather than raw-value form.
    """

    name: str
    flag_system: type[CategoricalFlag]
    list_mode: bool = False
    is_decoded: bool = False

    def decode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace raw flag values with their flag names.

        In scalar mode each value is replaced by its name (e.g. ``123 -> "good"``).
        In list mode each element of each list is replaced by its name.

        Args:
            df: The DataFrame containing the raw-value flag column.

        Returns:
            A new DataFrame with the flag column replaced by a ``Utf8`` (scalar) or ``List(Utf8)`` (list mode)
            column of flag names.
        """
        flag_map = self.flag_system.to_dict()
        old = list(flag_map.values())
        new = list(flag_map.keys())

        if self.list_mode:
            return df.with_columns(
                pl.col(self.name)
                .list.eval(pl.element().replace_strict(old=old, new=new, default=None, return_dtype=pl.Utf8))
                .alias(self.name)
            )
        else:
            return df.with_columns(
                pl.col(self.name).replace_strict(old=old, new=new, default=None, return_dtype=pl.Utf8).alias(self.name)
            )

    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace flag names back to their raw values.

        In scalar mode each name is replaced by its value (e.g. ``"good" -> 123``).
        In list mode each element of each list is replaced by its value.

        Args:
            df: The DataFrame containing the decoded (flag-name) flag column.

        Returns:
            A new DataFrame with the flag column replaced by its original value dtype.

        Raises:
            CategoricalFlagUnknownError: If any flag name in the column is not in the flag system.
        """
        flag_map = self.flag_system.to_dict()
        vtype = self.flag_system.value_type()
        return_dtype = pl.Int32 if vtype is int else pl.Utf8

        if self.list_mode:
            present = set(df[self.name].explode().drop_nulls().unique().to_list())
        else:
            present = set(df[self.name].drop_nulls().unique().to_list())

        unknown = present - flag_map.keys()
        if unknown:
            raise CategoricalFlagUnknownError(f"Unknown flag names in column '{self.name}': {sorted(unknown)}.")

        old = list(flag_map.keys())
        new = list(flag_map.values())

        if self.list_mode:
            return df.with_columns(
                pl.col(self.name)
                .list.eval(pl.element().replace_strict(old=old, new=new, default=None, return_dtype=return_dtype))
                .alias(self.name)
            )
        else:
            return df.with_columns(
                pl.col(self.name)
                .replace_strict(old=old, new=new, default=None, return_dtype=return_dtype)
                .alias(self.name)
            )

    def add_flag(
        self,
        df: pl.DataFrame,
        flag: int | str,
        expr: pl.Expr = pl.lit(True),
        overwrite: bool = True,
    ) -> pl.DataFrame:
        """Set or append a flag value on rows where ``expr`` is true.

        In scalar mode, sets the column value to ``flag`` where ``expr`` is true. When ``overwrite`` is
        ``False``, only rows whose current value is null are updated.

        In list mode, appends ``flag`` to the list where ``expr`` is true and the flag is not already
        present. ``overwrite`` is not applicable in list mode.

        If the column is currently decoded, it is encoded first, the flag is applied, and the result is
        decoded again.

        Args:
            df: The DataFrame containing the flag column.
            flag: The flag name or value to set/append.
            expr: A Polars expression defining which rows to update.
                  Defaults to ``pl.lit(True)``, meaning all rows.
            overwrite: Scalar mode only. If ``True`` (default), replaces any existing value. If ``False``,
                only updates rows whose current value is null.

        Returns:
            A new DataFrame with the flag column updated.
        """
        value = self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        col_dtype = df[self.name].dtype

        if self.list_mode:
            df = df.with_columns(
                pl.when(expr & ~pl.col(self.name).list.contains(pl.lit(value)))
                .then(pl.concat_list([pl.col(self.name), pl.lit(value).implode()]))
                .otherwise(pl.col(self.name))
                .alias(self.name)
            )
        else:
            condition = expr if overwrite else (expr & pl.col(self.name).is_null())
            df = df.with_columns(
                pl.when(condition).then(pl.lit(value, dtype=col_dtype)).otherwise(pl.col(self.name)).alias(self.name)
            )

        if self.is_decoded:
            df = self.decode(df)
        return df

    def remove_flag(
        self,
        df: pl.DataFrame,
        flag: int | str,
        expr: pl.Expr = pl.lit(True),
    ) -> pl.DataFrame:
        """Remove a flag value from rows where ``expr`` is true.

        In scalar mode, sets the column value to null where ``expr`` is true. In list mode, removes all
        occurrences of ``flag`` from the list where ``expr`` is true.

        If the column is currently decoded, it is encoded first, the flag is removed, and the result is
        decoded again.

        Args:
            df: The DataFrame containing the flag column.
            flag: The flag name or value to remove.
            expr: A Polars expression defining which rows to update.
                  Defaults to ``pl.lit(True)``, meaning all rows.

        Returns:
            A new DataFrame with the flag column updated.
        """
        value = self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        col_dtype = df[self.name].dtype

        if self.list_mode:
            df = df.with_columns(
                pl.when(expr)
                .then(pl.col(self.name).list.eval(pl.element().filter(pl.element() != pl.lit(value))))
                .otherwise(pl.col(self.name))
                .alias(self.name)
            )
        else:
            df = df.with_columns(
                pl.when(expr).then(pl.lit(None, dtype=col_dtype)).otherwise(pl.col(self.name)).alias(self.name)
            )

        if self.is_decoded:
            df = self.decode(df)
        return df

    def __eq__(self, other: object) -> bool:
        """Check if two ``CategoricalFlagColumn`` instances are equal.

        Compares ``name``, ``flag_system``, and ``list_mode``. ``is_decoded`` is runtime state and is
        excluded from the comparison.

        Args:
            other: The object to compare.

        Returns:
            True if both instances have the same name, flag system, and list mode, False otherwise.
        """
        if not isinstance(other, CategoricalFlagColumn):
            return False
        return self.name == other.name and self.flag_system == other.flag_system and self.list_mode == other.list_mode

    # Make class instances unhashable
    __hash__ = None


class FlagManager:
    """Registry for flag systems and flag columns.

    This class:
      * registers **flag systems** (bitwise or categorical) under a string name;
      * registers **flag columns** linked to a specific flag system.
    """

    def __init__(self) -> None:
        self._flag_systems: dict[str, type[FlagSystemBase]] = {}
        self._flag_columns: dict[str, FlagColumn] = {}

    @property
    def flag_systems(self) -> dict[str, type[FlagSystemBase]]:
        """Registered flag systems."""
        return self._flag_systems

    @property
    def flag_columns(self) -> dict[str, FlagColumn]:
        """Registered flag columns."""
        return self._flag_columns

    def register_flag_system(
        self,
        flag_system_name: str,
        flag_system: FlagSystemType = None,
        flag_type: FlagSystemLiteral = "bitwise",
    ) -> None:
        """Register a flag system with the flag manager.

        Two kinds of flag system are supported:

        **Bitwise** (``BitwiseFlag``): values are powers of two, enabling multiple flags to be combined
        on a single integer per row. Pass ``flag_type="bitwise"`` (the default).

        **Categorical** (``CategoricalFlag``): values are arbitrary ``int`` or ``str``; each row holds
        one value (or a list of values in list mode). Pass ``flag_type="categorical"``.

        Accepted inputs for ``flag_system``:

        - ``None`` - produces a default bitwise system with a single ``FLAGGED`` flag at value 1.
        - ``dict[str, int]`` - interpreted as bitwise or categorical depending on ``flag_type``.
          Bitwise values must be powers of two.
        - ``dict[str, str]`` - always categorical; ``flag_type`` is ignored.
        - ``list[str]`` - flag names are sorted; bitwise assigns powers of two, categorical assigns
          sequential integers starting from 0. An empty list produces the default ``FLAGGED`` flag.

        Args:
            flag_system_name: The name of the new flag system.
            flag_system: The flag system definition. Defaults to ``None``.
            flag_type: Whether to create a ``"bitwise"`` or ``"categorical"`` flag system. Only
                relevant when ``flag_system`` is a ``dict[str, int]`` or ``list[str]``.

        Raises:
            DuplicateFlagSystemError: If a flag system with the same name is already registered.
            FlagSystemTypeError: If the flag system is not a recognised type, or a list contains duplicate names.
        """
        if flag_system_name in self._flag_systems:
            raise DuplicateFlagSystemError(f"Flag system '{flag_system_name}' already exists.")

        default_flag_dict = {"FLAGGED": 1}

        if not flag_system:
            self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, default_flag_dict)
            return

        if isinstance(flag_system, list):
            if len(set(flag_system)) != len(flag_system):
                raise FlagSystemTypeError("Flag system list contains duplicate category names.")

            if flag_type == "categorical":
                sorted_categories = sorted(flag_system)
                flag_dict = {name: i for i, name in enumerate(sorted_categories)}
                self._flag_systems[flag_system_name] = CategoricalFlag(flag_system_name, flag_dict)
            else:
                sorted_categories = sorted(flag_system)
                flag_dict = {name: 2**i for i, name in enumerate(sorted_categories)}
                self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, flag_dict)

        elif isinstance(flag_system, dict):
            # Check if all values in dict are string, in which case we know this should be a categorical flag system
            if all(isinstance(v, str) for v in flag_system.values()):
                flag_type = "categorical"

            if flag_type == "categorical":
                self._flag_systems[flag_system_name] = CategoricalFlag(flag_system_name, flag_system)
            else:
                self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, flag_system)

        else:
            raise FlagSystemTypeError(
                f"Unknown type of flag system: {type(flag_system)}. "
                f"Expected dict[str, int], dict[str, str], or list[str]."
            )

    def get_flag_system(self, flag_system_name: str) -> type[FlagSystemBase]:
        """Return a registered flag system.

        Args:
            flag_system_name: The registered flag system name.

        Returns:
            The ``BitwiseFlag`` or ``CategoricalFlag`` enum class that defines the flag system.

        Raises:
            FlagSystemNotFoundError: If no flag system with ``flag_system_name`` is registered.
        """
        try:
            return self._flag_systems[flag_system_name]
        except KeyError:
            raise FlagSystemNotFoundError(f"No such flag system: '{flag_system_name}'")

    def register_flag_column(self, name: str, flag_system_name: str, list_mode: bool | None = None) -> None:
        """Mark the specified existing column as a flag column.

        For categorical flag systems, ``list_mode`` controls whether the column stores a list of values
        per row (``True``) or a single nullable value (``False``). Inferred from the column dtype by the
        caller; ignored for bitwise flag systems.

        Args:
            name: A column name to mark as a flag column.
            flag_system_name: The name of the flag system.
            list_mode: For categorical systems, whether the column is in list mode. ``None`` for bitwise.

        Raises:
            FlagSystemError: If a flag column with ``name`` is already registered.
            FlagSystemNotFoundError: If ``flag_system_name`` is not a registered flag system.
        """
        if name in self._flag_columns:
            raise FlagSystemError(f"Flag column '{name}' already registered. System: '{flag_system_name}'.")

        flag_system = self.get_flag_system(flag_system_name)
        if issubclass(flag_system, CategoricalFlag):
            self._flag_columns[name] = CategoricalFlagColumn(name, flag_system, bool(list_mode))
        elif issubclass(flag_system, BitwiseFlag):
            self._flag_columns[name] = BitwiseFlagColumn(name, flag_system)

    def get_flag_column(self, name: str) -> FlagColumn:
        """Look up a registered flag column by name.

        Args:
            name: Flag column name.

        Returns:
            The corresponding ``BitwiseFlagColumn`` or ``CategoricalFlagColumn`` instance.

        Raises:
            ColumnNotFoundError: If no flag column with ``name`` is registered.
        """
        try:
            return self._flag_columns[name]
        except KeyError:
            raise ColumnNotFoundError(f"No such flag column: '{name}'.")

    def copy(self) -> "FlagManager":
        """Create a deep copy of this ``FlagManager``, duplicating all registered systems and columns.

        Returns:
            A new ``FlagManager`` with the same flag systems, flag columns, and ``is_decoded`` state.
        """
        out = FlagManager()

        # register flag systems in the new copy under the same names as previous
        for name, flag_system in self._flag_systems.items():
            flag_type: FlagSystemLiteral = "categorical" if issubclass(flag_system, CategoricalFlag) else "bitwise"
            out.register_flag_system(name, flag_system.to_dict(), flag_type=flag_type)

        #  register flag columns in the new copy with their associated system names
        for name, flag_column in self._flag_columns.items():
            if isinstance(flag_column, CategoricalFlagColumn):
                out.register_flag_column(name, flag_column.flag_system.system_name(), flag_column.list_mode)
            else:
                out.register_flag_column(name, flag_column.flag_system.system_name())
            out.flag_columns[name].is_decoded = flag_column.is_decoded

        return out

    def __copy__(self) -> "FlagManager":
        return self.copy()

    def __eq__(self, other: object) -> bool:
        """Check if two ``FlagManager`` instances are equal.

        Args:
            other: The object to compare.

        Returns:
            True if both instances have the same flag systems and flag columns, False otherwise.
        """
        if not isinstance(other, FlagManager):
            return False

        return self._flag_systems == other._flag_systems and self._flag_columns == other._flag_columns

    # Make class instances unhashable
    __hash__ = None
