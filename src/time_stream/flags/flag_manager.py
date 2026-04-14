"""
Flag System Management Module.

This module provides a small registry for use by the ``TimeFrame`` class for flag systems and for flag columns
in the TimeFrame's DataFrame.

Three flag system types are supported:
  - ``BitwiseFlag`` - power-of-two integer values; multiple flags can be combined per row.
  - ``CategoricalSingleFlag`` - arbitrary int or str values; each row holds exactly one value (mutually exclusive).
  - ``CategoricalListFlag`` - arbitrary int or str values; each row holds a list of values (flags can coexist).

Typical use from within the ``TimeFrame`` class:
    1) Register a flag system by name
    2) Register a DataFrame column as a flag column linked to that system.
    3) Use ``add_flag`` / ``remove_flag`` on the flag column, with Polars expressions to modify the flag values.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
from time_stream.flags.categorical_flag_system import CategoricalListFlag, CategoricalSingleFlag
from time_stream.flags.flag_system import FlagSystemBase, FlagSystemLiteral

FlagSystemType = dict[str, int | str] | list[str] | None


class FlagColumn(ABC):
    """Abstract base class for flag columns in a TimeFrame.

    A flag column binds a DataFrame column to a flag system and provides ``add_flag`` and
    ``remove_flag`` operations. Concrete subclasses implement the appropriate semantics:

    - ``BitwiseFlagColumn`` - bitwise OR/AND operations on integer columns.
    - ``CategoricalSingleFlagColumn`` - set/clear a single value per row.
    - ``CategoricalListFlagColumn`` - append/remove values from a list per row.

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

    @abstractmethod
    def filter_expr(self, flags: list[int | str]) -> pl.Expr:
        """Return a boolean Polars expression that is True for rows where any of the given flags are set.

        Handles both encoded and decoded column states internally.

        Args:
            flags: One or more flag names or values to match against.

        Returns:
            A boolean Polars expression.
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

    def filter_expr(self, flags: list[int | str]) -> pl.Expr:
        """Return a boolean expression that is True for rows where any of the given flags are set.

        For an encoded column, does a bitwise OR to check for existence.
        For a decoded column, checks whether the list contains any of the flag names.

        Args:
            flags: One or more flag names or bit values to match against.

        Returns:
            A boolean Polars expression.

        Raises:
            BitwiseFlagUnknownError: If any flag is not in the flag system.
        """
        # Fetch the actual flag enum members based on the flag values provided
        flag_members = [self.flag_system.get_flag(f) for f in flags]

        if self.is_decoded:
            exprs = [pl.col(self.name).list.contains(pl.lit(f.name)) for f in flag_members]
            return pl.any_horizontal(exprs)

        combined = 0
        for f in flag_members:
            combined |= int(f)
        return (pl.col(self.name) & pl.lit(combined)) != 0

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
class CategoricalSingleFlagColumn(FlagColumn):
    """Represents a categorical flag column where each row holds exactly one flag value (or null).

    Flags are mutually exclusive - setting a new flag replaces the existing value. Governed by a
    ``CategoricalSingleFlag`` system.

    Attributes:
        name: Name of the flag column in the DataFrame.
        flag_system: The ``CategoricalSingleFlag`` enum class that defines the available flag values.
        is_decoded: Whether the column is currently in decoded (flag-name) form rather than raw-value form.
    """

    name: str
    flag_system: type[CategoricalSingleFlag]
    is_decoded: bool = False

    def decode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace raw flag values with their flag names.

        Each value is replaced by its name (e.g. ``123 -> "good"``).

        Args:
            df: The DataFrame containing the raw-value flag column.

        Returns:
            A new DataFrame with the flag column replaced by a ``Utf8`` column of flag names.
        """
        flag_map = self.flag_system.to_dict()
        old = list(flag_map.values())
        new = list(flag_map.keys())
        return df.with_columns(
            pl.col(self.name).replace_strict(old=old, new=new, default=None, return_dtype=pl.Utf8).alias(self.name)
        )

    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace flag names back to their raw values.

        Each name is replaced by its value (e.g. ``"good" -> 123``).

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

        present = set(df[self.name].drop_nulls().unique().to_list())
        unknown = present - flag_map.keys()
        if unknown:
            raise CategoricalFlagUnknownError(f"Unknown flag names in column '{self.name}': {sorted(unknown)}.")

        old = list(flag_map.keys())
        new = list(flag_map.values())
        return df.with_columns(
            pl.col(self.name).replace_strict(old=old, new=new, default=None, return_dtype=return_dtype).alias(self.name)
        )

    def add_flag(
        self,
        df: pl.DataFrame,
        flag: int | str,
        expr: pl.Expr = pl.lit(True),
        overwrite: bool = True,
    ) -> pl.DataFrame:
        """Set the flag value on rows where ``expr`` is true.

        When ``overwrite`` is ``False``, only rows whose current value is null are updated.

        If the column is currently decoded, it is encoded first, the flag is applied, and the result is
        decoded again.

        Args:
            df: The DataFrame containing the flag column.
            flag: The flag name or value to set.
            expr: A Polars expression defining which rows to update.
                  Defaults to ``pl.lit(True)``, meaning all rows.
            overwrite: If ``True`` (default), replaces any existing value. If ``False``,
                only updates rows whose current value is null.

        Returns:
            A new DataFrame with the flag column updated.
        """
        value = self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        col_dtype = df[self.name].dtype
        condition = expr if overwrite else (expr & pl.col(self.name).is_null())
        df = df.with_columns(
            pl.when(condition).then(pl.lit(value, dtype=col_dtype)).otherwise(pl.col(self.name)).alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def remove_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Set the column value to null on rows where ``expr`` is true.

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
        self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        col_dtype = df[self.name].dtype
        df = df.with_columns(
            pl.when(expr).then(pl.lit(None, dtype=col_dtype)).otherwise(pl.col(self.name)).alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def filter_expr(self, flags: list[int | str]) -> pl.Expr:
        """Return a boolean expression that is True for rows where the column value matches any of the given flags.

        Args:
            flags: One or more flag names or values to match against.

        Returns:
            A boolean Polars expression.

        Raises:
            CategoricalFlagUnknownError: If any flag is not in the flag system.
        """
        flag_members = [self.flag_system.get_flag(f) for f in flags]
        values = [f.name if self.is_decoded else f.value for f in flag_members]
        return pl.col(self.name).is_in(values)

    def __eq__(self, other: object) -> bool:
        """Check if two ``CategoricalSingleFlagColumn`` instances are equal.

        Compares ``name`` and ``flag_system`` only. ``is_decoded`` is runtime state and is excluded
        from the comparison.

        Args:
            other: The object to compare.

        Returns:
            True if both instances have the same name and flag system, False otherwise.
        """
        if not isinstance(other, CategoricalSingleFlagColumn):
            return False
        return self.name == other.name and self.flag_system == other.flag_system

    # Make class instances unhashable
    __hash__ = None


@dataclass
class CategoricalListFlagColumn(FlagColumn):
    """Represents a categorical flag column where each row holds a list of flag values.

    Flags can coexist - multiple flags can be present in a single row simultaneously. Governed by a
    ``CategoricalListFlag`` system.

    Attributes:
        name: Name of the flag column in the DataFrame.
        flag_system: The ``CategoricalListFlag`` enum class that defines the available flag values.
        is_decoded: Whether the column is currently in decoded (flag-name) form rather than raw-value form.
    """

    name: str
    flag_system: type[CategoricalListFlag]
    is_decoded: bool = False

    def decode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace raw flag values in each list with their flag names.

        Each element of each list is replaced by its name.

        Args:
            df: The DataFrame containing the raw-value flag column.

        Returns:
            A new DataFrame with the flag column replaced by a ``List(Utf8)`` column of flag names.
        """
        flag_map = self.flag_system.to_dict()
        old = list(flag_map.values())
        new = list(flag_map.keys())
        return df.with_columns(
            pl.col(self.name)
            .list.eval(pl.element().replace_strict(old=old, new=new, default=None, return_dtype=pl.Utf8))
            .alias(self.name)
        )

    def encode(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace flag names in each list back to their raw values.

        Each element of each list is replaced by its value.

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

        present = set(df[self.name].explode().drop_nulls().unique().to_list())
        unknown = present - flag_map.keys()
        if unknown:
            raise CategoricalFlagUnknownError(f"Unknown flag names in column '{self.name}': {sorted(unknown)}.")

        old = list(flag_map.keys())
        new = list(flag_map.values())
        return df.with_columns(
            pl.col(self.name)
            .list.eval(pl.element().replace_strict(old=old, new=new, default=None, return_dtype=return_dtype))
            .alias(self.name)
        )

    def add_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Append a flag value to the list on rows where ``expr`` is true.

        If the flag is already present in a row's list, it is not added again.

        If the column is currently decoded, it is encoded first, the flag is applied, and the result is
        decoded again.

        Args:
            df: The DataFrame containing the flag column.
            flag: The flag name or value to append.
            expr: A Polars expression defining which rows to update.
                  Defaults to ``pl.lit(True)``, meaning all rows.

        Returns:
            A new DataFrame with the flag column updated.
        """
        value = self.flag_system.get_flag(flag)
        if self.is_decoded:
            df = self.encode(df)
        df = df.with_columns(
            pl.when(expr & ~pl.col(self.name).list.contains(pl.lit(value)))
            .then(pl.concat_list([pl.col(self.name), pl.lit(value).implode()]))
            .otherwise(pl.col(self.name))
            .alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def remove_flag(self, df: pl.DataFrame, flag: int | str, expr: pl.Expr = pl.lit(True)) -> pl.DataFrame:
        """Remove all occurrences of a flag value from the list on rows where ``expr`` is true.

        If the flag is not present, the row is unchanged.

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
        df = df.with_columns(
            pl.when(expr)
            .then(pl.col(self.name).list.eval(pl.element().filter(pl.element() != pl.lit(value))))
            .otherwise(pl.col(self.name))
            .alias(self.name)
        )
        if self.is_decoded:
            df = self.decode(df)
        return df

    def filter_expr(self, flags: list[int | str]) -> pl.Expr:
        """Return a boolean expression that is True for rows where any of the given flags are set.

        For scalar mode, checks whether the column value is any of the given flag values.
        For list mode, checks whether the list contains any of the given flag values.

        Args:
            flags: One or more flag names or values to match against.

        Returns:
            A boolean Polars expression.

        Raises:
            CategoricalFlagUnknownError: If any flag is not in the flag system.
        """
        # Fetch the actual flag enum members based on the flag values provided
        flag_members = [self.flag_system.get_flag(f) for f in flags]
        values = [f.name if self.is_decoded else f.value for f in flag_members]
        exprs = [pl.col(self.name).list.contains(pl.lit(v)) for v in values]
        return pl.any_horizontal(exprs)

    def __eq__(self, other: object) -> bool:
        """Check if two ``CategoricalListFlagColumn`` instances are equal.

        Compares ``name`` and ``flag_system`` only. ``is_decoded`` is runtime state and is excluded
        from the comparison.

        Args:
            other: The object to compare.

        Returns:
            True if both instances have the same name and flag system, False otherwise.
        """
        if not isinstance(other, CategoricalListFlagColumn):
            return False
        return self.name == other.name and self.flag_system == other.flag_system

    # Make class instances unhashable
    __hash__ = None


class FlagManager:
    """Registry for flag systems and flag columns.

    This class:
      * registers **flag systems** (bitwise, categorical single, or categorical list) under a string name;
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

        Three kinds of flag system are supported:

        **Bitwise** (``BitwiseFlag``): values are powers of two, enabling multiple flags to be combined
        on a single integer per row. Pass ``flag_type="bitwise"`` (the default).

        **Categorical single** (``CategoricalSingleFlag``): values are arbitrary ``int`` or ``str``; each
        row holds exactly one value (flags are mutually exclusive). Pass ``flag_type="categorical"``.

        **Categorical list** (``CategoricalListFlag``): values are arbitrary ``int`` or ``str``; each row
        holds a list of values (flags can coexist). Pass ``flag_type="categorical_list"``.

        Accepted inputs for ``flag_system``:

        - ``None`` - produces a default bitwise system with a single ``FLAGGED`` flag at value 1.
        - ``dict[str, int]`` - interpreted as bitwise or categorical depending on ``flag_type``.
          Bitwise values must be powers of two.
        - ``dict[str, str]`` - categorical by default (``flag_type`` inferred as ``"categorical"`` unless
          ``"categorical_list"`` is explicitly passed).
        - ``list[str]`` - flag names are sorted; bitwise assigns powers of two, categorical uses
          each string as both the flag name and its value. An empty list produces the default ``FLAGGED`` flag.

        Args:
            flag_system_name: The name of the new flag system.
            flag_system: The flag system definition. Defaults to ``None``.
            flag_type: Whether to create a ``"bitwise"``, ``"categorical"``, or ``"categorical_list"``
                flag system. Only relevant when ``flag_system`` is a ``dict[str, int]`` or ``list[str]``.

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

            if flag_type == "categorical_list":
                sorted_categories = sorted(flag_system)
                flag_dict = {name: name for name in sorted_categories}
                self._flag_systems[flag_system_name] = CategoricalListFlag(flag_system_name, flag_dict)
            elif flag_type == "categorical":
                sorted_categories = sorted(flag_system)
                flag_dict = {name: name for name in sorted_categories}
                self._flag_systems[flag_system_name] = CategoricalSingleFlag(flag_system_name, flag_dict)
            else:
                sorted_categories = sorted(flag_system)
                flag_dict = {name: 2**i for i, name in enumerate(sorted_categories)}
                self._flag_systems[flag_system_name] = BitwiseFlag(flag_system_name, flag_dict)

        elif isinstance(flag_system, dict):
            # A dict[str, str] implies categorical; infer it if the caller hasn't specified a categorical type
            if all(isinstance(v, str) for v in flag_system.values()) and flag_type == "bitwise":
                flag_type = "categorical"

            if flag_type == "categorical_list":
                self._flag_systems[flag_system_name] = CategoricalListFlag(flag_system_name, flag_system)
            elif flag_type == "categorical":
                self._flag_systems[flag_system_name] = CategoricalSingleFlag(flag_system_name, flag_system)
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
            The ``BitwiseFlag``, ``CategoricalSingleFlag``, or ``CategoricalListFlag`` enum class.

        Raises:
            FlagSystemNotFoundError: If no flag system with ``flag_system_name`` is registered.
        """
        try:
            return self._flag_systems[flag_system_name]
        except KeyError:
            raise FlagSystemNotFoundError(f"No such flag system: '{flag_system_name}'")

    def register_flag_column(self, name: str, flag_system_name: str) -> None:
        """Mark the specified existing column as a flag column.

        The column type (``BitwiseFlagColumn``, ``CategoricalSingleFlagColumn``, or
        ``CategoricalListFlagColumn``) is determined by the flag system type.

        Args:
            name: A column name to mark as a flag column.
            flag_system_name: The name of the flag system.

        Raises:
            FlagSystemError: If a flag column with ``name`` is already registered.
            FlagSystemNotFoundError: If ``flag_system_name`` is not a registered flag system.
        """
        if name in self._flag_columns:
            raise FlagSystemError(f"Flag column '{name}' already registered. System: '{flag_system_name}'.")

        flag_system = self.get_flag_system(flag_system_name)
        if flag_system.flag_type == "categorical_list":
            self._flag_columns[name] = CategoricalListFlagColumn(name, flag_system)
        elif flag_system.flag_type == "categorical":
            self._flag_columns[name] = CategoricalSingleFlagColumn(name, flag_system)
        else:
            self._flag_columns[name] = BitwiseFlagColumn(name, flag_system)

    def get_flag_column(self, name: str) -> FlagColumn:
        """Look up a registered flag column by name.

        Args:
            name: Flag column name.

        Returns:
            The corresponding ``BitwiseFlagColumn``, ``CategoricalSingleFlagColumn``, or
            ``CategoricalListFlagColumn`` instance.

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

        for name, flag_system in self._flag_systems.items():
            out.register_flag_system(name, flag_system.to_dict(), flag_type=flag_system.flag_type)

        for name, flag_column in self._flag_columns.items():
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
