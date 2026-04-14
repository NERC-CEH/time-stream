"""
Categorical Flag Module.

Provides flag systems for arbitrary integer or string flag values.

Unlike ``BitwiseFlag``, values do not need to be powers of two - they are separate categories such as
``{"good": 0, "questionable": 1, "bad": 2}`` or string codes like ``{"good": "G", "bad": "B"}``.

Two categorical flag system types are supported:

- ``CategoricalSingleFlag`` - each row holds exactly one flag value (or null). Flags are mutually exclusive.
- ``CategoricalListFlag`` - each row holds a list of flag values. Flags can coexist.

Both inherit from ``FlagSystemBase`` for the shared flag system interface and from ``enum.Enum`` for member access.

Validation includes:
  - all values are ``int`` or all values are ``str``,
  - no duplicate keys or values within the mapping.
"""

from enum import Enum

import polars as pl

from time_stream.exceptions import (
    CategoricalFlagTypeError,
    CategoricalFlagUnknownError,
    CategoricalFlagValueError,
)
from time_stream.flags.flag_system import FlagMeta, FlagSystemBase, FlagSystemLiteral


class CategoricalSingleMeta(FlagMeta):
    """Metaclass for ``CategoricalSingleFlag`` enums.

    Kept distinct from ``BitwiseMeta`` and ``CategoricalListMeta`` so that flag system classes of
    different types are never considered equal.
    """

    flag_type: FlagSystemLiteral = "categorical"


class CategoricalListMeta(CategoricalSingleMeta):
    """Metaclass for ``CategoricalListFlag`` enums."""

    flag_type: FlagSystemLiteral = "categorical_list"


class CategoricalSingleFlag(FlagSystemBase, Enum, metaclass=CategoricalSingleMeta):
    """A categorical flag enum where each row holds exactly one flag value (or null).

    Flags are mutually exclusive - setting a new flag replaces the existing value. Inherits from
    ``FlagSystemBase`` for the shared flag system interface (``system_name``, ``to_dict``,
    ``get_flag``, ``value_type``) and from ``enum.Enum`` for member access.

    Flag values must all be the same type (``int`` or ``str``) and must be unique. Dynamic creation
    is supported, e.g.::

        QC = CategoricalSingleFlag("QC", {"good": 0, "questionable": 1, "bad": 2})
        CODES = CategoricalSingleFlag("CODES", {"good": "G", "bad": "B"})
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate member values on class definition."""
        super().__init_subclass__(**kwargs)
        if cls.__members__:
            mapping = {n: m.value for n, m in cls.__members__.items()}
            cls._check_value_types(mapping)
            cls._check_unique_values(cls.__name__, mapping)

    @staticmethod
    def _check_value_types(mapping: dict[str, int | str]) -> None:
        """Ensure all values in the mapping are ``int`` or all are ``str``.

        Args:
            mapping: The flag name-to-value mapping to validate.

        Raises:
            CategoricalFlagTypeError: If values are mixed types or contain types other than ``int``/``str``.
        """
        value_types = {type(v) for v in mapping.values()}
        if not value_types.issubset({int, str}):
            raise CategoricalFlagTypeError("All flag values must be int or str.")
        if len(value_types) > 1:
            raise CategoricalFlagTypeError("All flag values must be the same type (all int or all str).")

    @staticmethod
    def _check_unique_values(name: str, mapping: dict[str, int | str]) -> None:
        """Ensure no two flag names map to the same value.

        Args:
            name: The flag system name, used in the error message.
            mapping: The flag name-to-value mapping to validate.

        Raises:
            CategoricalFlagValueError: If any value appears more than once.
        """
        values = list(mapping.values())
        if len(set(values)) != len(values):
            raise CategoricalFlagValueError(f"Duplicate values in categorical flag mapping for '{name}': {values}.")

    @classmethod
    def get_flag(cls, flag: int | str) -> "CategoricalSingleFlag":
        """Look up a flag by name or by value.

        For string-valued systems, a string argument is checked against names first, then against values.
        For int-valued systems, a string argument is checked against names and an int argument is checked
        against values.

        Args:
            flag: The flag name (``str``) or value (``int`` for int-valued systems, ``str`` for str-valued systems).

        Returns:
            The corresponding ``CategoricalSingleFlag`` member.

        Raises:
            CategoricalFlagUnknownError: If ``flag`` does not match any name or value.
            CategoricalFlagTypeError: If ``flag`` is neither ``int`` nor ``str``.
        """
        if isinstance(flag, str):
            if flag in cls.__members__:
                return cls[flag]
            for member in cls.__members__.values():
                if member.value == flag:
                    return member
            raise CategoricalFlagUnknownError(f"Flag '{flag}' not found in categorical flag system '{cls.__name__}'.")

        elif isinstance(flag, int):
            for member in cls.__members__.values():
                if member.value == flag:
                    return member
            raise CategoricalFlagUnknownError(
                f"Flag value {flag} not found in categorical flag system '{cls.__name__}'."
            )

        else:
            raise CategoricalFlagTypeError(f"Flag must be int or str, not {type(flag).__name__}.")

    @classmethod
    def value_type(cls) -> type:
        """Return the Python type of the flag values (``int`` or ``str``).

        Returns:
            ``int`` if values are integers, ``str`` if values are strings.
        """
        first = next(iter(cls.__members__.values()))
        return type(first.value)

    @classmethod
    def validate_column(cls, series: pl.Series) -> None:
        """Validate that all non-null values in ``series`` are valid for this flag system.

        Args:
            series: The Polars Series to validate. Expected to contain scalar values.

        Raises:
            CategoricalFlagUnknownError: If the series contains values not in this flag system.
        """
        valid_values = set(cls.to_dict().values())
        unknown = set(series.drop_nulls().unique().to_list()) - valid_values

        if unknown:
            raise CategoricalFlagUnknownError(
                f"Column '{series.name}' contains values not in flag system '{cls.system_name()}': {sorted(unknown)}."
            )


class CategoricalListFlag(CategoricalSingleFlag, metaclass=CategoricalListMeta):
    """A categorical flag enum where each row holds a list of flag values.

    Flags can coexist - multiple flags can be present in a single row simultaneously. Inherits all
    validation and lookup behaviour from ``CategoricalSingleFlag``.
    """

    @classmethod
    def validate_column(cls, series: pl.Series) -> None:
        """Validate that all non-null values in ``series`` are valid for this flag system.

        Args:
            series: The Polars Series to validate. Expected to contain lists of values; the series
                is exploded before validation.

        Raises:
            CategoricalFlagUnknownError: If the series contains values not in this flag system.
        """
        valid_values = set(cls.to_dict().values())
        unknown = set(series.explode().drop_nulls().unique().to_list()) - valid_values

        if unknown:
            raise CategoricalFlagUnknownError(
                f"Column '{series.name}' contains values not in flag system '{cls.system_name()}': {sorted(unknown)}."
            )
