"""
Categorical Flag Module.

Provides a flag system for arbitrary integer or string flag values.

Unlike ``BitwiseFlag``, values do not need to be powers of two - they are separate categories such as
``{"good": 0, "questionable": 1, "bad": 2}`` or string codes like ``{"good": "G", "bad": "B"}``.

Inherits from ``FlagSystemBase`` for the shared flag system interface and from ``enum.Enum`` for member access.

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
from time_stream.flags.flag_system import FlagMeta, FlagSystemBase


class CategoricalMeta(FlagMeta):
    """Metaclass for ``CategoricalFlag`` enums.

    ``BitwiseMeta`` and ``CategoricalMeta`` are kept as distinct subclasses so that a ``BitwiseFlag``
    class and a ``CategoricalFlag`` class with the same name and member values are never equal.
    """


class CategoricalFlag(FlagSystemBase, Enum, metaclass=CategoricalMeta):
    """A categorical flag enum with arbitrary ``int`` or ``str`` values.

    Inherits from ``FlagSystemBase`` for the shared flag system interface (``system_name``, ``to_dict``,
    ``get_flag``, ``value_type``) and from ``enum.Enum`` for member access.

    Flag values must all be the same type (``int`` or ``str``) and must be unique. Dynamic creation
    is supported, e.g.::

        QC = CategoricalFlag("QC", {"good": 0, "questionable": 1, "bad": 2})
        CODES = CategoricalFlag("CODES", {"good": "G", "bad": "B"})
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
    def get_flag(cls, flag: int | str) -> "CategoricalFlag":
        """Look up a flag by name or by value.

        For string-valued systems, a string argument is checked against names first, then against values.
        For int-valued systems, a string argument is checked against names and an int argument is checked
        against values.

        Args:
            flag: The flag name (``str``) or value (``int`` for int-valued systems, ``str`` for str-valued systems).

        Returns:
            The corresponding ``CategoricalFlag`` member.

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
    def validate_column(cls, series: pl.Series, list_mode: bool | None = None) -> None:
        """Validate that all non-null values in ``series`` are valid for this flag system.

        Args:
            series: The Polars Series to validate.
            list_mode: Whether the series contains lists of values. If ``True``, the series is
                exploded before validation.

        Raises:
            CategoricalFlagUnknownError: If the series contains values not in this flag system.
        """
        valid_values = set(cls.to_dict().values())
        flat = series.explode().drop_nulls() if list_mode else series.drop_nulls()
        unknown = set(flat.unique().to_list()) - valid_values

        if unknown:
            raise CategoricalFlagUnknownError(
                f"Column '{series.name}' contains values not in flag system '{cls.system_name()}': {sorted(unknown)}."
            )
