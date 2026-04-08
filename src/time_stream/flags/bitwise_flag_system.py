"""
Bitwise Flag Module.

Provides ``BitwiseFlag``, the concrete base class for bitwise flag systems. Each flag value must be
a power of two, ensuring each flag occupies exactly one bit and that combined values are unique.

Inherits from ``FlagSystemBase`` for the shared flag system interface and from ``enum.Flag`` for
bitwise semantics.

Validation includes:
  - values are non-negative integers,
  - each value is a power of two,
  - each value is unique within the enum.
"""

from enum import Flag

import polars as pl

from time_stream.exceptions import (
    BitwiseFlagDuplicateError,
    BitwiseFlagTypeError,
    BitwiseFlagUnknownError,
    BitwiseFlagValueError,
)
from time_stream.flags.flag_system import FlagMeta, FlagSystemBase


class BitwiseMeta(FlagMeta):
    """Metaclass for ``BitwiseFlag`` enums.

    ``BitwiseMeta`` and ``CategoricalMeta`` are kept as distinct subclasses so that a ``BitwiseFlag``
    class and a ``CategoricalFlag`` class with the same name and member values are never equal.
    """


class BitwiseFlag(FlagSystemBase, int, Flag, metaclass=BitwiseMeta):
    """A flag enumeration that allows efficient flagging using bitwise operations.

    Inherits from ``FlagSystemBase`` for the shared flag system interface and from ``enum.Flag`` for
    bitwise OR/AND semantics.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate enum member values on class definition."""
        super().__init_subclass__(**kwargs)
        for member in cls.__members__.values():
            cls._check_type(member.value)
            cls._check_bitwise(member.value)
            cls._check_unique(member.value)

    @staticmethod
    def _check_type(value: int) -> None:
        """Ensure the flag value is a non-negative integer.

        Args:
            value: The flag value.

        Raises:
            BitwiseFlagTypeError: If the value is not a non-negative integer.
        """
        if not isinstance(value, int) or value < 0:
            raise BitwiseFlagTypeError(f"Flag value must be a non-negative integer: {value}")

    @staticmethod
    def _check_bitwise(value: int) -> None:
        """Ensure the flag value is a power of two.

        Args:
            value: The flag value.

        Raises:
            BitwiseFlagValueError: If the value is not a power of two.
        """
        if value == 0 or ((value & (value - 1)) != 0):
            raise BitwiseFlagValueError(f"Flag is not a bitwise value: {value}")

    @classmethod
    def _check_unique(cls, value: int) -> None:
        """Ensure the flag value is unique within the enumeration.

        Args:
            value: The flag value.

        Raises:
            BitwiseFlagDuplicateError: If the flag value is already defined in the enumeration.
        """
        all_values = [m.value for m in cls.__members__.values()]
        if all_values.count(value) > 1:
            raise BitwiseFlagDuplicateError(f"Flag is not unique: {value}")

    @classmethod
    def get_flag(cls, flag: int | str) -> "BitwiseFlag":
        """Look up a flag by name or by value.

        Only singular (non-combined) flag values are accepted. For example, with flags ``MISSING=1``,
        ``ESTIMATED=2``, ``CORRECTED=4``, passing ``3`` raises an error because it is a combination of
        ``MISSING`` and ``ESTIMATED``.

        Args:
            flag: The flag name (``str``) or bit value (``int``).

        Returns:
            The corresponding ``BitwiseFlag`` member.

        Raises:
            BitwiseFlagUnknownError: If the flag value is not valid or is a combination of multiple flags.
            BitwiseFlagTypeError: If ``flag`` is neither ``int`` nor ``str``.
        """
        if isinstance(flag, str):
            try:
                return cls.__getitem__(flag)
            except KeyError:
                raise BitwiseFlagUnknownError(f"Flag value '{flag}' is not a valid singular flag.")

        elif isinstance(flag, int):
            if flag not in cls.__members__.values():
                raise BitwiseFlagUnknownError(f"Flag value '{flag}' is not a valid singular flag.")
            return cls(flag)

        else:
            raise BitwiseFlagTypeError(f"Flag value must be an integer or string, not {type(flag)}.")

    @classmethod
    def value_type(cls) -> type:
        """Return ``int`` - bitwise flag values are always integers."""
        return int

    @classmethod
    def validate_column(cls, series: pl.Series, list_mode: bool | None = None) -> None:
        """Validate that all non-null values in ``series`` are valid bitwise combinations.

        A value is valid if all of its set bits correspond to flags defined in this system.

        Args:
            series: The Polars Series to validate.
            list_mode: Unused for bitwise flag systems.

        Raises:
            BitwiseFlagUnknownError: If the series contains values with bits not in this flag system.
        """
        known_mask = sum(cls.to_dict().values())
        s = series.drop_nulls()
        invalid = s.filter((s & ~known_mask) != 0).unique().to_list()  # type: ignore[arg-type]

        if invalid:
            raise BitwiseFlagUnknownError(
                f"Column '{series.name}' contains values with bits not defined in flag system "
                f"'{cls.system_name()}': {sorted(invalid)}."
            )
