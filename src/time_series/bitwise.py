from enum import EnumType, Flag
from typing import Union


class BitwiseMeta(EnumType):
    def __repr__(self):
        """Return a helpful representation of the flag, listing all enum members."""
        members = ", ".join(f"{name}={member.value}" for name, member in self.__members__.items())
        return f"<{self.__name__} ({members})>"


class BitwiseFlag(int, Flag, metaclass=BitwiseMeta):
    """A flag enumeration that allows efficient flagging using bitwise operations."""

    def __new__(cls, value: int) -> Union[int, Flag]:
        """Creates a new BitwiseFlag instance.

        Args:
            value: The integer representation of the flag.

        Raises:
            ValueError: If the value is not a positive power of two or is not unique.
        """
        cls._check_type(value)
        cls._check_bitwise(value)
        cls._check_unique(value)

        return super().__new__(cls, value)

    @staticmethod
    def _check_type(value: int) -> None:
        """Ensures the flag value is a positive integer.

        Args:
            value: The flag value.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Flag value must be a positive integer: {value}")

    @staticmethod
    def _check_bitwise(value: int) -> None:
        """Ensures the flag value is a power of two.

        This validation ensures that each flag represents a single unique state and can be combined efficiently
        using bitwise OR operations.

        Args:
            value: The flag value.

        Raises:
            ValueError: If the value is not a power of two.
        """
        if value == 0 or ((value & (value - 1)) != 0):
            raise ValueError(f"Flag is not a bitwise value: {value}")

    @classmethod
    def _check_unique(cls, value: int) -> None:
        """Ensures the flag value is unique within the enumeration.

        Args:
            value: The flag value.

        Raises:
            ValueError: If the flag value is already defined in the enumeration.
        """
        if value in cls:
            raise ValueError(f"Flag is not unique: {value}")

    @classmethod
    def get_single_flag(cls, flag: Union[int, str]) -> "BitwiseFlag":
        """Retrieves a single flag from an integer or string value.

        Can't be a combination of integer flag values, for example with the classification of:
        "MISSING": 1,
        "ESTIMATED": 2,
        "CORRECTED": 4
        > Can't ask for flag "3" - a combination of MISSING and ESTIMATED.

        Args:
            flag: The flag identifier.

        Returns:
            BitwiseFlag: The corresponding BitwiseFlag instance.

        Raises:
            KeyError: If the flag value is not valid or is a combination of multiple flags.
            TypeError: If the flag value is not an integer or string.
        """
        if isinstance(flag, str):
            return cls.__getitem__(flag)
        elif isinstance(flag, int):
            if flag not in cls:
                raise KeyError(f"Flag value {flag} is not a valid singular flag.")
            return cls(flag)
        else:
            raise TypeError(f"Flag value must be an integer or string, not {type(flag)}.")
