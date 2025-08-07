from enum import EnumType, Flag
from typing import Union


class BitwiseMeta(EnumType):
    @property
    def name(self) -> str:
        return self.__name__

    def __repr__(self):
        """Return a helpful representation of the flag, listing all enum members."""
        members = ", ".join(f"{name}={member.value}" for name, member in self.__members__.items())
        return f"<{self.name} ({members})>"


class BitwiseFlag(int, Flag, metaclass=BitwiseMeta):
    """A flag enumeration that allows efficient flagging using bitwise operations."""

    def __init_subclass__(cls, **kwargs):
        """Validate enum member values on class definition."""
        super().__init_subclass__(**kwargs)
        for name, member in cls.__members__.items():
            cls._check_type(member.value)
            cls._check_bitwise(member.value)
            cls._check_unique(member.value)

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
        all_values = [i.value for _, i in cls.__members__.items()]
        if all_values.count(value) > 1:
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
            if flag not in cls.__members__.values():
                raise KeyError(f"Flag value {flag} is not a valid singular flag.")
            return cls(flag)
        else:
            raise TypeError(f"Flag value must be an integer or string, not {type(flag)}.")
