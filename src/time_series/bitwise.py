from enum import Flag
from typing import Union


class BitwiseFlag(int, Flag):
    def __new__(cls, value):
        cls._check_type(value)
        cls._check_bitwise(value)
        cls._check_unique(value)

        return super().__new__(cls, value)

    @staticmethod
    def _check_type(value):
        """Checks that all test flags are bitwise.
        """
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Flag value must be a positive integer: {value}")

    @staticmethod
    def _check_bitwise(value):
        """Checks that all test flags are bitwise.
        """
        if value == 0 or ((value & (value - 1)) != 0):
            raise ValueError(f"Flag is not a bitwise value: {value}")

    @classmethod
    def _check_unique(cls, value):
        if value in cls:
            raise ValueError(f"Flag is not unique: {value}")

    @classmethod
    def get_single_flag(cls, flag: Union[int, str]) -> Flag:
        """ Create an instance of the flag from an integer or string value.

        This must be a single flag value, for example can't be a combination of integer flag values:
        "MISSING": 1,
        "ESTIMATED": 2,
        "CORRECTED": 4
        Can't ask for flag "3" - a combination of MISSING and ESTIMATED.
        """
        if isinstance(flag, str):
            return cls.__getitem__(flag)
        elif isinstance(flag, int):
            if flag not in cls:
                raise KeyError(f"Flag value {flag} is not a valid singular flag.")
            return cls(flag)
        else:
            raise TypeError(f"Flag value must be an integer or string, not {type(flag)}.")
