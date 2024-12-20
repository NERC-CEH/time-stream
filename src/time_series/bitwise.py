from enum import Flag
from typing import Dict, List, TypeVar

BWFlag = TypeVar("BWFlag", bound=Flag)


class BitwiseFlag(Flag):
    @classmethod
    def from_value(cls, value: int) -> "BitwiseFlag":
        """
        Create an instance of the flag from an integer value.
        """
        if value < 0:
            raise ValueError(f"Invalid value {value}")
        return cls(value)


class FlagManager:
    def __init__(self, flag_class: "BitwiseFlag") -> None:
        """
        Initialize the FlagManager with a specific flag class.

        Args:
            flag_class: The class of the flags to manage.
        """
        self.flag_class = flag_class
        self.flags = self.flag_class(0)

    def add_flag(self, flag: BWFlag) -> None:
        """
        Add a flag to the current flags.

        Args:
            flag (BWFlag): The flag to add.
        """
        self.flags |= flag

    def remove_flag(self, flag: BWFlag) -> None:
        """
        Remove a flag from the current flags.

        Args:
            flag (BWFlag): The flag to remove.
        """
        self.flags &= ~flag

    def has_flag(self, flag: BWFlag) -> bool:
        """
        Check if a specific flag is set.

        Args:
            flag (BWFlag): The flag to check.

        Returns:
            bool: True if the flag is set, False otherwise.
        """
        return (self.flags & flag) == flag


class BitWiseValidator:
    """
    Validates that a list of bitwise values are valid and non-wasteful.
    """

    @staticmethod
    def _check_flag_type(test_flag: int) -> None:
        """Checks the type of the flag and raises a TypeError if
        not an integer.

        Args:
            test_flag: A numeric value.
        Raises:
            TypeError: Raises if type is not int.
        """
        if not isinstance(test_flag, int):
            raise TypeError(f'A bitwise value must be an integer, received "{type(test_flag)}"')

    @staticmethod
    def _flags_are_unique(test_flags: List[int]) -> bool:
        """Checks if flags are unique

        Args:
            test_flags: A list of flag values.

        Raises:
            ValueError: If flags are not unique.
        """
        if len(test_flags) != len(set(test_flags)):
            raise ValueError(f"Flags are not unique: {test_flags}")

    @staticmethod
    def _flags_are_sequential(test_flags: List[int]) -> bool:
        """Checks that flags are sequential and start at number 1.

        Args:
            test_flags: A list of flag values.

        Raises:
            ValueError: If flags are not sequential starting at 1.
        """
        for i, test_flag in enumerate(test_flags):
            BitWiseValidator._check_flag_type(test_flag)

            if test_flag != 1 << i:
                raise ValueError(f"Flags are not sequential: {test_flags}")

    @staticmethod
    def _flags_are_bitwise(test_flags: List[int]) -> bool:
        """Checks that all test flags are bitwise.

        Args:
            test_flags: A list of flag values.

        Raises:
            ValueError: If any flag is not a bitwise value.
        """

        for test_flag in test_flags:
            BitWiseValidator._check_flag_type(test_flag)

            if test_flag == 0 or ((test_flag & (test_flag - 1)) != 0):
                raise ValueError(f"Flag is not a bitwise value: {test_flag}")

        return True

    @staticmethod
    def validate(flag_values: List[int]) -> bool:
        """Checks that flags in a list are valid.

        Args:
            flag_values: A list of flag values.
        """
        BitWiseValidator._flags_are_unique(flag_values)
        BitWiseValidator._flags_are_bitwise(flag_values)
        BitWiseValidator._flags_are_sequential(flag_values)


def create_flag_class(name: str, flag_dict: Dict[str, int]) -> BitwiseFlag:
    """
    Create a BitwiseFlag class dynamically based on metadata, with validation.

    Args:
        name (str): The name of the class.
        flag_dict (Dict[str, int]): The flag name and value pair dictionary.

    Returns:
        BitwiseFlag: The dynamically created class.
    """
    flag_values = list(flag_dict.values())
    BitWiseValidator.validate(flag_values)

    return BitwiseFlag(name, flag_dict)
