from typing import Any

import polars as pl
import pytest

from time_stream.bitwise import BitwiseFlag
from time_stream.exceptions import (
    BitwiseFlagDuplicateError,
    BitwiseFlagTypeError,
    BitwiseFlagUnknownError,
    BitwiseFlagValueError,
)


class Flags(BitwiseFlag):
    FLAG_A = 1
    FLAG_B = 2
    FLAG_C = 4


class TestBitwiseFlag:
    def test_valid_flags(self) -> None:
        """Test creating valid bitwise flags."""
        assert Flags.FLAG_A == 1
        assert Flags.FLAG_B == 2
        assert Flags.FLAG_C == 4

    def test_invalid_flag_value_not_power_of_two(self) -> None:
        """Test that a non-power-of-two flag value raises error."""
        with pytest.raises(BitwiseFlagValueError):

            class _Flags(BitwiseFlag):  # noqa - expect class unused warning
                INVALID_FLAG = 3

    def test_invalid_flag_value_negative(self) -> None:
        """Test that a negative flag value raises error."""
        with pytest.raises(BitwiseFlagTypeError):

            class _Flags(BitwiseFlag):  # noqa - expect class unused warning
                INVALID_FLAG = -2

    def test_flag_uniqueness(self) -> None:
        """Test that duplicate flag values raise ValueError."""
        with pytest.raises(BitwiseFlagDuplicateError):

            class _Flags(BitwiseFlag):  # noqa - expect class unused warning
                FLAG_A = 1
                FLAG_B = 2
                FLAG_C = 2

    def test_get_single_flag_by_integer(self) -> None:
        """Test retrieving a flag by integer value."""
        assert Flags.get_single_flag(1) == Flags.FLAG_A
        assert Flags.get_single_flag(2) == Flags.FLAG_B
        assert Flags.get_single_flag(4) == Flags.FLAG_C

    def test_get_single_flag_by_string(self) -> None:
        """Test retrieving a flag by string name."""
        assert Flags.get_single_flag("FLAG_A") == Flags.FLAG_A
        assert Flags.get_single_flag("FLAG_B") == Flags.FLAG_B
        assert Flags.get_single_flag("FLAG_C") == Flags.FLAG_C

    def test_get_single_flag_invalid_integer(self) -> None:
        """Test that requesting an invalid integer flag raises error."""
        with pytest.raises(BitwiseFlagUnknownError):
            Flags.get_single_flag(3)  # 3 is a combination of 1 and 2, not a valid single flag

    def test_get_single_flag_invalid_string(self) -> None:
        """Test that requesting an invalid string flag raises error."""
        with pytest.raises(BitwiseFlagUnknownError):
            Flags.get_single_flag("FLAG_D")

    def test_get_single_flag_invalid_type(self) -> None:
        """Test that requesting a flag with an invalid type raises error."""
        with pytest.raises(BitwiseFlagTypeError):
            Flags.get_single_flag(3.5)  # noqa - expecting wrong type

    def test_to_dict(self) -> None:
        """Test creating a dictionary from a bitwise flag."""
        result = Flags.to_dict()
        expected = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        assert result == expected


class TestBitwiseFlagEquality:
    def test_equality(self) -> None:
        """Test that the same bitwise flags are equal."""
        original = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        same = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        assert original == same

    def test_different_names(self) -> None:
        """Test that the bitwise flags with different names are not equal."""
        original = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        different = BitwiseFlag("different", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        assert original != different

    def test_different_values(self) -> None:
        """Test that the bitwise flags with different values are not equal."""
        original = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        different = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 8})
        assert original != different

    def test_different_keys(self) -> None:
        """Test that the bitwise flags with different keys are not equal."""
        original = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        different = BitwiseFlag("flags", {"FLAG_1": 1, "FLAG_2": 2, "FLAG_3": 4})
        assert original != different

    @pytest.mark.parametrize(
        "non_bw",
        [
            "hello",
            123,
            {"key": "value"},
            pl.DataFrame(),
        ],
        ids=["str", "int", "dict", "df"],
    )
    def test_different_object(self, non_bw: Any) -> None:
        """Test that comparing against a non-bitwise flag objects are not equal."""
        original = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        assert original != non_bw
