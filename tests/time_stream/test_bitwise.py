import unittest

from time_series.bitwise import BitwiseFlag


class Flags(BitwiseFlag):
    FLAG_A = 1
    FLAG_B = 2
    FLAG_C = 4


class TestBitwiseFlag(unittest.TestCase):
    def test_valid_flags(self):
        """Test creating valid bitwise flags."""
        self.assertEqual(Flags.FLAG_A, 1)
        self.assertEqual(Flags.FLAG_B, 2)
        self.assertEqual(Flags.FLAG_C, 4)

    def test_invalid_flag_value_not_power_of_two(self):
        """Test that a non-power-of-two flag value raises error."""
        with self.assertRaises(ValueError):
            class _Flags(BitwiseFlag):
                INVALID_FLAG = 3

    def test_invalid_flag_value_negative(self):
        """Test that a negative flag value raises error."""
        with self.assertRaises(ValueError):
            class _Flags(BitwiseFlag):
                INVALID_FLAG = -2

    def test_flag_uniqueness(self):
        """Test that duplicate flag values raise ValueError."""
        with self.assertRaises(ValueError):
            class _Flags(BitwiseFlag):
                FLAG_A = 1
                FLAG_B = 2
                FLAG_C = 2

    def test_get_single_flag_by_integer(self):
        """Test retrieving a flag by integer value."""
        self.assertEqual(Flags.get_single_flag(1), Flags.FLAG_A)
        self.assertEqual(Flags.get_single_flag(2), Flags.FLAG_B)
        self.assertEqual(Flags.get_single_flag(4), Flags.FLAG_C)

    def test_get_single_flag_by_string(self):
        """Test retrieving a flag by string name."""
        self.assertEqual(Flags.get_single_flag("FLAG_A"), Flags.FLAG_A)
        self.assertEqual(Flags.get_single_flag("FLAG_B"), Flags.FLAG_B)
        self.assertEqual(Flags.get_single_flag("FLAG_C"), Flags.FLAG_C)

    def test_get_single_flag_invalid_integer(self):
        """Test that requesting an invalid integer flag raises error."""
        with self.assertRaises(KeyError):
            Flags.get_single_flag(3)  # 3 is a combination of 1 and 2, not a valid single flag

    def test_get_single_flag_invalid_string(self):
        """Test that requesting an invalid string flag raises error."""
        with self.assertRaises(KeyError):
            Flags.get_single_flag("FLAG_D")

    def test_get_single_flag_invalid_type(self):
        """Test that requesting a flag with an invalid type raises error."""
        with self.assertRaises(TypeError):
            Flags.get_single_flag(3.5)
