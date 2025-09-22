import copy
import unittest
from unittest.mock import Mock

from time_stream.bitwise import BitwiseFlag
from time_stream.exceptions import (
    ColumnNotFoundError,
    DuplicateFlagSystemError,
    FlagSystemError,
    FlagSystemNotFoundError,
)
from time_stream.flag_manager import FlagColumn, FlagManager


class TestRegisterFlagSystem(unittest.TestCase):
    def test_add_valid_dict_flag_system(self) -> None:
        """Test adding a new valid dict based flag system."""
        flag_manager = FlagManager()
        new_flag_system = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        flag_manager.register_flag_system("new_flags", new_flag_system)
        self.assertIn("new_flags", flag_manager.flag_systems)

    def test_add_valid_class_flag_system(self) -> None:
        """Test adding a new valid bitwise class based flag system."""
        flag_manager = FlagManager()
        new_flag_system = BitwiseFlag("new_flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager.register_flag_system("new_flags", new_flag_system)
        self.assertIn("new_flags", flag_manager.flag_systems)

    def test_add_duplicate_flag_system_raises_error(self) -> None:
        """Test adding a duplicate flag system raises error."""
        flag_manager = FlagManager()
        new_flag_system = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        flag_manager.register_flag_system("quality_control", new_flag_system)
        with self.assertRaises(DuplicateFlagSystemError):
            flag_manager.register_flag_system("quality_control", new_flag_system)


class TestRegisterFlagColumn(unittest.TestCase):
    def setUp(self) -> None:
        self.flag_manager = FlagManager()
        self.flag_manager.register_flag_system("quality_control", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

    def test_flag_column_success(self) -> None:
        """Test registering a flag column with a valid flag system."""
        self.flag_manager.register_flag_column("flag_column", "base_column", "quality_control")
        self.assertIn("flag_column", self.flag_manager.flag_columns)

    def test_flag_column_invalid_flag_system_raises_error(self) -> None:
        """Test registering a flag column with a non-existent flag system raises error."""
        with self.assertRaises(FlagSystemNotFoundError):
            self.flag_manager.register_flag_column("flag_column", "base_column", "bad_system")

    def test_flag_column_invalid_column_name_raises_error(self) -> None:
        """Test registering a flag column with a duplicate column name raises error."""
        # Register once
        self.flag_manager.register_flag_column("flag_column", "base_column", "quality_control")
        with self.assertRaises(FlagSystemError):
            # Register twice - should raise error
            self.flag_manager.register_flag_column("flag_column", "base_column", "quality_control")


class TestGetFlagSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.flag_manager = FlagManager()
        self.flag_manager.register_flag_system("quality_control", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

    def test_get_flag_system_success(self) -> None:
        expected = BitwiseFlag("quality_control", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        result = self.flag_manager.get_flag_system("quality_control")
        self.assertEqual(result, expected)

    def test_get_flag_system_error(self) -> None:
        """Test that requesting an invalid system raises error"""
        with self.assertRaises(FlagSystemNotFoundError):
            self.flag_manager.get_flag_system("bad_system")


class TestGetFlagColumn(unittest.TestCase):
    def setUp(self) -> None:
        self.flag_manager = FlagManager()
        self.mock_flag_column = Mock(FlagColumn)
        self.flag_manager._flag_columns = {"flag_column": self.mock_flag_column}

    def test_get_flag_column_success(self) -> None:
        result = self.flag_manager.get_flag_column("flag_column")
        self.assertEqual(result, self.mock_flag_column)

    def test_get_flag_column_error(self) -> None:
        with self.assertRaises(ColumnNotFoundError):
            self.flag_manager.get_flag_column("no_column")


class TestCopy(unittest.TestCase):
    def setUp(self) -> None:
        self.flag_manager = FlagManager()
        self.flag_manager.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        self.flag_manager.register_flag_system("system2", BitwiseFlag("system2", {"FLAG_1": 1, "FLAG_2": 2}))
        self.flag_manager.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")
        self.flag_manager.register_flag_column(name="flag_col_2", base="base2", flag_system_name="system2")

    def assert_copy(self, flag_manager_copy: FlagManager) -> None:
        # Check we have a different object
        self.assertIsNot(flag_manager_copy, self.flag_manager)

        for name, system in self.flag_manager.flag_systems.items():
            copy_system = flag_manager_copy.flag_systems[name]
            # Systems should be a different object
            self.assertIsNot(system, copy_system)
            # But should have the same data
            self.assertEqual(system, copy_system)

        for name, flag_column in self.flag_manager.flag_columns.items():
            copy_flag_column = flag_manager_copy.flag_columns[name]
            self.assertEqual(copy_flag_column.name, flag_column.name)
            self.assertEqual(copy_flag_column.base, flag_column.base)
            self.assertEqual(copy_flag_column.flag_system_name, flag_column.flag_system_name)

        # Test that the copy created is independent of the original
        # Change the original
        self.flag_manager.register_flag_system("new_system", {"A": 1})
        self.flag_manager.register_flag_column(name="new_flags", base="base3", flag_system_name="new_system")

        # Copy should not see the new entries
        self.assertNotIn("new_sys", flag_manager_copy.flag_systems)
        self.assertNotIn("new_flags", flag_manager_copy.flag_columns)

    def test_copy(self) -> None:
        """Test that the copy method copies the full structure of the flag manager object."""
        flag_manager_copy = self.flag_manager.copy()
        self.assert_copy(flag_manager_copy)

    def test_standard_lib_copy(self) -> None:
        """Test that the standard library copy module works as expected."""
        flag_manager_copy = copy.copy(self.flag_manager)
        self.assert_copy(flag_manager_copy)
