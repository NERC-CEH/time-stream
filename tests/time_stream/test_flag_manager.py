import copy
import unittest
from datetime import datetime
from unittest.mock import Mock

import polars as pl
from polars.testing import assert_series_equal

from time_stream import TimeSeries
from time_stream.bitwise import BitwiseFlag
from time_stream.exceptions import (
    BitwiseFlagUnknownError,
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


class TestAddFlag(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = TimeSeries(
            pl.DataFrame(
                {"time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)], "value": [1, 2, 3]}
            ),
            "time",
        ).with_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        self.ts.init_flag_column("value", "system1", "flag_col_1")

    def test_add_flag_to_flag_column_no_expr(self) -> None:
        """Test that adding a flag to a valid flag column with no expression sets all values in the
        column to that flag"""
        self.ts.add_flag("flag_col_1", 1)
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_to_flag_column_with_expr(self) -> None:
        """Test that adding a flag to a valid flag column with an expression sets all values in the
        column that match that expression to that flag"""
        self.ts.add_flag("flag_col_1", 1, pl.col("value").gt(1))
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_by_name_to_flag_column(self) -> None:
        """Test that adding a flag to a valid flag column using the flag name (rather than value)
        sets all values in the column that match that expression to the correct flag value"""
        self.ts.add_flag("flag_col_1", "FLAG_C")
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [4, 4, 4])

        assert_series_equal(result, expected)

    def test_adding_non_existent_flag_to_flag_column_raises_error(self) -> None:
        """Test that trying to add an invalid flag to a valid flag column raises error"""
        with self.assertRaises(BitwiseFlagUnknownError):
            self.ts.add_flag("flag_col_1", 10)

    def test_add_flag_to_data_column_raises_error(self) -> None:
        """Test that trying to add a flag to a data column raises error"""
        with self.assertRaises(ColumnNotFoundError):
            self.ts.add_flag("value", 1)

    def test_add_flag_twice(self) -> None:
        """Test that adding a flag twice uses the bitwise math, so doesn't actually add the value twice"""

        self.ts.add_flag("flag_col_1", 1)
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 1])
        assert_series_equal(result, expected)

        # adding a second time should leave it as is
        self.ts.add_flag("flag_col_1", 1)
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 1])
        assert_series_equal(result, expected)


class TestRemoveFlag(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = TimeSeries(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                    "value": [1, 2, 3],
                    "flag_col_1": [1, 3, 7],
                }
            ),
            "time",
        ).with_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        self.ts.register_flag_column("flag_col_1", "value", "system1")

    def test_remove_flag_from_flag_column_no_expr(self) -> None:
        """Test that removing a flag from a valid flag column with no expression works as expected"""
        self.ts.remove_flag("flag_col_1", 1)
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 2, 6])

        assert_series_equal(result, expected)

    def test_remove_flag_twice(self) -> None:
        """Test that removing a flag twice uses the bitwise math, so doesn't actually remove the value twice"""
        self.ts.remove_flag("flag_col_1", 1)
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 2, 6])
        assert_series_equal(result, expected)

        # removing a second time should leave it as is
        self.ts.remove_flag("flag_col_1", 1)
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 2, 6])
        assert_series_equal(result, expected)

    def test_remove_flag_from_flag_column_with_expr(self) -> None:
        """Test that removing a flag from a valid flag column with an expression removes flag from only those rows"""
        self.ts.remove_flag("flag_col_1", 1, pl.col("value").gt(1))
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 2, 6])

        assert_series_equal(result, expected)

    def test_remove_flag_by_name_from_flag_column(self) -> None:
        """Test that removing a flag from a valid flag column using the flag name (rather than value) works as
        expected"""
        self.ts.remove_flag("flag_col_1", "FLAG_B")
        result = self.ts.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 5])

        assert_series_equal(result, expected)

    def test_removing_non_existent_flag_from_flag_column_raises_error(self) -> None:
        """Test that trying to remove an invalid flag to a valid flag column raises error"""
        with self.assertRaises(BitwiseFlagUnknownError):
            self.ts.add_flag("flag_col_1", 10)

    def test_remove_flag_from_data_column_raises_error(self) -> None:
        """Test that trying to remove a flag to a data column raises error"""
        with self.assertRaises(ColumnNotFoundError):
            self.ts.add_flag("value", 1)


class TestEqualityFlagManager(unittest.TestCase):
    def test_equal(self) -> None:
        """Test that two identical flag manager objects are considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_original.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")
        flag_manager_original.register_flag_column(name="flag_col_2", base="base2", flag_system_name="system2")

        flag_manager_same = FlagManager()
        flag_manager_same.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_same.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_same.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")
        flag_manager_same.register_flag_column(name="flag_col_2", base="base2", flag_system_name="system2")

        self.assertEqual(flag_manager_original, flag_manager_same)

    def test_different_system_names(self) -> None:
        """Test that flag manager objects with different flag system names are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("different1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_system("different2", {"FLAG_1": 1, "FLAG_2": 2})

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertNotEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)

    def test_different_system_values(self) -> None:
        """Test that flag manager objects with different flag system values are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 8})
        flag_manager_different.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 4})

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertNotEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)

    def test_different_system_additional(self) -> None:
        """Test that flag manager objects with additional flag systems are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_different.register_flag_system("system3", {"FLAG_Z": 1, "FLAG_Y": 2})

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertNotEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)

    def test_different_column_name(self) -> None:
        """Test that flag manager objects with different flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_column(name="different1", base="base1", flag_system_name="system1")

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertNotEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)

    def test_different_column_base(self) -> None:
        """Test that flag manager objects with different flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_column(name="flag_col_1", base="different1", flag_system_name="system1")

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertNotEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)

    def test_different_column_system(self) -> None:
        """Test that flag manager objects with different flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_original.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_different.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system2")

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertNotEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)

    def test_different_column_additional(self) -> None:
        """Test that flag manager objects with additional flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_column(name="flag_col_1", base="base1", flag_system_name="system1")
        flag_manager_different.register_flag_column(name="flag_col_2", base="base1", flag_system_name="system1")

        self.assertNotEqual(flag_manager_original, flag_manager_different)

        # Check the expected property is the difference
        self.assertEqual(flag_manager_original.flag_systems, flag_manager_different.flag_systems)
        self.assertNotEqual(flag_manager_original.flag_columns, flag_manager_different.flag_columns)


class TestEqualityFlagColumn(unittest.TestCase):
    def setUp(self) -> None:
        self.flag_system1 = BitwiseFlag("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        self.flag_system2 = BitwiseFlag("system2", {"FLAG_1": 1, "FLAG_2": 2})
        self.flag_column_original = FlagColumn("flag_col_1", "base1", self.flag_system1)

    def test_equal(self) -> None:
        """Test that two identical flag column objects are considered equal."""
        flag_column_same = FlagColumn("flag_col_1", "base1", self.flag_system1)

        self.assertEqual(self.flag_column_original, flag_column_same)

    def test_different_name(self) -> None:
        """Test that two flag column objects with different names are not considered equal."""
        flag_column_different = FlagColumn("different1", "base1", self.flag_system1)

        self.assertNotEqual(self.flag_column_original, flag_column_different)

        # Check the expected property is the difference
        self.assertNotEqual(self.flag_column_original.name, flag_column_different.name)
        self.assertEqual(self.flag_column_original.base, flag_column_different.base)
        self.assertEqual(self.flag_column_original.flag_system, flag_column_different.flag_system)

    def test_different_base(self) -> None:
        """Test that two flag column objects with different base columns are not considered equal."""
        flag_column_different = FlagColumn("flag_col_1", "different1", self.flag_system1)

        self.assertNotEqual(self.flag_column_original, flag_column_different)

        # Check the expected property is the difference
        self.assertEqual(self.flag_column_original.name, flag_column_different.name)
        self.assertNotEqual(self.flag_column_original.base, flag_column_different.base)
        self.assertEqual(self.flag_column_original.flag_system, flag_column_different.flag_system)

    def test_different_system(self) -> None:
        """Test that two flag column objects with different flag systems are not considered equal."""
        flag_column_different = FlagColumn("flag_col_1", "base1", self.flag_system2)

        self.assertNotEqual(self.flag_column_original, flag_column_different)

        # Check the expected property is the difference
        self.assertEqual(self.flag_column_original.name, flag_column_different.name)
        self.assertEqual(self.flag_column_original.base, flag_column_different.base)
        self.assertNotEqual(self.flag_column_original.flag_system, flag_column_different.flag_system)
