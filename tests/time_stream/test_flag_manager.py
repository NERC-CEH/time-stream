import copy
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from polars.testing import assert_series_equal

from time_stream import TimeFrame
from time_stream.bitwise import BitwiseFlag
from time_stream.exceptions import (
    BitwiseFlagUnknownError,
    ColumnNotFoundError,
    DuplicateFlagSystemError,
    FlagSystemError,
    FlagSystemNotFoundError,
    FlagSystemTypeError,
)
from time_stream.flag_manager import FlagColumn, FlagManager


class TestRegisterFlagSystem:
    def test_add_valid_dict_flag_system(self) -> None:
        """Test adding a new valid dict based flag system."""
        flag_manager = FlagManager()
        new_flag_system = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        flag_manager.register_flag_system("new_flags", new_flag_system)
        assert "new_flags" in flag_manager.flag_systems

    def test_add_valid_class_flag_system(self) -> None:
        """Test adding a new valid bitwise class based flag system."""
        flag_manager = FlagManager()
        new_flag_system = BitwiseFlag("new_flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager.register_flag_system("new_flags", new_flag_system)
        assert "new_flags" in flag_manager.flag_systems

    def test_add_duplicate_flag_system_raises_error(self) -> None:
        """Test adding a duplicate flag system raises error."""
        flag_manager = FlagManager()
        new_flag_system = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        flag_manager.register_flag_system("quality_control", new_flag_system)

        with pytest.raises(DuplicateFlagSystemError):
            flag_manager.register_flag_system("quality_control", new_flag_system)

    def test_add_valid_list_flag_system(self) -> None:
        """Test adding a flag system from a list of category names."""
        flag_manager = FlagManager()
        flag_manager.register_flag_system("new_flags", ["FLAG_A", "FLAG_B", "FLAG_C"])

        assert "new_flags" in flag_manager.flag_systems

        # check the bit values were correctly assigned
        flag_system = flag_manager.get_flag_system("new_flags")
        values = sorted(flag_system.to_dict().values())
        assert values == [1, 2, 4]

    def test_list_flag_system_is_sorted(self) -> None:
        """Test that unsorted list input produces the same flag system as sorted input."""
        fm1 = FlagManager()
        fm1.register_flag_system("flags", ["C", "A", "B"])
        fm2 = FlagManager()
        fm2.register_flag_system("flags", ["A", "B", "C"])
        assert fm1.get_flag_system("flags") == fm2.get_flag_system("flags")

    def test_empty_list_produces_default_flagged(self) -> None:
        """Test that an empty list produces a default FLAGGED flag at value 1."""
        flag_manager = FlagManager()
        flag_manager.register_flag_system("default_flags", [])
        flag_system = flag_manager.get_flag_system("default_flags")
        assert flag_system.to_dict() == {"FLAGGED": 1}

    def test_none_produces_default_flagged(self) -> None:
        """Test that None produces a default FLAGGED flag at value 1."""
        flag_manager = FlagManager()
        flag_manager.register_flag_system("default_flags")
        flag_system = flag_manager.get_flag_system("default_flags")
        assert flag_system.to_dict() == {"FLAGGED": 1}

    def test_duplicate_list_entries_raises_error(self) -> None:
        """Test that duplicate names in the list raise FlagSystemTypeError."""
        flag_manager = FlagManager()
        with pytest.raises(FlagSystemTypeError):
            flag_manager.register_flag_system("new_flags", ["FLAG_A", "FLAG_B", "FLAG_A"])


class TestRegisterFlagColumn:
    @staticmethod
    def setup_flag_manager() -> FlagManager:
        flag_manager = FlagManager()
        flag_manager.register_flag_system("quality_control", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        return flag_manager

    def test_flag_column_success(self) -> None:
        """Test registering a flag column with a valid flag system."""
        flag_manager = self.setup_flag_manager()
        flag_manager.register_flag_column("flag_column", "quality_control")
        assert "flag_column" in flag_manager.flag_columns

    def test_flag_column_invalid_flag_system_raises_error(self) -> None:
        """Test registering a flag column with a non-existent flag system raises error."""
        flag_manager = self.setup_flag_manager()

        with pytest.raises(FlagSystemNotFoundError):
            flag_manager.register_flag_column("flag_column", "bad_system")

    def test_flag_column_invalid_column_name_raises_error(self) -> None:
        """Test registering a flag column with a duplicate column name raises error."""
        # Register once
        flag_manager = self.setup_flag_manager()
        flag_manager.register_flag_column("flag_column", "quality_control")

        with pytest.raises(FlagSystemError):
            # Register twice - should raise error
            flag_manager.register_flag_column("flag_column", "quality_control")


class TestGetFlagSystem:
    @staticmethod
    def setup_flag_manager() -> FlagManager:
        flag_manager = FlagManager()
        flag_manager.register_flag_system("quality_control", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        return flag_manager

    def test_get_flag_system_success(self) -> None:
        flag_manager = self.setup_flag_manager()
        expected = BitwiseFlag("quality_control", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        result = flag_manager.get_flag_system("quality_control")

        assert result == expected

    def test_get_flag_system_error(self) -> None:
        """Test that requesting an invalid system raises error"""
        flag_manager = self.setup_flag_manager()

        with pytest.raises(FlagSystemNotFoundError):
            flag_manager.get_flag_system("bad_system")


class TestGetFlagColumn:
    @staticmethod
    def setup_flag_manager() -> tuple[FlagManager, Mock]:
        flag_manager = FlagManager()
        mock_flag_column = Mock(FlagColumn)
        flag_manager._flag_columns = {"flag_column": mock_flag_column}

        return flag_manager, mock_flag_column

    def test_get_flag_column_success(self) -> None:
        flag_manager, mock_flag_column = self.setup_flag_manager()

        result = flag_manager.get_flag_column("flag_column")
        assert result == mock_flag_column

    def test_get_flag_column_error(self) -> None:
        flag_manager, mock_flag_column = self.setup_flag_manager()

        with pytest.raises(ColumnNotFoundError):
            flag_manager.get_flag_column("no_column")


class TestCopy:
    @staticmethod
    def setup_flag_manager() -> FlagManager:
        flag_manager = FlagManager()
        flag_manager.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager.register_flag_system("system2", BitwiseFlag("system2", {"FLAG_1": 1, "FLAG_2": 2}))
        flag_manager.register_flag_column(name="flag_col_1", flag_system_name="system1")
        flag_manager.register_flag_column(name="flag_col_2", flag_system_name="system2")
        return flag_manager

    def assert_copy(self, flag_manager_copy: FlagManager) -> None:
        flag_manager = self.setup_flag_manager()

        # Check we have a different object but with the same contents
        assert flag_manager_copy == flag_manager
        assert flag_manager_copy is not flag_manager

        for name, system in flag_manager.flag_systems.items():
            copy_system = flag_manager_copy.flag_systems[name]

            # Systems should be a different object
            assert system is not copy_system

            # But should have the same data
            assert system == copy_system

        for name, flag_column in flag_manager.flag_columns.items():
            copy_flag_column = flag_manager_copy.flag_columns[name]
            assert copy_flag_column.name == flag_column.name

        # Test that the copy created is independent of the original
        # Change the original
        flag_manager.register_flag_system("new_system", {"A": 1})
        flag_manager.register_flag_column(name="new_flags", flag_system_name="new_system")

        # Copy should not see the new entries
        assert "new_sys" not in flag_manager_copy.flag_systems
        assert "new_flags" not in flag_manager_copy.flag_columns

    def test_copy(self) -> None:
        """Test that the copy method copies the full structure of the flag manager object."""
        flag_manager = self.setup_flag_manager()
        flag_manager_copy = flag_manager.copy()

        self.assert_copy(flag_manager_copy)

    def test_standard_lib_copy(self) -> None:
        """Test that the standard library copy module works as expected."""
        flag_manager = self.setup_flag_manager()

        flag_manager_copy = copy.copy(flag_manager)
        self.assert_copy(flag_manager_copy)


class TestAddFlag:
    @staticmethod
    def setup_tf() -> TimeFrame:
        tf = TimeFrame(
            pl.DataFrame(
                {"time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)], "value": [1, 2, 3]}
            ),
            "time",
        ).with_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        tf.init_flag_column("system1", "flag_col_1")
        return tf

    def test_add_flag_to_flag_column_no_expr(self) -> None:
        """Test that adding a flag to a valid flag column with no expression sets all values in the
        column to that flag"""
        tf = self.setup_tf()

        tf.add_flag("flag_col_1", 1)
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_to_flag_column_with_expr(self) -> None:
        """Test that adding a flag to a valid flag column with an expression sets all values in the
        column that match that expression to that flag"""
        tf = self.setup_tf()

        tf.add_flag("flag_col_1", 1, pl.col("value").gt(1))
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_by_name_to_flag_column(self) -> None:
        """Test that adding a flag to a valid flag column using the flag name (rather than value)
        sets all values in the column that match that expression to the correct flag value"""
        tf = self.setup_tf()

        tf.add_flag("flag_col_1", "FLAG_C")
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [4, 4, 4])

        assert_series_equal(result, expected)

    def test_adding_non_existent_flag_to_flag_column_raises_error(self) -> None:
        """Test that trying to add an invalid flag to a valid flag column raises error"""
        tf = self.setup_tf()

        with pytest.raises(BitwiseFlagUnknownError):
            tf.add_flag("flag_col_1", 10)

    def test_add_flag_to_data_column_raises_error(self) -> None:
        """Test that trying to add a flag to a data column raises error"""
        tf = self.setup_tf()

        with pytest.raises(ColumnNotFoundError):
            tf.add_flag("value", 1)

    def test_add_flag_twice(self) -> None:
        """Test that adding a flag twice uses the bitwise math, so doesn't actually add the value twice"""
        tf = self.setup_tf()

        tf.add_flag("flag_col_1", 1)

        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 1])

        assert_series_equal(result, expected)

        # adding a second time should leave it as is
        tf.add_flag("flag_col_1", 1)

        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_on_decoded_column(self) -> None:
        """Test that add_flag on a decoded column leaves the column in List(String) with the flag added."""
        tf = self.setup_tf()
        tf_decoded = tf.decode_flag_column("flag_col_1")

        tf_decoded.add_flag("flag_col_1", "FLAG_A")

        result = tf_decoded.df["flag_col_1"]
        assert result.dtype == pl.List(pl.String)
        assert result.to_list() == [["FLAG_A"], ["FLAG_A"], ["FLAG_A"]]


class TestRemoveFlag:
    @staticmethod
    def setup_tf() -> TimeFrame:
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                    "value": [1, 2, 3],
                    "flag_col_1": [1, 3, 7],
                }
            ),
            "time",
        ).with_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        tf.register_flag_column("flag_col_1", "system1")
        return tf

    def test_remove_flag_from_flag_column_no_expr(self) -> None:
        """Test that removing a flag from a valid flag column with no expression works as expected"""
        tf = self.setup_tf()

        tf.remove_flag("flag_col_1", 1)
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 2, 6])

        assert_series_equal(result, expected)

    def test_remove_flag_twice(self) -> None:
        """Test that removing a flag twice uses the bitwise math, so doesn't actually remove the value twice"""
        tf = self.setup_tf()

        tf.remove_flag("flag_col_1", 1)
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 2, 6])
        assert_series_equal(result, expected)

        # removing a second time should leave it as is
        tf.remove_flag("flag_col_1", 1)
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [0, 2, 6])
        assert_series_equal(result, expected)

    def test_remove_flag_from_flag_column_with_expr(self) -> None:
        """Test that removing a flag from a valid flag column with an expression removes flag from only those rows"""
        tf = self.setup_tf()

        tf.remove_flag("flag_col_1", 1, pl.col("value").gt(1))
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 2, 6])

        assert_series_equal(result, expected)

    def test_remove_flag_by_name_from_flag_column(self) -> None:
        """
        Test that removing a flag from a valid flag column using the flag name (rather than value) works as
        expected.
        """
        tf = self.setup_tf()

        tf.remove_flag("flag_col_1", "FLAG_B")
        result = tf.df["flag_col_1"]
        expected = pl.Series("flag_col_1", [1, 1, 5])

        assert_series_equal(result, expected)

    def test_removing_non_existent_flag_from_flag_column_raises_error(self) -> None:
        """Test that trying to remove an invalid flag to a valid flag column raises error"""
        tf = self.setup_tf()

        with pytest.raises(BitwiseFlagUnknownError):
            tf.add_flag("flag_col_1", 10)

    def test_remove_flag_from_data_column_raises_error(self) -> None:
        """Test that trying to remove a flag to a data column raises error"""
        tf = self.setup_tf()

        with pytest.raises(ColumnNotFoundError):
            tf.add_flag("value", 1)

    def test_remove_flag_on_decoded_column(self) -> None:
        """Test that remove_flag on a decoded column leaves the column in List(String) with the flag removed."""
        tf = self.setup_tf()
        tf_decoded = tf.decode_flag_column("flag_col_1")

        tf_decoded.remove_flag("flag_col_1", "FLAG_A")

        result = tf_decoded.df["flag_col_1"]
        assert result.dtype == pl.List(pl.String)
        assert result.to_list() == [[], ["FLAG_B"], ["FLAG_B", "FLAG_C"]]


class TestEqualityFlagManager:
    def test_equal(self) -> None:
        """Test that two identical flag manager objects are considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_original.register_flag_column(name="flag_col_1", flag_system_name="system1")
        flag_manager_original.register_flag_column(name="flag_col_2", flag_system_name="system2")

        flag_manager_same = FlagManager()
        flag_manager_same.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_same.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_same.register_flag_column(name="flag_col_1", flag_system_name="system1")
        flag_manager_same.register_flag_column(name="flag_col_2", flag_system_name="system2")

        assert flag_manager_original == flag_manager_same

    def test_different_system_names(self) -> None:
        """Test that flag manager objects with different flag system names are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("different1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_system("different2", {"FLAG_1": 1, "FLAG_2": 2})

        assert flag_manager_original != flag_manager_different

        # Check the expected property is the difference
        assert flag_manager_original.flag_systems != flag_manager_different.flag_systems
        assert flag_manager_original.flag_columns == flag_manager_different.flag_columns

    def test_different_system_values(self) -> None:
        """Test that flag manager objects with different flag system values are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 8})
        flag_manager_different.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 4})

        assert flag_manager_original != flag_manager_different

        # Check the expected property is the difference
        assert flag_manager_original.flag_systems != flag_manager_different.flag_systems
        assert flag_manager_original.flag_columns == flag_manager_different.flag_columns

    def test_different_system_additional(self) -> None:
        """Test that flag manager objects with additional flag systems are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_different.register_flag_system("system3", {"FLAG_Z": 1, "FLAG_Y": 2})

        assert flag_manager_original != flag_manager_different

        # Check the expected property is the difference
        assert flag_manager_original.flag_systems != flag_manager_different.flag_systems
        assert flag_manager_original.flag_columns == flag_manager_different.flag_columns

    def test_different_column_name(self) -> None:
        """Test that flag manager objects with different flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_column(name="flag_col_1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_column(name="different1", flag_system_name="system1")

        assert flag_manager_original != flag_manager_different

        # Check the expected property is the difference
        assert flag_manager_original.flag_systems == flag_manager_different.flag_systems
        assert flag_manager_original.flag_columns != flag_manager_different.flag_columns

    def test_different_column_system(self) -> None:
        """Test that flag manager objects with different flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_original.register_flag_column(name="flag_col_1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_manager_different.register_flag_column(name="flag_col_1", flag_system_name="system2")

        assert flag_manager_original != flag_manager_different

        # Check the expected property is the difference
        assert flag_manager_original.flag_systems == flag_manager_different.flag_systems
        assert flag_manager_original.flag_columns != flag_manager_different.flag_columns

    def test_different_column_additional(self) -> None:
        """Test that flag manager objects with additional flag columns are not considered equal."""
        flag_manager_original = FlagManager()
        flag_manager_original.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_original.register_flag_column(name="flag_col_1", flag_system_name="system1")

        flag_manager_different = FlagManager()
        flag_manager_different.register_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_manager_different.register_flag_column(name="flag_col_1", flag_system_name="system1")
        flag_manager_different.register_flag_column(name="flag_col_2", flag_system_name="system1")

        assert flag_manager_original != flag_manager_different

        # Check the expected property is the difference
        assert flag_manager_original.flag_systems == flag_manager_different.flag_systems
        assert flag_manager_original.flag_columns != flag_manager_different.flag_columns


class TestEqualityFlagColumn:
    @staticmethod
    def setup_flag_systems() -> tuple[BitwiseFlag, BitwiseFlag, FlagColumn]:
        flag_system1 = BitwiseFlag("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        flag_system2 = BitwiseFlag("system2", {"FLAG_1": 1, "FLAG_2": 2})
        flag_column_original = FlagColumn("flag_col_1", flag_system1)

        return flag_system1, flag_system2, flag_column_original

    def test_equal(self) -> None:
        """Test that two identical flag column objects are considered equal."""
        flag_system1, flag_system2, flag_column_original = self.setup_flag_systems()
        flag_column_same = FlagColumn("flag_col_1", flag_system1)

        assert flag_column_original == flag_column_same

    def test_different_name(self) -> None:
        """Test that two flag column objects with different names are not considered equal."""
        flag_system1, flag_system2, flag_column_original = self.setup_flag_systems()

        flag_column_different = FlagColumn("different1", flag_system1)

        assert flag_column_original != flag_column_different

        # Check the expected property is the difference
        assert flag_column_original.name != flag_column_different.name
        assert flag_column_original.flag_system == flag_column_different.flag_system

    def test_different_system(self) -> None:
        """Test that two flag column objects with different flag systems are not considered equal."""
        flag_system1, flag_system2, flag_column_original = self.setup_flag_systems()

        flag_column_different = FlagColumn("flag_col_1", flag_system2)

        assert flag_column_original != flag_column_different

        # Check the expected property is the difference
        assert flag_column_original.name == flag_column_different.name
        assert flag_column_original.flag_system != flag_column_different.flag_system

    @pytest.mark.parametrize(
        "non_fs",
        [
            "hello",
            123,
            {"key": "value"},
            pl.DataFrame(),
        ],
        ids=["str", "int", "dict", "df"],
    )
    def test_different_object(self, non_fs: Any) -> None:
        """Test that comparing against non-flag system objects are not equal."""
        flag_system1, flag_system2, flag_column_original = self.setup_flag_systems()

        assert flag_column_original != non_fs


class TestDecodeFlagColumn:
    @staticmethod
    def setup_tf() -> TimeFrame:
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3), datetime(2025, 1, 4)],
                    "value": [1, 2, 3, 4],
                    "flag_col_1": [0, 1, 5, 7],
                }
            ),
            "time",
        ).with_flag_system("system1", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        tf.register_flag_column("flag_col_1", "system1")
        return tf

    def test_output_dtype_is_list_string(self) -> None:
        """Test that the decoded column has dtype List(String)."""
        tf = self.setup_tf()
        tf_decoded = tf.decode_flag_column("flag_col_1")
        assert tf_decoded.df["flag_col_1"].dtype == pl.List(pl.String)

    def test_same_column_name_after_decode(self) -> None:
        """Test that the column name is preserved after decoding."""
        tf = self.setup_tf()
        tf_decoded = tf.decode_flag_column("flag_col_1")
        assert "flag_col_1" in tf_decoded.df.columns

    @pytest.mark.parametrize(
        "row, expected",
        [
            (0, []),
            (1, ["FLAG_A"]),
            (2, ["FLAG_A", "FLAG_C"]),
            (3, ["FLAG_A", "FLAG_B", "FLAG_C"]),
        ],
        ids=["zero", "single_flag", "two_flags", "all_flags_ascending_order"],
    )
    def test_integer_decodes_to_flag_names(self, row: int, expected: list[str]) -> None:
        """Test that integer values decode to the correct flag name lists in ascending bit-value order."""
        tf = self.setup_tf()
        tf_decoded = tf.decode_flag_column("flag_col_1")
        assert tf_decoded.df["flag_col_1"].to_list()[row] == expected
        assert tf_decoded.get_flag_column("flag_col_1").is_decoded is True

    def test_returns_new_timeframe(self) -> None:
        """Test that decode_flag_column returns a new TimeFrame (original unchanged)."""
        tf = self.setup_tf()
        tf_decoded = tf.decode_flag_column("flag_col_1")
        assert tf.get_flag_column("flag_col_1").is_decoded is False
        assert tf_decoded is not tf

    def test_unregistered_column_raises_column_not_found_error(self) -> None:
        """Test that decoding an unregistered column raises ColumnNotFoundError."""
        tf = self.setup_tf()
        with pytest.raises(ColumnNotFoundError):
            tf.decode_flag_column("value")


class TestEncodeFlagColumn:
    @staticmethod
    def setup_tf() -> TimeFrame:
        tf = TestDecodeFlagColumn.setup_tf()
        return tf.decode_flag_column("flag_col_1")

    def test_output_dtype_is_int64(self) -> None:
        """Test that the encoded column has dtype Int64."""
        tf_decoded = self.setup_tf()
        tf_encoded = tf_decoded.encode_flag_column("flag_col_1")
        assert tf_encoded.df["flag_col_1"].dtype == pl.Int64

    def test_same_column_name_after_encode(self) -> None:
        """Test that the column name is preserved after encoding."""
        tf_decoded = self.setup_tf()
        tf_encoded = tf_decoded.encode_flag_column("flag_col_1")
        assert "flag_col_1" in tf_encoded.df.columns

    @pytest.mark.parametrize(
        "row, expected",
        [
            (0, 0),
            (1, 1),
            (2, 5),
            (3, 7),
        ],
        ids=["empty_list", "single_flag", "two_flags", "all_flags"],
    )
    def test_flag_names_encode_to_integer(self, row: int, expected: int) -> None:
        """Test that flag name lists encode to the correct bitwise integer values."""
        tf_decoded = self.setup_tf()
        tf_encoded = tf_decoded.encode_flag_column("flag_col_1")
        assert tf_encoded.df["flag_col_1"].to_list()[row] == expected
        assert tf_encoded.get_flag_column("flag_col_1").is_decoded is False

    def test_returns_new_timeframe(self) -> None:
        """Test that encode_flag_column returns a new TimeFrame (original unchanged)."""
        tf_decoded = self.setup_tf()
        tf_encoded = tf_decoded.encode_flag_column("flag_col_1")
        assert tf_decoded.get_flag_column("flag_col_1").is_decoded is True
        assert tf_encoded is not tf_decoded

    def test_column_remains_registered(self) -> None:
        """Test that the column is still registered as a flag column after encoding."""
        tf_decoded = self.setup_tf()
        tf_encoded = tf_decoded.encode_flag_column("flag_col_1")
        assert tf_encoded.get_flag_column("flag_col_1") is not None

    def test_unregistered_column_raises_column_not_found_error(self) -> None:
        """Test that encoding an unregistered column raises ColumnNotFoundError."""
        tf_decoded = self.setup_tf()
        with pytest.raises(ColumnNotFoundError):
            tf_decoded.encode_flag_column("value")

    def test_unknown_flag_name_raises_error(self) -> None:
        """Test that encoding a column containing an unknown flag name raises BitwiseFlagUnknownError."""
        tf = TestDecodeFlagColumn.setup_tf()
        df_with_unknown = tf.df.with_columns(pl.Series("flag_col_1", [["FLAG_A", "UNKNOWN"], [], [], []]))
        tf_manual = tf.with_df(df_with_unknown)
        tf_manual._flag_manager.flag_columns["flag_col_1"].is_decoded = True
        with pytest.raises(BitwiseFlagUnknownError):
            tf_manual.encode_flag_column("flag_col_1")

    def test_decode_then_encode_produces_original_values(self) -> None:
        """Test that decode followed by encode produces the original integer values."""
        tf = TestDecodeFlagColumn.setup_tf()
        original = tf.df["flag_col_1"].clone()
        tf_roundtrip = tf.decode_flag_column("flag_col_1").encode_flag_column("flag_col_1")
        assert_series_equal(tf_roundtrip.df["flag_col_1"], original)
