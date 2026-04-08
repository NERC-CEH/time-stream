import copy
from datetime import datetime
from unittest.mock import Mock

import polars as pl
import pytest

from time_stream import TimeFrame
from time_stream.exceptions import (
    BitwiseFlagUnknownError,
    CategoricalFlagUnknownError,
    ColumnNotFoundError,
    DuplicateFlagSystemError,
    FlagSystemError,
    FlagSystemNotFoundError,
    FlagSystemTypeError,
)
from time_stream.flags.bitwise_flag_system import BitwiseFlag
from time_stream.flags.flag_manager import BitwiseFlagColumn, CategoricalFlagColumn, FlagManager


class TestRegisterFlagSystem:
    def test_add_valid_dict_flag_system(self) -> None:
        """Test adding a new valid dict based flag system."""
        flag_manager = FlagManager()
        new_flag_system = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        flag_manager.register_flag_system("new_flags", new_flag_system)
        assert "new_flags" in flag_manager.flag_systems

    def test_add_valid_dict_int_categorical_flag_system(self) -> None:
        """Test adding a categorical flag system from a dict[str, int]."""
        flag_manager = FlagManager()
        flag_manager.register_flag_system("new_flags", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")
        assert "new_flags" in flag_manager.flag_systems
        assert flag_manager.flag_systems["new_flags"].to_dict() == {"FLAG_A": 0, "FLAG_B": 1}

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

    def test_list_categorical_produces_sequential_integers(self) -> None:
        """Test that a list with flag_type='categorical' assigns sequential integers in sorted order."""
        flag_manager = FlagManager()
        flag_manager.register_flag_system("new_flags", ["FLAG_A", "FLAG_B", "FLAG_C"], flag_type="categorical")
        flag_system = flag_manager.get_flag_system("new_flags")
        assert flag_system.to_dict() == {"FLAG_A": 0, "FLAG_B": 1, "FLAG_C": 2}

    def test_dict_int_categorical_flag_system(self) -> None:
        """Test that a dict[str, int] with flag_type='categorical' creates a categorical system."""
        flag_manager = FlagManager()
        flag_manager.register_flag_system("new_flags", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")
        flag_system = flag_manager.get_flag_system("new_flags")
        assert flag_system.to_dict() == {"FLAG_A": 0, "FLAG_B": 1}

    def test_unsupported_type_raises(self) -> None:
        """Test that passing an unsupported type raises FlagSystemTypeError."""
        flag_manager = FlagManager()
        with pytest.raises(FlagSystemTypeError):
            flag_manager.register_flag_system("bad", 12345)  # type: ignore[arg-type]


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

    def test_categorical_flag_column_scalar(self) -> None:
        """Test registering a categorical flag column in scalar mode."""
        fm = FlagManager()
        fm.register_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")
        fm.register_flag_column("flag_col", "qc", list_mode=False)
        col = fm.get_flag_column("flag_col")
        assert isinstance(col, CategoricalFlagColumn)
        assert col.list_mode is False

    def test_categorical_flag_column_list(self) -> None:
        """Test registering a categorical flag column in list mode."""
        fm = FlagManager()
        fm.register_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")
        fm.register_flag_column("flag_col", "qc", list_mode=True)
        col = fm.get_flag_column("flag_col")
        assert isinstance(col, CategoricalFlagColumn)
        assert col.list_mode is True


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
        mock_flag_column = Mock(BitwiseFlagColumn)
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
        flag_manager.register_flag_system("system2", {"FLAG_1": 1, "FLAG_2": 2})
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


class TestRegisterFlagColumnValidation:
    def test_categorical_invalid_values_raise_error(self) -> None:
        """Test that registering a categorical flag column with invalid values raises an error."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                    "qc_flags": [0, 99],
                }
            ),
            "time",
        ).with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")

        with pytest.raises(CategoricalFlagUnknownError):
            tf.register_flag_column("qc_flags", "qc")

    def test_categorical_valid_values_succeed(self) -> None:
        """Test that registering a categorical flag column with only valid values succeeds."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                    "qc_flags": [0, 1],
                }
            ),
            "time",
        ).with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")

        tf.register_flag_column("qc_flags", "qc")
        assert "qc_flags" in tf.flag_columns

    def test_categorical_null_values_are_allowed(self) -> None:
        """Test that null values in a categorical column are accepted."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                    "qc_flags": pl.Series([0, None], dtype=pl.Int32),
                }
            ),
            "time",
        ).with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")

        tf.register_flag_column("qc_flags", "qc")
        assert "qc_flags" in tf.flag_columns

    def test_bitwise_valid_combinations_succeed(self) -> None:
        """Test that a bitwise column containing valid flag combinations is accepted."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                    "flags": [0, 3, 7],
                }
            ),
            "time",
        ).with_flag_system("qc", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        tf.register_flag_column("flags", "qc")
        assert "flags" in tf.flag_columns

    def test_bitwise_null_values_are_allowed(self) -> None:
        """Test that null values in a bitwise column are accepted."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                    "flags": pl.Series([1, None], dtype=pl.Int32),
                }
            ),
            "time",
        ).with_flag_system("qc", {"FLAG_A": 1, "FLAG_B": 2})

        tf.register_flag_column("flags", "qc")
        assert "flags" in tf.flag_columns

    def test_bitwise_invalid_values_raise_error(self) -> None:
        """Test that a bitwise column with bits outside the known mask raises BitwiseFlagUnknownError."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "time": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                    "flags": [1, 8],  # 8 has no corresponding flag
                }
            ),
            "time",
        ).with_flag_system("qc", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})

        with pytest.raises(BitwiseFlagUnknownError):
            tf.register_flag_column("flags", "qc")


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
