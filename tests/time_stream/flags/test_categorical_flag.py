"""Tests for the CategoricalFlag system and its integration with TimeFrame."""

from datetime import datetime
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_series_equal

from time_stream import TimeFrame
from time_stream.exceptions import (
    CategoricalFlagTypeError,
    CategoricalFlagUnknownError,
    CategoricalFlagValueError,
    ColumnTypeError,
)
from time_stream.flags.bitwise_flag_system import BitwiseFlag
from time_stream.flags.categorical_flag_system import CategoricalFlag
from time_stream.flags.flag_manager import CategoricalFlagColumn


class FlagsInt(CategoricalFlag):
    FLAG_A = 97
    FLAG_B = 98
    FLAG_C = 99


class FlagsStr(CategoricalFlag):
    FLAG_A = "A"
    FLAG_B = "B"
    FLAG_C = "C"


class TestCategoricalFlag:
    def test_valid_int_flags(self) -> None:
        assert FlagsInt.FLAG_A.value == 97
        assert FlagsInt.FLAG_B.value == 98
        assert FlagsInt.FLAG_C.value == 99

    def test_valid_str_flags(self) -> None:
        assert FlagsStr.FLAG_A.value == "A"
        assert FlagsStr.FLAG_B.value == "B"
        assert FlagsStr.FLAG_C.value == "C"

    def test_mixed_value_types_raises(self) -> None:
        """Test that mixing int and str values raises CategoricalFlagTypeError."""
        with pytest.raises(CategoricalFlagTypeError):
            CategoricalFlag._check_value_types({"FLAG_A": 0, "FLAG_B": "B"})

    @pytest.mark.parametrize("value", [1.23, [1], {"a": "B"}])
    def test_invalid_flag_value_not_int_or_str(self, value: Any) -> None:
        """Test that a non-int or str flag raises error."""
        with pytest.raises(CategoricalFlagTypeError):

            class _Flags(CategoricalFlag):  # noqa - expect class unused warning
                INVALID_FLAG = value

    def test_value_type_int(self) -> None:
        """Test that value_type() returns int for int-valued systems."""
        assert FlagsInt.value_type() is int

    def test_value_type_str(self) -> None:
        """Test that value_type() returns str for str-valued systems."""
        assert FlagsStr.value_type() is str

    def test_duplicate_values_raises(self) -> None:
        """Test that duplicate values raise CategoricalFlagValueError."""
        with pytest.raises(CategoricalFlagValueError):
            CategoricalFlag._check_unique_values("dup", {"FLAG_A": 0, "ALSO_FLAG_A": 0, "FLAG_B": 1})


class TestCategoricalFlagValidateColumn:
    def test_valid_int_values_pass(self) -> None:
        """Test that a series containing only valid int flag values passes."""
        FlagsInt.validate_column(pl.Series("flags", [97, 98, 99]))

    def test_valid_str_values_pass(self) -> None:
        """Test that a series containing only valid str flag values passes."""
        FlagsStr.validate_column(pl.Series("flags", ["A", "B", "C"]))

    def test_nulls_are_allowed(self) -> None:
        """Test that null values do not raise an error."""
        FlagsInt.validate_column(pl.Series("flags", [97, None], dtype=pl.Int32))
        FlagsStr.validate_column(pl.Series("flags", ["A", None], dtype=pl.Utf8))

    def test_empty_series_passes(self) -> None:
        """Test that an empty series does not raise an error."""
        FlagsInt.validate_column(pl.Series("flags", [], dtype=pl.Int32))
        FlagsStr.validate_column(pl.Series("flags", [], dtype=pl.Utf8))

    def test_invalid_int_raises(self) -> None:
        """Test that an unknown int value raises CategoricalFlagUnknownError."""
        with pytest.raises(CategoricalFlagUnknownError):
            FlagsInt.validate_column(pl.Series("flags", [0, 99]))

    def test_invalid_str_raises(self) -> None:
        """Test that an unknown str value raises CategoricalFlagUnknownError."""
        with pytest.raises(CategoricalFlagUnknownError):
            FlagsStr.validate_column(pl.Series("flags", ["A", "UNKNOWN"]))

    def test_list_mode_valid_values_pass(self) -> None:
        """Test that a list series with only valid values passes."""
        FlagsInt.validate_column(pl.Series("flags", [[98, 99], [97], []]), list_mode=True)

    def test_list_mode_invalid_value_raises(self) -> None:
        """Test that a list series containing an unknown value raises CategoricalFlagUnknownError."""
        with pytest.raises(CategoricalFlagUnknownError):
            FlagsInt.validate_column(pl.Series("flags", [[0, 99], [1]]), list_mode=True)


class TestCategoricalFlagEquality:
    def test_equality(self) -> None:
        """Test that two CategoricalFlag classes with the same name and mapping are equal."""
        original = CategoricalFlag("flags", {"FLAG_A": 0, "FLAG_B": 1})
        same = CategoricalFlag("flags", {"FLAG_A": 0, "FLAG_B": 1})
        assert original == same

    def test_inequality_different_name(self) -> None:
        """Test that different names produce unequal classes."""
        original = CategoricalFlag("flags", {"FLAG_A": 0, "FLAG_B": 1})
        different = CategoricalFlag("other", {"FLAG_A": 0, "FLAG_B": 1})
        assert original != different

    def test_inequality_different_values(self) -> None:
        """Test that different values produce unequal classes."""
        original = CategoricalFlag("flags", {"FLAG_A": 0, "FLAG_B": 1})
        different = CategoricalFlag("flags", {"FLAG_A": 0, "FLAG_B": 2})
        assert original != different

    @pytest.mark.parametrize(
        "non_cat",
        ["hello", 123, {"key": "value"}, pl.DataFrame(), BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})],
        ids=["str", "int", "dict", "df", "bitwise_flag"],
    )
    def test_different_object(self, non_cat: Any) -> None:
        """Test that comparing against a non-categorical flag objects are not equal."""
        original = CategoricalFlag("flags", {"FLAG_A": 0, "FLAG_B": 1, "FLAG_C": 2})
        assert original != non_cat


class TestCategoricalFlagGetFlag:
    def setup_method(self) -> None:
        """Set up int and str flag systems for each test."""
        self.int_cf = CategoricalFlag("qc", {"FLAG_A": 0, "FLAG_B": 1, "FLAG_C": 2})
        self.str_cf = CategoricalFlag("met", {"FLAG_A": "A", "FLAG_B": "B", "FLAG_C": "S"})

    def test_get_int_by_name(self) -> None:
        """Test looking up a member by name returns the correct enum member."""
        assert self.int_cf.get_flag("FLAG_A").value == 0
        assert self.int_cf.get_flag("FLAG_B").value == 1

    def test_get_int_by_value(self) -> None:
        """Test looking up a member by int value returns the correct enum member."""
        assert self.int_cf.get_flag(0).value == 0
        assert self.int_cf.get_flag(1).value == 1

    def test_get_str_by_name(self) -> None:
        """Test looking up a member by name returns the correct enum member."""
        assert self.str_cf.get_flag("FLAG_A").value == "A"
        assert self.str_cf.get_flag("FLAG_B").value == "B"

    def test_get_str_by_value(self) -> None:
        """Test looking up a member by its value returns the correct enum member."""
        assert self.str_cf.get_flag("A").value == "A"

    def test_unknown_name_raises(self) -> None:
        """Test that an unknown name raises CategoricalFlagUnknownError."""
        with pytest.raises(CategoricalFlagUnknownError):
            self.int_cf.get_flag("unknown")

    def test_unknown_int_value_raises(self) -> None:
        """Test that an unknown int value raises CategoricalFlagUnknownError."""
        with pytest.raises(CategoricalFlagUnknownError):
            self.int_cf.get_flag(99)

    def test_wrong_type_raises(self) -> None:
        """Test that passing a non-int/str type raises CategoricalFlagTypeError."""
        with pytest.raises(CategoricalFlagTypeError):
            self.int_cf.get_flag(1.0)  # type: ignore[arg-type]


def make_tf() -> TimeFrame:
    """Create a minimal TimeFrame for flag tests."""
    return TimeFrame(
        pl.DataFrame(
            {
                "time": [datetime(2025, 1, 1), datetime(2025, 1, 2), datetime(2025, 1, 3)],
                "value": [10, 20, 30],
            }
        ),
        "time",
    )


def make_int_scalar_tf() -> TimeFrame:
    """Create a TimeFrame with an int categorical flag system and scalar flag column."""
    tf = make_tf().with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1, "FLAG_C": 2}, flag_type="categorical")
    tf.init_flag_column("qc", "flag_col")
    return tf


def make_str_scalar_tf() -> TimeFrame:
    """Create a TimeFrame with a str categorical flag system and scalar flag column."""
    tf = make_tf().with_flag_system("met", {"FLAG_A": "A", "FLAG_B": "B", "FLAG_C": "C"})
    tf.init_flag_column("met", "flag_col")
    return tf


def make_int_list_tf() -> TimeFrame:
    """Create a TimeFrame with an int categorical flag system and list-mode flag column."""
    tf = make_tf().with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1, "FLAG_C": 2}, flag_type="categorical")
    tf.init_flag_column("qc", "flag_col", list_mode=True)
    return tf


def make_str_list_tf() -> TimeFrame:
    """Create a TimeFrame with a str categorical flag system and list-mode flag column."""
    tf = make_tf().with_flag_system("met", {"FLAG_A": "A", "FLAG_B": "B"})
    tf.init_flag_column("met", "flag_col", list_mode=True)
    return tf


class TestAddFlag:
    def test_int_scalar_all_rows(self) -> None:
        """Test adding an int flag to all rows by name."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].to_list() == [0, 0, 0]

    def test_int_scalar_by_value(self) -> None:
        """Test adding an int flag by its integer value."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", 1)
        assert tf.df["flag_col"].to_list() == [1, 1, 1]

    def test_int_scalar_with_expr(self) -> None:
        """Test that expr limits which rows are updated."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_B", pl.col("value").gt(15))
        result = tf.df["flag_col"].to_list()
        assert result[0] is None
        assert result[1] == 1
        assert result[2] == 1

    def test_int_scalar_overwrite_true(self) -> None:
        """Test that overwrite=True replaces an existing value."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_B", overwrite=True)
        assert tf.df["flag_col"].to_list() == [1, 1, 1]

    def test_int_scalar_overwrite_false(self) -> None:
        """Test that overwrite=False skips rows that already have a value."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A", pl.col("value").le(20))
        tf.add_flag("flag_col", "FLAG_B", overwrite=False)
        result = tf.df["flag_col"].to_list()
        assert result[0] == 0  # not overwritten
        assert result[1] == 0  # not overwritten
        assert result[2] == 1  # was null, now set to bad

    def test_int_scalar_unknown_name_raises(self) -> None:
        """Test that adding an unknown flag name raises CategoricalFlagUnknownError."""
        tf = make_int_scalar_tf()
        with pytest.raises(CategoricalFlagUnknownError):
            tf.add_flag("flag_col", "unknown")

    def test_int_scalar_unknown_value_raises(self) -> None:
        """Test that adding an unknown int value raises CategoricalFlagUnknownError."""
        tf = make_int_scalar_tf()
        with pytest.raises(CategoricalFlagUnknownError):
            tf.add_flag("flag_col", 99)

    def test_str_scalar_by_name(self) -> None:
        """Test adding a str flag by its name."""
        tf = make_str_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].to_list() == ["A", "A", "A"]

    def test_str_scalar_by_value(self) -> None:
        """Test adding a str flag by its value string."""
        tf = make_str_scalar_tf()
        tf.add_flag("flag_col", "A")
        assert tf.df["flag_col"].to_list() == ["A", "A", "A"]

    def test_str_scalar_overwrite_false(self) -> None:
        """Test that overwrite=False leaves already-set rows unchanged."""
        tf = make_str_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A", pl.col("value").eq(10))
        tf.add_flag("flag_col", "FLAG_B", overwrite=False)
        result = tf.df["flag_col"].to_list()
        assert result[0] == "A"  # already set, not overwritten
        assert result[1] == "B"
        assert result[2] == "B"

    def test_int_list_appends_to_list(self) -> None:
        """Test that add_flag appends the value to the list."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].to_list() == [[0], [0], [0]]

    def test_int_list_twice_does_not_duplicate(self) -> None:
        """Test that calling add_flag with the same flag twice does not duplicate it."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].to_list() == [[0], [0], [0]]

    def test_int_list_two_different_flags(self) -> None:
        """Test that adding two different flags appends both."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_B")
        assert tf.df["flag_col"].to_list() == [[0, 1], [0, 1], [0, 1]]

    def test_int_list_with_expr(self) -> None:
        """Test that expr limits which rows are updated in list mode."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_B", pl.col("value").gt(15))
        result = tf.df["flag_col"].to_list()
        assert result[0] == []
        assert result[1] == [1]
        assert result[2] == [1]


class TestRemoveFlag:
    def test_int_scalar_all_rows(self) -> None:
        """Test that remove_flag sets the value to null on all rows."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.remove_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].is_null().all()

    def test_int_scalar_with_expr(self) -> None:
        """Test that expr limits which rows are cleared."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.remove_flag("flag_col", "FLAG_A", pl.col("value").gt(15))
        result = tf.df["flag_col"].to_list()
        assert result[0] == 0
        assert result[1] is None
        assert result[2] is None

    def test_str_scalar_sets_null(self) -> None:
        """Test that remove_flag sets str values to null."""
        tf = make_str_scalar_tf()
        tf.add_flag("flag_col", "FLAG_B")
        tf.remove_flag("flag_col", "FLAG_B")
        assert tf.df["flag_col"].is_null().all()

    def test_int_list_removes_from_list(self) -> None:
        """Test that remove_flag removes the value from the list."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_B")
        tf.remove_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].to_list() == [[1], [1], [1]]

    def test_int_list_with_expr(self) -> None:
        """Test that expr limits which rows have the value removed."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_B")
        tf.remove_flag("flag_col", "FLAG_A", pl.col("value").gt(15))
        result = tf.df["flag_col"].to_list()
        assert result[0] == [0, 1]
        assert result[1] == [1]
        assert result[2] == [1]

    def test_int_list_absent_value_is_noop(self) -> None:
        """Test that removing a value not in the list is a no-op."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.remove_flag("flag_col", "FLAG_B")  # "FLAG_B" was never added
        assert tf.df["flag_col"].to_list() == [[0], [0], [0]]

    def test_str_list_add_and_remove(self) -> None:
        """Test adding and removing str values in list mode."""
        tf = make_str_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_B")
        tf.remove_flag("flag_col", "FLAG_A")
        assert tf.df["flag_col"].to_list() == [["B"], ["B"], ["B"]]


class TestDecodeFlagColumn:
    @staticmethod
    def setup_int_scalar_tf() -> TimeFrame:
        """Create a TimeFrame with populated int scalar flags for decode tests."""
        tf = make_int_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A", pl.col("value").eq(10))
        tf.add_flag("flag_col", "FLAG_B", pl.col("value").eq(20))
        # row 3 (value=30) remains null
        return tf

    def test_int_scalar_dtype_is_utf8(self) -> None:
        """Test that decoded int scalar column has Utf8 dtype."""
        tf_dec = self.setup_int_scalar_tf().decode_flag_column("flag_col")
        assert tf_dec.df["flag_col"].dtype == pl.Utf8

    def test_int_scalar_values_replaced_with_names(self) -> None:
        """Test that raw int values are replaced with flag names."""
        tf_dec = self.setup_int_scalar_tf().decode_flag_column("flag_col")
        assert tf_dec.df["flag_col"].to_list() == ["FLAG_A", "FLAG_B", None]

    def test_int_scalar_is_decoded_set_true(self) -> None:
        """Test that is_decoded is True on the decoded column."""
        tf_dec = self.setup_int_scalar_tf().decode_flag_column("flag_col")
        assert tf_dec.get_flag_column("flag_col").is_decoded is True

    def test_int_scalar_original_unchanged(self) -> None:
        """Test that decode returns a new TimeFrame and leaves the original intact."""
        tf = self.setup_int_scalar_tf()
        tf_dec = tf.decode_flag_column("flag_col")
        assert tf.get_flag_column("flag_col").is_decoded is False
        assert tf_dec is not tf

    def test_int_scalar_already_decoded_raises(self) -> None:
        """Test that decoding an already-decoded column raises ColumnTypeError."""
        tf_dec = self.setup_int_scalar_tf().decode_flag_column("flag_col")
        with pytest.raises(ColumnTypeError):
            tf_dec.decode_flag_column("flag_col")

    @staticmethod
    def setup_str_scalar_tf() -> TimeFrame:
        """Create a TimeFrame with populated str scalar flags for decode tests."""
        tf = make_str_scalar_tf()
        tf.add_flag("flag_col", "FLAG_A", pl.col("value").eq(10))
        tf.add_flag("flag_col", "FLAG_B", pl.col("value").eq(20))
        return tf

    def test_str_scalar_dtype_is_utf8(self) -> None:
        """Test that decoded str scalar column still has Utf8 dtype (names replace values)."""
        tf_dec = self.setup_str_scalar_tf().decode_flag_column("flag_col")
        assert tf_dec.df["flag_col"].dtype == pl.Utf8

    def test_str_scalar_values_replaced_with_names(self) -> None:
        """Test that str values are replaced with their flag names."""
        tf_dec = self.setup_str_scalar_tf().decode_flag_column("flag_col")
        assert tf_dec.df["flag_col"].to_list() == ["FLAG_A", "FLAG_B", None]

    @staticmethod
    def setup_int_list_tf() -> TimeFrame:
        """Create a TimeFrame with populated int list flags for decode tests."""
        tf = make_int_list_tf()
        tf.add_flag("flag_col", "FLAG_A")
        tf.add_flag("flag_col", "FLAG_B", pl.col("value").gt(15))
        return tf

    def test_int_list_dtype_is_list_utf8(self) -> None:
        """Test that decoded list column has List(Utf8) dtype."""
        tf_dec = self.setup_int_list_tf().decode_flag_column("flag_col")
        assert tf_dec.df["flag_col"].dtype == pl.List(pl.Utf8)

    def test_int_list_values_replaced_with_names(self) -> None:
        """Test that values in lists are replaced with flag names."""
        tf_dec = self.setup_int_list_tf().decode_flag_column("flag_col")
        assert tf_dec.df["flag_col"].to_list() == [["FLAG_A"], ["FLAG_A", "FLAG_B"], ["FLAG_A", "FLAG_B"]]

    def test_int_list_add_remove_on_decoded(self) -> None:
        """Test that add_flag and remove_flag work transparently on decoded list columns."""
        tf_dec = self.setup_int_list_tf().decode_flag_column("flag_col")
        tf_dec.add_flag("flag_col", "FLAG_C", pl.col("value").eq(10))
        tf_dec.remove_flag("flag_col", "FLAG_A", pl.col("value").eq(10))
        result = tf_dec.df["flag_col"].to_list()
        assert result[0] == ["FLAG_C"]


class TestEncodeFlagColumn:
    def test_int_scalar_restores_int_dtype(self) -> None:
        """Test that encoding a decoded int scalar column restores Int32 dtype."""
        tf = TestDecodeFlagColumn.setup_int_scalar_tf()
        tf_enc = tf.decode_flag_column("flag_col").encode_flag_column("flag_col")
        assert tf_enc.df["flag_col"].dtype == pl.Int32

    def test_int_scalar_round_trip(self) -> None:
        """Test that decode then encode restores the original int values."""
        tf = TestDecodeFlagColumn.setup_int_scalar_tf()
        original = tf.df["flag_col"].clone()
        tf_roundtrip = tf.decode_flag_column("flag_col").encode_flag_column("flag_col")
        assert_series_equal(tf_roundtrip.df["flag_col"], original)

    def test_int_scalar_is_decoded_false_after_encode(self) -> None:
        """Test that is_decoded is False after encoding."""
        tf = TestDecodeFlagColumn.setup_int_scalar_tf()
        tf_enc = tf.decode_flag_column("flag_col").encode_flag_column("flag_col")
        assert tf_enc.get_flag_column("flag_col").is_decoded is False

    def test_int_scalar_not_decoded_raises(self) -> None:
        """Test that encoding a column that is not decoded raises ColumnTypeError."""
        tf = TestDecodeFlagColumn.setup_int_scalar_tf()
        with pytest.raises(ColumnTypeError):
            tf.encode_flag_column("flag_col")

    def test_int_scalar_unknown_name_raises(self) -> None:
        """Test that encoding a column with an unknown flag name raises CategoricalFlagUnknownError."""
        tf = TestDecodeFlagColumn.setup_int_scalar_tf()
        tf_dec = tf.decode_flag_column("flag_col")
        df_bad = tf_dec.df.with_columns(pl.Series("flag_col", ["FLAG_A", "UNKNOWN", None], dtype=pl.Utf8))
        tf_bad = tf_dec.with_df(df_bad)
        with pytest.raises(CategoricalFlagUnknownError):
            tf_bad.encode_flag_column("flag_col")

    def test_str_scalar_restores_str_values(self) -> None:
        """Test that encoding restores the original str values."""
        tf = TestDecodeFlagColumn.setup_str_scalar_tf()
        tf_enc = tf.decode_flag_column("flag_col").encode_flag_column("flag_col")
        assert tf_enc.df["flag_col"].to_list() == ["A", "B", None]

    def test_str_scalar_round_trip(self) -> None:
        """Test that decode then encode restores original str values."""
        tf = TestDecodeFlagColumn.setup_str_scalar_tf()
        original = tf.df["flag_col"].clone()
        tf_roundtrip = tf.decode_flag_column("flag_col").encode_flag_column("flag_col")
        assert_series_equal(tf_roundtrip.df["flag_col"], original)

    def test_int_list_round_trip(self) -> None:
        """Test that decode then encode restores original list values."""
        tf = TestDecodeFlagColumn.setup_int_list_tf()
        original = tf.df["flag_col"].clone()
        tf_roundtrip = tf.decode_flag_column("flag_col").encode_flag_column("flag_col")
        assert_series_equal(tf_roundtrip.df["flag_col"], original)

    def test_int_list_unknown_name_raises(self) -> None:
        """Test that encoding a list column with an unknown name raises CategoricalFlagUnknownError."""
        tf = TestDecodeFlagColumn.setup_int_list_tf()
        tf_dec = tf.decode_flag_column("flag_col")
        df_bad = tf_dec.df.with_columns(pl.Series("flag_col", [["FLAG_A", "UNKNOWN"], [], []], dtype=pl.List(pl.Utf8)))
        tf_bad = tf_dec.with_df(df_bad)
        with pytest.raises(CategoricalFlagUnknownError):
            tf_bad.encode_flag_column("flag_col")


class TestRegisterFlagColumnInfersMode:
    def test_scalar_column_registered_as_scalar(self) -> None:
        """Test that an existing Int32 column is registered in scalar mode."""
        tf = make_tf().with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")
        df = tf.df.with_columns(pl.lit(None, dtype=pl.Int32).alias("my_flags"))
        tf2 = tf.with_df(df)
        tf2.register_flag_column("my_flags", "qc")
        col = tf2.get_flag_column("my_flags")
        assert isinstance(col, CategoricalFlagColumn)
        assert col.list_mode is False

    def test_list_column_registered_as_list(self) -> None:
        """Test that an existing List(Int32) column is inferred as list mode automatically."""
        tf = make_tf().with_flag_system("qc", {"FLAG_A": 0, "FLAG_B": 1}, flag_type="categorical")
        df = tf.df.with_columns(pl.Series("my_flags", [[], [], []], dtype=pl.List(pl.Int32)))
        tf2 = tf.with_df(df)
        tf2.register_flag_column("my_flags", "qc")
        col = tf2.get_flag_column("my_flags")
        assert isinstance(col, CategoricalFlagColumn)
        assert col.list_mode is True
