from datetime import datetime
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_series_equal

from time_stream import TimeFrame
from time_stream.exceptions import (
    BitwiseFlagDuplicateError,
    BitwiseFlagTypeError,
    BitwiseFlagUnknownError,
    BitwiseFlagValueError,
    ColumnNotFoundError,
)
from time_stream.flags.bitwise_flag_system import BitwiseFlag
from time_stream.flags.categorical_flag_system import CategoricalSingleFlag


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

    def test_get_flag_by_integer(self) -> None:
        """Test retrieving a flag value by integer."""
        assert Flags.get_flag(1) == Flags.FLAG_A
        assert Flags.get_flag(2) == Flags.FLAG_B
        assert Flags.get_flag(4) == Flags.FLAG_C

    def test_get_flag_by_string(self) -> None:
        """Test retrieving a flag value by name."""
        assert Flags.get_flag("FLAG_A") == Flags.FLAG_A
        assert Flags.get_flag("FLAG_B") == Flags.FLAG_B
        assert Flags.get_flag("FLAG_C") == Flags.FLAG_C

    def test_get_flag_invalid_integer(self) -> None:
        """Test that a combined (non-singular) integer raises an error."""
        with pytest.raises(BitwiseFlagUnknownError):
            Flags.get_flag(3)  # 3 is a combination of 1 and 2, not a valid single flag

    def test_get_flag_invalid_string(self) -> None:
        """Test that an unknown flag name raises an error."""
        with pytest.raises(BitwiseFlagUnknownError):
            Flags.get_flag("FLAG_D")

    def test_get_flag_invalid_type(self) -> None:
        """Test that a non-int/str argument raises an error."""
        with pytest.raises(BitwiseFlagTypeError):
            Flags.get_flag(3.5)  # noqa - expecting wrong type

    def test_to_dict(self) -> None:
        """Test creating a dictionary from a bitwise flag."""
        result = Flags.to_dict()
        expected = {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4}
        assert result == expected

    def test_value_type_returns_int(self) -> None:
        """Test that value_type() returns int."""
        assert Flags.value_type() is int


class TestBitwiseFlagValidateColumn:
    def test_single_flag_values_pass(self) -> None:
        """Test that individual flag values are accepted."""
        Flags.validate_column(pl.Series("flags", [1, 2, 4]))

    def test_combined_flag_values_pass(self) -> None:
        """Test that valid bitwise combinations are accepted."""
        Flags.validate_column(pl.Series("flags", [0, 3, 5, 6, 7]))

    def test_zero_passes(self) -> None:
        """Test that 0 (no flags set) is a valid value."""
        Flags.validate_column(pl.Series("flags", [0, 0, 0]))

    def test_nulls_are_allowed(self) -> None:
        """Test that null values do not raise an error."""
        Flags.validate_column(pl.Series("flags", [1, None, 4], dtype=pl.Int32))

    def test_empty_series_passes(self) -> None:
        """Test that an empty series does not raise an error."""
        Flags.validate_column(pl.Series("flags", [], dtype=pl.Int32))

    def test_unknown_bit_raises(self) -> None:
        """Test that a value with a bit outside the known mask raises BitwiseFlagUnknownError."""
        with pytest.raises(BitwiseFlagUnknownError):
            Flags.validate_column(pl.Series("flags", [1, 8]))  # 8 not defined in Flags


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
            CategoricalSingleFlag("flags", {"FLAG_A": 0, "FLAG_B": 1, "FLAG_C": 2}),
        ],
        ids=["str", "int", "dict", "df", "categorical_flag"],
    )
    def test_different_object(self, non_bw: Any) -> None:
        """Test that comparing against a non-bitwise flag objects are not equal."""
        original = BitwiseFlag("flags", {"FLAG_A": 1, "FLAG_B": 2, "FLAG_C": 4})
        assert original != non_bw


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
        """Test that removing a flag from a valid flag column using the flag name (rather than value) works as
        expected."""
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
