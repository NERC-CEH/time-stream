import unittest
from datetime import datetime

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream.base import TimeSeries
from time_stream.columns import DataColumn, FlagColumn, SupplementaryColumn
from time_stream.exceptions import BitwiseFlagUnknownError, ColumnTypeError, MetadataError


class BaseTimeSeriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Set up a mock TimeSeries instance for all test classes."""
        cls.df = pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "data_col1": [1, 2, 3],
                "data_col2": [4, 5, 6],
                "supp_col": ["a", "b", "c"],
                "flag_col": [0, 0, 0],
                "flag_col2": [2, 2, 2],
            }
        )
        cls.metadata = {
            "data_col1": {"key1": "1", "key2": "10", "key3": "100"},
            "data_col2": {"key1": "2", "key2": "20", "key3": "200"},
            "supp_col": {"key1": "3", "key2": "30", "key3": "300"},
        }
        cls.flag_systems = {
            "example_flag_system": {"OK": 1, "WARNING": 2, "ERROR": 4},
            "example_flag_system2": {"A": 8, "B": 16, "C": 32},
        }

        cls.ts = TimeSeries(
            df=cls.df,
            time_name="time",
            supplementary_columns=["supp_col"],
            flag_columns={"flag_col": "example_flag_system"},
            flag_systems=cls.flag_systems,
            column_metadata=cls.metadata,
        )


class TestMetadata(BaseTimeSeriesTest):
    def setUp(self) -> None:
        self.column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

    def test_retrieve_all_metadata(self) -> None:
        """Test retrieving all metadata for a column"""
        result = self.column.metadata()
        expected = {"key1": "1", "key2": "10", "key3": "100"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_specific_key(self) -> None:
        """Test retrieving a specific metadata key."""
        result = self.column.metadata("key1")
        expected = {"key1": "1"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self) -> None:
        """Test retrieving multiple metadata keys."""
        result = self.column.metadata(["key1", "key3"])
        expected = {"key1": "1", "key3": "100"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_key(self) -> None:
        """Test that error raised when requesting a non-existent metadata key"""
        with self.assertRaises(MetadataError):
            self.column.metadata("nonexistent_key")

    def test_retrieve_metadata_for_nonexistent_key_strict_false(self) -> None:
        """Test that an empty result is returned when strict is false for non-existent key."""
        expected = {"nonexistent_key": None}
        result = self.column.metadata("nonexistent_key", strict=False)
        self.assertEqual(result, expected)


class TestRemoveMetadata(BaseTimeSeriesTest):
    def setUp(self) -> None:
        self.column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

    def test_remove_all_metadata(self) -> None:
        """Test removing all metadata from a column"""
        self.column.remove_metadata()
        expected = {}
        self.assertEqual(self.column.metadata(), expected)

    def test_remove_single_metadata(self) -> None:
        """Test removing single metadata key from a column"""
        self.column.remove_metadata("key1")
        expected = {"key2": "10", "key3": "100"}
        self.assertEqual(self.column.metadata(), expected)

    def test_remove_multiple_metadata(self) -> None:
        """Test removing multiple metadata keys from a column"""
        self.column.remove_metadata(["key1", "key2"])
        expected = {"key3": "100"}
        self.assertEqual(self.column.metadata(), expected)


class TestSetAsSupplementary(BaseTimeSeriesTest):
    def setUp(self) -> None:
        """Ensure ts is reset before each test."""
        super().setUpClass()

    @parameterized.expand([("data_to_supplementary", "data_col1"), ("flag_to_supplementary", "flag_col")])
    def test_to_supplementary(self, _: str, col: str) -> None:
        """Test that a non-supplementary column gets converted to a supplementary column."""
        column = self.ts.columns[col].set_as_supplementary()

        self.assertIsInstance(column, SupplementaryColumn)
        self.assertIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)


class TestSetAsFlag(BaseTimeSeriesTest):
    def setUp(self) -> None:
        """Ensure ts is reset before each test."""
        super().setUpClass()

    @parameterized.expand([("data_to_flag", "data_col1"), ("supplementary_to_flag", "supp_col")])
    def test_to_flag(self, _: str, col: str) -> None:
        """Test that a non-flag column gets converted to a flag column."""
        column = self.ts.columns[col].set_as_flag("example_flag_system")

        self.assertIsInstance(column, FlagColumn)
        self.assertIn(column.name, self.ts.flag_columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)


class TestUnset(BaseTimeSeriesTest):
    def setUp(self) -> None:
        """Ensure ts is reset before each test."""
        super().setUpClass()

    @parameterized.expand(
        [("unset_data_col", "data_col1"), ("unset_supplementary_col", "supp_col"), ("unset_flag_col", "flag_col")]
    )
    def test_unset_column(self, _: str, col: str) -> None:
        """Test unsetting a column converts (or remains as) a data column"""
        column = self.ts.columns[col].unset()

        self.assertIsInstance(column, DataColumn)
        self.assertIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)


class TestRemove(BaseTimeSeriesTest):
    def setUp(self) -> None:
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_remove_data_column(self) -> None:
        """Test that removing a data column works as expected"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column.remove()

        self.assertIsNone(column._ts)
        self.assertNotIn(column.name, self.ts.columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.df.columns)

    def test_remove_supplementary_column(self) -> None:
        """Test that removing a supplementary column works as expected"""
        column = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        column.remove()

        self.assertIsNone(column._ts)
        self.assertNotIn(column.name, self.ts.columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.df.columns)

    def test_remove_flag_column(self) -> None:
        """Test that removing a flag column works as expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column.remove()

        self.assertIsNone(column._ts)
        self.assertNotIn(column.name, self.ts.columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.df.columns)


class TestAddFlag(BaseTimeSeriesTest):
    def setUp(self) -> None:
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_add_flag_to_flag_column_no_expr(self) -> None:
        """Test that adding a flag to a valid flag column with no expression sets all values in the
        column to that flag"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_to_flag_column_with_expr(self) -> None:
        """Test that adding a flag to a valid flag column with an expression sets all values in the
        column that match that expression to that flag"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag(1, pl.col("data_col1").gt(1))
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [0, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_by_name_to_flag_column(self) -> None:
        """Test that adding a flag to a valid flag column using the flag name (rather than value)
        sets all values in the column that match that expression to the correct flag value"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag("ERROR")
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [4, 4, 4])

        assert_series_equal(result, expected)

    def test_adding_non_existent_flag_to_flag_column_raises_error(self) -> None:
        """Test that trying to add an invalid flag to a valid flag column raises error"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        with self.assertRaises(BitwiseFlagUnknownError):
            column.add_flag(10)

    def test_add_flag_to_data_column_raises_error(self) -> None:
        """Test that trying to add a flag to a data column raises error"""
        column = DataColumn("data_col1", self.ts)
        with self.assertRaises(ColumnTypeError):
            column.add_flag(1)

    def test_add_flag_to_supplementary_column_raises_error(self) -> None:
        """Test that trying to add a flag to a supplementary column raises error"""
        column = SupplementaryColumn("supp_col", self.ts)
        with self.assertRaises(ColumnTypeError):
            column.add_flag(1)

    def test_add_flag_twice(self) -> None:
        """Test that adding a flag twice uses the bitwise math, so doesn't actually add the value twice"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 1, 1])
        assert_series_equal(result, expected)

        # adding a second time should leave it as is
        column.add_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 1, 1])
        assert_series_equal(result, expected)


class TestRemoveFlag(BaseTimeSeriesTest):
    def setUp(self) -> None:
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_remove_flag_from_flag_column_no_expr(self) -> None:
        """Test that removing a flag from a valid flag column with no expression works as expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 1, 1]))

        column.remove_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [0, 0, 0])

        assert_series_equal(result, expected)

    def test_remove_flag_from_flag_column_with_combined_flag_values(self) -> None:
        """Test that removing a flag from a valid flag column where the rows have combined flag values
        works as expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 3, 7]))

        column.remove_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [0, 2, 6])

        assert_series_equal(result, expected)

    def test_remove_flag_twice(self) -> None:
        """Test that removing a flag twice uses the bitwise math, so doesn't actually remove the value twice"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [3, 3, 3]))

        column.remove_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [2, 2, 2])
        assert_series_equal(result, expected)

        # removing a second time should leave it as is
        column.remove_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [2, 2, 2])
        assert_series_equal(result, expected)

    def test_remove_flag_from_flag_column_with_expr(self) -> None:
        """Test that removing a flag from a valid flag column with an expression removes flag from only those rows"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 1, 1]))

        column.remove_flag(1, pl.col("data_col1").gt(1))
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 0, 0])

        assert_series_equal(result, expected)

    def test_remove_flag_by_name_from_flag_column(self) -> None:
        """Test that removing a flag from a valid flag column using the flag name (rather than value) works as
        expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 4, 1]))

        column.remove_flag("ERROR")
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 0, 1])

        assert_series_equal(result, expected)

    def test_removing_non_existent_flag_from_flag_column_raises_error(self) -> None:
        """Test that trying to remove an invalid flag to a valid flag column raises error"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        with self.assertRaises(BitwiseFlagUnknownError):
            column.remove_flag(10)

    def test_remove_flag_from_data_column_raises_error(self) -> None:
        """Test that trying to remove a flag to a data column raises error"""
        column = DataColumn("data_col1", self.ts)
        with self.assertRaises(ColumnTypeError):
            column.remove_flag(1)

    def test_remove_flag_from_supplementary_column_raises_error(self) -> None:
        """Test that trying to remove a flag to a supplementary column raises error"""
        column = SupplementaryColumn("supp_col", self.ts)
        with self.assertRaises(ColumnTypeError):
            column.remove_flag(1)


class TestAsTimeSeries(BaseTimeSeriesTest):
    def test_as_timeseries(self) -> None:
        column = DataColumn("data_col1", self.ts)
        ts = column.as_timeseries()
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(list(ts.columns.keys()), ["data_col1"])


class TestGetattr(BaseTimeSeriesTest):
    def test_access_metadata_key(self) -> None:
        """Test accessing metadata key for a column."""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        result = column.key1
        expected = "1"
        self.assertEqual(result, expected)

    def test_access_nonexistent_attribute(self) -> None:
        """Test accessing metadata key that doesn't exist"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        with self.assertRaises(MetadataError):
            _ = column.key0


class TestPrimaryTimeColumn(BaseTimeSeriesTest):
    def test_set_as_supplementary_not_allowed(self) -> None:
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.set_as_supplementary()

    def test_set_as_flag_not_allowed(self) -> None:
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.set_as_flag("example_flag_system")

    def test_unset_not_allowed(self) -> None:
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.unset()


class TestEq(BaseTimeSeriesTest):
    def test_data_columns_are_equal(self) -> None:
        column1 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column2 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        self.assertEqual(column1, column2)

    def test_data_columns_are_not_equal(self) -> None:
        column1 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column2 = DataColumn("data_col2", self.ts, self.metadata["data_col2"])
        self.assertNotEqual(column1, column2)

    def test_supplementary_columns_are_equal(self) -> None:
        column1 = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        column2 = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        self.assertEqual(column1, column2)

    def test_supplementary_columns_are_not_equal(self) -> None:
        column1 = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        column2 = SupplementaryColumn("data_col1", self.ts, self.metadata["data_col1"])
        self.assertNotEqual(column1, column2)

    def test_flag_columns_are_equal(self) -> None:
        column1 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column2 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.assertEqual(column1, column2)

    def test_flag_columns_are_not_equal(self) -> None:
        column1 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column2 = FlagColumn("data_col1", self.ts, "example_flag_system", {})
        self.assertNotEqual(column1, column2)

    def test_flag_columns_different_flag_systems_are_not_equal(self) -> None:
        column1 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column2 = FlagColumn("flag_col", self.ts, "example_flag_system2", {})
        self.assertNotEqual(column1, column2)

    def test_flag_columns_different_types_are_not_equal(self) -> None:
        column1 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column2 = SupplementaryColumn("data_col1", self.ts, self.metadata["data_col1"])
        self.assertNotEqual(column1, column2)
