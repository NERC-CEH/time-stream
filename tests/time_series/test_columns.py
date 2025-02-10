import unittest
from datetime import datetime

import polars as pl
from polars.testing import assert_series_equal

from time_series.base import TimeSeries
from time_series.columns import DataColumn, FlagColumn, SupplementaryColumn


class BaseTimeSeriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a mock TimeSeries instance for all test classes."""
        cls.df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "data_col1": [1, 2, 3],
            "data_col2": [4, 5, 6],
            "supp_col": ["a", "b", "c"],
            "flag_col": [0, 0, 0],
        })
        cls.metadata = {
            "data_col1": {"key1": "1", "key2": "10", "key3": "100"},
            "data_col2": {"key1": "2", "key2": "20", "key3": "200"},
            "supp_col": {"key1": "3", "key2": "30", "key3": "300"},
        }
        cls.flag_systems = {"example_flag_system": {"OK": 1, "WARNING": 2, "ERROR": 4},
                            "example_flag_system2": {"A": 8, "B": 16, "C": 32}}

        cls.ts = TimeSeries(df=cls.df,
                            time_name="time",
                            supplementary_columns=["supp_col"],
                            flag_columns={"flag_col": "example_flag_system"},
                            flag_systems=cls.flag_systems,
                            column_metadata=cls.metadata)


class TestMetadata(BaseTimeSeriesTest):
    def setUp(self):
        self.column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

    def test_retrieve_all_metadata(self):
        """Test retrieving all metadata for a column"""
        result = self.column.metadata()
        expected = {"key1": "1", "key2": "10", "key3": "100"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_specific_key(self):
        """Test retrieving a specific metadata key."""
        result = self.column.metadata("key1")
        expected = {"key1": "1"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self):
        """Test retrieving multiple metadata keys."""
        result = self.column.metadata(["key1", "key3"])
        expected = {"key1": "1", "key3": "100"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_key(self):
        """Test that error raised when requesting a non-existent metadata key"""
        with self.assertRaises(KeyError):
            self.column.metadata("nonexistent_key")

    def test_retrieve_metadata_for_nonexistent_key_strict_false(self):
        """Test that an empty result is returned when strict is false for non-existent key."""
        expected = {'nonexistent_key': None}
        result = self.column.metadata("nonexistent_key", strict=False)
        self.assertEqual(result, expected)


class TestRemoveMetadata(BaseTimeSeriesTest):
    def setUp(self):
        self.column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

    def test_remove_all_metadata(self):
        """Test removing all metadata from a column"""
        self.column.remove_metadata()
        expected = {}
        self.assertEqual(self.column.metadata(), expected)

    def test_remove_single_metadata(self):
        """Test removing single metadata key from a column"""
        self.column.remove_metadata("key1")
        expected = {"key2": "10", "key3": "100"}
        self.assertEqual(self.column.metadata(), expected)

    def test_remove_multiple_metadata(self):
        """Test removing multiple metadata keys from a column"""
        self.column.remove_metadata(["key1", "key2"])
        expected = {"key3": "100"}
        self.assertEqual(self.column.metadata(), expected)


class TestSetAsSupplementary(BaseTimeSeriesTest):
    def test_data_to_supplementary(self):
        """ Test that a data column gets converted to a supplementary column"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

        column = column.set_as_supplementary()
        self.assertIsInstance(column, SupplementaryColumn)
        self.assertIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)

    def test_flag_to_supplementary(self):
        """ Test that a flag column gets converted to a supplementary column"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column = column.set_as_supplementary()
        self.assertIsInstance(column, SupplementaryColumn)
        self.assertIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)
        self.assertNotIn(column.name, self.ts.data_columns)


class TestSetAsFlag(BaseTimeSeriesTest):
    def test_data_to_flag(self):
        """ Test that a data column gets converted to a flag column"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

        column = column.set_as_flag("example_flag_system")
        self.assertIsInstance(column, FlagColumn)
        self.assertIn(column.name, self.ts.flag_columns)
        self.assertNotIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)

    def test_supplementary_to_flag(self):
        """ Test that a supplementary column gets converted to a flag column"""
        column = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])

        column = column.set_as_flag("example_flag_system")
        self.assertIsInstance(column, FlagColumn)
        self.assertIn(column.name, self.ts.flag_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.data_columns)

class TestUnset(BaseTimeSeriesTest):
    def test_unset_data_column(self):
        """ Test that unsetting a data column remains the same"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])

        column = column.unset()
        self.assertIsInstance(column, DataColumn)
        self.assertIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)

    def test_unset_supplementary_column(self):
        """ Test that unsetting a supplementary column converts it to a data column"""
        column = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])

        column = column.unset()
        self.assertIsInstance(column, DataColumn)
        self.assertIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)

    def test_unset_flag_column(self):
        """ Test that unsetting a supplementary column converts it to a data column"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column = column.unset()
        self.assertIsInstance(column, DataColumn)
        self.assertIn(column.name, self.ts.data_columns)
        self.assertNotIn(column.name, self.ts.supplementary_columns)
        self.assertNotIn(column.name, self.ts.flag_columns)


class TestRemove(BaseTimeSeriesTest):
    def test_remove_data_column(self):
        """ Test that removing a data column works as expected"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column.remove()

        self.assertIsNone(column._ts)
        self.assertNotIn(column.name, self.ts.columns)
        self.assertNotIn(column.name, self.ts.data_columns)

    def test_remove_supplementary_column(self):
        """ Test that removing a supplementary column works as expected"""
        column = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        column.remove()

        self.assertIsNone(column._ts)
        self.assertNotIn(column.name, self.ts.columns)
        self.assertNotIn(column.name, self.ts.data_columns)

    def test_remove_flag_column(self):
        """ Test that removing a flag column works as expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column.remove()

        self.assertIsNone(column._ts)
        self.assertNotIn(column.name, self.ts.columns)
        self.assertNotIn(column.name, self.ts.data_columns)

class TestAddFlag(BaseTimeSeriesTest):
    def setUp(self):
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_add_flag_to_flag_column_no_expr(self):
        """ Test that adding a flag to a valid flag column with no expression sets all values in the
        column to that flag"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_to_flag_column_with_expr(self):
        """ Test that adding a flag to a valid flag column with an expression sets all values in the
        column that match that expression to that flag"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag(1, pl.col("data_col1").gt(1))
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [0, 1, 1])

        assert_series_equal(result, expected)

    def test_add_flag_by_name_to_flag_column(self):
        """ Test that adding a flag to a valid flag column using the flag name (rather than value)
        sets all values in the column that match that expression to the correct flag value"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})

        column.add_flag("ERROR")
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [4, 4, 4])

        assert_series_equal(result, expected)

    def test_adding_non_existent_flag_to_flag_column_raises_error(self):
        """ Test that trying to add an invalid flag to a valid flag column raises error"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        with self.assertRaises(KeyError):
            column.add_flag(10)

    def test_add_flag_to_data_column_raises_error(self):
        """ Test that trying to add a flag to a data column raises error"""
        column = DataColumn("data_col1", self.ts)
        with self.assertRaises(TypeError):
            column.add_flag(1)

    def test_add_flag_to_supplementary_column_raises_error(self):
        """ Test that trying to add a flag to a supplementary column raises error"""
        column = SupplementaryColumn("supp_col", self.ts)
        with self.assertRaises(TypeError):
            column.add_flag(1)

    def test_add_flag_twice(self):
        """ Test that adding a flag twice uses the bitwise math, so doesn't actually add the value twice"""
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
    def setUp(self):
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_remove_flag_from_flag_column_no_expr(self):
        """ Test that removing a flag from a valid flag column with no expression works as expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 1, 1]))

        column.remove_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [0, 0, 0])

        assert_series_equal(result, expected)

    def test_remove_flag_from_flag_column_with_combined_flag_values(self):
        """ Test that removing a flag from a valid flag column where the rows have combined flag values
        works as expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 3, 7]))

        column.remove_flag(1)
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [0, 2, 6])

        assert_series_equal(result, expected)

    def test_remove_flag_twice(self):
        """ Test that removing a flag twice uses the bitwise math, so doesn't actually remove the value twice"""
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

    def test_remove_flag_from_flag_column_with_expr(self):
        """ Test that removing a flag from a valid flag column with an expression removes flag from only those rows"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 1, 1]))

        column.remove_flag(1, pl.col("data_col1").gt(1))
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 0, 0])

        assert_series_equal(result, expected)

    def test_remove_flag_by_name_from_flag_column(self):
        """ Test that removing a flag from a valid flag column using the flag name (rather than value) works as
        expected"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.ts.df = self.ts.df.with_columns(pl.Series("flag_col", [1, 4, 1]))

        column.remove_flag("ERROR")
        result = self.ts.df["flag_col"]
        expected = pl.Series("flag_col", [1, 0, 1])

        assert_series_equal(result, expected)

    def test_removing_non_existent_flag_from_flag_column_raises_error(self):
        """ Test that trying to remove an invalid flag to a valid flag column raises error"""
        column = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        with self.assertRaises(KeyError):
            column.remove_flag(10)

    def test_remove_flag_from_data_column_raises_error(self):
        """ Test that trying to remove a flag to a data column raises error"""
        column = DataColumn("data_col1", self.ts)
        with self.assertRaises(TypeError):
            column.remove_flag(1)

    def test_remove_flag_from_supplementary_column_raises_error(self):
        """ Test that trying to remove a flag to a supplementary column raises error"""
        column = SupplementaryColumn("supp_col", self.ts)
        with self.assertRaises(TypeError):
            column.remove_flag(1)


class TestAsTimeSeries(BaseTimeSeriesTest):
    def test_as_timeseries(self):
        column = DataColumn("data_col1", self.ts)
        ts = column.as_timeseries()
        self.assertIsInstance(ts, TimeSeries)
        self.assertEqual(list(ts.columns.keys()), ["data_col1"])


class TestGetattr(BaseTimeSeriesTest):
    def test_access_metadata_key(self):
        """Test accessing metadata key for a column."""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        result = column.key1
        expected = "1"
        self.assertEqual(result, expected)

    def test_access_nonexistent_attribute(self):
        """Test accessing metadata key that doesn't exist"""
        column = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        with self.assertRaises(AttributeError):
            _ = column.key0


class TestPrimaryTimeColumn(BaseTimeSeriesTest):
    def test_set_as_supplementary_not_allowed(self):
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.set_as_supplementary()

    def test_set_as_flag_not_allowed(self):
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.set_as_flag("example_flag_system")

    def test_unset_not_allowed(self):
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.unset()

    def test_remove_relationship_not_allowed(self):
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.remove_relationship()

    def test_add_relationship_not_allowed(self):
        column = self.ts.time_column
        with self.assertRaises(NotImplementedError):
            column.add_relationship()


class TestEq(BaseTimeSeriesTest):
    def test_data_columns_are_equal(self):
        column1 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column2 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        self.assertEqual(column1, column2)

    def test_data_columns_are_not_equal(self):
        column1 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column2 = DataColumn("data_col2", self.ts, self.metadata["data_col2"])
        self.assertNotEqual(column1, column2)

    def test_supplementary_columns_are_equal(self):
        column1 = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        column2 = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        self.assertEqual(column1, column2)

    def test_supplementary_columns_are_not_equal(self):
        column1 = SupplementaryColumn("supp_col", self.ts, self.metadata["supp_col"])
        column2 = SupplementaryColumn("data_col1", self.ts, self.metadata["data_col1"])
        self.assertNotEqual(column1, column2)

    def test_flag_columns_are_equal(self):
        column1 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column2 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        self.assertEqual(column1, column2)

    def test_flag_columns_are_not_equal(self):
        column1 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column2 = FlagColumn("data_col1", self.ts, "example_flag_system", {})
        self.assertNotEqual(column1, column2)

    def test_flag_columns_different_flag_systems_are_not_equal(self):
        column1 = FlagColumn("flag_col", self.ts, "example_flag_system", {})
        column2 = FlagColumn("flag_col", self.ts, "example_flag_system2", {})
        self.assertNotEqual(column1, column2)

    def test_flag_columns_different_types_are_not_equal(self):
        column1 = DataColumn("data_col1", self.ts, self.metadata["data_col1"])
        column2 = SupplementaryColumn("data_col1", self.ts, self.metadata["data_col1"])
        self.assertNotEqual(column1, column2)


class TestAddRelationship(BaseTimeSeriesTest):
    def setUp(self):
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_add_relationship_by_object(self):
        """Test that can add a relationship using column object."""
        self.ts.data_col1.add_relationship("supp_col")
        data_col1_rels = self.ts._relationship_manager._get_relationships(self.ts.data_col1)
        supp_col_rels = self.ts._relationship_manager._get_relationships(self.ts.supp_col)

        self.assertEqual(len(data_col1_rels), 1)
        self.assertEqual(len(supp_col_rels), 1)

    def test_add_relationship_by_str(self):
        """Test that can add a relationship using column string name."""
        self.ts.data_col1.add_relationship("supp_col")
        data_col1_rels = self.ts._relationship_manager._get_relationships(self.ts.data_col1)
        supp_col_rels = self.ts._relationship_manager._get_relationships(self.ts.supp_col)

        self.assertEqual(len(data_col1_rels), 1)
        self.assertEqual(len(supp_col_rels), 1)

    def test_add_multiple_relationships(self):
        """Test adding multiple relationships."""
        self.ts.data_col1.add_relationship("supp_col")
        self.ts.data_col1.add_relationship("flag_col")
        data_col1_rels = self.ts._relationship_manager._get_relationships(self.ts.data_col1)
        supp_col_rels = self.ts._relationship_manager._get_relationships(self.ts.supp_col)
        flag_col_rels = self.ts._relationship_manager._get_relationships(self.ts.flag_col)

        self.assertEqual(len(data_col1_rels), 2)
        self.assertEqual(len(supp_col_rels), 1)
        self.assertEqual(len(flag_col_rels), 1)

    def test_add_data_rel_to_data_column_raises_error(self):
        """ Test that can't set relationship between two data columns"""
        with self.assertRaises(TypeError):
            self.ts.data_col1.add_relationship("data_col2")

    def test_add_supp_rel_to_flag_column_raises_error(self):
        """ Test that can't set relationship between supplementary and flag columns"""
        with self.assertRaises(TypeError):
            self.ts.supp_col.add_relationship("flag_col")

    def test_add_flag_rel_to_supp_column_raises_error(self):
        """ Test that can't set relationship between flag and supplementary columns"""
        with self.assertRaises(TypeError):
            self.ts.flag_col.add_relationship("supp_col")


class TestRemoveRelationship(BaseTimeSeriesTest):
    def setUp(self):
        """Ensure ts is reset before each test."""
        super().setUpClass()

    def test_remove_relationship_by_object(self):
        """Test that can remove a relationship using column object, and it removes it from both directions."""
        self.ts.data_col1.add_relationship("supp_col")
        self.ts.data_col1.remove_relationship(self.ts.supp_col)
        data_col1_rels = self.ts._relationship_manager._get_relationships(self.ts.data_col1)
        supp_col_rels = self.ts._relationship_manager._get_relationships(self.ts.supp_col)

        self.assertEqual(data_col1_rels, [])
        self.assertEqual(supp_col_rels, [])

    def test_remove_relationship_by_str(self):
        """Test that can remove a relationship using column string name, and it removes it from both directions."""
        self.ts.data_col1.add_relationship("supp_col")
        self.ts.data_col1.remove_relationship("supp_col")
        data_col1_rels = self.ts._relationship_manager._get_relationships(self.ts.data_col1)
        supp_col_rels = self.ts._relationship_manager._get_relationships(self.ts.supp_col)

        self.assertEqual(data_col1_rels, [])
        self.assertEqual(supp_col_rels, [])

    def test_remove_relationship_leaves_other_relationships(self):
        """Test that removing one relationship maintains other relationships."""
        self.ts.data_col1.add_relationship("supp_col")
        self.ts.data_col1.add_relationship("flag_col")
        self.ts.data_col1.remove_relationship("supp_col")
        data_col1_rels = self.ts._relationship_manager._get_relationships(self.ts.data_col1)
        supp_col_rels = self.ts._relationship_manager._get_relationships(self.ts.supp_col)
        flag_col_rels = self.ts._relationship_manager._get_relationships(self.ts.flag_col)

        self.assertEqual(len(data_col1_rels), 1)
        self.assertEqual(data_col1_rels[0].other_column, self.ts.flag_col)
        self.assertEqual(len(supp_col_rels), 0)
        self.assertEqual(len(flag_col_rels), 1)
