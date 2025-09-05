import unittest
from datetime import date, datetime
from typing import Any
from unittest.mock import Mock, patch

import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal, assert_series_equal

from time_stream.base import TimeSeries
from time_stream.columns import DataColumn, FlagColumn, PrimaryTimeColumn, SupplementaryColumn, TimeSeriesColumn
from time_stream.exceptions import AggregationPeriodError, ColumnNotFoundError, ColumnTypeError, MetadataError
from time_stream.period import Period


class TestInitSupplementaryColumn(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "data_column": [1, 2, 3],
            "supp_column1": ["a", "b", "c"],
            "supp_column2": ["x", "y", "z"],
        }
    )

    @parameterized.expand(
        [
            ("int", "new_int_col", 5, pl.Int32),
            ("float", "new_float_col", 3.14, pl.Float64),
            ("str", "new_str_col", "test", pl.String),
            ("none", "new_null_col", None, pl.Null),
        ]
    )
    def test_init_supplementary_column_with_single_value(
        self, _: str, new_col_name: str, new_col_value: list, new_col_type: pl.DataType
    ) -> None:
        """Test initialising a supplementary column with a single value (None, int, float or str)."""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, [new_col_value] * len(self.df), new_col_type))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand(
        [
            ("int_int", "new_int_col", 5, pl.Int32),
            ("int_float", "new_float_col", 3, pl.Float64),
            ("float_int", "new_int_col", 3.4, pl.Int32),  # Expect this to convert to 3
            ("none_float", "new_float_col", None, pl.Float64),
            ("none_string", "new_str_col", None, pl.String),
            ("str_int", "new_int_col", "5", pl.Int32),
            ("str_float", "new_float_col", "3.5", pl.Float64),
            ("float_string", "new_null_col", 3.4, pl.String),
        ]
    )
    def test_init_supplementary_column_with_single_value_and_dtype(
        self, _: str, new_col_name: str, new_col_value: list, dtype: pl.DataType
    ) -> None:
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, [new_col_value] * len(self.df)).cast(dtype))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand(
        [
            ("str_int", "new_int_col", "test", pl.Int32),
            ("str_float", "new_float_col", "test", pl.Float64),
        ]
    )
    def test_init_supplementary_column_with_single_value_and_bad_dtype(
        self, _: str, new_col_name: str, new_col_value: list, dtype: pl.DataType
    ) -> None:
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(pl.exceptions.InvalidOperationError):
            ts.init_supplementary_column(new_col_name, new_col_value, dtype)

    @parameterized.expand(
        [
            ("int_iterable", "new_int_iter_col", [5, 6, 7]),
            ("float_iterable", "new_float_iter_col", [1.1, 1.2, 1.3]),
            ("str_iterable", "new_str_iter_col", ["a", "b", "c"]),
            ("none_iterable", "new_null_iter_col", [None, None, None]),
        ]
    )
    def test_init_supp_column_with_iterable(self, _: str, new_col_name: str, new_col_value: list) -> None:
        """Test initialising a supplementary column with an iterable."""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, new_col_value))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand(
        [("too_short", "new_too_short_iter", [5, 6]), ("too_long", "new_too_long_iter", [5, 6, 7, 8, 9, 10])]
    )
    def test_init_supp_column_with_wrong_len_iterable_raises_error(
        self, _: str, new_col_name: str, new_col_value: list
    ) -> None:
        """Test initialising a supplementary column with an iterable of the wrong length raises an error."""
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(pl.ShapeError):
            ts.init_supplementary_column(new_col_name, new_col_value)

    @parameterized.expand(
        [
            ("int_int", "new_int_col", [5, 6, 7], pl.Int32),
            ("int_float", "new_float_col", [3, 4, 5], pl.Float64),
            ("none_float", "new_float_col", [None, None, None], pl.Float64),
            ("str_int", "new_int_col", ["5", "6", "7"], pl.Int32),
            ("none_string", "new_str_col", [None, None, None], pl.String),
        ]
    )
    def test_init_supp_column_with_iterable_and_dtype(
        self, _: str, new_col_name: str, new_col_value: list, dtype: pl.DataType
    ) -> None:
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        expected_df = ts.df.with_columns(pl.Series(new_col_name, new_col_value).cast(dtype))

        pl.testing.assert_frame_equal(ts.df, expected_df)
        self.assertIn(new_col_name, ts.supplementary_columns)

    @parameterized.expand(
        [
            ("str_int", "new_int_col", ["t1", "t2", "t3"], pl.Int32),
            ("str_float", "new_float_col", ["t1", "t2", "t3"], pl.Float64),
            ("mix_float", "new_float_col", [4.5, "3", None], pl.Float64),
        ]
    )
    def test_init_supp_column_with_iterable_and_bad_dtype(
        self, name: str, new_col_name: str, new_col_value: list, dtype: pl.DataType
    ) -> None:
        """Test initialising a supplementary column with a single value set to sensible dtype"""
        ts = TimeSeries(self.df, time_name="time")
        if name == "mix_float":
            # Special case where mixed types cause TypeError
            with self.assertRaises(TypeError):
                ts.init_supplementary_column(new_col_name, new_col_value, dtype)
        else:
            with self.assertRaises(pl.exceptions.InvalidOperationError):
                ts.init_supplementary_column(new_col_name, new_col_value, dtype)


class TestSetSupplementaryColumns(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "data_column": [1, 2, 3],
            "supp_column1": ["a", "b", "c"],
            "supp_column2": ["x", "y", "z"],
        }
    )

    def test_no_supplementary_columns(self) -> None:
        """Test that a TimeSeries object is initialised without any supplementary columns set."""
        ts = TimeSeries(self.df, time_name="time")
        expected_data_columns = {
            "data_column": DataColumn("data_column", ts),
            "supp_column1": DataColumn("supp_column1", ts),
            "supp_column2": DataColumn("supp_column2", ts),
        }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, {})

    def test_empty_list(self) -> None:
        """Test that an empty list behaves the same as no list sent"""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column([])
        expected_data_columns = {
            "data_column": DataColumn("data_column", ts),
            "supp_column1": DataColumn("supp_column1", ts),
            "supp_column2": DataColumn("supp_column2", ts),
        }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, {})

    def test_single_supplementary_column(self) -> None:
        """Test that a single supplementary column is set correctly."""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column(["supp_column1"])
        expected_data_columns = {
            "data_column": DataColumn("data_column", ts),
            "supp_column2": DataColumn("supp_column2", ts),
        }
        expected_supp_columns = {
            "supp_column1": SupplementaryColumn("supp_column1", ts),
        }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)

    def test_multiple_supplementary_column(self) -> None:
        """Test that multiple supplementary columns are set correctly."""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column(["supp_column1", "supp_column2"])
        expected_data_columns = {
            "data_column": DataColumn("data_column", ts),
        }
        expected_supp_columns = {
            "supp_column1": SupplementaryColumn("supp_column1", ts),
            "supp_column2": SupplementaryColumn("supp_column2", ts),
        }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)

    @parameterized.expand(
        [
            ("One bad", ["data_column", "supp_column1", "supp_column2", "non_col"]),
            ("multi bad", ["data_column", "bad_col", "non_col"]),
            ("All bad", ["bad_col", "non_col"]),
        ]
    )
    def test_bad_supplementary_columns(self, _: str, supplementary_columns: list) -> None:
        """Test that error raised for supplementary columns specified that are not in df"""
        ts = TimeSeries(self.df, time_name="time")
        with self.assertRaises(KeyError):
            ts.set_supplementary_column(supplementary_columns)

    def test_appending_supplementary_column(self) -> None:
        """Test that adding supplementary columns maintains existing supplementary columns."""
        ts = TimeSeries(self.df, time_name="time")
        ts.set_supplementary_column(["supp_column1"])
        ts.set_supplementary_column(["supp_column2"])
        expected_data_columns = {
            "data_column": DataColumn("data_column", ts),
        }
        expected_supp_columns = {
            "supp_column1": SupplementaryColumn("supp_column1", ts),
            "supp_column2": SupplementaryColumn("supp_column2", ts),
        }
        self.assertEqual(ts.data_columns, expected_data_columns)
        self.assertEqual(ts.supplementary_columns, expected_supp_columns)


class TestSortTime(unittest.TestCase):
    def test_sort_random_dates(self) -> None:
        """Test that random dates are sorted appropriately"""
        times = [date(1990, 1, 1), date(2019, 5, 8), date(1967, 12, 25), date(2059, 8, 12)]
        expected = pl.Series("time", [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)])

        with patch.object(TimeSeries, "_setup"):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
            ts._setup_columns()
            ts.sort_time()
            assert_series_equal(ts.time_column.data, expected)

    def test_sort_sorted_dates(self) -> None:
        """Test that already sorted dates are maintained"""
        times = [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)]
        expected = pl.Series("time", times)

        with patch.object(TimeSeries, "_setup"):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
            ts._setup_columns()
            ts.sort_time()
            assert_series_equal(ts.time_column.data, expected)

    def test_sort_times(self) -> None:
        """Test that times are sorted appropriately"""
        times = [
            datetime(2024, 1, 1, 12, 59),
            datetime(2024, 1, 1, 12, 55),
            datetime(2024, 1, 1, 12, 18),
            datetime(2024, 1, 1, 1, 5),
        ]

        expected = pl.Series(
            "time",
            [
                datetime(2024, 1, 1, 1, 5),
                datetime(2024, 1, 1, 12, 18),
                datetime(2024, 1, 1, 12, 55),
                datetime(2024, 1, 1, 12, 59),
            ],
        )
        with patch.object(TimeSeries, "_setup"):
            ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
            ts._setup_columns()
            ts.sort_time()
            assert_series_equal(ts.time_column.data, expected)


class TestRemoveMissingColumns(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    metadata = {"col1": {"description": "meta1"}, "col2": {"description": "meta2"}, "col3": {"description": "meta3"}}

    def test_no_columns_removed(self) -> None:
        """Test that no columns are removed when all columns are present in the new DataFrame."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        ts._remove_missing_columns(ts._df, self.df)
        ts._df = self.df

        self.assertEqual(list(ts.columns.keys()), ["col1", "col2", "col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), ["col1"])
        self.assertEqual(list(ts.data_columns.keys()), ["col2", "col3"])
        self.assertEqual(ts.col1.metadata(), self.metadata["col1"])
        self.assertEqual(ts.col2.metadata(), self.metadata["col2"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

    def test_single_data_column_removed(self) -> None:
        """Test that single data column is removed correctly."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        new_df = self.df.drop("col2")
        ts._remove_missing_columns(ts._df, new_df)
        ts._df = new_df

        self.assertEqual(list(ts.columns.keys()), ["col1", "col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), ["col1"])
        self.assertEqual(list(ts.data_columns.keys()), ["col3"])
        self.assertEqual(ts.col1.metadata(), self.metadata["col1"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

        # try to access removed columns
        with self.assertRaises(AttributeError):
            _ = ts.col2

    def test_single_supplementary_column_removed(self) -> None:
        """Test that single supplementary column is removed correctly."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        new_df = self.df.drop("col1")
        ts._remove_missing_columns(ts._df, new_df)
        ts._df = new_df

        self.assertEqual(list(ts.columns.keys()), ["col2", "col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), [])
        self.assertEqual(list(ts.data_columns.keys()), ["col2", "col3"])
        self.assertEqual(ts.col2.metadata(), self.metadata["col2"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

        # try to access removed columns
        with self.assertRaises(AttributeError):
            _ = ts.col1

    def test_multiple_columns_removed(self) -> None:
        """Test that multiple columns are removed correctly."""
        ts = TimeSeries(self.df, time_name="time", supplementary_columns=["col1"], column_metadata=self.metadata)

        new_df = self.df.drop(["col1", "col2"])
        ts._remove_missing_columns(ts._df, new_df)
        ts._df = new_df

        self.assertEqual(list(ts.columns.keys()), ["col3"])
        self.assertEqual(list(ts.supplementary_columns.keys()), [])
        self.assertEqual(list(ts.data_columns.keys()), ["col3"])
        self.assertEqual(ts.col3.metadata(), self.metadata["col3"])

        # try to access removed columns
        with self.assertRaises(AttributeError):
            _ = ts.col1
        with self.assertRaises(AttributeError):
            _ = ts.col2


class TestAddNewColumns(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    ts = TimeSeries(df, time_name="time")

    def test_add_new_columns(self) -> None:
        """Test that new columns are correctly added as DataColumns."""
        new_df = self.df.with_columns(pl.Series("col4", [1.1, 2.2, 3.3]))
        with patch.object(TimeSeriesColumn, "_validate_name", lambda *args, **kwargs: None):
            self.ts._add_new_columns(self.ts._df, new_df)

        self.assertIn("col4", self.ts.columns)
        self.assertIsInstance(self.ts.col4, DataColumn)

    def test_no_changes_when_no_new_columns(self) -> None:
        """Test that does nothing if there are no new columns."""
        original_columns = self.ts._columns.copy()
        self.ts._add_new_columns(self.ts._df, self.df)

        self.assertEqual(original_columns, self.ts._columns)

    def test_columns_added_to_relationship_manager(self) -> None:
        """Test that new columns are initialised in the relationship manager, with no relationships"""
        new_df = self.df.with_columns(pl.Series("col4", [1.1, 2.2, 3.3]))
        with patch.object(TimeSeriesColumn, "_validate_name", lambda *args, **kwargs: None):
            self.ts._add_new_columns(self.ts._df, new_df)

        self.assertIn("col4", self.ts.columns)
        self.assertIn("col4", self.ts._relationship_manager._relationships)
        self.assertEqual(self.ts._relationship_manager._relationships["col4"], set())


class TestSelectColumns(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    df = pl.DataFrame({"time": times, "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_select_single_column(self) -> None:
        """Test selecting a single of column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        expected = TimeSeries(
            self.df.select(["time", "col1"]), time_name="time", column_metadata={"col1": self.metadata["col1"]}
        )
        result = ts.select(["col1"])
        self.assertEqual(result, expected)

    def test_select_multiple_columns(self) -> None:
        """Test selecting multiple columns."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        expected = TimeSeries(
            self.df.select(["time", "col1", "col2"]),
            time_name="time",
            column_metadata={"col1": self.metadata["col1"], "col2": self.metadata["col2"]},
        )
        result = ts.select(["col1", "col2"])
        self.assertEqual(result, expected)

    def test_select_no_columns_raises_error(self) -> None:
        """Test selecting no columns raises error."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            ts.select([])
        self.assertEqual("No columns specified.", str(err.exception))

    def test_select_nonexistent_column(self) -> None:
        """Test selecting a column that does not exist raises error."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            ts.select(["nonexistent_column"])
        self.assertEqual("Columns not found in dataframe: ['nonexistent_column']", str(err.exception))

    def test_select_existing_and_nonexistent_column(self) -> None:
        """Test selecting a column that does not exist, alongside existing columns, still raises error"""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            ts.select(["col1", "col2", "nonexistent_column"])
        self.assertEqual("Columns not found in dataframe: ['nonexistent_column']", str(err.exception))

    def test_select_column_doesnt_mutate_original_ts(self) -> None:
        """When selecting a column, the original ts should be unchanged"""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        original_df = ts.df

        expected = TimeSeries(
            self.df.select(["time", "col1"]), time_name="time", column_metadata={"col1": self.metadata["col1"]}
        )
        col1_ts = ts.select(["col1"])
        self.assertEqual(col1_ts, expected)
        assert_frame_equal(ts.df, original_df)

        expected = TimeSeries(
            self.df.select(["time", "col2"]), time_name="time", column_metadata={"col2": self.metadata["col2"]}
        )
        col2_ts = ts.select(["col2"])
        self.assertEqual(col2_ts, expected)
        assert_frame_equal(ts.df, original_df)


class TestGetattr(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    column_metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}

    def test_access_time_column(self) -> None:
        """Test accessing the time column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.column_metadata)
        result = ts.time
        expected = PrimaryTimeColumn("time", ts)
        self.assertEqual(result, expected)

    def test_access_data_column(self) -> None:
        """Test accessing a data column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.column_metadata)
        result = ts.col1
        expected = DataColumn("col1", ts, self.column_metadata["col1"])
        self.assertEqual(result, expected)

    @parameterized.expand(
        [
            ("int", "site_id", 1234),
            ("str", "network", "FDRI"),
            ("dict", "some_info", {1: "a", 2: "b", 3: "c"}),
        ]
    )
    def test_access_metadata_key(self, _: str, key: str, expected: Any) -> None:
        """Test accessing metadata key."""
        ts = TimeSeries(self.df, time_name="time", metadata=self.metadata)
        result = ts.__getattr__(key)
        self.assertEqual(result, expected)

    def test_access_nonexistent_attribute(self) -> None:
        """Test accessing metadata key that doesn't exist"""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.column_metadata)
        with self.assertRaises(MetadataError):
            _ = ts.col1.key0


class TestGetItem(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    def test_access_time_column(self) -> None:
        """Test accessing the time column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        result = ts["time"]
        expected = PrimaryTimeColumn("time", ts)
        self.assertEqual(result, expected)

    def test_access_data_column(self) -> None:
        """Test accessing a data column."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        result = ts["col1"]
        expected = TimeSeries(
            self.df.select(["time", "col1"]), time_name="time", column_metadata={"col1": self.metadata["col1"]}
        )
        self.assertEqual(result, expected)

    def test_access_multiple_data_columns(self) -> None:
        """Test accessing multiple data columns."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        result = ts[["col1", "col2"]]
        expected = TimeSeries(
            self.df.select(["time", "col1", "col2"]),
            time_name="time",
            column_metadata={"col1": self.metadata["col1"], "col2": self.metadata["col2"]},
        )
        self.assertEqual(result, expected)

    def test_non_existent_column(self) -> None:
        """Test accessing non-existent data column raises error."""
        ts = TimeSeries(self.df, time_name="time", column_metadata=self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            _ = ts["col0"]
        self.assertEqual("Columns not found in dataframe: ['col0']", str(err.exception))


class TestSetupColumns(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "data_col": [10, 20, 30],
            "supp_col": ["A", "B", "C"],
            "flag_col": [1, 2, 3],
        }
    )
    flag_systems = {"example_flag_system": {"OK": 1, "WARNING": 2}}

    def test_valid_setup_columns(self) -> None:
        """Test that valid columns are correctly classified."""
        ts = TimeSeries(
            df=self.df,
            time_name="time",
            supplementary_columns=["supp_col"],
            flag_columns={"flag_col": "example_flag_system"},
            flag_systems=self.flag_systems,
        )

        self.assertIsInstance(ts.time_column, PrimaryTimeColumn)
        self.assertIsInstance(ts.columns["supp_col"], SupplementaryColumn)
        self.assertIsInstance(ts.columns["flag_col"], FlagColumn)
        self.assertIsInstance(ts.columns["data_col"], DataColumn)

    def test_missing_supplementary_column_raises_error(self) -> None:
        """Test that an error raised when supplementary columns do not exist."""
        with self.assertRaises(ColumnNotFoundError) as err:
            TimeSeries(
                df=self.df,
                time_name="time",
                supplementary_columns=["missing_col"],
                flag_columns={"flag_col": "example_flag_system"},
                flag_systems=self.flag_systems,
            )
        self.assertEqual("Columns not found in dataframe: ['missing_col']", str(err.exception))

    def test_missing_flag_column_raises_error(self) -> None:
        """Test that an error raised when flag columns do not exist."""
        with self.assertRaises(ColumnNotFoundError) as err:
            TimeSeries(
                df=self.df,
                time_name="time",
                supplementary_columns=["supp_col"],
                flag_columns={"missing_col": "example_flag_system"},
                flag_systems=self.flag_systems,
            )
        self.assertEqual("Columns not found in dataframe: ['missing_col']", str(err.exception))


class TestTimeColumn(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )

    def test_valid_time_column(self) -> None:
        """Test that time_column correctly returns the PrimaryTimeColumn instance."""
        ts = TimeSeries(self.df, time_name="time")
        self.assertIsInstance(ts.time_column, PrimaryTimeColumn)
        self.assertEqual(ts.time_column.name, "time")

    def test_no_time_column_raises_error(self) -> None:
        """Test that error is raised if no primary time column is found."""
        ts = TimeSeries(self.df, time_name="time")
        with patch.object(ts, "_columns", {}):  # Simulate missing columns
            with self.assertRaises(ColumnNotFoundError) as err:
                _ = ts.time_column
            self.assertEqual("No single primary time column found.", str(err.exception))

    def test_multiple_time_columns_raises_error(self) -> None:
        """Test that error is raised if multiple primary time columns exist."""
        ts = TimeSeries(self.df, time_name="time")
        with patch.object(TimeSeriesColumn, "_validate_name", lambda *args, **kwargs: None):
            with patch.object(
                ts,
                "_columns",
                {
                    "time1": PrimaryTimeColumn("time1", ts),
                    "time2": PrimaryTimeColumn("time2", ts),
                },
            ):
                with self.assertRaises(ColumnNotFoundError) as err:
                    _ = ts.time_column
                self.assertEqual("No single primary time column found.", str(err.exception))


class TestTimeSeriesEquality(unittest.TestCase):
    def setUp(self) -> None:
        """Set up multiple TimeSeries objects for testing."""
        self.df_original = pl.DataFrame(
            {"time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)], "value": [10, 20, 30]}
        )
        self.df_same = pl.DataFrame(
            {"time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)], "value": [10, 20, 30]}
        )
        self.df_different_values = pl.DataFrame(
            {"time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)], "value": [100, 200, 300]}
        )
        self.df_different_times = pl.DataFrame(
            {"time": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)], "value": [10, 20, 30]}
        )

        self.flag_systems_1 = {"quality_flags": {"OK": 1, "WARNING": 2}}
        self.flag_systems_2 = {"quality_flags": {"NOT_OK": 4, "ERROR": 8}}

        self.ts_original = TimeSeries(
            df=self.df_original,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_1,
        )

    def test_equal_timeseries(self) -> None:
        """Test that two identical TimeSeries objects are considered equal."""
        ts_same = TimeSeries(
            df=self.df_same,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_1,
        )
        self.assertEqual(self.ts_original, ts_same)

    def test_different_data_values(self) -> None:
        """Test that TimeSeries objects with different data values are not equal."""
        ts_different_df = TimeSeries(
            df=self.df_different_values,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_1,
        )
        self.assertNotEqual(self.ts_original, ts_different_df)

    def test_different_time_values(self) -> None:
        """Test that TimeSeries objects with different time values are not equal."""
        ts_different_times = TimeSeries(
            df=self.df_different_times,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_1,
        )
        self.assertNotEqual(self.ts_original, ts_different_times)

    def test_different_time_name(self) -> None:
        """Test that TimeSeries objects with different time column name are not equal."""
        ts_different_time_name = TimeSeries(
            df=self.df_original.rename({"time": "timestamp"}),
            time_name="timestamp",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_1,
        )
        self.assertNotEqual(self.ts_original, ts_different_time_name)

    def test_different_periodicity(self) -> None:
        """Test that TimeSeries objects with different periodicity are not equal."""
        ts_different_periodicity = TimeSeries(
            df=self.df_original,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_months(1),
            flag_systems=self.flag_systems_1,
        )
        self.assertNotEqual(self.ts_original, ts_different_periodicity)

    def test_different_resolution(self) -> None:
        """Test that TimeSeries objects with different resolution are not equal."""
        ts_different_resolution = TimeSeries(
            df=self.df_original,
            time_name="time",
            resolution=Period.of_seconds(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_1,
        )
        self.assertNotEqual(self.ts_original, ts_different_resolution)

    def test_different_flag_systems(self) -> None:
        """Test that TimeSeries objects with different flag systems are not equal."""
        ts_different_flags = TimeSeries(
            df=self.df_original,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
            flag_systems=self.flag_systems_2,
        )
        self.assertNotEqual(self.ts_original, ts_different_flags)


class TestColumnMetadata(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }
    ts = TimeSeries(df, time_name="time", column_metadata=metadata)

    def test_retrieve_all_metadata(self) -> None:
        """Test retrieving all metadata for all columns."""
        self.assertEqual(self.ts.column_metadata(), self.metadata)

    def test_retrieve_metadata_for_single_column(self) -> None:
        """Test retrieving metadata for a single column."""
        expected = {"col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"}}
        result = self.ts.column_metadata("col1")
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_columns(self) -> None:
        """Test retrieving metadata for multiple columns."""
        expected = {
            "col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"},
            "col2": {"key1": "2", "key2": "20", "key3": "200"},
        }
        result = self.ts.column_metadata(["col1", "col2"])
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_specific_key(self) -> None:
        """Test retrieving a specific metadata key."""
        expected = {"col1": {"key1": "1"}}
        result = self.ts.column_metadata("col1", "key1")
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self) -> None:
        """Test retrieving multiple metadata keys."""
        expected = {"col1": {"key1": "1", "key3": "100"}}
        result = self.ts.column_metadata("col1", ["key1", "key3"])
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_column(self) -> None:
        """Test that a KeyError is raised when requesting a non-existent column."""
        with self.assertRaises(KeyError):
            self.ts.column_metadata("nonexistent_column")

    def test_retrieve_metadata_for_nonexistent_key_single_column(self) -> None:
        """Test that error raised when requesting a non-existent metadata key for an existing single column"""
        with self.assertRaises(MetadataError) as err:
            self.ts.column_metadata("col1", "nonexistent_key")
        self.assertEqual("Metadata key(s) ['nonexistent_key'] not found in any column.", str(err.exception))

    def test_retrieve_metadata_for_nonexistent_key_in_one_column(self) -> None:
        """Test that dict returned when requesting a metadata key exists in one column, but not another"""
        expected = {"col1": {"key4": "1000"}, "col2": {"key4": None}}
        result = self.ts.column_metadata(["col1", "col2"], "key4")
        self.assertEqual(result, expected)


class TestMetadata(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}
    ts = TimeSeries(df, time_name="time", metadata=metadata)

    def test_retrieve_all_metadata(self) -> None:
        """Test retrieving all metadata"""
        result = self.ts.metadata()
        self.assertEqual(result, self.metadata)

    @parameterized.expand(
        [
            ("int", "site_id"),
            ("str", "network"),
            ("dict", "some_info"),
        ]
    )
    def test_retrieve_metadata_for_specific_key(self, _: str, key: str) -> None:
        """Test retrieving a specific metadata key."""
        result = self.ts.metadata(key)
        expected = {key: self.metadata[key]}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self) -> None:
        """Test retrieving multiple metadata keys."""
        result = self.ts.metadata(["site_id", "network"])
        expected = {"site_id": 1234, "network": "FDRI"}
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_key_raises_error(self) -> None:
        """Test that an error is raised for a non-existent key."""
        with self.assertRaises(MetadataError) as err:
            self.ts.metadata("nonexistent_key")
        self.assertEqual("Metadata key 'nonexistent_key' not found", str(err.exception))

    def test_retrieve_metadata_for_nonexistent_key_strict_false(self) -> None:
        """Test that an empty result is returned when strict is false for non-existent key."""
        expected = {"nonexistent_key": None}
        result = self.ts.metadata("nonexistent_key", strict=False)
        self.assertEqual(result, expected)


class TestSetupMetadata(unittest.TestCase):
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )

    def test_setup_metadata_success(self) -> None:
        """Test that metadata entries with keys not in _columns are added successfully."""
        metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}
        with patch.object(TimeSeries, "_setup"):
            ts = TimeSeries(self.df, time_name="time")
            ts._setup_metadata(metadata)
        self.assertEqual(ts._metadata, metadata)

    def test_setup_metadata_conflict(self) -> None:
        """Test that providing a metadata key that conflicts with _columns raises a KeyError."""
        metadata = {"col1": 1234}
        with patch.object(TimeSeries, "_setup"):
            with self.assertRaises(MetadataError) as err:
                ts = TimeSeries(self.df, time_name="time")
                ts._setup_metadata(metadata)
            self.assertEqual("Metadata key 'col1' exists as a Column in the Time Series.", str(err.exception))

    def test_setup_metadata_empty(self) -> None:
        """Test that passing an empty metadata dictionary leaves _metadata unchanged."""
        metadata = {}
        with patch.object(TimeSeries, "_setup"):
            ts = TimeSeries(self.df, time_name="time")
            ts._setup_metadata(metadata)
        self.assertEqual(ts._metadata, {})


class TestGetFlagSystemColumn(unittest.TestCase):
    def setUp(self) -> None:
        df = pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "data_col": [10, 20, 30],
                "supp_col": ["A", "B", "C"],
                "flag_col": [1, 2, 3],
            }
        )
        flag_systems = {"example_flag_system": {"OK": 1, "WARNING": 2}}
        self.ts = TimeSeries(
            df=df,
            time_name="time",
            supplementary_columns=["supp_col"],
            flag_columns={"flag_col": "example_flag_system"},
            flag_systems=flag_systems,
        )
        self.ts.data_col.add_relationship(["flag_col"])

    def test_data_column_not_exist(self) -> None:
        """Test return when the specified data column doesn't exist"""
        data_column = "data_col_not_exist"
        flag_system = "example_flag_system"
        with self.assertRaises(ColumnNotFoundError) as err:
            self.ts.get_flag_system_column(data_column, flag_system)
        self.assertEqual("Data Column 'data_col_not_exist' not found.", str(err.exception))

    @parameterized.expand([("supp_col", SupplementaryColumn), ("flag_col", FlagColumn)])
    def test_data_column_not_a_data_column(self, data_column: str, column_type: type[TimeSeriesColumn]) -> None:
        """Test return when the specified data column isn't actually a data column"""
        flag_system = "example_flag_system"
        with self.assertRaises(ColumnTypeError) as err:
            self.ts.get_flag_system_column(data_column, flag_system)
        self.assertEqual(f"Column '{data_column}' is type {column_type}. Should be a data column.", str(err.exception))

    def test_get_expected_flag_column(self) -> None:
        """Test expected flag column returned for valid flag system"""
        data_column = "data_col"
        flag_system = "example_flag_system"
        expected = self.ts.flag_col
        result = self.ts.get_flag_system_column(data_column, flag_system)
        self.assertEqual(result, expected)


class TestAggregate(unittest.TestCase):
    def setUp(self) -> None:
        df = pl.DataFrame(
            {"time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], "data_col": [10, 20, 30]}
        )
        self.ts = TimeSeries(df=df, time_name="time", resolution="P1D", periodicity="P1D")

    def test_validate_aggregation_period_non_epoch_agnostic(self) -> None:
        """Test validation fails for non-epoch agnostic periods."""
        period = Mock()
        period.is_epoch_agnostic.return_value = False

        with self.assertRaises(AggregationPeriodError) as err:
            self.ts.aggregate(period, "mean", "data_col")
        self.assertEqual(f"Non-epoch agnostic aggregation periods are not supported: '{period}'.", str(err.exception))

    def test_validate_aggregation_period_not_subperiod(self) -> None:
        """Test validation fails when the aggregation period is not a subperiod."""
        period = Mock()
        period.is_epoch_agnostic.return_value = True
        period.is_subperiod_of.return_value = False

        self.ts._periodicity = period

        with self.assertRaises(AggregationPeriodError) as err:
            self.ts.aggregate(period, "mean", "data_col")
        self.assertEqual(
            f"Incompatible aggregation period '{period}' with TimeSeries periodicity '{self.ts.periodicity}'. "
            f"TimeSeries periodicity must be a subperiod of the aggregation period.",
            str(err.exception),
        )
