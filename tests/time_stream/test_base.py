import unittest
from datetime import date, datetime
from typing import Any

import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal, assert_frame_not_equal, assert_series_equal

from time_stream.base import TimeSeries
from time_stream.bitwise import BitwiseFlag
from time_stream.exceptions import ColumnNotFoundError, MetadataError
from time_stream.flag_manager import FlagColumn
from time_stream.period import Period


class TestSortTime(unittest.TestCase):
    def test_sort_random_dates(self) -> None:
        """Test that random dates are sorted appropriately"""
        times = [date(1990, 1, 1), date(2019, 5, 8), date(1967, 12, 25), date(2059, 8, 12)]
        expected = pl.Series("time", [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)])

        ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
        ts.sort_time()
        assert_series_equal(ts.df["time"], expected)

    def test_sort_sorted_dates(self) -> None:
        """Test that already sorted dates are maintained"""
        times = [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)]
        expected = pl.Series("time", times)

        ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
        ts.sort_time()
        assert_series_equal(ts.df["time"], expected)

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
        ts = TimeSeries(pl.DataFrame({"time": times}), time_name="time")
        ts.sort_time()
        assert_series_equal(ts.df["time"], expected)


class TestSelectColumns(unittest.TestCase):
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    df = pl.DataFrame({"time": times, "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    @parameterized.expand([("as_str", "col1"), ("as_list", ["col1"])])
    def test_select_single_column(self, _: str, col: str | list) -> None:
        """Test selecting a single of column."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )
        result = ts.select(col)
        self.assertEqual(result, expected)

    def test_select_multiple_columns(self) -> None:
        """Test selecting multiple columns."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        expected = TimeSeries(self.df.select(["time", "col1", "col2"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"], "col2": self.metadata["col2"]}
        )

        result = ts.select(["col1", "col2"])
        self.assertEqual(result, expected)

    def test_select_no_columns_raises_error(self) -> None:
        """Test selecting no columns raises error."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            ts.select([])
        self.assertEqual("No columns specified.", str(err.exception))

    def test_select_nonexistent_column(self) -> None:
        """Test selecting a column that does not exist raises error."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            ts.select(["nonexistent_column"])
        self.assertEqual("Columns not found in dataframe: ['nonexistent_column']", str(err.exception))

    def test_select_existing_and_nonexistent_column(self) -> None:
        """Test selecting a column that does not exist, alongside existing columns, still raises error"""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            ts.select(["col1", "col2", "nonexistent_column"])
        self.assertEqual("Columns not found in dataframe: ['nonexistent_column']", str(err.exception))

    def test_select_column_doesnt_mutate_original_ts(self) -> None:
        """When selecting a column, the original ts should be unchanged"""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        original_df = ts.df

        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )

        col1_ts = ts.select(["col1"])
        self.assertEqual(col1_ts, expected)
        assert_frame_equal(ts.df, original_df)

        expected = TimeSeries(self.df.select(["time", "col2"]), time_name="time").with_column_metadata(
            {"col2": self.metadata["col2"]}
        )

        col2_ts = ts.select(["col2"])
        self.assertEqual(col2_ts, expected)
        assert_frame_equal(ts.df, original_df)

    def test_select_column_with_flags(self) -> None:
        ts = TimeSeries(self.df, time_name="time").with_flag_system("system", {"A": 1, "B": 2, "C": 4})
        ts.init_flag_column("col1", "system", "flag_col")

        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time").with_flag_system(
            "system", {"A": 1, "B": 2, "C": 4}
        )
        expected.init_flag_column("col1", "system", "flag_col")

        result = ts.select(["col1"])
        self.assertEqual(result, expected)


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
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        result = ts["time"]
        expected = TimeSeries(self.df[["time"]], time_name="time")
        self.assertEqual(result, expected)

    def test_access_data_column(self) -> None:
        """Test accessing a data column."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        result = ts["col1"]
        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )
        self.assertEqual(result, expected)

    def test_access_multiple_data_columns(self) -> None:
        """Test accessing multiple data columns."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        result = ts[["col1", "col2"]]
        expected = TimeSeries(
            self.df.select(["time", "col1", "col2"]),
            time_name="time",
        ).with_column_metadata({"col1": self.metadata["col1"], "col2": self.metadata["col2"]})
        self.assertEqual(result, expected)

    def test_non_existent_column(self) -> None:
        """Test accessing non-existent data column raises error."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        with self.assertRaises(ColumnNotFoundError) as err:
            _ = ts["col0"]
        self.assertEqual("Columns not found in dataframe: ['col0']", str(err.exception))


class TestColumnMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
            }
        )
        self.metadata = {
            "col1": {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"},
            "col2": {"key1": "2", "key2": "20", "key3": "200"},
        }
        self.ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)

    def test_all_columns_have_metadata(self) -> None:
        """Test on the init that all columns get initialised with an empty dict if no metadata provided"""
        self.assertEqual(set(self.df.columns), set(self.ts.column_metadata.keys()))
        self.assertEqual(self.ts.column_metadata["col1"], {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"})
        self.assertEqual(self.ts.column_metadata["col2"], {"key1": "2", "key2": "20", "key3": "200"})
        self.assertEqual(self.ts.column_metadata["col3"], {})

    def test_retrieve_metadata_for_single_column(self) -> None:
        """Test retrieving metadata for a single column."""
        expected = {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"}
        result = self.ts.column_metadata["col1"]
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_specific_key(self) -> None:
        """Test retrieving a specific metadata key."""
        expected = "1"
        result = self.ts.column_metadata["col1"]["key1"]
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_column(self) -> None:
        """Test that a KeyError is raised when requesting a non-existent column."""
        with self.assertRaises(KeyError):
            _ = self.ts.column_metadata["nonexistent_column"]

    def test_retrieve_metadata_for_nonexistent_key_single_column_strict(self) -> None:
        """Test that error raised when requesting a non-existent metadata key for an existing single column"""
        with self.assertRaises(KeyError):
            _ = self.ts.column_metadata["col1"]["nonexistent_key"]

    def test_set_non_dict_raises_error(self) -> None:
        """Test that error raised when setting a non-dict object as the metadata for a column"""
        with self.assertRaises(MetadataError):
            _ = self.ts.column_metadata["col1"] = 123  # noqa - expecting typehint error

    def test_set_column_metadata(self) -> None:
        """Test that we can set the column metadata object directly. Existing metadata should be maintained"""
        column_metadata = {"col1": {"key": "value"}}
        self.ts.column_metadata = column_metadata
        column_metadata["col2"] = {"key1": "2", "key2": "20", "key3": "200"}
        column_metadata["col3"] = {}
        column_metadata["time"] = {}
        self.assertEqual(self.ts.column_metadata, column_metadata)

    def test_set_column_metadata_column_error(self) -> None:
        """Test that directly setting column metadata with incorrect format raises error."""
        column_metadata = {"missing_col": {"key": "value"}}
        with self.assertRaises(KeyError):
            self.ts.column_metadata = column_metadata

    def test_set_column_metadata_none(self) -> None:
        """Test that none resets column metadata to empty dicts for each column"""
        self.ts.column_metadata = None
        column_metadata = {"col1": {}, "col2": {}, "col3": {}, "time": {}}
        self.assertEqual(self.ts.column_metadata, column_metadata)

    @parameterized.expand(
        [
            ("int", 1234),
            ("str", "network: FDRI"),
            ("list", ["a", "b", "c"]),
        ]
    )
    def test_set_column_metadata_not_dict_error(self, _: str, metadata: Any) -> None:
        """Test that trying to set the column metadata object to not a dict raises error"""
        with self.assertRaises(MetadataError):
            self.ts.column_metadata = metadata

    @parameterized.expand(
        [
            ("int", 1234),
            ("str", "network: FDRI"),
            ("list", ["a", "b", "c"]),
        ]
    )
    def test_set_nested_column_metadata_not_dict_error(self, _: str, metadata: Any) -> None:
        """Test that trying to set the inner column metadata object to not a dict raises error"""
        with self.assertRaises(MetadataError):
            self.ts.column_metadata = {"col1": metadata}

    def test_clear_column_metadata(self) -> None:
        """Test that removing the column_metadata object sets it back to an empty dicts for each column"""
        del self.ts.column_metadata
        column_metadata = {"col1": {}, "col2": {}, "col3": {}, "time": {}}
        self.assertEqual(self.ts.column_metadata, column_metadata)


class TestMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
            }
        )
        self.metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}
        self.ts = TimeSeries(self.df, time_name="time").with_metadata(self.metadata)

    def test_retrieve_all_metadata(self) -> None:
        """Test retrieving all metadata"""
        result = self.ts.metadata
        self.assertEqual(result, self.metadata)

    @parameterized.expand(
        [
            ("int", "site_id", 1234),
            ("str", "network", "FDRI"),
            ("dict", "some_info", {1: "a", 2: "b", 3: "c"}),
        ]
    )
    def test_retrieve_metadata_for_specific_key(self, _: str, key: str, expected: Any) -> None:
        """Test retrieving a specific metadata key."""
        result = self.ts.metadata[key]
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_key_raises_error(self) -> None:
        """Test that an error is raised for a non-existent key."""
        with self.assertRaises(KeyError):
            _ = self.ts.metadata["nonexistent_key"]

    def test_set_metadata(self) -> None:
        """Test that we can set the metadata object directly"""
        metadata = {"key": "value"}
        self.ts.metadata = metadata
        self.assertEqual(self.ts.metadata, metadata)

    def test_set_metadata_none(self) -> None:
        """Test that none sets to empty dict"""
        self.ts.metadata = None
        self.assertEqual(self.ts.metadata, {})

    @parameterized.expand(
        [
            ("int", 1234),
            ("str", "network: FDRI"),
            ("list", ["a", "b", "c"]),
        ]
    )
    def test_set_metadata_not_dict_error(self, _: str, metadata: Any) -> None:
        """Test that trying to set the metadata object to not a dict raises error"""
        with self.assertRaises(MetadataError):
            self.ts.metadata = metadata

    def test_clear_metadata(self) -> None:
        """Test that removing the metadata object sets it back to an empty dict"""
        del self.ts.metadata
        self.assertEqual(self.ts.metadata, {})


class TestInitFlagColumn(unittest.TestCase):
    def setUp(self) -> None:
        """Set up a mock TimeSeries and FlagManager for testing."""
        self.df = pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "value": [10, 20, 30],
                "existing_flags": [0, 1, 2],
            }
        )
        self.flag_system = BitwiseFlag("quality_control", {"OUT_OF_RANGE": 1, "SPIKE": 2, "LOW_BATTERY": 4})
        self.ts = TimeSeries(df=self.df, time_name="time")
        self.ts.register_flag_system("quality_control", self.flag_system)

    def test_init_flag_column_success(self) -> None:
        """Test initialising a flag column with a valid flag system."""

        # Shouldn't have any flag columns to start with
        self.assertEqual(self.ts._flag_manager._flag_columns, {})

        self.ts.init_flag_column("value", "quality_control", "flag_column")
        self.assertEqual(
            self.ts._flag_manager._flag_columns,
            {"flag_column": FlagColumn("flag_column", "value", self.flag_system)},
        )
        self.assertIn("flag_column", self.ts.flag_columns)

    def test_init_column_adds_to_metadata(self) -> None:
        """Test that a new flag column gets added to the column metadata dict."""
        self.ts.init_flag_column("value", "quality_control", "flag_column")
        expected = {"time": {}, "value": {}, "existing_flags": {}, "flag_column": {}}
        self.assertEqual(self.ts.column_metadata, expected)

    def test_init_flag_column_with_single_value(self) -> None:
        """Test initialising a flag column with a valid flag system and a non-default single value"""

        # Shouldn't have any flag columns to start with
        self.assertEqual(self.ts._flag_manager._flag_columns, {})

        self.ts.init_flag_column("value", "quality_control", "flag_column", 1)
        expected_values = pl.Series("flag_column", [1, 1, 1], dtype=pl.Int64)
        assert_series_equal(self.ts.df["flag_column"], expected_values)

    def test_init_flag_column_with_list_value(self) -> None:
        """Test initialising a flag column with a valid flag system and a non-default list of values"""

        # Shouldn't have any flag columns to start with
        self.assertEqual(self.ts._flag_manager._flag_columns, {})

        self.ts.init_flag_column("value", "quality_control", "flag_column", [1, 2, 4])
        expected_values = pl.Series("flag_column", [1, 2, 4], dtype=pl.Int64)
        assert_series_equal(self.ts.df["flag_column"], expected_values)

    def test_with_no_flag_column_name(self) -> None:
        """Test that a default flag column name is used if not provided."""

        # Shouldn't have any flag columns to start with
        self.assertEqual(self.ts._flag_manager._flag_columns, {})

        self.ts.init_flag_column("value", "quality_control")
        default_name = "value__flag__quality_control"

        self.assertEqual(
            self.ts._flag_manager._flag_columns,
            {default_name: FlagColumn(default_name, "value", self.flag_system)},
        )
        self.assertIn(default_name, self.ts.flag_columns)

    def test_init_no_column_name_adds_to_metadata(self) -> None:
        """Test that a new flag column gets added to the column metadata dict."""
        self.ts.init_flag_column("value", "quality_control")
        expected = {"time": {}, "value": {}, "existing_flags": {}, "value__flag__quality_control": {}}
        self.assertEqual(self.ts.column_metadata, expected)


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

        self.flag_systems_1 = {"OK": 1, "WARNING": 2}
        self.flag_systems_2 = {"NOT_OK": 4, "ERROR": 8}
        self.metadata = {"site": "DUMMY", "organisation": "UKCEH"}

        self.ts_original = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

    def test_equal_timeseries(self) -> None:
        """Test that two identical TimeSeries objects are considered equal."""
        ts_same = (
            TimeSeries(
                df=self.df_same,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertEqual(self.ts_original, ts_same)

    def test_different_data_values(self) -> None:
        """Test that TimeSeries objects with different data values are not equal."""
        ts_different_df = (
            TimeSeries(
                df=self.df_different_values,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_df)

        # Test the expected property has changed
        assert_frame_not_equal(self.ts_original.df, ts_different_df.df)
        self.assertEqual(self.ts_original.time_name, ts_different_df.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_df.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_df.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_df.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_df._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_df.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_df.column_metadata)

    def test_different_time_values(self) -> None:
        """Test that TimeSeries objects with different time values are not equal."""
        ts_different_times = (
            TimeSeries(
                df=self.df_different_times,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_times)

        # Test the expected property has changed
        assert_frame_not_equal(self.ts_original.df, ts_different_times.df)
        self.assertEqual(self.ts_original.time_name, ts_different_times.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_times.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_times.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_times.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_times._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_times.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_times.column_metadata)

    def test_different_time_name(self) -> None:
        """Test that TimeSeries objects with different time column name are not equal."""
        ts_different_time_name = (
            TimeSeries(
                df=self.df_original.rename({"time": "timestamp"}),
                time_name="timestamp",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_time_name)

        # Test the expected property has changed
        assert_frame_not_equal(self.ts_original.df, ts_different_time_name.df)
        self.assertNotEqual(self.ts_original.time_name, ts_different_time_name.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_time_name.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_time_name.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_time_name.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_time_name._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_time_name.metadata)
        self.assertNotEqual(self.ts_original.column_metadata, ts_different_time_name.column_metadata)

    def test_different_periodicity(self) -> None:
        """Test that TimeSeries objects with different periodicity are not equal."""
        ts_different_periodicity = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_months(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_periodicity)

        # Test the expected property has changed
        assert_frame_equal(self.ts_original.df, ts_different_periodicity.df)
        self.assertEqual(self.ts_original.time_name, ts_different_periodicity.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_periodicity.resolution)
        self.assertNotEqual(self.ts_original.periodicity, ts_different_periodicity.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_periodicity.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_periodicity._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_periodicity.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_periodicity.column_metadata)

    def test_different_resolution(self) -> None:
        """Test that TimeSeries objects with different resolution are not equal."""
        ts_different_resolution = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_seconds(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_resolution)

        assert_frame_equal(self.ts_original.df, ts_different_resolution.df)
        self.assertEqual(self.ts_original.time_name, ts_different_resolution.time_name)
        self.assertNotEqual(self.ts_original.resolution, ts_different_resolution.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_resolution.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_resolution.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_resolution._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_resolution.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_resolution.column_metadata)

    def test_different_time_anchor(self) -> None:
        """Test that TimeSeries objects with different time anchor are not equal."""
        ts_different_time_anchor = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
                time_anchor="end",
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_time_anchor)

        # Test the expected property has changed
        assert_frame_equal(self.ts_original.df, ts_different_time_anchor.df)
        self.assertEqual(self.ts_original.time_name, ts_different_time_anchor.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_time_anchor.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_time_anchor.periodicity)
        self.assertNotEqual(self.ts_original.time_anchor, ts_different_time_anchor.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_time_anchor._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_time_anchor.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_time_anchor.column_metadata)

    def test_different_flag_systems(self) -> None:
        """Test that TimeSeries objects with different flag systems are not equal."""
        ts_different_flags = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_2)
            .with_metadata(self.metadata)
        )

        self.assertNotEqual(self.ts_original, ts_different_flags)

        assert_frame_equal(self.ts_original.df, ts_different_flags.df)
        self.assertEqual(self.ts_original.time_name, ts_different_flags.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_flags.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_flags.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_flags.time_anchor)
        self.assertNotEqual(self.ts_original._flag_manager, ts_different_flags._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_flags.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_flags.column_metadata)

    def test_different_metadata(self) -> None:
        """Test that TimeSeries objects with different metadata are not equal."""
        ts_different_metadata = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata({"site": "DIFFERENT", "organisation": "UKCEH"})
        )

        self.assertNotEqual(self.ts_original, ts_different_metadata)

        assert_frame_equal(self.ts_original.df, ts_different_metadata.df)
        self.assertEqual(self.ts_original.time_name, ts_different_metadata.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_metadata.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_metadata.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_metadata.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_metadata._flag_manager)
        self.assertNotEqual(self.ts_original.metadata, ts_different_metadata.metadata)
        self.assertEqual(self.ts_original.column_metadata, ts_different_metadata.column_metadata)

    def test_different_column_metadata(self) -> None:
        """Test that TimeSeries objects with different metadata are not equal."""
        ts_different_column_metadata = (
            TimeSeries(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
            .with_column_metadata({"value": {"units": "mm"}})
        )

        self.assertNotEqual(self.ts_original, ts_different_column_metadata)

        assert_frame_equal(self.ts_original.df, ts_different_column_metadata.df)
        self.assertEqual(self.ts_original.time_name, ts_different_column_metadata.time_name)
        self.assertEqual(self.ts_original.resolution, ts_different_column_metadata.resolution)
        self.assertEqual(self.ts_original.periodicity, ts_different_column_metadata.periodicity)
        self.assertEqual(self.ts_original.time_anchor, ts_different_column_metadata.time_anchor)
        self.assertEqual(self.ts_original._flag_manager, ts_different_column_metadata._flag_manager)
        self.assertEqual(self.ts_original.metadata, ts_different_column_metadata.metadata)
        self.assertNotEqual(self.ts_original.column_metadata, ts_different_column_metadata.column_metadata)

    @parameterized.expand(
        [
            ("str", "hello"),
            ("int", 123),
            ("dict", {"key": "value"}),
            ("df", pl.DataFrame()),
        ]
    )
    def test_different_object(self, _: str, non_ts: Any) -> None:
        """Test that comparing against a non TimeSeries objects are not equal."""
        self.assertNotEqual(self.ts_original, non_ts)
