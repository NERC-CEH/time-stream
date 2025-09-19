import unittest
from datetime import date, datetime

import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal, assert_series_equal

from time_stream.base import TimeSeries
from time_stream.exceptions import ColumnNotFoundError, MetadataError
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

    def test_select_single_column(self) -> None:
        """Test selecting a single of column."""
        ts = TimeSeries(self.df, time_name="time").with_column_metadata(self.metadata)
        expected = TimeSeries(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )
        result = ts.select(["col1"])
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
        )

    def test_equal_timeseries(self) -> None:
        """Test that two identical TimeSeries objects are considered equal."""
        ts_same = TimeSeries(
            df=self.df_same,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
        )
        self.assertEqual(self.ts_original, ts_same)

    def test_different_data_values(self) -> None:
        """Test that TimeSeries objects with different data values are not equal."""
        ts_different_df = TimeSeries(
            df=self.df_different_values,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
        )
        self.assertNotEqual(self.ts_original, ts_different_df)

    def test_different_time_values(self) -> None:
        """Test that TimeSeries objects with different time values are not equal."""
        ts_different_times = TimeSeries(
            df=self.df_different_times,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
        )
        self.assertNotEqual(self.ts_original, ts_different_times)

    def test_different_time_name(self) -> None:
        """Test that TimeSeries objects with different time column name are not equal."""
        ts_different_time_name = TimeSeries(
            df=self.df_original.rename({"time": "timestamp"}),
            time_name="timestamp",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
        )
        self.assertNotEqual(self.ts_original, ts_different_time_name)

    def test_different_periodicity(self) -> None:
        """Test that TimeSeries objects with different periodicity are not equal."""
        ts_different_periodicity = TimeSeries(
            df=self.df_original, time_name="time", resolution=Period.of_days(1), periodicity=Period.of_months(1)
        )
        self.assertNotEqual(self.ts_original, ts_different_periodicity)

    def test_different_resolution(self) -> None:
        """Test that TimeSeries objects with different resolution are not equal."""
        ts_different_resolution = TimeSeries(
            df=self.df_original,
            time_name="time",
            resolution=Period.of_seconds(1),
            periodicity=Period.of_days(1),
        )
        self.assertNotEqual(self.ts_original, ts_different_resolution)


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
    ts = TimeSeries(df, time_name="time").with_column_metadata(metadata)

    def test_retrieve_metadata_for_single_column(self) -> None:
        """Test retrieving metadata for a single column."""
        expected = {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"}
        result = self.ts.column_metadata("col1")
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_specific_key(self) -> None:
        """Test retrieving a specific metadata key."""
        expected = {"key1": "1"}
        result = self.ts.column_metadata("col1", "key1")
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_multiple_keys(self) -> None:
        """Test retrieving multiple metadata keys."""
        expected = {"key1": "1", "key3": "100"}
        result = self.ts.column_metadata("col1", ["key1", "key3"])
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_column(self) -> None:
        """Test that a KeyError is raised when requesting a non-existent column."""
        with self.assertRaises(ColumnNotFoundError):
            self.ts.column_metadata("nonexistent_column")

    def test_retrieve_metadata_for_nonexistent_key_single_column_not_strict(self) -> None:
        """Test that error NOT raised when requesting a non-existent metadata key for an existing single column,
        if strict is false"""
        expected = {"nonexistent_key": None}
        result = self.ts.column_metadata("col1", "nonexistent_key", strict=False)
        self.assertEqual(result, expected)

    def test_retrieve_metadata_for_nonexistent_key_single_column_strict(self) -> None:
        """Test that error raised when requesting a non-existent metadata key for an existing single column"""
        with self.assertRaises(MetadataError) as err:
            self.ts.column_metadata("col1", "nonexistent_key", strict=True)
        self.assertEqual("Metadata key 'nonexistent_key' not found for column 'col1'", str(err.exception))


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
    ts = TimeSeries(df, time_name="time").with_series_metadata(metadata)

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
            self.ts.metadata("nonexistent_key", strict=True)
        self.assertEqual("Metadata key 'nonexistent_key' not found", str(err.exception))

    def test_retrieve_metadata_for_nonexistent_key_strict_false(self) -> None:
        """Test that an empty result is returned when strict is false for non-existent key."""
        expected = {"nonexistent_key": None}
        result = self.ts.metadata("nonexistent_key", strict=False)
        self.assertEqual(result, expected)
