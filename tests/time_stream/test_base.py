import re
from datetime import date, datetime
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_frame_not_equal, assert_series_equal

from time_stream.aggregation import Percentile
from time_stream.base import TimeFrame
from time_stream.bitwise import BitwiseFlag
from time_stream.exceptions import ColumnNotFoundError, MetadataError
from time_stream.flag_manager import FlagColumn
from time_stream.period import Period


class TestSortTime:
    def test_sort_random_dates(self) -> None:
        """Test that random dates are sorted appropriately"""
        times = [date(1990, 1, 1), date(2019, 5, 8), date(1967, 12, 25), date(2059, 8, 12)]
        expected = pl.Series("time", [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)])

        tf = TimeFrame(pl.DataFrame({"time": times}), time_name="time")
        tf.sort_time()
        assert_series_equal(tf.df["time"], expected)

    def test_sort_sorted_dates(self) -> None:
        """Test that already sorted dates are maintained"""
        times = [date(1967, 12, 25), date(1990, 1, 1), date(2019, 5, 8), date(2059, 8, 12)]
        expected = pl.Series("time", times)

        tf = TimeFrame(pl.DataFrame({"time": times}), time_name="time")
        tf.sort_time()
        assert_series_equal(tf.df["time"], expected)

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
        tf = TimeFrame(pl.DataFrame({"time": times}), time_name="time")
        tf.sort_time()
        assert_series_equal(tf.df["time"], expected)


class TestSelectColumns:
    times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
    df = pl.DataFrame({"time": times, "col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    metadata = {
        "col1": {"key1": "1", "key2": "10", "key3": "100"},
        "col2": {"key1": "2", "key2": "20", "key3": "200"},
        "col3": {"key1": "3", "key2": "30", "key3": "300"},
    }

    @pytest.mark.parametrize("col", ["col1", ["col1"]], ids=["as str", "as list"])
    def test_select_single_column(self, col: str | list) -> None:
        """Test selecting a single of column."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)
        expected = TimeFrame(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )
        result = tf.select(col)
        assert result == expected

    def test_select_multiple_columns(self) -> None:
        """Test selecting multiple columns."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)
        expected = TimeFrame(self.df.select(["time", "col1", "col2"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"], "col2": self.metadata["col2"]}
        )

        result = tf.select(["col1", "col2"])
        assert result == expected

    def test_select_no_columns_raises_error(self) -> None:
        """Test selecting no columns raises error."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)

        expected_error = "No columns specified"

        with pytest.raises(ColumnNotFoundError, match=expected_error):
            tf.select([])

    def test_select_nonexistent_column(self) -> None:
        """Test selecting a column that does not exist raises error."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)

        expected_error = "Columns not found in dataframe: ['nonexistent_column']"
        with pytest.raises(ColumnNotFoundError, match=re.escape(expected_error)):
            tf.select(["nonexistent_column"])

    def test_select_existing_and_nonexistent_column(self) -> None:
        """Test selecting a column that does not exist, alongside existing columns, still raises error"""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)

        expected_error = "Columns not found in dataframe: ['nonexistent_column']"

        with pytest.raises(ColumnNotFoundError, match=re.escape(expected_error)):
            tf.select(["col1", "col2", "nonexistent_column"])

    def test_select_column_doesnt_mutate_original_tf(self) -> None:
        """When selecting a column, the original tf should be unchanged"""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)
        original_df = tf.df

        expected = TimeFrame(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )

        col1_tf = tf.select(["col1"])
        assert col1_tf == expected
        assert_frame_equal(tf.df, original_df)

        expected = TimeFrame(self.df.select(["time", "col2"]), time_name="time").with_column_metadata(
            {"col2": self.metadata["col2"]}
        )

        col2_tf = tf.select(["col2"])
        assert col2_tf == expected
        assert_frame_equal(tf.df, original_df)

    def test_select_column_with_flags(self) -> None:
        tf = TimeFrame(self.df, time_name="time").with_flag_system("system", {"A": 1, "B": 2, "C": 4})
        tf.init_flag_column("col1", "system", "flag_col")

        expected = TimeFrame(self.df.select(["time", "col1"]), time_name="time").with_flag_system(
            "system", {"A": 1, "B": 2, "C": 4}
        )
        expected.init_flag_column("col1", "system", "flag_col")

        result = tf.select(["col1"])
        assert result == expected


class TestGetItem:
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
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)
        result = tf["time"]
        expected = TimeFrame(self.df[["time"]], time_name="time")
        assert result == expected

    def test_access_data_column(self) -> None:
        """Test accessing a data column."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)
        result = tf["col1"]
        expected = TimeFrame(self.df.select(["time", "col1"]), time_name="time").with_column_metadata(
            {"col1": self.metadata["col1"]}
        )
        assert result == expected

    def test_access_multiple_data_columns(self) -> None:
        """Test accessing multiple data columns."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)
        result = tf[["col1", "col2"]]

        expected = TimeFrame(
            self.df.select(["time", "col1", "col2"]),
            time_name="time",
        ).with_column_metadata({"col1": self.metadata["col1"], "col2": self.metadata["col2"]})

        assert result == expected

    def test_non_existent_column(self) -> None:
        """Test accessing non-existent data column raises error."""
        tf = TimeFrame(self.df, time_name="time").with_column_metadata(self.metadata)

        expected_error = "Columns not found in dataframe: ['col0']"
        with pytest.raises(ColumnNotFoundError, match=re.escape(expected_error)):
            tf["col0"]


class TestColumnMetadata:
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
    }
    tf = TimeFrame(df, time_name="time").with_column_metadata(metadata)

    def test_all_columns_have_metadata(self) -> None:
        """Test on the init that all columns get initialised with an empty dict if no metadata provided"""
        assert set(self.df.columns) == set(self.tf.column_metadata.keys())
        assert self.tf.column_metadata["col1"] == {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"}
        assert self.tf.column_metadata["col2"] == {"key1": "2", "key2": "20", "key3": "200"}
        assert self.tf.column_metadata["col3"] == {}

    def test_retrieve_metadata_for_single_column(self) -> None:
        """Test retrieving metadata for a single column."""
        expected = {"key1": "1", "key2": "10", "key3": "100", "key4": "1000"}
        result = self.tf.column_metadata["col1"]
        assert result == expected

    def test_retrieve_metadata_for_specific_key(self) -> None:
        """Test retrieving a specific metadata key."""
        expected = "1"
        result = self.tf.column_metadata["col1"]["key1"]
        assert result == expected

    def test_retrieve_metadata_for_nonexistent_column(self) -> None:
        """Test that a KeyError is raised when requesting a non-existent column."""
        with pytest.raises(KeyError):
            _ = self.tf.column_metadata["nonexistent_column"]

    def test_retrieve_metadata_for_nonexistent_key_single_column_strict(self) -> None:
        """Test that error raised when requesting a non-existent metadata key for an existing single column"""
        with pytest.raises(KeyError):
            _ = self.tf.column_metadata["col1"]["nonexistent_key"]

    def test_set_non_dict_raises_error(self) -> None:
        """Test that error raised when setting a non-dict object as the metadata for a column"""
        with pytest.raises(MetadataError):
            _ = self.tf.column_metadata["col1"] = 123  # noqa - expecting typehint error

    def test_set_column_metadata(self) -> None:
        """Test that we can set the column metadata object directly. Existing metadata should be maintained"""
        column_metadata = {"col1": {"key": "value"}}
        self.tf.column_metadata = column_metadata
        column_metadata["col2"] = {"key1": "2", "key2": "20", "key3": "200"}
        column_metadata["col3"] = {}
        column_metadata["time"] = {}
        assert self.tf.column_metadata == column_metadata

    def test_set_column_metadata_column_error(self) -> None:
        """Test that directly setting column metadata with incorrect format raises error."""
        column_metadata = {"missing_col": {"key": "value"}}
        with pytest.raises(KeyError):
            self.tf.column_metadata = column_metadata

    def test_set_column_metadata_none(self) -> None:
        """Test that none resets column metadata to empty dicts for each column"""
        self.tf.column_metadata = None
        column_metadata = {"col1": {}, "col2": {}, "col3": {}, "time": {}}
        assert self.tf.column_metadata == column_metadata

    @pytest.mark.parametrize("metadata", [1234, "network: FDRI", ["a", "b", "c"]], ids=["int", "str", "list"])
    def test_set_column_metadata_not_dict_error(self, metadata: Any) -> None:
        """Test that trying to set the column metadata object to not a dict raises error"""
        with pytest.raises(MetadataError):
            self.tf.column_metadata = metadata

    @pytest.mark.parametrize("metadata", [1234, "network: FDRI", ["a", "b", "c"]], ids=["int", "str", "list"])
    def test_set_nested_column_metadata_not_dict_error(self, metadata: Any) -> None:
        """Test that trying to set the inner column metadata object to not a dict raises error"""
        with pytest.raises(MetadataError):
            self.tf.column_metadata = {"col1": metadata}

    def test_clear_column_metadata(self) -> None:
        """Test that removing the column_metadata object sets it back to an empty dicts for each column"""
        del self.tf.column_metadata
        column_metadata = {"col1": {}, "col2": {}, "col3": {}, "time": {}}

        assert self.tf.column_metadata == column_metadata


class TestMetadata:
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    metadata = {"site_id": 1234, "network": "FDRI", "some_info": {1: "a", 2: "b", 3: "c"}}
    tf = TimeFrame(df, time_name="time").with_metadata(metadata)

    def test_retrieve_all_metadata(self) -> None:
        """Test retrieving all metadata"""
        result = self.tf.metadata
        assert result == self.metadata

    @pytest.mark.parametrize(
        "key,expected",
        [
            ("site_id", 1234),
            ("network", "FDRI"),
            ("some_info", {1: "a", 2: "b", 3: "c"}),
        ],
        ids=["int", "str", "dict"],
    )
    def test_retrieve_metadata_for_specific_key(self, key: str, expected: Any) -> None:
        """Test retrieving a specific metadata key."""
        result = self.tf.metadata[key]
        assert result == expected

    def test_retrieve_metadata_for_nonexistent_key_raises_error(self) -> None:
        """Test that an error is raised for a non-existent key."""
        with pytest.raises(KeyError):
            self.tf.metadata["nonexistent_key"]

    def test_set_metadata(self) -> None:
        """Test that we can set the metadata object directly"""
        metadata = {"key": "value"}
        self.tf.metadata = metadata

        assert self.tf.metadata == metadata

    def test_set_metadata_none(self) -> None:
        """Test that none sets to empty dict"""
        self.tf.metadata = None
        assert self.tf.metadata == {}

    @pytest.mark.parametrize("metadata", [1234, "network: FDRI", ["a", "b", "c"]], ids=["int", "str", "list"])
    def test_set_metadata_not_dict_error(self, metadata: Any) -> None:
        """Test that trying to set the metadata object to not a dict raises error"""
        with pytest.raises(MetadataError):
            self.tf.metadata = metadata

    def test_clear_metadata(self) -> None:
        """Test that removing the metadata object sets it back to an empty dict"""
        del self.tf.metadata
        assert self.tf.metadata == {}


class TestInitFlagColumn:
    @staticmethod
    def setup_tf() -> tuple[TimeFrame, BitwiseFlag]:
        df = pl.DataFrame(
            {
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "value": [10, 20, 30],
                "existing_flags": [0, 1, 2],
            }
        )
        flag_system = BitwiseFlag("quality_control", {"OUT_OF_RANGE": 1, "SPIKE": 2, "LOW_BATTERY": 4})
        tf = TimeFrame(df=df, time_name="time")
        tf.register_flag_system("quality_control", flag_system)

        return tf, flag_system

    def test_init_flag_column_success(self) -> None:
        """Test initialising a flag column with a valid flag system."""
        tf, flag_system = self.setup_tf()

        # Shouldn't have any flag columns to start with
        assert tf._flag_manager._flag_columns == {}

        tf.init_flag_column("value", "quality_control", "flag_column")
        assert tf._flag_manager._flag_columns == {"flag_column": FlagColumn("flag_column", "value", flag_system)}

        assert "flag_column" in tf.flag_columns

    def test_init_column_adds_to_metadata(self) -> None:
        """Test that a new flag column gets added to the column metadata dict."""
        tf, flag_system = self.setup_tf()
        tf.init_flag_column("value", "quality_control", "flag_column")
        expected = {"time": {}, "value": {}, "existing_flags": {}, "flag_column": {}}
        assert tf.column_metadata == expected

    def test_init_flag_column_with_single_value(self) -> None:
        """Test initialising a flag column with a valid flag system and a non-default single value"""
        tf, flag_system = self.setup_tf()

        # Shouldn't have any flag columns to start with
        assert tf._flag_manager._flag_columns == {}

        tf.init_flag_column("value", "quality_control", "flag_column", 1)
        expected_values = pl.Series("flag_column", [1, 1, 1], dtype=pl.Int64)
        assert_series_equal(tf.df["flag_column"], expected_values)

    def test_init_flag_column_with_list_value(self) -> None:
        """Test initialising a flag column with a valid flag system and a non-default list of values"""
        tf, flag_system = self.setup_tf()

        # Shouldn't have any flag columns to start with
        assert tf._flag_manager._flag_columns == {}

        tf.init_flag_column("value", "quality_control", "flag_column", [1, 2, 4])
        expected_values = pl.Series("flag_column", [1, 2, 4], dtype=pl.Int64)
        assert_series_equal(tf.df["flag_column"], expected_values)

    def test_with_no_flag_column_name(self) -> None:
        """Test that a default flag column name is used if not provided."""
        tf, flag_system = self.setup_tf()

        # Shouldn't have any flag columns to start with
        assert tf._flag_manager._flag_columns == {}

        tf.init_flag_column("value", "quality_control")
        default_name = "value__flag__quality_control"

        assert tf._flag_manager._flag_columns == {default_name: FlagColumn(default_name, "value", flag_system)}

        assert default_name in tf.flag_columns

    def test_init_no_column_name_adds_to_metadata(self) -> None:
        """Test that a new flag column gets added to the column metadata dict."""
        tf, flag_system = self.setup_tf()

        tf.init_flag_column("value", "quality_control")
        expected = {"time": {}, "value": {}, "existing_flags": {}, "value__flag__quality_control": {}}
        assert tf.column_metadata == expected


class TestTimeSeriesEquality:
    def setUp(self) -> None:
        """Set up multiple TimeSeries objects for testing."""

    df_original = pl.DataFrame(
        {"time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)], "value": [10, 20, 30]}
    )
    df_same = pl.DataFrame(
        {"time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)], "value": [10, 20, 30]}
    )
    df_different_values = pl.DataFrame(
        {"time": [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)], "value": [100, 200, 300]}
    )
    df_different_times = pl.DataFrame(
        {"time": [datetime(2020, 1, 1), datetime(2020, 2, 1), datetime(2020, 3, 1)], "value": [10, 20, 30]}
    )

    flag_systems_1 = {"OK": 1, "WARNING": 2}
    flag_systems_2 = {"NOT_OK": 4, "ERROR": 8}
    metadata = {"site": "DUMMY", "organisation": "UKCEH"}

    tf_original = (
        TimeFrame(
            df=df_original,
            time_name="time",
            resolution=Period.of_days(1),
            periodicity=Period.of_days(1),
        )
        .with_flag_system("quality_flags", flag_systems_1)
        .with_metadata(metadata)
    )

    def test_equal_timeseries(self) -> None:
        """Test that two identical TimeSeries objects are considered equal."""
        tf_same = (
            TimeFrame(
                df=self.df_same,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original == tf_same

    def test_different_data_values(self) -> None:
        """Test that TimeSeries objects with different data values are not equal."""
        tf_different_df = (
            TimeFrame(
                df=self.df_different_values,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_df

        # Test the expected property has changed
        assert_frame_not_equal(self.tf_original.df, tf_different_df.df)
        assert self.tf_original.time_name == tf_different_df.time_name
        assert self.tf_original.resolution == tf_different_df.resolution
        assert self.tf_original.periodicity == tf_different_df.periodicity
        assert self.tf_original.time_anchor == tf_different_df.time_anchor
        assert self.tf_original._flag_manager == tf_different_df._flag_manager
        assert self.tf_original.metadata == tf_different_df.metadata
        assert self.tf_original.column_metadata == tf_different_df.column_metadata

    def test_different_time_values(self) -> None:
        """Test that TimeSeries objects with different time values are not equal."""
        tf_different_times = (
            TimeFrame(
                df=self.df_different_times,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_times

        # Test the expected property has changed
        assert_frame_not_equal(self.tf_original.df, tf_different_times.df)

        assert self.tf_original.time_name == tf_different_times.time_name
        assert self.tf_original.resolution == tf_different_times.resolution
        assert self.tf_original.periodicity == tf_different_times.periodicity
        assert self.tf_original.time_anchor == tf_different_times.time_anchor
        assert self.tf_original._flag_manager == tf_different_times._flag_manager
        assert self.tf_original.metadata == tf_different_times.metadata
        assert self.tf_original.column_metadata == tf_different_times.column_metadata

    def test_different_time_name(self) -> None:
        """Test that TimeSeries objects with different time column name are not equal."""
        tf_different_time_name = (
            TimeFrame(
                df=self.df_original.rename({"time": "timestamp"}),
                time_name="timestamp",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_time_name

        # Test the expected property has changed
        assert_frame_not_equal(self.tf_original.df, tf_different_time_name.df)

        assert self.tf_original.time_name != tf_different_time_name.time_name
        assert self.tf_original.resolution == tf_different_time_name.resolution
        assert self.tf_original.periodicity == tf_different_time_name.periodicity
        assert self.tf_original.time_anchor == tf_different_time_name.time_anchor
        assert self.tf_original._flag_manager == tf_different_time_name._flag_manager
        assert self.tf_original.metadata == tf_different_time_name.metadata
        assert self.tf_original.column_metadata != tf_different_time_name.column_metadata

    def test_different_periodicity(self) -> None:
        """Test that TimeSeries objects with different periodicity are not equal."""
        tf_different_periodicity = (
            TimeFrame(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_months(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_periodicity

        # Test the expected property has changed
        assert_frame_equal(self.tf_original.df, tf_different_periodicity.df)

        assert self.tf_original.time_name == tf_different_periodicity.time_name
        assert self.tf_original.resolution == tf_different_periodicity.resolution
        assert self.tf_original.periodicity != tf_different_periodicity.periodicity
        assert self.tf_original.time_anchor == tf_different_periodicity.time_anchor
        assert self.tf_original._flag_manager == tf_different_periodicity._flag_manager
        assert self.tf_original.metadata == tf_different_periodicity.metadata
        assert self.tf_original.column_metadata == tf_different_periodicity.column_metadata

    def test_different_resolution(self) -> None:
        """Test that TimeSeries objects with different resolution are not equal."""
        tf_different_resolution = (
            TimeFrame(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_seconds(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_resolution

        assert_frame_equal(self.tf_original.df, tf_different_resolution.df)

        assert self.tf_original.time_name == tf_different_resolution.time_name
        assert self.tf_original.resolution != tf_different_resolution.resolution
        assert self.tf_original.periodicity == tf_different_resolution.periodicity
        assert self.tf_original.time_anchor == tf_different_resolution.time_anchor
        assert self.tf_original._flag_manager == tf_different_resolution._flag_manager
        assert self.tf_original.metadata == tf_different_resolution.metadata
        assert self.tf_original.column_metadata == tf_different_resolution.column_metadata

    def test_different_time_anchor(self) -> None:
        """Test that TimeSeries objects with different time anchor are not equal."""
        tf_different_time_anchor = (
            TimeFrame(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
                time_anchor="end",
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_time_anchor

        # Test the expected property has changed
        assert_frame_equal(self.tf_original.df, tf_different_time_anchor.df)

        assert self.tf_original.time_name == tf_different_time_anchor.time_name
        assert self.tf_original.resolution == tf_different_time_anchor.resolution
        assert self.tf_original.periodicity == tf_different_time_anchor.periodicity
        assert self.tf_original.time_anchor != tf_different_time_anchor.time_anchor
        assert self.tf_original._flag_manager == tf_different_time_anchor._flag_manager
        assert self.tf_original.metadata == tf_different_time_anchor.metadata
        assert self.tf_original.column_metadata == tf_different_time_anchor.column_metadata

    def test_different_flag_systems(self) -> None:
        """Test that TimeSeries objects with different flag systems are not equal."""
        tf_different_flags = (
            TimeFrame(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_2)
            .with_metadata(self.metadata)
        )

        assert self.tf_original != tf_different_flags

        assert_frame_equal(self.tf_original.df, tf_different_flags.df)

        assert self.tf_original.time_name == tf_different_flags.time_name
        assert self.tf_original.resolution == tf_different_flags.resolution
        assert self.tf_original.periodicity == tf_different_flags.periodicity
        assert self.tf_original.time_anchor == tf_different_flags.time_anchor
        assert self.tf_original._flag_manager != tf_different_flags._flag_manager
        assert self.tf_original.metadata == tf_different_flags.metadata
        assert self.tf_original.column_metadata == tf_different_flags.column_metadata

    def test_different_metadata(self) -> None:
        """Test that TimeSeries objects with different metadata are not equal."""
        tf_different_metadata = (
            TimeFrame(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata({"site": "DIFFERENT", "organisation": "UKCEH"})
        )

        assert self.tf_original != tf_different_metadata

        assert_frame_equal(self.tf_original.df, tf_different_metadata.df)

        assert self.tf_original.time_name == tf_different_metadata.time_name
        assert self.tf_original.resolution == tf_different_metadata.resolution
        assert self.tf_original.periodicity == tf_different_metadata.periodicity
        assert self.tf_original.time_anchor == tf_different_metadata.time_anchor
        assert self.tf_original._flag_manager == tf_different_metadata._flag_manager
        assert self.tf_original.metadata != tf_different_metadata.metadata
        assert self.tf_original.column_metadata == tf_different_metadata.column_metadata

    def test_different_column_metadata(self) -> None:
        """Test that TimeSeries objects with different metadata are not equal."""
        tf_different_column_metadata = (
            TimeFrame(
                df=self.df_original,
                time_name="time",
                resolution=Period.of_days(1),
                periodicity=Period.of_days(1),
            )
            .with_flag_system("quality_flags", self.flag_systems_1)
            .with_metadata(self.metadata)
            .with_column_metadata({"value": {"units": "mm"}})
        )

        assert self.tf_original != tf_different_column_metadata

        assert_frame_equal(self.tf_original.df, tf_different_column_metadata.df)

        assert self.tf_original.time_name == tf_different_column_metadata.time_name
        assert self.tf_original.resolution == tf_different_column_metadata.resolution
        assert self.tf_original.periodicity == tf_different_column_metadata.periodicity
        assert self.tf_original.time_anchor == tf_different_column_metadata.time_anchor
        assert self.tf_original._flag_manager == tf_different_column_metadata._flag_manager
        assert self.tf_original.metadata == tf_different_column_metadata.metadata
        assert self.tf_original.column_metadata != tf_different_column_metadata.column_metadata

    @pytest.mark.parametrize(
        "non_tf",
        [
            "hello",
            123,
            {"key": "value"},
            pl.DataFrame(),
        ],
        ids=["str", "int", "dict", "df"],
    )
    def test_different_object(self, non_tf: Any) -> None:
        """Test that comparing against a non TimeSeries objects are not equal."""
        assert self.tf_original != non_tf


class TestAggregate:
    def test_aggregate_periodicity(self) -> None:
        period = Period.of_hours(1)
        length = 48
        df = pl.DataFrame(
            {
                "timestamp": [period.datetime(period.ordinal(datetime(2025, 1, 1)) + i) for i in range(length)],
                "value": list(range(length)),
            }
        )
        tf = TimeFrame(df=df, time_name="timestamp", resolution=period, periodicity=period)

        expected_df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, 0, 0, 0), datetime(2025, 1, 2, 0, 0, 0)],
                "percentile_value": [22, 46],
                "count_value": [24, 24],
                "expected_count_timestamp": [24, 24],
                "valid_value": [True, True],
            }
        )

        aggregated_tf = tf.aggregate(
            aggregation_period=Period.of_days(1), aggregation_function=Percentile, columns="value", p=95
        )

        assert_frame_equal(aggregated_tf.df, expected_df, check_dtypes=False)
