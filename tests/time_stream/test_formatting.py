from datetime import datetime
from unittest.mock import Mock

import polars as pl
import pytest

from time_stream import TimeFrame
from time_stream.bitwise import BitwiseFlag
from time_stream.formatting import (
    format_headers,
    format_key_value_pairs,
    format_object_size_str,
    format_repr_items,
    format_time_properties,
    format_timeframe_column_samples,
    format_timeframe_columns,
    string_preview,
    timeframe_repr,
)


class TestFormatObjectSizeStr:
    def test_bytes(self) -> None:
        """Test formatting for sizes under 1KB."""
        assert format_object_size_str(500) == "500.00 B"
        assert format_object_size_str(1023) == "1023.00 B"

    def test_kilobytes(self) -> None:
        """Test formatting for KB range."""
        assert format_object_size_str(1024) == "1.00 KB"
        assert format_object_size_str(2048) == "2.00 KB"
        assert format_object_size_str(1536) == "1.50 KB"

    def test_megabytes(self) -> None:
        """Test formatting for MB range."""
        assert format_object_size_str(1024 * 1024) == "1.00 MB"
        assert format_object_size_str(5 * 1024 * 1024) == "5.00 MB"

    def test_gigabytes(self) -> None:
        """Test formatting for GB range."""
        assert format_object_size_str(1024 * 1024 * 1024) == "1.00 GB"
        assert format_object_size_str(2.5 * 1024 * 1024 * 1024) == "2.50 GB"

    def test_terabytes(self) -> None:
        """Test formatting for TB range."""
        assert format_object_size_str(1024 * 1024 * 1024 * 1024) == "1.00 TB"

    def test_petabytes(self) -> None:
        """Test formatting for PB range."""
        assert format_object_size_str(1024**5) == "1.00 PB"
        assert format_object_size_str(2 * 1024**5) == "2.00 PB"

    def test_zero_bytes(self) -> None:
        """Test formatting for zero bytes."""
        assert format_object_size_str(0) == "0.00 B"


class TestFormatTimeframeColumnSamples:
    def test_no_values(self) -> None:
        """Test formatting with no values."""
        series = pl.Series("test", [])
        result = format_timeframe_column_samples(series)
        assert result == "[]"

    def test_single_value(self) -> None:
        """Test formatting with a single value."""
        series = pl.Series("test", [42])
        result = format_timeframe_column_samples(series)
        assert result == "[42]"

    def test_two_values(self) -> None:
        """Test formatting with two values (no ellipsis)."""
        series = pl.Series("test", [1, 2])
        result = format_timeframe_column_samples(series)
        assert result == "[1, 2]"

    def test_multiple_values(self) -> None:
        """Test formatting with more than two values (with ellipsis)."""
        series = pl.Series("test", range(999))
        result = format_timeframe_column_samples(series)
        assert result == "[0, ..., 998]"

    def test_string_values(self) -> None:
        """Test formatting with string values."""
        series = pl.Series("test", ["first", "middle", "last"])
        result = format_timeframe_column_samples(series)
        assert result == "[first, ..., last]"

    def test_float_values(self) -> None:
        """Test formatting with float values."""
        series = pl.Series("test", [1.5, 2.5, 3.5, 4.5])
        result = format_timeframe_column_samples(series)
        assert result == "[1.5, ..., 4.5]"

    def test_datetime_values(self) -> None:
        """Test formatting with datetime values."""
        series = pl.Series("test", [datetime(2025, 1, d + 1) for d in range(5)])
        result = format_timeframe_column_samples(series)
        assert result == "[2025-01-01 00:00:00, ..., 2025-01-05 00:00:00]"


class TestFormatHeaders:
    def test_lowercase_header(self) -> None:
        """Test formatting lowercase header."""
        assert format_headers("time properties") == "Time properties:"

    def test_uppercase_header(self) -> None:
        """Test formatting uppercase header."""
        assert format_headers("COLUMNS") == "Columns:"

    def test_mixed_case_header(self) -> None:
        """Test formatting mixed case header."""
        assert format_headers("Flag Systems") == "Flag systems:"

    def test_empty_header(self) -> None:
        """Test formatting empty header."""
        assert format_headers("") == ""


class TestFormatKeyValuePairs:
    def test_single_value(self) -> None:
        """Test formatting with a single value."""
        result = format_key_value_pairs("key", "value")
        assert result == "    key : value"

    def test_multiple_values(self) -> None:
        """Test formatting with multiple values."""
        result = format_key_value_pairs("key", "value1", "value2", "value3")
        assert result == "    key : value1  value2  value3"

    def test_none_key(self) -> None:
        """Test formatting with None key."""
        result = format_key_value_pairs(None, "value")
        assert result == ""

    def test_none_args(self) -> None:
        """Test formatting with None args."""
        result = format_key_value_pairs("key", None)
        assert result == "    key : None"

    def test_numeric_values(self) -> None:
        """Test formatting with numeric values."""
        result = format_key_value_pairs("count", 42, 3.14)
        assert result == "    count : 42  3.14"

    def test_padding_in_key(self) -> None:
        """Test that padding in key is preserved."""
        result = format_key_value_pairs("key   ", "value")
        assert result == "    key    : value"


class TestStringPreview:
    def test_short_string(self) -> None:
        """Test formatting string shorter than width."""
        result = string_preview("short text", width=80)
        assert result == "short text"

    def test_long_string(self) -> None:
        """Test formatting string longer than width."""
        long_text = "a" * 100
        result = string_preview(long_text, width=50)
        assert result == "a" * 45 + "[...]"

    def test_string_near_width(self) -> None:
        """Test formatting string near than width."""
        long_text = "a" * 10
        result = string_preview(long_text, width=9)
        assert result == "aaaa" + "[...]"

    def test_width_lt_placeholder(self) -> None:
        """Test formatting string when the width is less that the length of the placeholder"""
        long_text = "a" * 10
        result = string_preview(long_text, width=3)
        assert result == "[...]"

    def test_width_at_placeholder(self) -> None:
        """Test formatting string when the width is same length of the placeholder"""
        long_text = "a" * 10
        result = string_preview(long_text, width=5)
        assert result == "[...]"

    def test_exact_width(self) -> None:
        """Test formatting string exactly at width."""
        text = "a" * 80
        result = string_preview(text, width=80)
        # Should not be truncated
        assert result == text

    def test_numeric_value(self) -> None:
        """Test formatting numeric values."""
        result = string_preview(12345)
        assert result == "12345"

    def test_dict_value(self) -> None:
        """Test formatting with a dict."""
        d = {"a": 1, "b": 2, "c": 3}
        result = string_preview(d, width=20)
        assert result == "{'a': 1, 'b': 2[...]"

    def test_bitwise_flags_value(self) -> None:
        """Test formatting with bitwise flag object."""
        flags = BitwiseFlag("qc", {"BAD": 1, "GOOD": 2})
        result = string_preview(flags, width=20)
        assert result == "<qc (BAD=1, GOOD=2)>"

    def test_bitwise_flags_value_shorten(self) -> None:
        """Test formatting with bitwise flag object longer than width."""
        flags = BitwiseFlag("qc", {"BAD": 1, "GOOD": 2})
        result = string_preview(flags, width=10)
        assert result == "<qc ([...]"


class TestFormatReprItems:
    def test_single_section(self) -> None:
        """Test formatting single section."""
        items = {"Header": {"key1": "value1", "key2": "value2"}}
        result = format_repr_items(items)

        assert result == ["Header:", "    key1 : value1", "    key2 : value2"]

    def test_multiple_sections(self) -> None:
        """Test formatting multiple sections."""
        items = {"Header": {"key1": "value1", "key2": "value2"}, "Header2": {"key3": "value3", "key4": "value4"}}
        result = format_repr_items(items)

        assert result == [
            "Header:",
            "    key1 : value1",
            "    key2 : value2",
            "Header2:",
            "    key3 : value3",
            "    key4 : value4",
        ]

    def test_key_alignment(self) -> None:
        """Test that keys are properly aligned."""
        items = {"Header": {"short": "value1", "longer_key": "value2"}}
        result = format_repr_items(items)

        assert result == ["Header:", "    short      : value1", "    longer_key : value2"]

    def test_tuple_values(self) -> None:
        """Test formatting with tuple values."""
        items = {"Header": {"key": ("value1", "value2", "value3")}}
        result = format_repr_items(items)
        assert result == ["Header:", "    key : value1  value2  value3"]

    def test_empty_dict(self) -> None:
        """Test formatting empty dictionary."""
        result = format_repr_items({})
        assert result == []


class TestFormatTimeProperties:
    def test_properties(self) -> None:
        """Test formatting of time properties."""
        tf = Mock()
        tf.time_name = "timestamp"
        tf.resolution = "PT1H"
        tf.offset = "+T5M"
        tf.alignment = "PT1H+T5M"
        tf.periodicity = "PT1D+T5M"
        tf.time_anchor = "START"

        df = pl.DataFrame(
            {
                "timestamp": [1, 2, 3, 4, 5],
            }
        )
        tf.df = df

        result = format_time_properties(tf)

        assert result == {
            "Time column": ("timestamp", "[1, ..., 5]"),
            "Type": pl.Int64,
            "Resolution": "PT1H",
            "Offset": "+T5M",
            "Alignment": "PT1H+T5M",
            "Periodicity": "PT1D+T5M",
            "Anchor": "START",
        }


class TestFormatTimeframeColumns:
    def test_basic_columns(self) -> None:
        """Test formatting of regular columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d + 1) for d in range(5)],
                "col1": [10, 20, 30, 40, 50],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )
        tf = TimeFrame(df, "timestamp")

        result = format_timeframe_columns(tf)

        assert result == {
            "col1": (pl.Int64, "40.00 B", "[10, ..., 50]"),
            "col2": (pl.Float64, "40.00 B", "[1.1, ..., 5.5]"),
        }

    def test_flag_columns(self) -> None:
        """Test formatting of regular and flag columns."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d + 1) for d in range(5)],
                "col1": [10, 20, 30, 40, 50],
                "flag1": [1, 1, 2, 2, 2],
            }
        )
        tf = TimeFrame(df, "timestamp")
        tf.register_flag_system("qc", {"BAD": 1, "GOOD": 2})
        tf.register_flag_column("flag1", "col1", "qc")

        result = format_timeframe_columns(tf)

        assert result == {
            "col1": (pl.Int64, "40.00 B", "[10, ..., 50]"),
            "flag1": (pl.Int64, "40.00 B", "[1, ..., 2]", "(flags=qc)"),
        }


class TestTimeframeRepr:
    @pytest.fixture
    def timeframe(self) -> TimeFrame:
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, d + 1) for d in range(5)],
                "col1": [10, 20, 30, 40, 50],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "flag1": [1, 1, 2, 2, 2],
            }
        )
        return TimeFrame(df, "timestamp")

    def test_basic_repr(self, timeframe: TimeFrame) -> None:
        """Test basic TimeFrame representation with no optional sections"""
        result = timeframe_repr(timeframe)

        assert result == (
            "<time_stream.TimeFrame> Size (estimated): 208.00 B\n"
            "Time properties:\n"
            "    Time column : timestamp  [2025-01-01 00:00:00, ..., 2025-01-05 00:00:00]\n"
            "    Type        : Datetime(time_unit='us', time_zone=None)\n"
            "    Resolution  : PT0.000001S\n"
            "    Offset      : None\n"
            "    Alignment   : PT0.000001S\n"
            "    Periodicity : PT0.000001S\n"
            "    Anchor      : TimeAnchor.START\n"
            "Columns:\n"
            "    col1        : Int64  40.00 B  [10, ..., 50]\n"
            "    col2        : Float64  40.00 B  [1.1, ..., 5.5]\n"
            "    flag1       : Int64  40.00 B  [1, ..., 2]"
        )

    def test_repr_with_flag_systems(self, timeframe: TimeFrame) -> None:
        """Test representation with flag systems."""
        timeframe.register_flag_system("qc", {"BAD": 1, "GOOD": 2})
        timeframe.register_flag_column("flag1", "col1", "qc")

        result = timeframe_repr(timeframe)

        assert result == (
            "<time_stream.TimeFrame> Size (estimated): 208.00 B\n"
            "Time properties:\n"
            "    Time column : timestamp  [2025-01-01 00:00:00, ..., 2025-01-05 00:00:00]\n"
            "    Type        : Datetime(time_unit='us', time_zone=None)\n"
            "    Resolution  : PT0.000001S\n"
            "    Offset      : None\n"
            "    Alignment   : PT0.000001S\n"
            "    Periodicity : PT0.000001S\n"
            "    Anchor      : TimeAnchor.START\n"
            "Columns:\n"
            "    col1        : Int64  40.00 B  [10, ..., 50]\n"
            "    col2        : Float64  40.00 B  [1.1, ..., 5.5]\n"
            "    flag1       : Int64  40.00 B  [1, ..., 2]  (flags=qc)\n"
            "Flag systems:\n"
            "    qc          : <qc (BAD=1, GOOD=2)>"
        )

    def test_repr_with_metadata(self, timeframe: TimeFrame) -> None:
        """Test representation with metadata."""
        timeframe.metadata = {"source": "test_data", "version": "1.0"}
        result = timeframe_repr(timeframe)

        assert result == (
            "<time_stream.TimeFrame> Size (estimated): 208.00 B\n"
            "Time properties:\n"
            "    Time column : timestamp  [2025-01-01 00:00:00, ..., 2025-01-05 00:00:00]\n"
            "    Type        : Datetime(time_unit='us', time_zone=None)\n"
            "    Resolution  : PT0.000001S\n"
            "    Offset      : None\n"
            "    Alignment   : PT0.000001S\n"
            "    Periodicity : PT0.000001S\n"
            "    Anchor      : TimeAnchor.START\n"
            "Columns:\n"
            "    col1        : Int64  40.00 B  [10, ..., 50]\n"
            "    col2        : Float64  40.00 B  [1.1, ..., 5.5]\n"
            "    flag1       : Int64  40.00 B  [1, ..., 2]\n"
            "Metadata:\n"
            "    source      : test_data\n"
            "    version     : 1.0"
        )
