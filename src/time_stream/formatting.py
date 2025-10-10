"""
String formatting routines for __repr__ operations

This module provides utilities for creating human-readable string representations
of objects within Time-Stream.
"""

import sys
import textwrap
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from time_stream import TimeFrame


def timeframe_repr(tf: "TimeFrame") -> str:
    """Return a string representation of the TimeFrame dataframe.

    Args:
        tf: A TimeFrame object

    Returns:
        A formatted multi-line string representation of the TimeFrame object, including time properties, columns,
        flag systems, and metadata.
    """

    estimated_size = estimate_object_size(tf) + estimate_object_size(tf.df)

    lines = [f"<time_stream.TimeFrame> Size (estimated): {format_object_size_str(estimated_size)}"]

    repr_items = {
        "Time properties": format_time_properties(tf),
        "Columns": format_timeframe_columns(tf),
    }

    if tf.flag_systems:
        repr_items["Flag systems"] = tf.flag_systems

    if tf.metadata:
        repr_items["Metadata"] = tf.metadata

    lines.extend(format_repr_items(repr_items))
    return "\n".join(lines)


def format_time_properties(tf: "TimeFrame") -> dict:
    """Format TimeFrame time property information for display in TimeFrame representation.

    Args:
        tf: A TimeFrame object.

    Returns:
        A dictionary mapping time properties to their values
    """
    return {
        "Time column": (tf.time_name, format_timeframe_column_samples(tf.df[tf.time_name])),
        "Type": tf.df.schema[tf.time_name],
        "Resolution": tf.resolution,
        "Offset": tf.offset,
        "Alignment": tf.alignment,
        "Periodicity": tf.periodicity,
    }


def format_timeframe_columns(tf: "TimeFrame") -> dict:
    """Format TimeFrame column information for display in TimeFrame representation.

    Args:
        tf: A TimeFrame object with columns and associated metadata.

    Returns:
        A dictionary mapping column names to tuples containing:
        - dtype: The data type of the column
        - size: Formatted size string
        - samples: Sample values from the column
        - flags (optional): Flag system information if applicable
    """
    column_items = {}

    for col in tf.columns:
        dtype = tf.df.schema[col]
        size = estimate_object_size(tf.df[col])

        column_items[col] = (dtype, format_object_size_str(size), format_timeframe_column_samples(tf.df[col]))

        # Add flag system information if this column has flags
        if col in tf.flag_columns:
            flag_col = tf.get_flag_column(col)
            column_items[col] += (f"(flags={flag_col.flag_system.system_name()})",)

    return column_items


def estimate_object_size(obj: Any) -> int:
    """Estimate the memory size of an object in bytes.

    Args:
        obj: Any Python object. Special handling for Polars DataFrames and Series.

    Returns:
        Estimated size in bytes. Uses Polars estimation for DataFrame/Series objects,
        and ``sys.getsizeof()`` for other Python objects.
    """
    if isinstance(obj, (pl.DataFrame, pl.Series)):
        return int(obj.estimated_size())
    else:
        return sys.getsizeof(obj)


def format_object_size_str(size_bytes: int) -> str:
    """Convert byte size to human-readable format with appropriate units.

    Args:
        size_bytes: Size in bytes to format.

    Returns:
        Formatted string with size and unit (B, KB, MB, GB, TB, or PB).
    """
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    for u in units:
        size_bytes /= 1024.0
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {u}"
    return f"{size_bytes:.2f} PB"


def format_timeframe_column_samples(column_values: pl.Series) -> str:
    """Create a string showing first and last values of a TimeFrame column.

    Args:
        column_values: Polars Series containing the column data.

    Returns:
        String representation in the format "[first, ..., last]"
    """
    head_val = column_values.head(1).item()
    tail_val = column_values.tail(1).item()

    # Only show ellipsis if there are more than 2 values
    if len(column_values) > 2:
        final_values = [str(head_val), "...", str(tail_val)]
    else:
        final_values = [str(head_val), str(tail_val)]

    return f"[{', '.join(final_values)}]"


def format_repr_items(repr_items: dict) -> list[str]:
    """Format nested dictionary items into indented, aligned text lines.

    Args:
        repr_items: Dictionary with section headers as keys and nested dictionaries as values.

    Returns:
        List of formatted strings with headers and key-value pairs, properly aligned and indented for display.
    """

    lines = []

    # Calculate maximum key length for alignment across all sections
    max_key_length = max([len(key) for _, header_dict in repr_items.items() for key in header_dict])

    for header, header_dict in repr_items.items():
        lines.append(format_headers(header))

        for key, value in header_dict.items():
            key_length = len(key)

            # Work out how much padding needs to be added to the key for alignment
            padding = max_key_length - key_length

            # Ensure value is always a tuple for consistent processing
            values = value if isinstance(value, tuple) else (value,)

            lines.append(format_key_value_pairs(key + " " * padding, *values))
    return lines


def format_headers(header: str) -> str:
    """Format a section header string.

    Args:
        header: The header text to format.

    Returns:
        Capitalized header string with trailing colon.
    """
    return f"{header.capitalize()}:"


def format_key_value_pairs(key: str, *args) -> str:
    """Format a key with one or more values into an indented, aligned line.

    Args:
        key: The key string
        *args: Variable number of values to display after the key

    Returns:
        Formatted string with consistent indent, key, colon separator, and values.
    """
    if key is None or args is None:
        return ""
    values_str = "  ".join([textwrap_format(value) for value in args])
    return f"    {key} : {values_str}"


def textwrap_format(value: Any, width: int = 80) -> str:
    """Truncate long string representations to fit within a specified width.

    Args:
        value: Any value to be converted to string and potentially truncated.
        width: Maximum character width before truncation.

    Returns:
        String representation of value, truncated with "..." if exceeds width.
    """
    return textwrap.shorten(str(value), width=width, placeholder="...")
