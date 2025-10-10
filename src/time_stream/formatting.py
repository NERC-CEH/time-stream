"""String formatting routines for __repr__ operations"""
import sys
import textwrap
from typing import Any

import polars as pl



def timeframe_repr(tf) -> str:
    """Return a string representation of the TimeFrame dataframe."""
    time_values = tf.df[tf.time_name]
    estimated_size = estimate_object_size(tf) + estimate_object_size(tf.df)
    min_datetime = time_values.min()
    max_datetime = time_values.max()

    lines = [f"<time_stream.TimeFrame> Size: {format_object_size_str(estimated_size)}"]

    repr_items = {
        "Time properties": {
            "Time column": (tf.time_name, format_timeframe_column_samples(time_values, True)),
            "Resolution": tf.resolution,
            "Offset": tf.offset,
            "Alignment": tf.alignment,
            "Periodicity": tf.periodicity,
            "Duration": max_datetime - min_datetime
        },
        "Columns": format_timeframe_columns(tf)
    }

    if tf.flag_systems:
        repr_items["Flag systems"] = tf.flag_systems

    if tf.metadata:
        repr_items["Metadata"] = tf.metadata

    lines.extend(format_repr_items(repr_items))
    return "\n".join(lines)


def format_timeframe_columns(tf) -> dict:
    column_items = {}

    for col in tf.columns:
        dtype = tf.df.schema[col]
        size = estimate_object_size(tf.df[col])

        column_items[col] = (dtype, format_object_size_str(size), format_timeframe_column_samples(tf.df[col]))

        if col in tf.flag_columns:
            flag_col = tf.get_flag_column(col)
            column_items[col] += (f"(flags={flag_col.flag_system.system_name()})",)

    return column_items


def estimate_object_size(obj: Any) -> int:
    if isinstance(obj, (pl.DataFrame, pl.Series)):
        return int(obj.estimated_size())
    else:
        return sys.getsizeof(obj)


def format_object_size_str(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    units = ["KB", "MB", "GB", "TB", "PB"]
    for u in units:
        size_bytes /= 1024.0
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {u}"
    return f"{size_bytes:.2f} PB"


def format_timeframe_column_samples(column_values: pl.Series, format_time=False) -> str:
    if format_time:
        column_values = column_values.dt.to_string("%Y-%m-%d %H:%M:%S%.6f")

    head_val = column_values.head(1).item()
    tail_val = column_values.tail(1).item()

    mid_value = "..." if len(column_values) > 2 else ""
    final_values = [str(head_val), mid_value, str(tail_val)]

    return f"[{", ".join(final_values)}]"


def format_repr_items(repr_items: dict) -> list[str]:
    lines = []
    max_key_length = max([len(key) for _, header_dict in repr_items.items() for key in header_dict])

    for header, header_dict in repr_items.items():
        lines.append(format_headers(header))

        for key, value in header_dict.items():
            key_length = len(key)
            padding = max_key_length - key_length
            if not isinstance(value, tuple):
                value = (value,)
            lines.append(format_key_value_pairs(key + " " * padding, *value))
    return lines


def format_headers(header: str):
    return f"{header.capitalize()}:"


def format_key_value_pairs(key: str, *args):
    if key is None or args is None:
        return None
    values_str = "  ".join([textwrap_format(value) for value in args])
    return f"    {key} : {values_str}"


def textwrap_format(value: Any, width: int = 80) -> str:
    return textwrap.shorten(str(value), width=width, placeholder="...")
