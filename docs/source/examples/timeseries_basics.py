"""Examples for the "Quick start" page."""

from datetime import datetime, timedelta

import polars as pl

import time_stream as ts
from time_stream.exceptions import DuplicateTimeError

_DATES = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
_TEMPERATURE = [20.5, 21.0, 19.1, 26.0, 24.2, 26.6, 28.4, 30.9, 31.0, 29.1]
_PRECIPITATION = [0.0, 0.0, 5.1, 10.2, 2.0, 0.2, 0.0, 3.0, 1.6, 0.0]


def _daily_frame() -> pl.DataFrame:
    """Ten days of daily temperature and precipitation."""
    return pl.DataFrame({"time": _DATES, "temperature": _TEMPERATURE, "precipitation": _PRECIPITATION})


def _daily_frame_at_9am() -> pl.DataFrame:
    """The same ten-day frame, timestamped at 09:00 each day."""
    at_9am = [d.replace(hour=9) for d in _DATES]
    return pl.DataFrame({"time": at_9am, "temperature": _TEMPERATURE, "precipitation": _PRECIPITATION})


def _annual_maxima_frame() -> pl.DataFrame:
    """Three annual-maximum readings, timestamped at 09:00."""
    dates = [datetime(2023, 5, 21, 9), datetime(2024, 1, 18, 9), datetime(2025, 9, 1, 9)]
    return pl.DataFrame({"time": dates, "flow": [23.6, 42.1, 34.6]})


def _duplicate_rows_frame() -> pl.DataFrame:
    """A monthly frame with three repeated timestamps."""
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 1),  # duplicate
        datetime(2023, 2, 1),
        datetime(2023, 3, 1),
        datetime(2023, 4, 1),
        datetime(2023, 5, 1),
        datetime(2023, 6, 1),
        datetime(2023, 6, 1),  # duplicate
        datetime(2023, 6, 1),  # duplicate
        datetime(2023, 7, 1),
    ]
    return pl.DataFrame(
        {
            "time": dates,
            "temperature": [20, None, 19, 26, 24, 26, 28, 30, None, 29],
            "precipitation": [None, 0, 5, 10, 2, 0, None, 3, 4, 0],
        }
    )


def create_sample_dataframe() -> None:
    """Build a small Polars DataFrame with a time column and two value columns."""
    # [start:create_sample_dataframe]
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "time": dates,
            "temperature": [20.5, 21.0, 19.1, 26.0, 24.2, 26.6, 28.4, 30.9, 31.0, 29.1],
            "precipitation": [0.0, 0.0, 5.1, 10.2, 2.0, 0.2, 0.0, 3.0, 1.6, 0.0],
        }
    )
    # [end:create_sample_dataframe]
    del df


def create_simple_time_series() -> None:
    """Wrap a DataFrame in a TimeFrame with just a time column - everything else inferred."""
    df = _daily_frame()
    # [start:create_simple_time_series]
    tf = ts.TimeFrame(df=df, time_name="time")

    print(tf)
    # [end:create_simple_time_series]


def show_default_resolution() -> None:
    """The resolution / offset / periodicity / anchor a bare TimeFrame infers."""
    tf = ts.TimeFrame(_daily_frame(), time_name="time")
    # [start:show_default_resolution]
    print("resolution:", tf.resolution)
    print("offset:", tf.offset)
    print("periodicity:", tf.periodicity)
    print("time anchor:", tf.time_anchor)
    # [end:show_default_resolution]


def create_simple_time_series_with_periods() -> None:
    """Give the TimeFrame an explicit daily resolution."""
    df = _daily_frame()
    # [start:create_simple_time_series_with_periods]
    tf = ts.TimeFrame(df=df, time_name="time", resolution="P1D")

    print("resolution:", tf.resolution)
    print("offset:", tf.offset)
    print("periodicity:", tf.periodicity)
    # [end:create_simple_time_series_with_periods]


def create_simple_time_series_with_periods2() -> None:
    """Daily resolution with a +9h offset - readings taken at 09:00."""
    df = _daily_frame_at_9am()
    # [start:create_simple_time_series_with_periods2]
    tf = ts.TimeFrame(df=df, time_name="time", resolution="P1D", offset="+T9H")

    print("resolution:", tf.resolution)
    print("offset:", tf.offset)
    print("periodicity:", tf.periodicity)
    # [end:create_simple_time_series_with_periods2]


def create_simple_time_series_with_periods3() -> None:
    """Daily resolution, +9h offset, and a water-year periodicity (one value per year)."""
    df = _annual_maxima_frame()
    # [start:create_simple_time_series_with_periods3]
    tf = ts.TimeFrame(
        df=df,
        time_name="time",
        resolution="P1D",  # daily sampling interval
        offset="+T9H",  # readings taken at 09:00
        periodicity="P1Y+9MT9H",  # at most one value per water year (1 Oct 09:00)
    )

    print("resolution:", tf.resolution)
    print("offset:", tf.offset)
    print("periodicity:", tf.periodicity)
    # [end:create_simple_time_series_with_periods3]


def create_simple_time_series_with_metadata() -> None:
    """Attach dataset-level and per-column metadata to a TimeFrame."""
    tf = ts.TimeFrame(_daily_frame(), time_name="time", resolution="P1D")

    # [start:with_metadata]
    metadata = {"location": "UKCEH Wallingford", "station_id": "ABC123"}
    tf = tf.with_metadata(metadata)

    print(tf.metadata)
    # [end:with_metadata]

    # [start:with_column_metadata]
    column_metadata = {
        "temperature": {"units": "°C", "description": "Average temperature"},
        "precipitation": {"units": "mm", "description": "Precipitation amount", "instrument_type": "Tipping bucket"},
    }
    tf = tf.with_column_metadata(column_metadata)

    print(tf.column_metadata)
    # [end:with_column_metadata]


def show_time_series_metadata() -> None:
    """Read metadata back off a TimeFrame at the dataset, column, and key level."""
    tf = ts.TimeFrame(_daily_frame(), time_name="time", resolution="P1D").with_metadata(
        {"location": "UKCEH Wallingford", "station_id": "ABC123"}
    )
    tf = tf.with_column_metadata({"temperature": {"units": "°C", "description": "Average temperature"}})
    # [start:show_time_series_metadata]
    print("all dataset metadata:", tf.metadata)
    print("one key:", tf.metadata["location"])

    print("all column metadata:", tf.column_metadata)
    print("one column:", tf.column_metadata["temperature"])
    print("one column key:", tf.column_metadata["temperature"]["units"])
    # [end:show_time_series_metadata]


def accessing_data() -> None:
    """Get at the underlying data - the whole DataFrame, or a column subset as a new TimeFrame."""
    tf = ts.TimeFrame(_daily_frame(), time_name="time", resolution="P1D")
    # [start:accessing_data]
    df = tf.df  # the whole thing, as a Polars DataFrame

    selected = tf.select(["temperature"])  # a column subset, still a TimeFrame
    selected = tf[["temperature"]]  # the same, via indexing

    print(type(selected))
    print(selected.df)
    # [end:accessing_data]
    del df


def duplicate_rows_data() -> None:
    """A monthly frame with a few repeated timestamps, for the ``on_duplicates`` examples."""
    # [start:duplicate_rows_data]
    print(_duplicate_rows_frame())
    # [end:duplicate_rows_data]


def duplicate_row_example_error() -> None:
    """``on_duplicates='error'`` (the default) raises on repeated timestamps."""
    df = _duplicate_rows_frame()
    # [start:duplicate_row_example_error]
    try:
        ts.TimeFrame(df, "time", on_duplicates="error")
    except DuplicateTimeError as err:
        print(err)
    # [end:duplicate_row_example_error]


def duplicate_row_example_keep_first() -> None:
    """``on_duplicates='keep_first'`` keeps the first row of each duplicate group."""
    df = _duplicate_rows_frame()
    # [start:duplicate_row_example_keep_first]
    tf = ts.TimeFrame(df, "time", on_duplicates="keep_first")

    print(tf.df)
    # [end:duplicate_row_example_keep_first]


def duplicate_row_example_keep_last() -> None:
    """``on_duplicates='keep_last'`` keeps the last row of each duplicate group."""
    df = _duplicate_rows_frame()
    # [start:duplicate_row_example_keep_last]
    tf = ts.TimeFrame(df, "time", on_duplicates="keep_last")

    print(tf.df)
    # [end:duplicate_row_example_keep_last]


def duplicate_row_example_drop() -> None:
    """``on_duplicates='drop'`` removes every row involved in a duplicate."""
    df = _duplicate_rows_frame()
    # [start:duplicate_row_example_drop]
    tf = ts.TimeFrame(df, "time", on_duplicates="drop")

    print(tf.df)
    # [end:duplicate_row_example_drop]


def duplicate_row_example_merge() -> None:
    """``on_duplicates='merge'`` coalesces duplicate rows, first non-null wins."""
    df = _duplicate_rows_frame()
    # [start:duplicate_row_example_merge]
    tf = ts.TimeFrame(df, "time", on_duplicates="merge")

    print(tf.df)
    # [end:duplicate_row_example_merge]


def add_new_column_to_df() -> None:
    """Add a derived column by editing the DataFrame and handing it back with ``with_df``."""
    tf = ts.TimeFrame(_daily_frame(), time_name="time", resolution="P1D")
    # [start:add_new_column_to_df]
    new_df = tf.df.with_columns((pl.col("temperature") * 1.8 + 32).alias("temperature_f"))
    tf = tf.with_df(new_df)

    print(tf.df)
    # [end:add_new_column_to_df]


def create_misaligned_row_example() -> None:
    """Show a series with rows off the resolution grid, before and after they are removed."""
    all_rows = [
        datetime(2020, 1, 1, 0, 0),
        datetime(2020, 1, 1, 0, 30),
        datetime(2020, 1, 1, 1, 0),
        datetime(2020, 1, 1, 1, 30),
        datetime(2020, 1, 1, 1, 31),  # off the 30-minute grid
        datetime(2020, 1, 1, 1, 32),
        datetime(2020, 1, 1, 1, 33),
        datetime(2020, 1, 1, 2, 0),
        datetime(2020, 1, 1, 2, 30),
    ]
    aligned = [d for d in all_rows if d.minute in (0, 30)]

    print("initial data:")
    print(pl.DataFrame({"time": all_rows}))
    print("\nresolved data, misaligned rows removed:")
    print(pl.DataFrame({"time": aligned}))
