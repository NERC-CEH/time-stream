from datetime import datetime, timedelta

import polars as pl

import time_stream as ts
from time_stream.examples.utils import suppress_output
from time_stream.exceptions import DuplicateTimeError


def create_simple_dataframe() -> pl.DataFrame:
    # [start_block_1]
    from datetime import datetime, timedelta

    import polars as pl

    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [20.5, 21.0, 19.1, 26.0, 24.2, 26.6, 28.4, 30.9, 31.0, 29.1]
    precipitation = [0.0, 0.0, 5.1, 10.2, 2.0, 0.2, 0.0, 3.0, 1.6, 0.0]

    df = pl.DataFrame({"time": dates, "temperature": temperatures, "precipitation": precipitation})
    # [end_block_1]

    return df


def create_simple_dataframe_with_offset() -> pl.DataFrame:
    dates = [datetime(2023, 1, 1, 9) + timedelta(days=i) for i in range(10)]
    temperatures = [20.5, 21.0, 19.1, 26.0, 24.2, 26.6, 28.4, 30.9, 31.0, 29.1]
    precipitation = [0.0, 0.0, 5.1, 10.2, 2.0, 0.2, 0.0, 3.0, 1.6, 0.0]

    df = pl.DataFrame({"time": dates, "temperature": temperatures, "precipitation": precipitation})

    return df


def create_simple_dataframe_amax() -> pl.DataFrame:
    dates = [datetime(2023, 5, 21, 9), datetime(2024, 1, 18, 9), datetime(2025, 9, 1, 9)]
    flows = [23.6, 42.1, 34.6]
    df = pl.DataFrame({"time": dates, "flow": flows})

    return df


def create_simple_time_series() -> ts.TimeFrame:
    df = create_simple_dataframe()
    # [start_block_2]
    tf = ts.TimeFrame(
        df=df,
        time_name="time",  # Specify which column contains the primary datetime values
    )
    print(tf)
    # [end_block_2]
    return tf


def show_default_resolution() -> None:
    with suppress_output():
        tf = create_simple_time_series()
    # [start_block_3]
    print(tf.resolution)
    print(tf.offset)
    print(tf.periodicity)
    print(tf.time_anchor)
    # [end_block_3]


def create_simple_time_series_with_periods() -> ts.TimeFrame:
    df = create_simple_dataframe()

    # [start_block_4]
    tf = ts.TimeFrame(
        df=df,
        time_name="time",
        resolution="P1D",  # Sampling interval of 1 day
    )

    print("resolution=", tf.resolution)
    print("offset=", tf.offset)
    print("periodicity=", tf.periodicity)
    # [end_block_4]
    return tf


def create_simple_time_series_with_periods2() -> ts.TimeFrame:
    df_offset = create_simple_dataframe_with_offset()

    # [start_block_8]
    tf = ts.TimeFrame(
        df=df_offset,
        time_name="time",
        resolution="P1D",  # Sampling interval of 1 day
        offset="+T9H",  # Values are measured at 09:00am on each day
    )

    print("resolution=", tf.resolution)
    print("offset=", tf.offset)
    print("periodicity=", tf.periodicity)
    # [end_block_8]
    return tf


def create_simple_time_series_with_periods3() -> ts.TimeFrame:
    df_amax = create_simple_dataframe_amax()

    # [start_block_9]
    tf = ts.TimeFrame(
        df=df_amax,
        time_name="time",
        resolution="P1D",  # Sampling interval of 1 day
        offset="+T9H",  # Values are measured at 09:00am on each day
        periodicity="P1Y+9MT9H",  # We only expect 1 value per "water-year" (1st Oct 09:00)
    )

    print("resolution=", tf.resolution)
    print("offset=", tf.offset)
    print("periodicity=", tf.periodicity)
    # [end_block_9]
    return tf


def create_simple_time_series_with_metadata() -> ts.TimeFrame:
    with suppress_output():
        tf = create_simple_time_series_with_periods()

    # [start_block_5]
    metadata = {"location": "UKCEH Wallingford", "station_id": "ABC123"}

    tf = tf.with_metadata(metadata)
    # [end_block_5]

    # [start_block_6]
    column_metadata = {
        "temperature": {"units": "°C", "description": "Average temperature"},
        "precipitation": {
            "units": "mm",
            "description": "Precipitation amount",
            "instrument_type": "Tipping bucket",
            # Note that metadata keys are not required to be the same for all columns
        },
    }

    tf = tf.with_column_metadata(column_metadata)
    # [end_block_6]
    return tf


def show_time_series_metadata() -> None:
    tf = create_simple_time_series_with_metadata()
    # [start_block_7]
    print("Dataset-level metadata:")
    print("")
    print("All: ", tf.metadata)
    print("Specific key: ", tf.metadata["location"])
    print("")
    print("Column-level metadata:")
    print("")
    print("All: ", tf.column_metadata)
    print("Specific column: ", tf.column_metadata["temperature"])
    print("Specific column key: ", tf.column_metadata["temperature"]["units"])
    # [end_block_7]


def accessing_data() -> None:
    tf = create_simple_time_series_with_metadata()

    # [start_block_10]
    # Get the full DataFrame
    df = tf.df
    # [end_block_10]

    _ = df  # Just to get around ruff complaining that the variable is unused.

    # [start_block_12]
    # Select multiple columns as a TimeFrame
    selected_tf = tf.select(["temperature"])
    # or
    selected_tf = tf[["temperature"]]
    print("Type: ", type(selected_tf))
    print(selected_tf.df)
    # [end_block_12]


def create_df_with_duplicate_rows() -> pl.DataFrame:
    # [start_block_27]
    dates = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 1),  # Duplicate
        datetime(2023, 2, 1),
        datetime(2023, 3, 1),
        datetime(2023, 4, 1),
        datetime(2023, 5, 1),
        datetime(2023, 6, 1),
        datetime(2023, 6, 1),  # Duplicate
        datetime(2023, 6, 1),  # Duplicate
        datetime(2023, 7, 1),
    ]
    temperatures = [20, None, 19, 26, 24, 26, 28, 30, None, 29]
    precipitation = [None, 0, 5, 10, 2, 0, None, 3, 4, 0]

    df = pl.DataFrame({"time": dates, "temperature": temperatures, "precipitation": precipitation})

    print(df)
    # [end_block_27]
    return df


def duplicate_row_example_error() -> None:
    with suppress_output():
        df = create_df_with_duplicate_rows()

    try:
        # [start_block_28]
        ts.TimeFrame(df, "time", on_duplicates="error")
    # [end_block_28]
    except DuplicateTimeError as w:
        print(f"Warning: {w}")


def duplicate_row_example_keep_first() -> None:
    with suppress_output():
        df = create_df_with_duplicate_rows()

    # [start_block_29]
    tf = ts.TimeFrame(df, "time", on_duplicates="keep_first")
    # [end_block_29]
    print(tf.df)


def duplicate_row_example_keep_last() -> None:
    with suppress_output():
        df = create_df_with_duplicate_rows()

    # [start_block_30]
    tf = ts.TimeFrame(df, "time", on_duplicates="keep_last")
    # [end_block_30]
    print(tf.df)


def duplicate_row_example_drop() -> None:
    with suppress_output():
        df = create_df_with_duplicate_rows()

    # [start_block_31]
    tf = ts.TimeFrame(df, "time", on_duplicates="drop")
    # [end_block_31]
    print(tf.df)


def duplicate_row_example_merge() -> None:
    with suppress_output():
        df = create_df_with_duplicate_rows()

    # [start_block_32]
    tf = ts.TimeFrame(df, "time", on_duplicates="merge")
    # [end_block_32]
    print(tf.df)


def add_new_column_to_df() -> None:
    with suppress_output():
        tf = create_simple_time_series_with_periods()

    # [start_block_19]
    # Update the DataFrame by adding a new column
    new_df = tf.df.with_columns((pl.col("temperature") * 1.8 + 32).alias("temperature_f"))

    tf = tf.with_df(new_df)
    # [end_block_19]
    print(tf.df)
