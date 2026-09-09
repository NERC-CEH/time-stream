"""Examples for the "Aggregation" user guide page."""

from datetime import time

import polars as pl

import time_stream as ts
from examples._helpers import sample_frame


def sample_data() -> None:
    """The 15-minute river-flow series these examples aggregate."""
    # [start:sample_data]
    print(sample_frame())
    # [end:sample_data]


def pandas_example() -> None:
    """Annual maximum on the UK water year, done by hand with pandas ``resample`` - the comparison baseline."""
    # fmt: off
    # [start:pandas_example]
    import pandas as pd

    df = sample_frame(library="pandas")

    # Shift time back 9 hours to align water year boundaries with midnight Oct 1st
    # This makes Oct 1 9am appear as Oct 1 midnight for resampling purposes
    df.index = df.index - pd.Timedelta(hours=9)

    # Aggregate using resample
    # We can use "YS-OCT" (Year Start in October) as the resample period
    resampled = df.resample("YS-OCT")

    df_amax = pd.DataFrame(
        {
            "max_flow": resampled["flow"].max(),
            "count_flow": resampled["flow"].count(),
        }
    )

    # Get the datetime that each max value occurred on
    max_dates = resampled["flow"].idxmax() + pd.Timedelta(hours=9)
    df_amax["time_of_max_flow"] = max_dates

    # Adjust the time column back to represent actual water year starts (Oct 1 9am)
    df_amax.index = df_amax.index + pd.Timedelta(hours=9)
    df_amax.index.name = "time"

    # Calculate FULL water year expected counts
    # Each water year should have the number of 15-min intervals (900 seconds)
    #   from Oct 1 9am to next Oct 1 9am
    df_amax["expected_count_time"] = (
        (
            (df_amax.index + pd.DateOffset(years=1)) - df_amax.index
        ).total_seconds() / 900
    )

    df_amax = df_amax.reset_index()
    # [end:pandas_example]
    # fmt: on
    df_amax = df_amax[["time", "time_of_max_flow", "max_flow", "count_flow", "expected_count_time"]]
    print(pl.DataFrame(df_amax))


def polars_example() -> None:
    """The same water-year annual maximum, done by hand with polars ``group_by`` - the comparison baseline."""
    # fmt: off
    # [start:polars_example]
    import polars as pl

    df = sample_frame(library="polars")

    # Shift time back 9 hours to align water year boundaries with midnight Oct 1st
    # This makes Oct 1 9am appear as Oct 1 midnight for resampling purposes
    df = df.with_columns(pl.col("time").dt.offset_by("-9h").alias("shifted_time"))
    # Assign water year based on shifted time
    df = df.with_columns(
        [
            pl.when(pl.col("shifted_time").dt.month() >= 10)
            .then(pl.col("shifted_time").dt.year())
            .otherwise(pl.col("shifted_time").dt.year() - 1)
            .alias("water_year")
        ]
    )

    df_amax = df.group_by("water_year").agg([
        pl.col("flow").max().alias("max_flow"),
        pl.col("flow").count().alias("count_flow"),
        # Get the datetime that each max value occurred on
        pl.col("time").filter(pl.col("flow") == pl.col("flow").max())
            .first().alias("time_of_max_flow"),
    ]).sort("water_year")

    # Adjust the time column back to represent actual water year starts (Oct 1 9am)
    df_amax = df_amax.with_columns(
        pl.datetime(pl.col("water_year"), 10, 1, 9).alias("time")
    )

    # Calculate FULL water year expected counts
    # Each water year should have the number of 15-min intervals (900 seconds)
    #   from Oct 1 9am to next Oct 1 9am
    df_amax = df_amax.with_columns(
        ((pl.col("time").dt.offset_by("1y") - pl.col("time")).dt.total_seconds()
         // 900).alias("expected_count_time")
    )
    # [end:polars_example]
    # fmt: on
    df_amax = df_amax["time", "time_of_max_flow", "max_flow", "count_flow", "expected_count_time"]
    print(df_amax)


def time_stream_example() -> None:
    """The same water-year annual maximum, in a single ``TimeFrame.aggregate`` call."""
    # fmt: off
    # [start:time_stream_example]
    import time_stream as ts

    df = sample_frame(library="polars")

    # Wrap the DataFrame in a TimeFrame object
    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

    # Perform the aggregation to a water-year
    tf_amax = tf.aggregate("P1Y+9MT9H", "max", "flow")


























    # that's it...
    # [end:time_stream_example]
    # fmt: on
    df_amax = tf_amax.df["time", "time_of_max_flow", "max_flow", "count_flow", "expected_count_time"]
    print(df_amax)


def aggregation_time_window_example() -> None:
    """Aggregate only the readings between 09:00 and 17:00 of each day."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # [start:aggregation_time_window_example]
    tf_agg = tf.aggregate("P1D", "mean", "flow", time_window=(time(9, 0), time(17, 0)))

    print(tf_agg.df)
    # [end:aggregation_time_window_example]


def aggregation_missing_criteria_example() -> None:
    """Monthly mean that returns null for any month missing more than 150 readings."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # [start:aggregation_missing_criteria_example]
    tf_agg = tf.aggregate("P1M", "mean", "flow", missing_criteria=("missing", 150))

    print(tf_agg.df)
    # [end:aggregation_missing_criteria_example]


def aggregation_nth_example() -> None:
    """``nth`` aggregation - pick a fixed position (here midday) within each daily window."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # [start:aggregation_nth_example]
    # 15-minute data: the 49th reading of the day falls at 12:00 (noon)
    tf_agg = tf.aggregate("P1D", "nth", "flow", n=49)

    print(tf_agg.df)
    # [end:aggregation_nth_example]
