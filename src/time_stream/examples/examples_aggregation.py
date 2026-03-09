import polars as pl

import time_stream as ts
from time_stream.examples.utils import get_example_df, suppress_output


def pandas_example() -> None:
    with suppress_output():
        # fmt: off
        # [start_block_1]
        import pandas as pd

        df = get_example_df(library="pandas")

        # Shift time back by 9 hours to align water year boundaries with midnight Oct 1st
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
        # [end_block_1]
        df_amax = df_amax[[
            "time", "time_of_max_flow", "max_flow", "count_flow", "expected_count_time"
        ]]
    print(pl.DataFrame(df_amax))
    # fmt: on


def polars_example() -> None:
    with suppress_output():
        # fmt: off
        # [start_block_2]
        import polars as pl

        df = get_example_df(library="polars")

        # Shift time back by 9 hours to align water year boundaries with midnight Oct 1st
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
            pl.col("time").filter(
                pl.col("flow") == pl.col("flow").max()
            ).first().alias("time_of_max_flow"),
        ]).sort("water_year")

        # Adjust the time column back to represent actual water year starts (Oct 1 9am)
        df_amax = df_amax.with_columns(pl.datetime(pl.col("water_year"), 10, 1, 9).alias("time"))

        # Calculate FULL water year expected counts
        # Each water year should have the number of 15-min intervals (900 seconds)
        #   from Oct 1 9am to next Oct 1 9am
        df_amax = df_amax.with_columns(
            ((pl.col("time").dt.offset_by("1y") - pl.col("time")).dt.total_seconds() // 900
             ).alias("expected_count_time")
        )
        # [end_block_2]
        df_amax = df_amax[
            "time", "time_of_max_flow", "max_flow", "count_flow", "expected_count_time"
        ]
    print(df_amax)
    # fmt: on


def time_stream_example() -> None:
    # fmt: off
    with suppress_output():
        # [start_block_3]
        import time_stream as ts

        df = get_example_df(library="polars")

        # Wrap the DataFrame in a TimeFrame object
        tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

        # Perform the aggregation to a water-year
        tf_amax = tf.aggregate("P1Y+9MT9H", "max", "flow")


























        # that's it...
        # [end_block_3]
        df_amax = tf_amax.df[
            "time", "time_of_max_flow", "max_flow", "count_flow", "expected_count_time"
        ]
    print(df_amax)
    # fmt: on


def aggregation_time_window_example() -> None:
    from datetime import time

    with suppress_output():
        df = get_example_df(library="polars")

    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

    # [start_block_4]
    tf_agg = tf.aggregate("P1D", "mean", "flow", time_window=(time(9, 0), time(17, 0)))
    # [end_block_4]
    print(tf_agg.df)


def aggregation_missing_criteria_example() -> None:
    with suppress_output():
        df = get_example_df(library="polars")

    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")
    # [start_block_5]
    tf_agg = tf.aggregate("P1M", "mean", "flow", missing_criteria=("missing", 150))
    # [end_block_5]
    print(tf_agg.df)
