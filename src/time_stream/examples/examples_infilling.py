from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import time_stream as ts
from time_stream.examples.utils import get_example_df, suppress_output


def time_stream_example() -> None:
    # fmt: off
    with suppress_output():
        df = get_example_df(library="polars")

    # [start_block_1]
    import time_stream as ts

    # Wrap the DataFrame in a TimeFrame object
    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

    # Infill gaps
    tf_infill = tf.infill(
        "linear", "flow", max_gap_size=1
    ).infill(
        "pchip", "flow", max_gap_size=3
    )
    # [end_block_1]

    with pl.Config(tbl_rows=16):
        print(tf_infill)
    # fmt: on


def create_simple_time_series_with_gaps() -> ts.TimeFrame:
    np.random.seed(42)

    # Set up a daily time series with varying gaps
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),  # One-day gap,
        datetime(2024, 1, 4),
        datetime(2024, 1, 5),
        datetime(2024, 1, 6),
        datetime(2024, 1, 7),  # Two-day gap,
        datetime(2024, 1, 10),
        datetime(2024, 1, 11),
        datetime(2024, 1, 12),
        # Three-day gap,
        datetime(2024, 1, 16),
    ]
    df = pl.DataFrame(
        {
            "time": dates,
            "original": np.arange(len(dates)) * 0.5 + np.random.normal(0, 2, len(dates)),
        }
    )

    tf = ts.TimeFrame(df=df, time_name="time", resolution="P1D", periodicity="P1D")
    tf.pad()
    return tf


def all_infills() -> pl.DataFrame:
    with suppress_output():
        tf = create_simple_time_series_with_gaps()

    methods = ["linear", "quadratic", "cubic", "pchip", "akima"]
    result = tf.df.clone()
    for method in methods:
        tf_infilled = tf.infill(method, "original")
        tf_infilled = tf_infilled.with_df(tf_infilled.df.rename({"original": method}))
        result = result.join(tf_infilled.df, on="time", how="full").drop("time_right")

    with pl.Config(tbl_rows=-1):
        print(result)

    return result


def plot_all_infills() -> None:
    with suppress_output():
        df = all_infills()

    plt.figure(figsize=(10, 6))

    datetime_col = "time"
    x_values = df[datetime_col].to_list()
    for col in [col for col in df.columns if col != datetime_col]:
        y_values = df[col].to_list()
        if col == "original":
            plt.scatter(x_values, y_values, label=col, s=70, c="black", zorder=10)
        else:
            plt.plot(x_values, y_values, label=col, linewidth=2)
            plt.scatter(x_values, y_values, s=30)

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Different infilling methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
