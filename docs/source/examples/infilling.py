"""Examples for the "Infilling" user guide page."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import time_stream as ts
from examples._helpers import sample_frame


def _alt_frame() -> pl.DataFrame:
    """A complete companion series - ``alt_flow`` running 25% above ``flow`` - to stand in for the gaps."""
    return sample_frame(complete=True).with_columns(pl.col("flow").mul(1.25).alias("alt_flow")).drop("flow")


def _gappy_daily_series() -> ts.TimeFrame:
    """A padded daily series with one-, two- and three-day gaps, for the infill-comparison plot."""
    np.random.seed(42)
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),  # one-day gap
        datetime(2024, 1, 4),
        datetime(2024, 1, 5),
        datetime(2024, 1, 6),
        datetime(2024, 1, 7),  # two-day gap
        datetime(2024, 1, 10),
        datetime(2024, 1, 11),
        datetime(2024, 1, 12),
        datetime(2024, 1, 16),  # three-day gap
    ]
    df = pl.DataFrame({"time": dates, "original": np.arange(len(dates)) * 0.5 + np.random.normal(0, 2, len(dates))})
    return ts.TimeFrame(df=df, time_name="time", resolution="P1D", periodicity="P1D").pad()


def _infill_comparison() -> pl.DataFrame:
    """Run every interpolation method over one gappy series and join the results side by side."""
    tf = _gappy_daily_series()
    result = tf.df.clone()
    for method in ["linear", "quadratic", "cubic", "pchip", "akima"]:
        infilled = tf.infill(method, "original")
        infilled = infilled.with_df(infilled.df.rename({"original": method}))
        result = result.join(infilled.df, on="time", how="full").drop("time_right")
    return result


def time_stream_example() -> None:
    """Two-stage infill: linear for single-step gaps, PCHIP for gaps up to three steps."""
    df = sample_frame()
    # fmt: off
    # [start:time_stream_example]
    import time_stream as ts

    tf = ts.TimeFrame(df, "time", resolution="PT15M", periodicity="PT15M")

    tf_infill = tf.infill(
        "linear", "flow", max_gap_size=1
    ).infill(
        "pchip", "flow", max_gap_size=3
    )
    # [end:time_stream_example]
    # fmt: on
    with pl.Config(tbl_rows=16):
        print(tf_infill)


def main_data() -> None:
    """The primary 15-minute flow series, with gaps."""
    # [start:main_data]
    print(sample_frame())
    # [end:main_data]


def alt_data() -> None:
    """The complete companion series that stands in for the gaps."""
    # [start:alt_data]
    print(_alt_frame())
    # [end:alt_data]


def alt_data_infill() -> None:
    """Fill gaps from an alternative column with a single fixed correction factor."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    alt_df = _alt_frame()
    # fmt: off
    # [start:alt_data_infill]
    tf_infill = tf.infill(
        "alt_data",
        "flow",
        alt_df=alt_df,
        correction_factor=0.75,
        alt_data_column="alt_flow"
    )
    # [end:alt_data_infill]
    # fmt: on
    print(tf_infill.df)


def alt_data_dynamic_infill() -> None:
    """Fill gaps from an alternative column, computing the correction factor per gap from a time window."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    alt_df = _alt_frame()
    # [start:alt_data_dynamic_infill]
    tf_infill = tf.infill(
        "alt_data_dynamic",
        "flow",
        alt_df=alt_df,
        alt_data_column="alt_flow",
        window_size="PT1H",
    )
    # [end:alt_data_dynamic_infill]
    print(tf_infill.df)


def alt_data_dynamic_window_size_formats() -> None:
    """The dynamic window can be given as an ISO 8601 string, an isoperiod ``Period``, or a ``timedelta``."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    alt_df = _alt_frame()
    # [start:alt_data_dynamic_window_size_formats]
    from datetime import timedelta

    from isoperiod import Period

    # window_size can be specified as an ISO 8601 duration string:
    tf_infill_str = tf.infill(
        "alt_data_dynamic",
        "flow",
        alt_df=alt_df,
        alt_data_column="alt_flow",
        window_size="PT1H",
    )

    # ...as a isoperiod Period object:
    tf_infill_period = tf.infill(
        "alt_data_dynamic",
        "flow",
        alt_df=alt_df,
        alt_data_column="alt_flow",
        window_size=Period.of_hours(1),
    )

    # ...or as a timedelta:
    tf_infill_timedelta = tf.infill(
        "alt_data_dynamic",
        "flow",
        alt_df=alt_df,
        alt_data_column="alt_flow",
        window_size=timedelta(hours=1),
    )
    # [end:alt_data_dynamic_window_size_formats]
    del tf_infill_str, tf_infill_period, tf_infill_timedelta


def alt_data_dynamic_infill_with_thresholds() -> None:
    """Dynamic alt-data infill that only applies where the window holds 2-4 supporting points."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    alt_df = _alt_frame()
    # [start:alt_data_dynamic_infill_with_thresholds]
    tf_infill = tf.infill(
        "alt_data_dynamic",
        "flow",
        alt_df=alt_df,
        alt_data_column="alt_flow",
        window_size="PT2H",
        min_threshold=2,
        max_threshold=4,
    )
    # [end:alt_data_dynamic_infill_with_thresholds]
    print(tf_infill.df)


def alt_data_dynamic_infill_one_sided() -> None:
    """Dynamic alt-data infill whose window looks only backwards from each gap."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    alt_df = _alt_frame()
    # [start:alt_data_dynamic_infill_one_sided]
    tf_infill = tf.infill(
        "alt_data_dynamic",
        "flow",
        alt_df=alt_df,
        alt_data_column="alt_flow",
        window_size="PT1H",
        window_side="left",
    )
    # [end:alt_data_dynamic_infill_one_sided]
    print(tf_infill.df)


def flagged_infill() -> None:
    """Infill and record which rows were filled in a flag column."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # fmt: off
    # [start:flagged_infill]
    # Register a flag system and create a flag column before running infill
    tf.register_flag_system("INFILL_FLAGS", ["INFILLED"])
    tf.init_flag_column("INFILL_FLAGS", "flow_flags")

    tf_infill = tf.infill(
        "linear", "flow", max_gap_size=3, flag_params=("flow_flags", "INFILLED")
    )
    # [end:flagged_infill]
    # fmt: on
    with pl.Config(tbl_rows=16):
        print(tf_infill.df)


def _hourly_frame_with_missing_rows() -> ts.TimeFrame:
    """An hourly flow series with two rows missing altogether, one null value, and a flag column."""
    df = pl.DataFrame(
        {
            "time": [datetime(2024, 1, 1, hour) for hour in (0, 1, 4, 5, 6)],
            "flow": [10.0, 20.0, 50.0, None, 70.0],
        }
    )
    tf = ts.TimeFrame(df, "time", resolution="PT1H", periodicity="PT1H")
    tf.register_flag_system("INFILL_FLAGS", ["INFILLED"])
    tf.init_flag_column("INFILL_FLAGS", "flow_flags")
    return tf


def missing_rows_data() -> None:
    """An hourly series missing its 02:00 and 03:00 rows, with a null value at 05:00."""
    # [start:missing_rows_data]
    print(_hourly_frame_with_missing_rows().df)
    # [end:missing_rows_data]


def missing_rows_infill() -> None:
    """Infill a series whose gaps are missing rows rather than null values."""
    tf = _hourly_frame_with_missing_rows()
    # fmt: off
    # [start:missing_rows_infill]
    tf_infill = tf.infill(
        "linear", "flow", flag_params=("flow_flags", "INFILLED")
    )
    # [end:missing_rows_infill]
    # fmt: on
    print(tf_infill.df)


def all_infills() -> None:
    """Every interpolation method over one gappy series, side by side."""
    # [start:all_infills]
    print(_infill_comparison())
    # [end:all_infills]


def plot_all_infills() -> None:
    """Plot the infill comparison - one line per interpolation method."""
    df = _infill_comparison()

    plt.figure(figsize=(10, 6))
    x_values = df["time"].to_list()
    for col in [col for col in df.columns if col != "time"]:
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
