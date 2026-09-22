"""Examples for the "Rolling aggregation" user guide page."""

import time_stream as ts
from examples._helpers import sample_frame


def rolling_mean_example() -> None:
    """Rolling 3-hour mean over a 15-minute flow series."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # [start:rolling_mean_example]
    tf_rolling = tf.rolling_aggregate("PT3H", "mean", "flow")

    print(tf_rolling.df)
    # [end:rolling_mean_example]


def rolling_missing_criteria_example() -> None:
    """Rolling mean that only emits a value when at least 3 points are present in the window."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # [start:rolling_missing_criteria_example]
    tf_rolling = tf.rolling_aggregate(
        "PT3H",
        "mean",
        "flow",
        missing_criteria=("available", 3),
    )

    print(tf_rolling.df)
    # [end:rolling_missing_criteria_example]


def rolling_nth_example() -> None:
    """Rolling ``nth`` aggregation - pick a fixed position within each trailing window."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M", periodicity="PT15M")
    # [start:rolling_nth_example]
    # Trailing 1-hour window, n=1: the first time step in each window - i.e. the value from
    # (just under) an hour before the current timestamp, or null if that reading is missing.
    tf_rolling = tf.rolling_aggregate("PT1H", "nth", "flow", n=1)

    print(tf_rolling.df)
    # [end:rolling_nth_example]
