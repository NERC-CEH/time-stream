"""Examples for the "Time of year aggregation" user guide page."""

from datetime import datetime

import numpy as np
import polars as pl

import time_stream as ts
from examples._helpers import sample_frame


def min_max_example() -> None:
    """The highest and lowest flow seen on each 15 min of the year, and when it was seen."""
    tf = ts.TimeFrame(sample_frame(), "time", resolution="PT15M")
    # fmt: off
    # [start:min_max_example]
    tf_min_max = tf.time_of_year_aggregate(
        "max", "flow"
    ).time_of_year_aggregate(
        "min", "flow"
    )
    # [end:min_max_example]

    with pl.Config(tbl_cols=-1, set_tbl_width_chars=-1):
        print(tf_min_max.df)
    # fmt: on


def _daily_temperature() -> pl.DataFrame:
    """A century of daily mean air temperature, 1925-2024, with a slow warming trend, seeded so it never changes."""
    rng = np.random.default_rng(1925)
    time = pl.date_range(datetime(1925, 1, 1), datetime(2024, 12, 31), "1d", eager=True)
    day_of_year = time.dt.ordinal_day().to_numpy()
    years = (time.dt.year() - 1925).to_numpy()

    seasonal = 10.0 - 6.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    warming = 0.015 * years
    weather = np.convolve(rng.normal(0, 1.6, len(time) + 4), np.ones(5) / np.sqrt(5), mode="valid")
    return pl.DataFrame({"time": time, "temperature": (seasonal + warming + weather).round(1)})


def reference_interval_example() -> None:
    """Daily temperature anomalies against the 1990-2010 mean for each day of the year."""
    tf = ts.TimeFrame(_daily_temperature(), "time", resolution="P1D")
    # [start:reference_interval_example]
    tf_climatology = tf.time_of_year_aggregate(
        "mean",
        "temperature",
        reference_interval=(datetime(1990, 1, 1), datetime(2010, 12, 31)),
    )

    anomaly = tf_climatology.df.select(
        "time",
        "temperature",
        "mean_temperature",
        (pl.col("temperature") - pl.col("mean_temperature")).alias("anomaly"),
    )
    print(anomaly)
    # [end:reference_interval_example]
