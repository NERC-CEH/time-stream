"""Examples for the "Irregular time series" user guide page."""

from datetime import datetime

import polars as pl

import time_stream as ts
from time_stream.exceptions import AggregationError, MissingCriteriaError, PeriodicityError


def _gaugings() -> pl.DataFrame:
    """River flow gaugings, taken by hand on random days at random times."""
    return pl.DataFrame(
        {
            "time": [
                datetime(2024, 1, 9, 10, 15),
                datetime(2024, 1, 23, 14, 2),
                datetime(2024, 2, 20, 9, 41),
                datetime(2024, 3, 5, 11, 30),
                datetime(2024, 3, 19, 15, 47),
                datetime(2024, 5, 14, 10, 5),
                datetime(2024, 6, 11, 13, 20),
                datetime(2024, 8, 27, 9, 55),
            ],
            "flow": [3.42, 2.98, 4.11, 3.76, 3.05, 1.92, 1.47, 0.88],
        }
    )


def create_irregular() -> None:
    """An irregular TimeFrame of gaugings recorded to the minute."""
    df = _gaugings()
    # [start:create_irregular]
    tf = ts.TimeFrame(df, "time", resolution="PT1M", periodicity="irregular")

    print(tf)
    print("is_irregular:", tf.is_irregular)
    # [end:create_irregular]


def aggregate_irregular() -> None:
    """Monthly mean of the gaugings: the result is a regular TimeFrame."""
    tf = ts.TimeFrame(_gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:aggregate_irregular]
    tf_monthly = tf.aggregate("P1M", "mean", "flow")

    print(tf_monthly.df)
    print("resolution:", tf_monthly.resolution, "periodicity:", tf_monthly.periodicity)
    # [end:aggregate_irregular]


def aggregate_available_criteria() -> None:
    """Monthly mean, only valid where at least 2 gaugings were taken."""
    tf = ts.TimeFrame(_gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:aggregate_available_criteria]
    tf_monthly = tf.aggregate("P1M", "mean", "flow", missing_criteria=("available", 2))

    print(tf_monthly.df)
    # [end:aggregate_available_criteria]


def rolling_irregular() -> None:
    """Trailing 60-day mean at each gauging: the result stays irregular."""
    tf = ts.TimeFrame(_gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:rolling_irregular]
    tf_rolling = tf.rolling_aggregate("P60D", "mean", "flow")

    print(tf_rolling.df)
    print("is_irregular:", tf_rolling.is_irregular)
    # [end:rolling_irregular]


def unsupported_operations() -> None:
    """The operations that need an expected time step raise an error."""
    tf = ts.TimeFrame(_gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:unsupported_operations]
    try:
        tf.pad()
    except PeriodicityError as err:
        print("pad:", err)

    try:
        tf.infill("linear", "flow")
    except PeriodicityError as err:
        print("infill:", err)

    try:
        tf.aggregate("P1M", "mean_sum", "flow")
    except AggregationError as err:
        print("mean_sum:", err)

    try:
        tf.aggregate("P1M", "mean", "flow", missing_criteria=("percent", 50))
    except MissingCriteriaError as err:
        print("percent:", err)
    # [end:unsupported_operations]
