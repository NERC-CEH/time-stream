"""Examples for the "Irregular time series" user guide page."""

import time_stream as ts
from examples.helpers import gaugings
from time_stream.exceptions import AggregationError, MissingCriteriaError, PeriodicityError


def create_irregular() -> None:
    """An irregular TimeFrame of gaugings recorded to the minute."""
    df = gaugings()
    # [start:create_irregular]
    tf = ts.TimeFrame(df, "time", resolution="PT1M", periodicity="irregular")

    print(tf)
    print("is_irregular:", tf.is_irregular)
    # [end:create_irregular]


def aggregate_irregular() -> None:
    """Monthly mean of the gaugings: the result is a regular TimeFrame."""
    tf = ts.TimeFrame(gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:aggregate_irregular]
    tf_monthly = tf.aggregate("P1M", "mean", "flow")

    print(tf_monthly.df)
    print("resolution:", tf_monthly.resolution, "periodicity:", tf_monthly.periodicity)
    # [end:aggregate_irregular]


def aggregate_available_criteria() -> None:
    """Monthly mean, only valid where at least 2 gaugings were taken."""
    tf = ts.TimeFrame(gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:aggregate_available_criteria]
    tf_monthly = tf.aggregate("P1M", "mean", "flow", missing_criteria=("available", 2))

    print(tf_monthly.df)
    # [end:aggregate_available_criteria]


def rolling_irregular() -> None:
    """Trailing 60-day mean at each gauging: the result stays irregular."""
    tf = ts.TimeFrame(gaugings(), "time", resolution="PT1M", periodicity="irregular")
    # [start:rolling_irregular]
    tf_rolling = tf.rolling_aggregate("P60D", "mean", "flow")

    print(tf_rolling.df)
    print("is_irregular:", tf_rolling.is_irregular)
    # [end:rolling_irregular]


def unsupported_operations() -> None:
    """The operations that need an expected time step raise an error."""
    tf = ts.TimeFrame(gaugings(), "time", resolution="PT1M", periodicity="irregular")
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
