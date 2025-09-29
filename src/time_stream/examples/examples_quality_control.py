from datetime import datetime, time, timedelta

import polars as pl

import time_stream as ts
from time_stream.examples.utils import suppress_output


def create_simple_time_series() -> ts.TimeFrame:
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    temperatures = [24, 22, -35, 26, 24, 26, 28, 50, 52, 29]
    precipitation = [-3, 0, 5, 10, 2, 0, 0, 3, 1, 0]
    sensor_codes = [992, 1, 1, 1, 1, 1, 1, 991, 995, 1]

    df = pl.DataFrame(
        {"timestamp": dates, "temperature": temperatures, "precipitation": precipitation, "sensor_codes": sensor_codes}
    )

    tf = ts.TimeFrame(df=df, time_name="timestamp")

    return tf


def comparison_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_2]
    tf = tf.qc_check(
        "comparison", "temperature", compare_to=50, operator=">=", into=True
    )
    # [end_block_2]
    print(tf)
    # fmt: on


def comparison_qc_3() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_4]
    error_codes = [991, 992, 993, 994, 995]
    tf = tf.qc_check(
        "comparison", "sensor_codes", compare_to=error_codes, operator="is_in", into=True
    )
    # [end_block_4]
    print(tf)
    # fmt: on


def range_qc_1() -> None:
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_5]
    tf = tf.qc_check(
        "range",
        "temperature",
        min_value=-10,
        max_value=50,
        closed="none",  # Range is not inclusive of min and max value
        within=False,  # Flag values outside of this range
        into=True,
    )
    # [end_block_5]
    print(tf)


def spike_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_7]
    tf = tf.qc_check(
        "spike", "temperature", threshold=10.0, into=True
    )
    # [end_block_7]
    print(tf)
    # fmt: on


def time_range_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_9]
    tf = tf.qc_check(
        "time_range",
        "precipitation",
        min_value=time(1, 0),
        max_value=time(3, 0),
        into=True
    )
    # [end_block_9]
    print(tf)
    # fmt: on


def time_range_qc_2() -> None:
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_10]
    tf = tf.qc_check(
        "time_range",
        "temperature",
        min_value=datetime(2023, 1, 1, 3, 30),
        max_value=datetime(2023, 1, 1, 9, 30),
        into=True,
    )
    # [end_block_10]
    print(tf)
