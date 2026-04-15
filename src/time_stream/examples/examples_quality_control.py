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
    tf.register_flag_system("qc", {"FLAGGED": 1})

    return tf


def comparison_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_2]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "comparison", "temperature", compare_to=50, operator=">=", flag_params=("flag_temperature", "FLAGGED")
    )
    # [end_block_2]
    print(tf.df)
    # fmt: on


def comparison_qc_3() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_4]
    error_codes = [991, 992, 993, 994, 995]
    tf.init_flag_column("qc", "flag_sensor_codes")
    tf = tf.qc_check(
        "comparison", "sensor_codes", compare_to=error_codes, operator="is_in",
        flag_params=("flag_sensor_codes", "FLAGGED")
    )
    # [end_block_4]
    print(tf.df)
    # fmt: on


def range_qc_1() -> None:
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_5]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "range",
        "temperature",
        min_value=-30,
        max_value=50,
        closed="none",  # Range is not inclusive of min and max value
        within=False,  # Flag values outside of this range
        flag_params=("flag_temperature", "FLAGGED"),
    )
    # [end_block_5]
    print(tf.df)


def spike_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_7]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "spike", "temperature", threshold=10.0, flag_params=("flag_temperature", "FLAGGED")
    )
    # [end_block_7]
    print(tf.df)
    # fmt: on


def time_range_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_9]
    tf.init_flag_column("qc", "flag_precipitation")
    tf = tf.qc_check(
        "time_range",
        "precipitation",
        min_value=time(1, 0),
        max_value=time(3, 0),
        flag_params=("flag_precipitation", "FLAGGED")
    )
    # [end_block_9]
    print(tf.df)
    # fmt: on


def time_range_qc_2() -> None:
    with suppress_output():
        tf = create_simple_time_series()

    # [start_block_10]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "time_range",
        "temperature",
        min_value=datetime(2023, 1, 1, 3, 30),
        max_value=datetime(2023, 1, 1, 9, 30),
        flag_params=("flag_temperature", "FLAGGED"),
    )
    # [end_block_10]
    print(tf.df)


def create_flat_line_time_series() -> ts.TimeFrame:
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    temperature = [18.0, 20.0, 20.0, 20.0, 20.0, 22.0, 21.0, 0.0, 0.0, 0.0]
    df = pl.DataFrame({"timestamp": dates, "temperature": temperature})
    tf = ts.TimeFrame(df=df, time_name="timestamp")
    tf.register_flag_system("qc", {"FLAGGED": 1})
    return tf


def create_near_flat_line_time_series() -> ts.TimeFrame:
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    # Values drift slightly around 20, then jump to 22 and drift slightly around 21
    temperature = [18.0, 20.0, 20.005, 20.001, 19.991, 22.0, 20.99, 21.003, 21.009, 20.997]
    df = pl.DataFrame({"timestamp": dates, "temperature": temperature})
    tf = ts.TimeFrame(df=df, time_name="timestamp")
    tf.register_flag_system("qc", {"FLAGGED": 1})
    return tf


def flat_line_qc_1() -> None:
    # fmt: off
    with suppress_output():
        tf = create_flat_line_time_series()

    # [start_block_11]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "flat_line", "temperature", min_count=3, flag_params=("flag_temperature", "FLAGGED")
    )
    # [end_block_11]
    print(tf.df)
    # fmt: on


def flat_line_qc_2() -> None:
    # fmt: off
    with suppress_output():
        tf = create_flat_line_time_series()

    # [start_block_12]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "flat_line", "temperature", min_count=3, ignore_value=0.0, flag_params=("flag_temperature", "FLAGGED")
    )
    # [end_block_12]
    print(tf.df)
    # fmt: on


def flat_line_qc_3() -> None:
    # fmt: off
    with suppress_output():
        tf = create_near_flat_line_time_series()

    # [start_block_13]
    tf.init_flag_column("qc", "flag_temperature")
    tf = tf.qc_check(
        "flat_line", "temperature", min_count=3, tolerance=0.1, flag_params=("flag_temperature", "FLAGGED")
    )
    # [end_block_13]
    print(tf.df)
    # fmt: on
