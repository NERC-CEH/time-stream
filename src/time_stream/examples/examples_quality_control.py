from datetime import datetime, timedelta, time

import polars as pl

from time_stream import TimeSeries


from utils import suppress_output


def create_simple_time_series():
    # [start_block_1]
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    temperatures = [24, 22, -35, 26, 24, 26, 28, 50, 52, 29]
    precipitation = [-3, 0, 5, 10, 2, 0, 0, 3, 1, 0]
    sensor_codes = [992, 1, 1, 1, 1, 1, 1, 991, 995, 1]

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": temperatures,
        "precipitation": precipitation,
        "sensor_codes": sensor_codes
    })

    ts = TimeSeries(
        df=df,
        time_name="timestamp"
    )
    # [end_block_1]
    with pl.Config(tbl_rows=-1):
        print(ts)

    return ts

def comparison_qc_1():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_2]
    ts.df = ts.df.with_columns(
        ts.qc_check(
            "comparison",
            check_column="temperature",
            compare_to=50,
            operator=">="
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_2]

def comparison_qc_2():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_3]
    ts.df = ts.df.with_columns(
        ts.qc_check(
            "comparison",
            check_column="precipitation",
            compare_to=0,
            operator="<"
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_3]

def comparison_qc_3():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_4]
    ts.df = ts.df.with_columns(
        ts.qc_check(
            "comparison",
            check_column="sensor_codes",
            compare_to=[991, 992, 993, 994, 995],
            operator="is_in"
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_4]

def range_qc_1():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_5]
    ts.df = ts.df.with_columns(
        ts.qc_check(
            "range",
            check_column="temperature",
            min_value=-10,
            max_value=50,
            closed="none",  # Range is not inclusive of min and max value
            within=False, # Flag values outside of this range
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_5]

def range_qc_2():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_6]
    ts.df = ts.df.with_columns(
        ts.qc_check(
            "range",
            check_column="precipitation",
            min_value=-3,
            max_value=1,
            closed="both",  # Range is inclusive of min and max value
            within=True, # Flag values inside of this range
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_6]

def spike_qc_1():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_7]
    ts.df = ts.df.with_columns(
            ts.qc_check(
            "spike",
            check_column="temperature",
            threshold=10.0
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_7]

def observation_interval_example():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_8]
    # Only check data from specific dates
    start_date = datetime(2023, 1, 5)
    end_date = datetime(2023, 1, 10)

    ts.df = ts.df.with_columns(
        ts.qc_check(
            "range",
            check_column="temperature",
            min_value=-10,
            max_value=50,
            closed="none",
            within=False,
            observation_interval=(start_date, end_date)
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_8]


def time_range_qc_1():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_9]
    ts.df = ts.df.with_columns(
        ts.qc_check(
            "time_range",
            check_column="temperature",
            min_value=time(1, 0),
            max_value=time(3, 0)
        ).alias("qc_result")
    )

    print(ts)
    # [end_block_9]
