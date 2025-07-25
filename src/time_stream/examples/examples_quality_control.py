from datetime import datetime, timedelta

import polars as pl

from time_stream import TimeSeries


from utils import suppress_output


def create_simple_time_series():
    # [start_block_1]
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    temperatures = [24, 22, 19, 26, 24, 26, 28, 50, 52, 29]
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
    return ts

def comparison_qc_1():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_2]
    ts.qc_check(
        "comparison",
        check_column="temperature",
        flag_column="temperature_flag",
        compare_to=50,
        operator=">="
    )

    print(ts)
    # [end_block_2]

def comparison_qc_2():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_3]
    ts.qc_check(
        "comparison",
        check_column="precipitation",
        flag_column="precipitation_flag",
        compare_to=-0,
        operator="<"
    )

    print(ts)
    # [end_block_3]

def comparison_qc_3():
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_4]
    ts.qc_check(
        "comparison",
        check_column="sensor_codes",
        flag_column="sensor_codes_flag",
        compare_to=[991, 992, 993, 994, 995],
        operator="is_in"
    )

    print(ts)
    # [end_block_4]