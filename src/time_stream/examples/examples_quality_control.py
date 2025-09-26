from datetime import date, datetime, time, timedelta

import polars as pl

from time_stream import TimeFrame
from time_stream.examples.utils import suppress_output


def create_simple_time_series() -> TimeFrame:
    # [start_block_1]
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    temperatures = [24, 22, -35, 26, 24, 26, 28, 50, 52, 29]
    precipitation = [-3, 0, 5, 10, 2, 0, 0, 3, 1, 0]
    sensor_codes = [992, 1, 1, 1, 1, 1, 1, 991, 995, 1]

    df = pl.DataFrame(
        {"timestamp": dates, "temperature": temperatures, "precipitation": precipitation, "sensor_codes": sensor_codes}
    )

    ts = TimeFrame(df=df, time_name="timestamp")
    # [end_block_1]
    with pl.Config(tbl_rows=-1):
        print(ts)

    return ts


def comparison_qc_1() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_2]

    ts = ts.qc_check("comparison", column="temperature", compare_to=50, operator=">=", into="qc_result")

    print(ts)
    # [end_block_2]


def comparison_qc_2() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_3]
    ts = ts.qc_check("comparison", column="precipitation", compare_to=0, operator="<", into="qc_result")

    print(ts)
    # [end_block_3]


def comparison_qc_3() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_4]
    ts = ts.qc_check(
        "comparison", column="sensor_codes", compare_to=[991, 992, 993, 994, 995], operator="is_in", into="qc_result"
    )

    print(ts)
    # [end_block_4]


def range_qc_1() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_5]
    ts = ts.qc_check(
        "range",
        column="temperature",
        min_value=-10,
        max_value=50,
        closed="none",  # Range is not inclusive of min and max value
        within=False,  # Flag values outside of this range
        into="qc_result",
    )

    print(ts)
    # [end_block_5]


def range_qc_2() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_6]
    ts = ts.qc_check(
        "range",
        column="precipitation",
        min_value=-3,
        max_value=1,
        closed="both",  # Range is inclusive of min and max value
        within=True,  # Flag values inside of this range
        into="qc_result",
    )

    print(ts)
    # [end_block_6]


def spike_qc_1() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_7]
    ts = ts.qc_check("spike", column="temperature", threshold=10.0, into="qc_result")

    print(ts)
    # [end_block_7]


def observation_interval_example() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_8]
    # Only check data from specific dates
    start_date = datetime(2023, 1, 1, 5)
    end_date = datetime(2023, 1, 1, 10)

    ts = ts.qc_check(
        "range",
        column="temperature",
        min_value=-10,
        max_value=50,
        closed="none",
        within=False,
        observation_interval=(start_date, end_date),
        into="qc_result",
    )

    print(ts)
    # [end_block_8]


def time_range_qc_1() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_9]
    ts = ts.qc_check("time_range", column="temperature", min_value=time(1, 0), max_value=time(3, 0), into="qc_result")

    print(ts)
    # [end_block_9]


def time_range_qc_2() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_10]
    ts = ts.qc_check(
        "time_range",
        column="temperature",
        min_value=datetime(2023, 1, 1, 3, 30),
        max_value=datetime(2023, 1, 1, 9, 30),
        into="qc_result",
    )

    print(ts)
    # [end_block_10]


def time_range_qc_3() -> None:
    with suppress_output():
        ts = create_simple_time_series()

    # [start_block_11]
    ts = ts.qc_check(
        "time_range", column="temperature", min_value=date(2023, 1, 1), max_value=date(2023, 1, 2), into="qc_result"
    )

    print(ts)
    # [end_block_11]
