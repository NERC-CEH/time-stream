# [start_block_1]
from datetime import datetime, timedelta

import polars as pl

from time_stream import TimeSeries, Period
# [end_block_1]
from utils import suppress_output


def create_df():
    # [start_block_2]
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(31)]
    values = [10, 12, 15, 14, 13, 17, 19, 21, 18, 17,
              5, 9, 0, 1, 5, 11, 12, 10, 21, 16,
              10, 11, 8, 6, 14, 17, 12, 10, 10, 8, 5]

    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": values
    })
    # [end_block_2]
    return df

def create_timeseries():
    with suppress_output():
        df = create_df()

    # [start_block_3]
    # Specify the resolution and periodicity of the data
    resolution = Period.of_days(1)
    periodicity = Period.of_days(1)

    # Build the time series object
    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=resolution,
        periodicity=periodicity
    )
    # [end_block_3]
    print(ts)
    return ts


def aggregate_data():
    with suppress_output():
        ts = create_timeseries()

    # [start_block_4]
    from time_stream.aggregation import Mean

    # Aggregate temperature data by month
    monthly_period = Period.of_months(1)
    monthly_temp = ts.aggregate(monthly_period, Mean, "temperature")
    # [end_block_4]

    print(monthly_temp)


def create_flagging_system():
    with suppress_output():
        df = create_df()

    # [start_block_5]
    quality_flags = {"MISSING": 1, "SUSPICIOUS": 2, "ESTIMATED": 4}

    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        flag_systems={"quality": quality_flags}
    )

    print(ts.flag_systems)
    # [end_block_5]
    return ts


def use_flagging_system():
    with suppress_output():
        ts = create_flagging_system()

    # [start_block_6]
    # Create a flag column
    ts.init_flag_column("quality", "temperature_qc_flags")

    # Flag suspicious values
    ts.add_flag("temperature_qc_flags", "SUSPICIOUS", pl.col("temperature") > 15)
    # [end_block_6]

    with pl.Config(tbl_rows=31):
        print(ts)
    return ts
