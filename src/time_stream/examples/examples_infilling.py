from datetime import datetime

import numpy as np
import polars as pl

from time_stream import Period, TimeSeries

from utils import suppress_output


def create_simple_time_series_with_gaps():
    # [start_block_1]
    np.random.seed(42)

    # Set up a daily time series with varying gaps
    dates = [
        datetime(2024, 1, 1), datetime(2024, 1, 2), # One-day gap,
        datetime(2024, 1, 4), datetime(2024, 1, 5), datetime(2024, 1, 6),
        datetime(2024, 1, 7), # Two-day gap,
        datetime(2024, 1, 10), datetime(2024, 1, 11), datetime(2024, 1, 12),
        # Three-day gap,
        datetime(2024, 1, 16)
    ]

    # Create example random column data
    df = pl.DataFrame({
        "timestamp": dates,
        "temperature": np.arange(len(dates)) * 0.5 + np.random.normal(0, 2, len(dates)),
    })

    ts = TimeSeries(
        df=df,
        time_name="timestamp",
        resolution=Period.of_days(1),
        periodicity=Period.of_days(1),
        pad=True
    )
    # [end_block_1]
    with pl.Config(tbl_rows=-1):
        print(ts)

    return ts


def all_infills():
    with suppress_output():
        ts = create_simple_time_series_with_gaps()

    # [start_block_2]
    methods = ["linear", "cubic", "quadratic", "pchip", "akima"]
    result = ts.df.clone()
    for method in methods:
        ts_infilled = ts.infill(method, "temperature")
        ts_infilled.df = ts_infilled.df.rename({"temperature": method})
        result = result.join(ts_infilled.df, on="timestamp", how="full").drop("timestamp_right")
    # [end_block_2]

    with pl.Config(tbl_rows=-1):
        print(result)
    return ts

# 
# 
# 
# old_df = df.clone()
# 
# methods = ["linear", "cubic", "quadratic", "pchip", "akima", "bspline"]
# 
# for m in methods:
#     ts = TimeSeries(df, "timestamp", resolution=Period.of_days(1), periodicity=Period.of_days(1), pad=True)
# 
#     if m == "bspline":
#         i = InfillMethod.get(m, order=1)
#     else:
#         i = InfillMethod.get(m)
#     print(i)
# 
#     ts = ts.infill(i, "data", max_gap_size=2)
# 
#     old_df = ts.df.rename({"data": m}).join(old_df, on="timestamp", how="full").drop("timestamp_right")
# 
# with pl.Config(tbl_rows=-1):
#         print(old_df)