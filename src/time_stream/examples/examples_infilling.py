from datetime import datetime

import matplotlib.pyplot as plt
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
    methods = ["linear", "quadratic", "cubic", "pchip", "akima"]
    result = ts.df.clone()
    for method in methods:
        ts_infilled = ts.infill(method, "temperature")
        ts_infilled.df = ts_infilled.df.rename({"temperature": method})
        result = result.join(ts_infilled.df, on="timestamp", how="full").drop("timestamp_right")
    # [end_block_2]

    with pl.Config(tbl_rows=-1):
        print(result)
    return result


def plot_all_infills():
    with suppress_output():
        df = all_infills()

    plt.figure(figsize=(10, 6))

    datetime_col = 'timestamp'
    x_values = df[datetime_col].to_list()
    for col in [col for col in df.columns if col != datetime_col]:
        y_values = df[col].to_list()
        if col == "temperature":
            plt.scatter(x_values, y_values, label=col, s=70, c="black", zorder=10)
        else:
            plt.plot(x_values, y_values, label=col, linewidth=2)
            plt.scatter(x_values, y_values, s=30)

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Different infilling methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()


def infill_with_max_gap():
    with suppress_output():
        ts = create_simple_time_series_with_gaps()

    # [start_block_3]
    # The gap of 3 missing dates will not be infilled
    ts_infilled = ts.infill("linear", "temperature", max_gap_size=2)
    # [end_block_3]

    with pl.Config(tbl_rows=-1):
        print(ts_infilled)


def observation_interval_example():
    with suppress_output():
        ts = create_simple_time_series_with_gaps()

    # [start_block_4]
    # Only infill data between specific dates
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 5)

    ts_infilled = ts.infill("linear", "temperature", observation_interval=(start_date, end_date))
    # [end_block_4]

    with pl.Config(tbl_rows=-1):
        print(ts_infilled)

all_infills()