import contextlib
import os
import sys
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl


@contextlib.contextmanager
def suppress_output() -> Iterator:
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def get_example_df(library: str = "polars") -> pd.DataFrame:
    # Create sample data: 15-minute intervals from 2020-09-01 to 2023-11-01 with random flow data
    np.random.seed(31)
    date_range = pd.date_range(start="2020-09-01", end="2023-11-01", freq="15min")
    flow_data = np.random.uniform(10, 100, len(date_range)) + np.sin(np.arange(len(date_range)) * 0.01) * 20
    flow_data[[1, 3, 4, 5, 6, -2, -3]] = np.nan

    # Create input dataframe
    df = pd.DataFrame({"time": date_range, "flow": flow_data})

    # Add some NaN values to simulate incomplete data
    mask = np.random.random(len(df)) > 0.95
    df.loc[mask, "flow"] = np.nan
    if library == "polars":
        df = pl.DataFrame(df)
    else:
        df = df.set_index("time")

    with pl.Config(tbl_rows=16):
        print(pl.DataFrame(df))

    return df
