"""Shared fixtures for the user guide examples - not shown on any page."""

import numpy as np
import pandas as pd
import polars as pl


def sample_frame(*, library: str = "polars", complete: bool = False) -> pd.DataFrame | pl.DataFrame:
    """A 15-minute river-flow series from 2020-09-01 to 2023-11-01, seeded so it never changes.

    Args:
        library: ``"polars"`` returns a Polars DataFrame; anything else a time-indexed pandas DataFrame.
        complete: When False, a scattering of ``flow`` values (including a few fixed indices) is set to null.
    """
    np.random.seed(31)
    date_range = pd.date_range(start="2020-09-01", end="2023-11-01", freq="15min")
    flow = np.random.uniform(90, 100, len(date_range)) + np.sin(np.arange(len(date_range)) * 0.01) * 20

    df = pd.DataFrame({"time": date_range, "flow": flow})

    if not complete:
        df.loc[np.random.random(len(df)) > 0.95, "flow"] = np.nan
        df.iloc[[1, 3, 4, 5, 6, -2, -3], df.columns.get_loc("flow")] = np.nan

    if library == "polars":
        return pl.DataFrame(df)
    return df.set_index("time")
