"""Shared fixtures for the user guide examples - not shown on any page."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

DATA = Path(__file__).parent / "data"

# The 15-minute examples start here, just before a run of short gaps that the infilling examples fill
SAMPLE_START = datetime(2024, 1, 10, 2, 30)


def flow_15min() -> pl.DataFrame:
    """An example 15-minute river flow dataset (m³/s), from 2015 to 2025, with gaps."""
    return pl.read_parquet(DATA / "example_flow_15min.parquet")


def sample_frame(*, library: str = "polars") -> pd.DataFrame | pl.DataFrame:
    """The example 15-minute river flow from 2024-01-10 02:30 to the end of 2025.

    Args:
        library: ``"polars"`` returns a Polars DataFrame; anything else a time-indexed pandas DataFrame.
    """
    df = flow_15min().filter(pl.col("time") >= SAMPLE_START)
    if library == "polars":
        return df
    # Built from the column arrays, as to_pandas() would need pyarrow
    return pd.DataFrame({"flow": df["flow"].to_numpy()}, index=pd.Index(df["time"].to_numpy(), name="time"))


def secondary_flow_15min() -> pl.DataFrame:
    """A second example 15-minute river flow dataset (m³/s), over the same period as the sample."""
    return pl.read_parquet(DATA / "example_flow_15min_secondary.parquet").filter(pl.col("time") >= SAMPLE_START)


def rainfall_15min() -> pl.DataFrame:
    """An example 15-minute rainfall dataset (mm), from 2015 to 2025, with gaps."""
    return pl.read_parquet(DATA / "example_rainfall_15min.parquet")
