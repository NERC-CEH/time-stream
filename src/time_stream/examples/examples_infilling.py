
# Create sample data with gaps
from datetime import datetime, timedelta
import numpy as np
from time_stream import TimeSeries, Period
from time_stream.infill import InfillMethod
import polars as pl


n_rows = 50
missing_rate = 0.5

dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_rows)]

# Create simple data column with trend + noise
values = np.arange(n_rows) * 0.5 + np.random.normal(0, 2, n_rows)

# Create DataFrame first
df = pl.DataFrame({
    "timestamp": dates,
    "data": values,
})

# Remove random rows to create gaps
n_to_remove = int(n_rows * missing_rate)
indices_to_remove = np.random.choice(n_rows, n_to_remove, replace=False)

# Keep only rows not in the removal list
df = df.filter(pl.Series("keep", ~np.isin(np.arange(n_rows), indices_to_remove)))

old_df = df.clone()

methods = ["linear", "cubic", "quadratic", "pchip", "akima", "bspline"]

for m in methods:
    ts = TimeSeries(df, "timestamp", resolution=Period.of_days(1), periodicity=Period.of_days(1), pad=True)

    if m == "bspline":
        i = InfillMethod.get(m, order=1)
    else:
        i = InfillMethod.get(m)
    print(i)

    ts = ts.infill(i, "data", max_gap_size=2)

    old_df = ts.df.rename({"data": m}).join(old_df, on="timestamp", how="full").drop("timestamp_right")

with pl.Config(tbl_rows=-1):
        print(old_df)