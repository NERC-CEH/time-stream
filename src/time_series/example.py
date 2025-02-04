import random
from datetime import datetime

import polars as pl

from time_series import Period, TimeSeries

# Create a DataFrame to simulate random datetime and value data over several years
# this example simulates AMAX style data, with each year having one value on a random day
years = range(1900, 2000)
timestamps = []
values = []

# Generating random datetime values within each year
for year in years:
    day = random.randint(1, 28)
    month = random.randint(1, 9)
    hour = random.choice(range(0, 23, 9))
    minute = random.choice(range(0, 60, 15))
    second = 0
    random_dt = datetime(year, month, day, hour, minute, second)
    timestamps.append(random_dt)
    values.append(random.uniform(0, 100))  # Random values between 0 and 100

resolution = Period.of_minutes(15)
periodicity = Period.of_years(1).with_month_offset(9).with_hour_offset(9)

# Convert to a Polars DataFrame
df = pl.DataFrame(
    {"timestamp": timestamps, "pressure": values, "temperature": values, "supp1": values, "flagcol": values}
)

metadata = {
    "pressure": {"units": "hpa", "description": "Hello world!"},
    "temperature": {"description": "Hello temperature!"},
}

flag_systems = {"quality_flags": {"OK": 1, "WARNING": 2}}

ts = TimeSeries(
    df,
    "timestamp",
    resolution,
    periodicity,
    column_metadata=metadata,
    supplementary_columns=["supp1"],
    flag_columns={"flagcol": "quality_flags"},
    flag_systems=flag_systems,
)

ts.df = ts.df.with_columns((pl.col("pressure") * 2).alias("pressure"))


ts.supp1.add_relationship("pressure")
print(ts._relationship_manager._relationships)

ts.supp1.remove()
print(ts.columns)
print(ts._relationship_manager._relationships)


ts.temperature.add_relationship("flagcol")
print(ts._relationship_manager._relationships)

# ts.flagcol.add_relationship("temperature")
# print(ts._relationship_manager._relationships)

ts.flagcol.remove()
print(ts.columns)
print(ts._relationship_manager._relationships)
pass
