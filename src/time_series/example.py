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
df = pl.DataFrame({"timestamp": timestamps, "pressure": values, "temperature": values})

metadata = {
    "pressure": {"units": "hpa", "description": "Hello world!"},
    "temperature": {"description": "Hello temperature!"},
}

ts = TimeSeries(df, "timestamp", resolution, periodicity, column_metadata=metadata)

ts.df = ts.df.with_columns((pl.col("pressure") * 2).alias("pressure"))
print(ts)


print(ts.pressure, type(ts.pressure))
print(ts.pressure.as_timeseries(), type(ts.pressure.as_timeseries()))
print(ts.pressure.units)
print(ts["pressure"], type(ts["pressure"]))
print(ts[["pressure", "temperature"]])
print(ts.pressure.metadata("units"))
print(ts.shape)

# Adding flag columns
flag_dict = {
    "MISSING": 1,
    "ESTIMATED": 2,
    "CORRECTED": 4,
}
# First register the flag type
ts.add_flag_system("CORE", flag_dict)
# Add the flag column under the new flag type
ts.init_flag_column("CORE", "pressure_flags")
# Add a flag to the column. Can be done with flag name...
ts.add_flag("pressure_flags", "MISSING")
# or with flag value
ts.add_flag("pressure_flags", 2)

print(ts)

ts.remove_flag("pressure_flags", "ESTIMATED")
ts.pressure_flags.add_flag(4)

print(ts)

print(type(ts.pressure_flags))
ts.pressure_flags.unset()
print(type(ts.pressure_flags))
#ts.pressure_flags.add_flag(1)

ts.pressure_flags.remove()

print(ts)