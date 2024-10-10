import random
from datetime import datetime

import polars as pl

from time_series import TimeSeries
from time_series.period import Period

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
df = pl.DataFrame({"timestamp": timestamps, "value": values})

ts = TimeSeries.from_polars(df, "timestamp", resolution, periodicity)

print(ts)
