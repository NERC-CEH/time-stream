import random
from datetime import datetime

import polars as pl

from time_series import TimeSeries
from time_series.period import Period

# Create a DataFrame to simulate random datetime and value data over several years
# this example simulates AMAX style data, with each year having one value on a random day
years = range(2010, 2021)
timestamps = []
values = []

# Generating random datetime values within each year
for year in years:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    hour = random.choice(range(0, 23, 9))
    minute = random.choice(range(0, 60, 15))
    second = 0
    random_dt = datetime(year, month, day, hour, minute, second)
    timestamps.append(random_dt)
    values.append(random.uniform(0, 100))  # Random values between 0 and 100

# Convert to a Polars DataFrame
df = pl.DataFrame({"timestamp": timestamps, "value": values})

# ts = TimeSeries.from_polars(df, "timestamp")

# print(ts)


# Create a DataFrame to simulate random datetime and value data over several years
# this example simulates monthly data
years = range(2010, 2021)
timestamps = []
values = []

# Generating random datetime values within each year
for year in years:
    month = random.choice(range(1, 12))
    day = random.choice(range(1, 28))
    hour = random.choice(range(0, 23))
    minute = random.choice(range(0, 60, 15))
    second = 0
    dt = datetime(year, month, day, hour, minute, second)
    timestamps.append(dt)
    values.append(random.uniform(0, 100))  # Random values between 0 and 100

# timestamps = [
#     datetime(2000, 10, 1),
#     datetime(2001, 11, 1, 16),
#     datetime(2002, 12, 1, 4, 3),
#     datetime(2004, 1, 1, 7, 10),
# ]
# values= [1,2,3,4]

# Convert to a Polars DataFrame
df = pl.DataFrame({"timestamp": timestamps, "value": values})

resolution = Period.of_minutes(15)
periodicity = Period.of_years(1).with_month_offset(9).with_hour_offset(9)

ts = TimeSeries.from_polars(df, "timestamp", resolution, periodicity)