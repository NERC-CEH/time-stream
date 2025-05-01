from datetime import datetime, timedelta

import polars as pl

from time_stream import Period, TimeSeries
from time_stream.aggregation import Min

dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(31)]
values = [
    10,
    12,
    15,
    14,
    13,
    17,
    19,
    21,
    18,
    17,
    5,
    9,
    0,
    1,
    5,
    11,
    12,
    10,
    21,
    16,
    10,
    11,
    8,
    6,
    14,
    17,
    12,
    10,
    10,
    8,
    5,
]

df = pl.DataFrame({"timestamp": dates, "temperature": values})

ts = TimeSeries(df=df, time_name="timestamp", resolution="PT30M", periodicity="PT30M")

ts.aggregate(Period.of_months(1), Min, "temperature", {"available": 32})
