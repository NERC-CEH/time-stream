from typing import Iterator, Optional

import polars as pl
from period import Period
from time_series_base import TimeSeries


class TimeSeriesPolars(TimeSeries):
    def __init__(
        self,
        df: pl.DataFrame,
        time_name: Optional[str] = None,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        timezone: Optional[str] = None,
    ) -> None:
        self.__df = df
        super().__init__(time_name, resolution, periodicity, timezone)

    @property
    def df(self) -> pl.DataFrame:
        return self.__df

    def __len__(self):
        return self.__df.height

    def __iter__(self) -> Iterator:
        return self.__df.iter_rows()

    def __str__(self) -> str:
        return self.__df.__str__()
