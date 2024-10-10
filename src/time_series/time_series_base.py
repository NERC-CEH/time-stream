from abc import ABC, abstractmethod
from typing import Iterator, Optional

import polars as pl
from period import Period


class TimeSeries(ABC):
    def __init__(
        self,
        time_name: Optional[str] = None,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        timezone: Optional[str] = None,
    ) -> None:
        self.__time_name = time_name
        self.__resolution = resolution
        self.__periodicity = periodicity
        self.__timezone = timezone

    @property
    def time_name(self) -> str:
        return self.__time_name

    @property
    def resolution(self) -> Period:
        return self.__resolution

    @property
    def periodicity(self) -> Period:
        return self.__periodicity

    @property
    def timezone(self) -> str:
        return self.__timezone

    @staticmethod
    def from_polars(
        df: pl.DataFrame,
        time_name: Optional[str] = None,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        timezone: Optional[str] = None,
    ) -> "TimeSeries":
        # Lazy import to avoid recursive importing
        from time_series_polars import TimeSeriesPolars

        return TimeSeriesPolars(df, time_name, resolution, periodicity, timezone)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
