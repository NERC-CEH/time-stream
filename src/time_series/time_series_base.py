from abc import ABC, abstractmethod
from typing import Iterator, Optional

import polars as pl

from time_series.period import Period


class TimeSeries(ABC):
    def __init__(
        self,
        time_name: str,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        time_zone: Optional[str] = None,
    ) -> None:
        self.__time_name = time_name
        self.__resolution = resolution
        self.__periodicity = periodicity
        self.__time_zone = time_zone

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
    def time_zone(self) -> str:
        return self.__time_zone

    @staticmethod
    def from_polars(
        df: pl.DataFrame,
        time_name: str,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        time_zone: Optional[str] = None,
    ) -> "TimeSeries":
        # Lazy import to avoid recursive importing
        from time_series_polars import TimeSeriesPolars

        return TimeSeriesPolars(df, time_name, resolution, periodicity, time_zone)

    @abstractmethod
    def _validate_resolution(self)  -> None:
        pass

    @abstractmethod
    def _validate_periodicity(self) -> None:
        pass

    @abstractmethod
    def _set_time_zone(self) -> None:
        pass

    @abstractmethod
    def _sort_time(self) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
