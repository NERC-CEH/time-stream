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
        from time_series.time_series_polars import TimeSeriesPolars

        return TimeSeriesPolars(df, time_name, resolution, periodicity, time_zone)

    @abstractmethod
    def _validate_resolution(self) -> None:
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

    def aggregate(
        self, aggregation_period: Period, aggregation_function: "AggregationFunction", column_name: str
    ) -> "TimeSeries":
        """Apply an aggregation function to a column in this
        TimeSeries and return a new derived TimeSeries containing
        the aggregated data.

        The AggregationFunction class provides static methods that
        return aggregation function objects that can be used with
        this method.

        Note: This is the first attempt at a mechanism for aggregating
        time-series data.  The signature of this method is likely to
        evolve considerably.

        Args:
            aggregation_period: The period over which to aggregate
                                the data
            aggregation_function: The aggregation function to apply
            column_name: The column containing the data to be aggregated

        Returns:
            A TimeSeries containing the aggregated data.
        """
        return aggregation_function.apply(self, aggregation_period, column_name)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class AggregationFunction(ABC):
    """An aggregation function that can be applied to a field
    in a TimeSeries.

    A new aggregated TimeSeries can be created from an existing
    TimeSeries by passing a subclass of AggregationFunction
    into the TimeSeries.aggregate method.

    Attributes:
        name: The name of the aggregation function
    """

    @staticmethod
    def mean() -> "AggregationFunction":
        """Return an AggregationFunction that calculates
        an arithmetic mean.

        Returns:
            An AggregationFunction
        """
        # Lazy import to avoid recursive importing
        import time_series.aggregation

        return time_series.aggregation.Mean()

    @staticmethod
    def min() -> "AggregationFunction":
        """Return an AggregationFunction that returns the
        minimum value within each aggregation period, along
        with the datetime of the minimum

        Returns:
            An AggregationFunction
        """
        # Lazy import to avoid recursive importing
        import time_series.aggregation

        return time_series.aggregation.Min()

    @staticmethod
    def max() -> "AggregationFunction":
        """Return an AggregationFunction that returns the
        maximum value within each aggregation period, along
        with the datetime of the maximum

        Returns:
            An AggregationFunction
        """
        # Lazy import to avoid recursive importing
        import time_series.aggregation

        return time_series.aggregation.Max()

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this aggregation function"""
        return self._name

    @abstractmethod
    def apply(self, ts: TimeSeries, aggregation_period: Period, column_name: str) -> TimeSeries:
        """Apply this aggregation function to the supplied
        TimeSeries column and return a new TimeSeries containing
        the aggregated data

        Note: This is the first attempt at a mechanism for aggregating
        time-series data.  The signature of this method is likely to
        evolve considerably.

        Args:
            ts: The TimeSeries containing the data to be aggregated
            column_name: The column containing the data to be aggregated

        Returns:
            A TimeSeries containing the aggregated data
        """
        raise NotImplementedError()
