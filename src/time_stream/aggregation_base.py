"""
Time-series aggregation base classes and functions.

This module exists mainly to get round problems with cyclical module
initialisation in the "time_series.base" and "time_series.aggregation"
modules, which manifests itself with errors such as:

   ImportError: cannot import name 'blah1' from partially initialized module 'blah2'

This module defines the interface that is used by the TimeSeries class for
aggregation, which consists of the "apply_aggregation" function and the
"AggregationFunction" abstract base class.  The implementation of the
various aggregation functions is done in the "time_series.aggregation"
module, which imports "AggregationFunction" from this module.

It is intended (hoped?) that this module does not need any unit tests
itself. Testing of functions and classes in the "time_series.base" and
"time_series.aggregation" modules should indirectly test everything in
this module.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Type, Union

if TYPE_CHECKING:
    # Import is for type hinting only.  Make sure there is no runtime import, to avoid recursion.
    from time_stream import Period, TimeSeries


class AggregationFunction(ABC):
    """An aggregation function that can be applied to a field
    in a TimeSeries.

    A new aggregated TimeSeries can be created from an existing
    TimeSeries by passing a subclass of AggregationFunction
    into the TimeSeries.aggregate method.

    Attributes:
        name: The name of the aggregation function
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Return the name of this aggregation function"""
        return self._name

    @abstractmethod
    def apply(
        self,
        ts: "TimeSeries",
        aggregation_period: "Period",
        column_name: str,
        missing_criteria: Optional[Dict[str, Union[str, int]]] = None,
    ) -> "TimeSeries":
        """Apply this aggregation function to the supplied
        TimeSeries column and return a new TimeSeries containing
        the aggregated data

        Note: This is the first attempt at a mechanism for aggregating
        time-series data.  The signature of this method is likely to
        evolve considerably.

        Args:
            ts: The TimeSeries containing the data to be aggregated
            aggregation_period: The time period over which to aggregate
            column_name: The column containing the data to be aggregated
            missing_criteria: What level of missing data is acceptable. Ignores missing values by default

        Returns:
            A TimeSeries containing the aggregated data
        """
        raise NotImplementedError

    @classmethod
    def create(cls) -> "AggregationFunction":
        raise NotImplementedError


# The TimeSeries.aggregate method calls this function with self and all it's arguments
def apply_aggregation(
    ts: "TimeSeries",
    aggregation_period: "Period",
    aggregation_function: Type[AggregationFunction],
    column_name: str,
    missing_criteria: Union[None, Dict[str, Union[str, int]]],
) -> "TimeSeries":
    """Apply an aggregation function to a column in this TimeSeries, check the aggregation satisfies user requirements
    and return a new derived TimeSeries containing the aggregated data.

    The AggregationFunction class provides static methods that return aggregation function objects that can be used
    with this function.

    NOTE: This is the first attempt at a mechanism for aggregating time-series data.  The signature of this method
        is likely to evolve considerably.

    Args:
        aggregation_period: The period over which to aggregate the data
        aggregation_function: The aggregation function to apply
        column_name: The column containing the data to be aggregated
        missing_criteria: What level of missing data is acceptable. Ignores missing values by default.

    Returns:
        A TimeSeries containing the aggregated data.
    """
    if not ts.periodicity.is_subperiod_of(aggregation_period):
        raise UserWarning(
            f"Data periodicity {ts.periodicity} is not a subperiod of aggregation period {aggregation_period}"
        )
    return aggregation_function.create().apply(ts, aggregation_period, column_name, missing_criteria)
