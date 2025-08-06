from abc import ABC, abstractmethod
from datetime import date, datetime, time
from typing import List, Optional, Tuple, Type, Union

import polars as pl

from time_stream import TimeSeries
from time_stream.enums import ClosedInterval

# Registry for built-in QC checks
_QC_REGISTRY = {}


def register_qc_check(cls: Type["QCCheck"]) -> Type["QCCheck"]:
    """Decorator to register quality control check classes using their name attribute.

    Args:
        cls: The quality control class to register.

    Returns:
        The decorated class.
    """
    _QC_REGISTRY[cls.name] = cls
    return cls


class QCCheck(ABC):
    """Base class for quality control checks."""

    _ts = None

    @property
    def ts(self) -> TimeSeries:
        if self._ts is None:
            raise AttributeError("TimeSeries has not been initialised for this QC check.")
        return self._ts

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the QC check."""
        pass

    @abstractmethod
    def expr(self, check_column: str) -> pl.Expr:
        """Return the Polars expression for this QC check.

        Args:
            check_column: The column to apply the check to.

        Returns:
            Boolean expression where True indicates values that should be flagged.
        """
        pass

    @classmethod
    def get(cls, check: Union[str, Type["QCCheck"]], **kwargs) -> "QCCheck":
        """Factory method to get a QC check instance from string names or class type.

        Args:
            check: The QC check specification, which can be:
                - A string name: e.g. "threshold", "range", "spike"
                - A class type: ThresholdCheck, RangeCheck, etc.
                - An instance: ThresholdCheck(), or any QCCheck instance
            **kwargs: Parameters specific to the check type, used to initialise the class object.
                      Ignored if check is already an instance of the class.

        Returns:
            An instance of the appropriate QCCheck subclass.

        Raises:
            KeyError: If a string name is not registered as a QC check.
            TypeError: If the input type is not supported or a class doesn't inherit from QCCheck.

        Examples:
            >>> # From string
            >>> qc = QCCheck.get("threshold")
            >>>
            >>> # From class
            >>> qc = QCCheck.get(ThresholdCheck)
            >>>
            >>> # From instance
            >>> qc = QCCheck.get(ThresholdCheck(arg1, arg2))
        """
        # If it's already an instance, return it
        if isinstance(check, QCCheck):
            return check

        # If it's a string, look it up in the registry
        if isinstance(check, str):
            try:
                return _QC_REGISTRY[check](**kwargs)
            except KeyError:
                raise KeyError(f"Unknown QC check '{check}'.")

        # If it's a class, check the subclass type and return
        elif isinstance(check, type):
            if issubclass(check, QCCheck):
                return check(**kwargs)  # type: ignore[misc]
            else:
                raise TypeError(f"QC check class {check.__name__} must inherit from QCCheck")

        else:
            raise TypeError(f"QC check must be a string or a QCCheck class. Got {type(check).__name__}")

    def apply(
        self,
        ts: TimeSeries,
        check_column: str,
        observation_interval: Optional[datetime | Tuple[datetime, datetime | None]] = None,
    ) -> pl.Series:
        """Apply the QC check to the TimeSeries.

        Args:
            ts: The TimeSeries to check.
            check_column: The column to perform the check on.
            observation_interval: Optional time interval to limit the check to.

        Returns:
            pl.Series: Boolean series of the resolved expression on the TimeSeries.
        """
        # Validate column exists
        if check_column not in ts.columns and check_column != ts.time_name:
            raise KeyError(f"Check column '{check_column}' not found in TimeSeries.")

        # Set the timeseries property, in case class expr method needs access to properties in the object
        self._ts = ts

        # Get the check expression
        check_expr = self.expr(check_column)

        # Apply observation interval filter if specified
        if observation_interval:
            date_filter = self._get_date_filter(ts, observation_interval)
            check_expr = check_expr & date_filter

        # Evaluate and return the result of the QC check
        #   Naming the series to an empty string so as not to cause confusion.
        #   Up to user if they want to name it and add on to the TimeSeries dataframe.
        result = ts.df.select(check_expr).to_series().alias("")

        return result

    @staticmethod
    def _get_date_filter(ts: TimeSeries, observation_interval: datetime | Tuple[datetime, datetime | None]) -> pl.Expr:
        """Get Polars expression for observation date interval filtering.

        Args:
            ts: The TimeSeries to create the filter for.
            observation_interval: Tuple of (start_date, end_date) defining the time period.

        Returns:
            pl.Expr: Boolean polars expression for date filtering.
        """
        if isinstance(observation_interval, datetime):
            start_date = observation_interval
            end_date = None
        else:
            start_date, end_date = observation_interval

        if end_date is None:
            return pl.col(ts.time_name) >= start_date
        else:
            return pl.col(ts.time_name).is_between(start_date, end_date)


@register_qc_check
class ComparisonCheck(QCCheck):
    """Compares values against a given value using a comparison operator."""

    name = "comparison"

    def __init__(self, compare_to: float | List, operator: str, flag_na: Optional[bool] = False) -> None:
        """Initialize comparison check.

        Args:
            compare_to: The value for comparison.
            operator: Comparison operator. One of: '>', '>=', '<', '<=', '==', '!=', 'is_in'.
            flag_na: If True, also flag NaN/null values as failing the check. Defaults to False.
        """
        self.compare_to = compare_to
        self.operator = operator
        self.flag_na = flag_na

    def expr(self, check_column: str) -> pl.Expr:
        """Return the Polars expression for threshold checking."""
        operator_map = {
            ">": pl.col(check_column) > self.compare_to,
            ">=": pl.col(check_column) >= self.compare_to,
            "<": pl.col(check_column) < self.compare_to,
            "<=": pl.col(check_column) <= self.compare_to,
            "==": pl.col(check_column) == self.compare_to,
            "!=": pl.col(check_column) != self.compare_to,
            "is_in": pl.col(check_column).is_in(self.compare_to),
        }

        if self.operator not in operator_map:
            raise KeyError(f"Invalid operator '{self.operator}'. Use: {', '.join(operator_map.keys())}")

        operator_expr = operator_map[self.operator]
        if self.flag_na:
            operator_expr = operator_expr | pl.col(check_column).is_null()

        return operator_expr


@register_qc_check
class RangeCheck(QCCheck):
    """Check that values fall within an acceptable range."""

    name = "range"

    def __init__(
        self,
        min_value: float | time | date | datetime,
        max_value: float | time | date | datetime,
        closed: Optional[str | ClosedInterval] = "both",
        within: Optional[bool] = True,
    ) -> None:
        """Initialize range check.

        Args:
            min_value: Minimum of the range.
            max_value: Maximum of the range.
            closed: Define which sides of the interval are closed (inclusive) {'both', 'left', 'right', 'none'}
                    (default = "both")
            within: Whether values get flagged when within or outside the range (default = True (within)).
        """
        self.min_value = min_value
        self.max_value = max_value
        self.closed = ClosedInterval(closed)
        self.within = within

    def expr(self, check_column: str) -> pl.Expr:
        """Return the Polars expression for range checking."""
        if type(self.min_value) is not type(self.max_value):
            raise TypeError("'min_value' and 'max_value' must be of same type")

        check_type = type(self.min_value)

        # Check if we're doing a time-based range check
        if check_type is time:
            check_column = pl.col(check_column).dt.time()

            # Consider ranges that cross midnight, e.g. min_value = 11:00, max_value = 01:00
            if self.min_value > self.max_value:
                # Swap the values so the comparison operators work the correct way around
                self.min_value, self.max_value = self.max_value, self.min_value

                # Reverse the within parameter, as we've swapped the min/max logic
                self.within = not self.within

                # We also need to swap the close parameter (if "both" or "none")
                # Don't have to change "left" or "right" as it shakes out the same even when reversing the min/max
                if self.closed == ClosedInterval.BOTH:
                    self.closed = ClosedInterval.NONE
                elif self.closed == ClosedInterval.NONE:
                    self.closed = ClosedInterval.BOTH

        elif check_type is date:
            # For datetime.date objects (NOT datetime.datetime!), we want to consider the whole date part of the column
            check_column = pl.col(check_column).dt.date()

        else:
            # This should handle numeric objects and datetime.datetime objects
            check_column = pl.col(check_column)

        in_range = check_column.is_between(
            self.min_value,
            self.max_value,
            closed=self.closed.value,  # type: ignore[arg-type] ignore Literal typing as the enum constrains the values
        )
        return in_range if self.within else ~in_range


@register_qc_check
class TimeRangeCheck(RangeCheck):
    """Flag rows where the primary time column of the time series fall within an acceptable range.

    This can either be used with min / max values of:
        - datetime.time : Useful for scenarios where there are consistent errors at a certain time of day,
                          e.g., during an automated sensor calibration time.
        - datetime.date : Useful for scenarios where a specific date range is known to be bad,
                              e.g., during a time of sensor errors not picked up elsewhere.
        - datetime.datetime : As above, but where there you need to add a time to the date range as well.

    Note: This is equivalent to using `RangeCheck` with `check_column = ts.time_name`. However, adding this as a
          convenience method as it may not be obvious that the `RangeCheck` can be used for this purpose.
    """

    name = "time_range"

    def expr(self, _: str) -> pl.Expr:
        return super().expr(self.ts.time_name)


@register_qc_check
class SpikeCheck(QCCheck):
    """Detect spikes by assessing differences with neighboring values."""

    name = "spike"

    def __init__(self, threshold: float):
        """Initialize spike detection check.

        Args:
            threshold: The spike detection threshold.
        """
        self.threshold = threshold

    def expr(self, check_column: str) -> pl.Expr:
        """Return the Polars expression for spike detection.

        The algorithm:
        1. Calculate differences between current value and neighbors
        2. Compute total combined difference and skew
        3. Flag where (total_difference - skew) > threshold * 2
        """
        # Calculate differences with temporal neighbors
        prev_val = pl.col(check_column).shift(1)
        next_val = pl.col(check_column).shift(-1)

        diff_prev = pl.col(check_column) - prev_val
        diff_next = next_val - pl.col(check_column)

        # Calculate total difference and skew
        d = (diff_prev - diff_next).abs()
        skew = (diff_prev.abs() - diff_next.abs()).abs()
        d_no_skew = d - skew

        # Double the threshold since we're summing differences
        return d_no_skew > (self.threshold * 2.0)
