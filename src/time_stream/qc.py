from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import List, Optional, Tuple, Type, Union

import polars as pl

from time_stream import TimeSeries

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
        self, min_value: float, max_value: float, inclusive: Optional[bool] = True, within: Optional[bool] = False
    ) -> None:
        """Initialize range check.

        Args:
            min_value: Minimum acceptable value. Values below this will be flagged.
            max_value: Maximum acceptable value. Values above this will be flagged.
            inclusive: Whether the range bounds are inclusive.
            within: Whether values get flagged when within this range (within=True)
                    or not within this range (within=False, default).
        """
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
        self.within = within

    def expr(self, check_column: str) -> pl.Expr:
        """Return the Polars expression for range checking."""

        # Check if we're doing a time-based range check
        if isinstance(self.min_value, time) and isinstance(self.max_value, time):
            check_column = pl.col(check_column).dt.time()
        else:
            check_column = pl.col(check_column)

        # Set the operators based on "inclusive" option
        if self.inclusive:
            min_expr = check_column < self.min_value
            max_expr = check_column > self.max_value
        else:
            min_expr = check_column <= self.min_value
            max_expr = check_column >= self.max_value

        # Return expression based on "within" option
        base_expr = min_expr | max_expr
        if self.within:
            return ~base_expr  # Flag if INSIDE range
        else:
            return base_expr  # Flag if OUTSIDE range


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
