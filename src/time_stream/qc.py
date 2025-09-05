from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import ClassVar, Type

import polars as pl

from time_stream.enums import ClosedInterval
from time_stream.exceptions import (
    DuplicateRegistryKeyError,
    QcError,
    QcTypeError,
    QcUnknownOperatorError,
    UnknownQcError,
)
from time_stream.utils import check_columns_in_dataframe, get_date_filter

# Registry for built-in QC checks
_QC_REGISTRY = {}


def register_qc_check(cls: Type["QCCheck"]) -> Type["QCCheck"]:
    """Decorator to register quality control check classes using their name attribute.

    Args:
        cls: The quality control class to register.

    Returns:
        The decorated class.
    """
    key = cls.name.lower()
    if key in _QC_REGISTRY:
        raise DuplicateRegistryKeyError(f"QC check key: '{key}' already registered")

    _QC_REGISTRY[key] = cls
    return cls


def available_qc_checks() -> list[str]:
    """Return a sorted list of available qc check names."""
    return sorted(_QC_REGISTRY.keys())


@dataclass(frozen=True)
class QcCtx:
    """Immutable context passed to QC checks."""

    df: pl.DataFrame
    time_name: str


class QCCheck(ABC):
    """Base class for quality control checks."""

    name: ClassVar[str]

    @abstractmethod
    def expr(self, _ctx: QcCtx, _column: str) -> pl.Expr:
        """Return the boolean Polars expression for this QC check (result of True == Value failed the QC check)"""
        pass

    @classmethod
    def get(cls, check: "str | QCCheck | Type[QCCheck]", **kwargs) -> "QCCheck":
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
            UnknownQcError: If a string name is not registered as a QC check.
            QcTypeError: If the input type is not supported or a class doesn't inherit from QCCheck.

        Examples::
            # From string
            qc = QCCheck.get("threshold")

            # From class
            qc = QCCheck.get(ThresholdCheck)

            # From instance
            qc = QCCheck.get(ThresholdCheck(arg1, arg2))
        """
        # If it's already an instance, return it
        if isinstance(check, QCCheck):
            return check

        # If it's a string, look it up in the registry
        if isinstance(check, str):
            try:
                return _QC_REGISTRY[check.lower()](**kwargs)
            except KeyError:
                raise UnknownQcError(f"Unknown QC check '{check}'. Available methods: {available_qc_checks()}")

        # If it's a class, check the subclass type and return
        elif isinstance(check, type):
            if issubclass(check, QCCheck):
                return check(**kwargs)  # type: ignore[misc]
            else:
                raise QcTypeError(f"QC check class '{check.__name__}' must inherit from QCCheck")

        else:
            raise QcTypeError(f"QC check must be a string or a QCCheck class. Got '{type(check).__name__}'")

    def apply(
        self,
        df: pl.DataFrame,
        time_name: str,
        check_column: str,
        observation_interval: datetime | tuple[datetime, datetime | None] | None = None,
    ) -> pl.Series:
        """Apply the QC check to the data.

        Args:
            df: The Polars DataFrame containing the time series data to quality control
            time_name: Name of the time column in the dataframe
            check_column: The column to perform the check on.
            observation_interval: Optional time interval to limit the check to.

        Returns:
            pl.Series: Boolean series of the resolved expression.
        """
        ctx = QcCtx(df, time_name)
        pipeline = QcCheckPipeline(self, ctx, check_column, observation_interval)
        return pipeline.execute()


class QcCheckPipeline:
    """Encapsulates the logic for the QC pipeline steps."""

    def __init__(
        self,
        qc_check: QCCheck,
        ctx: QcCtx,
        column: str,
        observation_interval: datetime | tuple[datetime, datetime | None] | None = None,
    ):
        self.qc_check = qc_check
        self.ctx = ctx
        self.column = column
        self.observation_interval = observation_interval

    def execute(self) -> pl.Series:
        """Execute the quality control check pipeline

        Returns:

        """
        self._validate()

        # Get the check expression
        check_expr = self.qc_check.expr(self.ctx, self.column)

        # Apply observation interval filter if specified
        if self.observation_interval:
            date_filter = get_date_filter(self.ctx.time_name, self.observation_interval)
            check_expr = check_expr & date_filter

        # Evaluate and return the result of the QC check
        #   Name as empty string to avoid accidental collisions.
        #   Up to user if they want to name it and add on to the dataframe.
        result = self.ctx.df.select(check_expr.alias("")).to_series()

        return result

    def _validate(self) -> None:
        """Carry out validation that the QC check can actually be carried out."""
        if self.ctx.df.is_empty():
            raise QcError("Cannot perform QC check on an empty DataFrame.")
        check_columns_in_dataframe(self.ctx.df, [self.column, self.ctx.time_name])


@register_qc_check
class ComparisonCheck(QCCheck):
    """Compares values against a given value using a comparison operator."""

    name = "comparison"

    def __init__(self, compare_to: float | list, operator: str, flag_na: bool = False) -> None:
        """Initialize comparison check.

        Args:
            compare_to: The value for comparison.
            operator: Comparison operator. One of: '>', '>=', '<', '<=', '==', '!=', 'is_in'.
            flag_na: If True, also flag NaN/null values as failing the check. Defaults to False.
        """
        self.compare_to = compare_to
        self.operator = operator
        self.flag_na = flag_na

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for threshold checking."""
        operator_map = {
            ">": pl.col(column) > self.compare_to,
            ">=": pl.col(column) >= self.compare_to,
            "<": pl.col(column) < self.compare_to,
            "<=": pl.col(column) <= self.compare_to,
            "==": pl.col(column) == self.compare_to,
            "!=": pl.col(column) != self.compare_to,
            "is_in": pl.col(column).is_in(self.compare_to),
        }

        if self.operator not in operator_map:
            raise QcUnknownOperatorError(f"Invalid operator '{self.operator}'. Use: {', '.join(operator_map.keys())}")

        operator_expr = operator_map[self.operator]
        if self.flag_na:
            operator_expr = operator_expr | pl.col(column).is_null()

        return operator_expr


@register_qc_check
class RangeCheck(QCCheck):
    """Check that values fall within an acceptable range."""

    name = "range"

    def __init__(
        self,
        min_value: float | time | date | datetime,
        max_value: float | time | date | datetime,
        closed: str | ClosedInterval = "both",
        within: bool = True,
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

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for range checking."""
        if type(self.min_value) is not type(self.max_value):
            raise TypeError("'min_value' and 'max_value' must be of same type")

        check_type = type(self.min_value)

        # Check if we're doing a time-based range check
        if check_type is time:
            col_expr = pl.col(column).dt.time()

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
            col_expr = pl.col(column).dt.date()

        else:
            # This should handle numeric objects and datetime.datetime objects
            col_expr = pl.col(column)

        in_range = col_expr.is_between(
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

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        return super().expr(ctx, ctx.time_name)


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

    def expr(self, ctx: QcCtx, column: str) -> pl.Expr:
        """Return the Polars expression for spike detection.

        The algorithm:
        1. Calculate differences between current value and neighbors
        2. Compute total combined difference and skew
        3. Flag where (total_difference - skew) > threshold * 2
        """
        # Calculate differences with temporal neighbors
        prev_val = pl.col(column).shift(1)
        next_val = pl.col(column).shift(-1)

        diff_prev = pl.col(column) - prev_val
        diff_next = next_val - pl.col(column)

        # Calculate total difference and skew
        d = (diff_prev - diff_next).abs()
        skew = (diff_prev.abs() - diff_next.abs()).abs()
        d_no_skew = d - skew

        # Double the threshold since we're summing differences
        return d_no_skew > (self.threshold * 2.0)
