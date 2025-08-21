from collections.abc import Iterable

from time_stream.enums import DuplicateOption


class TimeStreamError(Exception):
    """Base class for custom errors in the time-stream package."""


class ColumnNotFoundError(TimeStreamError):
    """Raised when a requested column does not exist."""


class ColumnTypeError(TimeStreamError):
    """Raised when a requested column is not the expected type."""


class ColumnRelationshipError(TimeStreamError):
    """Raised when there is an issue with the relationship of one column to another"""


class DuplicateColumnError(TimeStreamError):
    """Raised when a column name is duplicated."""


class DuplicateTimeError(TimeStreamError):
    """Raised when duplicate time values are found."""

    def __init__(
        self,
        msg: str | None = None,
        duplicates: Iterable | None = None,
    ):
        if not msg:
            msg = (
                f"Duplicate time values found. A TimeSeries must have unique time values. "
                f"Options for dealing with duplicate rows include: {[o.name for o in DuplicateOption]}."
            )
        self.duplicates = duplicates
        super().__init__(msg)


class MetadataError(TimeStreamError):
    """Raised when there is an error with the metadata within time series object."""


class PeriodicityError(TimeStreamError):
    """Raised when datetime values do not conform to the specified periodicity."""


class ResolutionError(TimeStreamError):
    """Raised when datetime values are not aligned to the specified resolution."""


class TimeMutatedError(TimeStreamError):
    """Raised when the time values have been detected as being mutated."""

    def __init__(
        self, msg: str | None = None, old_timestamps: Iterable | None = None, new_timestamps: Iterable | None = None
    ):
        if not msg:
            msg = (
                "Time column has been modified. Adding, removing or modifying time rows requires creating a new "
                "TimeSeries instance."
            )
        self.old_timestamps = old_timestamps
        self.new_timestamps = new_timestamps
        super().__init__(msg)


class AggregationError(TimeStreamError):
    """Base class for aggregation-related errors"""


class UnknownAggregationError(AggregationError):
    """Raised when an unknown aggregation is requested."""


class AggregationTypeError(AggregationError):
    """Raised when an invalid aggregation type is provided."""


class AggregationPeriodError(AggregationError):
    """Raised when an aggregation period is invalid."""


class MissingCriteriaError(AggregationError):
    """Raised when invalid or inconsistent missing data criteria are provided."""


class BitwiseFlagError(TimeStreamError):
    """Base class for BitwiseFlag-related errors."""


class BitwiseFlagTypeError(BitwiseFlagError):
    """Raised when a flag value has the wrong type."""


class BitwiseFlagValueError(BitwiseFlagError):
    """Raised when a flag value is invalid (negative, not power of two, or zero)."""


class BitwiseFlagDuplicateError(BitwiseFlagError):
    """Raised when a flag value is not unique within the enumeration."""


class BitwiseFlagUnknownError(BitwiseFlagError):
    """Raised when a flag lookup fails or a non-singular flag is requested."""


class QcError(TimeStreamError):
    """Base class for qc-related errors"""


class UnknownQcError(QcError):
    """Raised when an unknown QC method is requested."""


class QcTypeError(QcError):
    """Raised when an invalid QC method type is provided."""


class QcUnknownOperatorError(QcError):
    """Raised when an invalid operator is passed to a QC check."""


class InfillError(TimeStreamError):
    """Base class for infill-related errors"""


class UnknownInfillError(InfillError):
    """Raised when an unknown infill method is requested."""


class InfillTypeError(InfillError):
    """Raised when an invalid infill method type is provided."""


class InfillInsufficientValuesError(InfillError):
    """Raised when there are insufficient number of values to carry out the infill method."""
