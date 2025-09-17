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


class DuplicateValueError(TimeStreamError):
    """Raised when values that should be unique are found to have duplicates."""


class DuplicateTimeError(DuplicateValueError):
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


class QcUnknownOperatorError(QcError):
    """Raised when an invalid operator is passed to a QC check."""


class InfillError(TimeStreamError):
    """Base class for infill-related errors"""


class InfillInsufficientValuesError(InfillError):
    """Raised when there are insufficient number of values to carry out the infill method."""


class PeriodError(Exception):
    """Base exception for all period-related errors."""


class PeriodConfigError(PeriodError):
    """Raised when constructing or configuring objects within the Period module with unsupported options."""


class PeriodParsingError(PeriodError):
    """Raised when things like a period string or timedelta cannot be parsed."""


class PeriodValidationError(PeriodError):
    """Raised when a validation on objects in the Period module fails."""


class FlagSystemError(TimeStreamError):
    """Base class for flag-system related errors."""


class DuplicateFlagSystemError(FlagSystemError):
    """Raised when a flag system already exists."""


class FlagSystemTypeError(FlagSystemError):
    """Raised when an invalid flag system type is provided."""


class FlagSystemNotFoundError(FlagSystemError):
    """Raised when a flag system can't be found."""


class UnhandledEnumError(TimeStreamError):
    """Base class for unhandled enumeration related errors."""


class RegistryError(TimeStreamError):
    """Base class for registry-related errors."""


class DuplicateRegistryKeyError(RegistryError):
    """Raised when a registry key already exists."""


class UnknownRegistryKeyError(RegistryError):
    """Raised when a registry key doesn't exist in the registry."""


class RegistryKeyTypeError(RegistryError):
    """Raised when a registry key is an incorrect type."""
