class TimeStreamError(Exception):
    """Base class for custom errors in the time-stream package."""


class ColumnNotFoundError(TimeStreamError):
    """Raised when a requested column does not exist."""


class ColumnTypeError(TimeStreamError):
    """Raised when a requested column is not the expected type."""


class DuplicateColumnError(TimeStreamError):
    """Raised when a column name is duplicated."""


class DuplicateTimeError(TimeStreamError):
    """Raised when duplicate time values are found."""


class MetadataError(TimeStreamError):
    """Raised when there is an error with the metadata within time series object."""


class PeriodicityError(TimeStreamError):
    """Raised when datetime values do not conform to the specified periodicity."""


class ResolutionError(TimeStreamError):
    """Raised when datetime values are not aligned to the specified resolution."""


class TimeMutatedError(TimeStreamError):
    """Raised when the time values have been detected as being mutated."""
