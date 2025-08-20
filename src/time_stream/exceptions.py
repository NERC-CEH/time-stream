from collections.abc import Iterable

from time_stream.enums import DuplicateOption


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
