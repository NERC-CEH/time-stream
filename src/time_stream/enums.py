"""
Time-Stream Enumerations.

This module defines several enums used throughout time_stream to standardise constants across the package.
"""

from enum import Enum


class DuplicateOption(Enum):
    """Enum representing the options for handling duplicate timestamp rows in a TimeFrame.

    Attributes:
        DROP: Raise an error if duplicate rows are found.
        KEEP_FIRST: Keep the first row of any duplicate groups.
        KEEP_LAST: Keep the last row of any duplicate groups.
        DROP: Drop all duplicate rows.
        MERGE: Merge duplicate rows using "coalesce" (the first non-null value for each column takes precedence).
    """

    DROP = "drop"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    ERROR = "error"
    MERGE = "merge"


class MissingCriteria(Enum):
    """Enum representing the missing criteria options available for aggregation

    Attributes:
        PERCENT: Require X% of data to be present
        MISSING: Allow at most X missing values
        AVAILABLE: Require at least X values present
        NA: Not applicable - no criteria for completeness
    """

    PERCENT = "percent"
    MISSING = "missing"
    AVAILABLE = "available"
    NA = "na"


class ClosedInterval(Enum):
    """Enum representing how to handle the edges of intervals

    Attributes:
        BOTH: Both edges are considered closed (inclusive)
        LEFT: Only the left edge is considered closed (left inclusive, right exclusive)
        RIGHT: Only the right edge is considered closed (left exclusive, right inclusive)
        NONE: Neither edge is considered closed (exclusive)
    """

    BOTH = "both"
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class TimeAnchor(Enum):
    """Enum representing how time is anchored at the measured point and therefore over which period of time
    a value is valid for.

    In the descriptions below, "x" is the time value, "r" stands for a single unit of the resolution of the data
    (15-minute, 1-hour, 1-day, etc.).

    Attributes:
        POINT: The time stamp is anchored for the instant of time "x".
               A value at "x" is considered valid only for the instant of time "x".
        START: The time stamp is anchored starting at "x".
               A value at "x" is considered valid starting at "x" (inclusive) and ending at "x+r" (exclusive).
        END: The time stamp is anchored ending at "x".
             A value at "x" is considered valid starting at "x-r" (exclusive) and ending at "x" (inclusive).
    """

    POINT = "point"
    START = "start"
    END = "end"


class ValidationErrorOptions(Enum):
    """Enum representing generic options for handling validation error.

    Attributes:
        ERROR: Raise an error.
        RESOLVE: Fix the validation issues automatically.

    """

    ERROR = "error"
    RESOLVE = "resolve"


class RollingAlignment(Enum):
    """Enum representing how a rolling window is aligned relative to each timestamp.

    Attributes:
        TRAILING: The window looks backward from the timestamp. Window: (t - window_size, t].
            Edge effects appear at the start of the series.
        LEADING: The window looks forward from the timestamp. Window: [t, t + window_size).
            Edge effects appear at the end of the series.
        CENTER: The window is centered on the timestamp. Window: [t - window_size/2, t + window_size/2].
            Edge effects appear at both the start and end of the series.
            Not supported for calendar-based periods (months, years).
    """

    TRAILING = "trailing"
    LEADING = "leading"
    CENTER = "center"
