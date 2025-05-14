from enum import Enum


class DuplicateOption(Enum):
    """Enum representing the options for handling duplicate timestamp rows in a TimeSeries."""
    DROP = "drop"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    ERROR = "error"
    MERGE = "merge"
