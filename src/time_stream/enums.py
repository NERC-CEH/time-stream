from enum import Enum


class DuplicateOption(Enum):
    """Enum representing the options for handling duplicate timestamp rows in a TimeSeries.

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


class MissingCriteriaOptions(Enum):
    """An enumeration to represent various options for handling missing data within aggregation operations.

    Attributes:
        MISSING: Calculate a value only if there are no more than n values missing in the period.
        AVAILABLE: Calculate a value only if there are at least n input values in the period.
        PERCENT: Calculate a value only if the data in the period is at least n percent complete.
    """

    MISSING = "missing"
    AVAILABLE = "available"
    PERCENT = "percent"


class RelationshipType(Enum):
    """Enum representing the type of relationship between columns.

    Attributes:
        ONE_TO_ONE: A one-to-one relationship
        ONE_TO_MANY: A one-to-many relationship (e.g., one data column can be linked to many flag columns).
        MANY_TO_ONE: A many-to-one relationship (e.g., many flag columns can be linked to one data column).
        MANY_TO_MANY: A many-to-many relationship (e.g., multiple supplementary columns to multiple data columns).
    """

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class DeletionPolicy(Enum):
    """Enum representing the policy when a related column is deleted.

    Attributes:
        CASCADE: Deletes the related column when the main column is removed.
        UNLINK: Unlinks the relationship but keeps the related column.
        RESTRICT: Prevents deletion if there are active relationships.
    """

    CASCADE = "cascade"
    UNLINK = "unlink"
    RESTRICT = "restrict"


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
