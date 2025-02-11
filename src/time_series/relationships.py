from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import is for type hinting only.  Make sure there is no runtime import, to avoid recursion.
    from time_series import TimeSeries
    from time_series.columns import TimeSeriesColumn


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


class Relationship:
    """Represents a relationship between columns in a TimeSeries."""

    def __init__(
        self,
        primary_column: "TimeSeriesColumn",
        other_column: "TimeSeriesColumn",
        relationship_type: RelationshipType,
        deletion_policy: DeletionPolicy,
    ) -> None:
        """Initializes a Relationship object.

        Args:
            primary_column: Primary column in the relationship
            other_column: Other related column in the relationship
            relationship_type: Type of relationship.
            deletion_policy: Policy for deletion.
        """
        self.primary_column = primary_column
        self.other_column = other_column
        self.relationship_type = relationship_type
        self.deletion_policy = deletion_policy

    def __repr__(self) -> str:
        """Returns a string representation of the Relationship instance, summarising key properties."""
        return f"{type(self).__name__}('{self.primary_column.name} - {self.other_column.name}')"

    def __hash__(self) -> int:
        """Returns a hash value based on column names, ensuring consistency across instances.

        Returns:
            int: The hash value of the relationship.
        """
        return hash((self.primary_column.name, self.other_column.name, self.relationship_type, self.deletion_policy))

    def __eq__(self, other: object) -> bool:
        """Check equality between two Relationship instances.

        Args:
            other: The object to compare.

        Returns:
            bool: True if both relationships are identical.
        """
        if not isinstance(other, Relationship):
            return NotImplemented

        return (
            self.primary_column.name == other.primary_column.name
            and self.other_column.name == other.other_column.name
            and self.relationship_type == other.relationship_type
            and self.deletion_policy == other.deletion_policy
        )

    def __ne__(self, other: object) -> bool:
        """Checks if two Relationship instances are not equal.

        Args:
            other: The object to compare.

        Returns:
            bool: True if the Relationship instances are not equal, False otherwise.
        """
        return not self.__eq__(other)


class RelationshipManager:
    """Manages column relationships for a TimeSeries object."""

    def __init__(self, ts: "TimeSeries"):
        """Initializes the column relationship manager for a TimeSeries.

        Args:
            ts: The parent TimeSeries instance.
        """
        self._ts = ts
        self._relationships = {}
        self._setup_relationships()

    def _setup_relationships(self) -> None:
        """Initializes relationship storage for each column in the TimeSeries."""
        for column in self._ts.columns.values():
            self._relationships[column.name] = set()

    def _add(self, relationship: Relationship) -> None:
        """Adds a new relationship.

        Args:
            relationship: The relationship to add.

        Raises:
            ValueError: If a one-to-one or many-to-one relationship already exists.
        """
        self._check_relationship_type_validity(relationship)
        self._relationships[relationship.primary_column.name].add(relationship)
        self._relationships[relationship.other_column.name].add(relationship)

    def _check_relationship_type_validity(self, relationship: Relationship) -> None:
        """Checks that the relationship type is valid, based on the relationship type and existing relationships.

        Args:
            relationship: The relationship to check.
        """

        def __check(col: "TimeSeriesColumn") -> None:
            existing_relationships = self._relationships[col.name]

            for _relationship in existing_relationships:
                try:
                    if _relationship.primary_column == col:
                        # If this column is the primary column in the relationship, any "_TO_ONE" relationship type
                        #   means that it can only have one relationship with another column
                        if _relationship.relationship_type in (
                            RelationshipType.ONE_TO_ONE,
                            RelationshipType.MANY_TO_ONE,
                        ):
                            raise ValueError

                    elif _relationship.other_column == col:
                        # If this column is the secondary column in the relationship, any "ONE_TO_" relationship type
                        #   means that it can only have one relationship with another column
                        if _relationship.relationship_type in (
                            RelationshipType.ONE_TO_ONE,
                            RelationshipType.ONE_TO_MANY,
                        ):
                            raise ValueError

                except ValueError:
                    raise ValueError(
                        f"{col.name} can only be related to one column. "
                        f"Existing relationships: {existing_relationships}"
                    )

        __check(relationship.primary_column)
        __check(relationship.other_column)

    def _remove(self, relationship: Relationship) -> None:
        """Removes a relationship (from both directions).

        Args:
            relationship: The relationship to remove.
        """
        if relationship in self._get_relationships(relationship.primary_column):
            self._relationships[relationship.primary_column.name].remove(relationship)
        if relationship in self._get_relationships(relationship.other_column):
            self._relationships[relationship.other_column.name].remove(relationship)

    def _get_relationships(self, column: "TimeSeriesColumn") -> list[Relationship]:
        """Retrieves relationships for a column.

        Args:
            column: The column.

        Returns:
            The set of relationships the column has.
        """
        return list(self._relationships.get(column.name, set()))

    def _handle_deletion(self, column: "TimeSeriesColumn") -> None:
        """Handles the deletion of a column based on the relationship's deletion policy.

        Args:
           column: The column being deleted.

        Raises:
           ValueError: If RESTRICT policy is enforced and relationships exist.
        """
        relationships = self._get_relationships(column)

        for relationship in relationships:
            if relationship.deletion_policy.value == DeletionPolicy.CASCADE.value:
                self._relationships[relationship.other_column.name].discard(relationship)
                self._remove(relationship)
                relationship.other_column.remove()

            elif relationship.deletion_policy.value == DeletionPolicy.UNLINK.value:
                self._remove(relationship)

            elif relationship.deletion_policy.value == DeletionPolicy.RESTRICT.value:
                raise UserWarning(
                    f"Cannot delete {column} because it has restricted relationship "
                    f"with: {relationship.other_column.name}"
                )

        self._relationships.pop(column.name, None)
