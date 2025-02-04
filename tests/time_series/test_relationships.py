from datetime import datetime
import unittest
import itertools

from parameterized import parameterized
import polars as pl
from polars.testing import assert_series_equal

from time_series import TimeSeries
from time_series.relationships import Relationship, RelationshipType, DeletionPolicy


class BaseRelationshipTest(unittest.TestCase):
    """Base class for setting up test fixtures for these tests."""

    def setUp(self):
        """Set up a mock TimeSeries testing."""
        self.df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "colA": [1, 2, 3],
            "colB": [10, 20, 30],
        })
        self.ts = TimeSeries(df=self.df, time_name="time")


def get_enum_combos(
        not_relationship_type: RelationshipType=None,
        not_deletion_policy: DeletionPolicy=None
) -> list[tuple[str, RelationshipType, DeletionPolicy]]:
    """ Generates a list of all possible combinations of RelationshipType and DeletionPolicy, filtering out those
    that match the specified exceptions.

    Args:
        not_relationship_type: A specific RelationshipType to exclude from the combinations.
        not_deletion_policy: A specific DeletionPolicy to exclude from the combinations.

    Returns:
        list: A list of tuples where each tuple represents a combination containing the formatted string,
            the RelationshipType, and the DeletionPolicy.
    """
    combos = [(f"{t}_{d}", t, d) for t, d in itertools.product(RelationshipType, DeletionPolicy)
              if not t is not_relationship_type and not d is not_deletion_policy]
    return combos


class TestRelationshipHash(BaseRelationshipTest):
    @parameterized.expand(get_enum_combos())
    def test_identical_hash(self, _, relationship_type, deletion_policy):
        """ Test that two relationships with identical attributes have the same hash. """
        rel1 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        rel2 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.assertEqual(rel1.__hash__(), rel2.__hash__())

    @parameterized.expand(get_enum_combos(not_relationship_type=RelationshipType.ONE_TO_ONE))
    def test_non_identical_relationship_type_hash(self, _, relationship_type, deletion_policy):
        """ Test that two relationships with different relationship types don't have the same hash. """
        rel1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_ONE, deletion_policy)
        rel2 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.assertNotEqual(rel1.__hash__(), rel2.__hash__())

    @parameterized.expand(get_enum_combos(not_deletion_policy=DeletionPolicy.CASCADE))
    def test_non_identical_deletion_policy_hash(self, _, relationship_type, deletion_policy):
        """ Test that two relationships with different deletion policies don't have the same hash. """
        rel1 = Relationship(self.ts.colA, self.ts.colB, relationship_type, DeletionPolicy.CASCADE)
        rel2 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.assertNotEqual(rel1.__hash__(), rel2.__hash__())

    def test_non_identical_column_hash(self):
        """ Test that two relationships with different columns don't have the same hash. """
        rel1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_MANY, DeletionPolicy.CASCADE)
        rel2 = Relationship(self.ts.colB, self.ts.colA, RelationshipType.ONE_TO_MANY, DeletionPolicy.CASCADE)
        self.assertNotEqual(rel1.__hash__(), rel2.__hash__())


class TestRelationshipEq(BaseRelationshipTest):
    @parameterized.expand(get_enum_combos())
    def test_identical_are_eq(self, _, relationship_type, deletion_policy):
        """ Test that two relationships with identical attributes are considered equal. """
        rel1 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        rel2 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.assertTrue(rel1 == rel2)
        self.assertFalse(rel1 != rel2)

    @parameterized.expand(get_enum_combos(not_relationship_type=RelationshipType.ONE_TO_ONE))
    def test_non_identical_relationship_type_noteq(self, _, relationship_type, deletion_policy):
        """ Test that two relationships with different relationship types are not considered equal."""
        rel1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_ONE, deletion_policy)
        rel2 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.assertFalse(rel1 == rel2)
        self.assertTrue(rel1 != rel2)

    @parameterized.expand(get_enum_combos(not_deletion_policy=DeletionPolicy.CASCADE))
    def test_non_identical_deletion_policy_noteq(self, _, relationship_type, deletion_policy):
        """ Test that two relationships with different deletion policies are not considered equal."""
        rel1 = Relationship(self.ts.colA, self.ts.colB, relationship_type, DeletionPolicy.CASCADE)
        rel2 = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.assertFalse(rel1 == rel2)
        self.assertTrue(rel1 != rel2)

    def test_non_identical_column_noteq(self):
        """ Test that two relationships with different columns are not considered equal."""
        rel1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_MANY, DeletionPolicy.CASCADE)
        rel2 = Relationship(self.ts.colB, self.ts.colA, RelationshipType.ONE_TO_MANY, DeletionPolicy.CASCADE)
        self.assertFalse(rel1 == rel2)
        self.assertTrue(rel1 != rel2)
