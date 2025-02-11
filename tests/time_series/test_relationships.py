from datetime import datetime
import unittest
import itertools

from parameterized import parameterized
import polars as pl

from time_series import TimeSeries
from time_series.relationships import Relationship, RelationshipManager, RelationshipType, DeletionPolicy


class BaseRelationshipTest(unittest.TestCase):
    """Base class for setting up test fixtures for these tests."""

    @classmethod
    def setUpClass(cls):
        """Set up a mock TimeSeries testing."""
        cls.df = pl.DataFrame({
            "time": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "colA": [1, 2, 3],
            "colB": [10, 20, 30],
            "colC": [100, 200, 300],
        })
        cls.ts = TimeSeries(df=cls.df, time_name="time")

    @classmethod
    def get_cols(cls):
        return cls.ts.columns

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


class TestRelationshipManagerAdd(BaseRelationshipTest):
    def setUp(self):
        self.relationship_manager = RelationshipManager(self.ts)

    @parameterized.expand(get_enum_combos())
    def test_add_relationship(self, _, relationship_type, deletion_policy):
        """Test adding a Relationship with all combos of relationship type and deletion policies to the manager"""
        relationship = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.relationship_manager._add(relationship)
        
        primary_col_relationships = self.relationship_manager._get_relationships(self.ts.colA)
        self.assertEqual(len(primary_col_relationships), 1)
        self.assertIn(relationship, primary_col_relationships)

        other_col_relationships = self.relationship_manager._get_relationships(self.ts.colB)
        self.assertEqual(len(other_col_relationships), 1)
        self.assertIn(relationship, other_col_relationships)

    @parameterized.expand([(str(d), d) for d in DeletionPolicy])
    def test_add_multiple_one_to_one_failure(self, _, deletion_policy):
        """Test adding a second one-to-one Relationship for the same primary column raises an error"""
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_ONE, deletion_policy)
        self.relationship_manager._add(relationship1)

        relationship2 = Relationship(self.ts.colA, self.ts.colC, RelationshipType.ONE_TO_ONE, deletion_policy)
        with self.assertRaises(ValueError):
            self.relationship_manager._add(relationship2)

    @parameterized.expand([(str(d), d) for d in DeletionPolicy])
    def test_add_multiple_many_to_one_failure(self, _, deletion_policy):
        """Test adding a second many-to-one Relationship for the same primary column raises an error"""
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.MANY_TO_ONE, deletion_policy)
        self.relationship_manager._add(relationship1)

        relationship2 = Relationship(self.ts.colA, self.ts.colC, RelationshipType.MANY_TO_ONE, deletion_policy)
        with self.assertRaises(ValueError):
            self.relationship_manager._add(relationship2)

    @parameterized.expand([(str(d), d) for d in DeletionPolicy])
    def test_add_multiple_many_to_many_success(self, _, deletion_policy):
        """Test adding multiple many-to-many Relationships for the same primary column is successful"""
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.MANY_TO_MANY, deletion_policy)
        relationship2 = Relationship(self.ts.colA, self.ts.colC, RelationshipType.MANY_TO_MANY, deletion_policy)

        self.relationship_manager._add(relationship1)
        self.relationship_manager._add(relationship2)

        primary_col_relationships = self.relationship_manager._get_relationships(self.ts.colA)
        self.assertEqual(len(primary_col_relationships), 2)
        self.assertIn(relationship1, primary_col_relationships)
        self.assertIn(relationship2, primary_col_relationships)

        other_col_relationships = self.relationship_manager._get_relationships(self.ts.colB)
        self.assertEqual(len(other_col_relationships), 1)
        self.assertIn(relationship1, other_col_relationships)

        other_col_relationships = self.relationship_manager._get_relationships(self.ts.colC)
        self.assertEqual(len(other_col_relationships), 1)
        self.assertIn(relationship2, other_col_relationships)

    @parameterized.expand([(str(d), d) for d in DeletionPolicy])
    def test_add_multiple_one_to_many_success(self, _, deletion_policy):
        """Test adding multiple one-to-many Relationships for the same primary column is successful"""
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_MANY, deletion_policy)
        relationship2 = Relationship(self.ts.colA, self.ts.colC, RelationshipType.ONE_TO_MANY, deletion_policy)

        self.relationship_manager._add(relationship1)
        self.relationship_manager._add(relationship2)

        colA_relationships = self.relationship_manager._get_relationships(self.ts.colA)
        colB_relationships = self.relationship_manager._get_relationships(self.ts.colB)
        colC_relationships = self.relationship_manager._get_relationships(self.ts.colC)

        self.assertEqual(len(colA_relationships), 2)
        self.assertEqual(len(colB_relationships), 1)
        self.assertEqual(len(colC_relationships), 1)
        self.assertIn(relationship1, colA_relationships)
        self.assertIn(relationship2, colA_relationships)
        self.assertIn(relationship1, colB_relationships)
        self.assertIn(relationship2, colC_relationships)

    @parameterized.expand([(str(d), d) for d in DeletionPolicy])
    def test_add_relationship_to_existing_one_to_many_fails(self, _, deletion_policy):
        """Test adding another Relationship to a column that has a ONE-TO-MANY Relationship already raises error."""
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_MANY, deletion_policy)
        relationship2 = Relationship(self.ts.colB, self.ts.colC, RelationshipType.MANY_TO_MANY, deletion_policy)

        self.relationship_manager._add(relationship1)
        with self.assertRaises(ValueError):
            self.relationship_manager._add(relationship2)


class TestRelationshipManagerRemove(BaseRelationshipTest):
    def setUp(self):
        self.relationship_manager = RelationshipManager(self.ts)

    @parameterized.expand(get_enum_combos())
    def test_remove(self, _, relationship_type, deletion_policy):
        """Test that removing a relationship removes it from both directions."""
        relationship = Relationship(self.ts.colA, self.ts.colB, relationship_type, deletion_policy)
        self.relationship_manager._add(relationship)
        self.assertIn(relationship, self.relationship_manager._get_relationships(self.ts.colA))
        self.assertIn(relationship, self.relationship_manager._get_relationships(self.ts.colB))

        self.relationship_manager._remove(relationship)
        self.assertNotIn(relationship, self.relationship_manager._get_relationships(self.ts.colA))
        self.assertNotIn(relationship, self.relationship_manager._get_relationships(self.ts.colB))


class TestRelationshipManagerGetRelationships(BaseRelationshipTest):
    def setUp(self):
        self.relationship_manager = RelationshipManager(self.ts)

    @parameterized.expand([(str(d), d) for d in DeletionPolicy])
    def test_get_relationships(self, _, deletion_policy):
        """Test getting relationships."""
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.MANY_TO_MANY, deletion_policy)
        relationship2 = Relationship(self.ts.colA, self.ts.colC, RelationshipType.MANY_TO_MANY, deletion_policy)
        self.relationship_manager._add(relationship1)
        self.relationship_manager._add(relationship2)

        result = self.relationship_manager._get_relationships(self.ts.colA)
        self.assertEqual(len(result), 2)
        self.assertIn(relationship1, result)
        self.assertIn(relationship2, result)

        result = self.relationship_manager._get_relationships(self.ts.colB)
        self.assertEqual(len(result), 1)
        self.assertIn(relationship1, result)

        result = self.relationship_manager._get_relationships(self.ts.colC)
        self.assertEqual(len(result), 1)
        self.assertIn(relationship2, result)

    def test_get_relationships_empty(self):
        """Test getting relationships when there are none for a column."""
        for _, col in self.ts.columns.items():
            result = self.relationship_manager._get_relationships(col)
            self.assertEqual(result, [])


class TestRelationshipManagerHandleDeletion(BaseRelationshipTest):
    def setUp(self):
        self.ts = TimeSeries(df=self.df, time_name="time")

    def test_cascade(self):
        """Test deletion handling with CASCADE policy. Deleting colA should cascade and remove colB.
        """
        relationship = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_ONE, DeletionPolicy.CASCADE)
        self.ts._relationship_manager._add(relationship)

        col_a = self.ts.colA
        col_b = self.ts.colB

        self.ts._relationship_manager._handle_deletion(self.ts.colA)

        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_a))
        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_b))
        self.assertNotIn(col_b.name, self.ts.columns)

    def test_multiple_cascade(self):
        """Test deletion handling with CASCADE policy, where the second column also has a CASCADE relationship.
        """
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.MANY_TO_MANY, DeletionPolicy.CASCADE)
        relationship2 = Relationship(self.ts.colB, self.ts.colC, RelationshipType.MANY_TO_MANY, DeletionPolicy.CASCADE)
        self.ts._relationship_manager._add(relationship1)
        self.ts._relationship_manager._add(relationship2)

        col_a = self.ts.colA
        col_b = self.ts.colB
        col_c = self.ts.colC

        self.ts._relationship_manager._handle_deletion(self.ts.colA)

        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_a))
        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_b))
        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_c))
        self.assertNotIn(col_b.name, self.ts.columns)
        self.assertNotIn(col_c.name, self.ts.columns)

    def test_unlink(self):
        """Test deletion handling with UNLINK policy. Deleting colA should unlink the relationship with colB,
        but not delete colB
        """
        relationship = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_ONE, DeletionPolicy.UNLINK)
        self.ts._relationship_manager._add(relationship)

        col_a = self.ts.colA
        col_b = self.ts.colB

        self.ts._relationship_manager._handle_deletion(self.ts.colA)

        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_a))
        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_b))
        self.assertIn(col_b.name, self.ts.columns)

    def test_multiple_unlink(self):
        """Test deletion handling with UNLINK policy, where second column has other relationships.
        """
        relationship1 = Relationship(self.ts.colA, self.ts.colB, RelationshipType.MANY_TO_MANY, DeletionPolicy.UNLINK)
        relationship2 = Relationship(self.ts.colB, self.ts.colC, RelationshipType.MANY_TO_MANY, DeletionPolicy.UNLINK)
        self.ts._relationship_manager._add(relationship1)
        self.ts._relationship_manager._add(relationship2)

        col_a = self.ts.colA
        col_b = self.ts.colB
        col_c = self.ts.colC

        self.ts._relationship_manager._handle_deletion(self.ts.colA)

        self.assertEqual([], self.ts._relationship_manager._get_relationships(col_a))
        self.assertEqual([relationship2], self.ts._relationship_manager._get_relationships(col_b))
        self.assertIn(col_b.name, self.ts.columns)
        self.assertIn(col_c.name, self.ts.columns)

    def test_restrict(self):
        """Test deletion handling with RESTRICT policy.  Should raise error.
        """
        relationship = Relationship(self.ts.colA, self.ts.colB, RelationshipType.ONE_TO_ONE, DeletionPolicy.RESTRICT)
        self.ts._relationship_manager._add(relationship)

        with self.assertRaises(UserWarning):
            self.ts._relationship_manager._handle_deletion(self.ts.colA)

        with self.assertRaises(UserWarning):
            self.ts._relationship_manager._handle_deletion(self.ts.colB)
