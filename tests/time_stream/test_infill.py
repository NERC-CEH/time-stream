import unittest
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream.infill import (InfillMethod, BSplineInterpolation, CubicInterpolation, QuadraticInterpolation,
                                LinearInterpolation, AkimaInterpolation, PchipInterpolation)

# Data used through the tests
LINEAR = pl.DataFrame({"values": [1., None, 3., None, 5.]})  # Linear progression
QUADRATIC = pl.DataFrame({"values": [0., None, 4., None, 16., None, 36.]})  # Quadratic data: y = x^2
CUBIC = pl.DataFrame({"values": [0., None, 8., None, 64., None, 216., None, 512.]})  # Cubic data: y = x^3
INSUFFICIENT_DATA = pl.DataFrame({"values": [1., None, None, None, None]}) # Insufficient data
COMPLETE = pl.DataFrame({"values": [1., 2., 3., 4., 5.]}) # No missing data
ALL_MISSING = pl.DataFrame({"values": [None, None, None, None, None]}) # All missing data
VARYING_GAPS = pl.DataFrame({"values": [1., None, 3., None, None, 6., None, None, None, 10.]})


class MockInfill(InfillMethod):
    name = "Mock"

    def _fill(self, df, infill_column):
        return df[infill_column]


class TestInfillMethod(unittest.TestCase):
    """Test the base InfillMethod class."""
    def setUp(self):
        """Set up test fixtures."""
        self.mock_ts = Mock()
        self.mock_ts.time_name = "timestamp"
        self.mock_ts.df = pl.DataFrame({"timestamp": [datetime(2025, m, 1) for m in range(1, 8)]})

    @parameterized.expand([
        ("linear", LinearInterpolation),
        ("cubic", CubicInterpolation),
        ("akima", AkimaInterpolation),
    ])
    def test_get_with_string(self, get_input, expected):
        """Test QCCheck.get() with string input."""
        infill = InfillMethod.get(get_input)
        self.assertIsInstance(infill, expected)

    @parameterized.expand([
        (LinearInterpolation, LinearInterpolation),
        (CubicInterpolation, CubicInterpolation),
        (AkimaInterpolation, AkimaInterpolation)
    ])
    def test_get_with_class(self, get_input, expected):
        """Test InfillMethod.get() with class input."""
        infill = InfillMethod.get(get_input)
        self.assertIsInstance(infill, expected)

    @parameterized.expand([
        (LinearInterpolation(), LinearInterpolation),
        (CubicInterpolation(), CubicInterpolation),
        (AkimaInterpolation(), AkimaInterpolation)
    ])
    def test_get_with_instance(self, get_input, expected):
        """Test InfillMethod.get() with instance input."""
        infill = InfillMethod.get(get_input)
        self.assertIsInstance(infill, expected)

    @parameterized.expand([
        "dummy", "RANGE", "123"
    ])
    def test_get_with_invalid_string(self, get_input):
        """Test InfillMethod.get() with invalid string."""
        with self.assertRaises(KeyError):
            InfillMethod.get(get_input)

    def test_get_with_invalid_class(self):
        """Test InfillMethod.get() with invalid class."""

        class InvalidClass:
            pass

        with self.assertRaises(TypeError):
            InfillMethod.get(InvalidClass)  # noqa - expecting type warning

    @parameterized.expand([
        (123,), ([LinearInterpolation, QuadraticInterpolation],), ({AkimaInterpolation},)
    ])
    def test_get_with_invalid_type(self, get_input):
        """Test InfillMethod.get() with invalid type."""
        with self.assertRaises(TypeError):
            InfillMethod.get(get_input)

    @parameterized.expand([
        (COMPLETE, 1, None, False),
        (LINEAR, 1, None, True),
        (INSUFFICIENT_DATA, 1, None, False),
        (ALL_MISSING, 1, None, False),
        (VARYING_GAPS, 1, None, True),
        (VARYING_GAPS, 2, None, True),
        (VARYING_GAPS, 3, None, True),
        (VARYING_GAPS, 1, (datetime(2025, 1, 3), datetime(2025, 1, 10)), False),
        (VARYING_GAPS, 2, (datetime(2025, 1, 6), datetime(2025, 1, 10)), False),
        (VARYING_GAPS, 3, datetime(2025, 1, 10), False),
    ])
    def test_anything_to_infill(self, df, max_gap_size, observation_interval, expected):
        """Test whether the anything to infill method returns expected boolean."""
        # Add timestamp column to input df
        df = df.with_columns(pl.Series("timestamp", [datetime(2025, 1, d) for d in range(1, len(df)+1)]))

        result = InfillMethod._anything_to_infill(
            df=df,
            time_name="timestamp",
            infill_column="values",
            max_gap_size=max_gap_size,
            observation_interval=observation_interval
        )
        self.assertEqual(result, expected)


class TestBSplineInterpolation(unittest.TestCase):
    def test_initialization(self):
        """Test BSplineInterpolation initialization."""
        # Custom order
        interp = BSplineInterpolation(order=2)
        self.assertEqual(interp.order, 2)
        self.assertEqual(interp.min_points_required, 3)

        # With scipy kwargs
        interp = BSplineInterpolation(order=1, bc_type="clamped")
        self.assertEqual(interp.scipy_kwargs['bc_type'], "clamped")


class TestLinearInterpolation(unittest.TestCase):

    @parameterized.expand([
        (LINEAR, [1., 2., 3., 4., 5.]),
        (QUADRATIC, [0., 2., 4., 10., 16., 26., 36.]),
        (CUBIC, [0., 4., 8., 36., 64., 140., 216., 364., 512.])
    ])
    def test_linear_interpolation_known_result(self, input_data, expected_data):
        """Test linear interpolation with known data."""
        result = LinearInterpolation()._fill(input_data, "values")

        expected = pl.Series("values_linear", expected_data)
        assert_series_equal(result["values_linear"], expected)

    @parameterized.expand([
        ("1 data points", INSUFFICIENT_DATA),
        ("0 data points", ALL_MISSING),
    ])
    def test_insufficient_data_raises_error(self, _, input_data):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            LinearInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self):
        """Test that complete data is unchanged."""
        result = LinearInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_linear", COMPLETE)
        assert_series_equal(result["values_linear"], expected)


class TestQuadraticInterpolation(unittest.TestCase):

    @parameterized.expand([
        (LINEAR, [1., 2., 3., 4., 5.]),
        (QUADRATIC, [0., 1., 4., 9., 16., 25., 36.]),
    ])
    def test_quadratic_interpolation_known_result(self, input_data, expected_data):
        """Test quadratic interpolation with known data."""
        result = QuadraticInterpolation()._fill(input_data, "values")

        expected = pl.Series("values_quadratic", expected_data)
        assert_series_equal(result["values_quadratic"], expected)

    @parameterized.expand([
        ("1 data points", INSUFFICIENT_DATA),
        ("0 data points", ALL_MISSING)
    ])
    def test_insufficient_data_raises_error(self, _, input_data):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            QuadraticInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self):
        """Test that complete data is unchanged."""
        result = QuadraticInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_quadratic", COMPLETE)
        assert_series_equal(result["values_quadratic"], expected)

class TestCubicInterpolation(unittest.TestCase):

    @parameterized.expand([
        (CUBIC, [0., 1., 8., 27., 64., 125., 216., 343., 512.]),
    ])
    def test_cubic_interpolation_known_result(self, input_data, expected_data):
        """Test cubic interpolation with known data."""
        result = CubicInterpolation()._fill(input_data, "values")

        expected = pl.Series("values_cubic", expected_data)
        assert_series_equal(result["values_cubic"], expected)

    @parameterized.expand([
        ("3 data points", LINEAR),
        ("1 data points", INSUFFICIENT_DATA),
        ("0 data points", ALL_MISSING)
    ])
    def test_insufficient_data_raises_error(self, _, input_data):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            CubicInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self):
        """Test that complete data is unchanged."""
        result = CubicInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_cubic", COMPLETE)
        assert_series_equal(result["values_cubic"], expected)

        
class TestAkimaInterpolation(unittest.TestCase):
    # Manually calculating the Akima interpolation isn't practical.
    #   Let's assume that SciPy is well tested and the Akima interpolation results are correct and let's just test
    #   behaviours of the interpolation class

    def test_initialization(self):
        """Test Akima initialization."""
        interp = AkimaInterpolation()
        self.assertEqual(interp.min_points_required, 5)
        self.assertEqual(interp.name, "akima")

        # With scipy kwargs
        interp = AkimaInterpolation(extrapolate=True)
        self.assertEqual(interp.scipy_kwargs['extrapolate'], True)

    def test_akima_interpolation_with_sufficient_data(self):
        """Test akima interpolation works when there is sufficient data (at least 5 points)."""
        result = AkimaInterpolation()._fill(CUBIC, "values")
        self.assertIn("values_akima", result.columns)

    @parameterized.expand([
        ("4 data points", QUADRATIC),
        ("3 data points", LINEAR),
        ("1 data points", INSUFFICIENT_DATA),
        ("0 data points", ALL_MISSING)
    ])
    def test_insufficient_data_raises_error(self, _, input_data):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            AkimaInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self):
        """Test that complete data is unchanged."""
        result = AkimaInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_akima", COMPLETE)
        assert_series_equal(result["values_akima"], expected)

        
class TestPchipInterpolation(unittest.TestCase):
    # Manually calculating the Pchip interpolation isn't practical.
    #   Let's assume that SciPy is well tested and the Pchip interpolation results are correct and let's just test
    #   behaviours of the interpolation class

    def test_initialization(self):
        """Test Akima initialization."""
        interp = PchipInterpolation()
        self.assertEqual(interp.min_points_required, 2)
        self.assertEqual(interp.name, "pchip")

        # With scipy kwargs
        interp = PchipInterpolation(extrapolate=True)
        self.assertEqual(interp.scipy_kwargs['extrapolate'], True)

    def test_pchip_interpolation_with_sufficient_data(self):
        """Test akima interpolation works when there is sufficient data (at least 2 points)."""
        result = PchipInterpolation()._fill(LINEAR, "values")
        self.assertIn("values_pchip", result.columns)

    @parameterized.expand([
        ("1 data points", INSUFFICIENT_DATA),
        ("0 data points", ALL_MISSING)
    ])
    def test_insufficient_data_raises_error(self, _, input_data):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            PchipInterpolation()._fill(input_data, "values")

    @parameterized.expand([
        (LINEAR,),
        (QUADRATIC,),
        (CUBIC,),
    ])
    def test_pchip_monotonic_preservation(self, input_data):
        """Part of the pchip behaviour is that it should preserve local monotonicity if the input data is monotonic."""
        result = PchipInterpolation()._fill(input_data, "values")
        interpolated = result["values_pchip"].to_numpy()

        # Check that result is monotonically increasing
        self.assertTrue(np.all(np.diff(interpolated) > 0))

    def test_complete_data_unchanged(self):
        """Test that complete data is unchanged."""
        result = PchipInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_pchip", COMPLETE)
        assert_series_equal(result["values_pchip"], expected)
