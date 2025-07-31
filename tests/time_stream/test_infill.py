import unittest
from datetime import datetime, time
from unittest.mock import Mock

import polars as pl
from parameterized import parameterized
from polars.testing import assert_series_equal

from time_stream.base import TimeSeries
from time_stream.infill import (InfillMethod, BSplineInterpolation, CubicInterpolation, QuadraticInterpolation,
                                LinearInterpolation, AkimaInterpolation, PchipInterpolation)

# Data used through the tests
LINEAR = pl.DataFrame({"values": [1.0, None, 3.0, None, 5.0]})  # Linear progression
QUADRATIC = pl.DataFrame({"values": [0.0, None, 4.0, None, 16.0, None, 36.0]})  # Quadratic data: y = x^2
CUBIC = pl.DataFrame({"values": [0.0, None, 8.0, None, 64.0, None, 216.0, None, 512.0]})  # Cubic data: y = x^3
INSUFFICIENT_DATA = pl.DataFrame({"values": [1.0, None, None, None, 5.0]}) # Insufficient data
COMPLETE = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0]}) # No missing data
ALL_MISSING = pl.DataFrame({"values": [None, None, None, None, None]}) # All missing data


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

    def test_all_missing_data_raises_error(self):
        """Test that all missing data raises ValueError."""
        with self.assertRaises(ValueError):
            LinearInterpolation()._fill(ALL_MISSING, "values")

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

    def test_all_missing_data_raises_error(self):
        """Test that all missing data raises ValueError."""
        with self.assertRaises(ValueError):
            QuadraticInterpolation()._fill(ALL_MISSING, "values")

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            QuadraticInterpolation()._fill(INSUFFICIENT_DATA, "values")


class TestCubicInterpolation(unittest.TestCase):

    @parameterized.expand([
        (CUBIC, [0., 1., 8., 27., 64., 125., 216., 343., 512.]),
    ])
    def test_cubic_interpolation_known_result(self, input_data, expected_data):
        """Test cubic interpolation with known data."""
        result = CubicInterpolation()._fill(input_data, "values")

        expected = pl.Series("values_cubic", expected_data)
        assert_series_equal(result["values_cubic"], expected)

    def test_all_missing_data_raises_error(self):
        """Test that all missing data raises ValueError."""
        with self.assertRaises(ValueError):
            CubicInterpolation()._fill(ALL_MISSING, "values")

    @parameterized.expand([
        (LINEAR,),
        (INSUFFICIENT_DATA,)
    ])
    def test_insufficient_data_raises_error(self, input_data):
        """Test that insufficient data raises ValueError."""
        with self.assertRaises(ValueError):
            CubicInterpolation()._fill(input_data, "values")