import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal, assert_series_equal

from time_stream import TimeSeries
from time_stream.infill import (InfillMethod, BSplineInterpolation, CubicInterpolation, QuadraticInterpolation,
                                LinearInterpolation, AkimaInterpolation, PchipInterpolation)
from time_stream.utils import gap_size_count

# Data used through the tests
LINEAR = pl.DataFrame({"values": [1., None, 3., None, 5.]})  # Linear progression
QUADRATIC = pl.DataFrame({"values": [0., None, 4., None, 16., None, 36.]})  # Quadratic data: y = x^2
CUBIC = pl.DataFrame({"values": [0., None, 8., None, 64., None, 216., None, 512.]})  # Cubic data: y = x^3
INSUFFICIENT_DATA = pl.DataFrame({"values": [1., None, None, None, None]}) # Insufficient data
COMPLETE = pl.DataFrame({"values": [1., 2., 3., 4., 5.]}) # No missing data
ALL_MISSING = pl.DataFrame({"values": [None, None, None, None, None]}) # All missing data
VARYING_GAPS = pl.DataFrame({"values": [1., None, 3., 4., 5., 6., None, None, 9., None, None, None, 13.]})
GAP_OF_TWO = pl.DataFrame({"values": [1., None, None, 4., 5.]})
START_GAP = pl.DataFrame({"values": [None, 2., 3., 4., 5., 6.]})
END_GAP = pl.DataFrame({"values": [1., 2., 3., 4., 5., None]})
START_GAP_WITH_MID_GAP = pl.DataFrame({"values": [None, 2., 3., None, 5., 6.]})
END_GAP_WITH_MID_GAP = pl.DataFrame({"values": [1., 2., 3., None, 5., None]})


class TestInfillMethod(unittest.TestCase):
    """Test the base InfillMethod class."""

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
        (COMPLETE, None, None, False),
        (LINEAR, 1, None, True),
        (INSUFFICIENT_DATA, 1, None, False),
        (INSUFFICIENT_DATA, None, None, False),
        (ALL_MISSING, 1, None, False),
        (VARYING_GAPS, 1, None, True),
        (VARYING_GAPS, 2, None, True),
        (VARYING_GAPS, 3, None, True),
        (VARYING_GAPS, 1, (datetime(2025, 1, 1), datetime(2025, 1, 3)), True),
        (VARYING_GAPS, 1, (datetime(2025, 1, 6), datetime(2025, 1, 9)), False),
        (VARYING_GAPS, 2, (datetime(2025, 1, 9), datetime(2025, 1, 11)), False),
        (VARYING_GAPS, None, datetime(2025, 1, 3), True),
        (VARYING_GAPS, None, None, True),
        (VARYING_GAPS, None, None, True),
        (START_GAP, None, None, False),
        (END_GAP, None, None, False),
        (START_GAP_WITH_MID_GAP, None, None, True),
        (END_GAP_WITH_MID_GAP, None, None, True),
    ])
    def test_infill_mask(self, df, max_gap_size, observation_interval, expected):
        """Test whether the infill_mask returns expected results."""
        # Add timestamp column to input df
        df = df.with_columns(pl.Series("timestamp", [datetime(2025, 1, d) for d in range(1, len(df)+1)]))

        # Get the mask
        mask = InfillMethod.infill_mask("timestamp", "values",
                                        max_gap_size=max_gap_size, observation_interval=observation_interval)

        # Apply the mask
        df = gap_size_count(df, "values")
        result = not df.filter(mask).is_empty()
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


class TestApply(unittest.TestCase):
    @staticmethod
    def create_ts(df):
        df = df.with_columns(pl.Series("timestamp", [datetime(2025, 1, d) for d in range(1, len(df) + 1)]))
        ts = TimeSeries(df, "timestamp", "P1D", "P1D")
        return ts

    @parameterized.expand([
        (LinearInterpolation(), LINEAR, [1., 2., 3., 4., 5.]),
        (LinearInterpolation(), QUADRATIC, [0., 2., 4., 10., 16., 26., 36.]),
        (LinearInterpolation(), CUBIC, [0., 4., 8., 36., 64., 140., 216., 364., 512.]),
        (QuadraticInterpolation(), LINEAR, [1., 2., 3., 4., 5.]),
        (QuadraticInterpolation(), QUADRATIC, [0., 1., 4., 9., 16., 25., 36.]),
        (CubicInterpolation(), CUBIC, [0., 1., 8., 27., 64., 125., 216., 343., 512.]),
    ])
    def test_apply(self, interpolator, df, expected):
        """Test that the apply method works as expected with good data."""
        ts = self.create_ts(df)
        result = interpolator.apply(ts, "values")
        expected = self.create_ts(pl.DataFrame({"values": expected}))
        assert_frame_equal(result.df, expected.df, check_column_order=False)

    @parameterized.expand([
        (COMPLETE, None, None),
        (START_GAP, None, None),
        (END_GAP, None, None),
        (GAP_OF_TWO, 1, None),
        (VARYING_GAPS, None, (datetime(2025, 1, 3), datetime(2025, 1, 6))),
        (VARYING_GAPS, 1, (datetime(2025, 1, 6), datetime(2025, 1, 9))),
    ])
    @patch.object(InfillMethod, "_fill")
    def test_apply_nothing_to_infill(self, df, max_gap_size, observation_interval, mock_fill):
        """Test that the apply method works when there is nothing to infill."""
        ts = self.create_ts(df)
        result = LinearInterpolation().apply(
            ts, "values", observation_interval=observation_interval, max_gap_size=max_gap_size
        )

        # The _fill method should not be called at all - the apply method should return early if nothing to infill
        mock_fill.assert_not_called()

        # Double-check the same data is returned
        expected = self.create_ts(df)
        assert_frame_equal(result.df, expected.df, check_column_order=False)

    @parameterized.expand([
        (START_GAP_WITH_MID_GAP, None, None, [None, 2., 3., 4., 5., 6.]),
        (END_GAP_WITH_MID_GAP, None, None, [1., 2., 3., 4., 5., None]),
        (VARYING_GAPS, 2, None, [1., 2., 3., 4., 5., 6., 7., 8., 9., None, None, None, 13.]),
        (VARYING_GAPS, None, datetime(2025, 1, 3), [1., None, 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.]),
        (VARYING_GAPS, 2, datetime(2025, 1, 3), [1., None, 3., 4., 5., 6., 7., 8., 9., None, None, None, 13.]),
    ])
    def test_apply_edge_cases(self, df, max_gap_size, observation_interval, expected):
        """Test that the apply method works when dealing with edge cases"""
        ts = self.create_ts(df)
        result = LinearInterpolation().apply(
            ts, "values", max_gap_size=max_gap_size, observation_interval=observation_interval
        )
        expected = self.create_ts(pl.DataFrame({"values": expected}))
        assert_frame_equal(result.df, expected.df, check_column_order=False)
