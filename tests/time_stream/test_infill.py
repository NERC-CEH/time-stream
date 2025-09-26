import unittest
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
from parameterized import parameterized
from polars.testing import assert_frame_equal, assert_series_equal

from time_stream import TimeFrame
from time_stream.exceptions import InfillInsufficientValuesError, RegistryKeyTypeError, UnknownRegistryKeyError
from time_stream.infill import (
    AkimaInterpolation,
    BSplineInterpolation,
    CubicInterpolation,
    InfillCtx,
    InfillMethod,
    InfillMethodPipeline,
    LinearInterpolation,
    PchipInterpolation,
    QuadraticInterpolation,
)
from time_stream.utils import gap_size_count

# Data used through the tests
LINEAR = pl.DataFrame({"values": [1.0, None, 3.0, None, 5.0]})  # Linear progression
QUADRATIC = pl.DataFrame({"values": [0.0, None, 4.0, None, 16.0, None, 36.0]})  # Quadratic data: y = x^2
CUBIC = pl.DataFrame({"values": [0.0, None, 8.0, None, 64.0, None, 216.0, None, 512.0]})  # Cubic data: y = x^3
INSUFFICIENT_DATA = pl.DataFrame({"values": [1.0, None, None, None, None]})  # Insufficient data
COMPLETE = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})  # No missing data
ALL_MISSING = pl.DataFrame({"values": [None, None, None, None, None]})  # All missing data
VARYING_GAPS = pl.DataFrame({"values": [1.0, None, 3.0, 4.0, 5.0, 6.0, None, None, 9.0, None, None, None, 13.0]})
GAP_OF_TWO = pl.DataFrame({"values": [1.0, None, None, 4.0, 5.0]})
START_GAP = pl.DataFrame({"values": [None, 2.0, 3.0, 4.0, 5.0, 6.0]})
END_GAP = pl.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0, None]})
START_GAP_WITH_MID_GAP = pl.DataFrame({"values": [None, 2.0, 3.0, None, 5.0, 6.0]})
END_GAP_WITH_MID_GAP = pl.DataFrame({"values": [1.0, 2.0, 3.0, None, 5.0, None]})
ALL_NULL = pl.DataFrame({"values": [None, None, None, None]})


class TestInfillMethod(unittest.TestCase):
    """Test the base InfillMethod class."""

    @parameterized.expand(
        [
            ("linear", LinearInterpolation),
            ("cubic", CubicInterpolation),
            ("akima", AkimaInterpolation),
        ]
    )
    def test_get_with_string(self, get_input: str, expected: type[InfillMethod]) -> None:
        """Test QCCheck.get() with string input."""
        infill = InfillMethod.get(get_input)
        self.assertIsInstance(infill, expected)

    @parameterized.expand(
        [
            (LinearInterpolation, LinearInterpolation),
            (CubicInterpolation, CubicInterpolation),
            (AkimaInterpolation, AkimaInterpolation),
        ]
    )
    def test_get_with_class(self, get_input: type[InfillMethod], expected: type[InfillMethod]) -> None:
        """Test InfillMethod.get() with class input."""
        infill = InfillMethod.get(get_input)
        self.assertIsInstance(infill, expected)

    @parameterized.expand(
        [
            (LinearInterpolation(), LinearInterpolation),
            (CubicInterpolation(), CubicInterpolation),
            (AkimaInterpolation(), AkimaInterpolation),
        ]
    )
    def test_get_with_instance(self, get_input: InfillMethod, expected: type[InfillMethod]) -> None:
        """Test InfillMethod.get() with instance input."""
        infill = InfillMethod.get(get_input)
        self.assertIsInstance(infill, expected)

    @parameterized.expand(["dummy", "RANGE", "123"])
    def test_get_with_invalid_string(self, get_input: str) -> None:
        """Test InfillMethod.get() with invalid string."""
        with self.assertRaises(UnknownRegistryKeyError):
            InfillMethod.get(get_input)

    def test_get_with_invalid_class(self) -> None:
        """Test InfillMethod.get() with invalid class."""

        class InvalidClass:
            pass

        with self.assertRaises(RegistryKeyTypeError):
            InfillMethod.get(InvalidClass)  # noqa - expecting type warning

    @parameterized.expand([(123,), ([LinearInterpolation, QuadraticInterpolation],), ({AkimaInterpolation},)])
    def test_get_with_invalid_type(self, get_input: Any) -> None:
        """Test InfillMethod.get() with invalid type."""
        with self.assertRaises(RegistryKeyTypeError):
            InfillMethod.get(get_input)


class TestInfillMethodPipeline(unittest.TestCase):
    @parameterized.expand(
        [
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
            (ALL_NULL, None, None, False),
        ]
    )
    def test_infill_mask(
        self,
        df: pl.DataFrame,
        max_gap_size: int,
        observation_interval: datetime | tuple[datetime, datetime | None] | None,
        expected: bool,
    ) -> None:
        """Test whether the infill_mask returns expected results."""
        df = df.with_columns(pl.Series("timestamp", [datetime(2025, 1, d) for d in range(1, len(df) + 1)]))

        ctx = InfillCtx(df, "timestamp", Mock())
        pipeline = InfillMethodPipeline(Mock(), ctx, "values", observation_interval, max_gap_size)

        # Get the mask
        mask = pipeline._infill_mask()

        # Apply the mask
        df = gap_size_count(df, "values")
        result = not df.filter(mask).is_empty()
        self.assertEqual(result, expected)


class TestBSplineInterpolation(unittest.TestCase):
    def test_initialization(self) -> None:
        """Test BSplineInterpolation initialization."""
        # Custom order
        interp = BSplineInterpolation(order=2)
        self.assertEqual(interp.order, 2)
        self.assertEqual(interp.min_points_required, 3)

        # With scipy kwargs
        interp = BSplineInterpolation(order=1, bc_type="clamped")
        self.assertEqual(interp.scipy_kwargs["bc_type"], "clamped")


class TestLinearInterpolation(unittest.TestCase):
    @parameterized.expand(
        [
            (LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (QUADRATIC, [0.0, 2.0, 4.0, 10.0, 16.0, 26.0, 36.0]),
            (CUBIC, [0.0, 4.0, 8.0, 36.0, 64.0, 140.0, 216.0, 364.0, 512.0]),
        ]
    )
    def test_linear_interpolation_known_result(self, input_data: pl.DataFrame, expected_data: list) -> None:
        """Test linear interpolation with known data."""
        result = LinearInterpolation()._fill(input_data, "values")
        expected = pl.Series("values_linear", expected_data)
        assert_series_equal(result["values_linear"], expected)

    @parameterized.expand(
        [
            ("1 data points", INSUFFICIENT_DATA),
            ("0 data points", ALL_MISSING),
        ]
    )
    def test_insufficient_data_raises_error(self, _: str, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        with self.assertRaises(InfillInsufficientValuesError):
            LinearInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        result = LinearInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_linear", COMPLETE)
        assert_series_equal(result["values_linear"], expected)


class TestQuadraticInterpolation(unittest.TestCase):
    @parameterized.expand(
        [
            (LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (QUADRATIC, [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
        ]
    )
    def test_quadratic_interpolation_known_result(self, input_data: pl.DataFrame, expected_data: list) -> None:
        """Test quadratic interpolation with known data."""
        result = QuadraticInterpolation()._fill(input_data, "values")

        expected = pl.Series("values_quadratic", expected_data)
        assert_series_equal(result["values_quadratic"], expected)

    @parameterized.expand([("1 data points", INSUFFICIENT_DATA), ("0 data points", ALL_MISSING)])
    def test_insufficient_data_raises_error(self, _: str, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        with self.assertRaises(InfillInsufficientValuesError):
            QuadraticInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        result = QuadraticInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_quadratic", COMPLETE)
        assert_series_equal(result["values_quadratic"], expected)


class TestCubicInterpolation(unittest.TestCase):
    @parameterized.expand(
        [
            (CUBIC, [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]),
        ]
    )
    def test_cubic_interpolation_known_result(self, input_data: pl.DataFrame, expected_data: list) -> None:
        """Test cubic interpolation with known data."""
        result = CubicInterpolation()._fill(input_data, "values")

        expected = pl.Series("values_cubic", expected_data)
        assert_series_equal(result["values_cubic"], expected)

    @parameterized.expand(
        [("3 data points", LINEAR), ("1 data points", INSUFFICIENT_DATA), ("0 data points", ALL_MISSING)]
    )
    def test_insufficient_data_raises_error(self, _: str, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        with self.assertRaises(InfillInsufficientValuesError):
            CubicInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        result = CubicInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_cubic", COMPLETE)
        assert_series_equal(result["values_cubic"], expected)


class TestAkimaInterpolation(unittest.TestCase):
    # Manually calculating the Akima interpolation isn't practical.
    #   Let's assume that SciPy is well tested and the Akima interpolation results are correct and let's just test
    #   behaviours of the interpolation class

    def test_initialization(self) -> None:
        """Test Akima initialization."""
        interp = AkimaInterpolation()
        self.assertEqual(interp.min_points_required, 5)
        self.assertEqual(interp.name, "akima")

        # With scipy kwargs
        interp = AkimaInterpolation(extrapolate=True)
        self.assertEqual(interp.scipy_kwargs["extrapolate"], True)

    def test_akima_interpolation_with_sufficient_data(self) -> None:
        """Test akima interpolation works when there is sufficient data (at least 5 points)."""
        result = AkimaInterpolation()._fill(CUBIC, "values")
        self.assertIn("values_akima", result.columns)

    @parameterized.expand(
        [
            ("4 data points", QUADRATIC),
            ("3 data points", LINEAR),
            ("1 data points", INSUFFICIENT_DATA),
            ("0 data points", ALL_MISSING),
        ]
    )
    def test_insufficient_data_raises_error(self, _: str, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        with self.assertRaises(InfillInsufficientValuesError):
            AkimaInterpolation()._fill(input_data, "values")

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        result = AkimaInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_akima", COMPLETE)
        assert_series_equal(result["values_akima"], expected)


class TestPchipInterpolation(unittest.TestCase):
    # Manually calculating the Pchip interpolation isn't practical.
    #   Let's assume that SciPy is well tested and the Pchip interpolation results are correct and let's just test
    #   behaviours of the interpolation class

    def test_initialization(self) -> None:
        """Test Akima initialization."""
        interp = PchipInterpolation()
        self.assertEqual(interp.min_points_required, 2)
        self.assertEqual(interp.name, "pchip")

        # With scipy kwargs
        interp = PchipInterpolation(extrapolate=True)
        self.assertEqual(interp.scipy_kwargs["extrapolate"], True)

    def test_pchip_interpolation_with_sufficient_data(self) -> None:
        """Test akima interpolation works when there is sufficient data (at least 2 points)."""
        result = PchipInterpolation()._fill(LINEAR, "values")
        self.assertIn("values_pchip", result.columns)

    @parameterized.expand([("1 data points", INSUFFICIENT_DATA), ("0 data points", ALL_MISSING)])
    def test_insufficient_data_raises_error(self, _: str, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        with self.assertRaises(InfillInsufficientValuesError):
            PchipInterpolation()._fill(input_data, "values")

    @parameterized.expand(
        [
            (LINEAR,),
            (QUADRATIC,),
            (CUBIC,),
        ]
    )
    def test_pchip_monotonic_preservation(self, input_data: pl.DataFrame) -> None:
        """Part of the pchip behaviour is that it should preserve local monotonicity if the input data is monotonic."""
        result = PchipInterpolation()._fill(input_data, "values")
        interpolated = result["values_pchip"].to_numpy()

        # Check that result is monotonically increasing
        self.assertTrue(np.all(np.diff(interpolated) > 0))

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        result = PchipInterpolation()._fill(COMPLETE, "values")
        expected = pl.Series("values_pchip", COMPLETE)
        assert_series_equal(result["values_pchip"], expected)


class TestApply(unittest.TestCase):
    @staticmethod
    def create_tf(df: pl.DataFrame) -> TimeFrame:
        df = df.with_columns(pl.Series("timestamp", [datetime(2025, 1, d) for d in range(1, len(df) + 1)]))
        tf = TimeFrame(df, "timestamp", "P1D", "P1D")
        return tf

    @parameterized.expand(
        [
            (LinearInterpolation(), LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (LinearInterpolation(), QUADRATIC, [0.0, 2.0, 4.0, 10.0, 16.0, 26.0, 36.0]),
            (LinearInterpolation(), CUBIC, [0.0, 4.0, 8.0, 36.0, 64.0, 140.0, 216.0, 364.0, 512.0]),
            (QuadraticInterpolation(), LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (QuadraticInterpolation(), QUADRATIC, [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
            (CubicInterpolation(), CUBIC, [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]),
        ]
    )
    def test_apply(self, interpolator: InfillMethod, df: pl.DataFrame, expected: list) -> None:
        """Test that the apply method works as expected with good data."""
        tf = self.create_tf(df)
        result = interpolator.apply(tf.df, tf.time_name, tf.periodicity, "values")
        expected = self.create_tf(pl.DataFrame({"values": expected}))
        assert_frame_equal(result, expected.df, check_column_order=False)

    @parameterized.expand(
        [
            (COMPLETE, None, None),
            (START_GAP, None, None),
            (END_GAP, None, None),
            (GAP_OF_TWO, 1, None),
            (VARYING_GAPS, None, (datetime(2025, 1, 3), datetime(2025, 1, 6))),
            (VARYING_GAPS, 1, (datetime(2025, 1, 6), datetime(2025, 1, 9))),
        ]
    )
    @patch.object(InfillMethod, "_fill")
    def test_apply_nothing_to_infill(
        self,
        df: pl.DataFrame,
        max_gap_size: int,
        observation_interval: datetime | tuple[datetime, datetime | None] | None,
        mock_fill: Mock,
    ) -> None:
        """Test that the apply method works when there is nothing to infill."""
        tf = self.create_tf(df)
        result = LinearInterpolation().apply(
            tf.df, tf.time_name, tf.periodicity, "values", observation_interval, max_gap_size
        )

        # The _fill method should not be called at all - the apply method should return early if nothing to infill
        mock_fill.assert_not_called()

        # Double-check the same data is returned
        expected = self.create_tf(df)
        assert_frame_equal(result, expected.df, check_column_order=False)

    @parameterized.expand(
        [
            (START_GAP_WITH_MID_GAP, None, None, [None, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (END_GAP_WITH_MID_GAP, None, None, [1.0, 2.0, 3.0, 4.0, 5.0, None]),
            (VARYING_GAPS, 2, None, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, None, None, None, 13.0]),
            (
                VARYING_GAPS,
                None,
                datetime(2025, 1, 3),
                [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
            ),
            (
                VARYING_GAPS,
                2,
                datetime(2025, 1, 3),
                [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, None, None, None, 13.0],
            ),
        ]
    )
    def test_apply_edge_cases(
        self, df: pl.DataFrame, max_gap_size: int, observation_interval: datetime, expected: list
    ) -> None:
        """Test that the apply method works when dealing with edge cases"""
        tf = self.create_tf(df)
        result = LinearInterpolation().apply(
            tf.df, tf.time_name, tf.periodicity, "values", observation_interval, max_gap_size
        )
        expected = self.create_tf(pl.DataFrame({"values": expected}))
        assert_frame_equal(result, expected.df, check_column_order=False)
