from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from time_stream import Period, TimeFrame
from time_stream.exceptions import (
    ColumnNotFoundError,
    InfillInsufficientValuesError,
    RegistryKeyTypeError,
    UnknownRegistryKeyError,
)
from time_stream.infill import (
    AkimaInterpolation,
    AltData,
    AltDataDynamic,
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

TIME_COLUMN = "timestamp"
PERIODICITY = Period.of_days(1)

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


class TestInfillMethod:
    """Test the base InfillMethod class."""

    @pytest.mark.parametrize(
        "get_input,expected",
        [
            ("linear", LinearInterpolation),
            ("cubic", CubicInterpolation),
            ("akima", AkimaInterpolation),
        ],
    )
    def test_get_with_string(self, get_input: str, expected: type[InfillMethod]) -> None:
        """Test QCCheck.get() with string input."""
        infill = InfillMethod.get(get_input)
        assert isinstance(infill, expected)

    @pytest.mark.parametrize(
        "get_input,expected",
        [
            (LinearInterpolation, LinearInterpolation),
            (CubicInterpolation, CubicInterpolation),
            (AkimaInterpolation, AkimaInterpolation),
        ],
    )
    def test_get_with_class(self, get_input: type[InfillMethod], expected: type[InfillMethod]) -> None:
        """Test InfillMethod.get() with class input."""
        infill = InfillMethod.get(get_input)
        assert isinstance(infill, expected)

    @pytest.mark.parametrize(
        "get_input,expected",
        [
            (LinearInterpolation(), LinearInterpolation),
            (CubicInterpolation(), CubicInterpolation),
            (AkimaInterpolation(), AkimaInterpolation),
        ],
    )
    def test_get_with_instance(self, get_input: InfillMethod, expected: type[InfillMethod]) -> None:
        """Test InfillMethod.get() with instance input."""
        infill = InfillMethod.get(get_input)
        assert isinstance(infill, expected)

    @pytest.mark.parametrize("get_input", ["dummy", "RANGE", "123"])
    def test_get_with_invalid_string(self, get_input: str) -> None:
        """Test InfillMethod.get() with invalid string."""
        with pytest.raises(UnknownRegistryKeyError):
            InfillMethod.get(get_input)

    def test_get_with_invalid_class(self) -> None:
        """Test InfillMethod.get() with invalid class."""

        class InvalidClass:
            pass

        with pytest.raises(RegistryKeyTypeError):
            InfillMethod.get(InvalidClass)  # type: ignore[arg-type]  # noqa - expecting type warning

    @pytest.mark.parametrize("get_input", [123, [LinearInterpolation, QuadraticInterpolation], {AkimaInterpolation}])
    def test_get_with_invalid_type(self, get_input: Any) -> None:
        """Test InfillMethod.get() with invalid type."""
        with pytest.raises(RegistryKeyTypeError):
            InfillMethod.get(get_input)


class TestInfillMethodPipeline:
    @pytest.mark.parametrize(
        "df,max_gap_size,observation_interval,expected",
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
        ],
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
        assert result == expected


class TestBSplineInterpolation:
    def test_initialization(self) -> None:
        """Test BSplineInterpolation initialization."""
        # Custom order
        interp = BSplineInterpolation(order=2)
        assert interp.order == 2
        assert interp.min_points_required == 3

        # With scipy kwargs
        interp = BSplineInterpolation(order=1, bc_type="clamped")
        assert interp.scipy_kwargs["bc_type"] == "clamped"


class TestLinearInterpolation:
    @pytest.mark.parametrize(
        "input_data,expected_data",
        [
            (LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (QUADRATIC, [0.0, 2.0, 4.0, 10.0, 16.0, 26.0, 36.0]),
            (CUBIC, [0.0, 4.0, 8.0, 36.0, 64.0, 140.0, 216.0, 364.0, 512.0]),
        ],
    )
    def test_linear_interpolation_known_result(self, input_data: pl.DataFrame, expected_data: list) -> None:
        """Test linear interpolation with known data."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        result = LinearInterpolation()._fill(input_data, "values", ctx)
        expected = pl.Series("values_linear", expected_data)
        assert_series_equal(result["values_linear"], expected)

    @pytest.mark.parametrize(
        "input_data",
        [
            INSUFFICIENT_DATA,
            ALL_MISSING,
        ],
        ids=["1 data points", "0 data points"],
    )
    def test_insufficient_data_raises_error(self, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        with pytest.raises(InfillInsufficientValuesError):
            LinearInterpolation()._fill(input_data, "values", ctx)

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        ctx = InfillCtx(COMPLETE, TIME_COLUMN, PERIODICITY)
        result = LinearInterpolation()._fill(COMPLETE, "values", ctx)
        expected = pl.Series("values_linear", COMPLETE)
        assert_series_equal(result["values_linear"], expected)


class TestQuadraticInterpolation:
    @pytest.mark.parametrize(
        "input_data,expected_data",
        [
            (LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (QUADRATIC, [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
        ],
    )
    def test_quadratic_interpolation_known_result(self, input_data: pl.DataFrame, expected_data: list) -> None:
        """Test quadratic interpolation with known data."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        result = QuadraticInterpolation()._fill(input_data, "values", ctx)

        expected = pl.Series("values_quadratic", expected_data)
        assert_series_equal(result["values_quadratic"], expected)

    @pytest.mark.parametrize("input_data", [INSUFFICIENT_DATA, ALL_MISSING], ids=["1 data points", "0 data points"])
    def test_insufficient_data_raises_error(self, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        with pytest.raises(InfillInsufficientValuesError):
            QuadraticInterpolation()._fill(input_data, "values", ctx)

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        ctx = InfillCtx(COMPLETE, TIME_COLUMN, PERIODICITY)
        result = QuadraticInterpolation()._fill(COMPLETE, "values", ctx)
        expected = pl.Series("values_quadratic", COMPLETE)
        assert_series_equal(result["values_quadratic"], expected)


class TestCubicInterpolation:
    @pytest.mark.parametrize(
        "input_data,expected_data",
        [
            (CUBIC, [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]),
        ],
    )
    def test_cubic_interpolation_known_result(self, input_data: pl.DataFrame, expected_data: list) -> None:
        """Test cubic interpolation with known data."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        result = CubicInterpolation()._fill(input_data, "values", ctx)

        expected = pl.Series("values_cubic", expected_data)
        assert_series_equal(result["values_cubic"], expected)

    @pytest.mark.parametrize(
        "input_data", [LINEAR, INSUFFICIENT_DATA, ALL_MISSING], ids=["3 data points", "1 data points", "0 data points"]
    )
    def test_insufficient_data_raises_error(self, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        with pytest.raises(InfillInsufficientValuesError):
            CubicInterpolation()._fill(input_data, "values", ctx)

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        ctx = InfillCtx(COMPLETE, TIME_COLUMN, PERIODICITY)
        result = CubicInterpolation()._fill(COMPLETE, "values", ctx)
        expected = pl.Series("values_cubic", COMPLETE)
        assert_series_equal(result["values_cubic"], expected)


class TestAkimaInterpolation:
    # Manually calculating the Akima interpolation isn't practical.
    #   Let's assume that SciPy is well tested and the Akima interpolation results are correct and let's just test
    #   behaviours of the interpolation class

    def test_initialization(self) -> None:
        """Test Akima initialization."""
        interp = AkimaInterpolation()
        assert interp.min_points_required == 5
        assert interp.name == "akima"

        # With scipy kwargs
        interp = AkimaInterpolation(extrapolate=True)
        assert interp.scipy_kwargs["extrapolate"]

    def test_akima_interpolation_with_sufficient_data(self) -> None:
        """Test akima interpolation works when there is sufficient data (at least 5 points)."""
        ctx = InfillCtx(CUBIC, TIME_COLUMN, PERIODICITY)
        result = AkimaInterpolation()._fill(CUBIC, "values", ctx)
        assert "values_akima" in result.columns

    @pytest.mark.parametrize(
        "input_data",
        [
            QUADRATIC,
            LINEAR,
            INSUFFICIENT_DATA,
            ALL_MISSING,
        ],
        ids=["4 data points", "3 data points", "1 data points", "0 data points"],
    )
    def test_insufficient_data_raises_error(self, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        with pytest.raises(InfillInsufficientValuesError):
            AkimaInterpolation()._fill(input_data, "values", ctx)

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        ctx = InfillCtx(COMPLETE, TIME_COLUMN, PERIODICITY)
        result = AkimaInterpolation()._fill(COMPLETE, "values", ctx)
        expected = pl.Series("values_akima", COMPLETE)
        assert_series_equal(result["values_akima"], expected)


class TestPchipInterpolation:
    # Manually calculating the Pchip interpolation isn't practical.
    #   Let's assume that SciPy is well tested and the Pchip interpolation results are correct and let's just test
    #   behaviours of the interpolation class

    def test_initialization(self) -> None:
        """Test Akima initialization."""
        interp = PchipInterpolation()
        assert interp.min_points_required == 2
        assert interp.name == "pchip"

        # With scipy kwargs
        interp = PchipInterpolation(extrapolate=True)
        assert interp.scipy_kwargs["extrapolate"]

    def test_pchip_interpolation_with_sufficient_data(self) -> None:
        """Test akima interpolation works when there is sufficient data (at least 2 points)."""
        ctx = InfillCtx(LINEAR, TIME_COLUMN, PERIODICITY)
        result = PchipInterpolation()._fill(LINEAR, "values", ctx)
        assert "values_pchip" in result.columns

    @pytest.mark.parametrize(
        "input_data",
        [INSUFFICIENT_DATA, ALL_MISSING],
        ids=[
            "1 data points",
            "0 data points",
        ],
    )
    def test_insufficient_data_raises_error(self, input_data: pl.DataFrame) -> None:
        """Test that insufficient data raises InfillInsufficientValuesError."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        with pytest.raises(InfillInsufficientValuesError):
            PchipInterpolation()._fill(input_data, "values", ctx)

    @pytest.mark.parametrize(
        "input_data",
        [
            LINEAR,
            QUADRATIC,
            CUBIC,
        ],
    )
    def test_pchip_monotonic_preservation(self, input_data: pl.DataFrame) -> None:
        """Part of the pchip behaviour is that it should preserve local monotonicity if the input data is monotonic."""
        ctx = InfillCtx(input_data, TIME_COLUMN, PERIODICITY)
        result = PchipInterpolation()._fill(input_data, "values", ctx)
        interpolated = result["values_pchip"].to_numpy()

        # Check that result is monotonically increasing
        assert np.all(np.diff(interpolated) > 0)

    def test_complete_data_unchanged(self) -> None:
        """Test that complete data is unchanged."""
        ctx = InfillCtx(COMPLETE, TIME_COLUMN, PERIODICITY)
        result = PchipInterpolation()._fill(COMPLETE, "values", ctx)
        expected = pl.Series("values_pchip", COMPLETE)
        assert_series_equal(result["values_pchip"], expected)


class TestApply:
    @staticmethod
    def create_tf(df: pl.DataFrame) -> TimeFrame:
        df = df.with_columns(pl.Series("timestamp", [datetime(2025, 1, d) for d in range(1, len(df) + 1)]))
        tf = TimeFrame(df, "timestamp", "P1D")
        return tf

    @pytest.mark.parametrize(
        "interpolator,df,expected_values",
        [
            (LinearInterpolation(), LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (LinearInterpolation(), QUADRATIC, [0.0, 2.0, 4.0, 10.0, 16.0, 26.0, 36.0]),
            (LinearInterpolation(), CUBIC, [0.0, 4.0, 8.0, 36.0, 64.0, 140.0, 216.0, 364.0, 512.0]),
            (QuadraticInterpolation(), LINEAR, [1.0, 2.0, 3.0, 4.0, 5.0]),
            (QuadraticInterpolation(), QUADRATIC, [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
            (CubicInterpolation(), CUBIC, [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0]),
        ],
    )
    def test_apply(self, interpolator: InfillMethod, df: pl.DataFrame, expected_values: list) -> None:
        """Test that the apply method works as expected with good data."""
        tf = self.create_tf(df)
        result = interpolator.apply(tf.df, tf.time_name, tf.periodicity, "values")
        meta = {"infill_method": interpolator.name}
        original_nulls = df["values"].is_null().to_list()
        expected_meta = [meta if null else None for null in original_nulls]
        expected_df = self.create_tf(pl.DataFrame({"values": expected_values})).df.with_columns(
            pl.Series("__INFILL_META__", expected_meta)
        )
        assert_frame_equal(result, expected_df, check_column_order=False)

    @pytest.mark.parametrize(
        "df,max_gap_size,observation_interval",
        [
            (COMPLETE, None, None),
            (START_GAP, None, None),
            (END_GAP, None, None),
            (GAP_OF_TWO, 1, None),
            (VARYING_GAPS, None, (datetime(2025, 1, 3), datetime(2025, 1, 6))),
            (VARYING_GAPS, 1, (datetime(2025, 1, 6), datetime(2025, 1, 9))),
        ],
    )
    @patch.object(InfillMethod, "_fill")
    def test_apply_nothing_to_infill(
        self,
        mock_fill: Mock,
        df: pl.DataFrame,
        max_gap_size: int,
        observation_interval: datetime | tuple[datetime, datetime | None] | None,
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

    @pytest.mark.parametrize(
        "df,max_gap_size,observation_interval,expected_values",
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
        ],
    )
    def test_apply_edge_cases(
        self, df: pl.DataFrame, max_gap_size: int, observation_interval: datetime, expected_values: list
    ) -> None:
        """Test that the apply method works when dealing with edge cases"""
        tf = self.create_tf(df)
        result = LinearInterpolation().apply(
            tf.df, tf.time_name, tf.periodicity, "values", observation_interval, max_gap_size
        )
        original_values = df["values"].to_list()
        expected_meta = [
            {"infill_method": "linear"} if orig is None and exp is not None else None
            for orig, exp in zip(original_values, expected_values)
        ]
        expected_df = self.create_tf(pl.DataFrame({"values": expected_values})).df.with_columns(
            pl.Series("__INFILL_META__", expected_meta)
        )
        assert_frame_equal(result, expected_df, check_column_order=False)


class TestAltData:
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 2),
                datetime(2025, 1, 3),
                datetime(2025, 1, 4),
                datetime(2025, 1, 5),
            ],
            "values": [1.0, None, 3.0, None, 5.0],
            "alt_values": [10.0, 20.0, 30.0, 40.0, 50.0],
            "alt_with_missing": [10.0, None, 30.0, 40.0, None],
        }
    )
    tf = TimeFrame(df, "timestamp", "P1D")
    meta = {"infill_method": "alt_data", "alt_dataset_name": "dep_ts"}

    def test_alt_data_infill(self) -> None:
        """Test basic infilling from an alternative column."""
        infiller = AltData(alt_data_column="alt_values")
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_df = self.df.with_columns(
            pl.Series("values", [1.0, 20.0, 3.0, 40.0, 5.0]),
            pl.Series("__INFILL_META__", [None, self.meta, None, self.meta, None]),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_alt_data_infill_with_correction(self) -> None:
        """Test infilling with a correction factor."""
        infiller = AltData(alt_data_column="alt_values", correction_factor=0.1)
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_df = self.df.with_columns(
            pl.Series("values", [1.0, 2.0, 3.0, 4.0, 5.0]),
            pl.Series("__INFILL_META__", [None, self.meta, None, self.meta, None]),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_alt_data_infill_no_missing_data(self) -> None:
        """Test that nothing happens when there is no missing data."""
        df_complete = self.df.with_columns(pl.Series("values", [1.0, 2.0, 3.0, 4.0, 5.0]))
        tf_complete = TimeFrame(df_complete, "timestamp", "P1D")
        infiller = AltData(alt_data_column="alt_values")
        result_df = infiller.apply(tf_complete.df, tf_complete.time_name, tf_complete.periodicity, "values")
        assert_frame_equal(result_df, tf_complete.df, check_column_order=False)

    def test_alt_data_infill_missing_alt_data(self) -> None:
        """Test that missing data in the alternative column is not used for infilling."""
        infiller = AltData(alt_data_column="alt_with_missing")
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_df = self.df.with_columns(
            pl.Series("values", [1.0, None, 3.0, 40.0, 5.0]),
            pl.Series("__INFILL_META__", [None, None, None, self.meta, None]),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_alt_data_infill_missing_alt_data_column_column(self) -> None:
        """Test that an error is raised if the alt_data_column column is missing."""
        infiller = AltData(alt_data_column="non_existent_column")
        with pytest.raises(ColumnNotFoundError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_alt_data_infill_restricting_date_range(self) -> None:
        """Test that only data in the observation_interval is infilled."""
        infiller = AltData(alt_data_column="alt_values")
        result_df = infiller.apply(
            self.tf.df,
            self.tf.time_name,
            self.tf.periodicity,
            "values",
            observation_interval=(datetime(2025, 1, 1), datetime(2025, 1, 2)),
        )
        expected_df = self.df.with_columns(
            pl.Series("values", [1.0, 20.0, 3.0, None, 5.0]),
            pl.Series("__INFILL_META__", [None, self.meta, None, None, None]),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_alt_data_infill_with_alt_data_provided(self) -> None:
        """Test infilling from a provided alternative DataFrame."""
        alt_df = pl.DataFrame(
            {
                "timestamp": self.df["timestamp"],
                "alt_values_df": [11.0, 22.0, 33.0, 44.0, 55.0],
            }
        )
        infiller = AltData(alt_data_column="alt_values_df", alt_df=alt_df)
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_df = self.df.with_columns(
            pl.Series("values", [1.0, 22.0, 3.0, 44.0, 5.0]),
            pl.Series("__INFILL_META__", [None, self.meta, None, self.meta, None]),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_alt_data_infill_with_alt_data_missing_time_column(self) -> None:
        """Test error when provided alt_data is missing the time column."""
        alt_df = pl.DataFrame({"alt_values_df": [11.0, 22.0, 33.0, 44.0, 55.0]})
        infiller = AltData(alt_data_column="alt_values", alt_df=alt_df)
        with pytest.raises(ColumnNotFoundError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_alt_data_infill_with_alt_data_missing_data_column(self) -> None:
        """Test error when provided alt_data is missing the data column."""
        alt_df = pl.DataFrame({"time": self.df["timestamp"]})
        infiller = AltData(alt_data_column="non_existent_column", alt_df=alt_df)
        with pytest.raises(ColumnNotFoundError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_alt_data_infill_with_alt_data_and_column_in_main_df(self) -> None:
        """Test that alt_data is prioritized when column name exists in main df."""
        alt_df = pl.DataFrame(
            {
                "timestamp": self.df["timestamp"],
                "values": [11.0, 22.0, 33.0, 44.0, 55.0],
            }
        )
        infiller = AltData(alt_data_column="values", alt_df=alt_df)

        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_df = self.df.with_columns(
            pl.Series("values", [1.0, 22.0, 3.0, 44.0, 5.0]),
            pl.Series("__INFILL_META__", [None, self.meta, None, self.meta, None]),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)


class TestAltDataDynamic:
    # When test data is structured, eg. increasing or decreasing, the correction_factor can have the same value
    # regardless of the position of the missing data in the values/alt_values.
    # Therefore we must use random data to test that all or only some data has been used.
    # Note: _infill_mask ensures missing values at edges are not infilled, and should remain as None.
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, d) for d in range(1, 13, 1)],
            "values": [7.6, 82.2, 89.6, None, 91.9, 82.6, 90.0, None, 48.4, None, 46.4, None],
            "alt_values": [20.2, 57.5, 96.3, 43.0, 78.4, 61.8, 55.1, 21.8, 100.2, 16.9, 4.2, 17.2],
            "alt_values_some_missing": [20.2, 57.5, 96.3, None, 78.4, None, 55.1, 21.8, 100.2, 16.9, 4.2, 17.2],
            "alt_values_all_missing": [None for _ in range(12)],
        }
    )
    tf = TimeFrame(df, "timestamp", "P1D")

    def test_basic_infill(self) -> None:
        """Test basic infilling from an alternative column. No thresholds.

        Test cases where original dataset has gaps such that, when using alt_values:
        Gap 1: The window around a gap has no missing data.
        Gap 2 and 3: The window around a gap has some missing data.
        Gap 4: A gap is at the edge of the dataset and data remains null.
        """
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P3D")
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        result_df = result_df.with_columns(pl.col("values").round(1))
        expected_df = self.df.with_columns(
            pl.Series("values", [7.6, 82.2, 89.6, 51.7, 91.9, 82.6, 90.0, 26.1, 48.4, 19.6, 46.4, None]),
            pl.Series(
                "__INFILL_META__",
                [
                    None,
                    None,
                    None,
                    {
                        "infill_method": "alt_data_dynamic",
                        "alt_dataset_name": "dep_ts",
                        "timestamps": [
                            datetime(2025, 1, 1, 0, 0),
                            datetime(2025, 1, 2, 0, 0),
                            datetime(2025, 1, 3, 0, 0),
                            datetime(2025, 1, 5, 0, 0),
                            datetime(2025, 1, 6, 0, 0),
                            datetime(2025, 1, 7, 0, 0),
                        ],
                        "correction_factor": 1.2020037909558623,
                    },
                    None,
                    None,
                    None,
                    {
                        "infill_method": "alt_data_dynamic",
                        "alt_dataset_name": "dep_ts",
                        "timestamps": [
                            datetime(2025, 1, 5, 0, 0),
                            datetime(2025, 1, 6, 0, 0),
                            datetime(2025, 1, 7, 0, 0),
                            datetime(2025, 1, 9, 0, 0),
                            datetime(2025, 1, 11, 0, 0),
                        ],
                        "correction_factor": 1.1988655321988657,
                    },
                    None,
                    {
                        "infill_method": "alt_data_dynamic",
                        "alt_dataset_name": "dep_ts",
                        "timestamps": [
                            datetime(2025, 1, 7, 0, 0),
                            datetime(2025, 1, 9, 0, 0),
                            datetime(2025, 1, 11, 0, 0),
                        ],
                        "correction_factor": 1.1586206896551725,
                    },
                    None,
                    None,
                ],
            ),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_basic_infill_some_alt_data_missing(self) -> None:
        """
        Test basic infilling from an alternative column, which is missing some data.

        Test cases where original dataset has gaps such that, when using alt_values_some_missing:
        1. alt_values_some_missing has data missing inside the window around a gap.
        2. alt_values_some_missing has a value missing in the gap.
        """
        infiller = AltDataDynamic(alt_data_column="alt_values_some_missing", window_size="P3D")
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        result_df = result_df.with_columns(pl.col("values").round(1))
        expected_df = self.df.with_columns(
            pl.Series("values", [7.6, 82.2, 89.6, None, 91.9, 82.6, 90.0, 25.4, 48.4, 19.6, 46.4, None]),
            pl.Series(
                "__INFILL_META__",
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    {
                        "infill_method": "alt_data_dynamic",
                        "alt_dataset_name": "dep_ts",
                        "timestamps": [
                            datetime(2025, 1, 5, 0, 0),
                            datetime(2025, 1, 7, 0, 0),
                            datetime(2025, 1, 9, 0, 0),
                            datetime(2025, 1, 11, 0, 0),
                        ],
                        "correction_factor": 1.1630937368642287,
                    },
                    None,
                    {
                        "infill_method": "alt_data_dynamic",
                        "alt_dataset_name": "dep_ts",
                        "timestamps": [
                            datetime(2025, 1, 7, 0, 0),
                            datetime(2025, 1, 9, 0, 0),
                            datetime(2025, 1, 11, 0, 0),
                        ],
                        "correction_factor": 1.1586206896551725,
                    },
                    None,
                    None,
                ],
            ),
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_valid_window_size(self) -> None:
        """Test window size smaller than periodicity raises an error."""
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="PT1H")
        with pytest.raises(ValueError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_window_duration_parser(self) -> None:
        """Test different inputs for window are parsed correctly as a duration,
        and that a window can be entered as an iso string, Period, or timedelta."""
        infiller_iso_hours = AltDataDynamic(alt_data_column="alt_values", window_size="PT1H")
        infiller_period_hours = AltDataDynamic(alt_data_column="alt_values", window_size=Period.of_hours(1))
        infiller_timedelta_hours = AltDataDynamic(alt_data_column="alt_values", window_size=timedelta(hours=1))
        infiller_iso_days = AltDataDynamic(alt_data_column="alt_values", window_size="P1D")
        infiller_period_days = AltDataDynamic(alt_data_column="alt_values", window_size=Period.of_days(1))
        infiller_timedelta_days = AltDataDynamic(alt_data_column="alt_values", window_size=timedelta(days=1))

        ctx = InfillCtx(self.tf.df, self.tf.time_name, Period.of_hours(1))
        assert infiller_iso_hours._window_duration(ctx) == timedelta(hours=1)
        assert infiller_period_hours._window_duration(ctx) == timedelta(hours=1)
        assert infiller_timedelta_hours._window_duration(ctx) == timedelta(hours=1)
        assert infiller_iso_days._window_duration(ctx) == timedelta(days=1)
        assert infiller_period_days._window_duration(ctx) == timedelta(days=1)
        assert infiller_timedelta_days._window_duration(ctx) == timedelta(days=1)

        # Month/year window size cannot be converted to a fixed timedelta
        infiller_month = AltDataDynamic(alt_data_column="alt_values", window_size="P1M")
        with pytest.raises(ValueError, match="Cannot resolve month or year"):
            infiller_month._window_duration(ctx)

        # When periodicity has no fixed timedelta (e.g. monthly data), skip size validation and return early
        ctx_monthly = InfillCtx(self.tf.df, self.tf.time_name, Period.of_iso_duration("P1M"))
        assert infiller_iso_days._window_duration(ctx_monthly) == timedelta(days=1)

    def test_window_is_empty(self) -> None:
        """Test that nothing happens if there is no data that can be used within the window around the gap."""
        infiller = AltDataDynamic(alt_data_column="alt_values_all_missing", window_size="P3D")
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_df = self.df.with_columns(
            [pl.Series("values", self.df["values"]), pl.Series("__INFILL_META__", [None for i in range(12)])]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_valid_window_smaller_than_min_threshold(self) -> None:
        """Test window size smaller than min_threshold raises error."""
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", min_threshold=10)
        with pytest.raises(ValueError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_infill_with_min_threshold(self) -> None:
        """Test infilling from an alternative column, with min_threshold specified, and max_threshold is None."""
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", min_threshold=4)
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        result_df = result_df.with_columns(pl.col("values").round(1))
        expected_df = self.df.with_columns(
            [
                pl.Series("values", [7.6, 82.2, 89.6, 51.7, 91.9, 82.6, 90.0, 26.1, 48.4, None, 46.4, None]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 1, 0, 0),
                                datetime(2025, 1, 2, 0, 0),
                                datetime(2025, 1, 3, 0, 0),
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                            ],
                            "correction_factor": 1.2020037909558623,
                        },
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 1.1988655321988655,
                        },
                        None,
                        None,
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_infill_with_max_threshold(self) -> None:
        """
        Test infilling from an alternative column, with max_threshold specified.
        Gap1: tests max_threshold is not None, symmetric.
        Gap2: tests max_threshold is not None, asymmetric.
        Gap3: tests max_threshold > window_df.height.
        Gap4: Remains None as _infill_mask ensures edges are not infilled.
        """
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", max_threshold=4)
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        result_df = result_df.with_columns(pl.col("values").round(1))
        expected_df = self.df.with_columns(
            [
                pl.Series("values", [7.6, 82.2, 89.6, 50.6, 91.9, 82.6, 90.0, 26.3, 48.4, 19.6, 46.4, None]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 2, 0, 0),
                                datetime(2025, 1, 3, 0, 0),
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                            ],
                            "correction_factor": 1.1778911564625851,
                        },
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 1.2083145051965658,
                        },
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 7, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 1.1586206896551725,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_infill_with_max_threshold_symmetry_preference(self) -> None:
        """
        Test uses same number of datapoints on each side of the gap before using 'closest' data.
        """
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, d) for d in range(1, 11, 1)],
                    "values": [1.0, 2.9, 3.8, 4.7, None, 6.5, 7.4, 8.3, 9.2, 10.1],
                    "alt_values": [10.0, 9.1, 8.2, 7.3, 6.4, 5.5, None, None, 2.8, 1.9],
                }
            ),
            "timestamp",
            "P1D",
        )

        infiller_symmetric = AltDataDynamic(alt_data_column="alt_values", window_size="P4D", max_threshold=4)
        result_symmetric_df = infiller_symmetric.apply(tf.df, tf.time_name, tf.periodicity, "values")
        result_symmetric_df = result_symmetric_df.with_columns(pl.col("values").round(1))
        expected_symmetric_df = tf.df.with_columns(
            [  # Uses sum(3.8,4.7,6.5,9.2)/sum(8.2,7.3,5.5,2.8)
                pl.Series("values", [1.0, 2.9, 3.8, 4.7, 6.5, 6.5, 7.4, 8.3, 9.2, 10.1]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 3, 0, 0),
                                datetime(2025, 1, 4, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                            ],
                            "correction_factor": 1.0168067226890756,
                        },
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_symmetric_df, expected_symmetric_df, check_column_order=False)

        infiller_asymmetric = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", max_threshold=4)
        result_asymmetric_df = infiller_asymmetric.apply(tf.df, tf.time_name, tf.periodicity, "values")
        result_asymmetric_df = result_asymmetric_df.with_columns(pl.col("values").round(1))
        expected_asymmetric_df = tf.df.with_columns(
            [  # Uses sum(2.93.8,4.7,6.5)/sum(9.1,8.2,7.3,5.5)
                pl.Series("values", [1.0, 2.9, 3.8, 4.7, 3.8, 6.5, 7.4, 8.3, 9.2, 10.1]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 2, 0, 0),
                                datetime(2025, 1, 3, 0, 0),
                                datetime(2025, 1, 4, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                            ],
                            "correction_factor": 0.5946843853820597,
                        },
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_asymmetric_df, expected_asymmetric_df, check_column_order=False)

    def test_window_side_parameter(self) -> None:
        """Test infilling from an alternative column, with window_side = "left", "right", "both" and None."""

        # left only
        infiller_left = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", window_side="left")
        result_left_df = infiller_left.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_left_df = self.df.with_columns(
            [
                pl.Series("values", [7.6, 82.2, 89.6, 44.3, 91.9, 82.6, 90.0, 29.5, 48.4, 15.1, 46.4, None]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 1, 0, 0),
                                datetime(2025, 1, 2, 0, 0),
                                datetime(2025, 1, 3, 0, 0),
                            ],
                            "correction_factor": 1.0310344827586206,
                        },
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                            ],
                            "correction_factor": 1.354326676907322,
                        },
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 7, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                            ],
                            "correction_factor": 0.8911783644558918,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )

        # right only
        infiller_right = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", window_side="right")
        result_right_df = infiller_right.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        expected_right_df = self.df.with_columns(
            [
                pl.Series("values", [7.6, 82.2, 89.6, 58.2, 91.9, 82.6, 90.0, 19.8, 48.4, 186.7, 46.4, None]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                            ],
                            "correction_factor": 1.354326676907322,
                        },
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 9, 0, 0),
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 0.9080459770114941,
                        },
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 11.047619047619047,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )

        # Both sides
        infiller_both = AltDataDynamic(alt_data_column="alt_values", window_size="P3D", window_side="both")
        result_both_df = infiller_both.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

        # Default returns both sides when window side not specified
        infiller_none = AltDataDynamic(alt_data_column="alt_values", window_size="P3D")
        result_none_df = infiller_none.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

        expected_both_df = self.df.with_columns(
            [
                pl.Series("values", [7.6, 82.2, 89.6, 51.7, 91.9, 82.6, 90.0, 26.1, 48.4, 19.6, 46.4, None]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 1, 0, 0),
                                datetime(2025, 1, 2, 0, 0),
                                datetime(2025, 1, 3, 0, 0),
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                            ],
                            "correction_factor": 1.2020037909558623,
                        },
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 5, 0, 0),
                                datetime(2025, 1, 6, 0, 0),
                                datetime(2025, 1, 7, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 1.1988655321988655,
                        },
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 7, 0, 0),
                                datetime(2025, 1, 9, 0, 0),
                                datetime(2025, 1, 11, 0, 0),
                            ],
                            "correction_factor": 1.1586206896551725,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )

        assert_frame_equal(
            result_left_df.with_columns(pl.col("values").round(1)), expected_left_df, check_column_order=False
        )
        assert_frame_equal(
            result_right_df.with_columns(pl.col("values").round(1)), expected_right_df, check_column_order=False
        )
        assert_frame_equal(
            result_both_df.with_columns(pl.col("values").round(1)), expected_both_df, check_column_order=False
        )
        assert_frame_equal(
            result_none_df.with_columns(pl.col("values").round(1)), expected_both_df, check_column_order=False
        )

    def test_alt_df_provided(self) -> None:
        """Test AltDataDynamic with alt_df provided as a separate DataFrame."""
        alt_df = pl.DataFrame(
            {
                "timestamp": self.df["timestamp"],
                "external_alt": self.df["alt_values"],
            }
        )
        infiller = AltDataDynamic(alt_data_column="external_alt", window_size="P3D", alt_df=alt_df)
        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        # Should produce same result as using an inline column
        infiller_inline = AltDataDynamic(alt_data_column="alt_values", window_size="P3D")
        expected_df = infiller_inline.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        assert_frame_equal(
            result_df.drop("__INFILL_META__"), expected_df.drop("__INFILL_META__"), check_column_order=False
        )

    def test_valid_thresholds(self) -> None:
        """Test that min_threshold > max_threshold raises ValueError at construction.
        and that max_threshold > 0."""
        with pytest.raises(ValueError):
            AltDataDynamic(alt_data_column="alt_values", window_size="P3D", min_threshold=5, max_threshold=3)

        with pytest.raises(ValueError):
            AltDataDynamic(alt_data_column="alt_values", window_size="P3D", max_threshold=0)

    def test_no_missing_data(self) -> None:
        """Test that nothing happens when there is no missing data."""
        df_complete = self.df.with_columns(pl.Series("values", [i * 1.0 for i in range(12)]))
        tf_complete = TimeFrame(df_complete, "timestamp", "P1D")
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P1D")
        result_df = infiller.apply(tf_complete.df, tf_complete.time_name, tf_complete.periodicity, "values")
        assert_frame_equal(result_df, tf_complete.df, check_column_order=False)

    def test_missing_alt_data_column_column(self) -> None:
        """Test that an error is raised if the alt_data_column column is missing."""
        infiller = AltDataDynamic(alt_data_column="non_existent_column", window_size="P3D")
        with pytest.raises(ColumnNotFoundError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_alt_data_missing_time_column(self) -> None:
        """Test error when provided alt_data is missing the time column."""
        alt_df = pl.DataFrame({"alt_values_df": [i * 1.0 for i in range(12)]})
        infiller = AltDataDynamic(alt_data_column="alt_values", alt_df=alt_df, window_size="P3D")
        with pytest.raises(ColumnNotFoundError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_alt_data_missing_data_column(self) -> None:
        """Test error when provided alt_data is missing the data column."""
        alt_df = pl.DataFrame({"time": self.df["timestamp"]})
        infiller = AltDataDynamic(alt_data_column="non_existent_column", alt_df=alt_df, window_size="P3D")
        with pytest.raises(ColumnNotFoundError):
            infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")

    def test_with_alt_data_and_column_in_main_df(self) -> None:
        """Test that alt_data is prioritized when column name exists in main df."""
        alt_df = pl.DataFrame(
            {
                "timestamp": self.df["timestamp"],
                "alt_values": self.df["alt_values_some_missing"],
            }
        )
        infiller = AltDataDynamic(alt_data_column="alt_values", alt_df=alt_df, window_size="P3D")

        result_df = infiller.apply(self.tf.df, self.tf.time_name, self.tf.periodicity, "values")
        result_df = result_df.with_columns(pl.col("values").round(1))
        expected_df = self.df.with_columns(
            [
                pl.Series("values", [7.6, 82.2, 89.6, None, 91.9, 82.6, 90.0, 25.4, 48.4, 19.6, 46.4, None]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [
                                datetime(2025, 1, 5),
                                datetime(2025, 1, 7),
                                datetime(2025, 1, 9),
                                datetime(2025, 1, 11),
                            ],
                            "correction_factor": 1.1630937368642287,
                        },
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [datetime(2025, 1, 7), datetime(2025, 1, 9), datetime(2025, 1, 11)],
                            "correction_factor": 1.1586206896551725,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_infill_with_max_threshold_one(self) -> None:
        """Test max_threshold=1 uses exactly one data point per gap.

        max_threshold=1 bypasses the symmetric filter (which requires max_threshold >= 2)
        and enters the asymmetric branch. Both sides have 2 rows each within the window,
        so the small-side cap (min(side_count, max_threshold)) ensures only 1 row is kept
        rather than both before-gap rows. In a tie (equal counts each side), the before-gap
        row is preferred.
        """
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, d) for d in range(1, 6)],
                    "values": [10.0, 30.0, None, 80.0, 50.0],
                    "alt_values": [1.0, 2.0, 5.0, 4.0, 8.0],
                }
            ),
            "timestamp",
            "P1D",
        )
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P2D", max_threshold=1)
        result_df = infiller.apply(tf.df, tf.time_name, tf.periodicity, "values")
        result_df = result_df.with_columns(pl.col("values").round(1))
        # Only day 2 is used: CF = 30.0 / 2.0 = 15.0, infilled = 15.0 * 5.0 = 75.0
        expected_df = tf.df.with_columns(
            [
                pl.Series("values", [10.0, 30.0, 75.0, 80.0, 50.0]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [datetime(2025, 1, 2)],
                            "correction_factor": 15.0,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_infill_gap_not_filled_when_alt_sum_is_zero(self) -> None:
        """Test that a gap is not infilled when alt_data sums to zero in its window, while a
        neighboring gap with a non-zero alt_sum is still infilled correctly.
        """
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, d) for d in range(1, 8)],
                    "values": [1.0, None, 3.0, 4.0, None, 6.0, 7.0],
                    "alt_values": [2.0, 0.0, -2.0, 4.0, 5.0, 6.0, 7.0],
                }
            ),
            "timestamp",
            "P1D",
        )
        infiller = AltDataDynamic(alt_data_column="alt_values", window_size="P1D")
        result_df = infiller.apply(tf.df, tf.time_name, tf.periodicity, "values")
        expected_df = tf.df.with_columns(
            [
                pl.Series("values", [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        # Gap 1: alt sums to zero (2.0 + -2.0 = 0), so CF is None
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [datetime(2025, 1, 1), datetime(2025, 1, 3)],
                            "correction_factor": None,
                        },
                        None,
                        None,
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [datetime(2025, 1, 4), datetime(2025, 1, 6)],
                            "correction_factor": 1.0,
                        },
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)

    def test_infill_with_max_threshold_and_one_sided_window(self) -> None:
        """Test that max_threshold limits data points correctly when combined with window_side."""
        tf = TimeFrame(
            pl.DataFrame(
                {
                    "timestamp": [datetime(2025, 1, d) for d in range(1, 7)],
                    "values": [10.0, 20.0, None, 40.0, 50.0, 60.0],
                    "alt_values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                }
            ),
            "timestamp",
            "P1D",
        )
        infiller = AltDataDynamic(
            alt_data_column="alt_values",
            window_size="P3D",
            max_threshold=2,
            min_threshold=2,
            window_side="right",
        )
        result_df = infiller.apply(tf.df, tf.time_name, tf.periodicity, "values")
        expected_df = tf.df.with_columns(
            [
                pl.Series("values", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
                pl.Series(
                    "__INFILL_META__",
                    [
                        None,
                        None,
                        # Right-side only, max_threshold=2: uses Jan 4 and Jan 5 (closest 2 after the gap)
                        # CF = (40.0 + 50.0) / (4.0 + 5.0) = 10.0
                        {
                            "infill_method": "alt_data_dynamic",
                            "alt_dataset_name": "dep_ts",
                            "timestamps": [datetime(2025, 1, 4), datetime(2025, 1, 5)],
                            "correction_factor": 10.0,
                        },
                        None,
                        None,
                        None,
                    ],
                ),
            ]
        )
        assert_frame_equal(result_df, expected_df, check_column_order=False)
