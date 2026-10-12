import sys
from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from isoperiod import Period
from matplotlib.collections import Collection, PolyCollection
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal

from time_stream.base import TimeFrame
from time_stream.exceptions import BitwiseFlagUnknownError, ColumnNotFoundError, ColumnTypeError
from time_stream.plotting import (
    FLAG_NONE,
    LOWER_FILL,
    MISSING_FILL,
    SERIES_COLORS,
    UPPER_FILL,
    _colors,
    _flag_colors,
    _flag_level,
    _flag_levels,
    _intervals,
    _panels,
    _period_label,
    _plot_style,
    _runs,
    _selected,
    _step_xy,
    _time_of_year_statistics,
    _window,
    _y_label,
    _y_limits,
)

matplotlib.use("Agg")

# Four days of data, on 1-4 January in 2023 and 2024
TWO_YEARS = pl.DataFrame(
    {
        "time": [datetime(y, 1, d) for y in (2023, 2024) for d in range(1, 5)],
        "flow": [1.0, 2.0, 3.0, 4.0, 3.0, None, 1.0, 6.0],
        "level": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "site": ["a"] * 8,
    }
)

FIVE_DAYS = pl.DataFrame(
    {
        "time": [datetime(2024, 1, d) for d in range(1, 6)],
        "flow": [1.0, None, 3.0, 4.0, 5.0],
    }
)


def _num(*times: datetime) -> np.ndarray:
    """Matplotlib date numbers of the given datetimes."""
    return np.asarray(mdates.date2num(list(times)), dtype=float)


def _face(collection: Collection) -> str:
    """The face colour of a collection, as hex."""
    return to_hex(np.asarray(collection.get_facecolor())[0])


def _two_years() -> TimeFrame:
    """The two-year daily TimeFrame."""
    return TimeFrame(TWO_YEARS, "time", resolution="P1D")


def _flagged() -> TimeFrame:
    """The five-day TimeFrame with a bitwise flag column."""
    tf = TimeFrame(FIVE_DAYS, "time", resolution="P1D")
    tf.register_flag_system("QC", {"CORRECTED": 1, "ESTIMATED": 2, "SUSPECT": 4})
    tf.init_flag_column("QC", "flow_flags")
    tf.add_flag("flow_flags", "CORRECTED", pl.col("time") <= datetime(2024, 1, 2))
    tf.add_flag("flow_flags", "SUSPECT", pl.col("time") == datetime(2024, 1, 2))
    tf.add_flag("flow_flags", "ESTIMATED", pl.col("time") == datetime(2024, 1, 4))
    return tf


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    """Close every figure a test opens."""
    yield
    plt.close("all")


class TestPanels:
    @pytest.mark.parametrize(
        "columns,subplots,expected",
        [
            (None, False, [["flow", "level"]]),
            ("flow", False, [["flow"]]),
            (["level", "flow"], False, [["level", "flow"]]),
            ([["flow"], ["level"]], False, [["flow"], ["level"]]),
            (["flow", ["level", "flow"]], False, [["flow"], ["level", "flow"]]),
            (["flow", "level"], True, [["flow"], ["level"]]),
            ([["flow", "level"]], True, [["flow"], ["level"]]),
        ],
        ids=["default numeric", "single", "list", "nested", "mixed", "subplots", "nested subplots"],
    )
    def test_panels(self, columns: Any, subplots: bool, expected: list[list[str]]) -> None:
        """Columns are grouped into panels."""
        assert _panels(_two_years(), columns, subplots) == expected

    @pytest.mark.parametrize("columns", ["missing", [["flow"], []], []], ids=["unknown", "empty panel", "none"])
    def test_no_or_unknown_columns(self, columns: Any) -> None:
        """Unknown columns, and nothing to plot, raise."""
        with pytest.raises(ColumnNotFoundError):
            _panels(_two_years(), columns, False)

    @pytest.mark.parametrize("column", ["site", "time"], ids=["string", "time"])
    def test_not_numeric(self, column: str) -> None:
        """Non-numeric columns raise."""
        with pytest.raises(ColumnTypeError):
            _panels(_two_years(), column, False)


class TestPlotStyle:
    @pytest.mark.parametrize(
        "plot_style,metadata,expected",
        [
            (None, {}, "step"),
            (None, {"plot_style": "bar"}, "bar"),
            ("line", {"plot_style": "bar"}, "line"),
            ({"flow": "scatter"}, {"plot_style": "bar"}, "scatter"),
            ({"level": "scatter"}, {"plot_style": "bar"}, "bar"),
        ],
        ids=["default", "metadata", "argument", "mapping", "mapping other column"],
    )
    def test_precedence(self, plot_style: Any, metadata: dict, expected: str) -> None:
        """The argument wins over the metadata, which wins over the default."""
        tf = _two_years()
        tf.column_metadata["flow"] = metadata
        assert _plot_style(tf, "flow", plot_style) == expected

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({"resolution": "P1D", "time_anchor": "start"}, "step"),
            ({"resolution": "P1D", "time_anchor": "end"}, "step"),
            ({"resolution": "P1D", "time_anchor": "point"}, "line"),
            ({"resolution": "P1D", "periodicity": "P1Y"}, "scatter"),
        ],
        ids=["start", "end", "point", "sparse"],
    )
    def test_default(self, kwargs: dict, expected: str) -> None:
        """The default follows the temporal properties."""
        df = pl.DataFrame({"time": [datetime(2023, 1, 1), datetime(2024, 1, 1)], "flow": [1.0, 2.0]})
        tf = TimeFrame(df, "time", **kwargs)
        assert _plot_style(tf, "flow", None) == expected

    def test_invalid(self) -> None:
        """An unknown style raises."""
        with pytest.raises(ValueError, match="Invalid plot_style"):
            _plot_style(_two_years(), "flow", "pie")  # type: ignore[arg-type]  # testing an invalid value


class TestSelected:
    @pytest.mark.parametrize(
        "value,expected",
        [(True, ["flow", "level"]), (False, []), ("level", ["level"]), (["level", "flow"], ["level", "flow"])],
        ids=["all", "none", "single", "list"],
    )
    def test_selected(self, value: Any, expected: list[str]) -> None:
        """Options select all, none or the named plotted columns."""
        assert _selected(value, ["flow", "level"], "envelope") == expected

    def test_not_plotted(self) -> None:
        """Selecting a column that is not plotted raises."""
        with pytest.raises(ColumnNotFoundError, match="'envelope' columns are not being plotted"):
            _selected("rain", ["flow"], "envelope")


class TestIntervals:
    TIMES = pl.Series([datetime(2024, 1, 1), datetime(2024, 1, 2)])

    @pytest.mark.parametrize(
        "time_anchor,expected_starts,expected_ends",
        [
            ("start", [(2024, 1, 1), (2024, 1, 2)], [(2024, 1, 2), (2024, 1, 3)]),
            ("end", [(2023, 12, 31), (2024, 1, 1)], [(2024, 1, 1), (2024, 1, 2)]),
            ("point", [(2024, 1, 1), (2024, 1, 2)], [(2024, 1, 2), (2024, 1, 3)]),
        ],
        ids=["start", "end", "point"],
    )
    def test_intervals(self, time_anchor: Any, expected_starts: list, expected_ends: list) -> None:
        """Each value is valid over the interval its time anchor gives."""
        times, starts, ends = _intervals(self.TIMES, Period.of_days(1), time_anchor)
        assert_array_equal(times, _num(datetime(2024, 1, 1), datetime(2024, 1, 2)))
        assert_array_equal(starts, _num(*(datetime(*d) for d in expected_starts)))
        assert_array_equal(ends, _num(*(datetime(*d) for d in expected_ends)))

    def test_calendar_resolution(self) -> None:
        """Monthly intervals follow the calendar."""
        _, starts, ends = _intervals(
            pl.Series([datetime(2024, 1, 1), datetime(2024, 2, 1)]), Period.of_months(1), "start"
        )
        assert_array_equal(ends - starts, [31.0, 29.0])

    def test_date_column(self) -> None:
        """Date columns are converted to datetimes."""
        _, starts, ends = _intervals(pl.Series([date(2024, 1, 1)]), Period.of_days(1), "start")
        assert_array_equal(starts, _num(datetime(2024, 1, 1)))
        assert_array_equal(ends, _num(datetime(2024, 1, 2)))


class TestStepXY:
    def test_contiguous(self) -> None:
        """Contiguous intervals join into one outline."""
        x, y = _step_xy(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.array([5.0, 6.0]))
        assert_array_equal(x, [0.0, 1.0, 1.0, 2.0])
        assert_array_equal(y, [5.0, 5.0, 6.0, 6.0])

    def test_gap(self) -> None:
        """Intervals that do not meet are separated by a break."""
        x, y = _step_xy(np.array([0.0, 2.0]), np.array([1.0, 3.0]), np.array([5.0, 6.0]))
        assert_array_equal(x, [0.0, 1.0, 1.0, 2.0, 3.0])
        assert_array_equal(y, [5.0, 5.0, np.nan, 6.0, 6.0])


class TestRuns:
    @pytest.mark.parametrize(
        "starts,ends,keys,expected",
        [
            ([0, 1, 2, 3], [1, 2, 3, 4], [1, 1, 2, 2], ([1, 2], [0, 2], [2, 4])),
            ([0, 1, 5, 6], [1, 2, 6, 7], [1, 1, 1, 1], ([1, 1], [0, 5], [2, 7])),
            ([0], [1], [3], ([3], [0], [1])),
        ],
        ids=["key change", "gap", "single"],
    )
    def test_runs(self, starts: list, ends: list, keys: list, expected: tuple) -> None:
        """Consecutive meeting intervals with the same key merge."""
        run_keys, run_starts, run_ends = _runs(np.array(starts, float), np.array(ends, float), np.array(keys))
        assert_array_equal(run_keys, expected[0])
        assert_array_equal(run_starts, expected[1])
        assert_array_equal(run_ends, expected[2])


class TestWindow:
    def test_filters_and_pads(self) -> None:
        """The window is filtered to the given times, with missing time steps padded with nulls."""
        df = FIVE_DAYS.filter(pl.col("time") != datetime(2024, 1, 3))
        tf = TimeFrame(df, "time", resolution="P1D")
        actual = _window(tf, ["flow"], datetime(2024, 1, 2), datetime(2024, 1, 4))
        expected = pl.DataFrame({"time": [datetime(2024, 1, d) for d in (2, 3, 4)], "flow": [None, None, 4.0]})
        assert_frame_equal(actual, expected)

    def test_pads_beyond_data(self) -> None:
        """A window larger than the data is padded to its ends."""
        tf = TimeFrame(FIVE_DAYS, "time", resolution="P1D")
        actual = _window(tf, ["flow"], datetime(2024, 1, 4), datetime(2024, 1, 7))
        expected = pl.DataFrame({"time": [datetime(2024, 1, d) for d in (4, 5, 6, 7)], "flow": [4.0, 5.0, None, None]})
        assert_frame_equal(actual, expected)

    def test_irregular_not_padded(self) -> None:
        """Data without a resolution is not padded."""
        df = pl.DataFrame({"time": [datetime(2024, 1, 1), datetime(2024, 1, 5)], "flow": [1.0, 2.0]})
        tf = TimeFrame(df, "time")
        assert_frame_equal(_window(tf, ["flow"], None, None), df)

    def test_empty(self) -> None:
        """Nothing in the window raises."""
        df = pl.DataFrame({"time": [datetime(2024, 1, 1), datetime(2024, 1, 5)], "flow": [1.0, 2.0]})
        tf = TimeFrame(df, "time")
        with pytest.raises(ValueError, match="No data to plot"):
            _window(tf, ["flow"], datetime(2024, 1, 2), datetime(2024, 1, 3))


class TestTimeOfYearStatistics:
    def test_statistics(self) -> None:
        """Each row gets the statistics of its day of the year, over all the data."""
        tf = _two_years()
        window = _window(tf, ["flow"], datetime(2024, 1, 1), datetime(2024, 1, 4))
        actual, record = _time_of_year_statistics(
            tf, window, ["flow"], ["flow"], datetime(2024, 1, 1), datetime(2024, 1, 4), None
        )
        expected = pl.DataFrame(
            {
                "min_flow": [1.0, 2.0, 1.0, 4.0],
                "max_flow": [3.0, 2.0, 3.0, 6.0],
                "mean_flow": [2.0, 2.0, 2.0, 5.0],
            }
        )
        assert_frame_equal(actual, expected)
        assert record == ("January 2023", "January 2024")

    def test_reference_interval(self) -> None:
        """Only data in the reference interval is used."""
        tf = _two_years()
        window = _window(tf, ["flow"], datetime(2024, 1, 1), datetime(2024, 1, 2))
        actual, record = _time_of_year_statistics(
            tf,
            window,
            ["flow"],
            ["flow"],
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            (None, datetime(2023, 12, 31)),
        )
        expected = pl.DataFrame({"min_flow": [1.0, 2.0], "max_flow": [1.0, 2.0], "mean_flow": [1.0, 2.0]})
        assert_frame_equal(actual, expected)
        assert record == ("January 2023", "January 2023")

    def test_missing_time_steps_in_window(self) -> None:
        """Missing time steps in the window get the statistics of their time of year, or nulls if it has no data."""
        tf = TimeFrame(TWO_YEARS.filter(pl.col("time") != datetime(2024, 1, 3)), "time", resolution="P1D")
        window = _window(tf, ["flow"], datetime(2024, 1, 3), datetime(2024, 1, 5))
        actual, _ = _time_of_year_statistics(tf, window, ["flow"], [], datetime(2024, 1, 3), datetime(2024, 1, 5), None)
        expected = pl.DataFrame({"min_flow": [3.0, 4.0, None], "max_flow": [3.0, 6.0, None]})
        assert_frame_equal(actual, expected)

    def test_sub_daily(self) -> None:
        """Sub-daily data is matched by time of day as well."""
        df = pl.DataFrame(
            {
                "time": [datetime(y, 1, 1, h) for y in (2023, 2024) for h in (0, 12)],
                "flow": [1.0, 10.0, 3.0, 30.0],
            }
        )
        tf = TimeFrame(df, "time", resolution="PT12H")
        window = _window(tf, ["flow"], datetime(2024, 1, 1), None)
        actual, _ = _time_of_year_statistics(tf, window, [], ["flow"], datetime(2024, 1, 1), None, None)
        expected = pl.DataFrame({"mean_flow": [2.0, 20.0]})
        assert_frame_equal(actual, expected)

    def test_no_columns(self) -> None:
        """Without columns there is nothing to calculate."""
        tf = _two_years()
        actual, record = _time_of_year_statistics(tf, tf.df, [], [], None, None, None)
        assert actual.is_empty()
        assert record is None


class TestFlags:
    def test_levels_default(self) -> None:
        """A flag column name shows every flag in its system's order."""
        assert _flag_levels(_flagged(), "flow_flags") == {"flow_flags": ["CORRECTED", "ESTIMATED", "SUSPECT"]}

    def test_levels_mapping(self) -> None:
        """A mapping shows the given flags, by name or value, in the given order."""
        assert _flag_levels(_flagged(), {"flow_flags": [4, "CORRECTED"]}) == {"flow_flags": ["SUSPECT", "CORRECTED"]}

    def test_levels_none(self) -> None:
        """No flags gives no levels."""
        assert _flag_levels(_flagged(), None) == {}

    def test_levels_empty(self) -> None:
        """A flag column given no flags raises."""
        with pytest.raises(ValueError, match="No flags given"):
            _flag_levels(_flagged(), {"flow_flags": []})

    def test_levels_unknown_flag(self) -> None:
        """An unknown flag raises."""
        with pytest.raises(BitwiseFlagUnknownError):
            _flag_levels(_flagged(), {"flow_flags": ["MISSING"]})

    def test_levels_not_flag_column(self) -> None:
        """A column that is not a flag column raises."""
        with pytest.raises(ColumnNotFoundError):
            _flag_levels(_flagged(), "flow")

    @pytest.mark.parametrize(
        "flags,expected",
        [
            (["CORRECTED", "ESTIMATED", "SUSPECT"], [1, 3, 0, 2, 0]),
            (["SUSPECT", "CORRECTED"], [2, 2, 0, 0, 0]),
            (["ESTIMATED"], [0, 0, 0, 1, 0]),
        ],
        ids=["all", "reordered", "one"],
    )
    def test_level(self, flags: list[str], expected: list[int]) -> None:
        """Each row gets its most severe flag."""
        tf = _flagged()
        assert_array_equal(_flag_level(tf.df, tf.get_flag_column("flow_flags"), flags), expected)

    def test_level_decoded(self) -> None:
        """Decoded flag columns work too."""
        tf = _flagged().decode_flag_column("flow_flags")
        levels = _flag_level(tf.df, tf.get_flag_column("flow_flags"), ["CORRECTED", "ESTIMATED", "SUSPECT"])
        assert_array_equal(levels, [1, 3, 0, 2, 0])

    @pytest.mark.parametrize("count", [1, 2, 5], ids=["one", "two", "five"])
    def test_colors_darken(self, count: int) -> None:
        """Flag colours get darker with severity."""
        brightness = [sum(color) for color in _flag_colors(count)]
        assert len(brightness) == count
        assert brightness == sorted(brightness, reverse=True)


class TestLabels:
    @pytest.mark.parametrize(
        "metadata,panel,expected",
        [
            ({}, ["flow"], "flow"),
            ({"flow": {"label": "Flow"}}, ["flow"], "Flow"),
            ({"flow": {"label": "Flow", "units": "m³/s"}}, ["flow"], "Flow (m³/s)"),
            ({"flow": {"units": "m"}, "level": {"units": "m"}}, ["flow", "level"], "m"),
            ({"flow": {"units": "m³/s"}, "level": {"units": "m"}}, ["flow", "level"], ""),
            ({}, ["flow", "level"], ""),
        ],
        ids=["name", "label", "label and units", "shared units", "different units", "no units"],
    )
    def test_y_label(self, metadata: dict, panel: list[str], expected: str) -> None:
        """The y axis label comes from the label and units metadata."""
        tf = _two_years()
        tf.column_metadata = metadata
        assert _y_label(tf, panel) == expected

    @pytest.mark.parametrize(
        "first,last,expected",
        [(2025, 2025, "2025"), (2025, 2026, "2025/2026"), (2013, 2026, "2013-2026")],
        ids=["one year", "two years", "many years"],
    )
    def test_period_label(self, first: int, last: int, expected: str) -> None:
        """The period label names the years in the window."""
        assert _period_label(first, last) == expected


class TestColors:
    def test_metadata_and_presets(self) -> None:
        """Metadata colours are kept, and other columns take the next unused TSP colour."""
        tf = _two_years()
        tf.column_metadata["flow"] = {"color": SERIES_COLORS[0]}
        assert _colors(tf, ["flow", "level"]) == {"flow": SERIES_COLORS[0], "level": SERIES_COLORS[1]}

    def test_cycles(self) -> None:
        """More columns than colours reuse colours."""
        df = pl.DataFrame({"time": [datetime(2024, 1, 1)], **{f"c{i}": [1.0] for i in range(12)}})
        tf = TimeFrame(df, "time", resolution="P1D")
        assert len(_colors(tf, df.columns[1:])) == 12


class TestYLimits:
    @pytest.mark.parametrize(
        "arrays,log_y,ylim,expected",
        [
            ([np.array([1.0, 55.2])], False, None, (0.0, 60.0)),
            ([np.array([-3.3, 17.1])], False, None, (-5.0, 20.0)),
            ([np.array([1.0, np.nan]), np.array([7.5])], False, None, (0.0, 8.0)),
            ([np.array([41.0, 47.0])], False, None, (40.0, 48.0)),
            ([np.array([5.0, 5.0])], False, None, (4.4, 5.6)),
            ([np.array([np.nan])], False, None, (0.0, 1.0)),
            ([np.array([1.0, 55.2])], False, (None, 100.0), (0.0, 100.0)),
            ([np.array([1.0, 55.2])], False, (-1.0, None), (-1.0, 60.0)),
            ([np.array([0.0, 2.0, 100.0])], True, None, (1.6, 125.0)),
            ([np.array([0.0])], True, None, (1.0, 10.0)),
        ],
        ids=[
            "round",
            "negative",
            "several arrays",
            "far from zero",
            "constant",
            "all missing",
            "user top",
            "user bottom",
            "log",
            "log no positives",
        ],
    )
    def test_y_limits(self, arrays: list[np.ndarray], log_y: bool, ylim: Any, expected: tuple) -> None:
        """Limits cover the data on round numbers, from zero if the data is close to it."""
        assert _y_limits(arrays, log_y, ylim) == pytest.approx(expected)


class TestPlot:
    """Test the figures drawn by TimeFrame.plot."""

    @staticmethod
    def setup_tf() -> TimeFrame:
        tf = _two_years()
        tf.metadata = {"title": "Site A"}
        tf.column_metadata = {"flow": {"label": "Flow", "units": "m³/s"}, "level": {"label": "Level", "units": "m"}}
        return tf

    @pytest.mark.parametrize(
        "kwargs,expected_axes",
        [
            ({}, 1),
            ({"subplots": True}, 2),
            ({"columns": [["flow"], ["level"]]}, 2),
        ],
        ids=["one panel", "subplots", "nested"],
    )
    def test_panels(self, kwargs: dict, expected_axes: int) -> None:
        """A panel is drawn for each group of columns."""
        fig = self.setup_tf().plot(**kwargs)
        assert len(fig.axes) == expected_axes

    def test_titles_and_labels(self) -> None:
        """The title, period label and y label come from the metadata."""
        fig = self.setup_tf().plot("flow")
        ax = fig.axes[0]
        assert ax.get_title(loc="left") == "Site A"
        assert ax.get_title(loc="right") == "2023/2024"
        assert ax.get_ylabel() == "Flow (m³/s)"

    def test_title_argument(self) -> None:
        """The title argument overrides the metadata."""
        fig = self.setup_tf().plot("flow", title="Other")
        assert fig.axes[0].get_title(loc="left") == "Other"

    def test_window(self) -> None:
        """The x axis covers the validity intervals of the values in the window."""
        fig = self.setup_tf().plot("flow", datetime(2024, 1, 2), datetime(2024, 1, 3))
        assert fig.axes[0].get_xlim() == tuple(_num(datetime(2024, 1, 2), datetime(2024, 1, 4)))
        assert fig.axes[0].get_title(loc="right") == "2024"

    def test_step_values(self) -> None:
        """Values are drawn as steps across their interval."""
        fig = self.setup_tf().plot("level", datetime(2024, 1, 1), datetime(2024, 1, 2))
        x, y = fig.axes[0].lines[0].get_data()
        assert_array_equal(x, _num(*(datetime(2024, 1, d) for d in (1, 2, 2, 3))))
        assert_array_equal(y, [14.0, 14.0, 15.0, 15.0])

    @pytest.mark.parametrize("plot_style,marker", [("line", "none"), ("scatter", "o")], ids=["line", "scatter"])
    def test_line_and_scatter(self, plot_style: Any, marker: str) -> None:
        """Lines and markers are drawn at each time."""
        fig = self.setup_tf().plot("level", datetime(2024, 1, 1), datetime(2024, 1, 2), plot_style=plot_style)
        line = fig.axes[0].lines[0]
        assert_array_equal(line.get_xdata(), _num(datetime(2024, 1, 1), datetime(2024, 1, 2)))
        assert line.get_marker() == marker

    def test_bar(self) -> None:
        """Bars span each value's interval, skipping missing values."""
        fig = self.setup_tf().plot("flow", datetime(2024, 1, 1), datetime(2024, 1, 3), plot_style="bar")
        bars = [patch for patch in fig.axes[0].patches if isinstance(patch, Rectangle)]
        assert [bar.get_height() for bar in bars] == [3.0, 1.0]
        assert [bar.get_width() for bar in bars] == [1.0, 1.0]

    def test_envelope_and_mean(self) -> None:
        """The envelope fills above the maximum and below the minimum, and the mean is dashed."""
        fig = self.setup_tf().plot("flow", datetime(2024, 1, 1), envelope=True, long_term_mean=True)
        ax = fig.axes[0]
        fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert [_face(f) for f in fills] == [UPPER_FILL, LOWER_FILL]
        mean = ax.lines[1]
        assert mean.get_linestyle() == "--"
        assert_array_equal(mean.get_ydata(), [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0])
        assert fig.get_supxlabel() == "Period of record: January 2023 to January 2024"
        legend = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend == ["Flow", "Long-term mean", "Maximum", "Minimum"]

    def test_envelope_colors(self) -> None:
        """The envelope colours can be given, for the area above the maximum and below the minimum."""
        fig = self.setup_tf().plot("flow", envelope=True, envelope_colors=("#ff0000", "#0000ff"))
        fills = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
        assert [_face(f) for f in fills] == ["#ff0000", "#0000ff"]
        # The legend's colour patches are the maximum and minimum
        assert [to_hex(patch.get_facecolor()) for patch in fig.legends[0].get_patches()] == ["#ff0000", "#0000ff"]

    def test_reference_interval_footer(self) -> None:
        """The footer names the reference interval, with the months of the data used."""
        fig = self.setup_tf().plot("flow", envelope=True, reference_interval=(None, datetime(2023, 12, 31)))
        assert fig.get_supxlabel() == "Reference interval: January 2023 to January 2023"

    def test_second_envelope_outlined(self) -> None:
        """Only the first envelope in a panel is filled."""
        fig = self.setup_tf().plot(["flow", "level"], envelope=True)
        ax = fig.axes[0]
        assert len([c for c in ax.collections if isinstance(c, PolyCollection)]) == 2
        assert [line.get_linestyle() for line in ax.lines] == ["-", "-", ":", ":"]
        legend = [text.get_text() for text in fig.legends[0].get_texts()]
        assert "Level maximum and minimum" in legend

    def test_no_legend_for_single_series(self) -> None:
        """A single series has no legend."""
        assert self.setup_tf().plot("flow").legends == []

    def test_show_missing(self) -> None:
        """Missing values are shaded."""
        fig = self.setup_tf().plot("flow", datetime(2024, 1, 1), show_missing=True)
        shading = [c for c in fig.axes[0].collections if _face(c) == MISSING_FILL]
        assert len(shading) == 1
        vertices = np.asarray(shading[0].get_paths()[0].vertices)
        assert vertices[:, 0].min() == _num(datetime(2024, 1, 2))[0]

    def test_flag_strip(self) -> None:
        """Flags are drawn in a strip below the panels, with a key."""
        fig = _flagged().plot("flow", flags="flow_flags")
        strip = fig.axes[-1]
        assert len(fig.axes) == 2
        assert [label.get_text() for label in strip.get_yticklabels()] == ["flow_flags"]
        colors = {_face(c) for c in strip.collections}
        assert FLAG_NONE in colors
        assert len(colors) == 4
        key = [text.get_text() for text in fig.legends[0].get_texts()]
        assert key == ["None", "CORRECTED", "ESTIMATED", "SUSPECT"]

    def test_several_flag_columns(self) -> None:
        """Every flag column gets a row, and the key names the column."""
        tf = _flagged()
        tf.init_flag_column("QC", "other_flags")
        fig = tf.plot("flow", flags={"flow_flags": ["SUSPECT"], "other_flags": ["CORRECTED"]})
        assert [label.get_text() for label in fig.axes[-1].get_yticklabels()] == ["flow_flags", "other_flags"]
        key = [text.get_text() for text in fig.legends[0].get_texts()]
        assert key == ["None", "flow_flags: SUSPECT", "other_flags: CORRECTED"]

    def test_metadata_color(self) -> None:
        """Series colours come from the metadata."""
        tf = self.setup_tf()
        tf.column_metadata["flow"]["color"] = "#123456"
        fig = tf.plot("flow")
        assert fig.axes[0].lines[0].get_color() == "#123456"

    def test_log_y(self) -> None:
        """The y axis can be logarithmic."""
        fig = self.setup_tf().plot("level", log_y=True)
        assert fig.axes[0].get_yscale() == "log"

    @pytest.mark.parametrize(
        "start,end,initials",
        [
            (datetime(2023, 1, 1), datetime(2023, 12, 31), True),
            (datetime(2023, 1, 1), datetime(2023, 1, 4), False),
        ],
        ids=["months", "days"],
    )
    def test_time_axis(self, start: datetime, end: datetime, initials: bool) -> None:
        """Windows of a few months to a year are labelled with month initials."""
        fig = self.setup_tf().plot("level", start, end)
        fig.canvas.draw()
        labels = [label.get_text() for label in fig.axes[0].get_xticklabels(which="both") if label.get_text()]
        assert all(len(label) == 1 for label in labels) == initials

    def test_figsize(self) -> None:
        """The figure size can be given."""
        fig = self.setup_tf().plot("flow", figsize=(6.0, 3.0))
        assert tuple(fig.get_size_inches()) == (6.0, 3.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start": datetime(2024, 1, 1, tzinfo=UTC)},
            {"end": datetime(2024, 1, 1, tzinfo=UTC)},
            {"reference_interval": (datetime(2024, 1, 1, tzinfo=UTC), None)},
        ],
        ids=["start", "end", "reference interval"],
    )
    def test_time_zone_mismatch(self, kwargs: dict) -> None:
        """Datetimes must match the time column's time zone."""
        with pytest.raises(TypeError, match="has a time zone but the column does not"):
            self.setup_tf().plot("flow", **kwargs)

    def test_start_after_end(self) -> None:
        """A start after the end raises."""
        with pytest.raises(ValueError, match="'start' must not be after 'end'"):
            self.setup_tf().plot("flow", datetime(2024, 1, 2), datetime(2024, 1, 1))

    def test_missing_matplotlib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without matplotlib, plotting raises a helpful ImportError."""
        monkeypatch.delitem(sys.modules, "time_stream.plotting")
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match="time-stream\\[plot\\]"):
            self.setup_tf().plot("flow")
