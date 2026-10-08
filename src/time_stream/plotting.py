"""
Plotting Module.

Plots the data in a :class:`~time_stream.TimeFrame` with matplotlib. Each value is drawn over the period of time it is
valid for, with an optional envelope of the maximum and minimum from the period of record, and a strip below the plot
marking flagged values.

Metadata keywords read from the TimeFrame:

- Column metadata: ``label`` (name in legends and axis labels), ``units`` (added to the axis label), ``color``
  (any matplotlib colour) and ``plot_style`` (``"line"``, ``"step"``, ``"bar"`` or ``"scatter"``).
- TimeFrame metadata: ``title`` (shown at the top left).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from isoperiod import Period
from matplotlib.artist import Artist
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator, NullFormatter

from time_stream.exceptions import ColumnNotFoundError, ColumnTypeError
from time_stream.types import PlotStyle, TimeAnchor
from time_stream.utils import check_columns_in_dataframe, check_literal_value, check_time_zone, pad_time

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from time_stream.base import TimeFrame
    from time_stream.flags.flag_manager import FlagColumn

# Metadata keywords
LABEL = "label"
UNITS = "units"
COLOR = "color"
PLOT_STYLE = "plot_style"
TITLE = "title"

# Default colours
SERIES_COLORS = ["#000000", "#ff0000", "#00b200", "#0000ff", "#ff00ff", "#00b2b2", "#b28c00", "#007c00", "#b27a7a"]
UPPER_FILL = "#b3b3ff"
LOWER_FILL = "#ffb3b3"
MISSING_FILL = "#b3ffb3"

# Flag strip colours: unflagged, then a grey scale from the least to the most severe flag
FLAG_NONE = "#e0e0e0"
FLAG_LEAST = "#bdbdbd"
FLAG_MOST = "#1a1a1a"

# Windows of this many days get one tick label per month, as month initials
MONTH_AXIS_DAYS = (85, 400)


def plot_timeframe(
    tf: TimeFrame,
    columns: str | Sequence[str | Sequence[str]] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    subplots: bool = False,
    plot_style: PlotStyle | dict[str, PlotStyle] | None = None,
    envelope: bool | str | Sequence[str] = False,
    long_term_mean: bool | str | Sequence[str] = False,
    period_of_record: tuple[datetime | None, datetime | None] | None = None,
    flags: str | Sequence[str] | dict[str, Sequence[int | str]] | None = None,
    show_missing: bool = False,
    title: str | None = None,
    log_y: bool = False,
    ylim: tuple[float | None, float | None] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot a TimeFrame. See :meth:`time_stream.TimeFrame.plot` for the arguments.

    Returns:
        The matplotlib figure.
    """
    panels = _panels(tf, columns, subplots)
    plotted = [column for panel in panels for column in panel]
    styles: dict[str, PlotStyle] = {column: _plot_style(tf, column, plot_style) for column in plotted}
    envelope_columns = _selected(envelope, plotted, "envelope")
    mean_columns = _selected(long_term_mean, plotted, "long_term_mean")
    flag_levels = _flag_levels(tf, flags)

    bounds = [("start", start), ("end", end), *(("period_of_record", value) for value in period_of_record or ())]
    for name, value in bounds:
        if value is not None:
            check_time_zone(value, tf.df[tf.time_name].dtype, name)
    if start is not None and end is not None and start > end:
        raise ValueError(f"'start' must not be after 'end'. Got: {start} > {end}")

    window = _window(tf, [*plotted, *flag_levels], start, end)
    times, starts, ends = _intervals(window[tf.time_name], tf.resolution, tf.time_anchor)

    statistics_columns = list(dict.fromkeys([*envelope_columns, *mean_columns]))
    statistics, record = _slot_statistics(tf, window, statistics_columns, period_of_record)

    width, height = figsize or (10.0, 4.0 * len(panels) + 0.3 * len(flag_levels) + 0.6)
    fig = plt.figure(figsize=(width, height), layout="constrained")
    height_ratios = [4.0] * len(panels) + ([max(0.3 * len(flag_levels), 0.4)] if flag_levels else [])
    axes = fig.subplots(len(height_ratios), 1, sharex=True, squeeze=False, height_ratios=height_ratios)[:, 0]

    colors = _colors(tf, plotted)
    legend: dict[str, Artist] = {}
    for ax, panel in zip(axes, panels):
        panel_legend = _draw_panel(
            ax,
            tf,
            panel,
            window,
            (times, starts, ends),
            styles,
            colors,
            statistics,
            envelope_columns,
            mean_columns,
            show_missing,
            log_y,
            ylim,
        )
        for label, handle in panel_legend.items():
            legend.setdefault(label, handle)

    if flag_levels:
        flag_key = _draw_flag_strip(axes[-1], tf, window, starts, ends, flag_levels)
        fig.legend(flag_key.values(), flag_key.keys(), loc="outside right lower", frameon=False, title="Flags")

    if len(legend) > 1:
        fig.legend(legend.values(), legend.keys(), loc="outside right upper", frameon=False)

    x_start, x_end = float(np.nanmin(starts)), float(np.nanmax(ends))
    _format_time_axis(axes[-1], x_start, x_end)

    axes[0].set_title(title or tf.metadata.get(TITLE) or "", loc="left", fontsize="x-large")
    years = window[tf.time_name].dt.year().to_numpy()
    axes[0].set_title(_period_label(int(years.min()), int(years.max())), loc="right", fontsize="x-large")
    if record is not None:
        fig.supxlabel(f"Period of record: {record[0]} to {record[1]}", x=0.0, ha="left", fontsize="medium")
    return fig


def _panels(tf: TimeFrame, columns: str | Sequence[str | Sequence[str]] | None, subplots: bool) -> list[list[str]]:
    """Group the columns to plot into panels.

    Args:
        tf: The TimeFrame being plotted.
        columns: A column, a list of columns sharing one panel, or a list of lists with one list per panel.
        subplots: Whether to give every column its own panel.

    Returns:
        The columns in each panel.

    Raises:
        ColumnNotFoundError: If there are no columns to plot, or a column does not exist.
        ColumnTypeError: If a column is not numeric.
    """
    if columns is None:
        panels = [[column for column in tf.data_columns if tf.df[column].dtype.is_numeric()]]
    elif isinstance(columns, str):
        panels = [[columns]]
    elif all(isinstance(column, str) for column in columns):
        panels = [[str(column) for column in columns]]
    else:
        panels = [[column] if isinstance(column, str) else list(column) for column in columns]

    if subplots:
        panels = [[column] for panel in panels for column in panel]

    if not panels or not all(panels):
        raise ColumnNotFoundError("No columns to plot.")

    plotted = [column for panel in panels for column in panel]
    check_columns_in_dataframe(tf.df, plotted)
    for column in plotted:
        if column == tf.time_name or not tf.df[column].dtype.is_numeric():
            raise ColumnTypeError(f"Column '{column}' is not numeric, so cannot be plotted.")
    return panels


def _plot_style(tf: TimeFrame, column: str, plot_style: PlotStyle | dict[str, PlotStyle] | None) -> PlotStyle:
    """The style to draw a column in: the argument, then the column metadata, then the default for the TimeFrame.

    Args:
        tf: The TimeFrame being plotted.
        column: The column to style.
        plot_style: A style for every column, or a mapping of column to style.

    Returns:
        The plot style.
    """
    style = plot_style.get(column) if isinstance(plot_style, dict) else plot_style
    if style is None:
        style = tf.column_metadata[column].get(PLOT_STYLE)
    if style is None:
        style = _default_style(tf)
    check_literal_value(style, PlotStyle, "plot_style")
    return style


def _default_style(tf: TimeFrame) -> PlotStyle:
    """The default style for a TimeFrame, from its temporal properties.

    Sparse data (periodicity coarser than the alignment) is drawn as points, instantaneous data as a line, and data
    valid over a period as steps across that period.

    Args:
        tf: The TimeFrame being plotted.

    Returns:
        The plot style.
    """
    if tf.periodicity != tf.alignment:
        return "scatter"
    if tf.time_anchor == "point":
        return "line"
    return "step"


def _selected(value: bool | str | Sequence[str], plotted: list[str], name: str) -> list[str]:
    """The columns an option applies to.

    Args:
        value: ``True`` for every plotted column, ``False`` for none, or the column name(s).
        plotted: The columns being plotted.
        name: The name of the option, used in the error message.

    Returns:
        The selected columns.

    Raises:
        ColumnNotFoundError: If a selected column is not being plotted.
    """
    if isinstance(value, bool):
        return list(plotted) if value else []
    selected = [value] if isinstance(value, str) else list(value)
    not_plotted = [column for column in selected if column not in plotted]
    if not_plotted:
        raise ColumnNotFoundError(f"'{name}' columns are not being plotted: {not_plotted}")
    return selected


def _flag_levels(
    tf: TimeFrame, flags: str | Sequence[str] | dict[str, Sequence[int | str]] | None
) -> dict[str, list[str]]:
    """The flags to show for each flag column, from the least to the most severe.

    Args:
        tf: The TimeFrame being plotted.
        flags: Flag column name(s), showing every flag in their flag system, or a mapping of flag column to flags.

    Returns:
        Mapping of flag column name to flag names.

    Raises:
        ValueError: If a flag column is given no flags.
    """
    if flags is None:
        return {}
    if isinstance(flags, str):
        flags = [flags]
    if not isinstance(flags, dict):
        flags = {name: list(tf.get_flag_column(name).flag_system.__members__) for name in flags}

    levels = {}
    for name, values in flags.items():
        if not values:
            raise ValueError(f"No flags given for flag column '{name}'.")
        flag_system = tf.get_flag_column(name).flag_system
        levels[name] = [_member_name(flag_system.__members__, flag_system.get_flag(value)) for value in values]
    return levels


def _member_name(members: Mapping[str, Any], member: Any) -> str:
    """The name of a flag system member.

    Args:
        members: The flag system's members, by name.
        member: The member.

    Returns:
        The member's name.
    """
    return next(name for name, candidate in members.items() if candidate is member)


def _window(tf: TimeFrame, columns: list[str], start: datetime | None, end: datetime | None) -> pl.DataFrame:
    """The rows to plot, padded with nulls for any missing time steps.

    Data at the default microsecond resolution has no regular time step, so is not padded.

    Args:
        tf: The TimeFrame being plotted.
        columns: The columns needed.
        start: Start of the window (inclusive).
        end: End of the window (inclusive).

    Returns:
        The time column and the given columns, within the window.

    Raises:
        ValueError: If there is nothing to plot in the window.
    """
    time = pl.col(tf.time_name)
    df = tf.df.select(tf.time_name, *dict.fromkeys(columns))
    if start is not None:
        df = df.filter(time >= start)
    if end is not None:
        df = df.filter(time <= end)

    regular = tf.periodicity != Period.of_microseconds(1)
    if regular and (not df.is_empty() or (start is not None and end is not None)):
        df = pad_time(df, tf.time_name, tf.periodicity, tf.time_anchor, start, end)

    if df.is_empty():
        raise ValueError("No data to plot in the time window.")
    return df


def _to_num(times: pl.Series) -> np.ndarray:
    """Convert datetimes to matplotlib date numbers.

    Args:
        times: Date or datetime values.

    Returns:
        Float days since the matplotlib epoch.
    """
    return np.asarray(mdates.date2num(times.to_numpy()), dtype=float)


def _intervals(
    times: pl.Series, resolution: Period, time_anchor: TimeAnchor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The time of each value, and the interval it is valid over.

    A ``start`` anchored value is valid over ``[t, t + resolution)`` and an ``end`` anchored one over
    ``(t - resolution, t]``. A ``point`` value is valid only at ``t``, so gets ``[t, t + resolution)`` for drawing as
    steps or bars.

    Args:
        times: The time column.
        resolution: The resolution of the TimeFrame.
        time_anchor: The time anchor of the TimeFrame.

    Returns:
        Matplotlib date numbers of each time, interval start and interval end.
    """
    if times.dtype == pl.Date:
        times = times.cast(pl.Datetime("us"))
    step = resolution.pl_interval
    if time_anchor == "end":
        starts, ends = times.dt.offset_by(f"-{step}"), times
    else:
        starts, ends = times, times.dt.offset_by(step)
    return _to_num(times), _to_num(starts), _to_num(ends)


def _step_xy(starts: np.ndarray, ends: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The outline of values held flat across their intervals, broken wherever the intervals do not meet.

    Args:
        starts: Interval starts.
        ends: Interval ends.
        values: The value over each interval.

    Returns:
        The x and y coordinates of the outline.
    """
    x = np.column_stack([starts, ends]).ravel()
    y = np.repeat(values, 2)
    breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
    return np.insert(x, 2 * breaks, ends[breaks - 1]), np.insert(y, 2 * breaks, np.nan)


def _runs(starts: np.ndarray, ends: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge consecutive intervals with the same key into runs.

    Args:
        starts: Interval starts.
        ends: Interval ends.
        keys: The key of each interval.

    Returns:
        The key, start and end of each run.
    """
    new_run = np.ones(len(keys), dtype=bool)
    new_run[1:] = (keys[1:] != keys[:-1]) | (starts[1:] != ends[:-1])
    first = np.flatnonzero(new_run)
    last = np.append(first[1:], len(keys)) - 1
    return keys[first], starts[first], ends[last]


def _slot_statistics(
    tf: TimeFrame,
    window: pl.DataFrame,
    columns: list[str],
    period_of_record: tuple[datetime | None, datetime | None] | None,
) -> tuple[pl.DataFrame, tuple[str, str] | None]:
    """The maximum, minimum and mean from the period of record for each row of the window.

    Every value is placed in a slot by its month, day and time of day, so for daily data 1 January in the window gets
    the statistics of every 1 January in the period of record.

    Args:
        tf: The TimeFrame being plotted.
        window: The rows being plotted.
        columns: The columns to calculate statistics for.
        period_of_record: Start and end (inclusive) of the data to calculate from. ``None`` for either uses all data.

    Returns:
        ``<column>_max``, ``<column>_min`` and ``<column>_mean`` for each row of the window, and the months of the
        first and last data in the period of record (e.g. ``"October 2013"``).
    """
    if not columns:
        return pl.DataFrame(), None

    time = pl.col(tf.time_name)
    source = tf.df.select(tf.time_name, *columns)
    record_start, record_end = period_of_record or (None, None)
    if record_start is not None:
        source = source.filter(time >= record_start)
    if record_end is not None:
        source = source.filter(time <= record_end)

    slot = [time.dt.month().alias("_month"), time.dt.day().alias("_day")]
    if source[tf.time_name].dtype != pl.Date:
        slot.append(time.dt.time().alias("_time"))
    slot_names = [expr.meta.output_name() for expr in slot]

    statistics = source.group_by(slot).agg(
        *[pl.col(c).max().alias(f"{c}_max") for c in columns],
        *[pl.col(c).min().alias(f"{c}_min") for c in columns],
        *[pl.col(c).mean().alias(f"{c}_mean") for c in columns],
    )
    by_row = (
        window.select(slot)
        .join(statistics, on=slot_names, how="left", maintain_order="left")
        .drop(slot_names)
        .cast(pl.Float64)
    )

    with_data = source.filter(pl.any_horizontal(pl.col(columns).is_not_null()))[tf.time_name].sort()
    months = with_data.dt.strftime("%B %Y")
    record = (months[0], months[-1]) if not months.is_empty() else None
    return by_row, record


def _colors(tf: TimeFrame, columns: list[str]) -> dict[str, Any]:
    """The colour of each column: from its metadata, otherwise the next unused default colour.

    Args:
        tf: The TimeFrame being plotted.
        columns: The columns being plotted.

    Returns:
        Mapping of column to colour.
    """
    chosen = {column: tf.column_metadata[column].get(COLOR) for column in columns}
    unused = [color for color in SERIES_COLORS if color not in chosen.values()]
    for column, color in chosen.items():
        if color is None:
            chosen[column] = unused.pop(0) if unused else SERIES_COLORS[len(chosen) % len(SERIES_COLORS)]
    return chosen


def _lighter(color: Any) -> tuple[float, float, float]:
    """A colour halfway to white.

    Args:
        color: Any matplotlib colour.

    Returns:
        The lighter colour as RGB.
    """
    r, g, b = to_rgb(color)
    return (r + 1) / 2, (g + 1) / 2, (b + 1) / 2


def _label(tf: TimeFrame, column: str) -> str:
    """The display name of a column: its ``label`` metadata, otherwise its name.

    Args:
        tf: The TimeFrame being plotted.
        column: The column.

    Returns:
        The label.
    """
    return str(tf.column_metadata[column].get(LABEL) or column)


def _y_label(tf: TimeFrame, panel: list[str]) -> str:
    """The y axis label of a panel: the label and units of a single column, or the units shared by all columns.

    Args:
        tf: The TimeFrame being plotted.
        panel: The columns in the panel.

    Returns:
        The axis label, empty if there is none.
    """
    units = {tf.column_metadata[column].get(UNITS) for column in panel}
    unit = units.pop() if len(units) == 1 else None
    if len(panel) == 1:
        label = _label(tf, panel[0])
        return f"{label} ({unit})" if unit else label
    return str(unit or "")


def _y_limits(
    arrays: list[np.ndarray], log_y: bool, ylim: tuple[float | None, float | None] | None
) -> tuple[float, float]:
    """Y axis limits covering the values, on round numbers, from zero when the values are closer to zero than
    they are spread.

    Args:
        arrays: Every set of values drawn on the panel.
        log_y: Whether the y axis is logarithmic.
        ylim: Limits given by the user, overriding either end.

    Returns:
        The lower and upper limits.
    """
    values = np.concatenate([array[np.isfinite(array)] for array in arrays]) if arrays else np.array([])
    if log_y:
        values = values[values > 0]
    if values.size == 0:
        lower, upper = (1.0, 10.0) if log_y else (0.0, 1.0)
    elif log_y:
        lower, upper = values.min() / 1.25, values.max() * 1.25
    else:
        lower, upper = float(values.min()), float(values.max())
        spread = upper - lower
        lower = 0.0 if 0 <= lower < spread else lower - spread * 0.02
        upper += spread * 0.02
        if upper <= lower:
            pad = abs(lower) * 0.1 or 1.0
            lower, upper = lower - pad, upper + pad
        ticks = np.asarray(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lower, upper))
        below, above = ticks[ticks <= lower], ticks[ticks >= upper]
        lower = below.max() if below.size else lower
        upper = above.min() if above.size else upper

    if ylim is not None:
        lower = lower if ylim[0] is None else ylim[0]
        upper = upper if ylim[1] is None else ylim[1]
    return float(lower), float(upper)


def _draw_panel(
    ax: Axes,
    tf: TimeFrame,
    panel: list[str],
    window: pl.DataFrame,
    intervals: tuple[np.ndarray, np.ndarray, np.ndarray],
    styles: dict[str, PlotStyle],
    colors: dict[str, Any],
    statistics: pl.DataFrame,
    envelope_columns: list[str],
    mean_columns: list[str],
    show_missing: bool,
    log_y: bool,
    ylim: tuple[float | None, float | None] | None,
) -> dict[str, Artist]:
    """Draw the columns of one panel.

    The first envelope in a panel is filled; any others are outlined in the colour of their column.

    Args:
        ax: The axes to draw on.
        tf: The TimeFrame being plotted.
        panel: The columns to draw.
        window: The rows being plotted.
        intervals: The time, interval start and interval end of each row.
        styles: The plot style of each column.
        colors: The colour of each column.
        statistics: The period of record statistics for each row.
        envelope_columns: Columns to draw the envelope of.
        mean_columns: Columns to draw the long-term mean of.
        show_missing: Whether to shade where values are missing.
        log_y: Whether the y axis is logarithmic.
        ylim: Y axis limits given by the user.

    Returns:
        Legend handles, by label.
    """
    times, starts, ends = intervals
    legend: dict[str, Artist] = {}
    drawn: list[np.ndarray] = []
    envelopes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    multiple = len(panel) > 1

    def xy(style: PlotStyle, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _step_xy(starts, ends, values) if style in ("step", "bar") else (times, values)

    for column in panel:
        style, color, label = styles[column], colors[column], _label(tf, column)
        values = window[column].cast(pl.Float64).to_numpy()
        drawn.append(values)

        if style == "bar":
            has_value = np.isfinite(values)
            widths = ends[has_value] - starts[has_value]
            ax.bar(starts[has_value], values[has_value], widths, align="edge", color=color, linewidth=0.5, zorder=3)
            legend[label] = Patch(color=color)
        else:
            x, y = xy(style, values)
            marker, linestyle = ("o", "none") if style == "scatter" else ("none", "-")
            (line,) = ax.plot(x, y, color=color, marker=marker, markersize=3, linestyle=linestyle, zorder=3)
            legend[label] = line

        if column in mean_columns:
            mean_label = f"{label} long-term mean" if multiple else "Long-term mean"
            x, y = xy(style, statistics[f"{column}_mean"].to_numpy())
            (line,) = ax.plot(x, y, color=_lighter(color), linestyle="--", linewidth=1.5, zorder=2)
            legend[mean_label] = line
            drawn.append(y)

        if column in envelope_columns:
            upper = statistics[f"{column}_max"].to_numpy()
            lower = statistics[f"{column}_min"].to_numpy()
            x, upper = xy(style, upper)
            _, lower = xy(style, lower)
            envelopes.append((column, x, upper, lower))
            drawn.extend([upper, lower])

    bottom, top = _y_limits(drawn, log_y, ylim)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(bottom, top)

    for index, (column, x, upper, lower) in enumerate(envelopes):
        prefix = f"{_label(tf, column)} " if multiple else ""
        if index == 0:
            ax.fill_between(x, upper, top, color=UPPER_FILL, linewidth=0, zorder=1)
            ax.fill_between(x, bottom, lower, color=LOWER_FILL, linewidth=0, zorder=1)
            legend[f"{prefix}Maximum"] = Patch(color=UPPER_FILL)
            legend[f"{prefix}Minimum"] = Patch(color=LOWER_FILL)
        else:
            (line,) = ax.plot(x, upper, color=colors[column], linestyle=":", linewidth=1, zorder=2)
            ax.plot(x, lower, color=colors[column], linestyle=":", linewidth=1, zorder=2)
            legend[f"{prefix}maximum and minimum"] = line

    if show_missing:
        missing = window.select(pl.any_horizontal(pl.col(panel).is_null()))[:, 0].to_numpy()
        keys, run_starts, run_ends = _runs(starts, ends, missing)
        spans = [(s, e - s) for s, e in zip(run_starts[keys], run_ends[keys])]
        if spans:
            ax.broken_barh(spans, (0, 1), transform=ax.get_xaxis_transform(), color=MISSING_FILL, zorder=1.5)
            legend["Missing"] = Patch(color=MISSING_FILL)

    ax.set_ylabel(_y_label(tf, panel))
    ax.tick_params(axis="y", which="both", right=True, labelright=True)
    if not log_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    return legend


def _flag_colors(count: int) -> list[tuple[float, ...]]:
    """Grey scale colours for flags, from the least to the most severe.

    Args:
        count: The number of flags.

    Returns:
        A colour per flag.
    """
    least, most = np.array(to_rgb(FLAG_LEAST)), np.array(to_rgb(FLAG_MOST))
    fractions = np.linspace(0, 1, count) if count > 1 else np.array([0.5])
    return [tuple(float(c) for c in least + (most - least) * f) for f in fractions]


def _flag_level(df: pl.DataFrame, flag_column: FlagColumn, flags: list[str]) -> np.ndarray:
    """The most severe flag set on each row, as its position in ``flags`` counting from 1, or 0 for none.

    Args:
        df: The rows being plotted.
        flag_column: The flag column.
        flags: Flag names, from the least to the most severe.

    Returns:
        The level of each row.
    """
    level = pl.lit(0)
    for index, flag in enumerate(flags, start=1):
        level = pl.when(flag_column.filter_expr([flag]).fill_null(False)).then(index).otherwise(level)
    return df.select(level.alias("level"))["level"].to_numpy()


def _draw_flag_strip(
    ax: Axes,
    tf: TimeFrame,
    window: pl.DataFrame,
    starts: np.ndarray,
    ends: np.ndarray,
    flag_levels: dict[str, list[str]],
) -> dict[str, Artist]:
    """Draw a strip with a row per flag column, each value coloured by its most severe flag.

    Args:
        ax: The axes to draw on.
        tf: The TimeFrame being plotted.
        window: The rows being plotted.
        starts: Interval starts.
        ends: Interval ends.
        flag_levels: Flag names for each flag column, from the least to the most severe.

    Returns:
        The colour key, by label.
    """
    key: dict[str, Artist] = {"None": Patch(color=FLAG_NONE)}
    multiple = len(flag_levels) > 1

    for row, (name, flags) in enumerate(flag_levels.items()):
        colors = [FLAG_NONE, *_flag_colors(len(flags))]
        levels = _flag_level(window, tf.get_flag_column(name), flags)
        run_levels, run_starts, run_ends = _runs(starts, ends, levels)
        for level in np.unique(run_levels):
            in_level = run_levels == level
            spans = list(zip(run_starts[in_level], run_ends[in_level] - run_starts[in_level]))
            color = colors[level]
            # The edge keeps runs shorter than a pixel visible
            ax.broken_barh(spans, (row - 0.4, 0.8), facecolors=color, edgecolors=color, linewidth=0.5)
        for flag, color in zip(flags, colors[1:]):
            key[f"{_label(tf, name)}: {flag}" if multiple else flag] = Patch(color=color)

    ax.set_yticks(range(len(flag_levels)), [_label(tf, name) for name in flag_levels])
    ax.set_ylim(len(flag_levels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    return key


def _format_time_axis(ax: Axes, x_start: float, x_end: float) -> None:
    """Set the time axis limits and ticks: month initials for windows of a few months to a year

    Args:
        ax: The bottom axes, sharing its time axis with the others.
        x_start: Start of the axis, as a matplotlib date number.
        x_end: End of the axis, as a matplotlib date number.
    """
    ax.set_xlim(x_start, x_end)
    if MONTH_AXIS_DAYS[0] <= x_end - x_start <= MONTH_AXIS_DAYS[1]:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=16))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: f"{mdates.num2date(x):%b}"[0]))
        ax.tick_params(axis="x", which="minor", length=0)
    else:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _period_label(first: int, last: int) -> str:
    """The years covered by the window, e.g. ``2025`` or ``2025/2026``.

    Args:
        first: First year in the window.
        last: Last year in the window.

    Returns:
        The label.
    """
    if first == last:
        return str(first)
    if last == first + 1:
        return f"{first}/{last}"
    return f"{first}-{last}"
