"""Examples for the "Plotting" user guide page."""

from datetime import date, datetime

import polars as pl

import time_stream as ts
from examples.helpers import DATA, gaugings


def _weather() -> pl.DataFrame:
    """Daily air temperature and precipitation, 2014 to 2025, from the COSMOS-UK site at Chimney Meadows."""
    df = pl.read_parquet(DATA / "chimn_daily_ta_precip_2014-2025.parquet")
    return df.rename({"Air_temperature": "temperature", "Precipitation": "precipitation"})


def _daily_flow() -> pl.DataFrame:
    """Gauged daily flow, 1968 to 2025, from the UK National River Flow Archive: station 39027, Pang at Pangbourne."""
    return pl.read_parquet(DATA / "39027_gdf.parquet")


def _flow_timeframe() -> ts.TimeFrame:
    """The daily river flow as a TimeFrame, with a title, label and units."""
    tf = ts.TimeFrame(_daily_flow(), "time", resolution="P1D")
    tf.metadata = {"title": "39027 Pang at Pangbourne"}
    tf.column_metadata["flow"] = {"label": "Flow", "units": "m³/s"}
    return tf


def _timeframe() -> ts.TimeFrame:
    """The daily weather as a TimeFrame."""
    return ts.TimeFrame(_weather(), "time", resolution="P1D")


def _with_metadata(tf: ts.TimeFrame) -> ts.TimeFrame:
    """The TimeFrame with the title, labels and units used in the later examples."""
    tf.metadata = {"title": "Chimney Meadows"}
    tf.column_metadata = {
        "temperature": {"label": "Air temperature", "units": "°C"},
        "precipitation": {"label": "Precipitation", "units": "mm", "plot_style": "bar", "color": "tab:blue"},
    }
    return tf


def quick_plot() -> None:
    """Plot one column of a TimeFrame."""
    tf = _timeframe()
    # [start:quick_plot]
    tf.plot("temperature")
    # [end:quick_plot]


def metadata_plot() -> None:
    """Labels, units and a title from the TimeFrame metadata."""
    tf = _timeframe()
    # fmt: off
    # [start:metadata_plot]
    tf.metadata = {"title": "Chimney Meadows"}
    tf.column_metadata["temperature"] = {
        "label": "Air temperature",
        "units": "°C",
        "color": "tab:red"
    }

    tf.plot("temperature", start=datetime(2025, 1, 1), end=datetime(2025, 12, 31))
    # [end:metadata_plot]
    # fmt: on


def envelope_plot() -> None:
    """One water year of river flow against the range and mean of every year since 1968, on a log scale."""
    tf = _flow_timeframe()
    # [start:envelope_plot]
    tf.plot(
        "flow",
        start=datetime(2024, 10, 1),
        end=datetime(2025, 9, 30),
        envelope=True,
        long_term_mean=True,
        log_y=True,
    )
    # [end:envelope_plot]


def envelope_colors_plot() -> None:
    """One year of temperature against the range of every year, red for hotter and blue for colder."""
    tf = _with_metadata(_timeframe())
    # [start:envelope_colors_plot]
    tf.plot(
        "temperature",
        start=datetime(2025, 1, 1),
        end=datetime(2025, 12, 31),
        envelope=True,
        envelope_colors=("#ffb3b3", "#b3b3ff"),
    )
    # [end:envelope_colors_plot]


def panels_plot() -> None:
    """Temperature and precipitation in their own panels, sharing the time axis."""
    tf = _with_metadata(_timeframe())
    # [start:panels_plot]
    tf.plot(
        [["temperature"], ["precipitation"]],
        start=datetime(2019, 1, 1),
        end=datetime(2019, 12, 31),
        show_missing=True,
    )
    # [end:panels_plot]


def rolling_mean_plot() -> None:
    """Daily temperature with its 7-day rolling mean in the same panel."""
    tf = _with_metadata(_timeframe())
    # [start:rolling_mean_plot]
    weekly = tf.rolling_aggregate("P7D", "mean", "temperature", alignment="center")

    # Join the rolling mean onto the TimeFrame, so both columns can be plotted together
    tf = tf.with_df(tf.df.join(weekly.df.select("time", "mean_temperature"), on="time"))
    tf = tf.with_column_metadata({"mean_temperature": {"label": "7-day mean", "units": "°C", "color": "tab:red"}})

    tf.plot(["temperature", "mean_temperature"], start=datetime(2025, 1, 1), end=datetime(2025, 12, 31))
    # [end:rolling_mean_plot]


def irregular_plot() -> None:
    """River flow gaugings, taken by hand at irregular times, drawn as points."""
    # [start:irregular_plot]
    tf = ts.TimeFrame(gaugings(), "time", resolution="PT1M", periodicity="irregular")
    tf = tf.with_column_metadata({"flow": {"label": "Gauged flow", "units": "m³/s"}})

    tf.plot("flow")
    # [end:irregular_plot]


def customise_plot() -> None:
    """Marking freezing point and the hottest day on the returned figure."""
    tf = _with_metadata(_timeframe())
    # [start:customise_plot]
    fig = tf.plot("temperature", start=datetime(2025, 1, 1), end=datetime(2025, 12, 31))

    # The figure is an ordinary matplotlib Figure, so anything can be added to it
    in_2025 = tf.df.filter(pl.col("time").dt.year() == 2025)
    hottest = in_2025.sort("temperature", descending=True, nulls_last=True)["time"][0]

    ax = fig.axes[0]
    ax.axhline(0, color="tab:blue", linestyle="--", linewidth=1, label="Freezing")
    ax.axvline(hottest, color="tab:orange", linewidth=1, label=f"Hottest day ({hottest:%d %B})")
    ax.legend(loc="upper left")
    # [end:customise_plot]


def _with_spikes(tf: ts.TimeFrame) -> ts.TimeFrame:
    """The TimeFrame with a few sensor glitches in 2019, giving single-day temperature spikes."""
    spikes = {date(2019, 3, 12): 14.0, date(2019, 7, 21): -12.0, date(2019, 11, 5): 15.0}
    return tf.with_df(tf.df.with_columns(pl.col("temperature") + pl.col("time").replace_strict(spikes, default=0.0)))


def flags_plot() -> None:
    """Spikes and missing values flagged by quality control, shown in a strip below the plot."""
    tf = _with_spikes(_with_metadata(_timeframe()))
    # [start:flags_plot]
    tf = tf.with_flag_system("QC", {"SUSPECT": 1, "MISSING": 2})
    tf = tf.with_flag_column("temperature_flags", "QC")

    # Flag spikes as suspect, and missing values as missing
    tf = tf.qc_check("spike", "temperature", threshold=6, flag_params=("temperature_flags", "SUSPECT"))
    tf = tf.add_flag("temperature_flags", "MISSING", pl.col("temperature").is_null())

    tf.plot("temperature", start=datetime(2019, 1, 1), end=datetime(2019, 12, 31), flags="temperature_flags")
    # [end:flags_plot]


def flags_multiple_plot() -> None:
    """Flags for temperature and precipitation, with short labels and chosen colours."""
    tf = _with_spikes(_with_metadata(_timeframe()))
    # [start:flags_multiple_plot]
    tf = tf.with_flag_system("QC", {"SUSPECT": 1, "MISSING": 2})
    tf = tf.with_flag_column("temperature_flags", "QC").with_flag_column("precipitation_flags", "QC")

    # Suspect: temperature spikes, and daily precipitation over 25 mm
    tf = tf.qc_check("spike", "temperature", threshold=6, flag_params=("temperature_flags", "SUSPECT"))
    tf = tf.qc_check(
        "range",
        "precipitation",
        min_value=0.0,
        max_value=25.0,
        within=False,
        flag_params=("precipitation_flags", "SUSPECT"),
    )
    for column in ("temperature", "precipitation"):
        tf = tf.add_flag(f"{column}_flags", "MISSING", pl.col(column).is_null())

    # Short labels for the rows of the flag strip
    tf = tf.with_column_metadata({"temperature_flags": {"label": "Temp."}, "precipitation_flags": {"label": "Precip."}})

    tf.plot(
        [["temperature"], ["precipitation"]],
        start=datetime(2019, 1, 1),
        end=datetime(2019, 12, 31),
        flags=["temperature_flags", "precipitation_flags"],
        flag_colors={"SUSPECT": "tab:orange", "MISSING": "tab:red"},
    )
    # [end:flags_multiple_plot]
