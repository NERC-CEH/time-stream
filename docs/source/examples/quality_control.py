"""Examples for the "Quality control" user guide page."""

from datetime import datetime, time, timedelta

import polars as pl

import time_stream as ts


def _with_qc_flags(tf: ts.TimeFrame) -> ts.TimeFrame:
    """Register the ``qc`` flag system and an empty ``flag_column`` the checks write into."""
    tf.register_flag_system("qc", {"FLAGGED": 1})
    tf.init_flag_column("qc", "flag_column")
    return tf


def _simple_series() -> ts.TimeFrame:
    """Ten hourly rows of temperature / precipitation / sensor-code data with deliberate spikes and error codes."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    df = pl.DataFrame(
        {
            "timestamp": dates,
            "temperature": [24, 22, -35, 26, 24, 26, 28, 50, 52, 29],
            "precipitation": [-3, 0, 5, 10, 2, 0, 0, 3, 1, 0],
            "sensor_codes": [992, 1, 1, 1, 1, 1, 1, 991, 995, 1],
        }
    )
    return _with_qc_flags(ts.TimeFrame(df=df, time_name="timestamp"))


def _flat_line_series() -> ts.TimeFrame:
    """Temperature series with a run of identical 20.0s and a run of 0.0s, for the flat-line checks."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    df = pl.DataFrame({"timestamp": dates, "temperature": [18.0, 20.0, 20.0, 20.0, 20.0, 22.0, 21.0, 0.0, 0.0, 0.0]})
    return _with_qc_flags(ts.TimeFrame(df=df, time_name="timestamp"))


def _near_flat_line_series() -> ts.TimeFrame:
    """Temperature series that only drifts within +/-0.01, for the ``tolerance`` flat-line check."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(10)]
    temperature = [18.0, 20.0, 20.005, 20.001, 19.991, 22.0, 20.99, 21.003, 21.009, 20.997]
    df = pl.DataFrame({"timestamp": dates, "temperature": temperature})
    return _with_qc_flags(ts.TimeFrame(df=df, time_name="timestamp"))


def setup_flags() -> None:
    """Register a flag system and an empty flag column for the checks to write into."""
    dates = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(5)]
    tf = ts.TimeFrame(pl.DataFrame({"timestamp": dates, "temperature": [24, 22, 26, 24, 28]}), time_name="timestamp")
    # [start:setup_flags]
    tf.register_flag_system("qc", {"FLAGGED": 1})
    tf.init_flag_column("qc", "flag_column")
    # [end:setup_flags]
    print(tf.df)


def comparison_qc_1() -> None:
    """Flag temperatures greater than or equal to 50."""
    tf = _simple_series()
    # fmt: off
    # [start:comparison_qc_1]
    tf = tf.qc_check(
        "comparison", "temperature", compare_to=50, operator=">=",
        flag_params=("flag_column", "FLAGGED")
    )
    # [end:comparison_qc_1]
    # fmt: on
    print(tf.df)


def comparison_qc_3() -> None:
    """Flag rows whose sensor code is one of the known error codes (``is_in``)."""
    tf = _simple_series()
    # fmt: off
    # [start:comparison_qc_3]
    error_codes = [991, 992, 993, 994, 995]
    tf = tf.qc_check(
        "comparison", "sensor_codes", compare_to=error_codes, operator="is_in",
        flag_params=("flag_column", "FLAGGED")
    )
    # [end:comparison_qc_3]
    # fmt: on
    print(tf.df)


def range_qc_1() -> None:
    """Flag temperatures outside the open range (-30, 50)."""
    tf = _simple_series()
    # [start:range_qc_1]
    tf = tf.qc_check(
        "range",
        "temperature",
        min_value=-30,
        max_value=50,
        closed="none",  # range excludes the min and max value
        within=False,  # flag values outside the range
        flag_params=("flag_column", "FLAGGED"),
    )
    # [end:range_qc_1]
    print(tf.df)


def spike_qc_1() -> None:
    """Flag points that jump more than 10 units from their neighbours."""
    tf = _simple_series()
    # fmt: off
    # [start:spike_qc_1]
    tf = tf.qc_check(
        "spike", "temperature", threshold=10.0,
        flag_params=("flag_column", "FLAGGED")
    )
    # [end:spike_qc_1]
    # fmt: on
    print(tf.df)


def time_range_qc_1() -> None:
    """Flag precipitation readings recorded between 01:00 and 03:00 (time-of-day)."""
    tf = _simple_series()
    # [start:time_range_qc_1]
    tf = tf.qc_check(
        "time_range",
        "precipitation",
        min_value=time(1, 0),
        max_value=time(3, 0),
        flag_params=("flag_column", "FLAGGED"),
    )
    # [end:time_range_qc_1]
    print(tf.df)


def time_range_qc_2() -> None:
    """Flag temperature readings falling within a fixed datetime window."""
    tf = _simple_series()
    # [start:time_range_qc_2]
    tf = tf.qc_check(
        "time_range",
        "temperature",
        min_value=datetime(2023, 1, 1, 3, 30),
        max_value=datetime(2023, 1, 1, 9, 30),
        flag_params=("flag_column", "FLAGGED"),
    )
    # [end:time_range_qc_2]
    print(tf.df)


def flat_line_qc_1() -> None:
    """Flag runs of three or more identical values."""
    tf = _flat_line_series()
    # fmt: off
    # [start:flat_line_qc_1]
    tf = tf.qc_check(
        "flat_line", "temperature", min_count=3,
        flag_params=("flag_column", "FLAGGED")
    )
    # [end:flat_line_qc_1]
    # fmt: on
    print(tf.df)


def flat_line_qc_2() -> None:
    """Flat-line check that ignores runs of 0.0"""
    tf = _flat_line_series()
    # fmt: off
    # [start:flat_line_qc_2]
    tf = tf.qc_check(
        "flat_line", "temperature", min_count=3, ignore_value=0.0,
        flag_params=("flag_column", "FLAGGED")
    )
    # [end:flat_line_qc_2]
    # fmt: on
    print(tf.df)


def flat_line_qc_3() -> None:
    """Flat-line check with a tolerance, so near-constant runs are flagged too."""
    tf = _near_flat_line_series()
    # fmt: off
    # [start:flat_line_qc_3]
    tf = tf.qc_check(
        "flat_line", "temperature", min_count=3, tolerance=0.1,
        flag_params=("flag_column", "FLAGGED")
    )
    # [end:flat_line_qc_3]
    # fmt: on
    print(tf.df)
