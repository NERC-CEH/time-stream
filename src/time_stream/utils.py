"""
Time-Stream Utility Module.

This module provides helper functions used across the time_stream package for working with temporal data.
"""

from collections.abc import Iterable
from datetime import datetime

import polars as pl

from time_stream import Period
from time_stream.enums import DuplicateOption, TimeAnchor
from time_stream.exceptions import ColumnNotFoundError, DuplicateValueError, UnhandledEnumError


def get_date_filter(time_name: str, observation_interval: datetime | tuple[datetime, datetime | None]) -> pl.Expr:
    """Get Polars expression for observation date interval filtering.

    Args:
        time_name: The name of the time column to create the filter for
        observation_interval: Tuple of (start_date, end_date) defining the time period.

    Returns:
        pl.Expr: Boolean polars expression for date filtering.
    """
    if isinstance(observation_interval, datetime):
        start_date = observation_interval
        end_date = None
    else:
        start_date, end_date = observation_interval

    if end_date is None:
        return pl.col(time_name) >= start_date
    else:
        return pl.col(time_name).is_between(start_date, end_date)


def truncate_to_period(date_times: pl.Series, period: Period, time_anchor: TimeAnchor | None = None) -> pl.Series:
    """Truncate a Series of date/time values to the given period.

    All the date/time values in the input series are "rounded" to the specified period, based on the time anchor
    strategy chosen.

    Args:
        date_times: A Series of date/times to be truncated.
        period: The period to which the date/times should be truncated.
        time_anchor: The time anchor to which the date/times should be truncated.

    Returns:
        A `Polars` Series with the truncated date/time values.
    """
    # Need to ensure we're dealing with datetimes rather than just "dates"
    if date_times.dtype == pl.Date:
        date_times = date_times.cast(pl.Datetime("us"))

    # 1. Remove any offset from the time series
    date_times = date_times.dt.offset_by("-" + period.pl_offset)

    # 2. Truncate the (non-offset) time series to the given resolution interval
    #    Here we need to determine where the anchor points are, and if we need to nudge the datetimes towards
    #    the anchor.
    if time_anchor == TimeAnchor.END:
        # In this case, the anchor point is at the END of the period.
        #   - Subtract a micro-second (to handle datetimes on 'boundary' points that are within their own period),
        #   - Truncate to the start of the period,
        #   - Add on 1 period to get to the end point.
        date_times = date_times.dt.offset_by("-1us")
        date_times = date_times.dt.truncate(period.pl_interval)
        date_times = date_times.dt.offset_by(period.pl_interval)
    else:
        # This is a "standard" case, where the anchor point is at the START of the period,
        #   so simply truncate to the start of the period
        date_times = date_times.dt.truncate(period.pl_interval)

    # 3. Re-apply the offset
    date_times = date_times.dt.offset_by(period.pl_offset)

    return date_times


def pad_time(
    df: pl.DataFrame, time_name: str, periodicity: Period, time_anchor: TimeAnchor = TimeAnchor.START
) -> pl.DataFrame:
    """Pad the time series in the DataFrame with missing datetime rows, filling in NULLs for missing values.

    This method ensures a complete time series by adding rows for any missing timestamps within the range of
    the original DataFrame's time column. The process:

    1. Truncates existing timestamps to align with the boundary of their periodicity interval
    2. Finds the minimum and maximum timestamps in the dataset
    3. Generates a complete series of timestamps at the correct periodicity between min and max
    4. Identifies which expected timestamps are missing from the actual data
    5. Creates a DataFrame with these missing timestamps
    6. Joins this with the original data to create a complete time series

    The resulting padded DataFrame maintains all original data and column types, with NULL values populated
    for all non-time columns in the added rows.

    Args:
        df: The DataFrame to pad.
        time_name: The name of the time column to pad.
        periodicity: The periodicity of the time series.
        time_anchor: The time anchor to which the date/times conform to.

    Returns:
        pl.DataFrame of padded data
    """
    # Extract the existing datetimes, truncated to the boundary of their periodicity period
    existing_datetimes = truncate_to_period(df[time_name], periodicity, time_anchor)

    # Get the min and max datetime from the existing datetimes
    min_datetime = existing_datetimes.min()
    max_datetime = existing_datetimes.max()

    # Generate a series of the datetimes we would expect with a full time series between the start and end date
    expected_datetimes = pl.datetime_range(
        min_datetime,
        max_datetime,
        interval=periodicity.pl_interval,
        eager=True,
        time_unit=df[time_name].dtype.time_unit,
    )

    # Find any missing datetimes between expected and existing
    missing_datetimes = expected_datetimes.filter(~expected_datetimes.is_in(existing_datetimes))
    missing_df = pl.DataFrame({time_name: missing_datetimes})

    # Perform a join to create a complete time series
    padded_df = missing_df.join(df, on=time_name, how="full", coalesce=True)

    # Sort on time column
    padded_df = padded_df.sort(time_name)

    return padded_df


def gap_size_count(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Count the gap sizes in the DataFrame column, considering groups of consecutive NULL rows.

    Args:
        df: DataFrame containing column data
        column: The column to count gaps within

    Returns:
        pl.DataFrame with gap size counts.
    """
    null_mask = pl.col(column).is_null()
    df = df.with_columns(
        [
            pl.when(null_mask)
            .then(pl.len().over((null_mask != null_mask.shift(1, fill_value=False)).cum_sum()))
            .otherwise(0)
            .alias("gap_size")
        ]
    )
    return df


def check_columns_in_dataframe(df: pl.DataFrame, columns: str | Iterable[str]) -> None:
    """Checks that columns exist in the dataframe.

    Args:
        df: DataFrame to check against
        columns: String or Iterable of column name(s) to check

    Raises:
        ColumnNotFoundError: If any of the columns in the list do not exist in the dataframe.
    """
    if isinstance(columns, str):
        columns = [columns]

    invalid_columns = sorted(set(columns) - set(df.columns))
    if invalid_columns:
        raise ColumnNotFoundError(f"Columns not found in dataframe: {invalid_columns}")


def configure_period_object(period: str | Period | None) -> Period:
    """Configure a time-stream Period object.

    Converts strings to Period objects (if required).

    Args:
          period: The period to configure.

    Returns:
         A Period object.
    """
    if isinstance(period, Period):
        return period

    if period is None:
        # Default to a period that accepts all datetimes
        return Period.of_microseconds(1)

    # If it's a string, let's assume it's provided as a valid ISO duration string. And create a Period object
    if isinstance(period, str):
        period = Period.of_duration(period)

    return period


def epoch_check(period: Period) -> None:
    """Check if the period is epoch-agnostic.

    A period is considered "epoch agnostic" if it divides the timeline into consistent intervals regardless of the
    epoch (starting point) used for calculations. This ensures that the intervals are aligned with natural
    calendar or clock units (e.g., days, months, years), rather than being influenced by the specific epoch used
    in arithmetic.

    Currently, Time-Stream does not allow working with non-epoch agnostic periods.

    For example:
        - Epoch-agnostic periods include:
            - `P1Y` (1 year): Intervals are aligned to calendar years.
            - `P1M` (1 month): Intervals are aligned to calendar months.
            - `P1D` (1 day): Intervals are aligned to whole days.
            - `PT15M` (15 minutes): Intervals are aligned to clock minutes.

        - Non-epoch-agnostic periods include:
            - `P7D` (7 days): Intervals depend on the epoch. For example, starting from 2023-01-01 vs. 2023-01-03
                would result in different alignments of 7-day periods.

    Args:
        period: The period to check.

    Raises:
        NotImplementedError: If the period is not epoch-agnostic.
    """
    if not period.is_epoch_agnostic():
        # E.g., 5 hours, 7 days, 9 months, etc.
        raise NotImplementedError(f"Non-epoch agnostic  periods are not supported: {period}")


def handle_duplicates(
    df: pl.DataFrame,
    column: str,
    on_duplicates: DuplicateOption,
) -> pl.DataFrame:
    """Handle duplicate values in a DataFrame column according to the specified option.

    Args:
        df: The Polars DataFrame to operate on.
        column: The name of the column to check for duplicates.
        on_duplicates: Strategy for handling duplicates:
            - ERROR: Raise a DuplicateValueError.
            - KEEP_FIRST: Keep the first occurrence of each duplicate.
            - KEEP_LAST: Keep the last occurrence of each duplicate.
            - MERGE: Merge duplicate rows by coalescing (taking the first non-null) values.
            - DROP: Drop all rows containing duplicates in the specified column.

    Returns:
        A new Polars DataFrame with duplicates handled according to the chosen option.

    Raises:
        DuplicateValueError: If on_duplicates is set to ERROR and duplicates exist.
    """
    duplicate_mask = df[column].is_duplicated()

    if not duplicate_mask.any():
        # Nothing to do!
        return df

    if on_duplicates == DuplicateOption.ERROR:
        raise DuplicateValueError()

    elif on_duplicates == DuplicateOption.KEEP_FIRST:
        new_df = df.unique(subset=column, keep="first")

    elif on_duplicates == DuplicateOption.KEEP_LAST:
        new_df = df.unique(subset=column, keep="last")

    elif on_duplicates == DuplicateOption.MERGE:
        merge_cols = [c for c in df.columns if c != column]
        new_df = df.group_by(column).agg([pl.col(col).drop_nulls().first().alias(col) for col in merge_cols])

    elif on_duplicates == DuplicateOption.DROP:
        new_df = df.filter(~duplicate_mask)

    else:
        # Should never reach here, unless a new enum value is added in the future and logic has not been added here
        raise UnhandledEnumError(f"Unhandled duplicates option: {on_duplicates}")

    return new_df


def check_alignment(date_times: pl.Series, alignment: Period, time_anchor: TimeAnchor) -> bool:
    """Check that a Series of date/time values conforms to a given alignment period.

    Alignment defines how "precise" the datetimes are; in effect defining the set of allowed timestamps positions
    along the timeline.

    Args:
       date_times: A Series of date/times to be tested.
       alignment: The alignment period that the date/times are checked against.
       time_anchor: The time anchor to which the date/times should conform to.

    Returns:
       True if the Series conforms to the alignment period.
    """
    return date_times.equals(truncate_to_period(date_times, alignment, time_anchor))


def check_periodicity(date_times: pl.Series, periodicity: Period, time_anchor: TimeAnchor) -> bool:
    """Check that a Series of date/time values conforms to given periodicity.

    Periodicity defines the allowed "frequency" of the datetimes, i.e., how many datetimes
     entries are allowed within a given period of time.

    Args:
       date_times: A Series of date/times to be tested.
       periodicity: The periodicity period that the date/times are checked against.
       time_anchor: The time anchor to which the date/times should conform to.

    Returns:
       True if the Series conforms to the periodicity.
    """
    # Check how many unique values are in the truncated times. It should equal the length of the original
    # time-series if all time values map to single periodicity
    return truncate_to_period(date_times, periodicity, time_anchor).n_unique() == date_times.len()
