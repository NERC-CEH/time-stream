from collections.abc import Iterable
from datetime import datetime

import polars as pl

from time_stream import Period
from time_stream.exceptions import ColumnNotFoundError


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


def truncate_to_period(date_times: pl.Series, period: Period) -> pl.Series:
    """Truncate a Series of date/time values to the given period.

    All the date/time values in the input series are "rounded down" to the specified period.

    Args:
       date_times: A Series of date/times to be truncated.
       period: The period to which the date/times should be truncated.

    Examples:
       For a period of one day (Period.of_days(1)) all the date/time values are rounded
       down, or truncated, to the start of the day (the hour, minute, second, and microsecond
       fields are all set to 0).

       For a period of fifteen minutes (Period.of_minutes(15)) all the date/time values are rounded
       down, or truncated, to the start of a fifteen-minute period (the minute field is rounded down
       to either 0, 15, 30 or 45, and the second, and microsecond fields are set to 0).

    Returns:
       A `Polars` Series with the truncated date/time values.
    """
    # Remove any offset from the time series
    time_series_no_offset = date_times.dt.offset_by("-" + period.pl_offset)

    # truncate the (non-offset) time series to the given resolution interval and add the offset back on
    truncated_times = time_series_no_offset.dt.truncate(period.pl_interval)
    truncated_times_with_offset = truncated_times.dt.offset_by(period.pl_offset)

    return truncated_times_with_offset


def pad_time(df: pl.DataFrame, time_name: str, periodicity: Period) -> pl.DataFrame:
    """Pad the time series in the DataFrame with missing datetime rows, filling in NULLs for missing values.

    This method ensures a complete time series by adding rows for any missing timestamps within the range of
    the original DataFrame's time column. The process:

    1. Truncates existing timestamps to align with the start of their periodicity interval
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

    Returns:
        pl.DataFrame of padded data
    """
    # Extract the existing datetimes, truncated to the start of their periodicity period
    existing_datetimes = truncate_to_period(df[time_name], periodicity)

    # Get the min and max datetime from the existing datetimes
    min_datetime = existing_datetimes.min()
    max_datetime = existing_datetimes.max()

    # Generate a series of the datetimes we would expect with a full time series between the start and end date
    expected_datetimes = pl.datetime_range(min_datetime, max_datetime, interval=periodicity.pl_interval, eager=True)

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


def check_columns_in_dataframe(df: pl.DataFrame, columns: Iterable[str]) -> None:
    """Checks that columns exist in the dataframe.

    Args:
        df: DataFrame to check against
        columns: Iterable of column names to check

    Raises:
        ColumnNotFoundError: If any of the columns in the list do not exist in the dataframe.
    """
    invalid_columns = set(columns) - set(df.columns)
    if invalid_columns:
        raise ColumnNotFoundError(f"Columns not found in dataframe: {invalid_columns}")
