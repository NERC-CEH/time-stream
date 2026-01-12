from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from time_stream.base import TimeFrame

RESOLUTION_MAPPING = {
    r"P\d+M": ["month"],  # Month
    r"P\d+D": ["day", "month"],  # Day
    r"PT\d+H": ["day", "month", "hour"],  # Hours
    r"PT\d+M": ["day", "month", "hour", "minute"],  # Minutes
    r"PT\d+S": ["day", "month", "hour", "minute", "second"],  # Seconds
    r"PT[0].\d+S": ["day", "month", "hour", "minute", "second", "microsecond"],  # Microseconds
}


def calculate_min_max_envelope(
    tf: TimeFrame,
) -> TimeFrame:
    """Calculate the min-max envelope for a TimeFrame.

    For each unique date-time find the historical min and max values across the time series. For example, for a daily
    time series that covers two years and contains all 365 days in each one, the max-min envelope for 01-Jan would be
    calculated from both instances of 01-Jan, the max-min envelope for 02-Jan would be calculated from both instances
    of 02-Jan etc

    For sub-daily time series, the min-max envelope is calculated from every instance of the day-time across the
    time series. For example, for hourly resolution, the min-max envelope would be calculated for all instances of
    01-Jan 00:00, 01-Jan 01:00 etc.

    Args:
        tf: TimeFrame object to calculate the min-max envelope for.

    Returns:
        A new polars DataFrame containing the original data alongside the calculated min-max envelope.

    """
    # Identify the resolution of the TimeFrame, and fetch the list of date related columns to group on (e.g. day, month)
    date_columns = get_date_columns(tf)

    if not date_columns:
        raise ValueError("The resolution of the TimeFrame is not supported.")

    # Construct the expressions to split out each date component into it's own column. Note that the name of the date
    # column is expected to match the corresponding datetime function name used to extract out the relevant data.
    exprs = [eval(f"pl.col('{tf.time_name}').dt.{date_col}().alias('{date_col}')") for date_col in date_columns]
    df = tf.df.with_columns(exprs)

    # Calculate the min-max values for each date by grouping by the date column combination and merging back to the
    # original dataframe so each date-time value has a corresponding min max
    min_max_df = df.group_by(date_columns, maintain_order=True).agg(
        [pl.max("value").alias("max"), pl.min("value").alias("min")]
    )

    merged_df = df.join(min_max_df, on=date_columns, how="left")

    # Remove the date columns from the final output
    output_df = merged_df.drop(date_columns)

    return output_df


def get_date_columns(tf: TimeFrame) -> list[str] | None:
    """Identify the separate date-related columns to extract from the time column.

    Using the resolution of the time frame, the various components of the date are split out into separate column names
    This allows grouping by the date components in order to calculate various properties of the data across each
    instance of that date-component-combination. For example, extracting every temperature value occurring on 1st Jan
    over a 10 year time series and calculating the maximum temperature value.

    Note that the date columns returned are expected to also match the corresponding datetime function name used to
    extract the relevant data. For example (datetime.minute()).

    Args:
        tf: TimeFrame to extract the resolution information from.

    Returns:
        List of date-component columns.
    """
    iso_duration = tf.resolution.iso_duration
    for iso_regex, date_columns in RESOLUTION_MAPPING.items():
        if re.fullmatch(iso_regex, iso_duration):
            return date_columns
