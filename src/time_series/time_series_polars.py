from typing import Iterable, Iterator, Optional, Union

import polars as pl

from time_series.period import Period
from time_series.time_series_base import TimeSeries


class TimeSeriesPolars(TimeSeries):
    """A class representing a time series data model using a Polars DataFrame.

    This class extends the `TimeSeries` base class and provides functionality for handling time series data
    using Polars.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        time_name: str,
        resolution: Optional[Period] = None,
        periodicity: Optional[Period] = None,
        time_zone: Optional[str] = None,
        supp_col_names: Optional[list] = [],
    ) -> None:
        """Initialize a TimeSeriesPolars instance.

        Args:
            df: The Polars DataFrame containing the time series data.
            time_name: The name of the time column in the DataFrame.
            resolution: The resolution of the time series. Defaults to None.
            periodicity: The periodicity of the time series. Defaults to None.
            time_zone: The time zone of the time series. Defaults to None.
        """
        super().__init__(time_name, resolution, periodicity, time_zone)
        self._df = df
        self._supp_col_names = supp_col_names
        self._setup()

    def _setup(self) -> None:
        """Perform initial setup for the time series object."""
        self._set_time_zone()
        self._sort_time()
        self._validate_resolution()
        self._validate_periodicity()
        self._set_column_types()

    @property
    def df(self) -> pl.DataFrame:
        """Get the underlying Polars DataFrame."""
        return self._df

    @property
    def data_col_names(self) -> list:
        return self._data_col_names

    @property
    def supp_col_names(self) -> list:
        return self._supp_col_names

    def add_supp_column(self, col_name: str, data: Union[int, float, str, Iterable]) -> None:
        """Add a supplementary column to df"""
        if isinstance(data, (float, int, str)):
            data = pl.lit(data)
        else:
            data = pl.Series(data)

        self._df = self._df.with_columns(data.alias(col_name))

        self._supp_col_names.append(col_name)

    def _validate_resolution(self) -> None:
        """Validate the resolution of the time series.

        Resolution defines how "precise" the datetimes are, i.e. to what precision of time unit should each
        datetime in the time series match to.

        Some examples:
        P0.000001S  Allow all datetime values, including microseconds.
        P1S	        Allow datetimes with a whole number of seconds. Microseconds must be "0".
        PT1M	    Allow datetimes to be specified to the minute. Seconds and Microseconds must be "0".
        PT15M	    Allow datetimes to be specified to a multiple of 15 minutes.
                    Seconds and Microseconds must be "0", and Minutes be one of ("00", "15", "30", "45")
        P1D	        Allow all dates, but the time must be "00:00:00"
        P1M	        Allow all years and months, but the day must be "1" and time "00:00:00"
        P3M	        Quarterly dates; month must be one of ("1", "4", "7", "10"), day must be "1" and time "00:00:00"
        P1Y+9M9H	Only dates at 09:00 am on the 1st of October are allowed.

        Raises:
            UserWarning: If the datetimes are not aligned to the resolution.
        """
        if self.resolution is None:
            # Default to a resolution that accepts all datetimes
            self._resolution = Period.of_microseconds(1)

        self._epoch_check(self.resolution)

        # Compare the original series to the rounded series.  If no match, it is not aligned to the resolution.
        rounded_times = self._round_time_to_period(self.resolution)
        aligned = self.df[self.time_name].equals(rounded_times)
        if not aligned:
            raise UserWarning(
                f'Values in time field: "{self.time_name}" are not aligned to ' f"resolution: {self.resolution}"
            )

    def _validate_periodicity(self) -> None:
        """Validate the periodicity of the time series.

        Periodicity defines the allowed "frequency" of the datetimes, i.e. how many datetimes entries are allowed within
        a given period of time.

        Some examples:
        P0.000001S	Effectively there is no "periodicity".
        P1S	        At most 1 datetime can occur within any given second.
        PT1M	    At most 1 datetime can occur within any given minute.
        PT15M	    At most 1 datetime can occur within any 15-minute duration.
                    Each 15-minute durations starts at ("00", "15", "30", "45") minutes past the hour.
        P1D	        At most 1 datetime can occur within any given calendar day
                    (from midnight of first day up to, but not including, midnight of the next day)
        P1M	        At most 1 datetime can occur within any given calendar month
                    (from midnight on the 1st of the month up to, but not including, midnight on the 1st of the
                    following month).
        P3M	        At most 1 datetime can occur within any given quarterly period.
        P1Y+9M9H	At most 1 datetime can occur within any given water year
                    (from 09:00 am on the 1st of October up to, but including, 09:00 am on the 1st of the
                    following year).

        Raises:
            UserWarning: If the datetimes do not conform to the periodicity.
        """
        if self.periodicity is None:
            # Default to a periodicity that accepts all datetimes
            self._periodicity = Period.of_microseconds(1)

        self._epoch_check(self.periodicity)

        # Check how many unique values are in the rounded times. It should equal the length of the original time series
        # if all time values map to a single periodicity
        rounded_times = self._round_time_to_period(self.periodicity)
        all_unique = rounded_times.n_unique() == len(self.df[self.time_name])
        if not all_unique:
            raise UserWarning(
                f'Values in time field: "{self.time_name}" do not conform to ' f"periodicity: {self.periodicity}"
            )

    def _set_time_zone(self) -> None:
        """Set the time zone for the time series.

        If a time zone is provided in class initialisation, this will overwrite any time zone set on the dataframe.
        If no time zone provided, defaults to either the dataframe time zone, or if this is not set either, then UTC.
        """
        default_time_zone = "UTC"
        df_time_zone = self.df.schema[self.time_name].time_zone

        if self.time_zone is not None:
            time_zone = self.time_zone
        elif self.time_zone is None and df_time_zone is not None:
            time_zone = df_time_zone
        else:
            time_zone = default_time_zone

        self._df = self.df.with_columns(pl.col(self.time_name).dt.replace_time_zone(time_zone))
        self._time_zone = time_zone

    def _sort_time(self) -> None:
        """Sort the DataFrame by the time column."""
        self._df = self.df.sort(self.time_name)

    def _set_column_types(self) -> None:
        """Check and define which columns are data columns and which are
        supplemenary columns.
        If not specifed, assume all but the time column are data columns
        """
        data_col_names = []
        supp_col_names = []

        for col in self._df.columns:
            if col == self.time_name:
                continue

            if col not in self._supp_col_names:
                data_col_names.append(col)
            else:
                supp_col_names.append(col)

        if len(supp_col_names) != len(self._supp_col_names):
            # Supplementary columns specified that are not in the dataframe
            col_diff = set(self._supp_col_names).difference(set(supp_col_names))
            raise ValueError(f"Cannot assign supplementary columns not found in dataframe: {", ".join(col_diff)}")

        self._data_col_names = data_col_names
        self._supp_col_names = supp_col_names

    def _round_time_to_period(self, period: Period) -> pl.Series:
        """Round the time column to the given period.

        Args:
           period: The period to which the time column should be rounded.

        Returns:
           A Polars Series with the rounded time values.
        """
        # Remove any offset from the time series
        time_series_no_offset = self.df[self.time_name].dt.offset_by("-" + period.pl_offset)

        # Round the (non offset) time series to the given resolution interval and add the offset back on
        rounded_times = time_series_no_offset.dt.truncate(period.pl_interval)
        rounded_times_with_offset = rounded_times.dt.offset_by(period.pl_offset)

        return rounded_times_with_offset

    @staticmethod
    def _epoch_check(period: Period) -> None:
        """Check if the period is epoch-agnostic.

        Args:
            period: The period to check.

        Raises:
            NotImplementedError: If the period is not epoch-agnostic.
        """
        if not period.is_epoch_agnostic():
            # E.g. 5 hours, 7 days, 9 months, etc.
            raise NotImplementedError("Not available for non-epoch agnostic periodicity")

    def __len__(self) -> int:
        """Get the number of rows in the time series."""
        return self._df.height

    def __iter__(self) -> Iterator:
        """Return an iterator over the rows of the DataFrame."""
        return self._df.iter_rows()

    def __str__(self) -> str:
        return self._df.__str__()
