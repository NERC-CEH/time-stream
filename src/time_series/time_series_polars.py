from typing import Iterator, Optional

import polars as pl
from time_series.period import Period
from time_series.time_series_base import TimeSeries


class TimeSeriesPolars(TimeSeries):
    def __init__(
            self,
            df: pl.DataFrame,
            time_name: str,
            resolution: Optional[Period] = None,
            periodicity: Optional[Period] = None,
            time_zone: Optional[str] = None,
    ) -> None:
        self._df = df
        super().__init__(time_name, resolution, periodicity, time_zone)

        self._setup()

    def _setup(self) -> None:
        self._set_time_zone()
        self._sort_time()
        self._validate_resolution()
        self._validate_periodicity()

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    def _validate_resolution(self):
        """ Resolution defines how 'precise' the datetimes are
        """
        if self.resolution is None:
            # Default to a resolution that accepts all datetimes
            self._resolution = Period.of_microseconds(1)

        if not self.resolution.is_epoch_agnostic:
            # E.g. 5 hours, 7 days, 9 months, etc.
            raise NotImplementedError("Not available for non-epoch agnostic resolutions")

        # Get the datetime series, and remove any offset
        original_times = self.df[self.time_name]
        original_times_no_offset = original_times.dt.offset_by("-" + self.resolution.dataframe_offset)

        # Round the (non offset) datetime series to the given resolution interval and add the offset back on
        rounded_times = original_times_no_offset.dt.truncate(self.resolution.dataframe_frequency)
        rounded_times_with_offset = rounded_times.dt.offset_by(self.resolution.dataframe_offset)

        # Compare the original series to the rounded series.  If they don't match, then the
        # datetimes are not aligned to the resolution.
        aligned = original_times.equals(rounded_times_with_offset)
        if not aligned:
            print(original_times, rounded_times_with_offset)
            raise UserWarning(f"Values in time field: \"{self.time_name}\" are not aligned to "
                              f"resolution: {self.resolution}")

    def _validate_periodicity(self):
        """ Periodicity defines the allowed frequency of the datetimes
        """
        if self.periodicity is None:
            # Default to a resolution that accepts all datetimes
            self._periodicity = Period.of_microseconds(1)

        original_times = self.df[self.time_name]

        rounded_times = original_times.dt.truncate(self.periodicity.dataframe_frequency)

        # periodicity_counts = self.df.group_by_dynamic(
        #     index_column=self.time_name,
        #     every=self.periodicity.dataframe_frequency,
        #     offset=self.periodicity.dataframe_offset,
        #     closed="left",
        #     start_by = "datapoint"
        # ).agg(pl.len().alias("count"))

        #count_check = (periodicity_counts["count"].eq(1)).all()
        #if not count_check:
        #    raise UserWarning(f"Values in time field: \"{self.time_name}\" do not conform to "
        #                      f"periodicity: {self.periodicity}")

    def _set_time_zone(self):
        default_time_zone = "UTC"
        df_time_zone = self.df.schema[self.time_name].time_zone

        if self.time_zone is not None:
            time_zone = self.time_zone
        elif self.time_zone is None and df_time_zone is not None:
            time_zone = df_time_zone
        else:
            time_zone = default_time_zone

        if df_time_zone is None:
            self._df = self.df.with_columns(
                pl.col(self.time_name).dt.replace_time_zone(time_zone)
            )
        self._time_zone = time_zone

    def _sort_time(self) -> None:
        self._df = self.df.sort(self.time_name)

    def __len__(self):
        return self._df.height

    def __iter__(self) -> Iterator:
        return self._df.iter_rows()

    def __str__(self) -> str:
        return self._df.__str__()
