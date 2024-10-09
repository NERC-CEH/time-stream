from typing import Iterator, Optional

import polars as pl
from time_series.period import Period
from time_series.time_series_base import TimeSeries


class TimeSeriesPolars(TimeSeries):
    def __init__(
            self,
            df: pl.DataFrame,
            time_name: str,
            resolution: Period,
            periodicity: Period,
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
        rounded_time = self.df.group_by_dynamic(
            index_column=self.time_name,
            every=self.resolution.dataframe_frequency,
            offset=self.resolution.dataframe_offset,
            closed="left",
            start_by="datapoint"
        ).agg(pl.len())

        aligned = self.df[self.time_name].equals(rounded_time[self.time_name])
        if not aligned:
            raise UserWarning(f"Values in time field: \"{self.time_name}\" are not aligned to "
                              f"resolution: {self.resolution}")

    def _validate_periodicity(self):
        """ Periodicity defines the allowed frequency of the datetimes
        """
        periodicity_counts = self.df.group_by_dynamic(
            index_column=self.time_name,
            every=self.periodicity.dataframe_frequency,
            offset=self.periodicity.dataframe_offset,
            closed="left",
            start_by = "datapoint"
        ).agg(pl.len().alias("count"))

        count_check = (periodicity_counts["count"].eq(1)).all()
        if not count_check:
            raise UserWarning(f"Values in time field: \"{self.time_name}\" do not conform to "
                              f"periodicity: {self.periodicity}")

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
