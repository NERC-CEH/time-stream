import copy
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import polars as pl

if TYPE_CHECKING:
    # Import is for type hinting only.  Make sure there is no runtime import, to avoid recursion.
    from time_series import TimeSeries


class TimeSeriesColumn(ABC):
    """Base class for all column types in a TimeSeries."""
    def __init__(self, name: str, ts: "TimeSeries", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._name = name
        self._ts = ts

        #  NOTE: Doing a deep copy of this mutable object, otherwise the original object will refer to the same
        #   object in memory and will be changed by class methods.
        self._metadata = copy.deepcopy(metadata) or {}

    @property
    def name(self) -> str:
        return self._name

    def metadata(self, key: Optional[Union[str, list[str], tuple[str, ...]]] = None) -> Dict[str, Any]:
        """Retrieve metadata for all or specific keys.

        Args:
            key: A specific key or list/tuple of keys to filter the metadata. If None, all metadata is returned.

        Returns:
            A dictionary of the requested metadata.

        Raises:
            KeyError: If the requested key(s) are not found in the metadata.
        """
        if isinstance(key, str):
            key = [key]

        if key is None:
            return self._metadata
        else:
            return {k: self._metadata.get(k) for k in key}

    def remove_metadata(self, key: Optional[Union[str, list[str], tuple[str, ...]]] = None) -> None:
        """Removes metadata associated with a column, either completely or for specific keys.

        Args:
            key: A specific key or list/tuple of keys to remove. If None, all metadata for the column is removed.
        """
        if isinstance(key, str):
            key = [key]

        if key is None:
            self._metadata = {}
        else:
            self._metadata = {k: v for k, v in self.metadata().items() if k not in key}

    def set_as_supplementary(self) -> None:
        supplementary_col = SupplementaryColumn(self.name, self._ts)
        self._ts._columns[self.name] = supplementary_col

    def set_as_flag(self, flag_system: str) -> None:
        flag_col = FlagColumn(self.name, self._ts, flag_system)
        self._ts._columns[self.name] = flag_col

    def unset(self):
        # Unsetting basically means it becomes a 'normal' data column
        data_col = DataColumn(self.name, self._ts)
        self._ts._columns[self.name] = data_col

    def remove(self):
        if self.name in self._ts.df.columns:
            self._ts.df = self._ts.df.drop(self.name)
        del self._ts._columns[self.name]
        self._ts = None  # Clear reference to time series prevent further modifications

    def add_flag(self, *args, **kwargs):
        raise TypeError(f"Column '{self.name}' is not set as a flag column.")

    def remove_flag(self, *args, **kwargs):
        raise TypeError(f"Column '{self.name}' is not set as a flag column.")

    def data(self):
        return self._ts.df[[self._ts.time_name, self.name]]

    def as_timeseries(self):
        return self._ts.select([self.name])

    def __str__(self) -> str:
        """Return the string representation of the Column."""
        return str(self.data())

    def __getattr__(self, name: str) -> Any:
        """ Dynamically handle metadata attribute access for the Column object.

        Args:
            name: The attribute name being accessed.

        Returns:
            Metadata key: The metadata value for the column.

        Raises:
            AttributeError: If attribute not found.
        """
        try:
            metadata_value = self.metadata(name)
            return metadata_value
        except (KeyError, AttributeError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self) -> list[str]:
        """Return a list of attributes associated with the TimeSeries class.

        This method extends the default attributes of the TimeSeries class by including the metadata keys of this
        Column. This allows for dynamic attribute access using dot notation or introspection tools like `dir()`.

        Returns:
            A sorted list of attributes, combining the Default attributes of the class along with the names of the
            Column's metadata keys.
        """
        default_attrs = list(super().__dir__())
        custom_attrs = default_attrs + list(self._metadata.keys())
        return sorted(set(custom_attrs))


class PrimaryTimeColumn(TimeSeriesColumn):
    """Represents the primary datetime column that controls the Time Series."""
    def set_as_supplementary(self, *args, **kwargs):
        raise NotImplementedError()

    def set_as_flag(self, *args, **kwargs):
        raise NotImplementedError()

    def unset(self, *args, **kwargs):
        raise NotImplementedError()

    def data(self):
        return self._ts.df[self.name]


class DataColumn(TimeSeriesColumn):
    """Represents primary data columns."""
    pass


class SupplementaryColumn(TimeSeriesColumn):
    """Represents supplementary columns (e.g., metadata, extra information)."""
    pass


class FlagColumn(SupplementaryColumn):
    """Represents a flag column."""
    def __init__(self,
                 name: str,
                 ts: "TimeSeries",
                 flag_system: str,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name, ts, metadata)
        self.flag_system = flag_system

    def add_flag(self, flag: Union[int, str], expr: pl.Expr = pl.lit(True)) -> None:
        """Bitwise OR operation to add a flag value to rows that fulfill the expression."""
        flag = self._ts._flag_manager.flag_systems[self.flag_system].get_single_flag(flag)
        self._ts.df = self._ts.df.with_columns(
            pl.when(expr)
            .then(pl.col(self.name) | flag.value)
            .otherwise(pl.col(self.name))
        )

    def remove_flag(self, flag: Union[int, str], expr: pl.Expr = pl.lit(True)) -> None:
        """Bitwise AND operation to remove a flag from rows that fulfill the expression"""
        flag = self._ts._flag_manager.flag_systems[self.flag_system].get_single_flag(flag)
        self._ts.df = self._ts.df.with_columns(
            pl.when(expr)
            .then(pl.col(self.name) & ~flag.value)
            .otherwise(pl.col(self.name))
        )
