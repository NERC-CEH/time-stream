import enum
from typing import TYPE_CHECKING, Sequence, Type

import polars as pl

from time_stream.bitwise import BitwiseFlag
from time_stream.columns import FlagColumn

if TYPE_CHECKING:
    from time_stream.base import TimeSeries


class TimeSeriesFlagManager:
    """Manages flagging operations for a TimeSeries object using bitwise flags."""

    def __init__(self, ts: "TimeSeries", flag_systems: dict[str, Type[enum.Enum]] = None):
        """Initializes the flag manager for a TimeSeries.

        Args:
            ts: The parent TimeSeries instance.
            flag_systems: A dictionary mapping flag system names to BitwiseFlag instances.
        """
        self._ts = ts
        self._flag_systems = {}
        self._setup_flag_systems(flag_systems or {})

    @property
    def flag_systems(self) -> dict[str, Type[enum.Enum]]:
        return self._flag_systems

    def _setup_flag_systems(self, flag_systems: dict[str, dict[str, int] | Type[enum.Enum]] | None = None) -> None:
        """Adds flag systems into the flag manager.

        Args:
            flag_systems: A dictionary of flag system names mapped to either BitwiseFlag objects or dictionaries
                          defining flag values (which will be turned into BitwiseFlag objects).
        """
        for name, flag_system in flag_systems.items():
            self.add_flag_system(name, flag_system)

    def add_flag_system(self, name: str, flag_system: dict[str, dict[str, int] | Type[enum.Enum]]) -> None:
        """Adds a flag system to the flag manager, which contains flag values and their meanings.

        A flag system can be used to create flag columns that are specific to a particular type of flag.
        The flag system must contain valid bitwise values and their name. Using bitwise values
        allows for multiple flags to be set on a single value.

        For example, if we had a quality_control flag system, we could define it as follows:

        > flag_system_name = "quality_control"
        > flag_dict = {
        >     "OUT_OF_RANGE": 1,
        >     "SPIKE": 2,
        >     "LOW_BATTERY": 4,
        > }

        The flag_dict itself can be passed to this method, or a BitwiseFlag object can be created from the dict:
        > flag_system = BitwiseFlag(flag_dict, name=flag_system_name)

        A new flag column can then be created as a "quality_control" flag column.

        Args:
            name: The name of the new flag system.
            flag_system: The flag system containing flag values and their meanings.

        Raises:
            TypeError: If the flag system is not a valid type.
        """
        if name in self._flag_systems:
            raise KeyError(f"Flag system '{name}' already exists.")
        if isinstance(flag_system, enum.EnumType):
            self._flag_systems[name] = flag_system
        elif isinstance(flag_system, dict):
            self._flag_systems[name] = BitwiseFlag(name, flag_system)
        else:
            raise TypeError(f"Unknown type of flag system: {type(flag_system)}")

    def init_flag_column(self, flag_system: str, col_name: str, data: int | Sequence[int] = 0) -> None:
        """Add a new column to the TimeSeries DataFrame, setting it as a Flag Column.

        Args:
            flag_system: The name of the flag system.
            col_name: The name of the new flag column.
            data: The default value to populate the flag column with. Can be a scalar or list-like. Defaults to 0.

        Raises:
            KeyError: If flag system has not been created, or if the flag column already exists in the DataFrame or
                      the data column does not exist in the DataFrame.
        """
        if flag_system not in self._flag_systems:
            raise KeyError(f"Flag system '{flag_system}' not found.")

        if col_name in self._ts.columns:
            raise KeyError(f"Column '{col_name}' already exists in the DataFrame.")

        if isinstance(data, int):
            data = pl.lit(data, dtype=pl.Int64)
        else:
            data = pl.Series(col_name, data, dtype=pl.Int64)

        self._ts.df = self._ts.df.with_columns(data.alias(col_name))
        flag_col = FlagColumn(col_name, self._ts, flag_system)
        self._ts._columns[col_name] = flag_col

    def set_flag_column(self, flag_system: str, col_name: str | list[str]) -> None:
        """Mark the specified existing column(s) as a flag column.

        Args:
            flag_system: The name of the flag system.
            col_name: A column name (or list of column names) to mark as a flag column.
        """
        if isinstance(col_name, str):
            col_name = [col_name]

        for col in col_name:
            self._ts.columns[col].set_as_flag(flag_system)

    def add_flag(self, col_name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """Add flag value (if not there) to flag column, where expression is True.

        Args:
            col_name: The name of the flag column
            flag_value: The flag value to add
            expr: Polars expression for which rows to add flag to
        """
        self._ts.columns[col_name].add_flag(flag_value, expr)

    def remove_flag(self, col_name: str, flag_value: int | str, expr: pl.Expr = pl.lit(True)) -> None:
        """
        Remove flag value (if there) from flag column.

        Args:
            col_name: The name of the flag column
            flag_value: The flag value to remove
            expr: Polars expression for which rows to remove flag from
        """
        self._ts.columns[col_name].remove_flag(flag_value, expr)
