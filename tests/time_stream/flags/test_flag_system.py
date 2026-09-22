from typing import Any, Callable

import polars as pl
import pytest

from time_stream.flags.flag_system import FlagSystemBase


class TestFlagSystemBase:
    @pytest.mark.parametrize(
        "method,args",
        [
            (FlagSystemBase.get_flag, (1,)),
            (FlagSystemBase.value_type, ()),
            (FlagSystemBase.column_dtype, ()),
            (FlagSystemBase.empty_value, ()),
            (FlagSystemBase.validate_column, (pl.Series([1]),)),
        ],
        ids=["get_flag", "value_type", "column_dtype", "empty_value", "validate_column"],
    )
    def test_methods_not_implemented(self, method: Callable, args: tuple[Any, ...]) -> None:
        """Test that the base methods must be overridden by a flag system type."""
        with pytest.raises(NotImplementedError):
            method(*args)
