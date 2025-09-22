from typing import Any, Callable

from time_stream.exceptions import MetadataError


class ColumnMetadataDict(dict[str, dict[str, Any]]):
    """Lightweight dict-like container for per-column metadata that handles validation of the keys and values.

    This mapping enforces two rules:
      1) The key must be an existing DataFrame column (checked via the current_columns callback).
      2) The value must be a mapping (e.g., dict[str, Any]).
    """

    def __init__(self, current_columns: Callable[[], list[str]]) -> None:
        """Initialise the column-metadata dict

        Args:
            current_columns: A callable returning the current list of valid DataFrame column names.
        """
        super().__init__()
        self._current_columns = current_columns

    def _validate_key(self, key: str) -> None:
        """Ensure the key is the name of an existing DataFrame column.

        Args:
            key: The name of the column to validate.
        """
        if key not in self._current_columns():
            raise KeyError(f"Metadata column key '{key}' not found in TimeSeries")

    @staticmethod
    def _validate_value(value: dict[str, Any]) -> None:
        """Ensure the value is a dict/mapping.

        Args:
            value: Value to validate
        """
        if not isinstance(value, dict):
            raise MetadataError("Column metadata values must be mappings (e.g. dict)")

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Overwrite the update method to update the mapping with key/value pairs from another mapping or iterable -
        but crucially adding in our validation.
        """
        if args:
            other = args[0]
            if isinstance(other, dict):
                for k, v in other.items():
                    self[k] = v
            else:
                # assume a sequence of (key, value) pairs
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key: str, value: dict[str, Any]) -> None:
        """Assign per-column metadata after validating key and value.

        Args:
            key: Column that metadata is valid for
            value: Metadata dict for the given column
        """
        self._validate_key(key)
        self._validate_value(value)
        super().__setitem__(key, value)
