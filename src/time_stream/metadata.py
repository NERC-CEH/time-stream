import copy
from typing import Any, Self


class MetadataStore:
    """Holds TimeSeries-level and per-column metadata."""

    def __init__(self) -> None:
        self._series_metadata: dict[str, Any] = {}
        self._columns_metadata: dict[str, dict[str, Any]] = {}

    def set_series_metadata(self, metadata: dict[str, Any] | None = None) -> None:
        """Create or replace the timeseries-level metadata.

        Args:
            metadata: New metadata mapping. If ``None``, the series metadata is cleared (set to an empty dict).
        """
        self._series_metadata = metadata or {}

    def update_series_metadata(self, metadata: dict[str, Any]) -> None:
        """Update and/or add new items into the existing timeseries-level metadata (existing keys are overwritten).

        Args:
            metadata: Metadata entries to merge
        """
        merged = {**self._series_metadata, **(metadata or {})}
        self._series_metadata = merged

    def get_series_metadata(self) -> dict[str, Any]:
        """Return a copy of the series-level metadata.

        Returns:
            A shallow copy of the current series metadata.
        """
        return dict(self._series_metadata)

    def set_column_metadata(self, name: str, metadata: dict[str, Any]) -> None:
        """Create or replace metadata for a specific column.

        Args:
            name: Column name
            metadata: New metadata mapping for the column. If ``None``, the column's metadata is
                        cleared (set to an empty dict).
        """
        self._columns_metadata[name] = metadata or {}

    def update_column_metadata(self, name: str, metadata: dict[str, Any]) -> None:
        """Update and/or add new items into the existing column-level metadata (existing keys are overwritten).

        Args:
            name: Column name.
            metadata: Metadata entries to merge.
        """
        current = self._columns_metadata.get(name, {})
        merged = {**current, **metadata}
        self._columns_metadata[name] = merged

    def drop_column_metadata(self, name: str) -> None:
        """Remove metadata for a specific column.

        Args:
            name: Column name (value or flag). Missing entries are ignored.
        """
        self._columns_metadata.pop(name, None)

    def reset_column_metadata(self) -> None:
        """Remove metadata for all columns."""
        self._columns_metadata.clear()

    def get_column_metadata(self, name: str) -> dict[str, Any]:
        """Return a copy of metadata for a specific column.

        Args:
            name: Column name.

        Returns:
            A shallow copy of the column's metadata.
        """
        return dict(self._columns_metadata.get(name, {}))

    def copy(self) -> Self:
        out = MetadataStore()
        out.set_series_metadata(copy.deepcopy(self._series_metadata))
        for column, meta in self._columns_metadata.items():
            out.set_column_metadata(column, copy.deepcopy(meta))

        return out
