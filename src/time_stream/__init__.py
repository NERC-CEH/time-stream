import autosemver

try:
    __version__ = autosemver.packaging.get_current_version(project_name="time_stream")
except Exception:
    __version__ = "0.0.0"


__all__ = ["TimeSeries", "Period"]  # noqa


def __getattr__(name: str) -> None:
    if name == "TimeSeries":
        from time_stream.base import TimeSeries

        return TimeSeries

    if name == "Period":
        from time_stream.period import Period

        return Period

    raise AttributeError(f"module {__name__} has no attribute {name}")
