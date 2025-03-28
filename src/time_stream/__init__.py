import autosemver

from time_stream.base import TimeSeries
from time_stream.period import Period

__all__ = ["TimeSeries", "Period"]


try:
    __version__ = autosemver.packaging.get_current_version(project_name="time_stream")
except Exception:
    __version__ = "0.0.0"
