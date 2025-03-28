import autosemver

from time_series.base import TimeSeries
from time_series.period import Period

__all__ = ["TimeSeries", "Period"]


try:
    __version__ = autosemver.packaging.get_current_version(project_name="time_series")
except Exception:
    __version__ = "0.0.0"
