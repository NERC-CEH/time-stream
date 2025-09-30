from typing import TYPE_CHECKING, Any

import autosemver

if TYPE_CHECKING:
    # These imports are only for static type checkers (e.g., Pyright, IDEs).
    # At runtime, they are not executed, so the modules won't be imported unless needed.
    from time_stream.base import TimeFrame
    from time_stream.period import Period

try:
    __version__ = autosemver.packaging.get_current_version(project_name="time_stream")
except Exception:
    __version__ = "0.0.0"


# Declare the public API of the package. This tells `from time_stream import *` what to include.
__all__ = ["TimeFrame", "Period"]  # noqa


def __getattr__(name: str) -> Any:
    # NOTE: We use __getattr__ for lazy imports instead of top-level imports because setuptools may evaluate
    #   this module during build (e.g., to use the value of time_stream.__version__), before submodules like
    #   `time_stream.base` or `time_stream.period` exist. This avoids import-time errors when building from source
    #   using pyproject.toml and ensures compatibility with dynamic versioning tools like autosemver.
    #
    #   This 'lazy loading' also has additional benefits in that it can help with performance by avoiding unnecessary
    #   dependencies being loaded at startup and also gives us control over what to expose from the package.

    if name == "TimeFrame":
        from time_stream.base import TimeFrame  # noqa: PLC0415

        return TimeFrame

    if name == "Period":
        from time_stream.period import Period  # noqa: PLC0415

        return Period

    raise AttributeError(f"module {__name__} has no attribute {name}")
