"""
Time-Stream Type Aliases.

This module defines ``Literal`` type aliases used throughout time_stream to document the accepted
string values for key parameters.
"""

from typing import Literal

DuplicateOption = Literal["drop", "keep_first", "keep_last", "error", "merge"]
MissingCriteria = Literal["percent", "missing", "available", "na"]
ClosedInterval = Literal["both", "left", "right", "none"]
TimeAnchor = Literal["start", "end", "point"]
ValidationErrorOptions = Literal["error", "resolve"]
RollingAlignment = Literal["trailing", "leading", "center"]
