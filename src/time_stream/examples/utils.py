import contextlib
import os
import sys
from typing import Iterator


@contextlib.contextmanager
def suppress_output() -> Iterator:
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
