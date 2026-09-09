"""Run the example code the user guide renders, checking only that it executes.

The guide displays these examples with ``literalinclude`` and renders their output with ``jupyter-execute``, so a
broken example breaks the documentation build. Running them here catches that in seconds instead.

These are not behavioural tests - they assert nothing. The rest of the suite covers what the library does; this
covers only that every example the documentation shows still runs.
"""

import importlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import matplotlib
import pytest

matplotlib.use("Agg")  # example plots must not need a display

_EXAMPLES_DIR = Path(__file__).parents[3] / "docs" / "source" / "examples"
_MODULES = sorted(path.stem for path in _EXAMPLES_DIR.glob("*.py") if not path.stem.startswith("_"))


def _examples_in(module: ModuleType) -> list[tuple[str, Callable[..., object]]]:
    """Return the example functions a module defines, in definition order.

    Args:
        module: The imported example module

    Returns:
        A list of (name, function) pairs, excluding anything the module merely imported
    """
    return [
        (name, value)
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_") and getattr(value, "__module__", None) == module.__name__
    ]


def test_every_guide_page_has_examples() -> None:
    """Test that the examples package was found, so an import failure cannot silently skip everything."""
    assert _MODULES, f"No example modules found in {_EXAMPLES_DIR}"


@pytest.mark.parametrize("module_name", _MODULES)
def test_examples_run(module_name: str) -> None:
    """Test that every example in a module runs without raising."""
    module = importlib.import_module(f"examples.{module_name}")
    examples = _examples_in(module)
    assert examples, f"examples/{module_name}.py defines no examples"
    for _, example in examples:
        example()
