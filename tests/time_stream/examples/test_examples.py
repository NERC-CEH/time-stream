import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Callable, Iterator

import pytest

import time_stream.examples as examples


def iter_example_modules() -> Iterator[ModuleType]:
    """Get all example modules"""
    for module_info in pkgutil.iter_modules(examples.__path__, examples.__name__ + "."):
        yield importlib.import_module(module_info.name)


def get_example_functions(module: ModuleType) -> list[Callable]:
    """Get all functions from the specific examples module."""
    return [
        func
        for name, func in inspect.getmembers(module, inspect.isfunction)
        if not inspect.signature(func).parameters  # Skip functions that require arguments (most shouldn't!)
    ]


def all_example_funcs() -> Iterator[tuple[str, Callable]]:
    """Get all functions from all the examples modules."""
    for module in iter_example_modules():
        for func in get_example_functions(module):
            yield f"{module.__name__.split('.')[-1]}.{func.__name__}", func


class TestExampleFunctions:
    @pytest.mark.parametrize("name,func", all_example_funcs())
    def test_examples(self, name: str, func: Callable) -> None:
        """Test that each example function can be called without errors."""
        func()
