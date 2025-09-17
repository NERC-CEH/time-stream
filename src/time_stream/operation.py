"""
Operation Base Class

This module defines the `Operation` abstract base class, which provides a pattern for any class system that provides
an "operation" on the time series data. It enables flexible instantiation of operations from strings, types, or
existing instances with a class registry system.

Key features:
- Centralised registry with per-subclass isolation.
- Class decorator (`@Operation.register`) to add new operations.
- Factory method (`Operation.get`) for resolving operations by name, class, or instance.
"""

from abc import ABC
from typing import ClassVar, Self

from time_stream.exceptions import DuplicateRegistryKeyError, RegistryKeyTypeError, UnknownRegistryKeyError


class Operation(ABC):
    name: ClassVar[str]
    _REGISTRY: ClassVar[dict[str, type[Self]]]

    def __init_subclass__(cls, **kwargs) -> None:
        """Called on subclass init, to ensure each subclass gets its own registry dict (not shared)."""
        super().__init_subclass__(**kwargs)
        # Only create a new dict if the subclass didn't define one explicitly.
        if "_REGISTRY" not in cls.__dict__:
            cls._REGISTRY = {}

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Define the structure of the keys for the registry dict.

        Args:
            key: The key to normalize.

        Returns:
            The normalized key.
        """
        return key.lower()

    @classmethod
    def available(cls) -> list[str]:
        """Return a sorted list of available registered keys."""
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def register(cls, impl: type[Self]) -> type[Self]:
        """A method used as a decorator for subclasses to add to the register by its `name` attribute.

        Args:
            impl: The class to register.

        Returns:
            The decorated class.

        Raises:
            DuplicateRegistryKeyError: If the subclass `name` already exists in the registry.
        """
        key = cls._normalize_key(impl.name)
        if key in cls._REGISTRY:
            raise DuplicateRegistryKeyError(f"Key '{key}' already registered for {cls.__name__}")
        cls._REGISTRY[key] = impl
        return impl

    @classmethod
    def get(
        cls,
        spec: str | type[Self] | Self,
        **kwargs,
    ) -> Self:
        """Resolve `spec` to an instance of `Self`:

        Args:
            spec: The object to resolve.
                    - str: look up in registry; construct with **kwargs
                    - type: must subclass `cls`; construct with **kwargs
                    - instance: returned as-is
            **kwargs: Parameters specific to the class, used to initialise the class object.
                      Ignored if check is already an instance.

        Returns:
            A concrete instance of the `spec` object.

        Raises:
            UnknownRegistryKeyError: If `spec` is a string, and doesn't exist in the registry.
            RegistryKeyTypeError: If `spec` is an unexpected type.
        """
        # If it's already an instance, return it
        if isinstance(spec, cls):
            return spec

        # If it's a string, look it up in the registry
        elif isinstance(spec, str):
            try:
                return cls._REGISTRY[cls._normalize_key(spec)](**kwargs)
            except KeyError:
                raise UnknownRegistryKeyError(
                    f"Unknown name '{spec}' for class type '{cls.__name__}'. Available: {cls.available()}."
                )

        # If it's a class, instantiate it
        elif isinstance(spec, type):
            if issubclass(spec, cls):
                return spec(**kwargs)  # type: ignore[call-arg] - subclass __init__ will vary
            else:
                raise RegistryKeyTypeError(f"Class '{spec.__name__}' must inherit from '{cls.__name__}'.")

        else:
            raise RegistryKeyTypeError(
                f"Check object must be a string, {cls.__name__} class, or instance. Got {type(spec).__name__}."
            )
