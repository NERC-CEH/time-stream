"""
Flag System Base Module.

Provides two building blocks used by all flag system types:

- ``FlagMeta`` - shared metaclass base for flag system enum classes (``__repr__``, ``__eq__``, and ``__hash__``)
- ``FlagSystemBase`` - mixin defining the shared public interface (``system_name``, ``to_dict``,
  ``get_flag``, ``value_type``).

Flag system types are enum-based and created from a name and a ``dict[str, int | str]`` mapping.
"""

from collections.abc import Mapping
from enum import EnumType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

if TYPE_CHECKING:
    import polars as pl


FlagSystemLiteral = Literal["bitwise", "categorical", "categorical_list"]


class FlagMeta(EnumType):
    """Shared metaclass base for ``BitwiseMeta`` and ``CategoricalMeta``.

    ``__repr__``, ``__eq__``, and ``__hash__`` must live on the metaclass rather than the class
    itself because methods defined on a class apply to its *instances*. Comparing two flag system
    classes as objects (e.g. ``FlagA == FlagB``) requires these methods on the metaclass.

    This matters for dynamic creation - e.g. ``BitwiseFlag("QC", {"MISSING": 1})`` always produces
    a new class object, so two calls with identical members would otherwise compare unequal.
    """

    flag_type: FlagSystemLiteral

    def __repr__(cls) -> str:
        """Return a string representation listing all enum members and their values."""
        members = ", ".join(f"{name}={member.value}" for name, member in cls.__members__.items())  # type: ignore[attr-defined]
        return f"<{cls.__name__} ({members})>"

    def __eq__(cls, other: object) -> bool:
        """Check equality by metaclass type, class name, and member names/values.

        Args:
            other: The object to compare.

        Returns:
            True if both share the same metaclass type and have the same name and members.
        """
        if not isinstance(other, FlagMeta) or type(cls) is not type(other):
            return False
        cls_members = {n: m.value for n, m in cls.__members__.items()}  # type: ignore[attr-defined]
        other_members = {n: m.value for n, m in other.__members__.items()}  # type: ignore[attr-defined]
        return cls.__name__ == other.__name__ and cls_members == other_members

    def __hash__(cls) -> int:
        """Hash based on class name and member name/value pairs."""
        return hash((cls.__name__, tuple(sorted((n, m.value) for n, m in cls.__members__.items()))))  # type: ignore[attr-defined]


class FlagSystemBase:
    """Mixin providing the shared interface for flag system types.

    This is a mixin rather than an ABC because Python's enum metaclasses conflict with ``ABCMeta``.
    """

    __members__: ClassVar[Mapping[str, Any]]

    flag_type: FlagSystemLiteral

    @classmethod
    def system_name(cls) -> str:
        """Return the name of this flag system (the enum class name).

        Returns:
            The flag system name.
        """
        return cls.__name__

    @classmethod
    def to_dict(cls) -> dict[str, int | str]:
        """Return a mapping of flag names to their values.

        Returns:
            A dict mapping each flag name to its value.
        """
        return {name: member.value for name, member in cls.__members__.items()}

    @classmethod
    def get_flag(cls, flag: int | str) -> Self:
        """Look up a flag member by name or by value.

        Subclasses must override this method with the appropriate lookup semantics.

        Args:
            flag: The flag name (``str``) or flag value (``int`` or ``str``).

        Returns:
            The matching enum member.
        """
        raise NotImplementedError

    @classmethod
    def value_type(cls) -> type:
        """Return the Python type of the flag values (``int`` or ``str``).

        Subclasses must override this method.

        Returns:
            ``int`` or ``str``.
        """
        raise NotImplementedError

    @classmethod
    def validate_column(cls, series: "pl.Series") -> None:
        """Validate that all non-null values in ``series`` are valid for this flag system.

        Subclasses must override this method.

        Args:
            series: The Polars Series to validate.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError
