"""Provider contract primitives shared by platform capabilities."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class ProviderKind(StrEnum):
    """Stable categories of replaceable platform providers."""

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VECTOR_STORE = "vector_store"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Human-readable identity and capability category for a provider."""

    name: str
    kind: ProviderKind

    def __post_init__(self) -> None:
        """Validate immutable provider metadata."""
        if not self.name.strip():
            raise ValueError("provider name must not be empty")


@runtime_checkable
class Provider(Protocol):
    """Minimum structural contract implemented by platform providers."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return provider identity and capability metadata."""
        ...
