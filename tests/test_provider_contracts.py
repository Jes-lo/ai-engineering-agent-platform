"""Tests for foundational provider contracts."""

from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_agent_platform.contracts import (
    Provider,
    ProviderDescriptor,
    ProviderKind,
)


class ExampleProvider:
    """Minimal structural provider used only for contract testing."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider metadata."""
        return ProviderDescriptor(
            name="example",
            kind=ProviderKind.LLM,
        )


def test_provider_descriptor_is_immutable() -> None:
    """Provider metadata should be an immutable value object."""
    descriptor = ProviderDescriptor(
        name="example",
        kind=ProviderKind.LLM,
    )

    assert descriptor.name == "example"
    assert descriptor.kind is ProviderKind.LLM

    with pytest.raises(FrozenInstanceError):
        descriptor.name = "changed"  # type: ignore[misc]


def test_provider_descriptor_rejects_empty_name() -> None:
    """Provider identifiers must contain a meaningful name."""
    with pytest.raises(
        ValueError,
        match="provider name must not be empty",
    ):
        ProviderDescriptor(
            name="   ",
            kind=ProviderKind.LLM,
        )


def test_provider_protocol_supports_structural_typing() -> None:
    """Adapters should satisfy the provider contract structurally."""
    provider = ExampleProvider()

    assert isinstance(provider, Provider)
    assert provider.descriptor.name == "example"


def test_provider_kinds_have_stable_wire_values() -> None:
    """Provider categories should expose stable serialized values."""
    assert ProviderKind.LLM.value == "llm"
    assert ProviderKind.EMBEDDING.value == "embedding"
    assert ProviderKind.RERANKER.value == "reranker"
    assert ProviderKind.VECTOR_STORE.value == "vector_store"
    assert ProviderKind.TOOL.value == "tool"
