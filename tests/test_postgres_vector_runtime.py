"""Tests for PostgreSQL vector-store runtime composition."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import pytest

import ai_engineering_agent_platform.runtime.vector_store as runtime_module
from ai_engineering_agent_platform.adapters.postgres import (
    PostgreSQLVectorStoreProvider,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.contracts import (
    ProviderKind,
)
from ai_engineering_agent_platform.runtime import (
    PostgresPool,
    postgres_vector_store_runtime,
)


class DummyPool:
    """Opaque pool object used only to test runtime ownership."""


def test_vector_store_runtime_composes_provider_and_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime should open, inject, and close the PostgreSQL pool."""
    events: list[str] = []

    pool = cast(
        PostgresPool,
        DummyPool(),
    )

    @asynccontextmanager
    async def fake_postgres_pool_runtime(
        settings: Settings,
    ) -> AsyncIterator[PostgresPool]:
        assert settings.postgres_user == "ai_platform_runtime"

        events.append("open")

        try:
            yield pool
        finally:
            events.append("close")

    monkeypatch.setattr(
        runtime_module,
        "postgres_pool_runtime",
        fake_postgres_pool_runtime,
    )

    async def exercise() -> None:
        settings = Settings()

        async with postgres_vector_store_runtime(settings) as provider:
            assert isinstance(
                provider,
                PostgreSQLVectorStoreProvider,
            )

            assert provider.descriptor.kind is ProviderKind.VECTOR_STORE

            assert provider.descriptor.name == "postgresql-pgvector"

            assert events == ["open"]

        assert events == [
            "open",
            "close",
        ]

    asyncio.run(exercise())


def test_vector_store_runtime_closes_pool_after_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pool lifecycle must close even if provider use raises."""
    events: list[str] = []

    pool = cast(
        PostgresPool,
        DummyPool(),
    )

    @asynccontextmanager
    async def fake_postgres_pool_runtime(
        _settings: Settings,
    ) -> AsyncIterator[PostgresPool]:
        events.append("open")

        try:
            yield pool
        finally:
            events.append("close")

    monkeypatch.setattr(
        runtime_module,
        "postgres_pool_runtime",
        fake_postgres_pool_runtime,
    )

    async def exercise() -> None:
        async with postgres_vector_store_runtime(Settings()):
            raise RuntimeError("synthetic provider body failure")

    with pytest.raises(
        RuntimeError,
        match="synthetic provider body failure",
    ):
        asyncio.run(exercise())

    assert events == [
        "open",
        "close",
    ]
