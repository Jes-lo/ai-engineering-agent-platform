"""Tests for PostgreSQL async-pool runtime composition."""

import asyncio
from typing import cast

import pytest
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

import ai_engineering_agent_platform.runtime.postgres as postgres_runtime
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.runtime.postgres import (
    PostgresPool,
    create_postgres_pool,
)


def test_postgres_pool_requires_password() -> None:
    """Pool creation must fail closed when no DB secret is configured."""
    with pytest.raises(
        ValueError,
        match="postgres_password must be configured",
    ):
        create_postgres_pool(Settings())


def test_postgres_pool_is_constructed_closed() -> None:
    """Async pools must never perform I/O from their constructor."""
    value = "-".join(("test", "only", "credential"))

    pool = create_postgres_pool(
        Settings(
            postgres_password=SecretStr(value),
        )
    )

    assert isinstance(pool, AsyncConnectionPool)
    assert pool.closed
    assert pool.name == "ai-platform-postgres"
    assert pool.min_size == 1
    assert pool.max_size == 5


class _FakePool:
    def __init__(self) -> None:
        self.open_calls: list[tuple[bool, float]] = []
        self.close_calls = 0

    async def open(
        self,
        *,
        wait: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.open_calls.append((wait, timeout))

    async def close(
        self,
        timeout: float = 5.0,
    ) -> None:
        del timeout
        self.close_calls += 1


def test_postgres_runtime_opens_waits_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime ownership must explicitly open and close the pool."""
    fake = _FakePool()

    monkeypatch.setattr(
        postgres_runtime,
        "create_postgres_pool",
        lambda settings: cast(
            PostgresPool,
            fake,
        ),
    )

    settings = Settings(
        postgres_pool_timeout_seconds=7.5,
    )

    async def exercise() -> None:
        async with postgres_runtime.postgres_pool_runtime(settings) as pool:
            assert pool is cast(PostgresPool, fake)
            assert fake.close_calls == 0

    asyncio.run(exercise())

    assert fake.open_calls == [(True, 7.5)]
    assert fake.close_calls == 1


def test_postgres_runtime_closes_after_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owned DB resources must close even when caller logic fails."""
    fake = _FakePool()

    monkeypatch.setattr(
        postgres_runtime,
        "create_postgres_pool",
        lambda settings: cast(
            PostgresPool,
            fake,
        ),
    )

    async def exercise() -> None:
        with pytest.raises(
            RuntimeError,
            match="synthetic failure",
        ):
            async with postgres_runtime.postgres_pool_runtime(Settings()):
                raise RuntimeError("synthetic failure")

    asyncio.run(exercise())

    assert fake.close_calls == 1
