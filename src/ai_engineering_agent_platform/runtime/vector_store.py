"""Runtime composition for vector-store provider implementations."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ai_engineering_agent_platform.adapters.postgres import (
    PostgreSQLVectorStoreProvider,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.runtime.postgres import (
    postgres_pool_runtime,
)


@asynccontextmanager
async def postgres_vector_store_runtime(
    settings: Settings,
) -> AsyncIterator[PostgreSQLVectorStoreProvider]:
    """Yield a PostgreSQL vector provider while owning its pool lifecycle."""
    async with postgres_pool_runtime(settings) as pool:
        yield PostgreSQLVectorStoreProvider(pool)
