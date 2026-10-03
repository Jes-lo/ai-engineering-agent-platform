"""Runtime composition for PostgreSQL connectivity."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import TupleRow, tuple_row
from psycopg_pool import AsyncConnectionPool

from ai_engineering_agent_platform.config import Settings

type PostgresPool = AsyncConnectionPool[AsyncConnection[TupleRow]]


def create_postgres_pool(
    settings: Settings,
) -> PostgresPool:
    """Create a closed async PostgreSQL pool from validated settings."""
    if settings.postgres_password is None:
        raise ValueError(
            "postgres_password must be configured before creating the pool"
        )

    return AsyncConnectionPool(
        conninfo="",
        connection_class=AsyncConnection[TupleRow],
        kwargs={
            "host": settings.postgres_host,
            "port": settings.postgres_port,
            "dbname": settings.postgres_database,
            "user": settings.postgres_user,
            "password": settings.postgres_password.get_secret_value(),
            "sslmode": settings.postgres_sslmode,
            "connect_timeout": settings.postgres_connect_timeout_seconds,
            "application_name": "ai-engineering-agent-platform",
            "row_factory": tuple_row,
        },
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        timeout=settings.postgres_pool_timeout_seconds,
        open=False,
        check=AsyncConnectionPool.check_connection,
        name="ai-platform-postgres",
    )


@asynccontextmanager
async def postgres_pool_runtime(
    settings: Settings,
) -> AsyncIterator[PostgresPool]:
    """Open, validate, yield, and deterministically close the DB pool."""
    pool = create_postgres_pool(settings)

    try:
        await pool.open(
            wait=True,
            timeout=settings.postgres_pool_timeout_seconds,
        )
        yield pool
    finally:
        await pool.close()
