"""Live PostgreSQL + pgvector integration tests.

These tests are opt-in and require an ephemeral or explicitly prepared
PostgreSQL instance. Normal unit/quality test runs skip this module.
"""

import asyncio
import math
import os
from pathlib import Path

import pytest
from psycopg import AsyncConnection

from ai_engineering_agent_platform.adapters.postgres import (
    PostgreSQLVectorStoreProvider,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.contracts import (
    VectorDeleteRequest,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorRecord,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
)
from ai_engineering_agent_platform.runtime.postgres import (
    postgres_pool_runtime,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_PLATFORM_RUN_POSTGRES_INTEGRATION") != "1",
    reason=("live PostgreSQL integration is opt-in"),
)

RUNTIME_KEYS = {
    "AI_PLATFORM_POSTGRES_HOST",
    "AI_PLATFORM_POSTGRES_PORT",
    "AI_PLATFORM_POSTGRES_DATABASE",
    "AI_PLATFORM_POSTGRES_USER",
    "AI_PLATFORM_POSTGRES_PASSWORD",
    "AI_PLATFORM_POSTGRES_SSLMODE",
    "AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS",
    "AI_PLATFORM_POSTGRES_POOL_MIN_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_MAX_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_TIMEOUT_SECONDS",
}

BOOTSTRAP_KEYS = {
    "POSTGRES_USER",
    "POSTGRES_DB",
    "POSTGRES_PASSWORD",
}


def _load_env(
    path: Path,
    expected: set[str],
) -> dict[str, str]:
    """Load an exact external KEY=VALUE file."""
    values: dict[str, str] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue

        key, separator, value = line.partition("=")

        if separator != "=":
            raise RuntimeError(f"Malformed environment file: {path}")

        if key in values:
            raise RuntimeError(f"Duplicate environment key: {key}")

        values[key] = value

    if set(values) != expected:
        raise RuntimeError(f"Unexpected environment keys: {path}")

    return values


async def _cleanup(
    *,
    runtime: dict[str, str],
    bootstrap: dict[str, str],
) -> None:
    """Restore project tables to pristine state."""
    connection = await AsyncConnection.connect(
        host=runtime["AI_PLATFORM_POSTGRES_HOST"],
        port=int(runtime["AI_PLATFORM_POSTGRES_PORT"]),
        dbname=bootstrap["POSTGRES_DB"],
        user=bootstrap["POSTGRES_USER"],
        password=bootstrap["POSTGRES_PASSWORD"],
        sslmode=runtime["AI_PLATFORM_POSTGRES_SSLMODE"],
        connect_timeout=5,
        application_name=("ai-platform-integration-cleanup"),
    )

    try:
        await connection.execute(
            """
            TRUNCATE TABLE
                ai_platform.vector_records,
                ai_platform.vector_collections
            RESTART IDENTITY
            """
        )

        await connection.commit()
    finally:
        await connection.close()


async def _expect_execution_error(
    operation: object,
) -> None:
    """Require one provider operation to fail semantically."""
    try:
        await operation  # type: ignore[misc]
    except ProviderExecutionError:
        return

    raise RuntimeError("Expected ProviderExecutionError")


async def _exercise_live_provider() -> None:
    runtime_path = Path(os.environ["AI_PLATFORM_POSTGRES_INTEGRATION_RUNTIME_ENV"])

    bootstrap_path = Path(os.environ["AI_PLATFORM_POSTGRES_INTEGRATION_BOOTSTRAP_ENV"])

    runtime = _load_env(
        runtime_path,
        RUNTIME_KEYS,
    )

    bootstrap = _load_env(
        bootstrap_path,
        BOOTSTRAP_KEYS,
    )

    os.environ.update(runtime)

    settings = Settings()

    if settings.postgres_user != "ai_platform_runtime":
        raise RuntimeError("Integration test must use runtime role")

    try:
        async with postgres_pool_runtime(settings) as pool:
            provider = PostgreSQLVectorStoreProvider(pool)

            missing = await provider.query(
                VectorQueryRequest(
                    namespace="integration-missing",
                    vector=(
                        1.0,
                        0.0,
                        0.0,
                    ),
                    top_k=5,
                )
            )

            assert missing.results == ()

            metadata = (
                VectorMetadataItem(
                    key="string",
                    value="default",
                ),
                VectorMetadataItem(
                    key="integer",
                    value=7,
                ),
                VectorMetadataItem(
                    key="float",
                    value=1.25,
                ),
                VectorMetadataItem(
                    key="boolean",
                    value=True,
                ),
                VectorMetadataItem(
                    key="nullable",
                    value=None,
                ),
            )

            await provider.upsert(
                VectorUpsertRequest(
                    records=(
                        VectorRecord(
                            record_id="shared-id",
                            vector=(
                                1.0,
                                0.0,
                                0.0,
                            ),
                            text="default exact",
                            metadata=metadata,
                        ),
                        VectorRecord(
                            record_id="near",
                            vector=(
                                2.0,
                                0.0,
                                0.0,
                            ),
                            text="default near",
                        ),
                        VectorRecord(
                            record_id="far",
                            vector=(
                                4.0,
                                0.0,
                                0.0,
                            ),
                            text="default far",
                        ),
                    )
                )
            )

            await provider.upsert(
                VectorUpsertRequest(
                    namespace="integration-named",
                    records=(
                        VectorRecord(
                            record_id="shared-id",
                            vector=(
                                1.0,
                                0.0,
                                0.0,
                            ),
                            text="named exact",
                        ),
                    ),
                )
            )

            result = await provider.query(
                VectorQueryRequest(
                    vector=(
                        1.0,
                        0.0,
                        0.0,
                    ),
                    top_k=2,
                )
            )

            assert tuple(item.record_id for item in result.results) == (
                "shared-id",
                "near",
            )

            exact, near = result.results

            assert exact.rank == 1
            assert near.rank == 2
            assert exact.score == 1.0

            assert math.isclose(
                near.score,
                0.5,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )

            assert exact.metadata == metadata

            named = await provider.query(
                VectorQueryRequest(
                    namespace="integration-named",
                    vector=(
                        1.0,
                        0.0,
                        0.0,
                    ),
                    top_k=1,
                )
            )

            assert len(named.results) == 1
            assert named.results[0].record_id == "shared-id"
            assert named.results[0].text == "named exact"

            updated_metadata = (
                VectorMetadataItem(
                    key="version",
                    value=2,
                ),
                VectorMetadataItem(
                    key="updated",
                    value=True,
                ),
            )

            await provider.upsert(
                VectorUpsertRequest(
                    records=(
                        VectorRecord(
                            record_id="shared-id",
                            vector=(
                                1.25,
                                0.0,
                                0.0,
                            ),
                            text="default updated",
                            metadata=updated_metadata,
                        ),
                    )
                )
            )

            updated_result = await provider.query(
                VectorQueryRequest(
                    vector=(
                        1.0,
                        0.0,
                        0.0,
                    ),
                    top_k=1,
                )
            )

            updated = updated_result.results[0]

            assert updated.record_id == "shared-id"
            assert updated.text == "default updated"
            assert updated.metadata == updated_metadata

            assert math.isclose(
                updated.score,
                0.8,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )

            await _expect_execution_error(
                provider.upsert(
                    VectorUpsertRequest(
                        records=(
                            VectorRecord(
                                record_id=("wrong-dimensions"),
                                vector=(
                                    1.0,
                                    2.0,
                                ),
                            ),
                        )
                    )
                )
            )

            await _expect_execution_error(
                provider.query(
                    VectorQueryRequest(
                        vector=(
                            1.0,
                            2.0,
                        ),
                        top_k=1,
                    )
                )
            )

            await provider.delete(
                VectorDeleteRequest(
                    record_ids=(
                        "near",
                        "does-not-exist",
                    ),
                )
            )

            await provider.delete(
                VectorDeleteRequest(
                    record_ids=("near",),
                )
            )

            after_delete = await provider.query(
                VectorQueryRequest(
                    vector=(
                        1.0,
                        0.0,
                        0.0,
                    ),
                    top_k=10,
                )
            )

            assert tuple(item.record_id for item in after_delete.results) == (
                "shared-id",
                "far",
            )

            named_after_delete = await provider.query(
                VectorQueryRequest(
                    namespace=("integration-named"),
                    vector=(
                        1.0,
                        0.0,
                        0.0,
                    ),
                    top_k=10,
                )
            )

            assert tuple(item.record_id for item in named_after_delete.results) == (
                "shared-id",
            )

            await provider.delete(
                VectorDeleteRequest(
                    record_ids=(
                        "shared-id",
                        "far",
                    ),
                )
            )

            await provider.delete(
                VectorDeleteRequest(
                    namespace="integration-named",
                    record_ids=("shared-id",),
                )
            )

            async with pool.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT
                        (
                            SELECT count(*)
                            FROM ai_platform.vector_collections
                        ),
                        (
                            SELECT count(*)
                            FROM ai_platform.vector_records
                        )
                    """
                )

                counts = await cursor.fetchone()

                assert counts == (2, 0)
    finally:
        await _cleanup(
            runtime=runtime,
            bootstrap=bootstrap,
        )


def test_live_postgres_vector_provider() -> None:
    """Exercise the real provider against PostgreSQL + pgvector."""
    asyncio.run(_exercise_live_provider())
