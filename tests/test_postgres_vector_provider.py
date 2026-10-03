"""Tests for PostgreSQLVectorStoreProvider without a live database."""

import asyncio
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, cast

import pytest
from psycopg import OperationalError

from ai_engineering_agent_platform.adapters.postgres import (
    PostgreSQLVectorStoreProvider,
)
from ai_engineering_agent_platform.contracts import (
    ProviderKind,
    VectorDeleteRequest,
    VectorMetadataItem,
    VectorQueryRequest,
    VectorRecord,
    VectorStoreProvider,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    ProviderUnavailableError,
)
from ai_engineering_agent_platform.runtime import (
    PostgresPool,
)


class FakeCursor:
    """Return deterministic scripted rows."""

    def __init__(
        self,
        *,
        one: tuple[Any, ...] | None = None,
        many: Sequence[tuple[Any, ...]] = (),
    ) -> None:
        self._one = one
        self._many = tuple(many)

    async def fetchone(
        self,
    ) -> tuple[Any, ...] | None:
        """Return one scripted row."""
        return self._one

    async def fetchall(
        self,
    ) -> list[tuple[Any, ...]]:
        """Return all scripted rows."""
        return list(self._many)


class FakeConnection:
    """Execute a deterministic sequence of cursor responses."""

    def __init__(
        self,
        cursors: Sequence[FakeCursor],
    ) -> None:
        self._cursors = list(cursors)
        self.calls: list[tuple[str, object]] = []

    async def execute(
        self,
        statement: str,
        params: object = None,
    ) -> FakeCursor:
        """Record SQL and return the next scripted cursor."""
        self.calls.append(
            (
                statement,
                params,
            )
        )

        if not self._cursors:
            raise AssertionError("Unexpected SQL execution")

        return self._cursors.pop(0)


class FakeConnectionContext(AbstractAsyncContextManager[FakeConnection]):
    """Async context yielding a scripted connection."""

    def __init__(
        self,
        connection: FakeConnection,
        *,
        error: Exception | None = None,
    ) -> None:
        self._connection = connection
        self._error = error

    async def __aenter__(
        self,
    ) -> FakeConnection:
        """Yield connection or raise scripted acquisition error."""
        if self._error is not None:
            raise self._error

        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool | None:
        """Propagate provider exceptions."""
        return None


class FakePool:
    """Expose the pool connection-context shape used by the adapter."""

    def __init__(
        self,
        connection: FakeConnection,
        *,
        error: Exception | None = None,
    ) -> None:
        self._connection = connection
        self._error = error

    def connection(
        self,
    ) -> FakeConnectionContext:
        """Return a deterministic connection context."""
        return FakeConnectionContext(
            self._connection,
            error=self._error,
        )


def _provider(
    cursors: Sequence[FakeCursor],
    *,
    error: Exception | None = None,
) -> tuple[
    PostgreSQLVectorStoreProvider,
    FakeConnection,
]:
    """Create provider with a structurally compatible fake pool."""
    connection = FakeConnection(cursors)

    pool = cast(
        PostgresPool,
        FakePool(
            connection,
            error=error,
        ),
    )

    return (
        PostgreSQLVectorStoreProvider(pool),
        connection,
    )


def _record(
    *,
    record_id: str = "record-1",
    vector: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
    ),
) -> VectorRecord:
    """Build deterministic vector record."""
    return VectorRecord(
        record_id=record_id,
        vector=vector,
        text="synthetic text",
        metadata=(
            VectorMetadataItem(
                key="source",
                value="unit-test",
            ),
        ),
    )


def test_provider_satisfies_vector_store_contract() -> None:
    """PostgreSQL adapter should satisfy the structural contract."""
    provider, _connection = _provider(())

    typed: VectorStoreProvider = provider

    assert isinstance(
        typed,
        VectorStoreProvider,
    )
    assert provider.descriptor.kind is ProviderKind.VECTOR_STORE
    assert provider.descriptor.name == "postgresql-pgvector"


def test_upsert_creates_collection_and_records() -> None:
    """Upsert should create missing namespace before record writes."""
    provider, connection = _provider(
        (
            FakeCursor(one=None),
            FakeCursor(one=(41, 3)),
            FakeCursor(),
            FakeCursor(),
        )
    )

    async def exercise() -> None:
        await provider.upsert(
            VectorUpsertRequest(
                space_id="test-space",
                namespace="example",
                records=(
                    _record(record_id="record-1"),
                    _record(record_id="record-2"),
                ),
            )
        )

    asyncio.run(exercise())

    assert len(connection.calls) == 4

    assert "WHERE namespace IS NOT DISTINCT FROM %s" in connection.calls[0][0]

    assert "INSERT INTO ai_platform.vector_collections" in connection.calls[1][0]

    assert "ON CONFLICT" in connection.calls[2][0]

    assert "ON CONFLICT" in connection.calls[3][0]


def test_upsert_reuses_existing_collection() -> None:
    """Existing namespaces should not be recreated."""
    provider, connection = _provider(
        (
            FakeCursor(one=(7, 3)),
            FakeCursor(),
        )
    )

    asyncio.run(
        provider.upsert(
            VectorUpsertRequest(
                space_id="test-space",
                records=(_record(),),
            )
        )
    )

    assert len(connection.calls) == 2
    assert "INSERT INTO ai_platform.vector_collections" not in connection.calls[1][0]


def test_upsert_rejects_namespace_dimension_change() -> None:
    """One namespace must retain one vector dimensionality."""
    provider, connection = _provider((FakeCursor(one=(7, 4)),))

    with pytest.raises(
        ProviderExecutionError,
        match="dimensionality",
    ):
        asyncio.run(
            provider.upsert(
                VectorUpsertRequest(
                    space_id="test-space",
                    records=(_record(),),
                )
            )
        )

    assert len(connection.calls) == 1


def test_query_returns_normalized_ranked_results() -> None:
    """Nearest-neighbor rows should map into portable response objects."""
    provider, connection = _provider(
        (
            FakeCursor(one=(7, 3)),
            FakeCursor(
                many=(
                    (
                        "record-a",
                        "alpha",
                        [
                            {
                                "key": "source",
                                "value": "a",
                            }
                        ],
                        0.0,
                    ),
                    (
                        "record-b",
                        None,
                        [],
                        1.0,
                    ),
                )
            ),
        )
    )

    response = asyncio.run(
        provider.query(
            VectorQueryRequest(
                space_id="test-space",
                vector=(
                    1.0,
                    2.0,
                    3.0,
                ),
                top_k=2,
            )
        )
    )

    assert len(connection.calls) == 2

    assert "embedding <-> %s::public.vector" in connection.calls[1][0]

    assert tuple(result.record_id for result in response.results) == (
        "record-a",
        "record-b",
    )

    assert tuple(result.rank for result in response.results) == (
        1,
        2,
    )

    assert response.results[0].score == 1.0
    assert response.results[1].score == pytest.approx(0.5)


def test_query_missing_namespace_returns_empty_response() -> None:
    """Queries should not create namespaces as a side effect."""
    provider, connection = _provider((FakeCursor(one=None),))

    response = asyncio.run(
        provider.query(
            VectorQueryRequest(
                space_id="test-space",
                vector=(1.0,),
                top_k=3,
                namespace="missing",
            )
        )
    )

    assert response.results == ()
    assert len(connection.calls) == 1


def test_query_rejects_dimension_mismatch() -> None:
    """Query dimensions must agree with the stored namespace."""
    provider, _connection = _provider((FakeCursor(one=(7, 2)),))

    with pytest.raises(
        ProviderExecutionError,
        match="dimensionality",
    ):
        asyncio.run(
            provider.query(
                VectorQueryRequest(
                    space_id="test-space",
                    vector=(
                        1.0,
                        2.0,
                        3.0,
                    ),
                    top_k=1,
                )
            )
        )


def test_delete_existing_namespace_uses_scoped_delete() -> None:
    """Deletes must always be scoped to the resolved collection."""
    provider, connection = _provider(
        (
            FakeCursor(one=(7, 3)),
            FakeCursor(),
        )
    )

    asyncio.run(
        provider.delete(
            VectorDeleteRequest(
                space_id="test-space",
                record_ids=(
                    "record-1",
                    "record-2",
                ),
            )
        )
    )

    assert len(connection.calls) == 2

    statement, params = connection.calls[1]

    assert "DELETE FROM ai_platform.vector_records" in statement
    assert "WHERE collection_id = %s" in statement
    assert "ANY(%s::text[])" in statement

    assert params == (
        7,
        [
            "record-1",
            "record-2",
        ],
    )


def test_delete_missing_namespace_is_noop() -> None:
    """Deleting from an absent namespace should be idempotent."""
    provider, connection = _provider((FakeCursor(one=None),))

    asyncio.run(
        provider.delete(
            VectorDeleteRequest(
                space_id="test-space",
                namespace="missing",
                record_ids=("record-1",),
            )
        )
    )

    assert len(connection.calls) == 1


def test_operational_error_is_normalized() -> None:
    """Connectivity failures should use ProviderUnavailableError."""
    provider, _connection = _provider(
        (),
        error=OperationalError("synthetic connectivity failure"),
    )

    with pytest.raises(
        ProviderUnavailableError,
        match="unavailable",
    ):
        asyncio.run(
            provider.query(
                VectorQueryRequest(
                    space_id="test-space",
                    vector=(1.0,),
                    top_k=1,
                )
            )
        )


def test_malformed_database_result_is_normalized() -> None:
    """Invalid database payloads should become ProviderExecutionError."""
    provider, _connection = _provider(
        (
            FakeCursor(one=(7, 1)),
            FakeCursor(
                many=(
                    (
                        "record",
                        None,
                        {"wrong": "shape"},
                        0.0,
                    ),
                )
            ),
        )
    )

    with pytest.raises(
        ProviderExecutionError,
        match="invalid vector result data",
    ):
        asyncio.run(
            provider.query(
                VectorQueryRequest(
                    space_id="test-space",
                    vector=(1.0,),
                    top_k=1,
                )
            )
        )


def test_upsert_scopes_collection_by_namespace_and_space_id() -> None:
    """Collection creation must use namespace and vector-space identity."""
    provider, connection = _provider(
        (
            FakeCursor(one=None),
            FakeCursor(one=(41, 3)),
            FakeCursor(),
        )
    )

    asyncio.run(
        provider.upsert(
            VectorUpsertRequest(
                namespace="shared",
                space_id="model-a",
                records=(_record(),),
            )
        )
    )

    assert len(connection.calls) == 3

    lookup_statement, lookup_params = connection.calls[0]

    assert "namespace IS NOT DISTINCT FROM %s" in lookup_statement
    assert "space_id = %s" in lookup_statement
    assert lookup_params == (
        "shared",
        "model-a",
    )

    insert_statement, insert_params = connection.calls[1]

    assert "INSERT INTO ai_platform.vector_collections" in insert_statement
    assert "ON CONFLICT (namespace, space_id)" in insert_statement
    assert insert_params == (
        "shared",
        "model-a",
        3,
    )


def test_query_scopes_collection_by_namespace_and_space_id() -> None:
    """Queries must resolve exactly one vector space."""
    provider, connection = _provider(
        (
            FakeCursor(one=(7, 3)),
            FakeCursor(many=()),
        )
    )

    response = asyncio.run(
        provider.query(
            VectorQueryRequest(
                namespace="shared",
                space_id="model-b",
                vector=(
                    1.0,
                    2.0,
                    3.0,
                ),
                top_k=1,
            )
        )
    )

    assert response.results == ()
    assert len(connection.calls) == 2

    lookup_statement, lookup_params = connection.calls[0]

    assert "namespace IS NOT DISTINCT FROM %s" in lookup_statement
    assert "space_id = %s" in lookup_statement
    assert lookup_params == (
        "shared",
        "model-b",
    )


def test_delete_scopes_collection_by_namespace_and_space_id() -> None:
    """Delete must never cross a vector-space boundary."""
    provider, connection = _provider(
        (
            FakeCursor(one=(7, 3)),
            FakeCursor(),
        )
    )

    asyncio.run(
        provider.delete(
            VectorDeleteRequest(
                namespace="shared",
                space_id="model-c",
                record_ids=("record-1",),
            )
        )
    )

    assert len(connection.calls) == 2

    lookup_statement, lookup_params = connection.calls[0]

    assert "namespace IS NOT DISTINCT FROM %s" in lookup_statement
    assert "space_id = %s" in lookup_statement
    assert lookup_params == (
        "shared",
        "model-c",
    )

    delete_statement, delete_params = connection.calls[1]

    assert "WHERE collection_id = %s" in delete_statement
    assert delete_params == (
        7,
        ["record-1"],
    )
