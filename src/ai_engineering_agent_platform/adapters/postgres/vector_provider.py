"""PostgreSQL + pgvector implementation of the vector-store contract."""

from typing import Any

from psycopg import (
    AsyncConnection,
    OperationalError,
)
from psycopg import (
    Error as PsycopgError,
)
from psycopg.rows import TupleRow
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ai_engineering_agent_platform.adapters.postgres.vector_mapping import (
    build_metadata_payload,
    build_vector_literal,
    distance_to_score,
    parse_metadata_payload,
)
from ai_engineering_agent_platform.contracts import (
    ProviderDescriptor,
    ProviderKind,
    VectorDeleteRequest,
    VectorQueryRequest,
    VectorQueryResponse,
    VectorQueryResult,
    VectorUpsertRequest,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
    ProviderUnavailableError,
)

type AdapterPostgresPool = AsyncConnectionPool[AsyncConnection[TupleRow]]

type CollectionRow = tuple[int, int]


class PostgreSQLVectorStoreProvider:
    """Persist and query provider-neutral vectors through PostgreSQL."""

    _descriptor = ProviderDescriptor(
        name="postgresql-pgvector",
        kind=ProviderKind.VECTOR_STORE,
    )

    def __init__(
        self,
        pool: AdapterPostgresPool,
    ) -> None:
        """Use an injected pool without owning its lifecycle."""
        self._pool = pool

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return stable vector-store provider metadata."""
        return self._descriptor

    async def upsert(
        self,
        request: VectorUpsertRequest,
    ) -> None:
        """Insert or replace all records atomically within one namespace."""
        try:
            async with self._pool.connection() as connection:
                collection_id = await self._ensure_collection(
                    connection,
                    namespace=request.namespace,
                    dimensions=request.dimensions,
                )

                for record in request.records:
                    await connection.execute(
                        """
                        INSERT INTO ai_platform.vector_records (
                            collection_id,
                            record_id,
                            embedding,
                            text,
                            metadata
                        )
                        VALUES (
                            %s,
                            %s,
                            %s::public.vector,
                            %s,
                            %s
                        )
                        ON CONFLICT (
                            collection_id,
                            record_id
                        )
                        DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata
                        """,
                        (
                            collection_id,
                            record.record_id,
                            build_vector_literal(record.vector),
                            record.text,
                            Jsonb(build_metadata_payload(record.metadata)),
                        ),
                    )
        except ProviderExecutionError:
            raise
        except OperationalError as exc:
            raise ProviderUnavailableError(
                "PostgreSQL vector-store upsert unavailable"
            ) from exc
        except PsycopgError as exc:
            raise ProviderExecutionError(
                "PostgreSQL vector-store upsert failed"
            ) from exc

    async def query(
        self,
        request: VectorQueryRequest,
    ) -> VectorQueryResponse:
        """Return exact L2 nearest neighbors with normalized scores."""
        try:
            async with self._pool.connection() as connection:
                collection = await self._find_collection(
                    connection,
                    request.namespace,
                )

                if collection is None:
                    return VectorQueryResponse(results=())

                collection_id, dimensions = collection

                if dimensions != request.dimensions:
                    raise ProviderExecutionError(
                        "Vector namespace dimensionality does not match query"
                    )

                query_vector = build_vector_literal(request.vector)

                cursor = await connection.execute(
                    """
                    SELECT
                        record_id,
                        text,
                        metadata,
                        embedding <-> %s::public.vector
                            AS distance
                    FROM ai_platform.vector_records
                    WHERE collection_id = %s
                    ORDER BY
                        embedding <-> %s::public.vector ASC,
                        record_id ASC
                    LIMIT %s
                    """,
                    (
                        query_vector,
                        collection_id,
                        query_vector,
                        request.top_k,
                    ),
                )

                rows = await cursor.fetchall()

                results = tuple(
                    self._parse_query_row(
                        row,
                        rank=index + 1,
                    )
                    for index, row in enumerate(rows)
                )

                return VectorQueryResponse(results=results)
        except ProviderExecutionError:
            raise
        except OperationalError as exc:
            raise ProviderUnavailableError(
                "PostgreSQL vector-store query unavailable"
            ) from exc
        except PsycopgError as exc:
            raise ProviderExecutionError(
                "PostgreSQL vector-store query failed"
            ) from exc

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> None:
        """Delete requested records from one namespace."""
        try:
            async with self._pool.connection() as connection:
                collection = await self._find_collection(
                    connection,
                    request.namespace,
                )

                if collection is None:
                    return

                collection_id, _dimensions = collection

                await connection.execute(
                    """
                    DELETE FROM ai_platform.vector_records
                    WHERE collection_id = %s
                      AND record_id = ANY(%s::text[])
                    """,
                    (
                        collection_id,
                        list(request.record_ids),
                    ),
                )
        except OperationalError as exc:
            raise ProviderUnavailableError(
                "PostgreSQL vector-store delete unavailable"
            ) from exc
        except PsycopgError as exc:
            raise ProviderExecutionError(
                "PostgreSQL vector-store delete failed"
            ) from exc

    async def _ensure_collection(
        self,
        connection: AsyncConnection[TupleRow],
        *,
        namespace: str | None,
        dimensions: int,
    ) -> int:
        """Return collection id, creating the namespace when absent."""
        existing = await self._find_collection(
            connection,
            namespace,
        )

        if existing is not None:
            collection_id, existing_dimensions = existing

            if existing_dimensions != dimensions:
                raise ProviderExecutionError(
                    "Vector namespace dimensionality does not match upsert"
                )

            return collection_id

        cursor = await connection.execute(
            """
            INSERT INTO ai_platform.vector_collections (
                namespace,
                dimensions
            )
            VALUES (%s, %s)
            ON CONFLICT (namespace)
            DO NOTHING
            RETURNING
                collection_id,
                dimensions
            """,
            (
                namespace,
                dimensions,
            ),
        )

        inserted = await cursor.fetchone()

        if inserted is not None:
            collection_id, inserted_dimensions = self._parse_collection_row(inserted)

            if inserted_dimensions != dimensions:
                raise ProviderExecutionError(
                    "PostgreSQL returned unexpected collection dimensionality"
                )

            return collection_id

        concurrent = await self._find_collection(
            connection,
            namespace,
        )

        if concurrent is None:
            raise ProviderExecutionError(
                "PostgreSQL could not resolve vector namespace"
            )

        collection_id, concurrent_dimensions = concurrent

        if concurrent_dimensions != dimensions:
            raise ProviderExecutionError(
                "Vector namespace dimensionality does not match upsert"
            )

        return collection_id

    async def _find_collection(
        self,
        connection: AsyncConnection[TupleRow],
        namespace: str | None,
    ) -> CollectionRow | None:
        """Find collection identity and dimensionality by namespace."""
        cursor = await connection.execute(
            """
            SELECT
                collection_id,
                dimensions
            FROM ai_platform.vector_collections
            WHERE namespace IS NOT DISTINCT FROM %s
            """,
            (namespace,),
        )

        row = await cursor.fetchone()

        if row is None:
            return None

        return self._parse_collection_row(row)

    @staticmethod
    def _parse_collection_row(
        row: tuple[Any, ...],
    ) -> CollectionRow:
        """Validate the collection row returned by PostgreSQL."""
        if len(row) != 2:
            raise ProviderExecutionError(
                "PostgreSQL returned malformed collection data"
            )

        collection_id, dimensions = row

        if (
            isinstance(collection_id, bool)
            or not isinstance(collection_id, int)
            or collection_id <= 0
        ):
            raise ProviderExecutionError(
                "PostgreSQL returned invalid collection identity"
            )

        if (
            isinstance(dimensions, bool)
            or not isinstance(dimensions, int)
            or dimensions <= 0
        ):
            raise ProviderExecutionError(
                "PostgreSQL returned invalid collection dimensions"
            )

        return (
            collection_id,
            dimensions,
        )

    @staticmethod
    def _parse_query_row(
        row: tuple[Any, ...],
        *,
        rank: int,
    ) -> VectorQueryResult:
        """Normalize one PostgreSQL nearest-neighbor row."""
        if len(row) != 4:
            raise ProviderExecutionError("PostgreSQL returned malformed vector result")

        record_id, text, metadata, distance = row

        if not isinstance(record_id, str):
            raise ProviderExecutionError(
                "PostgreSQL returned invalid vector record identity"
            )

        if text is not None and not isinstance(text, str):
            raise ProviderExecutionError(
                "PostgreSQL returned invalid vector record text"
            )

        try:
            parsed_metadata = parse_metadata_payload(metadata)
            score = distance_to_score(distance)
            return VectorQueryResult(
                record_id=record_id,
                score=score,
                rank=rank,
                text=text,
                metadata=parsed_metadata,
            )
        except (TypeError, ValueError) as exc:
            raise ProviderExecutionError(
                "PostgreSQL returned invalid vector result data"
            ) from exc
