"""PostgreSQL durable agent continuation store."""

import hmac
import json
import re
from hashlib import sha256

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationConflictError,
    AgentContinuationIntegrityError,
    AgentContinuationKind,
    AgentContinuationStore,
    agent_continuation_key,
    validate_agent_continuation_identity,
)
from ai_engineering_agent_platform.services.agent_continuation_persistence import (
    AGENT_CONTINUATION_FORMAT_VERSION,
    AgentContinuationCodec,
    AgentContinuationSerializationError,
    AgentContinuationSnapshot,
)

PostgresPool = AsyncConnectionPool[AsyncConnection[TupleRow]]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_SELECT_SQL = """
SELECT
    continuation_key,
    continuation_kind,
    identity_json,
    continuation_status,
    continuation_format,
    state_version,
    continuation_payload,
    continuation_sha256
FROM ai_platform.agent_continuations
WHERE continuation_key = %s
"""

_INSERT_SQL = """
INSERT INTO ai_platform.agent_continuations (
    continuation_key,
    continuation_kind,
    identity_json,
    continuation_status,
    continuation_format,
    state_version,
    continuation_payload,
    continuation_sha256
)
VALUES (
    %s,
    %s,
    %s,
    'active',
    %s,
    1,
    %s,
    %s
)
ON CONFLICT DO NOTHING
RETURNING continuation_key
"""

_REPLACE_SQL = """
UPDATE ai_platform.agent_continuations
SET
    state_version = state_version + 1,
    continuation_payload = %s,
    continuation_sha256 = %s,
    updated_at = now()
WHERE
    continuation_key = %s
    AND continuation_kind = %s
    AND identity_json = %s
    AND continuation_status = 'active'
    AND state_version = %s
    AND continuation_sha256 = %s
RETURNING state_version
"""

_CONSUME_SQL = """
UPDATE ai_platform.agent_continuations
SET
    continuation_status = 'consumed',
    state_version = state_version + 1,
    updated_at = now(),
    consumed_at = now()
WHERE
    continuation_key = %s
    AND continuation_kind = %s
    AND identity_json = %s
    AND continuation_status = 'active'
    AND state_version = %s
    AND continuation_sha256 = %s
RETURNING state_version
"""


def _identity_json(
    identity: tuple[str, ...],
) -> str:
    validated = validate_agent_continuation_identity(identity)

    return json.dumps(
        list(validated),
        ensure_ascii=False,
        allow_nan=False,
        separators=(
            ",",
            ":",
        ),
    )


class PostgreSQLAgentContinuationStore(AgentContinuationStore):
    """CAS-backed durable continuation persistence."""

    def __init__(
        self,
        *,
        pool: PostgresPool,
        codec: AgentContinuationCodec | None = None,
    ) -> None:
        self._pool = pool
        self._codec = AgentContinuationCodec() if codec is None else codec

    async def create(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        snapshot: object,
    ) -> None:
        typed = self._validate_snapshot(
            kind=kind,
            identity=identity,
            snapshot=snapshot,
        )

        payload = self._codec.dumps(
            kind=kind,
            identity=identity,
            snapshot=typed,
        )

        digest = sha256(payload.encode("utf-8")).hexdigest()

        key = agent_continuation_key(
            kind,
            identity,
        )

        identity_json = _identity_json(identity)

        async with (
            self._pool.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                _INSERT_SQL,
                (
                    key,
                    kind.value,
                    identity_json,
                    AGENT_CONTINUATION_FORMAT_VERSION,
                    payload,
                    digest,
                ),
            )

            row = await cursor.fetchone()

        if row is None:
            raise AgentContinuationConflictError(
                "continuation already exists or was previously consumed"
            )

    async def load(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
    ) -> object | None:
        row = await self._load_row(
            kind=kind,
            identity=identity,
        )

        if row is None:
            return None

        (
            status,
            _version,
            _digest,
            snapshot,
        ) = self._decode_row(
            row,
            expected_kind=kind,
            expected_identity=identity,
        )

        if status == "consumed":
            return None

        return snapshot

    async def replace(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        expected: object,
        snapshot: object,
    ) -> None:
        typed_expected = self._validate_snapshot(
            kind=kind,
            identity=identity,
            snapshot=expected,
        )

        typed_snapshot = self._validate_snapshot(
            kind=kind,
            identity=identity,
            snapshot=snapshot,
        )

        row = await self._load_row(
            kind=kind,
            identity=identity,
        )

        if row is None:
            raise AgentContinuationConflictError("continuation is not active")

        (
            status,
            version,
            current_digest,
            current_snapshot,
        ) = self._decode_row(
            row,
            expected_kind=kind,
            expected_identity=identity,
        )

        if status != "active" or current_snapshot != typed_expected:
            raise AgentContinuationConflictError(
                "continuation changed or is not active"
            )

        payload = self._codec.dumps(
            kind=kind,
            identity=identity,
            snapshot=typed_snapshot,
        )

        digest = sha256(payload.encode("utf-8")).hexdigest()

        key = agent_continuation_key(
            kind,
            identity,
        )

        identity_json = _identity_json(identity)

        async with (
            self._pool.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                _REPLACE_SQL,
                (
                    payload,
                    digest,
                    key,
                    kind.value,
                    identity_json,
                    version,
                    current_digest,
                ),
            )

            updated = await cursor.fetchone()

        if updated is None:
            raise AgentContinuationConflictError(
                "continuation replacement lost compare-and-swap"
            )

    async def consume(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        expected: object,
    ) -> None:
        typed_expected = self._validate_snapshot(
            kind=kind,
            identity=identity,
            snapshot=expected,
        )

        row = await self._load_row(
            kind=kind,
            identity=identity,
        )

        if row is None:
            raise AgentContinuationConflictError("continuation is not active")

        (
            status,
            version,
            current_digest,
            current_snapshot,
        ) = self._decode_row(
            row,
            expected_kind=kind,
            expected_identity=identity,
        )

        if status != "active" or current_snapshot != typed_expected:
            raise AgentContinuationConflictError(
                "continuation changed or is not active"
            )

        key = agent_continuation_key(
            kind,
            identity,
        )

        identity_json = _identity_json(identity)

        async with (
            self._pool.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                _CONSUME_SQL,
                (
                    key,
                    kind.value,
                    identity_json,
                    version,
                    current_digest,
                ),
            )

            consumed = await cursor.fetchone()

        if consumed is None:
            raise AgentContinuationConflictError(
                "continuation consumption lost compare-and-swap"
            )

    async def _load_row(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
    ) -> tuple[object, ...] | None:
        key = agent_continuation_key(
            kind,
            identity,
        )

        async with (
            self._pool.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                _SELECT_SQL,
                (key,),
            )

            row = await cursor.fetchone()

        return row

    def _decode_row(
        self,
        row: tuple[object, ...],
        *,
        expected_kind: AgentContinuationKind,
        expected_identity: tuple[str, ...],
    ) -> tuple[
        str,
        int,
        str,
        AgentContinuationSnapshot,
    ]:
        if len(row) != 8:
            raise AgentContinuationIntegrityError(
                "stored continuation row shape is invalid"
            )

        (
            raw_key,
            raw_kind,
            raw_identity_json,
            raw_status,
            raw_format,
            raw_version,
            raw_payload,
            raw_digest,
        ) = row

        if not isinstance(
            raw_key,
            str,
        ):
            raise AgentContinuationIntegrityError("stored continuation key is invalid")

        if not isinstance(
            raw_kind,
            str,
        ):
            raise AgentContinuationIntegrityError("stored continuation kind is invalid")

        if not isinstance(
            raw_identity_json,
            str,
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation identity is invalid"
            )

        if not isinstance(
            raw_status,
            str,
        ) or raw_status not in {
            "active",
            "consumed",
        }:
            raise AgentContinuationIntegrityError(
                "stored continuation status is invalid"
            )

        if (
            not isinstance(
                raw_format,
                int,
            )
            or isinstance(
                raw_format,
                bool,
            )
            or raw_format != AGENT_CONTINUATION_FORMAT_VERSION
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation format is invalid"
            )

        if (
            not isinstance(
                raw_version,
                int,
            )
            or isinstance(
                raw_version,
                bool,
            )
            or raw_version < 1
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation version is invalid"
            )

        if not isinstance(
            raw_payload,
            str,
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation payload is invalid"
            )

        if (
            not isinstance(
                raw_digest,
                str,
            )
            or _SHA256_PATTERN.fullmatch(raw_digest) is None
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation checksum is invalid"
            )

        expected_key = agent_continuation_key(
            expected_kind,
            expected_identity,
        )

        expected_identity_json = _identity_json(expected_identity)

        if (
            raw_key != expected_key
            or raw_kind != expected_kind.value
            or raw_identity_json != expected_identity_json
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation metadata mismatch"
            )

        computed = sha256(raw_payload.encode("utf-8")).hexdigest()

        if not hmac.compare_digest(
            computed,
            raw_digest,
        ):
            raise AgentContinuationIntegrityError(
                "stored continuation checksum mismatch"
            )

        try:
            (
                decoded_kind,
                decoded_identity,
                snapshot,
            ) = self._codec.loads(raw_payload)
        except (
            AgentContinuationSerializationError,
            TypeError,
            ValueError,
        ) as exc:
            raise AgentContinuationIntegrityError(
                "stored continuation payload is invalid"
            ) from exc

        if decoded_kind is not expected_kind or decoded_identity != expected_identity:
            raise AgentContinuationIntegrityError(
                "stored continuation payload metadata mismatch"
            )

        canonical = self._codec.dumps(
            kind=decoded_kind,
            identity=decoded_identity,
            snapshot=snapshot,
        )

        if canonical != raw_payload:
            raise AgentContinuationIntegrityError(
                "stored continuation payload is not canonical"
            )

        return (
            raw_status,
            raw_version,
            raw_digest,
            snapshot,
        )

    @staticmethod
    def _validate_snapshot(
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        snapshot: object,
    ) -> AgentContinuationSnapshot:
        codec = AgentContinuationCodec()

        if kind is AgentContinuationKind.TURN:
            from ai_engineering_agent_platform.services.agent import (
                AgentTurnResult,
            )

            if not isinstance(
                snapshot,
                AgentTurnResult,
            ):
                raise AgentContinuationIntegrityError(
                    "turn continuation snapshot has invalid type"
                )

            typed: AgentContinuationSnapshot = snapshot

        elif kind is AgentContinuationKind.LOOP:
            from ai_engineering_agent_platform.services.agent_loop import (
                AgentLoopResult,
            )

            if not isinstance(
                snapshot,
                AgentLoopResult,
            ):
                raise AgentContinuationIntegrityError(
                    "loop continuation snapshot has invalid type"
                )

            typed = snapshot

        else:
            raise AgentContinuationIntegrityError("continuation kind is unsupported")

        codec.dumps(
            kind=kind,
            identity=identity,
            snapshot=typed,
        )

        return typed
