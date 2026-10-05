"""PostgreSQL adapter for durable authenticated human approvals."""

import hmac
import re
from datetime import datetime
from hashlib import sha256

from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool

from ai_engineering_agent_platform.services.approval import (
    ApprovalConflictError,
    ApprovalDecision,
    ApprovalError,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalStore,
    AuthenticatedApprovalActor,
    apply_consumption_transition,
    apply_decision_transition,
)
from ai_engineering_agent_platform.services.approval_persistence import (
    APPROVAL_RECORD_FORMAT_VERSION,
    ApprovalPersistenceError,
    ApprovalPersistenceIntegrityError,
    ApprovalRecordCodec,
)

type ApprovalPostgresPool = AsyncConnectionPool[AsyncConnection[TupleRow]]


_APPROVAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


_CREATE_SQL = """
INSERT INTO ai_platform.approval_requests (
    approval_id,
    run_id,
    workflow_id,
    workflow_version,
    step_id,
    call_id,
    tool_name,
    approval_status,
    approval_format,
    state_version,
    approval_payload,
    approval_sha256,
    requested_at,
    expires_at
)
VALUES (
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s,
    %s
)
ON CONFLICT (approval_id)
DO NOTHING
RETURNING approval_id
"""


_LOAD_SQL = """
SELECT
    approval_id,
    run_id,
    workflow_id,
    workflow_version,
    step_id,
    call_id,
    tool_name,
    approval_status,
    approval_format,
    state_version,
    approval_payload,
    approval_sha256,
    requested_at,
    expires_at
FROM ai_platform.approval_requests
WHERE approval_id = %s
"""


_UPDATE_SQL = """
UPDATE ai_platform.approval_requests
SET
    approval_status = %s,
    state_version = %s,
    approval_payload = %s,
    approval_sha256 = %s,
    updated_at = clock_timestamp()
WHERE
    approval_id = %s
    AND state_version = %s
    AND approval_status = %s
    AND run_id = %s
    AND workflow_id = %s
    AND workflow_version = %s
    AND step_id = %s
    AND call_id = %s
    AND tool_name = %s
RETURNING approval_id
"""


class PostgreSQLApprovalStore(ApprovalStore):
    """Persist canonical approval state with optimistic CAS transitions."""

    def __init__(
        self,
        pool: ApprovalPostgresPool,
        *,
        codec: ApprovalRecordCodec | None = None,
    ) -> None:
        """Store injected PostgreSQL pool and canonical codec."""
        if pool is None:
            raise ValueError("pool must not be None")

        self._pool = pool
        self._codec = ApprovalRecordCodec() if codec is None else codec

    @staticmethod
    def _validate_approval_id(
        approval_id: str,
    ) -> None:
        if not isinstance(
            approval_id,
            str,
        ) or not _APPROVAL_ID_PATTERN.fullmatch(approval_id):
            raise ValueError("approval_id must be a portable approval identifier")

    def _payload(
        self,
        request: ApprovalRequest,
    ) -> tuple[
        str,
        str,
    ]:
        payload = self._codec.dumps(request)

        digest = sha256(payload.encode("utf-8")).hexdigest()

        return (
            payload,
            digest,
        )

    async def create(
        self,
        request: ApprovalRequest,
    ) -> None:
        """Insert one pending approval exactly once."""
        if not isinstance(
            request,
            ApprovalRequest,
        ):
            raise ValueError("request must be ApprovalRequest")

        if request.status is not ApprovalStatus.PENDING:
            raise ValueError("new approval request must be pending")

        self._validate_approval_id(request.approval_id)

        payload, digest = self._payload(request)

        invocation = request.invocation

        parameters = (
            request.approval_id,
            request.run_id,
            request.workflow_id,
            request.workflow_version,
            request.step_id,
            invocation.call_id,
            invocation.tool_name,
            request.status.value,
            APPROVAL_RECORD_FORMAT_VERSION,
            request.version,
            payload,
            digest,
            request.requested_at,
            request.expires_at,
        )

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    _CREATE_SQL,
                    parameters,
                )

                row = await cursor.fetchone()

        except ApprovalError:
            raise

        except Exception as exc:
            raise ApprovalPersistenceError("PostgreSQL approval create failed") from exc

        if row is None:
            raise ApprovalConflictError("approval request already exists")

        if (
            not isinstance(
                row,
                tuple,
            )
            or len(row) != 1
            or row[0] != request.approval_id
        ):
            raise ApprovalPersistenceIntegrityError(
                "unexpected PostgreSQL approval create result"
            )

    async def load(
        self,
        approval_id: str,
    ) -> ApprovalRequest | None:
        """Load and fully revalidate one durable approval snapshot."""
        self._validate_approval_id(approval_id)

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    _LOAD_SQL,
                    (approval_id,),
                )

                row = await cursor.fetchone()

        except ApprovalError:
            raise

        except Exception as exc:
            raise ApprovalPersistenceError("PostgreSQL approval load failed") from exc

        if row is None:
            return None

        if (
            not isinstance(
                row,
                tuple,
            )
            or len(row) != 14
        ):
            raise ApprovalPersistenceIntegrityError(
                "unexpected PostgreSQL approval row shape"
            )

        (
            stored_approval_id,
            run_id,
            workflow_id,
            workflow_version,
            step_id,
            call_id,
            tool_name,
            approval_status,
            approval_format,
            state_version,
            payload,
            stored_digest,
            requested_at,
            expires_at,
        ) = row

        if (
            not isinstance(
                stored_approval_id,
                str,
            )
            or not isinstance(
                run_id,
                str,
            )
            or not isinstance(
                workflow_id,
                str,
            )
            or not isinstance(
                workflow_version,
                str,
            )
            or not isinstance(
                step_id,
                str,
            )
            or not isinstance(
                call_id,
                str,
            )
            or not isinstance(
                tool_name,
                str,
            )
            or not isinstance(
                approval_status,
                str,
            )
            or type(approval_format) is not int
            or type(state_version) is not int
            or not isinstance(
                payload,
                str,
            )
            or not isinstance(
                stored_digest,
                str,
            )
        ):
            raise ApprovalPersistenceIntegrityError(
                "invalid PostgreSQL approval field types"
            )

        if stored_approval_id != approval_id:
            raise ApprovalPersistenceIntegrityError(
                "stored approval identifier mismatch"
            )

        if approval_format != APPROVAL_RECORD_FORMAT_VERSION:
            raise ApprovalPersistenceIntegrityError("stored approval format differs")

        if state_version < 1:
            raise ApprovalPersistenceIntegrityError(
                "stored approval version is invalid"
            )

        if not _SHA256_PATTERN.fullmatch(stored_digest):
            raise ApprovalPersistenceIntegrityError("stored approval digest is invalid")

        actual_digest = sha256(payload.encode("utf-8")).hexdigest()

        if not hmac.compare_digest(
            stored_digest,
            actual_digest,
        ):
            raise ApprovalPersistenceIntegrityError("stored approval checksum mismatch")

        try:
            request = self._codec.loads(payload)
        except (
            ApprovalPersistenceError,
            TypeError,
            ValueError,
        ) as exc:
            raise ApprovalPersistenceIntegrityError(
                "stored approval payload is invalid"
            ) from exc

        invocation = request.invocation

        if (
            request.approval_id != stored_approval_id
            or request.run_id != run_id
            or request.workflow_id != workflow_id
            or request.workflow_version != workflow_version
            or request.step_id != step_id
            or invocation.call_id != call_id
            or invocation.tool_name != tool_name
            or request.status.value != approval_status
            or request.version != state_version
            or request.requested_at != requested_at
            or request.expires_at != expires_at
        ):
            raise ApprovalPersistenceIntegrityError("stored approval metadata mismatch")

        canonical = self._codec.dumps(request)

        if not hmac.compare_digest(
            payload,
            canonical,
        ):
            raise ApprovalPersistenceIntegrityError(
                "stored approval payload is not canonical"
            )

        return request

    async def _update(
        self,
        request: ApprovalRequest,
        *,
        expected_version: int,
        expected_status: ApprovalStatus,
    ) -> ApprovalRequest:
        payload, digest = self._payload(request)

        invocation = request.invocation

        parameters = (
            request.status.value,
            request.version,
            payload,
            digest,
            request.approval_id,
            expected_version,
            expected_status.value,
            request.run_id,
            request.workflow_id,
            request.workflow_version,
            request.step_id,
            invocation.call_id,
            invocation.tool_name,
        )

        try:
            async with self._pool.connection() as connection:
                cursor = await connection.execute(
                    _UPDATE_SQL,
                    parameters,
                )

                row = await cursor.fetchone()

        except ApprovalError:
            raise

        except Exception as exc:
            raise ApprovalPersistenceError(
                "PostgreSQL approval transition failed"
            ) from exc

        if row is None:
            raise ApprovalConflictError(
                "approval transition rejected by optimistic concurrency guard"
            )

        if (
            not isinstance(
                row,
                tuple,
            )
            or len(row) != 1
            or row[0] != request.approval_id
        ):
            raise ApprovalPersistenceIntegrityError(
                "unexpected PostgreSQL approval transition result"
            )

        return request

    async def decide(
        self,
        *,
        approval_id: str,
        expected_version: int,
        decision: ApprovalDecision,
        actor: AuthenticatedApprovalActor,
        decided_at: datetime,
        reason: str | None,
    ) -> ApprovalRequest:
        """Atomically CAS one pending approval to approve or reject."""
        if not isinstance(
            decided_at,
            datetime,
        ):
            raise ValueError("decided_at must be a datetime")

        current = await self.load(approval_id)

        if current is None:
            raise ApprovalConflictError("approval request no longer exists")

        updated = apply_decision_transition(
            current,
            expected_version=expected_version,
            decision=decision,
            actor=actor,
            decided_at=decided_at,
            reason=reason,
        )

        return await self._update(
            updated,
            expected_version=expected_version,
            expected_status=ApprovalStatus.PENDING,
        )

    async def consume(
        self,
        *,
        approval_id: str,
        expected_version: int,
        consumed_at: datetime,
    ) -> ApprovalRequest:
        """Atomically CAS one approved approval to consumed."""
        if not isinstance(
            consumed_at,
            datetime,
        ):
            raise ValueError("consumed_at must be a datetime")

        current = await self.load(approval_id)

        if current is None:
            raise ApprovalConflictError("approval request no longer exists")

        updated = apply_consumption_transition(
            current,
            expected_version=expected_version,
            consumed_at=consumed_at,
        )

        return await self._update(
            updated,
            expected_version=expected_version,
            expected_status=ApprovalStatus.APPROVED,
        )
