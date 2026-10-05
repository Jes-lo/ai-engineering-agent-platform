"""Unit tests for PostgreSQL durable approval storage."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest

from ai_engineering_agent_platform.adapters.postgres.approval_store import (
    PostgreSQLApprovalStore,
)
from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.services.approval import (
    APPROVAL_DECISION_SCOPE,
    ApprovalConflictError,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuthenticatedApprovalActor,
    apply_decision_transition,
)
from ai_engineering_agent_platform.services.approval_persistence import (
    APPROVAL_RECORD_FORMAT_VERSION,
    ApprovalPersistenceIntegrityError,
    ApprovalRecordCodec,
)

NOW = datetime(
    2026,
    10,
    4,
    6,
    15,
    tzinfo=UTC,
)


@dataclass
class FakeCursor:
    row: tuple[object, ...] | None

    async def fetchone(
        self,
    ) -> tuple[object, ...] | None:
        return self.row


@dataclass
class FakeConnection:
    rows: list[tuple[object, ...] | None] = field(default_factory=list)
    calls: list[
        tuple[
            str,
            tuple[object, ...],
        ]
    ] = field(default_factory=list)

    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...],
    ) -> FakeCursor:
        self.calls.append(
            (
                query,
                parameters,
            )
        )

        row = self.rows.pop(0) if self.rows else None

        return FakeCursor(row)


@dataclass
class FakeConnectionContext:
    connection: FakeConnection

    async def __aenter__(
        self,
    ) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        del exc_type
        del exc
        del traceback


@dataclass
class FakePool:
    connection_value: FakeConnection

    def connection(
        self,
    ) -> FakeConnectionContext:
        return FakeConnectionContext(self.connection_value)


def _actor() -> AuthenticatedApprovalActor:
    return AuthenticatedApprovalActor(
        subject="human-1",
        client_id="portfolio-console",
        authentication_method="oidc",
        scopes=(APPROVAL_DECISION_SCOPE,),
    )


def _invocation() -> ToolInvocation:
    return ToolInvocation(
        call_id="agent:run-1:tool:1:nonce-1",
        tool_name="dangerous_change",
        arguments=(
            ToolArgument(
                name="value",
                value=7,
            ),
        ),
    )


def _pending() -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="approval:postgres",
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=_invocation(),
        status=ApprovalStatus.PENDING,
        requested_at=NOW,
        expires_at=(NOW + timedelta(hours=1)),
    )


def _approved() -> ApprovalRequest:
    return apply_decision_transition(
        _pending(),
        expected_version=1,
        decision=ApprovalDecision.APPROVE,
        actor=_actor(),
        decided_at=NOW,
        reason="reviewed",
    )


def _stored_row(
    request: ApprovalRequest,
) -> tuple[object, ...]:
    codec = ApprovalRecordCodec()

    payload = codec.dumps(request)

    digest = sha256(payload.encode("utf-8")).hexdigest()

    return (
        request.approval_id,
        request.run_id,
        request.workflow_id,
        request.workflow_version,
        request.step_id,
        request.invocation.call_id,
        request.invocation.tool_name,
        request.status.value,
        APPROVAL_RECORD_FORMAT_VERSION,
        request.version,
        payload,
        digest,
        request.requested_at,
        request.expires_at,
    )


def _store(
    connection: FakeConnection,
) -> PostgreSQLApprovalStore:
    return PostgreSQLApprovalStore(
        FakePool(connection)  # type: ignore[arg-type]
    )


@pytest.mark.anyio
async def test_create_uses_insert_only_and_no_delete() -> None:
    request = _pending()

    connection = FakeConnection(
        rows=[
            (request.approval_id,),
        ]
    )

    await _store(connection).create(request)

    assert len(connection.calls) == 1

    query, parameters = connection.calls[0]

    assert "INSERT INTO ai_platform.approval_requests" in query
    assert "ON CONFLICT (approval_id)" in query
    assert "DO NOTHING" in query
    assert "DELETE" not in query

    assert parameters[0] == request.approval_id

    assert parameters[7] == ApprovalStatus.PENDING.value

    assert parameters[8] == APPROVAL_RECORD_FORMAT_VERSION


@pytest.mark.anyio
async def test_duplicate_create_fails_closed() -> None:
    connection = FakeConnection(
        rows=[
            None,
        ]
    )

    with pytest.raises(
        ApprovalConflictError,
        match="already exists",
    ):
        await _store(connection).create(_pending())


@pytest.mark.anyio
async def test_load_reconstructs_valid_request() -> None:
    request = _approved()

    connection = FakeConnection(
        rows=[
            _stored_row(request),
        ]
    )

    restored = await _store(connection).load(request.approval_id)

    assert restored == request


@pytest.mark.anyio
async def test_load_missing_returns_none() -> None:
    connection = FakeConnection(
        rows=[
            None,
        ]
    )

    restored = await _store(connection).load("approval:missing")

    assert restored is None


@pytest.mark.anyio
async def test_load_detects_checksum_corruption() -> None:
    request = _pending()

    row = list(_stored_row(request))

    row[10] = "{}"

    connection = FakeConnection(
        rows=[
            tuple(row),
        ]
    )

    with pytest.raises(
        ApprovalPersistenceIntegrityError,
        match="checksum mismatch",
    ):
        await _store(connection).load(request.approval_id)


@pytest.mark.anyio
async def test_load_detects_metadata_mismatch() -> None:
    request = _pending()

    row = list(_stored_row(request))

    row[1] = "run-tampered"

    connection = FakeConnection(
        rows=[
            tuple(row),
        ]
    )

    with pytest.raises(
        ApprovalPersistenceIntegrityError,
        match="metadata mismatch",
    ):
        await _store(connection).load(request.approval_id)


@pytest.mark.anyio
async def test_decide_uses_pending_version_cas() -> None:
    pending = _pending()

    connection = FakeConnection(
        rows=[
            _stored_row(pending),
            (pending.approval_id,),
        ]
    )

    updated = await _store(connection).decide(
        approval_id=pending.approval_id,
        expected_version=1,
        decision=ApprovalDecision.APPROVE,
        actor=_actor(),
        decided_at=NOW,
        reason="reviewed",
    )

    assert updated.status is ApprovalStatus.APPROVED
    assert updated.decision is ApprovalDecision.APPROVE
    assert updated.version == 2

    assert len(connection.calls) == 2

    query, parameters = connection.calls[1]

    assert "UPDATE ai_platform.approval_requests" in query
    assert "state_version = %s" in query
    assert "approval_status = %s" in query
    assert "DELETE" not in query

    assert parameters[5] == 1

    assert parameters[6] == ApprovalStatus.PENDING.value


@pytest.mark.anyio
async def test_decide_rejects_concurrent_transition() -> None:
    pending = _pending()

    connection = FakeConnection(
        rows=[
            _stored_row(pending),
            None,
        ]
    )

    with pytest.raises(
        ApprovalConflictError,
        match="optimistic concurrency",
    ):
        await _store(connection).decide(
            approval_id=pending.approval_id,
            expected_version=1,
            decision=ApprovalDecision.APPROVE,
            actor=_actor(),
            decided_at=NOW,
            reason=None,
        )


@pytest.mark.anyio
async def test_consume_uses_approved_version_cas() -> None:
    approved = _approved()

    connection = FakeConnection(
        rows=[
            _stored_row(approved),
            (approved.approval_id,),
        ]
    )

    consumed = await _store(connection).consume(
        approval_id=approved.approval_id,
        expected_version=2,
        consumed_at=(NOW + timedelta(seconds=1)),
    )

    assert consumed.status is ApprovalStatus.CONSUMED
    assert consumed.decision is ApprovalDecision.APPROVE
    assert consumed.version == 3

    query, parameters = connection.calls[1]

    assert parameters[5] == 2

    assert parameters[6] == ApprovalStatus.APPROVED.value

    assert "DELETE" not in query


def test_invalid_portable_approval_id_fails_before_sql() -> None:
    connection = FakeConnection()

    with pytest.raises(
        ValueError,
        match="portable approval identifier",
    ):
        asyncio.run(_store(connection).load(" approval invalid "))

    assert connection.calls == []


@pytest.mark.anyio
async def test_load_maps_semantically_invalid_payload_to_integrity_error() -> None:
    """A recomputed checksum cannot make invalid lifecycle JSON trusted."""
    request = _pending()

    row = list(_stored_row(request))

    payload = row[10]

    assert isinstance(
        payload,
        str,
    )

    tampered = payload.replace(
        '"decision":null',
        '"decision":"not-a-decision"',
        1,
    )

    assert tampered != payload

    row[10] = tampered
    row[11] = sha256(tampered.encode("utf-8")).hexdigest()

    connection = FakeConnection(
        rows=[
            tuple(row),
        ]
    )

    with pytest.raises(
        ApprovalPersistenceIntegrityError,
        match="stored approval payload is invalid",
    ):
        await _store(connection).load(request.approval_id)
