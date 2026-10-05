"""Tests for authenticated, expiring, single-use human approval."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.services.approval import (
    APPROVAL_DECISION_SCOPE,
    ApprovalAuthorizationError,
    ApprovalConflictError,
    ApprovalDecision,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    ApprovalRequest,
    ApprovalService,
    ApprovalStateError,
    ApprovalStatus,
    ApprovalStore,
    AuthenticatedApprovalActor,
    apply_consumption_transition,
    apply_decision_transition,
)

NOW = datetime(
    2026,
    10,
    4,
    5,
    30,
    tzinfo=UTC,
)


def _actor(
    *,
    subject: str = "human-1",
    scopes: tuple[str, ...] = (APPROVAL_DECISION_SCOPE,),
) -> AuthenticatedApprovalActor:
    return AuthenticatedApprovalActor(
        subject=subject,
        client_id="portfolio-console",
        authentication_method="oidc",
        scopes=scopes,
    )


def _invocation(
    *,
    call_id: str = "agent:run-1:tool:1:nonce-1",
    value: int = 7,
) -> ToolInvocation:
    return ToolInvocation(
        call_id=call_id,
        tool_name="dangerous_change",
        arguments=(
            ToolArgument(
                name="value",
                value=value,
            ),
        ),
    )


@dataclass
class MemoryApprovalStore(ApprovalStore):
    records: dict[str, ApprovalRequest] = field(default_factory=dict)

    async def create(
        self,
        request: ApprovalRequest,
    ) -> None:
        if request.approval_id in self.records:
            raise ApprovalConflictError("approval already exists")

        self.records[request.approval_id] = request

    async def load(
        self,
        approval_id: str,
    ) -> ApprovalRequest | None:
        return self.records.get(approval_id)

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
        current = self.records[approval_id]

        updated = apply_decision_transition(
            current,
            expected_version=expected_version,
            decision=decision,
            actor=actor,
            decided_at=decided_at,
            reason=reason,
        )

        self.records[approval_id] = updated

        return updated

    async def consume(
        self,
        *,
        approval_id: str,
        expected_version: int,
        consumed_at: datetime,
    ) -> ApprovalRequest:
        current = self.records[approval_id]

        updated = apply_consumption_transition(
            current,
            expected_version=expected_version,
            consumed_at=consumed_at,
        )

        self.records[approval_id] = updated

        return updated


def _service(
    store: MemoryApprovalStore,
    *,
    clock_value: datetime = NOW,
) -> ApprovalService:
    return ApprovalService(
        store=store,
        approval_id_factory=(lambda: "approval:test-1"),
        clock=lambda: clock_value,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "subject": "",
        },
        {
            "subject": " human ",
        },
        {
            "client_id": "",
        },
        {
            "authentication_method": "",
        },
        {
            "scopes": (
                APPROVAL_DECISION_SCOPE,
                APPROVAL_DECISION_SCOPE,
            ),
        },
    ),
)
def test_actor_snapshot_rejects_invalid_identity(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "subject": "human-1",
        "client_id": "portfolio-console",
        "authentication_method": "oidc",
        "scopes": (APPROVAL_DECISION_SCOPE,),
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        AuthenticatedApprovalActor(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_request_freezes_exact_tool_invocation() -> None:
    store = MemoryApprovalStore()
    service = _service(store)
    invocation = _invocation()

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=invocation,
        ttl_seconds=600,
    )

    assert request.status is ApprovalStatus.PENDING
    assert request.version == 1
    assert request.invocation == invocation
    assert request.requested_at == NOW
    assert request.expires_at == (NOW + timedelta(seconds=600))
    assert request.decided_by is None
    assert request.consumed_at is None
    assert store.records[request.approval_id] == request


@pytest.mark.anyio
async def test_unknown_approval_fails_closed() -> None:
    service = _service(MemoryApprovalStore())

    with pytest.raises(ApprovalNotFoundError):
        await service.get("approval:unknown")


@pytest.mark.anyio
async def test_decision_requires_authenticated_scope() -> None:
    store = MemoryApprovalStore()
    service = _service(store)

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=_invocation(),
    )

    with pytest.raises(
        ApprovalAuthorizationError,
        match="lacks approval decision scope",
    ):
        await service.decide(
            request.approval_id,
            actor=_actor(scopes=("tool:read",)),
            decision=ApprovalDecision.APPROVE,
        )

    assert store.records[request.approval_id].status is ApprovalStatus.PENDING


@pytest.mark.anyio
async def test_expired_pending_request_cannot_be_approved() -> None:
    store = MemoryApprovalStore()

    service = ApprovalService(
        store=store,
        approval_id_factory=(lambda: "approval:expired"),
        clock=lambda: NOW,
    )

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=_invocation(),
        ttl_seconds=1,
    )

    expired_service = ApprovalService(
        store=store,
        clock=lambda: NOW + timedelta(seconds=1),
    )

    with pytest.raises(ApprovalExpiredError):
        await expired_service.decide(
            request.approval_id,
            actor=_actor(),
            decision=ApprovalDecision.APPROVE,
        )

    assert store.records[request.approval_id].status is ApprovalStatus.PENDING


@pytest.mark.anyio
async def test_approved_request_mints_exact_single_use_grant() -> None:
    store = MemoryApprovalStore()
    service = _service(store)
    invocation = _invocation()

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=invocation,
    )

    approved = await service.decide(
        request.approval_id,
        actor=_actor(),
        decision=ApprovalDecision.APPROVE,
        reason="Reviewed intended change.",
    )

    assert approved.status is ApprovalStatus.APPROVED
    assert approved.decision is ApprovalDecision.APPROVE
    assert approved.version == 2
    assert approved.decided_by == _actor()
    assert approved.decided_at == NOW
    assert approved.decision_reason == "Reviewed intended change."

    grant = await service.consume_grant(
        request.approval_id,
        invocation=invocation,
    )

    assert grant.call_id == invocation.call_id
    assert grant.tool_name == invocation.tool_name

    consumed = store.records[request.approval_id]

    assert consumed.status is ApprovalStatus.CONSUMED
    assert consumed.decision is ApprovalDecision.APPROVE
    assert consumed.version == 3
    assert consumed.consumed_at == NOW

    with pytest.raises(
        ApprovalStateError,
        match="already been consumed",
    ):
        await service.consume_grant(
            request.approval_id,
            invocation=invocation,
        )


@pytest.mark.anyio
async def test_rejection_never_mints_execution_grant() -> None:
    store = MemoryApprovalStore()
    service = _service(store)
    invocation = _invocation()

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=invocation,
    )

    rejected = await service.decide(
        request.approval_id,
        actor=_actor(),
        decision=ApprovalDecision.REJECT,
        reason="Change not authorized.",
    )

    assert rejected.status is ApprovalStatus.REJECTED
    assert rejected.decision is ApprovalDecision.REJECT

    with pytest.raises(
        ApprovalStateError,
        match="rejected approval",
    ):
        await service.consume_grant(
            request.approval_id,
            invocation=invocation,
        )


@pytest.mark.anyio
async def test_approved_request_is_bound_to_frozen_arguments() -> None:
    store = MemoryApprovalStore()
    service = _service(store)

    original = _invocation(value=7)

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=original,
    )

    await service.decide(
        request.approval_id,
        actor=_actor(),
        decision=ApprovalDecision.APPROVE,
    )

    forged = _invocation(
        call_id=original.call_id,
        value=999,
    )

    with pytest.raises(
        ApprovalStateError,
        match="differs from frozen",
    ):
        await service.consume_grant(
            request.approval_id,
            invocation=forged,
        )

    assert store.records[request.approval_id].status is ApprovalStatus.APPROVED


@pytest.mark.anyio
async def test_approved_request_is_bound_to_exact_call_identity() -> None:
    store = MemoryApprovalStore()
    service = _service(store)

    original = _invocation()

    request = await service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=original,
    )

    await service.decide(
        request.approval_id,
        actor=_actor(),
        decision=ApprovalDecision.APPROVE,
    )

    with pytest.raises(
        ApprovalStateError,
        match="differs from frozen",
    ):
        await service.consume_grant(
            request.approval_id,
            invocation=_invocation(call_id="agent:other:tool:1:nonce-2"),
        )


@pytest.mark.anyio
async def test_approved_request_can_expire_before_consumption() -> None:
    store = MemoryApprovalStore()

    creation_service = ApprovalService(
        store=store,
        approval_id_factory=(lambda: "approval:short"),
        clock=lambda: NOW,
    )

    invocation = _invocation()

    request = await creation_service.request(
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=invocation,
        ttl_seconds=2,
    )

    approved = await creation_service.decide(
        request.approval_id,
        actor=_actor(),
        decision=ApprovalDecision.APPROVE,
    )

    assert approved.status is ApprovalStatus.APPROVED

    expired_service = ApprovalService(
        store=store,
        clock=lambda: NOW + timedelta(seconds=2),
    )

    with pytest.raises(ApprovalExpiredError):
        await expired_service.consume_grant(
            request.approval_id,
            invocation=invocation,
        )


def test_decision_transition_rejects_stale_version() -> None:
    pending = ApprovalRequest(
        approval_id="approval:version",
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="step-1",
        invocation=_invocation(),
        status=ApprovalStatus.PENDING,
        requested_at=NOW,
        expires_at=(NOW + timedelta(hours=1)),
    )

    with pytest.raises(
        ApprovalConflictError,
        match="version changed",
    ):
        apply_decision_transition(
            pending,
            expected_version=2,
            decision=ApprovalDecision.APPROVE,
            actor=_actor(),
            decided_at=NOW,
            reason=None,
        )


def test_consumption_transition_rejects_nonapproved_state() -> None:
    pending = ApprovalRequest(
        approval_id="approval:not-approved",
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="step-1",
        invocation=_invocation(),
        status=ApprovalStatus.PENDING,
        requested_at=NOW,
        expires_at=(NOW + timedelta(hours=1)),
    )

    with pytest.raises(
        ApprovalConflictError,
        match="no longer consumable",
    ):
        apply_consumption_transition(
            pending,
            expected_version=1,
            consumed_at=NOW,
        )


@pytest.mark.parametrize(
    "ttl",
    (
        0,
        -1,
        86401,
        True,
    ),
)
@pytest.mark.anyio
async def test_request_rejects_invalid_ttl(
    ttl: object,
) -> None:
    service = _service(MemoryApprovalStore())

    with pytest.raises(
        ValueError,
        match="ttl_seconds",
    ):
        await service.request(
            run_id="run-1",
            workflow_id="workflow-1",
            workflow_version="1",
            step_id="step-1",
            invocation=_invocation(),
            ttl_seconds=ttl,  # type: ignore[arg-type]
        )


def test_pending_state_cannot_smuggle_decision_identity() -> None:
    with pytest.raises(
        ValueError,
        match="pending approval",
    ):
        ApprovalRequest(
            approval_id="approval:smuggled",
            run_id="run-1",
            workflow_id="workflow-1",
            workflow_version="1",
            step_id="step-1",
            invocation=_invocation(),
            status=ApprovalStatus.PENDING,
            requested_at=NOW,
            expires_at=(NOW + timedelta(hours=1)),
            decided_by=_actor(),
        )


def test_actor_snapshot_does_not_store_bearer_credentials() -> None:
    actor_fields = {field_name for field_name in (_actor().__dataclass_fields__)}

    assert "token" not in actor_fields
    assert "credential" not in actor_fields
    assert "authorization" not in actor_fields


def test_consumed_state_requires_prior_approve_decision() -> None:
    """Consumed state cannot erase or forge its human decision provenance."""
    with pytest.raises(
        ValueError,
        match="consumed status requires prior approve decision",
    ):
        ApprovalRequest(
            approval_id="approval:bad-consumed",
            run_id="run-1",
            workflow_id="workflow-1",
            workflow_version="1",
            step_id="step-1",
            invocation=_invocation(),
            status=ApprovalStatus.CONSUMED,
            requested_at=NOW,
            expires_at=(NOW + timedelta(hours=1)),
            version=3,
            decision=ApprovalDecision.REJECT,
            decided_by=_actor(),
            decided_at=NOW,
            consumed_at=NOW,
        )


def test_decided_state_rejects_decision_at_expiration() -> None:
    """Persisted human decisions must happen before approval expiration."""
    with pytest.raises(
        ValueError,
        match="decided_at must precede expires_at",
    ):
        ApprovalRequest(
            approval_id="approval:late-decision",
            run_id="run-1",
            workflow_id="workflow-1",
            workflow_version="1",
            step_id="step-1",
            invocation=_invocation(),
            status=ApprovalStatus.APPROVED,
            requested_at=NOW,
            expires_at=(NOW + timedelta(seconds=10)),
            version=2,
            decision=ApprovalDecision.APPROVE,
            decided_by=_actor(),
            decided_at=(NOW + timedelta(seconds=10)),
        )


def test_consumed_state_rejects_consumption_at_expiration() -> None:
    """Approved authority cannot be consumed at or after expiration."""
    with pytest.raises(
        ValueError,
        match="consumed_at must precede expires_at",
    ):
        ApprovalRequest(
            approval_id="approval:late-consume",
            run_id="run-1",
            workflow_id="workflow-1",
            workflow_version="1",
            step_id="step-1",
            invocation=_invocation(),
            status=ApprovalStatus.CONSUMED,
            requested_at=NOW,
            expires_at=(NOW + timedelta(seconds=10)),
            version=3,
            decision=ApprovalDecision.APPROVE,
            decided_by=_actor(),
            decided_at=(NOW + timedelta(seconds=5)),
            consumed_at=(NOW + timedelta(seconds=10)),
        )
