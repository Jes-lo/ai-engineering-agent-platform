"""Durable authenticated human-approval application boundary.

This module owns approval lifecycle semantics without owning authentication
transport, PostgreSQL, workflow orchestration, or tool execution.

A trusted outer authentication boundary is responsible for constructing
``AuthenticatedApprovalActor``. Merely constructing the dataclass does not
authenticate a caller.

Approved evidence is consumed before it becomes a structural
``ToolApprovalGrant``. The existing ``ToolExecutionService`` remains the final
authority that validates tool policy, invocation identity, inputs, and the
grant before any provider side effect.

Consumption is intentionally single-use. This feature does not claim
exactly-once external side effects: process loss after approval consumption
but before a provider side effect can produce an authorized-but-not-executed
outcome, and automatic retry is deliberately outside this boundary.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from ai_engineering_agent_platform.contracts import (
    ToolInvocation,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolApprovalGrant,
)

MAX_APPROVAL_TTL_SECONDS = 86400
MAX_APPROVAL_REASON_LENGTH = 2000
MAX_APPROVAL_SCOPES = 64

APPROVAL_DECISION_SCOPE = "approval:decide"


class ApprovalError(Exception):
    """Base error for controlled human approval."""


class ApprovalNotFoundError(ApprovalError):
    """Raised when an approval identifier does not exist."""


class ApprovalAuthorizationError(ApprovalError):
    """Raised when an authenticated actor lacks approval authority."""


class ApprovalStateError(ApprovalError):
    """Raised when an approval lifecycle transition is invalid."""


class ApprovalExpiredError(ApprovalStateError):
    """Raised when approval evidence is no longer valid."""


class ApprovalConflictError(ApprovalError):
    """Raised when optimistic approval state changed concurrently."""


class ApprovalDecision(StrEnum):
    """Human decision accepted by the approval boundary."""

    APPROVE = "approve"
    REJECT = "reject"


class ApprovalStatus(StrEnum):
    """Durable lifecycle state for one exact approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONSUMED = "consumed"


def _require_text(
    name: str,
    value: object,
    *,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")

    if value != value.strip() or not value:
        raise ValueError(f"{name} must be a non-empty trimmed string")

    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length")

    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains control characters")

    return value


def _require_aware_datetime(
    name: str,
    value: object,
) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")

    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _approval_id() -> str:
    return "approval:" + uuid4().hex


@dataclass(frozen=True, slots=True)
class AuthenticatedApprovalActor:
    """Privacy-minimized authenticated actor snapshot.

    This is application data, not an authentication mechanism. Only a trusted
    authenticated outer boundary may construct it for ApprovalService use.
    Raw bearer credentials and tokens must never be stored here.
    """

    subject: str
    client_id: str
    authentication_method: str
    scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate bounded identity evidence without accepting credentials."""
        _require_text(
            "subject",
            self.subject,
            max_length=160,
        )
        _require_text(
            "client_id",
            self.client_id,
            max_length=160,
        )
        _require_text(
            "authentication_method",
            self.authentication_method,
            max_length=80,
        )

        if not isinstance(
            self.scopes,
            tuple,
        ):
            raise ValueError("scopes must be a tuple")

        if len(self.scopes) > MAX_APPROVAL_SCOPES:
            raise ValueError("too many actor scopes")

        for scope in self.scopes:
            _require_text(
                "scope",
                scope,
                max_length=160,
            )

        if len(self.scopes) != len(set(self.scopes)):
            raise ValueError("actor scopes must be unique")

    def has_scope(
        self,
        scope: str,
    ) -> bool:
        """Return whether exact authenticated authority is present."""
        return scope in self.scopes


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Immutable snapshot of approval state for one frozen tool invocation."""

    approval_id: str
    run_id: str
    workflow_id: str
    workflow_version: str
    step_id: str
    invocation: ToolInvocation
    status: ApprovalStatus
    requested_at: datetime
    expires_at: datetime
    version: int = 1
    decision: ApprovalDecision | None = None
    decided_by: AuthenticatedApprovalActor | None = None
    decided_at: datetime | None = None
    decision_reason: str | None = None
    consumed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Enforce lifecycle and identity invariants fail closed."""
        _require_text(
            "approval_id",
            self.approval_id,
            max_length=80,
        )
        _require_text(
            "run_id",
            self.run_id,
            max_length=80,
        )
        _require_text(
            "workflow_id",
            self.workflow_id,
            max_length=80,
        )
        _require_text(
            "workflow_version",
            self.workflow_version,
            max_length=80,
        )
        _require_text(
            "step_id",
            self.step_id,
            max_length=80,
        )

        if not isinstance(
            self.invocation,
            ToolInvocation,
        ):
            raise ValueError("invocation must be ToolInvocation")

        if not isinstance(
            self.status,
            ApprovalStatus,
        ):
            raise ValueError("status must be ApprovalStatus")

        requested_at = _require_aware_datetime(
            "requested_at",
            self.requested_at,
        )
        expires_at = _require_aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if expires_at <= requested_at:
            raise ValueError("expires_at must be after requested_at")

        if (
            not isinstance(
                self.version,
                int,
            )
            or isinstance(
                self.version,
                bool,
            )
            or self.version < 1
        ):
            raise ValueError("version must be a positive integer")

        if self.decision_reason is not None:
            _require_text(
                "decision_reason",
                self.decision_reason,
                max_length=MAX_APPROVAL_REASON_LENGTH,
            )

        if self.status is ApprovalStatus.PENDING:
            if any(
                value is not None
                for value in (
                    self.decision,
                    self.decided_by,
                    self.decided_at,
                    self.decision_reason,
                    self.consumed_at,
                )
            ):
                raise ValueError(
                    "pending approval must not contain decision or consumption data"
                )
            return

        if not isinstance(
            self.decision,
            ApprovalDecision,
        ):
            raise ValueError("decided approval requires explicit decision")

        if (
            self.status is ApprovalStatus.APPROVED
            and self.decision is not ApprovalDecision.APPROVE
        ):
            raise ValueError("approved status requires approve decision")

        if (
            self.status is ApprovalStatus.REJECTED
            and self.decision is not ApprovalDecision.REJECT
        ):
            raise ValueError("rejected status requires reject decision")

        if (
            self.status is ApprovalStatus.CONSUMED
            and self.decision is not ApprovalDecision.APPROVE
        ):
            raise ValueError("consumed status requires prior approve decision")

        if not isinstance(
            self.decided_by,
            AuthenticatedApprovalActor,
        ):
            raise ValueError("decided approval requires authenticated actor snapshot")

        decided_at = _require_aware_datetime(
            "decided_at",
            self.decided_at,
        )

        if decided_at < requested_at:
            raise ValueError("decided_at must not precede requested_at")

        if decided_at >= expires_at:
            raise ValueError("decided_at must precede expires_at")

        if self.status in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
        ):
            if self.consumed_at is not None:
                raise ValueError(
                    "unconsumed terminal decision must not contain consumed_at"
                )
            return

        if self.status is ApprovalStatus.CONSUMED:
            consumed_at = _require_aware_datetime(
                "consumed_at",
                self.consumed_at,
            )

            if consumed_at < decided_at:
                raise ValueError("consumed_at must not precede decided_at")
            if consumed_at >= expires_at:
                raise ValueError("consumed_at must precede expires_at")

            return

        raise ValueError("unsupported approval status")

    def is_expired(
        self,
        at: datetime,
    ) -> bool:
        """Return whether the approval validity window has closed."""
        checked = _require_aware_datetime(
            "at",
            at,
        )
        return checked >= self.expires_at


class ApprovalStore(Protocol):
    """Persistence boundary for optimistic durable approval state."""

    async def create(
        self,
        request: ApprovalRequest,
    ) -> None:
        """Persist one new pending approval exactly once."""
        ...

    async def load(
        self,
        approval_id: str,
    ) -> ApprovalRequest | None:
        """Return the current approval snapshot when present."""
        ...

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
        """Atomically transition one pending request to a human decision."""
        ...

    async def consume(
        self,
        *,
        approval_id: str,
        expected_version: int,
        consumed_at: datetime,
    ) -> ApprovalRequest:
        """Atomically consume one approved request exactly once."""
        ...


@dataclass(frozen=True, slots=True)
class ApprovalService:
    """Coordinate authenticated decisions and exact single-use grants."""

    store: ApprovalStore
    approval_id_factory: Callable[[], str] = _approval_id
    clock: Callable[[], datetime] = _utc_now
    required_decision_scope: str = APPROVAL_DECISION_SCOPE
    max_ttl_seconds: int = MAX_APPROVAL_TTL_SECONDS

    def __post_init__(self) -> None:
        """Validate injected trusted dependencies and policy bounds."""
        if self.store is None:
            raise ValueError("store must not be None")

        if not callable(self.approval_id_factory):
            raise ValueError("approval_id_factory must be callable")

        if not callable(self.clock):
            raise ValueError("clock must be callable")

        _require_text(
            "required_decision_scope",
            self.required_decision_scope,
            max_length=160,
        )

        if (
            not isinstance(
                self.max_ttl_seconds,
                int,
            )
            or isinstance(
                self.max_ttl_seconds,
                bool,
            )
            or self.max_ttl_seconds < 1
            or self.max_ttl_seconds > MAX_APPROVAL_TTL_SECONDS
        ):
            raise ValueError("max_ttl_seconds is outside supported approval bound")

    def _now(
        self,
    ) -> datetime:
        return _require_aware_datetime(
            "clock result",
            self.clock(),
        )

    async def request(
        self,
        *,
        run_id: str,
        workflow_id: str,
        workflow_version: str,
        step_id: str,
        invocation: ToolInvocation,
        ttl_seconds: int = 3600,
    ) -> ApprovalRequest:
        """Create durable approval state for one exact frozen invocation."""
        if (
            not isinstance(
                ttl_seconds,
                int,
            )
            or isinstance(
                ttl_seconds,
                bool,
            )
            or ttl_seconds < 1
            or ttl_seconds > self.max_ttl_seconds
        ):
            raise ValueError("ttl_seconds is outside configured approval bound")

        now = self._now()

        request = ApprovalRequest(
            approval_id=_require_text(
                "approval_id",
                self.approval_id_factory(),
                max_length=80,
            ),
            run_id=run_id,
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            step_id=step_id,
            invocation=invocation,
            status=ApprovalStatus.PENDING,
            requested_at=now,
            expires_at=(now + timedelta(seconds=ttl_seconds)),
        )

        await self.store.create(request)

        return request

    async def get(
        self,
        approval_id: str,
    ) -> ApprovalRequest:
        """Load one approval or fail closed when unknown."""
        _require_text(
            "approval_id",
            approval_id,
            max_length=80,
        )

        request = await self.store.load(approval_id)

        if request is None:
            raise ApprovalNotFoundError("approval request does not exist")

        return request

    async def decide(
        self,
        approval_id: str,
        *,
        actor: AuthenticatedApprovalActor,
        decision: ApprovalDecision,
        reason: str | None = None,
    ) -> ApprovalRequest:
        """Apply one authenticated human decision to pending approval."""
        if not isinstance(
            actor,
            AuthenticatedApprovalActor,
        ):
            raise ApprovalAuthorizationError(
                "approval decision requires authenticated actor snapshot"
            )

        if not actor.has_scope(self.required_decision_scope):
            raise ApprovalAuthorizationError(
                "authenticated actor lacks approval decision scope"
            )

        if not isinstance(
            decision,
            ApprovalDecision,
        ):
            raise ValueError("decision must be ApprovalDecision")

        if reason is not None:
            _require_text(
                "reason",
                reason,
                max_length=MAX_APPROVAL_REASON_LENGTH,
            )

        current = await self.get(approval_id)

        if current.status is not ApprovalStatus.PENDING:
            raise ApprovalStateError("approval request is not pending")

        now = self._now()

        if current.is_expired(now):
            raise ApprovalExpiredError("approval request has expired")

        return await self.store.decide(
            approval_id=current.approval_id,
            expected_version=current.version,
            decision=decision,
            actor=actor,
            decided_at=now,
            reason=reason,
        )

    async def consume_grant(
        self,
        approval_id: str,
        *,
        invocation: ToolInvocation,
    ) -> ToolApprovalGrant:
        """Consume approved evidence and mint one exact structural grant.

        Consumption occurs before the grant is returned. The caller must pass
        the grant to ToolExecutionService, which remains authoritative.
        """
        if not isinstance(
            invocation,
            ToolInvocation,
        ):
            raise ValueError("invocation must be ToolInvocation")

        current = await self.get(approval_id)

        if current.status is ApprovalStatus.REJECTED:
            raise ApprovalStateError("rejected approval cannot authorize execution")

        if current.status is ApprovalStatus.CONSUMED:
            raise ApprovalStateError("approval has already been consumed")

        if current.status is not ApprovalStatus.APPROVED:
            raise ApprovalStateError("approval request is not approved")

        now = self._now()

        if current.is_expired(now):
            raise ApprovalExpiredError("approved request has expired")

        if current.invocation != invocation:
            raise ApprovalStateError("invocation differs from frozen approval request")

        consumed = await self.store.consume(
            approval_id=current.approval_id,
            expected_version=current.version,
            consumed_at=now,
        )

        if (
            consumed.status is not ApprovalStatus.CONSUMED
            or consumed.invocation != invocation
        ):
            raise ApprovalStateError("approval store returned invalid consumed state")

        return ToolApprovalGrant(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
        )


def apply_decision_transition(
    request: ApprovalRequest,
    *,
    expected_version: int,
    decision: ApprovalDecision,
    actor: AuthenticatedApprovalActor,
    decided_at: datetime,
    reason: str | None,
) -> ApprovalRequest:
    """Pure helper for persistence adapters and deterministic test stores."""
    if request.version != expected_version:
        raise ApprovalConflictError("approval version changed")

    if request.status is not ApprovalStatus.PENDING:
        raise ApprovalConflictError("approval is no longer pending")

    status = (
        ApprovalStatus.APPROVED
        if decision is ApprovalDecision.APPROVE
        else ApprovalStatus.REJECTED
    )

    return replace(
        request,
        status=status,
        version=request.version + 1,
        decision=decision,
        decided_by=actor,
        decided_at=decided_at,
        decision_reason=reason,
    )


def apply_consumption_transition(
    request: ApprovalRequest,
    *,
    expected_version: int,
    consumed_at: datetime,
) -> ApprovalRequest:
    """Pure single-use approval consumption transition."""
    if request.version != expected_version:
        raise ApprovalConflictError("approval version changed")

    if request.status is not ApprovalStatus.APPROVED:
        raise ApprovalConflictError("approval is no longer consumable")

    return replace(
        request,
        status=ApprovalStatus.CONSUMED,
        version=request.version + 1,
        consumed_at=consumed_at,
    )
