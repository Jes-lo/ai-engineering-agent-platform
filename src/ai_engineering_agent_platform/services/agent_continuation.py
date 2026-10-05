"""Durable service-issued agent continuation boundary.

A continuation snapshot is execution state, not execution authority.

Only service-issued approval-required snapshots may be stored through this
boundary. A stored continuation never manufactures ToolApprovalGrant values
and never bypasses ToolExecutionService authorization or policy.

Concrete durable adapters must implement atomic compare-and-swap semantics for
replace and consume. Consumption must occur before provider side effects begin.

The application layer intentionally accepts ``object`` snapshots here to avoid
a dependency cycle between the continuation protocol and the agent/loop state
types. ControlledAgentService and BoundedAgentLoopService perform exact runtime
type and value validation before trusting a loaded snapshot.
"""

from enum import StrEnum
from hashlib import sha256
from typing import Protocol, runtime_checkable

MAX_CONTINUATION_IDENTITIES = 8
MAX_CONTINUATION_IDENTITY_LENGTH = 512


class AgentContinuationError(Exception):
    """Base failure for service-issued continuation state."""


class AgentContinuationConflictError(AgentContinuationError):
    """Raised when continuation state changed or already exists."""


class AgentContinuationIntegrityError(AgentContinuationError):
    """Raised when persisted continuation state fails validation."""


class AgentContinuationKind(StrEnum):
    """Kinds of resumable agent state persisted by the platform."""

    TURN = "turn"
    LOOP = "loop"


def validate_agent_continuation_identity(
    identity: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate exact immutable pending-call identity."""
    if not isinstance(
        identity,
        tuple,
    ):
        raise ValueError("continuation identity must be a tuple")

    if not identity:
        raise ValueError("continuation identity must not be empty")

    if len(identity) > MAX_CONTINUATION_IDENTITIES:
        raise ValueError("continuation identity exceeds maximum items")

    for value in identity:
        if not isinstance(
            value,
            str,
        ):
            raise ValueError("continuation identity values must be strings")

        if (
            not value
            or value != value.strip()
            or len(value) > MAX_CONTINUATION_IDENTITY_LENGTH
        ):
            raise ValueError("continuation identity value is invalid")

        if any(ord(character) < 32 for character in value):
            raise ValueError("continuation identity contains control characters")

    if len(identity) != len(set(identity)):
        raise ValueError("continuation identity values must be unique")

    return identity


def agent_continuation_key(
    kind: AgentContinuationKind,
    identity: tuple[str, ...],
) -> str:
    """Return a stable non-secret persistence key for exact call identity."""
    if not isinstance(
        kind,
        AgentContinuationKind,
    ):
        raise ValueError("kind must be AgentContinuationKind")

    validated = validate_agent_continuation_identity(identity)

    material = (kind.value + "\x00" + "\x00".join(validated)).encode("utf-8")

    return sha256(material).hexdigest()


@runtime_checkable
class AgentContinuationStore(Protocol):
    """Persistence boundary for exact service-issued continuation snapshots."""

    async def create(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        snapshot: object,
    ) -> None:
        """Persist a new active continuation exactly once."""
        ...

    async def load(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
    ) -> object | None:
        """Load the current active continuation when one exists."""
        ...

    async def replace(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        expected: object,
        snapshot: object,
    ) -> None:
        """Atomically replace one exact active snapshot."""
        ...

    async def consume(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        expected: object,
    ) -> None:
        """Atomically make one exact active continuation non-replayable."""
        ...
