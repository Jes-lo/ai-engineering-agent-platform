"""Tests for PostgreSQL durable agent continuation adapter."""

from hashlib import sha256

import pytest

from ai_engineering_agent_platform.adapters.postgres.agent_continuation_store import (
    PostgreSQLAgentContinuationStore,
)
from ai_engineering_agent_platform.contracts.llm import (
    FinishReason,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    MessageRole,
)
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.services.agent import (
    AgentPlannedToolCall,
    AgentTurnResult,
    AgentTurnStatus,
)
from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationConflictError,
    AgentContinuationIntegrityError,
    AgentContinuationKind,
    agent_continuation_key,
)
from ai_engineering_agent_platform.services.agent_continuation_persistence import (
    AGENT_CONTINUATION_FORMAT_VERSION,
    AgentContinuationCodec,
)


def _turn(
    *,
    run_id: str = "durable",
) -> AgentTurnResult:
    proposal = LLMToolCall(
        tool_name="change",
        arguments=(
            ToolArgument(
                name="target",
                value="production",
            ),
        ),
        provider_call_id="provider-call",
    )

    invocation = ToolInvocation(
        call_id="agent:durable:tool:1:nonce",
        tool_name="change",
        arguments=proposal.arguments,
    )

    return AgentTurnResult(
        run_id=run_id,
        status=AgentTurnStatus.APPROVAL_REQUIRED,
        response=LLMResponse(
            model="test-model",
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="approval required",
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(proposal,),
        ),
        planned_steps=(
            AgentPlannedToolCall(
                step_number=1,
                proposal=proposal,
                invocation=invocation,
            ),
        ),
        pending_approval_call_ids=(invocation.call_id,),
    )


def _identity(
    turn: AgentTurnResult,
) -> tuple[str, ...]:
    return tuple(step.invocation.call_id for step in turn.planned_steps)


def _row(
    turn: AgentTurnResult,
    *,
    status: str = "active",
    payload: str | None = None,
    digest: str | None = None,
    version: int = 1,
) -> tuple[object, ...]:
    identity = _identity(turn)

    codec = AgentContinuationCodec()

    actual_payload = (
        codec.dumps(
            kind=AgentContinuationKind.TURN,
            identity=identity,
            snapshot=turn,
        )
        if payload is None
        else payload
    )

    actual_digest = (
        sha256(actual_payload.encode("utf-8")).hexdigest() if digest is None else digest
    )

    return (
        agent_continuation_key(
            AgentContinuationKind.TURN,
            identity,
        ),
        AgentContinuationKind.TURN.value,
        '["' + identity[0] + '"]',
        status,
        AGENT_CONTINUATION_FORMAT_VERSION,
        version,
        actual_payload,
        actual_digest,
    )


class _FakeCursor:
    def __init__(
        self,
        results: list[tuple[object, ...] | None],
        queries: list[
            tuple[
                str,
                tuple[object, ...] | None,
            ]
        ],
    ) -> None:
        self._results = results
        self._queries = queries
        self._current: tuple[object, ...] | None = None

    async def __aenter__(
        self,
    ) -> "_FakeCursor":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None

    async def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> None:
        self._queries.append(
            (
                query,
                params,
            )
        )

        if not self._results:
            raise AssertionError("unexpected SQL execution")

        self._current = self._results.pop(0)

    async def fetchone(
        self,
    ) -> tuple[object, ...] | None:
        return self._current


class _FakeConnection:
    def __init__(
        self,
        results: list[tuple[object, ...] | None],
        queries: list[
            tuple[
                str,
                tuple[object, ...] | None,
            ]
        ],
    ) -> None:
        self._results = results
        self._queries = queries

    async def __aenter__(
        self,
    ) -> "_FakeConnection":
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None

    def cursor(
        self,
    ) -> _FakeCursor:
        return _FakeCursor(
            self._results,
            self._queries,
        )


class _FakePool:
    def __init__(
        self,
        results: list[tuple[object, ...] | None],
    ) -> None:
        self.results = results
        self.queries: list[
            tuple[
                str,
                tuple[object, ...] | None,
            ]
        ] = []

    def connection(
        self,
    ) -> _FakeConnection:
        return _FakeConnection(
            self.results,
            self.queries,
        )


def _store(
    *results: tuple[object, ...] | None,
) -> tuple[
    PostgreSQLAgentContinuationStore,
    _FakePool,
]:
    pool = _FakePool(list(results))

    store = PostgreSQLAgentContinuationStore(
        pool=pool,  # type: ignore[arg-type]
    )

    return (
        store,
        pool,
    )


@pytest.mark.anyio
async def test_create_uses_insert_conflict_guard() -> None:
    turn = _turn()
    identity = _identity(turn)

    key = agent_continuation_key(
        AgentContinuationKind.TURN,
        identity,
    )

    store, pool = _store((key,))

    await store.create(
        kind=AgentContinuationKind.TURN,
        identity=identity,
        snapshot=turn,
    )

    assert len(pool.queries) == 1
    assert "ON CONFLICT DO NOTHING" in pool.queries[0][0]


@pytest.mark.anyio
async def test_create_conflict_fails_closed() -> None:
    turn = _turn()

    store, _ = _store(None)

    with pytest.raises(
        AgentContinuationConflictError,
    ):
        await store.create(
            kind=AgentContinuationKind.TURN,
            identity=_identity(turn),
            snapshot=turn,
        )


@pytest.mark.anyio
async def test_load_active_round_trips_snapshot() -> None:
    turn = _turn()

    store, _ = _store(_row(turn))

    loaded = await store.load(
        kind=AgentContinuationKind.TURN,
        identity=_identity(turn),
    )

    assert loaded == turn


@pytest.mark.anyio
async def test_load_consumed_returns_none() -> None:
    turn = _turn()

    store, _ = _store(
        _row(
            turn,
            status="consumed",
        )
    )

    loaded = await store.load(
        kind=AgentContinuationKind.TURN,
        identity=_identity(turn),
    )

    assert loaded is None


@pytest.mark.anyio
async def test_checksum_tampering_fails_closed() -> None:
    turn = _turn()

    store, _ = _store(
        _row(
            turn,
            digest="0" * 64,
        )
    )

    with pytest.raises(
        AgentContinuationIntegrityError,
        match="checksum mismatch",
    ):
        await store.load(
            kind=AgentContinuationKind.TURN,
            identity=_identity(turn),
        )


@pytest.mark.anyio
async def test_replace_uses_compare_and_swap() -> None:
    turn = _turn()

    store, pool = _store(
        _row(turn),
        (2,),
    )

    await store.replace(
        kind=AgentContinuationKind.TURN,
        identity=_identity(turn),
        expected=turn,
        snapshot=turn,
    )

    assert len(pool.queries) == 2
    assert "state_version = %s" in pool.queries[1][0]
    assert "continuation_sha256 = %s" in pool.queries[1][0]


@pytest.mark.anyio
async def test_consume_uses_compare_and_swap() -> None:
    turn = _turn()

    store, pool = _store(
        _row(turn),
        (2,),
    )

    await store.consume(
        kind=AgentContinuationKind.TURN,
        identity=_identity(turn),
        expected=turn,
    )

    assert len(pool.queries) == 2
    assert "continuation_status = 'consumed'" in pool.queries[1][0]
    assert "state_version = %s" in pool.queries[1][0]


@pytest.mark.anyio
async def test_stale_expected_snapshot_fails_before_update() -> None:
    stored = _turn()
    stale = _turn(run_id="different")

    store, pool = _store(_row(stored))

    with pytest.raises(
        AgentContinuationConflictError,
    ):
        await store.consume(
            kind=AgentContinuationKind.TURN,
            identity=_identity(stored),
            expected=stale,
        )

    assert len(pool.queries) == 1


@pytest.mark.anyio
async def test_metadata_mismatch_fails_closed() -> None:
    turn = _turn()
    row = list(_row(turn))

    row[1] = "loop"

    store, _ = _store(tuple(row))

    with pytest.raises(
        AgentContinuationIntegrityError,
        match="metadata mismatch",
    ):
        await store.load(
            kind=AgentContinuationKind.TURN,
            identity=_identity(turn),
        )
