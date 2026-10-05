"""Tests for the bounded conversational agent-loop service."""

from itertools import count

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMAssistantToolCallMessage,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolResultMessage,
    MessageRole,
    ProviderDescriptor,
    ProviderKind,
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)
from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailCategory,
    GuardrailSeverity,
    GuardrailStage,
)
from ai_engineering_agent_platform.services import (
    MAX_AGENT_LOOP_MODEL_TURNS,
    MAX_AGENT_LOOP_TOOL_CALLS,
    AgentGuardrailBlockedError,
    AgentLoopBudgetError,
    AgentLoopRequest,
    AgentLoopResumeError,
    AgentLoopStatus,
    BoundedAgentLoopService,
    ControlledAgentService,
    ToolApprovalGrant,
    ToolExecutionAuthorization,
    ToolExecutionPolicy,
    ToolExecutionService,
    ToolRegistry,
)
from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationConflictError,
    AgentContinuationKind,
)
from ai_engineering_agent_platform.services.guardrails import (
    GuardrailLiteralPattern,
    GuardrailPolicy,
    GuardrailService,
    LiteralPatternGuardrailRule,
)


class SequenceLLMProvider:
    """Return deterministic responses in order and record every request."""

    def __init__(
        self,
        responses: tuple[
            LLMResponse,
            ...,
        ],
    ) -> None:
        """Store deterministic response sequence."""
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="loop-llm",
            kind=ProviderKind.LLM,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Record request and return the next deterministic response."""
        self.requests.append(request)

        if not self._responses:
            raise AssertionError("unexpected extra LLM generation")

        return self._responses.pop(0)


class SyntheticToolProvider:
    """Record controlled tool executions."""

    def __init__(
        self,
    ) -> None:
        """Initialize invocation history."""
        self.invocations: list[ToolInvocation] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        """Return deterministic provider identity."""
        return ProviderDescriptor(
            name="loop-tools",
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(
        self,
    ) -> tuple[
        ToolDefinition,
        ...,
    ]:
        """Expose one read tool and one approval-required change tool."""
        return (
            ToolDefinition(
                name="lookup",
                description="Read synthetic information.",
                parameters=(
                    ToolParameter(
                        name="query",
                        parameter_type=(ToolParameterType.STRING),
                        required=True,
                    ),
                ),
            ),
            ToolDefinition(
                name="change",
                description="Perform a synthetic change.",
                parameters=(
                    ToolParameter(
                        name="value",
                        parameter_type=(ToolParameterType.NUMBER),
                        required=True,
                    ),
                ),
            ),
        )

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        """Record one normalized provider side effect."""
        self.invocations.append(invocation)

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content=("result:" + invocation.tool_name),
        )


def _stop(
    content: str = "done",
) -> LLMResponse:
    """Return one terminal response."""
    return LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content=content,
        ),
        finish_reason=FinishReason.STOP,
    )


def _lookup_call(
    value: str,
    *,
    provider_call_id: str | None = None,
) -> LLMToolCall:
    """Return one synthetic read proposal."""
    return LLMToolCall(
        tool_name="lookup",
        arguments=(
            ToolArgument(
                name="query",
                value=value,
            ),
        ),
        provider_call_id=provider_call_id,
    )


def _change_call(
    value: int = 1,
    *,
    provider_call_id: str | None = None,
) -> LLMToolCall:
    """Return one synthetic approval-required proposal."""
    return LLMToolCall(
        tool_name="change",
        arguments=(
            ToolArgument(
                name="value",
                value=value,
            ),
        ),
        provider_call_id=provider_call_id,
    )


def _tool_calls(
    *calls: LLMToolCall,
) -> LLMResponse:
    """Return one assistant tool-call response."""
    return LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content="",
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(calls),
    )


def _authorization(
    *tool_names: str,
    grants: tuple[
        ToolApprovalGrant,
        ...,
    ] = (),
) -> ToolExecutionAuthorization:
    """Return explicit platform authorization."""
    return ToolExecutionAuthorization(
        allowed_tool_names=tuple(tool_names),
        approval_grants=grants,
    )


def _loop_request(
    *,
    run_id: str = "loop-run",
    exposed_tool_names: tuple[
        str,
        ...,
    ] = ("lookup",),
    max_model_turns: int = 4,
    max_tool_calls: int = 4,
) -> AgentLoopRequest:
    """Return one bounded-loop request."""
    return AgentLoopRequest(
        run_id=run_id,
        llm_request=LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="Complete the synthetic task.",
                ),
            ),
        ),
        exposed_tool_names=(exposed_tool_names),
        max_model_turns=(max_model_turns),
        max_tool_calls=(max_tool_calls),
    )


def _agent_guardrail_service(
    *,
    literal: str = "__agent_guardrail_marker_not_present__",
    severity: GuardrailSeverity = GuardrailSeverity.LOW,
) -> GuardrailService:
    """Return complete deterministic loop guardrail coverage."""
    stages = (
        GuardrailStage.USER_INPUT,
        GuardrailStage.TOOL_RESULT,
        GuardrailStage.MODEL_OUTPUT,
    )

    return GuardrailService(
        policy=GuardrailPolicy(
            enabled_stages=stages,
            block_at_or_above=GuardrailSeverity.HIGH,
            max_content_chars=100_000,
            max_findings=64,
        ),
        rules=(
            LiteralPatternGuardrailRule(
                rule_id="agent-loop-runtime-test-rule",
                stages=stages,
                patterns=(
                    GuardrailLiteralPattern(
                        literal=literal,
                        category=GuardrailCategory.OTHER,
                        severity=severity,
                        message="configured synthetic loop signal",
                    ),
                ),
            ),
        ),
    )


def _stack(
    *responses: LLMResponse,
) -> tuple[
    BoundedAgentLoopService,
    SequenceLLMProvider,
    SyntheticToolProvider,
]:
    """Build the real controlled-turn service under the bounded loop."""
    llm_provider = SequenceLLMProvider(tuple(responses))

    tool_provider = SyntheticToolProvider()

    registry = ToolRegistry(
        (tool_provider,),
        policies=(
            ToolExecutionPolicy(
                tool_name="lookup",
                enabled=True,
                requires_approval=False,
            ),
            ToolExecutionPolicy(
                tool_name="change",
                enabled=True,
                requires_approval=True,
            ),
        ),
    )

    tool_execution = ToolExecutionService(registry)

    sequence = count(start=1)

    controlled_agent = ControlledAgentService(
        llm_provider=llm_provider,
        guardrail_service=_agent_guardrail_service(),
        registry=registry,
        tool_execution=tool_execution,
        call_id_factory=(
            lambda run_id, step_number: (
                f"agent:{run_id}:tool:{step_number}:loop-nonce-{next(sequence)}"
            )
        ),
    )

    return (
        BoundedAgentLoopService(agent_service=controlled_agent),
        llm_provider,
        tool_provider,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "max_model_turns",
            0,
        ),
        (
            "max_model_turns",
            MAX_AGENT_LOOP_MODEL_TURNS + 1,
        ),
        (
            "max_tool_calls",
            0,
        ),
        (
            "max_tool_calls",
            MAX_AGENT_LOOP_TOOL_CALLS + 1,
        ),
    ),
)
def test_loop_request_enforces_absolute_budgets(
    field: str,
    value: int,
) -> None:
    """Callers cannot disable or exceed hard loop ceilings."""
    kwargs = {
        field: value,
    }

    with pytest.raises(
        ValueError,
        match=("must be between"),
    ):
        _loop_request(
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_stop_completes_without_tools() -> None:
    """A terminal response completes after one model turn."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(_stop("finished"))

    result = await service.start(
        _loop_request(),
        authorization=_authorization("lookup"),
    )

    assert result.status is (AgentLoopStatus.COMPLETED)

    assert result.model_turns_used == 1
    assert result.tool_calls_used == 0
    assert result.final_response is not None
    assert result.final_response.message.content == ("finished")

    assert len(llm_provider.requests) == 1

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_tool_result_round_trip_then_stop() -> None:
    """Validated tool output is returned as typed context before next model turn."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(
            _lookup_call(
                "status",
                provider_call_id="provider-1",
            )
        ),
        _stop("final answer"),
    )

    result = await service.start(
        _loop_request(run_id="roundtrip"),
        authorization=_authorization("lookup"),
    )

    assert result.status is (AgentLoopStatus.COMPLETED)

    assert result.model_turns_used == 2
    assert result.tool_calls_used == 1

    assert len(tool_provider.invocations) == 1

    assert len(llm_provider.requests) == 2

    second_messages = llm_provider.requests[1].messages

    assert len(second_messages) == 3

    assert isinstance(
        second_messages[1],
        LLMAssistantToolCallMessage,
    )

    assert isinstance(
        second_messages[2],
        LLMToolResultMessage,
    )

    tool_message = second_messages[2]

    assert tool_message.result.call_id == ("agent:roundtrip:tool:1:loop-nonce-1")

    assert tool_message.provider_call_id == ("provider-1")

    assert result.final_response is not None

    assert result.messages[-1] == (result.final_response.message)


@pytest.mark.anyio
async def test_multiple_model_tool_cycles_are_globally_counted() -> None:
    """Tool usage accumulates across model turns rather than resetting."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(
            _lookup_call(
                "first",
                provider_call_id="provider-1",
            )
        ),
        _tool_calls(
            _lookup_call(
                "second",
                provider_call_id="provider-2",
            )
        ),
        _stop(),
    )

    result = await service.start(
        _loop_request(
            max_model_turns=3,
            max_tool_calls=2,
        ),
        authorization=_authorization("lookup"),
    )

    assert result.model_turns_used == 3
    assert result.tool_calls_used == 2

    assert len(llm_provider.requests) == 3

    assert len(tool_provider.invocations) == 2


@pytest.mark.anyio
async def test_model_turn_budget_stops_before_extra_generation() -> None:
    """The loop cannot request another model turn after its global ceiling."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(_lookup_call("first")),
        _stop(),
    )

    with pytest.raises(
        AgentLoopBudgetError,
        match="model-turn budget exhausted",
    ):
        await service.start(
            _loop_request(
                max_model_turns=1,
                max_tool_calls=2,
            ),
            authorization=_authorization("lookup"),
        )

    assert len(llm_provider.requests) == 1

    assert len(tool_provider.invocations) == 1


@pytest.mark.anyio
async def test_global_tool_budget_prevents_second_execution() -> None:
    """Once the global tool budget is spent, tools are no longer exposed."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(_lookup_call("first")),
        _tool_calls(_lookup_call("second")),
    )

    with pytest.raises(
        AgentLoopBudgetError,
        match="after global tool-call budget exhaustion",
    ):
        await service.start(
            _loop_request(
                max_model_turns=2,
                max_tool_calls=1,
            ),
            authorization=_authorization("lookup"),
        )

    assert len(llm_provider.requests) == 2

    assert llm_provider.requests[1].tools == ()

    assert len(tool_provider.invocations) == 1


@pytest.mark.anyio
async def test_oversized_first_batch_fails_before_side_effect() -> None:
    """A proposal batch larger than remaining global budget executes nothing."""
    (
        service,
        _,
        tool_provider,
    ) = _stack(
        _tool_calls(
            _lookup_call("first"),
            _lookup_call("second"),
        )
    )

    with pytest.raises(
        AgentLoopBudgetError,
        match="remaining global tool-call budget",
    ):
        await service.start(
            _loop_request(
                max_tool_calls=1,
            ),
            authorization=_authorization("lookup"),
        )

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_approval_pause_resume_continues_without_regeneration() -> None:
    """Approved continuation executes the frozen decision then resumes the loop."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(_change_call(provider_call_id="provider-change-1")),
        _stop("change complete"),
    )

    pending = await service.start(
        _loop_request(
            run_id="approval-loop",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.status is (AgentLoopStatus.APPROVAL_REQUIRED)

    assert pending.pending_turn is not None

    assert len(llm_provider.requests) == 1

    assert tool_provider.invocations == []

    call_id = pending.pending_turn.pending_approval_call_ids[0]

    completed = await service.resume(
        pending,
        authorization=_authorization(
            "change",
            grants=(
                ToolApprovalGrant(
                    call_id=call_id,
                    tool_name="change",
                ),
            ),
        ),
    )

    assert completed.status is (AgentLoopStatus.COMPLETED)

    assert completed.model_turns_used == 2
    assert completed.tool_calls_used == 1

    assert len(llm_provider.requests) == 2

    assert len(tool_provider.invocations) == 1


@pytest.mark.anyio
async def test_missing_approval_remains_paused_without_model_call() -> None:
    """An unapproved resume does not execute or regenerate the decision."""
    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(_change_call()),
        _stop(),
    )

    pending = await service.start(
        _loop_request(
            run_id="still-pending",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    still_pending = await service.resume(
        pending,
        authorization=_authorization("change"),
    )

    assert still_pending.status is (AgentLoopStatus.APPROVAL_REQUIRED)

    assert len(llm_provider.requests) == 1

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_consumed_loop_continuation_cannot_be_replayed() -> None:
    """A successfully consumed approval continuation cannot run twice."""
    (
        service,
        _,
        tool_provider,
    ) = _stack(
        _tool_calls(_change_call()),
        _stop(),
    )

    pending = await service.start(
        _loop_request(
            run_id="loop-replay",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.pending_turn is not None

    call_id = pending.pending_turn.pending_approval_call_ids[0]

    authorization = _authorization(
        "change",
        grants=(
            ToolApprovalGrant(
                call_id=call_id,
                tool_name="change",
            ),
        ),
    )

    await service.resume(
        pending,
        authorization=authorization,
    )

    with pytest.raises(
        AgentLoopResumeError,
        match="not an active service-issued continuation",
    ):
        await service.resume(
            pending,
            authorization=authorization,
        )

    assert len(tool_provider.invocations) == 1


def test_agent_loop_service_is_public() -> None:
    """Feature 14 loop primitives are exported by services."""
    import ai_engineering_agent_platform.services as services

    expected = {
        "AgentLoopBudgetError",
        "AgentLoopError",
        "AgentLoopRequest",
        "AgentLoopResult",
        "AgentLoopResumeError",
        "AgentLoopStatus",
        "BoundedAgentLoopService",
        "MAX_AGENT_LOOP_MODEL_TURNS",
        "MAX_AGENT_LOOP_TOOL_CALLS",
    }

    assert expected <= set(services.__all__)

    for name in expected:
        assert hasattr(
            services,
            name,
        )


class _MemoryLoopContinuationStore:
    """Shared CAS store that survives service object replacement in tests."""

    def __init__(self) -> None:
        self.active: dict[
            tuple[
                AgentContinuationKind,
                tuple[str, ...],
            ],
            object,
        ] = {}
        self.consumed: set[
            tuple[
                AgentContinuationKind,
                tuple[str, ...],
            ]
        ] = set()

    async def create(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        snapshot: object,
    ) -> None:
        key = (
            kind,
            identity,
        )

        if key in self.active or key in self.consumed:
            raise AgentContinuationConflictError("continuation already exists")

        self.active[key] = snapshot

    async def load(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
    ) -> object | None:
        return self.active.get(
            (
                kind,
                identity,
            )
        )

    async def replace(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        expected: object,
        snapshot: object,
    ) -> None:
        key = (
            kind,
            identity,
        )

        if self.active.get(key) != expected:
            raise AgentContinuationConflictError("continuation changed")

        self.active[key] = snapshot

    async def consume(
        self,
        *,
        kind: AgentContinuationKind,
        identity: tuple[str, ...],
        expected: object,
    ) -> None:
        key = (
            kind,
            identity,
        )

        if self.active.get(key) != expected:
            raise AgentContinuationConflictError("continuation changed")

        del self.active[key]

        self.consumed.add(key)


@pytest.mark.anyio
async def test_durable_loop_continuation_survives_service_restart() -> None:
    """Fresh loop resumes frozen tool decision before making its next LLM call."""
    store = _MemoryLoopContinuationStore()

    (
        original,
        original_llm,
        original_tool_provider,
    ) = _stack(
        _tool_calls(_change_call(provider_call_id="provider-durable-1")),
        _stop("unused before restart"),
    )

    original._agent_service._continuation_store = store
    original._continuation_store = store

    pending = await original.start(
        _loop_request(
            run_id="durable-loop",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.status is AgentLoopStatus.APPROVAL_REQUIRED
    assert pending.pending_turn is not None
    assert len(original_llm.requests) == 1
    assert original_tool_provider.invocations == []

    identity = pending.pending_turn.pending_approval_call_ids

    (
        fresh,
        fresh_llm,
        fresh_tool_provider,
    ) = _stack(_stop("completed after durable resume"))

    fresh._agent_service._continuation_store = store
    fresh._continuation_store = store

    restored = await fresh.load_pending(identity)

    assert restored == pending

    call_id = identity[0]

    completed = await fresh.resume(
        restored,
        authorization=_authorization(
            "change",
            grants=(
                ToolApprovalGrant(
                    call_id=call_id,
                    tool_name="change",
                ),
            ),
        ),
    )

    assert completed.status is AgentLoopStatus.COMPLETED
    assert completed.model_turns_used == 2
    assert completed.tool_calls_used == 1

    # One model decision happened before the simulated restart. The fresh
    # service calls the model only after executing that frozen decision.
    assert len(original_llm.requests) == 1
    assert len(fresh_llm.requests) == 1
    assert len(fresh_tool_provider.invocations) == 1

    replay, _, replay_tool_provider = _stack(_stop("must not be used"))

    replay._agent_service._continuation_store = store
    replay._continuation_store = store

    with pytest.raises(
        AgentLoopResumeError,
        match="not an active service-issued continuation",
    ):
        await replay.resume(
            restored,
            authorization=_authorization(
                "change",
                grants=(
                    ToolApprovalGrant(
                        call_id=call_id,
                        tool_name="change",
                    ),
                ),
            ),
        )

    assert replay_tool_provider.invocations == []


@pytest.mark.anyio
async def test_durable_loop_missing_approval_remains_active_without_model_call() -> (
    None
):
    """Fail-closed resume may replace state but cannot execute or regenerate."""
    store = _MemoryLoopContinuationStore()

    (
        original,
        original_llm,
        original_tool_provider,
    ) = _stack(
        _tool_calls(_change_call()),
        _stop(),
    )

    original._agent_service._continuation_store = store
    original._continuation_store = store

    pending = await original.start(
        _loop_request(
            run_id="durable-loop-pending",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.pending_turn is not None

    identity = pending.pending_turn.pending_approval_call_ids

    (
        fresh,
        fresh_llm,
        fresh_tool_provider,
    ) = _stack(_stop("must not be called"))

    fresh._agent_service._continuation_store = store
    fresh._continuation_store = store

    restored = await fresh.load_pending(identity)

    still_pending = await fresh.resume(
        restored,
        authorization=_authorization("change"),
    )

    assert still_pending.status is AgentLoopStatus.APPROVAL_REQUIRED
    assert fresh_llm.requests == []
    assert fresh_tool_provider.invocations == []
    assert len(original_llm.requests) == 1
    assert original_tool_provider.invocations == []

    restored_again = await fresh.load_pending(identity)

    assert restored_again == still_pending


@pytest.mark.anyio
async def test_durable_loop_pause_uses_only_parent_loop_continuation() -> None:
    """A bounded loop persists LOOP only; its child TURN is not durable."""
    store = _MemoryLoopContinuationStore()

    service, llm_provider, tool_provider = _stack(
        _tool_calls(_change_call()),
        _stop(),
    )

    service._agent_service._continuation_store = store
    service._continuation_store = store

    pending = await service.start(
        _loop_request(
            run_id="parent-owned-loop-pause",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.status is AgentLoopStatus.APPROVAL_REQUIRED
    assert pending.pending_turn is not None

    identity = tuple(
        step.invocation.call_id for step in pending.pending_turn.planned_steps
    )

    assert (
        await store.load(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
        )
        == pending
    )

    assert (
        await store.load(
            kind=AgentContinuationKind.TURN,
            identity=identity,
        )
        is None
    )

    assert len(llm_provider.requests) == 1
    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_durable_loop_partial_approval_keeps_only_parent_continuation() -> None:
    """A fail-closed partial resume replaces LOOP without creating TURN."""
    store = _MemoryLoopContinuationStore()

    service, llm_provider, tool_provider = _stack(
        _tool_calls(_change_call()),
        _stop(),
    )

    service._agent_service._continuation_store = store
    service._continuation_store = store

    pending = await service.start(
        _loop_request(
            run_id="parent-owned-loop-partial",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.pending_turn is not None

    identity = tuple(
        step.invocation.call_id for step in pending.pending_turn.planned_steps
    )

    still_pending = await service.resume(
        pending,
        authorization=_authorization("change"),
    )

    assert still_pending.status is AgentLoopStatus.APPROVAL_REQUIRED

    assert (
        await store.load(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
        )
        == still_pending
    )

    assert (
        await store.load(
            kind=AgentContinuationKind.TURN,
            identity=identity,
        )
        is None
    )

    assert len(llm_provider.requests) == 1
    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_parent_loop_gate_consumes_before_tool_side_effects() -> None:
    """A parent-gate persistence conflict prevents provider execution."""

    class RejectParentConsumeStore(_MemoryLoopContinuationStore):
        async def consume(
            self,
            *,
            kind: AgentContinuationKind,
            identity: tuple[str, ...],
            expected: object,
        ) -> None:
            if kind is AgentContinuationKind.LOOP:
                raise AgentContinuationConflictError(
                    "synthetic parent consume conflict"
                )

            await super().consume(
                kind=kind,
                identity=identity,
                expected=expected,
            )

    store = RejectParentConsumeStore()

    service, llm_provider, tool_provider = _stack(
        _tool_calls(_change_call()),
        _stop(),
    )

    service._agent_service._continuation_store = store
    service._continuation_store = store

    pending = await service.start(
        _loop_request(
            run_id="parent-gate-before-side-effect",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.pending_turn is not None

    identity = tuple(
        step.invocation.call_id for step in pending.pending_turn.planned_steps
    )

    call_id = pending.pending_turn.pending_approval_call_ids[0]

    with pytest.raises(
        AgentLoopResumeError,
        match="not an active service-issued continuation",
    ):
        await service.resume(
            pending,
            authorization=_authorization(
                "change",
                grants=(
                    ToolApprovalGrant(
                        call_id=call_id,
                        tool_name="change",
                    ),
                ),
            ),
        )

    # The LOOP consume failed before ToolExecutionService/provider execution.
    assert tool_provider.invocations == []

    assert (
        await store.load(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
        )
        == pending
    )

    assert (
        await store.load(
            kind=AgentContinuationKind.TURN,
            identity=identity,
        )
        is None
    )

    # No post-pause model regeneration occurred either.
    assert len(llm_provider.requests) == 1


@pytest.mark.anyio
async def test_tool_result_guardrail_blocks_before_next_llm_turn() -> None:
    """Blocked tool output must not reach a subsequent model turn."""
    marker = "result:lookup"

    (
        service,
        llm_provider,
        tool_provider,
    ) = _stack(
        _tool_calls(
            _lookup_call("guarded-tool-result"),
        ),
        _stop("must not be generated"),
    )

    service._agent_service._guardrail_service = _agent_guardrail_service(
        literal=marker,
        severity=GuardrailSeverity.HIGH,
    )

    with pytest.raises(
        AgentGuardrailBlockedError,
        match="blocked by deterministic guardrail policy",
    ) as exc_info:
        await service.start(
            _loop_request(),
            authorization=_authorization("lookup"),
        )

    error = exc_info.value

    assert error.stage is GuardrailStage.TOOL_RESULT
    assert error.blocking_rule_ids == ("agent-loop-runtime-test-rule",)

    assert len(llm_provider.requests) == 1
    assert len(tool_provider.invocations) == 1

    assert tool_provider.invocations[0].tool_name == "lookup"

    assert marker not in str(error)
    assert marker not in repr(error)
