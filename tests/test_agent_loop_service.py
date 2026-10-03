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
from ai_engineering_agent_platform.services import (
    MAX_AGENT_LOOP_MODEL_TURNS,
    MAX_AGENT_LOOP_TOOL_CALLS,
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
