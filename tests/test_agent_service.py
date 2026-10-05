"""Tests for controlled single-turn agent orchestration."""

from itertools import count

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
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
    MAX_AGENT_TURN_STEPS,
    AgentOrchestrationError,
    AgentPlannedToolCall,
    AgentResponseError,
    AgentResumeError,
    AgentStepLimitError,
    AgentTurnRequest,
    AgentTurnResult,
    AgentTurnStatus,
    ControlledAgentService,
    ToolApprovalGrant,
    ToolAuthorizationError,
    ToolExecutionAuthorization,
    ToolExecutionPolicy,
    ToolExecutionPreflight,
    ToolExecutionService,
    ToolInputValidationError,
    ToolRegistry,
)
from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationConflictError,
    AgentContinuationKind,
)


class SyntheticLLMProvider:
    """Deterministic LLM provider for orchestration tests."""

    def __init__(
        self,
        response: LLMResponse,
    ) -> None:
        """Store one deterministic response."""
        self._response = response
        self.requests: list[LLMRequest] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        """Return deterministic LLM identity."""
        return ProviderDescriptor(
            name="synthetic-llm",
            kind=ProviderKind.LLM,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Record the request and return the configured response."""
        self.requests.append(request)
        return self._response


class SyntheticToolProvider:
    """Deterministic tool provider for orchestration tests."""

    def __init__(self) -> None:
        """Initialize execution recording."""
        self.invocations: list[ToolInvocation] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        """Return deterministic tool-provider identity."""
        return ProviderDescriptor(
            name="synthetic-tools",
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(
        self,
    ) -> tuple[
        ToolDefinition,
        ...,
    ]:
        """Return deterministic synthetic tools."""
        return _definitions()

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        """Record one provider side effect."""
        self.invocations.append(invocation)

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content=("synthetic result for " + invocation.tool_name),
        )


def _definitions() -> tuple[
    ToolDefinition,
    ...,
]:
    """Return one read tool and one approval-required tool."""
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


def _policies(
    *,
    lookup_enabled: bool = True,
    change_enabled: bool = True,
) -> tuple[
    ToolExecutionPolicy,
    ...,
]:
    """Return complete synthetic policy coverage."""
    return (
        ToolExecutionPolicy(
            tool_name="lookup",
            enabled=lookup_enabled,
            requires_approval=False,
        ),
        ToolExecutionPolicy(
            tool_name="change",
            enabled=change_enabled,
            requires_approval=True,
        ),
    )


def _authorization(
    *tool_names: str,
    grants: tuple[
        ToolApprovalGrant,
        ...,
    ] = (),
) -> ToolExecutionAuthorization:
    """Return explicit execution authorization."""
    return ToolExecutionAuthorization(
        allowed_tool_names=tuple(tool_names),
        approval_grants=grants,
    )


def _base_llm_request(
    *,
    tools: tuple[
        ToolDefinition,
        ...,
    ] = (),
) -> LLMRequest:
    """Return a provider-neutral request."""
    return LLMRequest(
        model="synthetic-model",
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="Perform the synthetic task.",
            ),
        ),
        tools=tools,
    )


def _turn_request(
    *,
    run_id: str = "run-1",
    exposed_tool_names: tuple[
        str,
        ...,
    ] = ("lookup",),
    max_steps: int = 4,
) -> AgentTurnRequest:
    """Return one controlled agent-turn request."""
    return AgentTurnRequest(
        run_id=run_id,
        llm_request=_base_llm_request(),
        exposed_tool_names=(exposed_tool_names),
        max_steps=max_steps,
    )


def _stop_response() -> LLMResponse:
    """Return a normal STOP response."""
    return LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content="Synthetic completion.",
        ),
        finish_reason=FinishReason.STOP,
    )


def _proposal(
    tool_name: str,
    *,
    arguments: tuple[
        ToolArgument,
        ...,
    ],
    provider_call_id: str | None = None,
) -> LLMToolCall:
    """Return one inert model proposal."""
    return LLMToolCall(
        tool_name=tool_name,
        arguments=arguments,
        provider_call_id=provider_call_id,
    )


def _tool_response(
    *proposals: LLMToolCall,
) -> LLMResponse:
    """Return one TOOL_CALLS response."""
    return LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content="",
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(proposals),
    )


def _lookup_proposal(
    value: object = "synthetic",
    *,
    provider_call_id: str | None = None,
) -> LLMToolCall:
    """Return one synthetic lookup proposal."""
    return _proposal(
        "lookup",
        arguments=(
            ToolArgument(
                name="query",
                value=value,  # type: ignore[arg-type]
            ),
        ),
        provider_call_id=provider_call_id,
    )


def _change_proposal(
    value: object = 2,
) -> LLMToolCall:
    """Return one approval-required proposal."""
    return _proposal(
        "change",
        arguments=(
            ToolArgument(
                name="value",
                value=value,  # type: ignore[arg-type]
            ),
        ),
    )


def _service(
    response: LLMResponse,
    *,
    policies: tuple[
        ToolExecutionPolicy,
        ...,
    ]
    | None = None,
) -> tuple[
    ControlledAgentService,
    SyntheticLLMProvider,
    SyntheticToolProvider,
    ToolRegistry,
    ToolExecutionService,
]:
    """Build deterministic orchestration with fresh test call IDs."""
    llm_provider = SyntheticLLMProvider(response)

    tool_provider = SyntheticToolProvider()

    registry = ToolRegistry(
        (tool_provider,),
        policies=(_policies() if policies is None else policies),
    )

    tool_execution = ToolExecutionService(registry)

    sequence = count(start=1)

    def call_id_factory(
        run_id: str,
        step_number: int,
    ) -> str:
        """Return deterministic-but-fresh platform test identity."""
        return f"agent:{run_id}:tool:{step_number}:nonce-{next(sequence)}"

    service = ControlledAgentService(
        llm_provider=llm_provider,
        registry=registry,
        tool_execution=tool_execution,
        call_id_factory=call_id_factory,
    )

    return (
        service,
        llm_provider,
        tool_provider,
        registry,
        tool_execution,
    )


def test_agent_request_rejects_caller_supplied_llm_tools() -> None:
    """The agent owns which registered definitions reach the model."""
    with pytest.raises(
        ValueError,
        match="must not contain caller-supplied tools",
    ):
        AgentTurnRequest(
            run_id="run-tools",
            llm_request=_base_llm_request(tools=(_definitions()[0],)),
            exposed_tool_names=("lookup",),
        )


@pytest.mark.parametrize(
    "run_id",
    (
        "",
        "contains space",
        ":provider-id",
        "a" * 65,
    ),
)
def test_agent_request_rejects_non_portable_run_id(
    run_id: str,
) -> None:
    """Caller correlation IDs remain bounded and portable."""
    with pytest.raises(
        ValueError,
        match="run_id must use",
    ):
        _turn_request(run_id=run_id)


@pytest.mark.parametrize(
    "max_steps",
    (
        0,
        -1,
        MAX_AGENT_TURN_STEPS + 1,
    ),
)
def test_agent_request_enforces_absolute_step_bound(
    max_steps: int,
) -> None:
    """Callers cannot disable or exceed the hard platform ceiling."""
    with pytest.raises(
        ValueError,
        match="max_steps must be between",
    ):
        _turn_request(max_steps=max_steps)


@pytest.mark.anyio
async def test_stop_response_completes_without_tool_execution() -> None:
    """STOP completes one turn without tool side effects."""
    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(_stop_response())

    result = await service.start(
        _turn_request(),
        authorization=_authorization("lookup"),
    )

    assert result.status is (AgentTurnStatus.COMPLETED)

    assert result.planned_steps == ()
    assert result.executions == ()
    assert tool_provider.invocations == []

    assert len(llm_provider.requests) == 1

    assert tuple(tool.name for tool in llm_provider.requests[0].tools) == ("lookup",)


@pytest.mark.anyio
async def test_unauthorized_tool_is_not_exposed_to_model() -> None:
    """Execution authorization also bounds model-visible capabilities."""
    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(_stop_response())

    with pytest.raises(
        ToolAuthorizationError,
        match="outside the execution allowlist",
    ):
        await service.start(
            _turn_request(exposed_tool_names=("change",)),
            authorization=_authorization("lookup"),
        )

    assert llm_provider.requests == []
    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_disabled_tool_is_not_exposed_to_model() -> None:
    """Platform-disabled tools never become model-visible capabilities."""
    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(
        _stop_response(),
        policies=_policies(
            change_enabled=False,
        ),
    )

    with pytest.raises(
        ToolAuthorizationError,
        match="disabled by platform policy",
    ):
        await service.start(
            _turn_request(exposed_tool_names=("change",)),
            authorization=_authorization("change"),
        )

    assert llm_provider.requests == []
    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_safe_proposal_gets_fresh_platform_call_id() -> None:
    """Provider call metadata cannot become controlled execution identity."""
    response = _tool_response(
        _lookup_proposal(
            provider_call_id="provider-call-999",
        )
    )

    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(response)

    result = await service.start(
        _turn_request(run_id="run-safe"),
        authorization=_authorization("lookup"),
    )

    assert result.status is (AgentTurnStatus.TOOL_RESULTS_AVAILABLE)

    planned = result.planned_steps[0]

    assert planned.proposal.provider_call_id == "provider-call-999"

    assert planned.invocation.call_id == "agent:run-safe:tool:1:nonce-1"

    assert planned.invocation.call_id != planned.proposal.provider_call_id

    assert tool_provider.invocations == [planned.invocation]

    assert len(llm_provider.requests) == 1


@pytest.mark.anyio
async def test_same_correlation_run_id_gets_fresh_call_identity() -> None:
    """Reusing correlation metadata does not reuse an approval identity."""
    response = _tool_response(_lookup_proposal())

    (
        service,
        _,
        _,
        _,
        _,
    ) = _service(response)

    first = await service.start(
        _turn_request(run_id="same-run"),
        authorization=_authorization("lookup"),
    )

    second = await service.start(
        _turn_request(run_id="same-run"),
        authorization=_authorization("lookup"),
    )

    assert (
        first.planned_steps[0].invocation.call_id
        != second.planned_steps[0].invocation.call_id
    )


@pytest.mark.anyio
async def test_provider_neutral_boundary_rejects_unexposed_tool() -> None:
    """The service rechecks exposure independently of the LLM adapter."""
    response = _tool_response(_change_proposal())

    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(response)

    with pytest.raises(
        AgentResponseError,
        match="outside the agent exposure set",
    ):
        await service.start(
            _turn_request(exposed_tool_names=("lookup",)),
            authorization=_authorization(
                "lookup",
                "change",
            ),
        )

    assert len(llm_provider.requests) == 1

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_step_limit_fails_before_tool_execution() -> None:
    """Too many proposals fail closed before provider side effects."""
    response = _tool_response(
        _lookup_proposal(),
        _lookup_proposal("second"),
    )

    (
        service,
        _,
        tool_provider,
        _,
        _,
    ) = _service(response)

    with pytest.raises(
        AgentStepLimitError,
        match="exceeds the agent turn step limit",
    ):
        await service.start(
            _turn_request(
                max_steps=1,
            ),
            authorization=_authorization("lookup"),
        )

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_missing_approval_creates_service_issued_pause() -> None:
    """Approval-required tools create an active in-memory continuation."""
    response = _tool_response(_change_proposal())

    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(response)

    pending = await service.start(
        _turn_request(
            run_id="run-approval",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.status is (AgentTurnStatus.APPROVAL_REQUIRED)

    assert pending.executions == ()

    assert pending.pending_approval_call_ids == ("agent:run-approval:tool:1:nonce-1",)

    assert tool_provider.invocations == []

    assert len(llm_provider.requests) == 1


@pytest.mark.anyio
async def test_resume_exact_approval_executes_without_llm_regeneration() -> None:
    """Resume executes the exact frozen plan without another model decision."""
    response = _tool_response(_change_proposal())

    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(response)

    pending = await service.start(
        _turn_request(
            run_id="run-resume",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    call_id = pending.pending_approval_call_ids[0]

    resumed = await service.resume(
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

    assert resumed.status is (AgentTurnStatus.TOOL_RESULTS_AVAILABLE)

    assert resumed.pending_approval_call_ids == ()

    assert len(resumed.executions) == 1

    assert len(tool_provider.invocations) == 1

    assert len(llm_provider.requests) == 1


@pytest.mark.anyio
async def test_resume_replay_is_rejected_after_execution() -> None:
    """The same approval continuation cannot execute twice."""
    response = _tool_response(_change_proposal())

    (
        service,
        _,
        tool_provider,
        _,
        _,
    ) = _service(response)

    pending = await service.start(
        _turn_request(
            run_id="run-replay",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    call_id = pending.pending_approval_call_ids[0]

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
        AgentResumeError,
        match="not an active service-issued continuation",
    ):
        await service.resume(
            pending,
            authorization=authorization,
        )

    assert len(tool_provider.invocations) == 1


@pytest.mark.anyio
async def test_mismatched_approval_remains_pending_then_can_succeed() -> None:
    """Bad approval cannot execute and does not destroy the valid continuation."""
    response = _tool_response(_change_proposal())

    (
        service,
        llm_provider,
        tool_provider,
        _,
        _,
    ) = _service(response)

    pending = await service.start(
        _turn_request(
            run_id="run-still-pending",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    still_pending = await service.resume(
        pending,
        authorization=_authorization(
            "change",
            grants=(
                ToolApprovalGrant(
                    call_id="other-call",
                    tool_name="change",
                ),
            ),
        ),
    )

    assert still_pending.status is (AgentTurnStatus.APPROVAL_REQUIRED)

    assert tool_provider.invocations == []

    valid_call_id = still_pending.pending_approval_call_ids[0]

    completed = await service.resume(
        still_pending,
        authorization=_authorization(
            "change",
            grants=(
                ToolApprovalGrant(
                    call_id=valid_call_id,
                    tool_name="change",
                ),
            ),
        ),
    )

    assert completed.status is (AgentTurnStatus.TOOL_RESULTS_AVAILABLE)

    assert len(tool_provider.invocations) == 1

    assert len(llm_provider.requests) == 1


@pytest.mark.anyio
async def test_unissued_pending_plan_cannot_be_resumed() -> None:
    """A fabricated result cannot become an active continuation."""
    response = _tool_response(_change_proposal())

    (
        service,
        _,
        tool_provider,
        _,
        _,
    ) = _service(response)

    proposal = response.tool_calls[0]

    forged = AgentTurnResult(
        run_id="run-forged",
        status=(AgentTurnStatus.APPROVAL_REQUIRED),
        response=response,
        planned_steps=(
            AgentPlannedToolCall(
                step_number=1,
                proposal=proposal,
                invocation=ToolInvocation(
                    call_id=("agent:run-forged:tool:1:forged"),
                    tool_name="change",
                    arguments=(proposal.arguments),
                ),
            ),
        ),
        pending_approval_call_ids=("agent:run-forged:tool:1:forged",),
    )

    with pytest.raises(
        AgentResumeError,
        match="not an active service-issued continuation",
    ):
        await service.resume(
            forged,
            authorization=_authorization(
                "change",
                grants=(
                    ToolApprovalGrant(
                        call_id=("agent:run-forged:tool:1:forged"),
                        tool_name="change",
                    ),
                ),
            ),
        )

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_all_calls_preflight_before_first_side_effect() -> None:
    """A later invalid call prevents an earlier valid call from executing."""
    response = _tool_response(
        _lookup_proposal("valid"),
        _lookup_proposal(
            123,
        ),
    )

    (
        service,
        _,
        tool_provider,
        _,
        _,
    ) = _service(response)

    with pytest.raises(
        ToolInputValidationError,
        match="invalid type: query",
    ):
        await service.start(
            _turn_request(
                max_steps=2,
            ),
            authorization=_authorization("lookup"),
        )

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_non_terminal_non_tool_finish_reason_fails_closed() -> None:
    """Length/filter/other responses are not treated as completed."""
    response = LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content="partial",
        ),
        finish_reason=FinishReason.LENGTH,
    )

    (
        service,
        _,
        tool_provider,
        _,
        _,
    ) = _service(response)

    with pytest.raises(
        AgentResponseError,
        match="requires STOP or TOOL_CALLS",
    ):
        await service.start(
            _turn_request(),
            authorization=_authorization("lookup"),
        )

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_call_id_factory_must_not_reuse_provider_id() -> None:
    """Even an injected factory cannot collapse provider and execution identity."""
    response = _tool_response(
        _lookup_proposal(
            provider_call_id="provider-call-1",
        )
    )

    llm_provider = SyntheticLLMProvider(response)

    tool_provider = SyntheticToolProvider()

    registry = ToolRegistry(
        (tool_provider,),
        policies=_policies(),
    )

    service = ControlledAgentService(
        llm_provider=llm_provider,
        registry=registry,
        tool_execution=(ToolExecutionService(registry)),
        call_id_factory=(lambda _run_id, _step: "provider-call-1"),
    )

    with pytest.raises(
        AgentOrchestrationError,
        match="must not reuse provider_call_id",
    ):
        await service.start(
            _turn_request(),
            authorization=_authorization("lookup"),
        )

    assert tool_provider.invocations == []


@pytest.mark.anyio
async def test_call_id_factory_must_generate_unique_batch_ids() -> None:
    """Duplicate platform invocation identities fail before execution."""
    response = _tool_response(
        _lookup_proposal("first"),
        _lookup_proposal("second"),
    )

    llm_provider = SyntheticLLMProvider(response)

    tool_provider = SyntheticToolProvider()

    registry = ToolRegistry(
        (tool_provider,),
        policies=_policies(),
    )

    service = ControlledAgentService(
        llm_provider=llm_provider,
        registry=registry,
        tool_execution=(ToolExecutionService(registry)),
        call_id_factory=(lambda _run_id, _step: "duplicate-call"),
    )

    with pytest.raises(
        AgentOrchestrationError,
        match="unique invocation identities",
    ):
        await service.start(
            _turn_request(
                max_steps=2,
            ),
            authorization=_authorization("lookup"),
        )

    assert tool_provider.invocations == []


def test_tool_execution_validate_has_no_provider_side_effect() -> None:
    """Public preflight validates without calling the provider."""
    (
        _,
        _,
        tool_provider,
        _,
        tool_execution,
    ) = _service(_stop_response())

    invocation = ToolInvocation(
        call_id="manual-call-1",
        tool_name="lookup",
        arguments=(
            ToolArgument(
                name="query",
                value="synthetic",
            ),
        ),
    )

    preflight = tool_execution.validate(
        invocation,
        authorization=_authorization("lookup"),
    )

    assert isinstance(
        preflight,
        ToolExecutionPreflight,
    )

    assert preflight.invocation == invocation
    assert preflight.approval_required is False
    assert preflight.approval_used is False
    assert tool_provider.invocations == []


def test_feature_13_service_surface_is_public() -> None:
    """Controlled agent-turn primitives remain exported by services."""
    import ai_engineering_agent_platform.services as services

    expected = {
        "AgentOrchestrationError",
        "AgentPlannedToolCall",
        "AgentResponseError",
        "AgentResumeError",
        "AgentStepLimitError",
        "AgentTurnRequest",
        "AgentTurnResult",
        "AgentTurnStatus",
        "ControlledAgentService",
        "MAX_AGENT_TURN_STEPS",
        "ToolExecutionPreflight",
    }

    assert expected <= set(services.__all__)

    for name in expected:
        assert hasattr(
            services,
            name,
        )


class _MemoryAgentContinuationStore:
    """Deterministic CAS store used to exercise fresh-service continuation."""

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
async def test_durable_turn_continuation_resumes_in_fresh_service() -> None:
    """Fresh service resumes the exact frozen plan without LLM regeneration."""
    store = _MemoryAgentContinuationStore()
    response = _tool_response(_change_proposal())

    (
        original_service,
        original_llm,
        original_tool_provider,
        _,
        _,
    ) = _service(response)

    original_service._continuation_store = store

    pending = await original_service.start(
        _turn_request(
            run_id="durable-turn",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    assert pending.status is AgentTurnStatus.APPROVAL_REQUIRED
    assert len(original_llm.requests) == 1
    assert original_tool_provider.invocations == []

    call_id = pending.pending_approval_call_ids[0]

    (
        fresh_service,
        fresh_llm,
        fresh_tool_provider,
        _,
        _,
    ) = _service(response)

    fresh_service._continuation_store = store

    resumed = await fresh_service.resume(
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

    assert resumed.status is AgentTurnStatus.TOOL_RESULTS_AVAILABLE
    assert len(resumed.executions) == 1

    assert fresh_llm.requests == []
    assert len(fresh_tool_provider.invocations) == 1

    replay_service, _, replay_tool_provider, _, _ = _service(response)

    replay_service._continuation_store = store

    with pytest.raises(
        AgentResumeError,
        match="not an active service-issued continuation",
    ):
        await replay_service.resume(
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

    assert replay_tool_provider.invocations == []


@pytest.mark.anyio
async def test_durable_turn_continuation_rejects_forged_snapshot() -> None:
    """Durable issuance does not make caller-fabricated state trusted."""
    store = _MemoryAgentContinuationStore()
    response = _tool_response(_change_proposal())

    service, _, _, _, _ = _service(response)

    service._continuation_store = store

    pending = await service.start(
        _turn_request(
            run_id="durable-forged",
            exposed_tool_names=("change",),
        ),
        authorization=_authorization("change"),
    )

    forged = AgentTurnResult(
        run_id="forged-run",
        status=pending.status,
        response=pending.response,
        planned_steps=pending.planned_steps,
        pending_approval_call_ids=(pending.pending_approval_call_ids),
    )

    fresh, _, tool_provider, _, _ = _service(response)

    fresh._continuation_store = store

    with pytest.raises(
        AgentResumeError,
        match="not an active service-issued continuation",
    ):
        await fresh.resume(
            forged,
            authorization=_authorization("change"),
        )

    assert tool_provider.invocations == []
