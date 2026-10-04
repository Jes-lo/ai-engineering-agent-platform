"""Tests for controlled workflow adapters over existing platform services."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import pytest

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
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
from ai_engineering_agent_platform.domain import (
    GroundedAnswer,
    GroundedAnswerStatus,
    RetrievalRequest,
    RetrievalResponse,
)
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowStepDefinition,
    WorkflowStepExecution,
)
from ai_engineering_agent_platform.services.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
    AgentLoopStatus,
)
from ai_engineering_agent_platform.services.rag import (
    RAGResult,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolApprovalGrant,
    ToolApprovalRequiredError,
    ToolExecutionAuthorization,
    ToolExecutionPolicy,
    ToolExecutionService,
    ToolRegistry,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowStepContext,
)
from ai_engineering_agent_platform.services.workflow_executors import (
    AgentLoopWorkflowExecutor,
    RAGWorkflowExecutor,
    ToolWorkflowExecutor,
    WorkflowAgentApprovalRequiredError,
    WorkflowToolRequest,
)


@dataclass
class SyntheticToolProvider:
    """Deterministic tool provider behind the real control service."""

    invocations: list[ToolInvocation] = field(default_factory=list)

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        return ProviderDescriptor(
            name="workflow-synthetic-tools",
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(
        self,
    ) -> tuple[
        ToolDefinition,
        ...,
    ]:
        return (
            ToolDefinition(
                name="lookup",
                description="Synthetic lookup.",
                parameters=(
                    ToolParameter(
                        name="query",
                        parameter_type=ToolParameterType.STRING,
                        required=True,
                    ),
                ),
            ),
            ToolDefinition(
                name="change",
                description="Synthetic approval-required change.",
                parameters=(),
            ),
        )

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        self.invocations.append(invocation)

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content=("tool-output:" + invocation.tool_name),
        )


@dataclass
class RecordingRAGRunner:
    """Structural RAG runner recording exact requests."""

    result: RAGResult
    requests: list[RetrievalRequest] = field(default_factory=list)

    async def run(
        self,
        request: RetrievalRequest,
    ) -> RAGResult:
        self.requests.append(request)

        return self.result


@dataclass
class RecordingAgentLoopRunner:
    """Structural bounded-agent runner recording starts."""

    result: AgentLoopResult
    starts: list[
        tuple[
            AgentLoopRequest,
            ToolExecutionAuthorization,
        ]
    ] = field(default_factory=list)

    async def start(
        self,
        request: AgentLoopRequest,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentLoopResult:
        self.starts.append(
            (
                request,
                authorization,
            )
        )

        return self.result


def _context(
    *,
    step_id: str,
    dependencies: tuple[
        WorkflowStepExecution,
        ...,
    ] = (),
) -> WorkflowStepContext:
    return WorkflowStepContext(
        run_id="workflow-run:test",
        workflow_id="workflow-test",
        workflow_version="1",
        step_id=step_id,
        dependency_results=dependencies,
    )


def _authorization(
    *tool_names: str,
    grants: tuple[
        ToolApprovalGrant,
        ...,
    ] = (),
) -> ToolExecutionAuthorization:
    return ToolExecutionAuthorization(
        allowed_tool_names=tuple(tool_names),
        approval_grants=grants,
    )


def _tool_service(
    *,
    requires_lookup_approval: bool = False,
) -> tuple[
    ToolExecutionService,
    SyntheticToolProvider,
]:
    provider = SyntheticToolProvider()

    registry = ToolRegistry(
        (provider,),
        policies=(
            ToolExecutionPolicy(
                tool_name="lookup",
                enabled=True,
                requires_approval=(requires_lookup_approval),
            ),
            ToolExecutionPolicy(
                tool_name="change",
                enabled=True,
                requires_approval=True,
            ),
        ),
    )

    return (
        ToolExecutionService(registry),
        provider,
    )


def _rag_result(
    *,
    query: str = "synthetic query",
) -> RAGResult:
    retrieval = RetrievalResponse(
        query=query,
        results=(),
        namespace="workflow",
    )

    answer = GroundedAnswer(
        query=query,
        status=GroundedAnswerStatus.ABSTAINED,
        answer=None,
        citations=(),
        namespace="workflow",
    )

    return RAGResult(
        retrieval=retrieval,
        grounding_input=retrieval,
        answer=answer,
        generation=None,
    )


def _agent_request() -> AgentLoopRequest:
    return AgentLoopRequest(
        run_id="workflow-agent",
        llm_request=LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="Complete the workflow task.",
                ),
            ),
        ),
        exposed_tool_names=(),
        max_model_turns=2,
        max_tool_calls=1,
    )


def _completed_agent_result() -> AgentLoopResult:
    request = _agent_request()

    final_message = LLMMessage(
        role=MessageRole.ASSISTANT,
        content="workflow complete",
    )

    final_response = LLMResponse(
        model="synthetic-model",
        message=final_message,
        finish_reason=FinishReason.STOP,
    )

    return AgentLoopResult(
        request=request,
        status=AgentLoopStatus.COMPLETED,
        messages=(
            *request.llm_request.messages,
            final_message,
        ),
        model_turns_used=1,
        tool_calls_used=0,
        final_response=final_response,
    )


@pytest.mark.anyio
async def test_tool_executor_uses_real_controlled_service_once() -> None:
    service, provider = _tool_service()

    executor = ToolWorkflowExecutor(
        name="tool",
        tool_execution=service,
        authorization=_authorization("lookup"),
        request_factory=(
            lambda context: WorkflowToolRequest(
                tool_name="lookup",
                arguments=(
                    ToolArgument(
                        name="query",
                        value=context.workflow_id,
                    ),
                ),
            )
        ),
    )

    step = WorkflowStepDefinition(
        step_id="tool-step",
        executor_name="tool",
    )

    result = await executor.execute(
        step=step,
        context=_context(step_id="tool-step"),
    )

    assert result.result.content == ("tool-output:lookup")

    assert len(provider.invocations) == 1

    invocation = provider.invocations[0]

    assert invocation.call_id == ("workflow:workflow-run:test:tool-step:tool")

    assert invocation.tool_name == ("lookup")

    assert invocation.arguments == (
        ToolArgument(
            name="query",
            value="workflow-test",
        ),
    )


def test_tool_executor_rejects_preinjected_approval_grants() -> None:
    service, _ = _tool_service()

    with pytest.raises(
        ValueError,
        match="cannot accept approval grants",
    ):
        ToolWorkflowExecutor(
            name="tool",
            tool_execution=service,
            authorization=_authorization(
                "lookup",
                grants=(
                    ToolApprovalGrant(
                        call_id="some-call",
                        tool_name="lookup",
                    ),
                ),
            ),
            request_factory=(lambda context: WorkflowToolRequest(tool_name="lookup")),
        )


@pytest.mark.anyio
async def test_tool_executor_fails_closed_for_approval_required_tool() -> None:
    service, provider = _tool_service(requires_lookup_approval=True)

    executor = ToolWorkflowExecutor(
        name="tool",
        tool_execution=service,
        authorization=_authorization("lookup"),
        request_factory=(
            lambda context: WorkflowToolRequest(
                tool_name="lookup",
                arguments=(
                    ToolArgument(
                        name="query",
                        value="synthetic",
                    ),
                ),
            )
        ),
    )

    step = WorkflowStepDefinition(
        step_id="approval-step",
        executor_name="tool",
    )

    with pytest.raises(
        ToolApprovalRequiredError,
    ):
        await executor.execute(
            step=step,
            context=_context(step_id="approval-step"),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_rag_executor_builds_request_from_declared_dependency() -> None:
    expected_result = _rag_result(query="tool-output")

    runner = RecordingRAGRunner(result=expected_result)

    dependency = WorkflowStepExecution(
        step_id="tool-step",
        executor_name="tool",
        output="tool-output",
    )

    def build_request(
        context: WorkflowStepContext,
    ) -> RetrievalRequest:
        source = context.result_for("tool-step")

        return RetrievalRequest(
            query=cast(
                str,
                source.output,
            ),
            top_k=3,
            namespace="workflow",
        )

    executor = RAGWorkflowExecutor(
        name="rag",
        rag_service=runner,
        request_factory=build_request,
    )

    step = WorkflowStepDefinition(
        step_id="rag-step",
        executor_name="rag",
        depends_on=("tool-step",),
    )

    result = await executor.execute(
        step=step,
        context=_context(
            step_id="rag-step",
            dependencies=(dependency,),
        ),
    )

    assert result is expected_result

    assert runner.requests == [
        RetrievalRequest(
            query="tool-output",
            top_k=3,
            namespace="workflow",
        )
    ]


@pytest.mark.anyio
async def test_agent_executor_starts_bounded_loop_once() -> None:
    expected = _completed_agent_result()

    runner = RecordingAgentLoopRunner(result=expected)

    authorization = _authorization()

    executor = AgentLoopWorkflowExecutor(
        name="agent",
        agent_loop=runner,
        authorization=authorization,
        request_factory=(lambda context: _agent_request()),
    )

    step = WorkflowStepDefinition(
        step_id="agent-step",
        executor_name="agent",
    )

    result = await executor.execute(
        step=step,
        context=_context(step_id="agent-step"),
    )

    assert result is expected

    assert runner.starts == [
        (
            _agent_request(),
            authorization,
        )
    ]


def test_agent_executor_rejects_preinjected_approval_grants() -> None:
    runner = RecordingAgentLoopRunner(result=_completed_agent_result())

    with pytest.raises(
        ValueError,
        match="cannot accept approval grants",
    ):
        AgentLoopWorkflowExecutor(
            name="agent",
            agent_loop=runner,
            authorization=_authorization(
                grants=(
                    ToolApprovalGrant(
                        call_id="some-call",
                        tool_name="change",
                    ),
                )
            ),
            request_factory=(lambda context: _agent_request()),
        )


@pytest.mark.anyio
async def test_agent_approval_pause_fails_closed_without_resume() -> None:
    pending = cast(
        AgentLoopResult,
        SimpleNamespace(status=(AgentLoopStatus.APPROVAL_REQUIRED)),
    )

    runner = RecordingAgentLoopRunner(result=pending)

    executor = AgentLoopWorkflowExecutor(
        name="agent",
        agent_loop=runner,
        authorization=_authorization("lookup"),
        request_factory=(lambda context: _agent_request()),
    )

    step = WorkflowStepDefinition(
        step_id="agent-step",
        executor_name="agent",
    )

    with pytest.raises(
        WorkflowAgentApprovalRequiredError,
        match="durable authenticated approval",
    ):
        await executor.execute(
            step=step,
            context=_context(step_id="agent-step"),
        )

    assert len(runner.starts) == 1


@pytest.mark.anyio
async def test_executor_rejects_step_binding_mismatch() -> None:
    runner = RecordingRAGRunner(result=_rag_result())

    executor = RAGWorkflowExecutor(
        name="rag",
        rag_service=runner,
        request_factory=(
            lambda context: RetrievalRequest(
                query="synthetic",
                top_k=1,
            )
        ),
    )

    step = WorkflowStepDefinition(
        step_id="step",
        executor_name="different",
    )

    with pytest.raises(
        Exception,
        match="different executor",
    ):
        await executor.execute(
            step=step,
            context=_context(step_id="step"),
        )

    assert runner.requests == []
