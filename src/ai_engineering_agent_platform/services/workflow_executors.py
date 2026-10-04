"""Controlled application executors for the project-owned workflow engine."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.domain import (
    RetrievalRequest,
)
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowStepDefinition,
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
    ControlledToolExecutionResult,
    ToolExecutionAuthorization,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowStepContext,
)

_EXECUTOR_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$")


class WorkflowExecutorAdapterError(Exception):
    """Base error for workflow-to-service adapters."""


class WorkflowAgentApprovalRequiredError(WorkflowExecutorAdapterError):
    """Raised when a workflow agent reaches a non-durable approval pause."""


@dataclass(
    frozen=True,
    slots=True,
)
class WorkflowToolRequest:
    """Tool request without caller-controlled platform execution identity."""

    tool_name: str
    arguments: tuple[
        ToolArgument,
        ...,
    ] = ()

    def __post_init__(
        self,
    ) -> None:
        """Validate one portable workflow tool request."""
        if (
            not isinstance(
                self.tool_name,
                str,
            )
            or not self.tool_name.strip()
        ):
            raise ValueError("tool_name must be a non-empty string")

        if not isinstance(
            self.arguments,
            tuple,
        ):
            raise ValueError("arguments must be a tuple")

        if any(
            not isinstance(
                argument,
                ToolArgument,
            )
            for argument in self.arguments
        ):
            raise ValueError("arguments must contain ToolArgument values")

        names = tuple(argument.name for argument in self.arguments)

        if len(names) != len(set(names)):
            raise ValueError("workflow tool argument names must be unique")


class ToolExecutionRunner(Protocol):
    """Minimal controlled tool-service surface required by workflows."""

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> ControlledToolExecutionResult:
        """Execute one controlled invocation."""


class RAGRunner(Protocol):
    """Minimal end-to-end RAG surface required by workflows."""

    async def run(
        self,
        request: RetrievalRequest,
    ) -> RAGResult:
        """Execute one RAG request."""


class AgentLoopRunner(Protocol):
    """Minimal bounded-agent start surface required by workflows."""

    async def start(
        self,
        request: AgentLoopRequest,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentLoopResult:
        """Start one bounded agent loop."""


type WorkflowToolRequestFactory = Callable[
    [
        WorkflowStepContext,
    ],
    WorkflowToolRequest,
]

type WorkflowRAGRequestFactory = Callable[
    [
        WorkflowStepContext,
    ],
    RetrievalRequest,
]

type WorkflowAgentRequestFactory = Callable[
    [
        WorkflowStepContext,
    ],
    AgentLoopRequest,
]


def _require_executor_name(
    value: object,
) -> str:
    """Return one workflow-compatible executor registry name."""
    if (
        not isinstance(
            value,
            str,
        )
        or _EXECUTOR_NAME_PATTERN.fullmatch(value) is None
    ):
        raise ValueError("executor name must be a portable workflow identifier")

    return value


def _validate_step_binding(
    *,
    expected_executor_name: str,
    step: WorkflowStepDefinition,
    context: WorkflowStepContext,
) -> None:
    """Require executor, step, and workflow context identity to agree."""
    if not isinstance(
        step,
        WorkflowStepDefinition,
    ):
        raise WorkflowExecutorAdapterError("step must be a WorkflowStepDefinition")

    if not isinstance(
        context,
        WorkflowStepContext,
    ):
        raise WorkflowExecutorAdapterError("context must be a WorkflowStepContext")

    if step.executor_name != expected_executor_name:
        raise WorkflowExecutorAdapterError(
            "workflow step is bound to a different executor"
        )

    if context.step_id != step.step_id:
        raise WorkflowExecutorAdapterError("workflow context step identity mismatch")


def _workflow_tool_call_id(
    context: WorkflowStepContext,
) -> str:
    """Create internal tool identity from platform workflow state."""
    return "workflow:" + context.run_id + ":" + context.step_id + ":tool"


class ToolWorkflowExecutor:
    """Adapt one workflow step to ToolExecutionService controls.

    Tool approval evidence is deliberately not accepted by this adapter.
    Durable authenticated HITL must exist before a workflow can execute
    approval-required tools.
    """

    def __init__(
        self,
        *,
        name: str,
        tool_execution: ToolExecutionRunner,
        authorization: ToolExecutionAuthorization,
        request_factory: WorkflowToolRequestFactory,
    ) -> None:
        """Store the controlled tool boundary and trusted request factory."""
        self._name = _require_executor_name(name)

        if not isinstance(
            authorization,
            ToolExecutionAuthorization,
        ):
            raise ValueError("authorization must be ToolExecutionAuthorization")

        if authorization.approval_grants:
            raise ValueError("workflow tool executor cannot accept approval grants")

        if not callable(request_factory):
            raise ValueError("request_factory must be callable")

        self._tool_execution = tool_execution

        self._authorization = authorization

        self._request_factory = request_factory

    @property
    def name(
        self,
    ) -> str:
        """Return stable workflow executor identity."""
        return self._name

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> ControlledToolExecutionResult:
        """Execute one workflow-generated tool request exactly once."""
        _validate_step_binding(
            expected_executor_name=self._name,
            step=step,
            context=context,
        )

        request = self._request_factory(context)

        if not isinstance(
            request,
            WorkflowToolRequest,
        ):
            raise WorkflowExecutorAdapterError(
                "tool request factory must return WorkflowToolRequest"
            )

        invocation = ToolInvocation(
            call_id=_workflow_tool_call_id(context),
            tool_name=request.tool_name,
            arguments=request.arguments,
        )

        return await self._tool_execution.execute(
            invocation,
            authorization=self._authorization,
        )


class RAGWorkflowExecutor:
    """Adapt one workflow step to the existing end-to-end RAG service."""

    def __init__(
        self,
        *,
        name: str,
        rag_service: RAGRunner,
        request_factory: WorkflowRAGRequestFactory,
    ) -> None:
        """Store the RAG boundary and trusted request factory."""
        self._name = _require_executor_name(name)

        if not callable(request_factory):
            raise ValueError("request_factory must be callable")

        self._rag_service = rag_service

        self._request_factory = request_factory

    @property
    def name(
        self,
    ) -> str:
        """Return stable workflow executor identity."""
        return self._name

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> RAGResult:
        """Run one workflow-generated RetrievalRequest exactly once."""
        _validate_step_binding(
            expected_executor_name=self._name,
            step=step,
            context=context,
        )

        request = self._request_factory(context)

        if not isinstance(
            request,
            RetrievalRequest,
        ):
            raise WorkflowExecutorAdapterError(
                "RAG request factory must return RetrievalRequest"
            )

        return await self._rag_service.run(request)


class AgentLoopWorkflowExecutor:
    """Adapt one workflow step to the bounded conversational agent loop.

    This adapter starts a loop but never resumes one. If the bounded agent
    reaches APPROVAL_REQUIRED, execution fails closed because Feature 17 does
    not yet implement durable authenticated HITL or durable workflow resume.
    """

    def __init__(
        self,
        *,
        name: str,
        agent_loop: AgentLoopRunner,
        authorization: ToolExecutionAuthorization,
        request_factory: WorkflowAgentRequestFactory,
    ) -> None:
        """Store the bounded agent boundary and trusted request factory."""
        self._name = _require_executor_name(name)

        if not isinstance(
            authorization,
            ToolExecutionAuthorization,
        ):
            raise ValueError("authorization must be ToolExecutionAuthorization")

        if authorization.approval_grants:
            raise ValueError("workflow agent executor cannot accept approval grants")

        if not callable(request_factory):
            raise ValueError("request_factory must be callable")

        self._agent_loop = agent_loop

        self._authorization = authorization

        self._request_factory = request_factory

    @property
    def name(
        self,
    ) -> str:
        """Return stable workflow executor identity."""
        return self._name

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> AgentLoopResult:
        """Start one bounded agent loop and require terminal completion."""
        _validate_step_binding(
            expected_executor_name=self._name,
            step=step,
            context=context,
        )

        request = self._request_factory(context)

        if not isinstance(
            request,
            AgentLoopRequest,
        ):
            raise WorkflowExecutorAdapterError(
                "agent request factory must return AgentLoopRequest"
            )

        result = await self._agent_loop.start(
            request,
            authorization=self._authorization,
        )

        if result.status is AgentLoopStatus.APPROVAL_REQUIRED:
            raise WorkflowAgentApprovalRequiredError(
                "workflow agent requires durable authenticated approval"
            )

        if result.status is not AgentLoopStatus.COMPLETED:
            raise WorkflowExecutorAdapterError(
                "bounded agent returned unsupported workflow state"
            )

        return result
