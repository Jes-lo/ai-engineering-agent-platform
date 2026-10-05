"""Controlled application executors for the project-owned workflow engine."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.domain import (
    RetrievalRequest,
)
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowApprovalPause,
    WorkflowStepDefinition,
)
from ai_engineering_agent_platform.services.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
    AgentLoopStatus,
)
from ai_engineering_agent_platform.services.approval import (
    ApprovalRequest,
    ApprovalService,
    ApprovalStatus,
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


class WorkflowAgentApprovalResumeError(WorkflowExecutorAdapterError):
    """Raised when durable workflow approval continuation cannot resume safely."""


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


@runtime_checkable
class ResumableAgentLoopRunner(AgentLoopRunner, Protocol):
    """Bounded-agent continuation surface required by workflow HITL."""

    async def load_pending(
        self,
        identity: tuple[str, ...],
    ) -> AgentLoopResult:
        """Load one exact active durable loop continuation."""

    async def resume(
        self,
        pending: AgentLoopResult,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentLoopResult:
        """Resume one frozen loop without regenerating its pending decision."""


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

    ``execute()`` intentionally preserves the pre-HITL fail-closed behavior so
    an older WorkflowEngine cannot accidentally interpret a pause as completed
    output.

    ``execute_with_approval_pause()`` and ``resume_approval()`` form the
    explicit Feature 19 approval-aware surface. WorkflowEngine integration is
    added separately so there is no unsafe intermediate state.

    Durable workflow state receives only ``WorkflowApprovalPause`` references.
    Approval records remain authoritative in ``ApprovalService`` and the
    bounded agent continuation store remains authoritative for frozen loop
    execution state.
    """

    def __init__(
        self,
        *,
        name: str,
        agent_loop: AgentLoopRunner,
        authorization: ToolExecutionAuthorization,
        request_factory: WorkflowAgentRequestFactory,
        approval_service: ApprovalService | None = None,
        approval_ttl_seconds: int = 3600,
    ) -> None:
        """Store controlled agent, authorization, and optional HITL boundary."""
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

        if approval_service is not None and not isinstance(
            approval_service,
            ApprovalService,
        ):
            raise ValueError("approval_service must be ApprovalService or None")

        if (
            not isinstance(
                approval_ttl_seconds,
                int,
            )
            or isinstance(
                approval_ttl_seconds,
                bool,
            )
            or approval_ttl_seconds < 1
            or approval_ttl_seconds > 86400
        ):
            raise ValueError("approval_ttl_seconds must be between 1 and 86400")

        if approval_service is not None and not isinstance(
            agent_loop,
            ResumableAgentLoopRunner,
        ):
            raise ValueError(
                "approval-aware workflow agent requires resumable agent loop"
            )

        self._agent_loop = agent_loop
        self._authorization = authorization
        self._request_factory = request_factory
        self._approval_service = approval_service
        self._approval_ttl_seconds = approval_ttl_seconds

    @property
    def name(
        self,
    ) -> str:
        """Return stable workflow executor identity."""
        return self._name

    def _require_approval_bridge(
        self,
    ) -> tuple[
        ResumableAgentLoopRunner,
        ApprovalService,
    ]:
        """Resolve the configured durable HITL dependencies fail closed."""
        approval_service = self._approval_service

        if approval_service is None:
            raise WorkflowAgentApprovalRequiredError(
                "workflow agent approval bridge is not configured"
            )

        agent_loop = self._agent_loop

        if not isinstance(
            agent_loop,
            ResumableAgentLoopRunner,
        ):
            raise WorkflowAgentApprovalResumeError(
                "workflow agent does not expose durable resume"
            )

        return (
            agent_loop,
            approval_service,
        )

    async def _start(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> AgentLoopResult:
        """Start one exactly bound bounded-agent loop."""
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

        if not isinstance(
            result,
            AgentLoopResult,
        ):
            raise WorkflowExecutorAdapterError(
                "bounded agent returned invalid workflow result"
            )

        if result.status not in (
            AgentLoopStatus.COMPLETED,
            AgentLoopStatus.APPROVAL_REQUIRED,
        ):
            raise WorkflowExecutorAdapterError(
                "bounded agent returned unsupported workflow state"
            )

        return result

    async def execute(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> AgentLoopResult:
        """Preserve legacy fail-closed execution until D2-B engine wiring.

        The historical executor contract treats any bounded-agent
        ``APPROVAL_REQUIRED`` signal as fail-closed, including structural test
        doubles that expose only ``status``. The approval-aware D2 surface remains
        stricter and requires a real ``AgentLoopResult`` before durable approval
        state can be materialized.
        """
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

        # Preserve the original fail-closed signal ordering. This check occurs
        # before concrete-result validation intentionally: an approval pause must
        # never be downgraded into a generic adapter/type failure by an older
        # WorkflowEngine that does not yet understand durable HITL.
        if (
            getattr(
                result,
                "status",
                None,
            )
            is AgentLoopStatus.APPROVAL_REQUIRED
        ):
            raise WorkflowAgentApprovalRequiredError(
                "workflow agent requires durable authenticated approval"
            )

        if not isinstance(
            result,
            AgentLoopResult,
        ):
            raise WorkflowExecutorAdapterError(
                "bounded agent returned invalid workflow result"
            )

        if result.status is not AgentLoopStatus.COMPLETED:
            raise WorkflowExecutorAdapterError(
                "bounded agent returned unsupported workflow state"
            )

        return result

    async def execute_with_approval_pause(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
    ) -> str | WorkflowApprovalPause:
        """Start through the explicit durable approval-aware workflow boundary."""
        result = await self._start(
            step=step,
            context=context,
        )

        if result.status is AgentLoopStatus.COMPLETED:
            return self._workflow_output(result)

        return await self._materialize_approval_pause(
            context=context,
            result=result,
        )

    @staticmethod
    def _workflow_output(
        result: AgentLoopResult,
    ) -> str:
        """Return the bounded durable workflow output for one completed loop."""
        if (
            result.status is not AgentLoopStatus.COMPLETED
            or result.final_response is None
        ):
            raise WorkflowExecutorAdapterError(
                "completed bounded agent result requires final response"
            )

        return result.final_response.message.content

    @staticmethod
    def _pending_invocations(
        result: AgentLoopResult,
    ) -> tuple[
        ToolInvocation,
        ...,
    ]:
        """Return approval-required invocations in frozen pending-call order."""
        if (
            result.status is not AgentLoopStatus.APPROVAL_REQUIRED
            or result.pending_turn is None
        ):
            raise WorkflowAgentApprovalResumeError(
                "bounded agent result is not an approval pause"
            )

        pending_turn = result.pending_turn

        call_ids = pending_turn.pending_approval_call_ids

        if not call_ids:
            raise WorkflowAgentApprovalResumeError(
                "approval pause has no pending approval call IDs"
            )

        by_call_id = {
            planned.invocation.call_id: planned.invocation
            for planned in pending_turn.planned_steps
        }

        if len(by_call_id) != len(pending_turn.planned_steps):
            raise WorkflowAgentApprovalResumeError(
                "frozen tool plan contains duplicate invocation identity"
            )

        try:
            invocations = tuple(by_call_id[call_id] for call_id in call_ids)
        except KeyError as exc:
            raise WorkflowAgentApprovalResumeError(
                "pending approval identity is absent from frozen tool plan"
            ) from exc

        if len(invocations) != len(call_ids):
            raise WorkflowAgentApprovalResumeError(
                "pending approval invocation mapping is incomplete"
            )

        return invocations

    @staticmethod
    def _continuation_identity(
        result: AgentLoopResult,
    ) -> tuple[
        str,
        ...,
    ]:
        """Return the full frozen-plan identity used by the durable loop gate."""
        if (
            result.status is not AgentLoopStatus.APPROVAL_REQUIRED
            or result.pending_turn is None
        ):
            raise WorkflowAgentApprovalResumeError(
                "bounded agent result is not an approval pause"
            )

        identity = tuple(
            planned.invocation.call_id for planned in result.pending_turn.planned_steps
        )

        if not identity:
            raise WorkflowAgentApprovalResumeError(
                "approval pause has empty continuation identity"
            )

        if len(identity) != len(set(identity)):
            raise WorkflowAgentApprovalResumeError(
                "approval pause continuation identity is not unique"
            )

        return identity

    async def _materialize_approval_pause(
        self,
        *,
        context: WorkflowStepContext,
        result: AgentLoopResult,
    ) -> WorkflowApprovalPause:
        """Create durable approvals for exactly the frozen missing invocations."""
        _agent_loop, approval_service = self._require_approval_bridge()

        continuation_identity = self._continuation_identity(result)

        invocations = self._pending_invocations(result)

        approval_ids: list[str] = []

        for invocation in invocations:
            approval = await approval_service.request(
                run_id=context.run_id,
                workflow_id=(context.workflow_id),
                workflow_version=(context.workflow_version),
                step_id=context.step_id,
                invocation=invocation,
                ttl_seconds=(self._approval_ttl_seconds),
            )

            approval_ids.append(approval.approval_id)

        return WorkflowApprovalPause(
            continuation_identity=(continuation_identity),
            approval_ids=tuple(approval_ids),
        )

    @staticmethod
    def _validate_bound_approval(
        *,
        approval: ApprovalRequest,
        context: WorkflowStepContext,
        invocation: ToolInvocation,
    ) -> None:
        """Require one approval record to match the exact workflow/invocation."""
        if (
            approval.run_id != context.run_id
            or approval.workflow_id != context.workflow_id
            or approval.workflow_version != context.workflow_version
            or approval.step_id != context.step_id
            or approval.invocation != invocation
        ):
            raise WorkflowAgentApprovalResumeError(
                "approval request binding differs from frozen workflow pause"
            )

        if approval.status is not ApprovalStatus.APPROVED:
            raise WorkflowAgentApprovalResumeError(
                "approval request is not approved for resume"
            )

    async def resume_approval(
        self,
        *,
        step: WorkflowStepDefinition,
        context: WorkflowStepContext,
        pause: WorkflowApprovalPause,
    ) -> str | WorkflowApprovalPause:
        """Resume one exact authenticated durable approval continuation.

        This method does not authenticate a human and never manufactures
        approval authority. Human decisions must already have passed through
        ``ApprovalService.decide``. The method validates the durable workflow
        binding, consumes approved single-use evidence, constructs structural
        grants, and resumes the frozen bounded-agent continuation.

        All approval records are validated before the first one is consumed.
        Multi-record consumption is not transactional; a crash or persistence
        conflict during consumption may require operational recovery, but tool
        execution is never attempted until the complete grant set exists.
        """
        _validate_step_binding(
            expected_executor_name=self._name,
            step=step,
            context=context,
        )

        if not isinstance(
            pause,
            WorkflowApprovalPause,
        ):
            raise WorkflowAgentApprovalResumeError(
                "pause must be WorkflowApprovalPause"
            )

        agent_loop, approval_service = self._require_approval_bridge()

        try:
            pending = await agent_loop.load_pending(pause.continuation_identity)
        except Exception as exc:
            # Loading occurs before approval consumption or provider execution.
            # Normalize the boundary without treating a stale continuation as
            # a workflow step execution failure.
            raise WorkflowAgentApprovalResumeError(
                "durable agent continuation is not resumable"
            ) from exc

        if not isinstance(
            pending,
            AgentLoopResult,
        ):
            raise WorkflowAgentApprovalResumeError(
                "durable agent continuation has invalid type"
            )

        expected_identity = self._continuation_identity(pending)

        if expected_identity != pause.continuation_identity:
            raise WorkflowAgentApprovalResumeError(
                "workflow pause continuation identity mismatch"
            )

        invocations = self._pending_invocations(pending)

        if len(pause.approval_ids) != len(invocations):
            raise WorkflowAgentApprovalResumeError(
                "workflow pause approval count differs from frozen plan"
            )

        bound: list[
            tuple[
                ApprovalRequest,
                ToolInvocation,
            ]
        ] = []

        for (
            approval_id,
            invocation,
        ) in zip(
            pause.approval_ids,
            invocations,
            strict=True,
        ):
            try:
                approval = await approval_service.get(approval_id)
            except Exception as exc:
                raise WorkflowAgentApprovalResumeError(
                    "workflow approval reference is not loadable"
                ) from exc

            self._validate_bound_approval(
                approval=approval,
                context=context,
                invocation=invocation,
            )

            bound.append(
                (
                    approval,
                    invocation,
                )
            )

        grants = []

        for (
            approval,
            invocation,
        ) in bound:
            try:
                grant = await approval_service.consume_grant(
                    approval.approval_id,
                    invocation=invocation,
                )
            except Exception as exc:
                # Approval consumption occurs before agent/tool execution.
                # Some earlier grants may already be consumed because multiple
                # approval rows are deliberately not claimed transactional.
                raise WorkflowAgentApprovalResumeError(
                    "approved workflow grant could not be consumed"
                ) from exc

            grants.append(grant)

        authorization = ToolExecutionAuthorization(
            allowed_tool_names=(self._authorization.allowed_tool_names),
            approval_grants=tuple(grants),
        )

        result = await agent_loop.resume(
            pending,
            authorization=authorization,
        )

        if not isinstance(
            result,
            AgentLoopResult,
        ):
            raise WorkflowExecutorAdapterError(
                "resumed bounded agent returned invalid result"
            )

        if result.status is AgentLoopStatus.APPROVAL_REQUIRED:
            return await self._materialize_approval_pause(
                context=context,
                result=result,
            )

        if result.status is not AgentLoopStatus.COMPLETED:
            raise WorkflowExecutorAdapterError(
                "resumed bounded agent returned unsupported workflow state"
            )

        return self._workflow_output(result)
