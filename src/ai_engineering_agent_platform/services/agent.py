"""Controlled single-turn agent orchestration over LLM and tool boundaries."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    ToolDefinition,
    ToolInvocation,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ControlledToolExecutionResult,
    ToolApprovalRequiredError,
    ToolAuthorizationError,
    ToolExecutionAuthorization,
    ToolExecutionService,
    ToolRegistry,
)

MAX_AGENT_TURN_STEPS = 8

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AgentOrchestrationError(Exception):
    """Base error for controlled agent-turn orchestration."""


class AgentResponseError(AgentOrchestrationError):
    """Raised when an LLM response is invalid for this orchestration layer."""


class AgentStepLimitError(AgentOrchestrationError):
    """Raised before execution when a proposal batch exceeds its step budget."""


class AgentResumeError(AgentOrchestrationError):
    """Raised when a pending agent result cannot be resumed safely."""


class AgentTurnStatus(StrEnum):
    """Normalized status of one controlled agent turn."""

    COMPLETED = "completed"
    APPROVAL_REQUIRED = "approval_required"
    TOOL_RESULTS_AVAILABLE = "tool_results_available"


@dataclass(frozen=True, slots=True)
class AgentTurnRequest:
    """Input for one bounded LLM-to-tool orchestration turn.

    run_id is caller correlation metadata. It is not execution authority and
    is not sufficient by itself to create a reusable approval identity.
    """

    run_id: str
    llm_request: LLMRequest
    exposed_tool_names: tuple[str, ...] = ()
    max_steps: int = 4

    def __post_init__(self) -> None:
        """Validate turn identity, tool exposure, and step budget."""
        if (
            not isinstance(
                self.run_id,
                str,
            )
            or _RUN_ID_PATTERN.fullmatch(self.run_id) is None
        ):
            raise ValueError("run_id must use 1-64 portable identifier characters")

        if not isinstance(
            self.llm_request,
            LLMRequest,
        ):
            raise ValueError("llm_request must be an LLMRequest")

        if self.llm_request.tools:
            raise ValueError("agent llm_request must not contain caller-supplied tools")

        if not isinstance(
            self.exposed_tool_names,
            tuple,
        ):
            raise ValueError("exposed_tool_names must be a tuple")

        for tool_name in self.exposed_tool_names:
            if (
                not isinstance(
                    tool_name,
                    str,
                )
                or not tool_name.strip()
            ):
                raise ValueError("exposed tool names must be non-empty strings")

        if len(self.exposed_tool_names) != len(set(self.exposed_tool_names)):
            raise ValueError("exposed tool names must be unique")

        if (
            not isinstance(
                self.max_steps,
                int,
            )
            or isinstance(
                self.max_steps,
                bool,
            )
            or self.max_steps <= 0
            or self.max_steps > MAX_AGENT_TURN_STEPS
        ):
            raise ValueError(f"max_steps must be between 1 and {MAX_AGENT_TURN_STEPS}")


@dataclass(frozen=True, slots=True)
class AgentPlannedToolCall:
    """One platform-owned invocation derived from an untrusted LLM proposal."""

    step_number: int
    proposal: LLMToolCall
    invocation: ToolInvocation

    def __post_init__(self) -> None:
        """Validate proposal-to-invocation identity without granting authority."""
        if (
            not isinstance(
                self.step_number,
                int,
            )
            or isinstance(
                self.step_number,
                bool,
            )
            or self.step_number <= 0
        ):
            raise ValueError("step_number must be positive")

        if not isinstance(
            self.proposal,
            LLMToolCall,
        ):
            raise ValueError("proposal must be an LLMToolCall")

        if not isinstance(
            self.invocation,
            ToolInvocation,
        ):
            raise ValueError("invocation must be a ToolInvocation")

        if self.proposal.tool_name != self.invocation.tool_name:
            raise ValueError("proposal tool name does not match invocation")

        if self.proposal.arguments != self.invocation.arguments:
            raise ValueError("proposal arguments do not match invocation")


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """Immutable result or resumable approval pause for one agent turn."""

    run_id: str
    status: AgentTurnStatus
    response: LLMResponse
    planned_steps: tuple[
        AgentPlannedToolCall,
        ...,
    ] = ()
    executions: tuple[
        ControlledToolExecutionResult,
        ...,
    ] = ()
    pending_approval_call_ids: tuple[
        str,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        """Validate status-specific orchestration invariants."""
        if (
            not isinstance(
                self.run_id,
                str,
            )
            or _RUN_ID_PATTERN.fullmatch(self.run_id) is None
        ):
            raise ValueError("run_id must use 1-64 portable identifier characters")

        if not isinstance(
            self.status,
            AgentTurnStatus,
        ):
            raise ValueError("status must be an AgentTurnStatus")

        if not isinstance(
            self.response,
            LLMResponse,
        ):
            raise ValueError("response must be an LLMResponse")

        if not isinstance(
            self.planned_steps,
            tuple,
        ) or any(
            not isinstance(
                step,
                AgentPlannedToolCall,
            )
            for step in self.planned_steps
        ):
            raise ValueError("planned_steps must contain AgentPlannedToolCall values")

        if not isinstance(
            self.executions,
            tuple,
        ) or any(
            not isinstance(
                execution,
                ControlledToolExecutionResult,
            )
            for execution in self.executions
        ):
            raise ValueError(
                "executions must contain ControlledToolExecutionResult values"
            )

        if not isinstance(
            self.pending_approval_call_ids,
            tuple,
        ):
            raise ValueError("pending_approval_call_ids must be a tuple")

        if len(self.pending_approval_call_ids) != len(
            set(self.pending_approval_call_ids)
        ):
            raise ValueError("pending approval call IDs must be unique")

        if self.status is AgentTurnStatus.COMPLETED:
            if self.response.finish_reason is not FinishReason.STOP:
                raise ValueError("completed agent turn requires STOP response")

            if self.planned_steps or self.executions or self.pending_approval_call_ids:
                raise ValueError(
                    "completed agent turn cannot contain tool execution state"
                )

            return

        if self.response.finish_reason is not FinishReason.TOOL_CALLS:
            raise ValueError("tool-bearing agent turn requires TOOL_CALLS response")

        if len(self.planned_steps) != len(self.response.tool_calls):
            raise ValueError("planned steps must match response tool-call count")

        for step, proposal in zip(
            self.planned_steps,
            self.response.tool_calls,
            strict=True,
        ):
            if step.proposal != proposal:
                raise ValueError("planned step proposal does not match response")

        planned_call_ids = tuple(step.invocation.call_id for step in self.planned_steps)

        if len(planned_call_ids) != len(set(planned_call_ids)):
            raise ValueError("planned invocation call IDs must be unique")

        if self.status is AgentTurnStatus.APPROVAL_REQUIRED:
            if self.executions:
                raise ValueError("approval-required turn cannot contain executions")

            if not self.pending_approval_call_ids:
                raise ValueError("approval-required turn must identify pending calls")

            if not set(self.pending_approval_call_ids).issubset(set(planned_call_ids)):
                raise ValueError("pending approval IDs must belong to planned calls")

            return

        if self.status is AgentTurnStatus.TOOL_RESULTS_AVAILABLE:
            if self.pending_approval_call_ids:
                raise ValueError("executed turn cannot retain pending approvals")

            if len(self.executions) != len(self.planned_steps):
                raise ValueError("executions must match planned step count")

            for step, execution in zip(
                self.planned_steps,
                self.executions,
                strict=True,
            ):
                if execution.invocation != step.invocation:
                    raise ValueError("execution does not match planned invocation")

            return

        raise ValueError("unsupported agent turn status")


def _default_call_id(
    run_id: str,
    step_number: int,
) -> str:
    """Create fresh platform-owned invocation identity."""
    return f"agent:{run_id}:tool:{step_number}:{uuid4().hex}"


class ControlledAgentService:
    """Coordinate one bounded LLM decision and controlled tool batch.

    This service intentionally does not feed tool results back to the LLM and
    therefore is not yet a full conversational agent loop.

    Pending approval state is held in memory. It is neither durable across
    process restarts nor an authenticated human-approval system.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        registry: ToolRegistry,
        tool_execution: ToolExecutionService,
        call_id_factory: (
            Callable[
                [str, int],
                str,
            ]
            | None
        ) = None,
    ) -> None:
        """Store explicit model, registry, execution, and identity boundaries."""
        self._llm_provider = llm_provider
        self._registry = registry
        self._tool_execution = tool_execution
        self._call_id_factory = (
            _default_call_id if call_id_factory is None else call_id_factory
        )
        self._pending_plans: dict[
            tuple[str, ...],
            AgentTurnResult,
        ] = {}

    async def start(
        self,
        request: AgentTurnRequest,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentTurnResult:
        """Run one LLM decision and execute or pause its bounded tool plan."""
        if not isinstance(
            request,
            AgentTurnRequest,
        ):
            raise AgentOrchestrationError("request must be an AgentTurnRequest")

        if not isinstance(
            authorization,
            ToolExecutionAuthorization,
        ):
            raise ToolAuthorizationError(
                "authorization must be a ToolExecutionAuthorization"
            )

        definitions = self._resolve_exposed_tools(
            request,
            authorization=authorization,
        )

        runtime_request = LLMRequest(
            model=request.llm_request.model,
            messages=(request.llm_request.messages),
            temperature=(request.llm_request.temperature),
            max_output_tokens=(request.llm_request.max_output_tokens),
            tools=definitions,
        )

        response = await self._llm_provider.generate(runtime_request)

        if response.finish_reason is FinishReason.STOP:
            return AgentTurnResult(
                run_id=request.run_id,
                status=(AgentTurnStatus.COMPLETED),
                response=response,
            )

        if response.finish_reason is not FinishReason.TOOL_CALLS:
            raise AgentResponseError("agent turn requires STOP or TOOL_CALLS response")

        if len(response.tool_calls) > request.max_steps:
            raise AgentStepLimitError(
                "LLM tool-call proposal count exceeds the agent turn step limit"
            )

        exposed_tool_names = set(request.exposed_tool_names)

        for proposal in response.tool_calls:
            if proposal.tool_name not in exposed_tool_names:
                raise AgentResponseError(
                    "LLM proposed a tool outside "
                    "the agent exposure set: " + proposal.tool_name
                )

        planned_steps = self._build_planned_steps(
            run_id=request.run_id,
            proposals=response.tool_calls,
        )

        missing_approvals = self._preflight_plan(
            planned_steps,
            authorization=authorization,
        )

        if missing_approvals:
            pending = AgentTurnResult(
                run_id=request.run_id,
                status=(AgentTurnStatus.APPROVAL_REQUIRED),
                response=response,
                planned_steps=planned_steps,
                pending_approval_call_ids=(missing_approvals),
            )

            self._register_pending(pending)

            return pending

        return await self._execute_plan(
            run_id=request.run_id,
            response=response,
            planned_steps=planned_steps,
            authorization=authorization,
        )

    async def resume(
        self,
        pending: AgentTurnResult,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentTurnResult:
        """Resume one exact active service-issued approval continuation."""
        if not isinstance(
            pending,
            AgentTurnResult,
        ):
            raise AgentResumeError("pending must be an AgentTurnResult")

        if pending.status is not AgentTurnStatus.APPROVAL_REQUIRED:
            raise AgentResumeError("only an approval-required turn can be resumed")

        key = self._plan_key(pending.planned_steps)

        issued_pending = self._pending_plans.get(key)

        if issued_pending is None or issued_pending != pending:
            raise AgentResumeError(
                "pending turn is not an active service-issued continuation"
            )

        missing_approvals = self._preflight_plan(
            pending.planned_steps,
            authorization=authorization,
        )

        if missing_approvals:
            still_pending = AgentTurnResult(
                run_id=pending.run_id,
                status=(AgentTurnStatus.APPROVAL_REQUIRED),
                response=pending.response,
                planned_steps=(pending.planned_steps),
                pending_approval_call_ids=(missing_approvals),
            )

            self._pending_plans[key] = still_pending

            return still_pending

        # Consume the continuation before provider execution. If execution
        # fails partway through the batch, automatic replay is deliberately
        # prohibited because earlier tool side effects may already exist.
        del self._pending_plans[key]

        return await self._execute_plan(
            run_id=pending.run_id,
            response=pending.response,
            planned_steps=(pending.planned_steps),
            authorization=authorization,
        )

    def _resolve_exposed_tools(
        self,
        request: AgentTurnRequest,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> tuple[
        ToolDefinition,
        ...,
    ]:
        """Resolve registered, enabled, explicitly authorized definitions."""
        definitions: list[ToolDefinition] = []

        for tool_name in request.exposed_tool_names:
            registration = self._registry.registration(tool_name)

            if not registration.policy.enabled:
                raise ToolAuthorizationError(
                    "agent cannot expose a tool disabled by platform policy"
                )

            if not authorization.allows(tool_name):
                raise ToolAuthorizationError(
                    "agent cannot expose a tool outside the execution allowlist"
                )

            definitions.append(registration.definition)

        return tuple(definitions)

    def _build_planned_steps(
        self,
        *,
        run_id: str,
        proposals: tuple[
            LLMToolCall,
            ...,
        ],
    ) -> tuple[
        AgentPlannedToolCall,
        ...,
    ]:
        """Create fresh platform-controlled invocation identities."""
        planned_steps: list[AgentPlannedToolCall] = []

        generated_call_ids: set[str] = set()

        for step_number, proposal in enumerate(
            proposals,
            start=1,
        ):
            call_id = self._call_id_factory(
                run_id,
                step_number,
            )

            if (
                not isinstance(
                    call_id,
                    str,
                )
                or not call_id.strip()
            ):
                raise AgentOrchestrationError(
                    "call_id_factory must return a non-empty string"
                )

            if call_id in generated_call_ids:
                raise AgentOrchestrationError(
                    "call_id_factory must produce unique invocation identities"
                )

            if (
                proposal.provider_call_id is not None
                and call_id == proposal.provider_call_id
            ):
                raise AgentOrchestrationError(
                    "platform call_id must not reuse provider_call_id"
                )

            generated_call_ids.add(call_id)

            planned_steps.append(
                AgentPlannedToolCall(
                    step_number=step_number,
                    proposal=proposal,
                    invocation=ToolInvocation(
                        call_id=call_id,
                        tool_name=(proposal.tool_name),
                        arguments=(proposal.arguments),
                    ),
                )
            )

        return tuple(planned_steps)

    def _preflight_plan(
        self,
        planned_steps: tuple[
            AgentPlannedToolCall,
            ...,
        ],
        *,
        authorization: ToolExecutionAuthorization,
    ) -> tuple[
        str,
        ...,
    ]:
        """Validate every planned call before the first provider side effect."""
        missing_approvals: list[str] = []

        for step in planned_steps:
            try:
                self._tool_execution.validate(
                    step.invocation,
                    authorization=authorization,
                )
            except ToolApprovalRequiredError:
                missing_approvals.append(step.invocation.call_id)

        return tuple(missing_approvals)

    async def _execute_plan(
        self,
        *,
        run_id: str,
        response: LLMResponse,
        planned_steps: tuple[
            AgentPlannedToolCall,
            ...,
        ],
        authorization: ToolExecutionAuthorization,
    ) -> AgentTurnResult:
        """Execute one already-preflighted plan sequentially.

        This is not a transactional batch. A later provider failure does not
        roll back an earlier successful provider side effect.
        """
        executions: list[ControlledToolExecutionResult] = []

        for step in planned_steps:
            executions.append(
                await self._tool_execution.execute(
                    step.invocation,
                    authorization=authorization,
                )
            )

        return AgentTurnResult(
            run_id=run_id,
            status=(AgentTurnStatus.TOOL_RESULTS_AVAILABLE),
            response=response,
            planned_steps=planned_steps,
            executions=tuple(executions),
        )

    def _register_pending(
        self,
        pending: AgentTurnResult,
    ) -> None:
        """Register one active in-memory continuation exactly once."""
        key = self._plan_key(pending.planned_steps)

        if key in self._pending_plans:
            raise AgentResumeError("pending plan identity is already active")

        self._pending_plans[key] = pending

    @staticmethod
    def _plan_key(
        planned_steps: tuple[
            AgentPlannedToolCall,
            ...,
        ],
    ) -> tuple[
        str,
        ...,
    ]:
        """Return immutable continuation identity for a planned tool batch."""
        return tuple(step.invocation.call_id for step in planned_steps)
