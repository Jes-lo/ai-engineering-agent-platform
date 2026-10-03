"""Bounded conversational agent loop over the controlled agent-turn boundary."""

import re
from dataclasses import dataclass
from enum import StrEnum

from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMAssistantToolCallMessage,
    LLMConversationMessage,
    LLMRequest,
    LLMResponse,
    LLMToolResultMessage,
)
from ai_engineering_agent_platform.services.agent import (
    MAX_AGENT_TURN_STEPS,
    AgentResponseError,
    AgentStepLimitError,
    AgentTurnRequest,
    AgentTurnResult,
    AgentTurnStatus,
    ControlledAgentService,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolExecutionAuthorization,
)

MAX_AGENT_LOOP_MODEL_TURNS = 8
MAX_AGENT_LOOP_TOOL_CALLS = 8

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AgentLoopError(Exception):
    """Base error for bounded conversational orchestration."""


class AgentLoopBudgetError(AgentLoopError):
    """Raised when a global conversational loop budget is exhausted."""


class AgentLoopResumeError(AgentLoopError):
    """Raised when a loop continuation cannot be resumed safely."""


class AgentLoopStatus(StrEnum):
    """Normalized bounded-loop state."""

    COMPLETED = "completed"
    APPROVAL_REQUIRED = "approval_required"


@dataclass(frozen=True, slots=True)
class AgentLoopRequest:
    """Input and absolute budgets for one bounded conversational run."""

    run_id: str
    llm_request: LLMRequest
    exposed_tool_names: tuple[str, ...] = ()
    max_model_turns: int = 4
    max_tool_calls: int = 4

    def __post_init__(self) -> None:
        """Validate loop identity, tool exposure, and hard budgets."""
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
            raise ValueError(
                "agent-loop llm_request must not contain caller-supplied tools"
            )

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
                self.max_model_turns,
                int,
            )
            or isinstance(
                self.max_model_turns,
                bool,
            )
            or self.max_model_turns <= 0
            or self.max_model_turns > MAX_AGENT_LOOP_MODEL_TURNS
        ):
            raise ValueError(
                f"max_model_turns must be between 1 and {MAX_AGENT_LOOP_MODEL_TURNS}"
            )

        if (
            not isinstance(
                self.max_tool_calls,
                int,
            )
            or isinstance(
                self.max_tool_calls,
                bool,
            )
            or self.max_tool_calls <= 0
            or self.max_tool_calls > MAX_AGENT_LOOP_TOOL_CALLS
        ):
            raise ValueError(
                f"max_tool_calls must be between 1 and {MAX_AGENT_LOOP_TOOL_CALLS}"
            )


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Immutable completed result or resumable conversational approval pause."""

    request: AgentLoopRequest
    status: AgentLoopStatus
    messages: tuple[
        LLMConversationMessage,
        ...,
    ]
    model_turns_used: int
    tool_calls_used: int
    final_response: LLMResponse | None = None
    pending_turn: AgentTurnResult | None = None

    def __post_init__(self) -> None:
        """Validate loop state, budgets, and provider-valid transcript."""
        if not isinstance(
            self.request,
            AgentLoopRequest,
        ):
            raise ValueError("request must be an AgentLoopRequest")

        if not isinstance(
            self.status,
            AgentLoopStatus,
        ):
            raise ValueError("status must be an AgentLoopStatus")

        if not isinstance(
            self.messages,
            tuple,
        ):
            raise ValueError("messages must be a tuple")

        if (
            not isinstance(
                self.model_turns_used,
                int,
            )
            or isinstance(
                self.model_turns_used,
                bool,
            )
            or self.model_turns_used <= 0
            or self.model_turns_used > self.request.max_model_turns
        ):
            raise ValueError("model_turns_used is outside the request budget")

        if (
            not isinstance(
                self.tool_calls_used,
                int,
            )
            or isinstance(
                self.tool_calls_used,
                bool,
            )
            or self.tool_calls_used < 0
            or self.tool_calls_used > self.request.max_tool_calls
        ):
            raise ValueError("tool_calls_used is outside the request budget")

        # Reconstructing a request validates the committed transcript,
        # including exact tool-call/tool-result ordering.
        LLMRequest(
            model=(self.request.llm_request.model),
            messages=self.messages,
            temperature=(self.request.llm_request.temperature),
            max_output_tokens=(self.request.llm_request.max_output_tokens),
        )

        if self.status is AgentLoopStatus.COMPLETED:
            if self.pending_turn is not None:
                raise ValueError("completed loop cannot retain a pending turn")

            if self.final_response is None:
                raise ValueError("completed loop requires final_response")

            if self.final_response.finish_reason is not FinishReason.STOP:
                raise ValueError("completed loop requires terminal STOP")

            if not self.messages or self.messages[-1] != self.final_response.message:
                raise ValueError("completed transcript must end with final response")

            return

        if self.status is AgentLoopStatus.APPROVAL_REQUIRED:
            if self.final_response is not None:
                raise ValueError("approval-required loop cannot contain final_response")

            if self.pending_turn is None:
                raise ValueError("approval-required loop requires pending_turn")

            if self.pending_turn.status is not AgentTurnStatus.APPROVAL_REQUIRED:
                raise ValueError("pending_turn must require approval")

            return

        raise ValueError("unsupported agent loop status")


class BoundedAgentLoopService:
    """Run model/tool cycles only through ControlledAgentService.

    Tool outputs remain untrusted model context. Any subsequent tool proposal
    must pass the same registration, exposure, authorization, approval,
    preflight, and execution controls on every iteration.

    Loop continuation state is process-local and in-memory.
    """

    def __init__(
        self,
        *,
        agent_service: ControlledAgentService,
    ) -> None:
        """Store the existing controlled agent-turn boundary."""
        if not isinstance(
            agent_service,
            ControlledAgentService,
        ):
            raise ValueError("agent_service must be a ControlledAgentService")

        self._agent_service = agent_service

        self._pending_loops: dict[
            tuple[str, ...],
            AgentLoopResult,
        ] = {}

    async def start(
        self,
        request: AgentLoopRequest,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentLoopResult:
        """Run a new bounded conversational loop."""
        if not isinstance(
            request,
            AgentLoopRequest,
        ):
            raise AgentLoopError("request must be an AgentLoopRequest")

        return await self._run(
            request=request,
            messages=request.llm_request.messages,
            model_turns_used=0,
            tool_calls_used=0,
            authorization=authorization,
        )

    async def resume(
        self,
        pending: AgentLoopResult,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> AgentLoopResult:
        """Resume one exact active loop continuation without model regeneration."""
        if not isinstance(
            pending,
            AgentLoopResult,
        ):
            raise AgentLoopResumeError("pending must be an AgentLoopResult")

        if (
            pending.status is not AgentLoopStatus.APPROVAL_REQUIRED
            or pending.pending_turn is None
        ):
            raise AgentLoopResumeError("only an approval-required loop can be resumed")

        key = self._pending_key(pending.pending_turn)

        issued = self._pending_loops.get(key)

        if issued is None or issued != pending:
            raise AgentLoopResumeError(
                "pending loop is not an active service-issued continuation"
            )

        # Consume the loop continuation before delegating to the controlled
        # resume boundary. A failure after tool execution begins must not make
        # the conversational continuation automatically replayable.
        del self._pending_loops[key]

        resumed_turn = await self._agent_service.resume(
            pending.pending_turn,
            authorization=authorization,
        )

        if resumed_turn.status is AgentTurnStatus.APPROVAL_REQUIRED:
            still_pending = AgentLoopResult(
                request=pending.request,
                status=(AgentLoopStatus.APPROVAL_REQUIRED),
                messages=pending.messages,
                model_turns_used=(pending.model_turns_used),
                tool_calls_used=(pending.tool_calls_used),
                pending_turn=resumed_turn,
            )

            self._register_pending(still_pending)

            return still_pending

        if resumed_turn.status is not AgentTurnStatus.TOOL_RESULTS_AVAILABLE:
            raise AgentLoopResumeError("resumed turn returned an invalid loop state")

        next_tool_calls_used = pending.tool_calls_used + len(resumed_turn.executions)

        if next_tool_calls_used > pending.request.max_tool_calls:
            raise AgentLoopBudgetError(
                "resumed tool batch exceeds global tool-call budget"
            )

        messages = self._append_tool_exchange(
            pending.messages,
            resumed_turn,
        )

        return await self._run(
            request=pending.request,
            messages=messages,
            model_turns_used=(pending.model_turns_used),
            tool_calls_used=(next_tool_calls_used),
            authorization=authorization,
        )

    async def _run(
        self,
        *,
        request: AgentLoopRequest,
        messages: tuple[
            LLMConversationMessage,
            ...,
        ],
        model_turns_used: int,
        tool_calls_used: int,
        authorization: ToolExecutionAuthorization,
    ) -> AgentLoopResult:
        """Continue until STOP, approval pause, or an absolute budget failure."""
        current_messages = messages
        current_model_turns = model_turns_used
        current_tool_calls = tool_calls_used

        while True:
            if current_model_turns >= request.max_model_turns:
                raise AgentLoopBudgetError(
                    "global model-turn budget exhausted before terminal STOP"
                )

            remaining_tool_calls = request.max_tool_calls - current_tool_calls

            if remaining_tool_calls < 0:
                raise AgentLoopBudgetError(
                    "global tool-call budget is already exceeded"
                )

            if remaining_tool_calls == 0:
                exposed_tool_names: tuple[
                    str,
                    ...,
                ] = ()

                turn_max_steps = 1
            else:
                exposed_tool_names = request.exposed_tool_names

                turn_max_steps = min(
                    remaining_tool_calls,
                    MAX_AGENT_TURN_STEPS,
                )

            turn_llm_request = LLMRequest(
                model=(request.llm_request.model),
                messages=current_messages,
                temperature=(request.llm_request.temperature),
                max_output_tokens=(request.llm_request.max_output_tokens),
            )

            turn_request = AgentTurnRequest(
                run_id=request.run_id,
                llm_request=turn_llm_request,
                exposed_tool_names=(exposed_tool_names),
                max_steps=turn_max_steps,
            )

            try:
                turn = await self._agent_service.start(
                    turn_request,
                    authorization=authorization,
                )
            except AgentStepLimitError as exc:
                raise AgentLoopBudgetError(
                    "model proposal exceeds remaining global tool-call budget"
                ) from exc
            except AgentResponseError as exc:
                if remaining_tool_calls == 0:
                    raise AgentLoopBudgetError(
                        "model proposed another tool after "
                        "global tool-call budget exhaustion"
                    ) from exc

                raise

            current_model_turns += 1

            if turn.status is AgentTurnStatus.COMPLETED:
                final_messages = (
                    *current_messages,
                    turn.response.message,
                )

                return AgentLoopResult(
                    request=request,
                    status=(AgentLoopStatus.COMPLETED),
                    messages=final_messages,
                    model_turns_used=(current_model_turns),
                    tool_calls_used=(current_tool_calls),
                    final_response=(turn.response),
                )

            if turn.status is AgentTurnStatus.APPROVAL_REQUIRED:
                pending = AgentLoopResult(
                    request=request,
                    status=(AgentLoopStatus.APPROVAL_REQUIRED),
                    messages=current_messages,
                    model_turns_used=(current_model_turns),
                    tool_calls_used=(current_tool_calls),
                    pending_turn=turn,
                )

                self._register_pending(pending)

                return pending

            if turn.status is not AgentTurnStatus.TOOL_RESULTS_AVAILABLE:
                raise AgentLoopError(
                    "controlled agent turn returned an unsupported state"
                )

            executed_count = len(turn.executions)

            current_tool_calls += executed_count

            if current_tool_calls > request.max_tool_calls:
                raise AgentLoopBudgetError("global tool-call budget exceeded")

            current_messages = self._append_tool_exchange(
                current_messages,
                turn,
            )

    @staticmethod
    def _append_tool_exchange(
        messages: tuple[
            LLMConversationMessage,
            ...,
        ],
        turn: AgentTurnResult,
    ) -> tuple[
        LLMConversationMessage,
        ...,
    ]:
        """Append one validated assistant-call/result exchange to history."""
        if turn.status is not AgentTurnStatus.TOOL_RESULTS_AVAILABLE:
            raise AgentLoopError("tool exchange requires executed tool results")

        assistant_message = LLMAssistantToolCallMessage(
            content=(turn.response.message.content),
            tool_calls=(turn.response.tool_calls),
        )

        tool_messages = tuple(
            LLMToolResultMessage(
                result=execution.result,
                provider_call_id=(planned.proposal.provider_call_id),
            )
            for planned, execution in zip(
                turn.planned_steps,
                turn.executions,
                strict=True,
            )
        )

        return (
            *messages,
            assistant_message,
            *tool_messages,
        )

    def _register_pending(
        self,
        pending: AgentLoopResult,
    ) -> None:
        """Register one loop continuation exactly once in process memory."""
        if pending.pending_turn is None:
            raise AgentLoopResumeError("pending loop has no pending turn")

        key = self._pending_key(pending.pending_turn)

        if key in self._pending_loops:
            raise AgentLoopResumeError("pending loop identity is already active")

        self._pending_loops[key] = pending

    @staticmethod
    def _pending_key(
        pending_turn: AgentTurnResult,
    ) -> tuple[
        str,
        ...,
    ]:
        """Return the controlled pending-call identity for one loop pause."""
        if pending_turn.status is not AgentTurnStatus.APPROVAL_REQUIRED:
            raise AgentLoopResumeError("pending turn does not require approval")

        if not pending_turn.pending_approval_call_ids:
            raise AgentLoopResumeError("pending turn has no approval call IDs")

        return pending_turn.pending_approval_call_ids
