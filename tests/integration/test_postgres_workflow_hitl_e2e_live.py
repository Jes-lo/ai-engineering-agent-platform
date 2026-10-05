"""Live PostgreSQL end-to-end workflow HITL approval integration.

This test crosses the real durable boundaries introduced by Feature 19:

WorkflowEngine
    -> PostgreSQL workflow checkpoint
    -> bounded agent LOOP continuation
    -> PostgreSQL agent continuation
    -> durable approval request
    -> PostgreSQL approval state
    -> authenticated approval decision
    -> exact grant consumption
    -> parent LOOP replay-gate consumption
    -> ToolExecutionService
    -> provider side effect
    -> terminal model response
    -> completed workflow checkpoint

The test deliberately does not claim exactly-once external side effects.
It verifies the implemented consume-before-side-effect replay boundary and
fail-closed stale/binding behavior.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_engineering_agent_platform.adapters.postgres.agent_continuation_store import (
    PostgreSQLAgentContinuationStore,
)
from ai_engineering_agent_platform.adapters.postgres.approval_store import (
    PostgreSQLApprovalStore,
)
from ai_engineering_agent_platform.adapters.postgres.workflow_state_store import (
    PostgreSQLWorkflowStateStore,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.contracts import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    MessageRole,
    ProviderDescriptor,
    ProviderKind,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
)
from ai_engineering_agent_platform.domain.guardrails import (
    GuardrailCategory,
    GuardrailSeverity,
    GuardrailStage,
)
from ai_engineering_agent_platform.domain.workflow import (
    WorkflowDefinition,
    WorkflowEventType,
    WorkflowRunResult,
    WorkflowRunStatus,
    WorkflowStepDefinition,
    WorkflowStepStatus,
)
from ai_engineering_agent_platform.runtime.postgres import (
    postgres_pool_runtime,
)
from ai_engineering_agent_platform.services.agent import (
    ControlledAgentService,
)
from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationKind,
    agent_continuation_key,
)
from ai_engineering_agent_platform.services.agent_loop import (
    AgentLoopRequest,
    BoundedAgentLoopService,
)
from ai_engineering_agent_platform.services.approval import (
    APPROVAL_DECISION_SCOPE,
    ApprovalAuthorizationError,
    ApprovalDecision,
    ApprovalService,
    ApprovalStatus,
    AuthenticatedApprovalActor,
)
from ai_engineering_agent_platform.services.guardrails import (
    GuardrailLiteralPattern,
    GuardrailPolicy,
    GuardrailService,
    LiteralPatternGuardrailRule,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolExecutionAuthorization,
    ToolExecutionPolicy,
    ToolExecutionService,
    ToolRegistry,
)
from ai_engineering_agent_platform.services.workflow import (
    WorkflowApprovalPending,
    WorkflowEngine,
    WorkflowExecutorRegistry,
    WorkflowResumeError,
    WorkflowStepContext,
)
from ai_engineering_agent_platform.services.workflow_executors import (
    AgentLoopWorkflowExecutor,
    WorkflowAgentApprovalResumeError,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_PLATFORM_RUN_POSTGRES_INTEGRATION") != "1",
    reason="live PostgreSQL integration is opt-in",
)


WORKFLOW_RUN_ID = "workflow-run:live-hitl-e2e"
WORKFLOW_ID = "workflow-live-hitl-e2e"
WORKFLOW_VERSION = "1"
STEP_ID = "agent-step"
EXECUTOR_NAME = "agent-live"
AGENT_RUN_ID = "agent-live-hitl-e2e"
CALL_ID = "live-hitl-call-1"
APPROVAL_ID = "approval:live-hitl-e2e"

NOW = datetime(
    2026,
    10,
    5,
    16,
    0,
    tzinfo=UTC,
)


class SequenceLLMProvider:
    """Return a deterministic response sequence and record requests."""

    def __init__(
        self,
        responses: tuple[
            LLMResponse,
            ...,
        ],
    ) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        return ProviderDescriptor(
            name="live-hitl-llm",
            kind=ProviderKind.LLM,
        )

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.requests.append(request)

        if not self._responses:
            raise AssertionError("unexpected extra LLM generation")

        return self._responses.pop(0)


@dataclass
class RecordingToolProvider:
    """Record the synthetic side effect after an optional ordering gate."""

    before_execute: (
        Callable[
            [ToolInvocation],
            Awaitable[None],
        ]
        | None
    ) = None

    invocations: list[ToolInvocation] = field(default_factory=list)

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        return ProviderDescriptor(
            name="live-hitl-tools",
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
                name="change",
                description=("Perform one synthetic approval-required change."),
                parameters=(),
            ),
        )

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        if self.before_execute is not None:
            await self.before_execute(invocation)

        self.invocations.append(invocation)

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content="live-hitl-change-applied",
        )


def _load_runtime_env(
    path: Path,
) -> dict[
    str,
    str,
]:
    """Load the isolated runtime-role environment without printing secrets."""
    values: dict[
        str,
        str,
    ] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        key, separator, value = line.partition("=")

        if separator != "=" or not key:
            raise RuntimeError(f"Malformed environment line in {path}")

        if key in values:
            raise RuntimeError(f"Duplicate environment key: {key}")

        values[key] = value

    if not values:
        raise RuntimeError("PostgreSQL runtime environment is empty")

    return values


def _tool_call_response() -> LLMResponse:
    """Return the exact pre-approval model decision."""
    return LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content="",
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            LLMToolCall(
                tool_name="change",
                arguments=(),
                provider_call_id=("provider-live-hitl-1"),
            ),
        ),
    )


def _stop_response() -> LLMResponse:
    """Return the terminal response after the approved tool result."""
    return LLMResponse(
        model="synthetic-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content=("Approved workflow completed."),
        ),
        finish_reason=FinishReason.STOP,
    )


def _authorization() -> ToolExecutionAuthorization:
    """Return base workflow tool authorization without fabricated grants."""
    return ToolExecutionAuthorization(
        allowed_tool_names=("change",),
    )


def _definition() -> WorkflowDefinition:
    """Return the single-step workflow used by the live HITL scenario."""
    return WorkflowDefinition(
        workflow_id=WORKFLOW_ID,
        version=WORKFLOW_VERSION,
        steps=(
            WorkflowStepDefinition(
                step_id=STEP_ID,
                executor_name=(EXECUTOR_NAME),
            ),
        ),
    )


def _request_factory(
    _context: WorkflowStepContext,
) -> AgentLoopRequest:
    """Return the bounded agent request owned by the trusted workflow adapter."""
    return AgentLoopRequest(
        run_id=AGENT_RUN_ID,
        llm_request=LLMRequest(
            model="synthetic-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content=("Perform the approved synthetic change."),
                ),
            ),
        ),
        exposed_tool_names=("change",),
        max_model_turns=2,
        max_tool_calls=1,
    )


def _agent_guardrail_service() -> GuardrailService:
    """Return non-blocking runtime guards for the durable HITL E2E."""
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
                rule_id="live-hitl-agent-runtime-rule",
                stages=stages,
                patterns=(
                    GuardrailLiteralPattern(
                        literal=("__live_hitl_guardrail_marker_not_present__"),
                        category=GuardrailCategory.OTHER,
                        severity=GuardrailSeverity.LOW,
                        message="configured live HITL guardrail signal",
                    ),
                ),
            ),
        ),
    )


def _loop(
    *,
    llm_provider: SequenceLLMProvider,
    tool_provider: RecordingToolProvider,
    continuation_store: PostgreSQLAgentContinuationStore,
) -> BoundedAgentLoopService:
    """Build the real controlled agent/tool stack over one durable LOOP store."""
    registry = ToolRegistry(
        (tool_provider,),
        policies=(
            ToolExecutionPolicy(
                tool_name="change",
                enabled=True,
                requires_approval=True,
            ),
        ),
    )

    agent = ControlledAgentService(
        llm_provider=llm_provider,
        guardrail_service=_agent_guardrail_service(),
        registry=registry,
        tool_execution=(ToolExecutionService(registry)),
        continuation_store=(continuation_store),
        call_id_factory=(lambda _run_id, _step: CALL_ID),
    )

    return BoundedAgentLoopService(
        agent_service=agent,
        continuation_store=(continuation_store),
    )


def _executor(
    *,
    agent_loop: BoundedAgentLoopService,
    approval_service: ApprovalService,
) -> AgentLoopWorkflowExecutor:
    """Build the approval-aware workflow agent executor."""
    return AgentLoopWorkflowExecutor(
        name=EXECUTOR_NAME,
        agent_loop=agent_loop,
        authorization=_authorization(),
        request_factory=(_request_factory),
        approval_service=(approval_service),
        approval_ttl_seconds=600,
    )


async def _exercise() -> None:
    runtime_path = Path(os.environ["AI_PLATFORM_POSTGRES_INTEGRATION_RUNTIME_ENV"])

    os.environ.update(_load_runtime_env(runtime_path))

    settings = Settings()

    if settings.postgres_user != "ai_platform_runtime":
        raise RuntimeError("Integration test must use runtime role")

    definition = _definition()

    async with postgres_pool_runtime(settings) as pool:
        # --------------------------------------------------------------
        # PROCESS 1:
        # Start a real workflow and durably pause before tool execution.
        # --------------------------------------------------------------

        workflow_store_1 = PostgreSQLWorkflowStateStore(pool)

        approval_store_1 = PostgreSQLApprovalStore(pool)

        continuation_store_1 = PostgreSQLAgentContinuationStore(pool=pool)

        approval_service_1 = ApprovalService(
            store=approval_store_1,
            approval_id_factory=(lambda: APPROVAL_ID),
            clock=lambda: NOW,
        )

        initial_llm = SequenceLLMProvider((_tool_call_response(),))

        initial_tools = RecordingToolProvider()

        initial_loop = _loop(
            llm_provider=initial_llm,
            tool_provider=initial_tools,
            continuation_store=(continuation_store_1),
        )

        initial_executor = _executor(
            agent_loop=initial_loop,
            approval_service=(approval_service_1),
        )

        initial_engine = WorkflowEngine(
            WorkflowExecutorRegistry((initial_executor,)),
            run_id_factory=(lambda: WORKFLOW_RUN_ID),
            state_store=(workflow_store_1),
        )

        pending = await initial_engine.run(definition)

        assert isinstance(
            pending,
            WorkflowApprovalPending,
        )

        assert pending.state.status is WorkflowRunStatus.AWAITING_APPROVAL

        assert len(pending.state.steps) == 1

        paused_step = pending.state.steps[0]

        assert paused_step.status is WorkflowStepStatus.AWAITING_APPROVAL

        pause = paused_step.approval_pause

        assert pause is not None

        assert pause.continuation_identity == (CALL_ID,)

        assert pause.approval_ids == (APPROVAL_ID,)

        assert pending.events[-1].event_type is WorkflowEventType.STEP_AWAITING_APPROVAL

        # One model decision created the frozen proposal.
        # No provider/tool side effect occurred before approval.
        assert len(initial_llm.requests) == 1

        assert initial_tools.invocations == []

        persisted_pause = await workflow_store_1.load(WORKFLOW_RUN_ID)

        assert persisted_pause is not None

        assert persisted_pause.state.status is WorkflowRunStatus.AWAITING_APPROVAL

        approval = await approval_store_1.load(APPROVAL_ID)

        assert approval is not None
        assert approval.status is ApprovalStatus.PENDING

        assert approval.run_id == WORKFLOW_RUN_ID

        assert approval.workflow_id == WORKFLOW_ID

        assert approval.workflow_version == WORKFLOW_VERSION

        assert approval.step_id == STEP_ID

        assert approval.invocation.call_id == CALL_ID

        assert approval.invocation.tool_name == "change"

        loop_key = agent_continuation_key(
            AgentContinuationKind.LOOP,
            pause.continuation_identity,
        )

        turn_key = agent_continuation_key(
            AgentContinuationKind.TURN,
            pause.continuation_identity,
        )

        frozen_loop = await continuation_store_1.load(
            kind=AgentContinuationKind.LOOP,
            identity=(pause.continuation_identity),
        )

        assert frozen_loop is not None

        # 19C3 invariant:
        # bounded workflow pause owns only the parent LOOP replay record.
        assert (
            await continuation_store_1.load(
                kind=AgentContinuationKind.TURN,
                identity=(pause.continuation_identity),
            )
            is None
        )

        async with pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    run_status,
                    checkpoint_format
                FROM ai_platform.workflow_checkpoints
                WHERE run_id = %s
                """,
                (WORKFLOW_RUN_ID,),
            )

            workflow_row = await cursor.fetchone()

            cursor = await connection.execute(
                """
                SELECT approval_status
                FROM ai_platform.approval_requests
                WHERE approval_id = %s
                """,
                (APPROVAL_ID,),
            )

            approval_row = await cursor.fetchone()

            cursor = await connection.execute(
                """
                SELECT
                    continuation_kind,
                    continuation_status
                FROM ai_platform.agent_continuations
                WHERE continuation_key = %s
                """,
                (loop_key,),
            )

            loop_row = await cursor.fetchone()

            cursor = await connection.execute(
                """
                SELECT continuation_status
                FROM ai_platform.agent_continuations
                WHERE continuation_key = %s
                """,
                (turn_key,),
            )

            turn_row = await cursor.fetchone()

        assert workflow_row == (
            WorkflowRunStatus.AWAITING_APPROVAL.value,
            2,
        )

        assert approval_row == (ApprovalStatus.PENDING.value,)

        assert loop_row == (
            AgentContinuationKind.LOOP.value,
            "active",
        )

        assert turn_row is None

        # --------------------------------------------------------------
        # AUTHENTICATED DECISION BOUNDARY:
        # A caller without approval:decide cannot approve.
        # --------------------------------------------------------------

        decision_store = PostgreSQLApprovalStore(pool)

        decision_service = ApprovalService(
            store=decision_store,
            clock=(lambda: NOW + timedelta(seconds=1)),
        )

        unauthorized_actor = AuthenticatedApprovalActor(
            subject="reviewer-live",
            client_id="integration-client",
            authentication_method="oidc",
            scopes=("workflow:read",),
        )

        with pytest.raises(ApprovalAuthorizationError):
            await decision_service.decide(
                APPROVAL_ID,
                actor=(unauthorized_actor),
                decision=(ApprovalDecision.APPROVE),
                reason=("must not be accepted"),
            )

        still_pending = await decision_service.get(APPROVAL_ID)

        assert still_pending.status is ApprovalStatus.PENDING

        actor = AuthenticatedApprovalActor(
            subject="reviewer-live",
            client_id="integration-client",
            authentication_method="oidc",
            scopes=(APPROVAL_DECISION_SCOPE,),
        )

        approved = await decision_service.decide(
            APPROVAL_ID,
            actor=actor,
            decision=(ApprovalDecision.APPROVE),
            reason=("live integration reviewed"),
        )

        assert approved.status is ApprovalStatus.APPROVED

        assert approved.decision is ApprovalDecision.APPROVE

        assert approved.decided_by == actor

        # --------------------------------------------------------------
        # PROCESS 2:
        # Fresh adapters/services simulate restart from only durable state.
        # --------------------------------------------------------------

        workflow_store_2 = PostgreSQLWorkflowStateStore(pool)

        approval_store_2 = PostgreSQLApprovalStore(pool)

        continuation_store_2 = PostgreSQLAgentContinuationStore(pool=pool)

        approval_service_2 = ApprovalService(
            store=approval_store_2,
            clock=(lambda: NOW + timedelta(seconds=2)),
        )

        resumed_llm = SequenceLLMProvider((_stop_response(),))

        ordering_checks: list[str] = []

        async def assert_consumed_before_tool_side_effect(
            invocation: ToolInvocation,
        ) -> None:
            # No model regeneration is allowed before execution of the
            # already-frozen approved tool proposal.
            assert resumed_llm.requests == []

            assert invocation.call_id == CALL_ID

            async with pool.connection() as connection:
                cursor = await connection.execute(
                    """
                    SELECT approval_status
                    FROM ai_platform.approval_requests
                    WHERE approval_id = %s
                    """,
                    (APPROVAL_ID,),
                )

                current_approval = await cursor.fetchone()

                cursor = await connection.execute(
                    """
                    SELECT continuation_status
                    FROM ai_platform.agent_continuations
                    WHERE continuation_key = %s
                    """,
                    (loop_key,),
                )

                current_loop = await cursor.fetchone()

            # Both durable replay/authority records are consumed before
            # the external tool provider is entered.
            assert current_approval == (ApprovalStatus.CONSUMED.value,)

            assert current_loop == ("consumed",)

            ordering_checks.append(invocation.call_id)

        resumed_tools = RecordingToolProvider(
            before_execute=(assert_consumed_before_tool_side_effect)
        )

        resumed_loop = _loop(
            llm_provider=resumed_llm,
            tool_provider=resumed_tools,
            continuation_store=(continuation_store_2),
        )

        resumed_executor = _executor(
            agent_loop=resumed_loop,
            approval_service=(approval_service_2),
        )

        resumed_engine = WorkflowEngine(
            WorkflowExecutorRegistry((resumed_executor,)),
            state_store=(workflow_store_2),
        )

        restarted_checkpoint = await workflow_store_2.load(WORKFLOW_RUN_ID)

        assert restarted_checkpoint is not None

        assert restarted_checkpoint.state.status is WorkflowRunStatus.AWAITING_APPROVAL

        restarted_loop = await continuation_store_2.load(
            kind=AgentContinuationKind.LOOP,
            identity=(pause.continuation_identity),
        )

        assert restarted_loop == frozen_loop

        assert (
            await continuation_store_2.load(
                kind=AgentContinuationKind.TURN,
                identity=(pause.continuation_identity),
            )
            is None
        )

        # --------------------------------------------------------------
        # WRONG BINDING:
        # Even an approved record cannot be consumed under a different
        # workflow version.
        # --------------------------------------------------------------

        wrong_context = WorkflowStepContext(
            run_id=WORKFLOW_RUN_ID,
            workflow_id=WORKFLOW_ID,
            workflow_version="2",
            step_id=STEP_ID,
        )

        with pytest.raises(WorkflowAgentApprovalResumeError):
            await resumed_executor.resume_approval(
                step=(definition.steps[0]),
                context=wrong_context,
                pause=pause,
            )

        after_wrong_binding = await approval_service_2.get(APPROVAL_ID)

        assert after_wrong_binding.status is ApprovalStatus.APPROVED

        assert (
            await continuation_store_2.load(
                kind=AgentContinuationKind.LOOP,
                identity=(pause.continuation_identity),
            )
            == frozen_loop
        )

        assert resumed_tools.invocations == []

        assert resumed_llm.requests == []

        # --------------------------------------------------------------
        # CORRECT RESUME:
        # Workflow loads the AWAITING checkpoint, executor validates exact
        # binding, consumes approval evidence, consumes LOOP replay state,
        # executes the tool, then obtains the terminal LLM response.
        # --------------------------------------------------------------

        result = await resumed_engine.resume_approval(
            definition,
            run_id=WORKFLOW_RUN_ID,
        )

        assert isinstance(
            result,
            WorkflowRunResult,
        )

        assert result.run_id == WORKFLOW_RUN_ID

        assert len(result.executions) == 1

        assert result.executions[0].step_id == STEP_ID

        assert result.state is not None

        assert result.state.status is WorkflowRunStatus.COMPLETED

        assert result.trace is not None

        assert result.trace.events[-1].event_type is WorkflowEventType.RUN_COMPLETED

        assert ordering_checks == [CALL_ID]

        assert len(resumed_tools.invocations) == 1

        assert resumed_tools.invocations[0].call_id == CALL_ID

        # Exactly one post-tool model generation occurs.
        # The frozen pre-approval decision was not regenerated.
        assert len(resumed_llm.requests) == 1

        consumed_approval = await approval_service_2.get(APPROVAL_ID)

        assert consumed_approval.status is ApprovalStatus.CONSUMED

        assert consumed_approval.decided_by == actor

        assert (
            await continuation_store_2.load(
                kind=AgentContinuationKind.LOOP,
                identity=(pause.continuation_identity),
            )
            is None
        )

        assert (
            await continuation_store_2.load(
                kind=AgentContinuationKind.TURN,
                identity=(pause.continuation_identity),
            )
            is None
        )

        terminal_checkpoint = await workflow_store_2.load(WORKFLOW_RUN_ID)

        assert terminal_checkpoint is not None

        assert terminal_checkpoint.state.status is WorkflowRunStatus.COMPLETED

        # Durable tombstones remain visible in PostgreSQL and prove the
        # approval/LOOP replay gates were consumed rather than deleted.
        async with pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT run_status
                FROM ai_platform.workflow_checkpoints
                WHERE run_id = %s
                """,
                (WORKFLOW_RUN_ID,),
            )

            final_workflow_row = await cursor.fetchone()

            cursor = await connection.execute(
                """
                SELECT approval_status
                FROM ai_platform.approval_requests
                WHERE approval_id = %s
                """,
                (APPROVAL_ID,),
            )

            final_approval_row = await cursor.fetchone()

            cursor = await connection.execute(
                """
                SELECT
                    continuation_kind,
                    continuation_status
                FROM ai_platform.agent_continuations
                WHERE continuation_key = %s
                """,
                (loop_key,),
            )

            final_loop_row = await cursor.fetchone()

            cursor = await connection.execute(
                """
                SELECT continuation_status
                FROM ai_platform.agent_continuations
                WHERE continuation_key = %s
                """,
                (turn_key,),
            )

            final_turn_row = await cursor.fetchone()

        assert final_workflow_row == (WorkflowRunStatus.COMPLETED.value,)

        assert final_approval_row == (ApprovalStatus.CONSUMED.value,)

        assert final_loop_row == (
            AgentContinuationKind.LOOP.value,
            "consumed",
        )

        # No child TURN row was ever created by the bounded-loop workflow.
        assert final_turn_row is None

        # --------------------------------------------------------------
        # REPLAY:
        # Neither the consumed agent continuation nor the terminal workflow
        # checkpoint may be replayed automatically.
        # --------------------------------------------------------------

        correct_context = WorkflowStepContext(
            run_id=WORKFLOW_RUN_ID,
            workflow_id=WORKFLOW_ID,
            workflow_version=(WORKFLOW_VERSION),
            step_id=STEP_ID,
        )

        with pytest.raises(WorkflowAgentApprovalResumeError):
            await resumed_executor.resume_approval(
                step=(definition.steps[0]),
                context=correct_context,
                pause=pause,
            )

        with pytest.raises(WorkflowResumeError):
            await resumed_engine.resume_approval(
                definition,
                run_id=WORKFLOW_RUN_ID,
            )

        # Replay attempts did not duplicate any provider side effect or
        # regenerate any model response.
        assert len(resumed_tools.invocations) == 1

        assert len(resumed_llm.requests) == 1


def test_live_authenticated_workflow_hitl_end_to_end() -> None:
    """Exercise durable authenticated workflow approval across restart."""
    asyncio.run(_exercise())
