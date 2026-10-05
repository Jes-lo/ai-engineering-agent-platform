"""Live PostgreSQL validation for durable agent continuations."""

import os
from hashlib import sha256

import pytest

from ai_engineering_agent_platform.adapters.postgres.agent_continuation_store import (
    PostgreSQLAgentContinuationStore,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.contracts.llm import (
    FinishReason,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    MessageRole,
)
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.runtime.postgres import (
    postgres_pool_runtime,
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
    AgentContinuationCodec,
)
from ai_engineering_agent_platform.services.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
    AgentLoopStatus,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_PLATFORM_RUN_POSTGRES_INTEGRATION") != "1",
    reason="live PostgreSQL integration is opt-in",
)


def _snapshots() -> tuple[
    AgentTurnResult,
    AgentLoopResult,
    tuple[str, ...],
]:
    argument = ToolArgument(
        name="target",
        value="production",
    )

    proposal = LLMToolCall(
        tool_name="change",
        arguments=(argument,),
        provider_call_id="provider-live",
    )

    invocation = ToolInvocation(
        call_id="agent:live-continuation:tool:1:nonce",
        tool_name="change",
        arguments=(argument,),
    )

    response = LLMResponse(
        model="test-model",
        message=LLMMessage(
            role=MessageRole.ASSISTANT,
            content="approval required",
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(proposal,),
    )

    turn = AgentTurnResult(
        run_id="live-continuation",
        status=AgentTurnStatus.APPROVAL_REQUIRED,
        response=response,
        planned_steps=(
            AgentPlannedToolCall(
                step_number=1,
                proposal=proposal,
                invocation=invocation,
            ),
        ),
        pending_approval_call_ids=(invocation.call_id,),
    )

    request = AgentLoopRequest(
        run_id="live-continuation",
        llm_request=LLMRequest(
            model="test-model",
            messages=(
                LLMMessage(
                    role=MessageRole.USER,
                    content="change production",
                ),
            ),
        ),
        exposed_tool_names=("change",),
        max_model_turns=4,
        max_tool_calls=4,
    )

    loop = AgentLoopResult(
        request=request,
        status=AgentLoopStatus.APPROVAL_REQUIRED,
        messages=request.llm_request.messages,
        model_turns_used=1,
        tool_calls_used=0,
        pending_turn=turn,
    )

    identity = turn.pending_approval_call_ids

    return (
        turn,
        loop,
        identity,
    )


@pytest.mark.anyio
async def test_postgres_continuation_restart_cas_and_tamper() -> None:
    settings = Settings()

    assert settings.postgres_user == "ai_platform_runtime"

    turn, loop, identity = _snapshots()

    async with postgres_pool_runtime(settings) as pool:
        store = PostgreSQLAgentContinuationStore(pool=pool)

        await store.create(
            kind=AgentContinuationKind.TURN,
            identity=identity,
            snapshot=turn,
        )

        await store.create(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
            snapshot=loop,
        )

        # Fresh adapter object represents process-level restart.
        restarted = PostgreSQLAgentContinuationStore(pool=pool)

        loaded_turn = await restarted.load(
            kind=AgentContinuationKind.TURN,
            identity=identity,
        )

        loaded_loop = await restarted.load(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
        )

        assert loaded_turn == turn
        assert loaded_loop == loop

        await restarted.replace(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
            expected=loop,
            snapshot=loop,
        )

        await restarted.consume(
            kind=AgentContinuationKind.TURN,
            identity=identity,
            expected=turn,
        )

        assert (
            await restarted.load(
                kind=AgentContinuationKind.TURN,
                identity=identity,
            )
            is None
        )

        with pytest.raises(
            AgentContinuationConflictError,
        ):
            await restarted.consume(
                kind=AgentContinuationKind.TURN,
                identity=identity,
                expected=turn,
            )

        await restarted.consume(
            kind=AgentContinuationKind.LOOP,
            identity=identity,
            expected=loop,
        )

        assert (
            await restarted.load(
                kind=AgentContinuationKind.LOOP,
                identity=identity,
            )
            is None
        )

        async with (
            pool.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                    SELECT
                        has_table_privilege(
                            current_user,
                            'ai_platform.agent_continuations',
                            'SELECT'
                        ),
                        has_table_privilege(
                            current_user,
                            'ai_platform.agent_continuations',
                            'INSERT'
                        ),
                        has_table_privilege(
                            current_user,
                            'ai_platform.agent_continuations',
                            'UPDATE'
                        ),
                        has_table_privilege(
                            current_user,
                            'ai_platform.agent_continuations',
                            'DELETE'
                        )
                    """
            )

            privileges = await cursor.fetchone()

        assert privileges == (
            True,
            True,
            True,
            False,
        )

        # Create one additional active row then corrupt only the payload.
        tamper_turn, _, tamper_identity = _snapshots()

        tamper_identity = (tamper_identity[0] + ":tamper",)

        tamper_proposal = tamper_turn.planned_steps[0].proposal

        tamper_invocation = ToolInvocation(
            call_id=tamper_identity[0],
            tool_name=tamper_proposal.tool_name,
            arguments=tamper_proposal.arguments,
        )

        tamper_turn = AgentTurnResult(
            run_id=tamper_turn.run_id,
            status=tamper_turn.status,
            response=tamper_turn.response,
            planned_steps=(
                AgentPlannedToolCall(
                    step_number=1,
                    proposal=tamper_proposal,
                    invocation=tamper_invocation,
                ),
            ),
            pending_approval_call_ids=tamper_identity,
        )

        await restarted.create(
            kind=AgentContinuationKind.TURN,
            identity=tamper_identity,
            snapshot=tamper_turn,
        )

        key = agent_continuation_key(
            AgentContinuationKind.TURN,
            tamper_identity,
        )

        codec = AgentContinuationCodec()

        original_payload = codec.dumps(
            kind=AgentContinuationKind.TURN,
            identity=tamper_identity,
            snapshot=tamper_turn,
        )

        corrupted_payload = original_payload.replace(
            "approval required",
            "approval corrupted",
            1,
        )

        assert corrupted_payload != original_payload

        async with (
            pool.connection() as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute(
                """
                    UPDATE ai_platform.agent_continuations
                    SET continuation_payload = %s
                    WHERE continuation_key = %s
                    """,
                (
                    corrupted_payload,
                    key,
                ),
            )

        with pytest.raises(
            AgentContinuationIntegrityError,
            match="checksum mismatch",
        ):
            await restarted.load(
                kind=AgentContinuationKind.TURN,
                identity=tamper_identity,
            )

        assert (
            sha256(original_payload.encode("utf-8")).hexdigest()
            != sha256(corrupted_payload.encode("utf-8")).hexdigest()
        )
