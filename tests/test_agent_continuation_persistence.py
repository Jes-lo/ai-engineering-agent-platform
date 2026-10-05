"""Tests for canonical durable agent-continuation persistence."""

import json

import pytest

from ai_engineering_agent_platform.contracts.llm import (
    FinishReason,
    LLMAssistantToolCallMessage,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolResultMessage,
    MessageRole,
    TokenUsage,
)
from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
)
from ai_engineering_agent_platform.services.agent import (
    AgentPlannedToolCall,
    AgentTurnResult,
    AgentTurnStatus,
)
from ai_engineering_agent_platform.services.agent_continuation import (
    AgentContinuationKind,
)
from ai_engineering_agent_platform.services.agent_continuation_persistence import (
    AgentContinuationCodec,
    AgentContinuationSerializationError,
)
from ai_engineering_agent_platform.services.agent_loop import (
    AgentLoopRequest,
    AgentLoopResult,
    AgentLoopStatus,
    BoundedAgentLoopService,
)


def _proposal(
    *,
    provider_call_id: str = "provider-2",
) -> LLMToolCall:
    return LLMToolCall(
        tool_name="change",
        arguments=(
            ToolArgument(
                name="target",
                value="production",
            ),
            ToolArgument(
                name="force",
                value=False,
            ),
            ToolArgument(
                name="count",
                value=2,
            ),
            ToolArgument(
                name="ratio",
                value=0.5,
            ),
            ToolArgument(
                name="optional",
                value=None,
            ),
        ),
        provider_call_id=provider_call_id,
    )


def _turn() -> AgentTurnResult:
    proposal = _proposal()

    invocation = ToolInvocation(
        call_id="agent:durable:tool:2:nonce",
        tool_name=proposal.tool_name,
        arguments=proposal.arguments,
    )

    return AgentTurnResult(
        run_id="durable",
        status=AgentTurnStatus.APPROVAL_REQUIRED,
        response=LLMResponse(
            model="test-model",
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="I need approval.",
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=TokenUsage(
                input_tokens=7,
                output_tokens=3,
            ),
            tool_calls=(proposal,),
        ),
        planned_steps=(
            AgentPlannedToolCall(
                step_number=1,
                proposal=proposal,
                invocation=invocation,
            ),
        ),
        pending_approval_call_ids=(invocation.call_id,),
    )


def _loop() -> AgentLoopResult:
    previous_call = LLMToolCall(
        tool_name="read",
        arguments=(
            ToolArgument(
                name="name",
                value="status",
            ),
        ),
        provider_call_id="provider-1",
    )

    previous_result = ToolResult(
        call_id="agent:durable:tool:1:nonce",
        tool_name="read",
        status=ToolExecutionStatus.SUCCESS,
        content="healthy",
    )

    messages = (
        LLMMessage(
            role=MessageRole.USER,
            content="Inspect and then change production.",
        ),
        LLMAssistantToolCallMessage(
            content="Checking first.",
            tool_calls=(previous_call,),
        ),
        LLMToolResultMessage(
            result=previous_result,
            provider_call_id="provider-1",
        ),
    )

    request = AgentLoopRequest(
        run_id="durable",
        llm_request=LLMRequest(
            model="test-model",
            messages=(messages[0],),
            temperature=0.2,
            max_output_tokens=128,
        ),
        exposed_tool_names=(
            "read",
            "change",
        ),
        max_model_turns=4,
        max_tool_calls=4,
    )

    return AgentLoopResult(
        request=request,
        status=AgentLoopStatus.APPROVAL_REQUIRED,
        messages=messages,
        model_turns_used=2,
        tool_calls_used=1,
        pending_turn=_turn(),
    )


def test_turn_round_trip_is_canonical_and_type_preserving() -> None:
    codec = AgentContinuationCodec()
    turn = _turn()
    identity = tuple(step.invocation.call_id for step in turn.planned_steps)

    payload = codec.dumps(
        kind=AgentContinuationKind.TURN,
        identity=identity,
        snapshot=turn,
    )

    kind, decoded_identity, decoded = codec.loads(payload)

    assert kind is AgentContinuationKind.TURN
    assert decoded_identity == identity
    assert decoded == turn
    assert (
        codec.dumps(
            kind=kind,
            identity=decoded_identity,
            snapshot=decoded,
        )
        == payload
    )


def test_loop_round_trip_preserves_transcript_variants() -> None:
    codec = AgentContinuationCodec()
    loop = _loop()

    assert loop.pending_turn is not None

    identity = loop.pending_turn.pending_approval_call_ids

    payload = codec.dumps(
        kind=AgentContinuationKind.LOOP,
        identity=identity,
        snapshot=loop,
    )

    kind, decoded_identity, decoded = codec.loads(payload)

    assert kind is AgentContinuationKind.LOOP
    assert decoded_identity == identity
    assert decoded == loop


def test_codec_rejects_non_approval_turn() -> None:
    completed = AgentTurnResult(
        run_id="completed",
        status=AgentTurnStatus.COMPLETED,
        response=LLMResponse(
            model="test-model",
            message=LLMMessage(
                role=MessageRole.ASSISTANT,
                content="done",
            ),
            finish_reason=FinishReason.STOP,
        ),
    )

    with pytest.raises(
        AgentContinuationSerializationError,
        match="only approval-required turn",
    ):
        AgentContinuationCodec().dumps(
            kind=AgentContinuationKind.TURN,
            identity=("non-approval-sentinel",),
            snapshot=completed,
        )


def test_codec_rejects_turn_identity_mismatch() -> None:
    with pytest.raises(
        AgentContinuationSerializationError,
        match="identity does not match planned calls",
    ):
        AgentContinuationCodec().dumps(
            kind=AgentContinuationKind.TURN,
            identity=("forged-call",),
            snapshot=_turn(),
        )


def test_codec_rejects_loop_identity_mismatch() -> None:
    with pytest.raises(
        AgentContinuationSerializationError,
        match="identity does not match planned calls",
    ):
        AgentContinuationCodec().dumps(
            kind=AgentContinuationKind.LOOP,
            identity=("forged-call",),
            snapshot=_loop(),
        )


def test_codec_rejects_noncanonical_json() -> None:
    codec = AgentContinuationCodec()
    turn = _turn()
    identity = tuple(step.invocation.call_id for step in turn.planned_steps)

    canonical = codec.dumps(
        kind=AgentContinuationKind.TURN,
        identity=identity,
        snapshot=turn,
    )

    parsed = json.loads(canonical)

    noncanonical = json.dumps(
        parsed,
        indent=2,
        sort_keys=False,
    )

    with pytest.raises(
        AgentContinuationSerializationError,
        match="not canonical",
    ):
        codec.loads(noncanonical)


def test_codec_rejects_invalid_kind() -> None:
    codec = AgentContinuationCodec()
    turn = _turn()
    identity = tuple(step.invocation.call_id for step in turn.planned_steps)

    raw = json.loads(
        codec.dumps(
            kind=AgentContinuationKind.TURN,
            identity=identity,
            snapshot=turn,
        )
    )

    raw["kind"] = "override"

    tampered = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    with pytest.raises(
        AgentContinuationSerializationError,
        match="kind is invalid",
    ):
        codec.loads(tampered)


def test_codec_rejects_collection_tool_argument_wire_value() -> None:
    codec = AgentContinuationCodec()
    turn = _turn()
    identity = tuple(step.invocation.call_id for step in turn.planned_steps)

    raw = json.loads(
        codec.dumps(
            kind=AgentContinuationKind.TURN,
            identity=identity,
            snapshot=turn,
        )
    )

    raw["snapshot"]["planned_steps"][0]["invocation"]["arguments"][0]["value"] = {
        "type": "list",
        "value": [],
    }

    tampered = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )

    with pytest.raises(
        AgentContinuationSerializationError,
        match="scalar type is unsupported",
    ):
        codec.loads(tampered)


def test_loop_key_remains_stable_when_pending_approval_subset_changes() -> None:
    """Loop continuation identity follows the frozen plan, not pending subset."""
    first = _turn()
    first_step = first.planned_steps[0]

    second_proposal = LLMToolCall(
        tool_name="change",
        arguments=(
            ToolArgument(
                name="target",
                value="staging",
            ),
        ),
        provider_call_id="provider-3",
    )

    second_invocation = ToolInvocation(
        call_id="agent:durable:tool:3:nonce",
        tool_name="change",
        arguments=second_proposal.arguments,
    )

    second_step = AgentPlannedToolCall(
        step_number=2,
        proposal=second_proposal,
        invocation=second_invocation,
    )

    partial = AgentTurnResult(
        run_id=first.run_id,
        status=AgentTurnStatus.APPROVAL_REQUIRED,
        response=LLMResponse(
            model=first.response.model,
            message=first.response.message,
            finish_reason=FinishReason.TOOL_CALLS,
            usage=first.response.usage,
            tool_calls=(
                first_step.proposal,
                second_proposal,
            ),
        ),
        planned_steps=(
            first_step,
            second_step,
        ),
        pending_approval_call_ids=(second_invocation.call_id,),
    )

    expected = (
        first_step.invocation.call_id,
        second_invocation.call_id,
    )

    assert BoundedAgentLoopService._pending_key(partial) == expected

    codec = AgentContinuationCodec()

    loop = AgentLoopResult(
        request=AgentLoopRequest(
            run_id=partial.run_id,
            llm_request=LLMRequest(
                model="test-model",
                messages=(
                    LLMMessage(
                        role=MessageRole.USER,
                        content="perform both changes",
                    ),
                ),
            ),
            exposed_tool_names=("change",),
            max_model_turns=4,
            max_tool_calls=4,
        ),
        status=AgentLoopStatus.APPROVAL_REQUIRED,
        messages=(
            LLMMessage(
                role=MessageRole.USER,
                content="perform both changes",
            ),
        ),
        model_turns_used=1,
        tool_calls_used=0,
        pending_turn=partial,
    )

    payload = codec.dumps(
        kind=AgentContinuationKind.LOOP,
        identity=expected,
        snapshot=loop,
    )

    kind, identity, decoded = codec.loads(payload)

    assert kind is AgentContinuationKind.LOOP
    assert identity == expected
    assert decoded == loop
