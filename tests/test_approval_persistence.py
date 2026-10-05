"""Tests for canonical durable approval serialization."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.services.approval import (
    APPROVAL_DECISION_SCOPE,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    AuthenticatedApprovalActor,
)
from ai_engineering_agent_platform.services.approval_persistence import (
    APPROVAL_RECORD_FORMAT_VERSION,
    MAX_APPROVAL_PAYLOAD_BYTES,
    ApprovalRecordCodec,
    ApprovalSerializationError,
)

NOW = datetime(
    2026,
    10,
    4,
    6,
    0,
    tzinfo=UTC,
)


def _invocation() -> ToolInvocation:
    return ToolInvocation(
        call_id="agent:run-1:tool:1:nonce-1",
        tool_name="dangerous_change",
        arguments=(
            ToolArgument(
                name="value",
                value=7,
            ),
            ToolArgument(
                name="enabled",
                value=True,
            ),
            ToolArgument(
                name="note",
                value="approved",
            ),
        ),
    )


def _actor() -> AuthenticatedApprovalActor:
    return AuthenticatedApprovalActor(
        subject="human-1",
        client_id="portfolio-console",
        authentication_method="oidc",
        scopes=(APPROVAL_DECISION_SCOPE,),
    )


def _request(
    *,
    status: ApprovalStatus = ApprovalStatus.PENDING,
) -> ApprovalRequest:
    kwargs: dict[str, object] = {}

    if status is not ApprovalStatus.PENDING:
        kwargs.update(
            decision=(
                ApprovalDecision.REJECT
                if status is ApprovalStatus.REJECTED
                else ApprovalDecision.APPROVE
            ),
            decided_by=_actor(),
            decided_at=NOW,
            decision_reason="reviewed",
        )

    if status is ApprovalStatus.CONSUMED:
        kwargs["consumed_at"] = NOW + timedelta(seconds=1)

    return ApprovalRequest(
        approval_id="approval:codec",
        run_id="run-1",
        workflow_id="workflow-1",
        workflow_version="1",
        step_id="agent-step",
        invocation=_invocation(),
        status=status,
        requested_at=NOW,
        expires_at=(NOW + timedelta(hours=1)),
        version=(
            1
            if status is ApprovalStatus.PENDING
            else (3 if status is ApprovalStatus.CONSUMED else 2)
        ),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "status",
    tuple(ApprovalStatus),
)
def test_codec_round_trip_preserves_lifecycle(
    status: ApprovalStatus,
) -> None:
    codec = ApprovalRecordCodec()
    request = _request(status=status)

    payload = codec.dumps(request)

    assert codec.loads(payload) == request

    assert codec.dumps(codec.loads(payload)) == payload


def test_codec_is_versioned_and_canonical() -> None:
    codec = ApprovalRecordCodec()

    payload = codec.dumps(_request())

    decoded = json.loads(payload)

    assert decoded["format"] == APPROVAL_RECORD_FORMAT_VERSION

    assert payload == json.dumps(
        decoded,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def test_codec_rejects_unknown_root_fields() -> None:
    codec = ApprovalRecordCodec()

    decoded = json.loads(codec.dumps(_request()))

    decoded["unexpected"] = True

    with pytest.raises(
        ApprovalSerializationError,
        match="unexpected fields",
    ):
        codec.loads(json.dumps(decoded))


def test_codec_rejects_unknown_format() -> None:
    codec = ApprovalRecordCodec()

    decoded = json.loads(codec.dumps(_request()))

    decoded["format"] = 999

    with pytest.raises(
        ApprovalSerializationError,
        match="format differs",
    ):
        codec.loads(json.dumps(decoded))


def test_codec_rejects_oversized_payload() -> None:
    codec = ApprovalRecordCodec()

    with pytest.raises(
        ApprovalSerializationError,
        match="payload size",
    ):
        codec.loads("x" * (MAX_APPROVAL_PAYLOAD_BYTES + 1))


def test_codec_preserves_consumed_approve_provenance() -> None:
    codec = ApprovalRecordCodec()

    restored = codec.loads(codec.dumps(_request(status=ApprovalStatus.CONSUMED)))

    assert restored.status is ApprovalStatus.CONSUMED
    assert restored.decision is ApprovalDecision.APPROVE
    assert restored.decided_by == _actor()


def test_codec_rejects_collection_tool_argument_wire_type() -> None:
    """Persistence must not widen the scalar ToolArgument contract."""
    codec = ApprovalRecordCodec()

    decoded = json.loads(codec.dumps(_request()))

    decoded["request"]["invocation"]["arguments"][0]["value"] = {
        "type": "list",
        "value": [],
    }

    with pytest.raises(
        ApprovalSerializationError,
        match="encoded tool argument type is unsupported",
    ):
        codec.loads(json.dumps(decoded))


def test_codec_rejects_invalid_decision_value() -> None:
    """Unknown persisted human decisions fail inside the codec boundary."""
    codec = ApprovalRecordCodec()

    decoded = json.loads(codec.dumps(_request(status=ApprovalStatus.APPROVED)))

    decoded["request"]["decision"] = "override"

    with pytest.raises(
        ApprovalSerializationError,
        match="approval decision is invalid",
    ):
        codec.loads(json.dumps(decoded))
