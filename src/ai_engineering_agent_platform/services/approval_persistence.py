"""Canonical persistence codec for durable authenticated approvals."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import cast

from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.services.approval import (
    ApprovalDecision,
    ApprovalError,
    ApprovalRequest,
    ApprovalStatus,
    AuthenticatedApprovalActor,
)

APPROVAL_RECORD_FORMAT_VERSION = 1

MAX_APPROVAL_PAYLOAD_BYTES = 262144
MAX_APPROVAL_COLLECTION_ITEMS = 128
MAX_APPROVAL_STRING_LENGTH = 65536
MAX_APPROVAL_INTEGER_DIGITS = 128


class ApprovalPersistenceError(ApprovalError):
    """Base error for durable approval persistence."""


class ApprovalSerializationError(ApprovalPersistenceError):
    """Raised when durable approval payloads cannot be encoded safely."""


class ApprovalPersistenceIntegrityError(ApprovalPersistenceError):
    """Raised when stored durable approval state fails integrity validation."""


def _require_exact_keys(
    value: object,
    *,
    keys: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        dict,
    ):
        raise ApprovalSerializationError(f"{label} must be an object")

    if any(not isinstance(key, str) for key in value):
        raise ApprovalSerializationError(f"{label} keys must be strings")

    typed = cast(
        dict[str, object],
        value,
    )

    if set(typed) != keys:
        raise ApprovalSerializationError(f"{label} has unexpected fields")

    return typed


def _require_string(
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise ApprovalSerializationError(f"{label} must be a string")

    if len(value) > MAX_APPROVAL_STRING_LENGTH:
        raise ApprovalSerializationError(f"{label} exceeds maximum string length")

    return value


def _datetime_to_wire(
    value: datetime,
) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ApprovalSerializationError("approval datetime must be timezone-aware")

    normalized = value.astimezone(UTC)

    return normalized.isoformat(timespec="microseconds").replace(
        "+00:00",
        "Z",
    )


def _datetime_from_wire(
    value: object,
    *,
    label: str,
) -> datetime:
    raw = _require_string(
        value,
        label=label,
    )

    if not raw.endswith("Z"):
        raise ApprovalSerializationError(f"{label} must use canonical UTC form")

    try:
        parsed = datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise ApprovalSerializationError(f"{label} is invalid") from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ApprovalSerializationError(f"{label} must be timezone-aware")

    return parsed


def _encode_value(
    value: object,
) -> object:
    """Encode exactly the scalar value domain accepted by ToolArgument."""
    if value is None:
        return {
            "type": "null",
            "value": None,
        }

    if type(value) is bool:
        return {
            "type": "bool",
            "value": value,
        }

    if type(value) is int:
        if len(str(abs(value))) > MAX_APPROVAL_INTEGER_DIGITS:
            raise ApprovalSerializationError(
                "tool argument integer exceeds maximum digits"
            )

        return {
            "type": "int",
            "value": str(value),
        }

    if type(value) is float:
        if not math.isfinite(value):
            raise ApprovalSerializationError("tool argument float must be finite")

        return {
            "type": "float",
            "value": repr(value),
        }

    if isinstance(
        value,
        str,
    ):
        if len(value) > MAX_APPROVAL_STRING_LENGTH:
            raise ApprovalSerializationError(
                "tool argument string exceeds maximum length"
            )

        return {
            "type": "str",
            "value": value,
        }

    raise ApprovalSerializationError("tool argument value type is not persistable")


def _decode_value(
    raw: object,
) -> str | int | float | bool | None:
    """Decode only the scalar value domain accepted by ToolArgument."""
    encoded = _require_exact_keys(
        raw,
        keys={
            "type",
            "value",
        },
        label="encoded tool argument value",
    )

    kind = _require_string(
        encoded["type"],
        label="encoded tool argument type",
    )

    value = encoded["value"]

    if kind == "null":
        if value is not None:
            raise ApprovalSerializationError("null value payload is invalid")

        return None

    if kind == "bool":
        if type(value) is not bool:
            raise ApprovalSerializationError("bool value payload is invalid")

        return value

    if kind == "int":
        raw_int = _require_string(
            value,
            label="integer value",
        )

        if not raw_int or raw_int in {
            "+",
            "-",
        }:
            raise ApprovalSerializationError("integer value is invalid")

        digits = raw_int[1:] if raw_int.startswith("-") else raw_int

        if not digits.isdigit() or len(digits) > MAX_APPROVAL_INTEGER_DIGITS:
            raise ApprovalSerializationError("integer value is invalid")

        if len(digits) > 1 and digits.startswith("0"):
            raise ApprovalSerializationError("integer value is not canonical")

        if raw_int == "-0":
            raise ApprovalSerializationError("integer value is not canonical")

        return int(raw_int)

    if kind == "float":
        raw_float = _require_string(
            value,
            label="float value",
        )

        try:
            parsed_float = float(raw_float)
        except ValueError as exc:
            raise ApprovalSerializationError("float value is invalid") from exc

        if not math.isfinite(parsed_float):
            raise ApprovalSerializationError("float value must be finite")

        if repr(parsed_float) != raw_float:
            raise ApprovalSerializationError("float value is not canonical")

        return parsed_float

    if kind == "str":
        return _require_string(
            value,
            label="string value",
        )

    raise ApprovalSerializationError("encoded tool argument type is unsupported")


class ApprovalRecordCodec:
    """Encode and decode bounded canonical approval records."""

    def dumps(
        self,
        request: ApprovalRequest,
    ) -> str:
        """Serialize one validated approval snapshot canonically."""
        if not isinstance(
            request,
            ApprovalRequest,
        ):
            raise ApprovalSerializationError("request must be ApprovalRequest")

        actor: object = None

        if request.decided_by is not None:
            actor = {
                "authentication_method": (request.decided_by.authentication_method),
                "client_id": request.decided_by.client_id,
                "scopes": list(request.decided_by.scopes),
                "subject": request.decided_by.subject,
            }

        invocation = {
            "arguments": [
                {
                    "name": argument.name,
                    "value": _encode_value(argument.value),
                }
                for argument in request.invocation.arguments
            ],
            "call_id": request.invocation.call_id,
            "tool_name": request.invocation.tool_name,
        }

        payload = {
            "format": APPROVAL_RECORD_FORMAT_VERSION,
            "request": {
                "approval_id": request.approval_id,
                "consumed_at": (
                    None
                    if request.consumed_at is None
                    else _datetime_to_wire(request.consumed_at)
                ),
                "decided_at": (
                    None
                    if request.decided_at is None
                    else _datetime_to_wire(request.decided_at)
                ),
                "decided_by": actor,
                "decision": (
                    None if request.decision is None else request.decision.value
                ),
                "decision_reason": request.decision_reason,
                "expires_at": _datetime_to_wire(request.expires_at),
                "invocation": invocation,
                "requested_at": _datetime_to_wire(request.requested_at),
                "run_id": request.run_id,
                "status": request.status.value,
                "step_id": request.step_id,
                "version": request.version,
                "workflow_id": request.workflow_id,
                "workflow_version": (request.workflow_version),
            },
        }

        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ApprovalSerializationError(
                "approval payload is not JSON serializable"
            ) from exc

        if len(encoded.encode("utf-8")) > MAX_APPROVAL_PAYLOAD_BYTES:
            raise ApprovalSerializationError("approval payload exceeds maximum size")

        return encoded

    def loads(
        self,
        payload: str,
    ) -> ApprovalRequest:
        """Deserialize one exact versioned canonical approval payload."""
        if not isinstance(
            payload,
            str,
        ):
            raise ApprovalSerializationError("approval payload must be a string")

        if not payload or len(payload.encode("utf-8")) > MAX_APPROVAL_PAYLOAD_BYTES:
            raise ApprovalSerializationError("approval payload size is invalid")

        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ApprovalSerializationError(
                "approval payload is invalid JSON"
            ) from exc

        root = _require_exact_keys(
            decoded,
            keys={
                "format",
                "request",
            },
            label="approval payload",
        )

        if (
            type(root["format"]) is not int
            or root["format"] != APPROVAL_RECORD_FORMAT_VERSION
        ):
            raise ApprovalSerializationError("approval payload format differs")

        data = _require_exact_keys(
            root["request"],
            keys={
                "approval_id",
                "consumed_at",
                "decided_at",
                "decided_by",
                "decision",
                "decision_reason",
                "expires_at",
                "invocation",
                "requested_at",
                "run_id",
                "status",
                "step_id",
                "version",
                "workflow_id",
                "workflow_version",
            },
            label="approval request",
        )

        invocation_data = _require_exact_keys(
            data["invocation"],
            keys={
                "arguments",
                "call_id",
                "tool_name",
            },
            label="approval invocation",
        )

        raw_arguments = invocation_data["arguments"]

        if not isinstance(
            raw_arguments,
            list,
        ):
            raise ApprovalSerializationError(
                "approval invocation arguments must be a list"
            )

        if len(raw_arguments) > MAX_APPROVAL_COLLECTION_ITEMS:
            raise ApprovalSerializationError(
                "approval invocation has too many arguments"
            )

        arguments = []

        for raw_argument in raw_arguments:
            argument_data = _require_exact_keys(
                raw_argument,
                keys={
                    "name",
                    "value",
                },
                label="approval invocation argument",
            )

            arguments.append(
                ToolArgument(
                    name=_require_string(
                        argument_data["name"],
                        label="tool argument name",
                    ),
                    value=_decode_value(argument_data["value"]),
                )
            )

        actor_data = data["decided_by"]

        actor = None

        if actor_data is not None:
            actor_fields = _require_exact_keys(
                actor_data,
                keys={
                    "authentication_method",
                    "client_id",
                    "scopes",
                    "subject",
                },
                label="authenticated approval actor",
            )

            raw_scopes = actor_fields["scopes"]

            if not isinstance(
                raw_scopes,
                list,
            ):
                raise ApprovalSerializationError("approval actor scopes must be a list")

            if len(raw_scopes) > MAX_APPROVAL_COLLECTION_ITEMS:
                raise ApprovalSerializationError("approval actor has too many scopes")

            actor = AuthenticatedApprovalActor(
                subject=_require_string(
                    actor_fields["subject"],
                    label="approval actor subject",
                ),
                client_id=_require_string(
                    actor_fields["client_id"],
                    label="approval actor client_id",
                ),
                authentication_method=_require_string(
                    actor_fields["authentication_method"],
                    label=("approval actor authentication_method"),
                ),
                scopes=tuple(
                    _require_string(
                        raw_scope,
                        label="approval actor scope",
                    )
                    for raw_scope in raw_scopes
                ),
            )

        decision_raw = data["decision"]

        try:
            decision = (
                None
                if decision_raw is None
                else ApprovalDecision(
                    _require_string(
                        decision_raw,
                        label="approval decision",
                    )
                )
            )
        except ValueError as exc:
            raise ApprovalSerializationError("approval decision is invalid") from exc
        decided_at_raw = data["decided_at"]

        consumed_at_raw = data["consumed_at"]

        reason_raw = data["decision_reason"]

        if reason_raw is not None and not isinstance(
            reason_raw,
            str,
        ):
            raise ApprovalSerializationError(
                "approval decision reason must be a string or null"
            )

        if type(data["version"]) is not int:
            raise ApprovalSerializationError("approval version must be an integer")

        try:
            return ApprovalRequest(
                approval_id=_require_string(
                    data["approval_id"],
                    label="approval_id",
                ),
                run_id=_require_string(
                    data["run_id"],
                    label="run_id",
                ),
                workflow_id=_require_string(
                    data["workflow_id"],
                    label="workflow_id",
                ),
                workflow_version=_require_string(
                    data["workflow_version"],
                    label="workflow_version",
                ),
                step_id=_require_string(
                    data["step_id"],
                    label="step_id",
                ),
                invocation=ToolInvocation(
                    call_id=_require_string(
                        invocation_data["call_id"],
                        label="tool call_id",
                    ),
                    tool_name=_require_string(
                        invocation_data["tool_name"],
                        label="tool_name",
                    ),
                    arguments=tuple(arguments),
                ),
                status=ApprovalStatus(
                    _require_string(
                        data["status"],
                        label="approval status",
                    )
                ),
                requested_at=_datetime_from_wire(
                    data["requested_at"],
                    label="requested_at",
                ),
                expires_at=_datetime_from_wire(
                    data["expires_at"],
                    label="expires_at",
                ),
                version=data["version"],
                decision=decision,
                decided_by=actor,
                decided_at=(
                    None
                    if decided_at_raw is None
                    else _datetime_from_wire(
                        decided_at_raw,
                        label="decided_at",
                    )
                ),
                decision_reason=reason_raw,
                consumed_at=(
                    None
                    if consumed_at_raw is None
                    else _datetime_from_wire(
                        consumed_at_raw,
                        label="consumed_at",
                    )
                ),
            )
        except ApprovalSerializationError:
            raise
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ApprovalSerializationError(
                "approval payload violates domain invariants"
            ) from exc
