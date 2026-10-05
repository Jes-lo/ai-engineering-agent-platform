"""Live PostgreSQL tests for durable authenticated human approval."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_engineering_agent_platform.adapters.postgres.approval_store import (
    PostgreSQLApprovalStore,
)
from ai_engineering_agent_platform.config import (
    Settings,
)
from ai_engineering_agent_platform.contracts import (
    ToolArgument,
    ToolInvocation,
)
from ai_engineering_agent_platform.runtime.postgres import (
    postgres_pool_runtime,
)
from ai_engineering_agent_platform.services.approval import (
    APPROVAL_DECISION_SCOPE,
    ApprovalDecision,
    ApprovalService,
    ApprovalStateError,
    ApprovalStatus,
    AuthenticatedApprovalActor,
)
from ai_engineering_agent_platform.services.approval_persistence import (
    ApprovalPersistenceIntegrityError,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_PLATFORM_RUN_POSTGRES_INTEGRATION") != "1",
    reason=("live PostgreSQL integration is opt-in"),
)


RUNTIME_KEYS = {
    "AI_PLATFORM_POSTGRES_HOST",
    "AI_PLATFORM_POSTGRES_PORT",
    "AI_PLATFORM_POSTGRES_DATABASE",
    "AI_PLATFORM_POSTGRES_USER",
    "AI_PLATFORM_POSTGRES_PASSWORD",
    "AI_PLATFORM_POSTGRES_SSLMODE",
    "AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS",
    "AI_PLATFORM_POSTGRES_POOL_MIN_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_MAX_SIZE",
    "AI_PLATFORM_POSTGRES_POOL_TIMEOUT_SECONDS",
}


def _load_env(
    path: Path,
) -> dict[str, str]:
    values: dict[
        str,
        str,
    ] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        key, separator, value = line.partition("=")

        if separator != "=" or key not in RUNTIME_KEYS:
            continue

        values[key] = value

    missing = RUNTIME_KEYS - values.keys()

    if missing:
        raise RuntimeError(
            "Integration runtime environment is incomplete: "
            + ", ".join(sorted(missing))
        )

    return values


def _actor() -> AuthenticatedApprovalActor:
    return AuthenticatedApprovalActor(
        subject="live-human",
        client_id="live-console",
        authentication_method="oidc",
        scopes=(APPROVAL_DECISION_SCOPE,),
    )


def _invocation() -> ToolInvocation:
    return ToolInvocation(
        call_id="agent:live-run:tool:1:nonce-1",
        tool_name="dangerous_change",
        arguments=(
            ToolArgument(
                name="value",
                value=9,
            ),
        ),
    )


async def _exercise() -> None:
    runtime_path = Path(os.environ["AI_PLATFORM_POSTGRES_INTEGRATION_RUNTIME_ENV"])

    runtime = _load_env(runtime_path)

    os.environ.update(runtime)

    settings = Settings()

    if settings.postgres_user != "ai_platform_runtime":
        raise RuntimeError("Integration test must use runtime role")

    now = datetime(
        2026,
        10,
        4,
        6,
        30,
        tzinfo=UTC,
    )

    async with postgres_pool_runtime(settings) as pool:
        store = PostgreSQLApprovalStore(pool)

        service = ApprovalService(
            store=store,
            approval_id_factory=(lambda: "approval:live-restart"),
            clock=lambda: now,
        )

        invocation = _invocation()

        pending = await service.request(
            run_id="live-run",
            workflow_id="live-workflow",
            workflow_version="1",
            step_id="agent-step",
            invocation=invocation,
            ttl_seconds=600,
        )

        assert pending.status is ApprovalStatus.PENDING

        approved = await service.decide(
            pending.approval_id,
            actor=_actor(),
            decision=ApprovalDecision.APPROVE,
            reason="live reviewed",
        )

        assert approved.status is ApprovalStatus.APPROVED
        assert approved.decision is ApprovalDecision.APPROVE
        assert approved.version == 2

        persisted = await store.load(pending.approval_id)

        assert persisted == approved

        restarted_store = PostgreSQLApprovalStore(pool)

        restarted_service = ApprovalService(
            store=restarted_store,
            clock=lambda: now + timedelta(seconds=1),
        )

        grant = await restarted_service.consume_grant(
            pending.approval_id,
            invocation=invocation,
        )

        assert grant.call_id == invocation.call_id
        assert grant.tool_name == invocation.tool_name

        consumed = await restarted_store.load(pending.approval_id)

        assert consumed is not None
        assert consumed.status is ApprovalStatus.CONSUMED
        assert consumed.decision is ApprovalDecision.APPROVE
        assert consumed.version == 3
        assert consumed.decided_by == _actor()

        with pytest.raises(
            ApprovalStateError,
            match="already been consumed",
        ):
            await restarted_service.consume_grant(
                pending.approval_id,
                invocation=invocation,
            )

        async with pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT
                    has_table_privilege(
                        current_user,
                        'ai_platform.approval_requests',
                        'SELECT'
                    ),
                    has_table_privilege(
                        current_user,
                        'ai_platform.approval_requests',
                        'INSERT'
                    ),
                    has_table_privilege(
                        current_user,
                        'ai_platform.approval_requests',
                        'UPDATE'
                    ),
                    has_table_privilege(
                        current_user,
                        'ai_platform.approval_requests',
                        'DELETE'
                    )
                """
            )

            privilege_row = await cursor.fetchone()

        assert privilege_row == (
            True,
            True,
            True,
            False,
        )

        async with pool.connection() as connection:
            await connection.execute(
                """
                UPDATE ai_platform.approval_requests
                SET approval_payload = '{}'
                WHERE approval_id = %s
                """,
                (pending.approval_id,),
            )

        with pytest.raises(
            ApprovalPersistenceIntegrityError,
            match="checksum mismatch",
        ):
            await restarted_store.load(pending.approval_id)


def test_live_postgres_durable_authenticated_approval() -> None:
    """Exercise approval persistence across a fresh service/store instance."""
    asyncio.run(_exercise())
