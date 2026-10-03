"""Tests for controlled registry, authorization, and tool execution."""

from collections.abc import Callable

import pytest

from ai_engineering_agent_platform.contracts import (
    ProviderDescriptor,
    ProviderKind,
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)
from ai_engineering_agent_platform.domain import (
    ProviderExecutionError,
)
from ai_engineering_agent_platform.services import (
    ControlledToolExecutionResult,
    ToolApprovalGrant,
    ToolApprovalRequiredError,
    ToolAuthorizationError,
    ToolExecutionAuthorization,
    ToolExecutionPolicy,
    ToolExecutionService,
    ToolInputValidationError,
    ToolRegistry,
    ToolRegistryError,
    ToolResultValidationError,
)


class SyntheticToolProvider:
    """Deterministic provider used to verify platform-owned controls."""

    def __init__(
        self,
        *,
        provider_name: str = "synthetic-tools",
        definitions: tuple[
            ToolDefinition,
            ...,
        ]
        | None = None,
        result_factory: Callable[
            [ToolInvocation],
            object,
        ]
        | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store deterministic definitions and execution behavior."""
        self._provider_name = provider_name
        self._definitions = _definitions() if definitions is None else definitions
        self._result_factory = result_factory
        self._error = error
        self.invocations: list[ToolInvocation] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        """Return provider identity."""
        return ProviderDescriptor(
            name=self._provider_name,
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(
        self,
    ) -> tuple[
        ToolDefinition,
        ...,
    ]:
        """Return deterministic tool definitions."""
        return self._definitions

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        """Record execution and return deterministic provider output."""
        self.invocations.append(invocation)

        if self._error is not None:
            raise self._error

        if self._result_factory is not None:
            result = self._result_factory(invocation)

            return result  # type: ignore[return-value]

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content="synthetic execution result",
        )


def _definitions() -> tuple[
    ToolDefinition,
    ...,
]:
    """Return deterministic safe/read and side-effecting definitions."""
    return (
        ToolDefinition(
            name="lookup",
            description="Read synthetic information.",
            parameters=(
                ToolParameter(
                    name="query",
                    parameter_type=(ToolParameterType.STRING),
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    parameter_type=(ToolParameterType.INTEGER),
                    required=False,
                ),
            ),
        ),
        ToolDefinition(
            name="change",
            description="Perform a synthetic state change.",
            parameters=(
                ToolParameter(
                    name="value",
                    parameter_type=(ToolParameterType.NUMBER),
                    required=True,
                ),
                ToolParameter(
                    name="confirmed",
                    parameter_type=(ToolParameterType.BOOLEAN),
                    required=False,
                ),
            ),
        ),
    )


def _policies(
    *,
    lookup_enabled: bool = True,
    change_enabled: bool = True,
) -> tuple[
    ToolExecutionPolicy,
    ...,
]:
    """Return complete explicit policy coverage."""
    return (
        ToolExecutionPolicy(
            tool_name="lookup",
            enabled=lookup_enabled,
            requires_approval=False,
        ),
        ToolExecutionPolicy(
            tool_name="change",
            enabled=change_enabled,
            requires_approval=True,
        ),
    )


def _registry(
    provider: SyntheticToolProvider | None = None,
    *,
    policies: tuple[
        ToolExecutionPolicy,
        ...,
    ]
    | None = None,
) -> tuple[
    ToolRegistry,
    SyntheticToolProvider,
]:
    """Build one controlled deterministic registry."""
    actual_provider = SyntheticToolProvider() if provider is None else provider

    registry = ToolRegistry(
        (actual_provider,),
        policies=(_policies() if policies is None else policies),
    )

    return (
        registry,
        actual_provider,
    )


def _lookup_invocation(
    *,
    call_id: str = "call-lookup-1",
    query: object = "synthetic",
    extra_arguments: tuple[
        ToolArgument,
        ...,
    ] = (),
) -> ToolInvocation:
    """Return one lookup invocation."""
    return ToolInvocation(
        call_id=call_id,
        tool_name="lookup",
        arguments=(
            ToolArgument(
                name="query",
                value=query,  # type: ignore[arg-type]
            ),
            *extra_arguments,
        ),
    )


def _change_invocation(
    *,
    call_id: str = "call-change-1",
    value: object = 2,
) -> ToolInvocation:
    """Return one approval-required invocation."""
    return ToolInvocation(
        call_id=call_id,
        tool_name="change",
        arguments=(
            ToolArgument(
                name="value",
                value=value,  # type: ignore[arg-type]
            ),
        ),
    )


def test_registry_preserves_provider_definition_policy_binding() -> None:
    """Registration should bind provider identity to exact tool policy."""
    registry, _ = _registry()

    assert tuple(
        registration.definition.name for registration in registry.registrations
    ) == (
        "lookup",
        "change",
    )

    lookup = registry.registration("lookup")

    assert lookup.provider_name == ("synthetic-tools")
    assert lookup.definition.name == "lookup"
    assert lookup.policy.requires_approval is False

    change = registry.registration("change")

    assert change.policy.requires_approval is True


def test_registry_requires_explicit_policy_for_every_tool() -> None:
    """No provider tool should become executable by implicit policy."""
    provider = SyntheticToolProvider()

    with pytest.raises(
        ToolRegistryError,
        match="missing explicit policies: change",
    ):
        ToolRegistry(
            (provider,),
            policies=(
                ToolExecutionPolicy(
                    tool_name="lookup",
                ),
            ),
        )


def test_registry_rejects_policy_for_unknown_tool() -> None:
    """Policy entries cannot authorize nonexistent tool identities."""
    provider = SyntheticToolProvider()

    with pytest.raises(
        ToolRegistryError,
        match="policies reference unknown tools: ghost",
    ):
        ToolRegistry(
            (provider,),
            policies=(
                *_policies(),
                ToolExecutionPolicy(
                    tool_name="ghost",
                ),
            ),
        )


def test_registry_rejects_duplicate_provider_identity() -> None:
    """Two providers cannot silently share one provider identity."""
    first = SyntheticToolProvider(
        provider_name="duplicate-provider",
        definitions=(
            ToolDefinition(
                name="first",
                description="First tool.",
            ),
        ),
    )

    second = SyntheticToolProvider(
        provider_name="duplicate-provider",
        definitions=(
            ToolDefinition(
                name="second",
                description="Second tool.",
            ),
        ),
    )

    with pytest.raises(
        ToolRegistryError,
        match="provider names must be unique",
    ):
        ToolRegistry(
            (
                first,
                second,
            ),
            policies=(
                ToolExecutionPolicy(
                    tool_name="first",
                ),
                ToolExecutionPolicy(
                    tool_name="second",
                ),
            ),
        )


def test_registry_rejects_globally_duplicate_tool_name() -> None:
    """Tool names must resolve unambiguously across providers."""
    definition = ToolDefinition(
        name="duplicate-tool",
        description="Duplicate.",
    )

    first = SyntheticToolProvider(
        provider_name="provider-a",
        definitions=(definition,),
    )

    second = SyntheticToolProvider(
        provider_name="provider-b",
        definitions=(definition,),
    )

    with pytest.raises(
        ToolRegistryError,
        match=("registered tool names must be globally unique"),
    ):
        ToolRegistry(
            (
                first,
                second,
            ),
            policies=(
                ToolExecutionPolicy(
                    tool_name="duplicate-tool",
                ),
            ),
        )


def test_unknown_tool_lookup_fails_closed() -> None:
    """Unknown tool identities should never resolve implicitly."""
    registry, _ = _registry()

    with pytest.raises(
        ToolRegistryError,
        match="tool is not registered: ghost",
    ):
        registry.registration("ghost")


@pytest.mark.anyio
async def test_allowlisted_tool_executes_and_preserves_identity() -> None:
    """Authorized valid input should execute exactly once."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _lookup_invocation(
        extra_arguments=(
            ToolArgument(
                name="limit",
                value=3,
            ),
        )
    )

    result = await service.execute(
        invocation,
        authorization=(
            ToolExecutionAuthorization(
                allowed_tool_names=("lookup",),
            )
        ),
    )

    assert isinstance(
        result,
        ControlledToolExecutionResult,
    )

    assert provider.invocations == [invocation]

    assert result.registration.definition.name == ("lookup")

    assert result.result.call_id == (invocation.call_id)

    assert result.result.tool_name == "lookup"
    assert result.result.status is (ToolExecutionStatus.SUCCESS)

    assert result.approval_required is False
    assert result.approval_used is False


@pytest.mark.anyio
async def test_tool_not_in_allowlist_is_denied_before_provider() -> None:
    """Execution authorization must be explicit per tool."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    with pytest.raises(
        ToolAuthorizationError,
        match="not present in the execution allowlist",
    ):
        await service.execute(
            _lookup_invocation(),
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=(),
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_disabled_tool_is_denied_before_provider() -> None:
    """Disabled platform policy must override an allowlist."""
    registry, provider = _registry(
        policies=_policies(
            lookup_enabled=False,
        ),
    )

    service = ToolExecutionService(registry)

    with pytest.raises(
        ToolAuthorizationError,
        match="disabled by platform policy",
    ):
        await service.execute(
            _lookup_invocation(),
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_approval_required_tool_fails_without_exact_grant() -> None:
    """Sensitive policy must fail closed when approval evidence is absent."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _change_invocation()

    with pytest.raises(
        ToolApprovalRequiredError,
        match="requires exact approval",
    ):
        await service.execute(
            invocation,
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("change",),
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_approval_grant_cannot_be_reused_for_different_call() -> None:
    """Approval evidence is bound to an exact call and tool identity."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _change_invocation(
        call_id="call-change-2",
    )

    authorization = ToolExecutionAuthorization(
        allowed_tool_names=("change",),
        approval_grants=(
            ToolApprovalGrant(
                call_id="call-change-1",
                tool_name="change",
            ),
        ),
    )

    with pytest.raises(
        ToolApprovalRequiredError,
        match="requires exact approval",
    ):
        await service.execute(
            invocation,
            authorization=authorization,
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_exact_approval_grant_allows_required_tool() -> None:
    """Matching structural approval evidence permits policy execution."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _change_invocation()

    result = await service.execute(
        invocation,
        authorization=(
            ToolExecutionAuthorization(
                allowed_tool_names=("change",),
                approval_grants=(
                    ToolApprovalGrant(
                        call_id=(invocation.call_id),
                        tool_name=(invocation.tool_name),
                    ),
                ),
            )
        ),
    )

    assert provider.invocations == [invocation]

    assert result.approval_required is True
    assert result.approval_used is True


@pytest.mark.anyio
async def test_missing_required_argument_fails_before_provider() -> None:
    """Required parameter presence is checked before tool execution."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = ToolInvocation(
        call_id="missing-required",
        tool_name="lookup",
        arguments=(),
    )

    with pytest.raises(
        ToolInputValidationError,
        match="missing required arguments: query",
    ):
        await service.execute(
            invocation,
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_none_does_not_satisfy_required_argument() -> None:
    """A required argument cannot be satisfied by a null value."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _lookup_invocation(
        query=None,
    )

    with pytest.raises(
        ToolInputValidationError,
        match="missing required arguments: query",
    ):
        await service.execute(
            invocation,
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_unexpected_argument_fails_before_provider() -> None:
    """Arguments absent from the registered definition fail closed."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _lookup_invocation(
        extra_arguments=(
            ToolArgument(
                name="unknown",
                value="value",
            ),
        )
    )

    with pytest.raises(
        ToolInputValidationError,
        match="unexpected arguments: unknown",
    ):
        await service.execute(
            invocation,
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    (
        "invocation",
        "expected_argument",
    ),
    (
        (
            _lookup_invocation(
                query=123,
            ),
            "query",
        ),
        (
            _lookup_invocation(
                extra_arguments=(
                    ToolArgument(
                        name="limit",
                        value=True,
                    ),
                )
            ),
            "limit",
        ),
        (
            _change_invocation(
                value=True,
            ),
            "value",
        ),
    ),
)
async def test_argument_types_are_strict_without_coercion(
    invocation: ToolInvocation,
    expected_argument: str,
) -> None:
    """Portable tool parameter types should not silently coerce values."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    grants: tuple[
        ToolApprovalGrant,
        ...,
    ] = ()

    if invocation.tool_name == "change":
        grants = (
            ToolApprovalGrant(
                call_id=invocation.call_id,
                tool_name=(invocation.tool_name),
            ),
        )

    with pytest.raises(
        ToolInputValidationError,
        match=("invalid type: " + expected_argument),
    ):
        await service.execute(
            invocation,
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=(invocation.tool_name,),
                    approval_grants=grants,
                )
            ),
        )

    assert provider.invocations == []


@pytest.mark.anyio
async def test_optional_null_argument_is_allowed() -> None:
    """Optional scalar parameters may explicitly carry null."""
    registry, provider = _registry()

    service = ToolExecutionService(registry)

    invocation = _lookup_invocation(
        extra_arguments=(
            ToolArgument(
                name="limit",
                value=None,
            ),
        )
    )

    result = await service.execute(
        invocation,
        authorization=(
            ToolExecutionAuthorization(
                allowed_tool_names=("lookup",),
            )
        ),
    )

    assert provider.invocations == [invocation]

    assert result.result.status is ToolExecutionStatus.SUCCESS


@pytest.mark.anyio
async def test_provider_result_call_id_must_match_invocation() -> None:
    """A provider cannot return a result for another call identity."""
    provider = SyntheticToolProvider(
        result_factory=lambda invocation: ToolResult(
            call_id="forged-call",
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content="forged",
        )
    )

    registry, provider = _registry(provider)

    service = ToolExecutionService(registry)

    with pytest.raises(
        ToolResultValidationError,
        match="call_id does not match invocation",
    ):
        await service.execute(
            _lookup_invocation(),
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert len(provider.invocations) == 1


@pytest.mark.anyio
async def test_provider_result_tool_name_must_match_invocation() -> None:
    """A provider cannot relabel another tool result as this call."""
    provider = SyntheticToolProvider(
        result_factory=lambda invocation: ToolResult(
            call_id=invocation.call_id,
            tool_name="change",
            status=ToolExecutionStatus.SUCCESS,
            content="forged",
        )
    )

    registry, provider = _registry(provider)

    service = ToolExecutionService(registry)

    with pytest.raises(
        ToolResultValidationError,
        match="tool_name does not match invocation",
    ):
        await service.execute(
            _lookup_invocation(),
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert len(provider.invocations) == 1


@pytest.mark.anyio
async def test_provider_execution_failure_is_not_hidden_or_retried() -> None:
    """Provider failures propagate after one authorized execution attempt."""
    provider = SyntheticToolProvider(
        error=ProviderExecutionError("synthetic tool failure")
    )

    registry, provider = _registry(provider)

    service = ToolExecutionService(registry)

    with pytest.raises(
        ProviderExecutionError,
        match="synthetic tool failure",
    ):
        await service.execute(
            _lookup_invocation(),
            authorization=(
                ToolExecutionAuthorization(
                    allowed_tool_names=("lookup",),
                )
            ),
        )

    assert len(provider.invocations) == 1


def test_authorization_requires_unique_allowlist_and_grants() -> None:
    """Duplicate authorization state should be rejected deterministically."""
    with pytest.raises(
        ValueError,
        match="allowed tool names must be unique",
    ):
        ToolExecutionAuthorization(
            allowed_tool_names=(
                "lookup",
                "lookup",
            )
        )

    grant = ToolApprovalGrant(
        call_id="call-1",
        tool_name="change",
    )

    with pytest.raises(
        ValueError,
        match="approval grants must be unique",
    ):
        ToolExecutionAuthorization(
            allowed_tool_names=("change",),
            approval_grants=(
                grant,
                grant,
            ),
        )


def test_controlled_tool_execution_surface_is_public() -> None:
    """New execution controls should be exported by services."""
    import ai_engineering_agent_platform.services as services

    expected = {
        "ControlledToolExecutionResult",
        "ToolApprovalGrant",
        "ToolApprovalRequiredError",
        "ToolAuthorizationError",
        "ToolExecutionAuthorization",
        "ToolExecutionControlError",
        "ToolExecutionPolicy",
        "ToolExecutionService",
        "ToolInputValidationError",
        "ToolRegistration",
        "ToolRegistry",
        "ToolRegistryError",
        "ToolResultValidationError",
    }

    assert expected <= set(services.__all__)

    for name in expected:
        assert hasattr(
            services,
            name,
        )
