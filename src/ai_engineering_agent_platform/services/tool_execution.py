"""Controlled registry, authorization, and execution for platform tools."""

from dataclasses import dataclass

from ai_engineering_agent_platform.contracts import (
    ProviderKind,
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolParameter,
    ToolParameterType,
    ToolProvider,
    ToolResult,
)


class ToolExecutionControlError(Exception):
    """Base error for platform-owned tool execution controls."""


class ToolRegistryError(ToolExecutionControlError):
    """Raised when tool registration or lookup is invalid."""


class ToolAuthorizationError(ToolExecutionControlError):
    """Raised when an invocation is not authorized for execution."""


class ToolApprovalRequiredError(ToolAuthorizationError):
    """Raised when an invocation lacks its required approval grant."""


class ToolInputValidationError(ToolExecutionControlError):
    """Raised when invocation arguments violate the registered definition."""


class ToolResultValidationError(ToolExecutionControlError):
    """Raised when a provider returns a result with invalid identity."""


def _require_non_empty_string(
    field_name: str,
    value: object,
) -> None:
    """Require one meaningful string value."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    """Explicit platform policy attached to one registered tool."""

    tool_name: str
    enabled: bool = True
    requires_approval: bool = False

    def __post_init__(self) -> None:
        """Validate tool-policy identity and booleans."""
        _require_non_empty_string(
            "tool_name",
            self.tool_name,
        )

        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

        if not isinstance(
            self.requires_approval,
            bool,
        ):
            raise ValueError("requires_approval must be a boolean")


@dataclass(frozen=True, slots=True)
class ToolApprovalGrant:
    """Structural approval evidence for one exact tool invocation.

    This is intentionally not an identity-bound, persistent, expiring, signed,
    or single-use human-approval system. Those properties belong to a later
    HITL feature.
    """

    call_id: str
    tool_name: str

    def __post_init__(self) -> None:
        """Validate exact invocation identity."""
        _require_non_empty_string(
            "call_id",
            self.call_id,
        )
        _require_non_empty_string(
            "tool_name",
            self.tool_name,
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionAuthorization:
    """Per-execution allowlist plus optional approval evidence."""

    allowed_tool_names: tuple[str, ...]
    approval_grants: tuple[
        ToolApprovalGrant,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        """Validate authorization collections and uniqueness."""
        if not isinstance(
            self.allowed_tool_names,
            tuple,
        ):
            raise ValueError("allowed_tool_names must be a tuple")

        for tool_name in self.allowed_tool_names:
            _require_non_empty_string(
                "allowed tool name",
                tool_name,
            )

        if len(self.allowed_tool_names) != len(set(self.allowed_tool_names)):
            raise ValueError("allowed tool names must be unique")

        if not isinstance(
            self.approval_grants,
            tuple,
        ):
            raise ValueError("approval_grants must be a tuple")

        if any(
            not isinstance(
                grant,
                ToolApprovalGrant,
            )
            for grant in self.approval_grants
        ):
            raise ValueError("approval_grants must contain ToolApprovalGrant values")

        grant_keys = tuple(
            (
                grant.call_id,
                grant.tool_name,
            )
            for grant in self.approval_grants
        )

        if len(grant_keys) != len(set(grant_keys)):
            raise ValueError("approval grants must be unique")

    def allows(
        self,
        tool_name: str,
    ) -> bool:
        """Return whether a tool name is explicitly allowlisted."""
        return tool_name in self.allowed_tool_names

    def has_approval_for(
        self,
        invocation: ToolInvocation,
    ) -> bool:
        """Return whether exact call/tool approval evidence exists."""
        return any(
            grant.call_id == invocation.call_id
            and grant.tool_name == invocation.tool_name
            for grant in self.approval_grants
        )


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """Public immutable registration metadata for one executable tool."""

    provider_name: str
    definition: ToolDefinition
    policy: ToolExecutionPolicy

    def __post_init__(self) -> None:
        """Validate registration ownership and policy identity."""
        _require_non_empty_string(
            "provider_name",
            self.provider_name,
        )

        if not isinstance(
            self.definition,
            ToolDefinition,
        ):
            raise ValueError("definition must be a ToolDefinition")

        if not isinstance(
            self.policy,
            ToolExecutionPolicy,
        ):
            raise ValueError("policy must be a ToolExecutionPolicy")

        if self.definition.name != self.policy.tool_name:
            raise ValueError("tool policy identity does not match definition")


@dataclass(frozen=True, slots=True)
class ToolExecutionPreflight:
    """Validated execution metadata produced without provider side effects."""

    registration: ToolRegistration
    invocation: ToolInvocation
    approval_required: bool
    approval_used: bool

    def __post_init__(self) -> None:
        """Validate preflight identity and approval relationships."""
        if self.registration.definition.name != self.invocation.tool_name:
            raise ValueError("registration does not match invocation")

        if not isinstance(
            self.approval_required,
            bool,
        ):
            raise ValueError("approval_required must be a boolean")

        if not isinstance(
            self.approval_used,
            bool,
        ):
            raise ValueError("approval_used must be a boolean")

        if self.approval_used and not self.approval_required:
            raise ValueError("approval cannot be used when it was not required")


@dataclass(frozen=True, slots=True)
class ControlledToolExecutionResult:
    """Auditable application-level result of one authorized tool call."""

    registration: ToolRegistration
    invocation: ToolInvocation
    result: ToolResult
    approval_required: bool
    approval_used: bool

    def __post_init__(self) -> None:
        """Validate execution-result identity relationships."""
        if self.registration.definition.name != self.invocation.tool_name:
            raise ValueError("registration does not match invocation")

        if self.result.call_id != self.invocation.call_id:
            raise ValueError("result call_id does not match invocation")

        if self.result.tool_name != self.invocation.tool_name:
            raise ValueError("result tool_name does not match invocation")

        if self.approval_used and not self.approval_required:
            raise ValueError("approval cannot be used when it was not required")


@dataclass(frozen=True, slots=True)
class _ResolvedTool:
    """Internal binding between registration metadata and provider."""

    registration: ToolRegistration
    provider: ToolProvider


def _parameter_accepts_value(
    parameter: ToolParameter,
    argument: ToolArgument,
) -> bool:
    """Return whether one argument has the declared portable scalar type."""
    value = argument.value

    if value is None:
        return not parameter.required

    if parameter.parameter_type is ToolParameterType.STRING:
        return isinstance(
            value,
            str,
        )

    if parameter.parameter_type is ToolParameterType.BOOLEAN:
        return isinstance(
            value,
            bool,
        )

    if parameter.parameter_type is ToolParameterType.INTEGER:
        return isinstance(
            value,
            int,
        ) and not isinstance(
            value,
            bool,
        )

    if parameter.parameter_type is ToolParameterType.NUMBER:
        return isinstance(
            value,
            (int, float),
        ) and not isinstance(
            value,
            bool,
        )

    return False


def _validate_invocation_against_definition(
    definition: ToolDefinition,
    invocation: ToolInvocation,
) -> None:
    """Validate arguments without coercion before provider execution."""
    if invocation.tool_name != definition.name:
        raise ToolInputValidationError("tool invocation name does not match definition")

    parameters = {parameter.name: parameter for parameter in definition.parameters}

    arguments = {argument.name: argument for argument in invocation.arguments}

    unexpected = tuple(sorted(set(arguments) - set(parameters)))

    if unexpected:
        raise ToolInputValidationError(
            "tool invocation contains unexpected arguments: " + ", ".join(unexpected)
        )

    missing = tuple(
        parameter.name
        for parameter in definition.parameters
        if (
            parameter.required
            and (
                parameter.name not in arguments
                or arguments[parameter.name].value is None
            )
        )
    )

    if missing:
        raise ToolInputValidationError(
            "tool invocation is missing required arguments: " + ", ".join(missing)
        )

    for argument_name, argument in arguments.items():
        parameter = parameters[argument_name]

        if not _parameter_accepts_value(
            parameter,
            argument,
        ):
            raise ToolInputValidationError(
                "tool argument has invalid type: " + argument_name
            )


def _validate_provider_result(
    invocation: ToolInvocation,
    result: object,
) -> ToolResult:
    """Require a normalized provider result bound to the exact invocation."""
    if not isinstance(
        result,
        ToolResult,
    ):
        raise ToolResultValidationError("tool provider must return ToolResult")

    if result.call_id != invocation.call_id:
        raise ToolResultValidationError("tool result call_id does not match invocation")

    if result.tool_name != invocation.tool_name:
        raise ToolResultValidationError(
            "tool result tool_name does not match invocation"
        )

    if not isinstance(
        result.status,
        ToolExecutionStatus,
    ):
        raise ToolResultValidationError("tool result status is invalid")

    return result


class ToolRegistry:
    """Explicit provider/tool registry with complete policy coverage."""

    def __init__(
        self,
        providers: tuple[
            ToolProvider,
            ...,
        ],
        *,
        policies: tuple[
            ToolExecutionPolicy,
            ...,
        ],
    ) -> None:
        """Build one fail-closed registry from providers and policies."""
        if not isinstance(
            providers,
            tuple,
        ):
            raise ToolRegistryError("providers must be a tuple")

        if not providers:
            raise ToolRegistryError("at least one tool provider is required")

        if not isinstance(
            policies,
            tuple,
        ):
            raise ToolRegistryError("policies must be a tuple")

        provider_names: list[str] = []
        provider_tools: list[
            tuple[
                ToolProvider,
                ToolDefinition,
            ]
        ] = []
        tool_names: list[str] = []

        for provider in providers:
            if not isinstance(
                provider,
                ToolProvider,
            ):
                raise ToolRegistryError("providers must satisfy ToolProvider")

            if provider.descriptor.kind is not ProviderKind.TOOL:
                raise ToolRegistryError("tool provider descriptor kind must be tool")

            provider_names.append(provider.descriptor.name)

            definitions = provider.definitions

            if not isinstance(
                definitions,
                tuple,
            ):
                raise ToolRegistryError("tool provider definitions must be a tuple")

            for definition in definitions:
                if not isinstance(
                    definition,
                    ToolDefinition,
                ):
                    raise ToolRegistryError(
                        "tool provider definitions must contain ToolDefinition values"
                    )

                provider_tools.append(
                    (
                        provider,
                        definition,
                    )
                )
                tool_names.append(definition.name)

        if len(provider_names) != len(set(provider_names)):
            raise ToolRegistryError("tool provider names must be unique")

        if not provider_tools:
            raise ToolRegistryError("at least one tool definition is required")

        if len(tool_names) != len(set(tool_names)):
            raise ToolRegistryError("registered tool names must be globally unique")

        if any(
            not isinstance(
                policy,
                ToolExecutionPolicy,
            )
            for policy in policies
        ):
            raise ToolRegistryError("policies must contain ToolExecutionPolicy values")

        policy_names = tuple(policy.tool_name for policy in policies)

        if len(policy_names) != len(set(policy_names)):
            raise ToolRegistryError("tool policy names must be unique")

        tool_name_set = set(tool_names)
        policy_name_set = set(policy_names)

        missing_policies = tuple(sorted(tool_name_set - policy_name_set))

        unknown_policies = tuple(sorted(policy_name_set - tool_name_set))

        if missing_policies:
            raise ToolRegistryError(
                "registered tools are missing explicit policies: "
                + ", ".join(missing_policies)
            )

        if unknown_policies:
            raise ToolRegistryError(
                "policies reference unknown tools: " + ", ".join(unknown_policies)
            )

        policy_by_name = {policy.tool_name: policy for policy in policies}

        bindings: dict[
            str,
            _ResolvedTool,
        ] = {}

        registrations: list[ToolRegistration] = []

        for provider, definition in provider_tools:
            registration = ToolRegistration(
                provider_name=(provider.descriptor.name),
                definition=definition,
                policy=policy_by_name[definition.name],
            )

            binding = _ResolvedTool(
                registration=registration,
                provider=provider,
            )

            bindings[definition.name] = binding

            registrations.append(registration)

        self._bindings = bindings
        self._registrations = tuple(registrations)

    @property
    def registrations(
        self,
    ) -> tuple[
        ToolRegistration,
        ...,
    ]:
        """Return immutable registrations in deterministic provider order."""
        return self._registrations

    def registration(
        self,
        tool_name: str,
    ) -> ToolRegistration:
        """Return public registration metadata for one known tool."""
        return self._resolve(tool_name).registration

    def _resolve(
        self,
        tool_name: str,
    ) -> _ResolvedTool:
        """Resolve one registered tool or fail closed."""
        binding = self._bindings.get(tool_name)

        if binding is None:
            raise ToolRegistryError(f"tool is not registered: {tool_name}")

        return binding


class ToolExecutionService:
    """Validate and execute controlled tool invocations.

    Validation is available separately so orchestration can verify every
    planned invocation before the first provider side effect.
    """

    def __init__(
        self,
        registry: ToolRegistry,
    ) -> None:
        """Store the explicit tool registry."""
        self._registry = registry

    def validate(
        self,
        invocation: ToolInvocation,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> ToolExecutionPreflight:
        """Validate one invocation without calling its provider."""
        if not isinstance(
            invocation,
            ToolInvocation,
        ):
            raise ToolInputValidationError("invocation must be a ToolInvocation")

        if not isinstance(
            authorization,
            ToolExecutionAuthorization,
        ):
            raise ToolAuthorizationError(
                "authorization must be a ToolExecutionAuthorization"
            )

        binding = self._registry._resolve(invocation.tool_name)

        registration = binding.registration
        policy = registration.policy

        if not policy.enabled:
            raise ToolAuthorizationError("tool is disabled by platform policy")

        if not authorization.allows(invocation.tool_name):
            raise ToolAuthorizationError(
                "tool is not present in the execution allowlist"
            )

        approval_used = False

        if policy.requires_approval:
            if not authorization.has_approval_for(invocation):
                raise ToolApprovalRequiredError(
                    "tool invocation requires exact approval"
                )

            approval_used = True

        _validate_invocation_against_definition(
            registration.definition,
            invocation,
        )

        return ToolExecutionPreflight(
            registration=registration,
            invocation=invocation,
            approval_required=(policy.requires_approval),
            approval_used=approval_used,
        )

    async def execute(
        self,
        invocation: ToolInvocation,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> ControlledToolExecutionResult:
        """Execute one invocation after the same deterministic preflight."""
        preflight = self.validate(
            invocation,
            authorization=authorization,
        )

        binding = self._registry._resolve(invocation.tool_name)

        provider_result = await binding.provider.execute(invocation)

        result = _validate_provider_result(
            invocation,
            provider_result,
        )

        return ControlledToolExecutionResult(
            registration=preflight.registration,
            invocation=invocation,
            result=result,
            approval_required=(preflight.approval_required),
            approval_used=preflight.approval_used,
        )
