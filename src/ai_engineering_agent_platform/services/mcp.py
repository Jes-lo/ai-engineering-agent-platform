"""MCP trust boundaries built on the existing controlled tool architecture."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from ai_engineering_agent_platform.contracts import (
    MCPClient,
    MCPRemoteToolResult,
    MCPToolBinding,
    MCPToolCallRequest,
    MCPToolCallResponse,
    ProviderDescriptor,
    ProviderKind,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolResult,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolExecutionAuthorization,
    ToolExecutionService,
    ToolRegistry,
    ToolRegistryError,
)

_SERVER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


class MCPControlError(Exception):
    """Base error for MCP interoperability policy and trust boundaries."""


class MCPTrustPolicyError(MCPControlError):
    """Raised when explicit MCP trust configuration is invalid."""


class MCPDiscoveryError(MCPControlError):
    """Raised when remote MCP discovery violates pinned expectations."""


class MCPRemoteExecutionError(MCPControlError):
    """Raised when a remote MCP tool fails or returns mismatched identity."""


class MCPServerPolicyError(MCPControlError):
    """Raised when the project-owned MCP service refuses an inbound call."""


@dataclass(frozen=True, slots=True)
class MCPTrustPolicy:
    """Explicit remote MCP capability bindings controlled by the platform."""

    server_name: str
    bindings: tuple[MCPToolBinding, ...]

    def __post_init__(self) -> None:
        """Validate server identity and exact capability bindings."""
        if (
            not isinstance(
                self.server_name,
                str,
            )
            or _SERVER_NAME_PATTERN.fullmatch(self.server_name) is None
        ):
            raise ValueError("server_name must use 1-40 portable identifier characters")

        if not isinstance(
            self.bindings,
            tuple,
        ):
            raise ValueError("bindings must be a tuple")

        if not self.bindings:
            raise ValueError("bindings must not be empty")

        if any(
            not isinstance(
                binding,
                MCPToolBinding,
            )
            for binding in self.bindings
        ):
            raise ValueError("bindings must contain MCPToolBinding values")

        remote_names = tuple(binding.remote_name for binding in self.bindings)

        if len(remote_names) != len(set(remote_names)):
            raise ValueError("remote MCP tool names must be unique in policy")

        local_names = tuple(binding.local_definition.name for binding in self.bindings)

        if len(local_names) != len(set(local_names)):
            raise ValueError("local MCP tool names must be unique in policy")


@dataclass(frozen=True, slots=True)
class _ResolvedRemoteTool:
    """Internal immutable remote/local identity mapping."""

    remote_name: str
    local_definition: ToolDefinition


class MCPToolProviderAdapter:
    """Expose explicitly pinned remote MCP tools as a normal ToolProvider.

    Discovery never grants execution authority. Remote capabilities are
    surfaced only when they have an explicit platform binding whose parameter
    schema exactly matches the discovered remote schema.

    The adapter performs no authorization or approval decisions itself.
    Once registered, ToolRegistry and ToolExecutionService retain those
    responsibilities exactly as for every other ToolProvider.
    """

    def __init__(
        self,
        *,
        client: MCPClient,
        policy: MCPTrustPolicy,
        resolved_tools: tuple[_ResolvedRemoteTool, ...],
    ) -> None:
        """Store one validated immutable remote capability snapshot."""
        self._client = client
        self._policy = policy
        self._resolved_tools = resolved_tools

        self._remote_by_local = {
            resolved.local_definition.name: (resolved.remote_name)
            for resolved in resolved_tools
        }

    @classmethod
    async def discover(
        cls,
        *,
        client: MCPClient,
        policy: MCPTrustPolicy,
    ) -> "MCPToolProviderAdapter":
        """Discover and pin only explicitly trusted remote capabilities."""
        if not isinstance(
            policy,
            MCPTrustPolicy,
        ):
            raise MCPTrustPolicyError("policy must be an MCPTrustPolicy")

        if client.server_name != policy.server_name:
            raise MCPTrustPolicyError(
                "MCP client server identity does not match trust policy"
            )

        discovered = await client.list_tools()

        if not isinstance(
            discovered,
            tuple,
        ):
            raise MCPDiscoveryError("MCP discovery must return a tuple")

        if any(
            not isinstance(
                definition,
                ToolDefinition,
            )
            for definition in discovered
        ):
            raise MCPDiscoveryError("MCP discovery returned a non-ToolDefinition value")

        discovered_names = tuple(definition.name for definition in discovered)

        if len(discovered_names) != len(set(discovered_names)):
            raise MCPDiscoveryError(
                "remote MCP discovery contains duplicate tool names"
            )

        by_remote_name = {definition.name: definition for definition in discovered}

        resolved: list[_ResolvedRemoteTool] = []

        for binding in policy.bindings:
            remote_definition = by_remote_name.get(binding.remote_name)

            if remote_definition is None:
                raise MCPDiscoveryError(
                    "explicitly bound remote MCP tool is missing: "
                    f"{binding.remote_name}"
                )

            if remote_definition.parameters != binding.local_definition.parameters:
                raise MCPDiscoveryError(
                    "remote MCP parameter schema differs from pinned "
                    f"platform definition: {binding.remote_name}"
                )

            resolved.append(
                _ResolvedRemoteTool(
                    remote_name=binding.remote_name,
                    local_definition=(binding.local_definition),
                )
            )

        return cls(
            client=client,
            policy=policy,
            resolved_tools=tuple(resolved),
        )

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        """Return provider identity suitable for ToolRegistry."""
        return ProviderDescriptor(
            name=("mcp-" + self._policy.server_name),
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(
        self,
    ) -> tuple[ToolDefinition, ...]:
        """Return platform-owned definitions only."""
        return tuple(resolved.local_definition for resolved in self._resolved_tools)

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        """Execute one already-controlled platform invocation remotely."""
        if not isinstance(
            invocation,
            ToolInvocation,
        ):
            raise MCPRemoteExecutionError("invocation must be a ToolInvocation")

        remote_name = self._remote_by_local.get(invocation.tool_name)

        if remote_name is None:
            raise MCPRemoteExecutionError("invocation refers to an unbound MCP tool")

        remote_result = await self._client.call_tool(
            tool_name=remote_name,
            arguments=invocation.arguments,
        )

        if not isinstance(
            remote_result,
            MCPRemoteToolResult,
        ):
            raise MCPRemoteExecutionError("remote MCP result has invalid type")

        if remote_result.tool_name != remote_name:
            raise MCPRemoteExecutionError("remote MCP result tool identity mismatch")

        if remote_result.is_error:
            raise MCPRemoteExecutionError("remote MCP tool reported an execution error")

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content=remote_result.content,
        )


def _default_server_call_id() -> str:
    """Create an internal execution identity unrelated to MCP request IDs."""
    return "mcp-server:" + uuid4().hex


class OwnedMCPToolService:
    """Project-owned transport-neutral MCP tool service core.

    This service is intended to sit behind a future authenticated MCP wire
    transport. It never accepts a platform execution call_id from the remote
    caller and never returns that internal call_id in its response.

    Approval-required tools are deliberately not exposed by this foundation.
    A future authenticated and durable HITL lifecycle must exist before those
    tools can be safely exported through MCP.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        tool_execution: ToolExecutionService,
        exposed_tool_names: tuple[str, ...],
        call_id_factory: Callable[[], str] | None = None,
    ) -> None:
        """Store explicit server exposure and controlled execution boundary."""
        if not isinstance(
            registry,
            ToolRegistry,
        ):
            raise ValueError("registry must be a ToolRegistry")

        if not isinstance(
            tool_execution,
            ToolExecutionService,
        ):
            raise ValueError("tool_execution must be a ToolExecutionService")

        if not isinstance(
            exposed_tool_names,
            tuple,
        ):
            raise ValueError("exposed_tool_names must be a tuple")

        if not exposed_tool_names:
            raise ValueError("exposed_tool_names must not be empty")

        if len(exposed_tool_names) != len(set(exposed_tool_names)):
            raise ValueError("exposed MCP tool names must be unique")

        for tool_name in exposed_tool_names:
            if (
                not isinstance(
                    tool_name,
                    str,
                )
                or not tool_name.strip()
            ):
                raise ValueError("exposed tool names must be non-empty strings")

            try:
                registry.registration(tool_name)
            except ToolRegistryError as exc:
                raise ValueError(
                    f"exposed MCP tool must already exist in ToolRegistry: {tool_name}"
                ) from exc

        self._registry = registry
        self._tool_execution = tool_execution
        self._exposed_tool_names = exposed_tool_names
        self._call_id_factory = call_id_factory or _default_server_call_id

    def list_tools(
        self,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> tuple[ToolDefinition, ...]:
        """List only enabled, authorized, non-approval tools."""
        if not isinstance(
            authorization,
            ToolExecutionAuthorization,
        ):
            raise MCPServerPolicyError(
                "authorization must be ToolExecutionAuthorization"
            )

        definitions: list[ToolDefinition] = []

        for tool_name in self._exposed_tool_names:
            registration = self._registry.registration(tool_name)

            if not registration.policy.enabled:
                continue

            if registration.policy.requires_approval:
                continue

            if not authorization.allows(tool_name):
                continue

            definitions.append(registration.definition)

        return tuple(definitions)

    async def call_tool(
        self,
        request: MCPToolCallRequest,
        *,
        authorization: ToolExecutionAuthorization,
    ) -> MCPToolCallResponse:
        """Execute one explicitly exposed MCP tool through ToolExecutionService."""
        if not isinstance(
            request,
            MCPToolCallRequest,
        ):
            raise MCPServerPolicyError("request must be an MCPToolCallRequest")

        if not isinstance(
            authorization,
            ToolExecutionAuthorization,
        ):
            raise MCPServerPolicyError(
                "authorization must be ToolExecutionAuthorization"
            )

        if request.tool_name not in self._exposed_tool_names:
            raise MCPServerPolicyError("MCP tool is not explicitly exposed")

        registration = self._registry.registration(request.tool_name)

        if not registration.policy.enabled:
            raise MCPServerPolicyError("MCP tool is disabled by platform policy")

        if registration.policy.requires_approval:
            raise MCPServerPolicyError(
                "approval-required tools are not exportable by "
                "the current MCP server foundation"
            )

        internal_call_id = self._call_id_factory()

        if (
            not isinstance(
                internal_call_id,
                str,
            )
            or not internal_call_id.strip()
        ):
            raise MCPServerPolicyError(
                "internal call_id factory returned an invalid value"
            )

        if internal_call_id == request.request_id:
            raise MCPServerPolicyError(
                "internal call_id must differ from external MCP request_id"
            )

        invocation = ToolInvocation(
            call_id=internal_call_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
        )

        execution = await self._tool_execution.execute(
            invocation,
            authorization=authorization,
        )

        result = execution.result

        return MCPToolCallResponse(
            request_id=request.request_id,
            tool_name=result.tool_name,
            status=result.status,
            content=result.content,
        )
