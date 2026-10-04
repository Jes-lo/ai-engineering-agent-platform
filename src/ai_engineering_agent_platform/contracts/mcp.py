"""Transport-neutral contracts for MCP interoperability boundaries."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.tool import (
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
)


def _require_non_empty_string(
    value: object,
    *,
    field_name: str,
) -> str:
    """Return one validated non-empty string."""
    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ValueError(f"{field_name} must not be empty")

    return value


@dataclass(frozen=True, slots=True)
class MCPToolBinding:
    """Bind one remote MCP capability to platform-owned tool metadata.

    The remote server controls the remote identifier and runtime result.
    The platform controls the local ToolDefinition that can enter its registry
    and model-visible surface.
    """

    remote_name: str
    local_definition: ToolDefinition

    def __post_init__(self) -> None:
        """Validate one explicit remote-to-local capability binding."""
        _require_non_empty_string(
            self.remote_name,
            field_name="remote_name",
        )

        if not isinstance(
            self.local_definition,
            ToolDefinition,
        ):
            raise ValueError("local_definition must be a ToolDefinition")


@dataclass(frozen=True, slots=True)
class MCPRemoteToolResult:
    """Transport-neutral untrusted result returned by a remote MCP tool."""

    tool_name: str
    content: str
    is_error: bool = False

    def __post_init__(self) -> None:
        """Validate remote result shape without granting it authority."""
        _require_non_empty_string(
            self.tool_name,
            field_name="tool_name",
        )

        if not isinstance(
            self.content,
            str,
        ):
            raise ValueError("content must be a string")

        if not isinstance(
            self.is_error,
            bool,
        ):
            raise ValueError("is_error must be a bool")


@dataclass(frozen=True, slots=True)
class MCPToolCallRequest:
    """Transport-neutral inbound request to a project-owned MCP tool core."""

    request_id: str
    tool_name: str
    arguments: tuple[ToolArgument, ...] = ()

    def __post_init__(self) -> None:
        """Validate caller correlation and portable tool arguments."""
        _require_non_empty_string(
            self.request_id,
            field_name="request_id",
        )

        _require_non_empty_string(
            self.tool_name,
            field_name="tool_name",
        )

        if not isinstance(
            self.arguments,
            tuple,
        ):
            raise ValueError("arguments must be a tuple")

        if any(
            not isinstance(
                argument,
                ToolArgument,
            )
            for argument in self.arguments
        ):
            raise ValueError("arguments must contain ToolArgument values")

        argument_names = tuple(argument.name for argument in self.arguments)

        if len(argument_names) != len(set(argument_names)):
            raise ValueError("argument names must be unique")


@dataclass(frozen=True, slots=True)
class MCPToolCallResponse:
    """Transport-neutral response without internal execution identity."""

    request_id: str
    tool_name: str
    status: ToolExecutionStatus
    content: str

    def __post_init__(self) -> None:
        """Validate response correlation and normalized result shape."""
        _require_non_empty_string(
            self.request_id,
            field_name="request_id",
        )

        _require_non_empty_string(
            self.tool_name,
            field_name="tool_name",
        )

        if not isinstance(
            self.status,
            ToolExecutionStatus,
        ):
            raise ValueError("status must be a ToolExecutionStatus")

        if not isinstance(
            self.content,
            str,
        ):
            raise ValueError("content must be a string")


@runtime_checkable
class MCPClient(Protocol):
    """Transport adapter contract for one explicitly configured MCP server.

    Concrete JSON-RPC, stdio, Streamable HTTP, authentication, and connection
    lifecycle behavior remain outside this transport-neutral foundation.
    """

    @property
    def server_name(
        self,
    ) -> str:
        """Return the configured server identity."""
        ...

    async def list_tools(
        self,
    ) -> tuple[ToolDefinition, ...]:
        """Discover remote tool metadata as untrusted input."""
        ...

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: tuple[ToolArgument, ...],
    ) -> MCPRemoteToolResult:
        """Invoke one already-authorized remote capability through transport."""
        ...
