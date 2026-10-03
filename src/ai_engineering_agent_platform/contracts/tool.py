"""Provider-neutral contracts for controlled tool execution."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol, runtime_checkable

from ai_engineering_agent_platform.contracts.provider import Provider


class ToolParameterType(StrEnum):
    """Portable scalar parameter types for platform tools."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"


class ToolExecutionStatus(StrEnum):
    """Normalized outcome of a tool invocation."""

    SUCCESS = "success"
    ERROR = "error"


type ToolArgumentValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ToolParameter:
    """Immutable definition of one tool input parameter."""

    name: str
    parameter_type: ToolParameterType
    required: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate portable parameter metadata."""
        if not self.name.strip():
            raise ValueError("parameter name must not be empty")

        if self.description is not None and not self.description.strip():
            raise ValueError("parameter description must not be empty")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Provider-neutral description of one executable tool."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()

    def __post_init__(self) -> None:
        """Validate tool definition invariants."""
        if not self.name.strip():
            raise ValueError("tool name must not be empty")

        if not self.description.strip():
            raise ValueError("tool description must not be empty")

        parameter_names = [parameter.name for parameter in self.parameters]

        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("tool parameter names must be unique")


@dataclass(frozen=True, slots=True)
class ToolArgument:
    """Immutable scalar argument supplied to a tool."""

    name: str
    value: ToolArgumentValue

    def __post_init__(self) -> None:
        """Validate portable tool argument values."""
        if not self.name.strip():
            raise ValueError("argument name must not be empty")

        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("tool argument float values must be finite")


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """Immutable request to execute one named tool."""

    call_id: str
    tool_name: str
    arguments: tuple[ToolArgument, ...] = ()

    def __post_init__(self) -> None:
        """Validate tool invocation invariants."""
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")

        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")

        argument_names = [argument.name for argument in self.arguments]

        if len(argument_names) != len(set(argument_names)):
            raise ValueError("tool argument names must be unique")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized result returned from a tool invocation."""

    call_id: str
    tool_name: str
    status: ToolExecutionStatus
    content: str

    def __post_init__(self) -> None:
        """Validate normalized tool result identity."""
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")

        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")


@runtime_checkable
class ToolProvider(Provider, Protocol):
    """Asynchronous provider contract for controlled tool execution."""

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return tools exposed by this provider."""
        ...

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        """Execute one normalized tool invocation."""
        ...
