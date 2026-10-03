"""Tests for provider-neutral tool contracts."""

import asyncio
from dataclasses import FrozenInstanceError

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
    ToolProvider,
    ToolResult,
)


class ExampleToolProvider:
    """Minimal structural implementation of the tool contract."""

    @property
    def descriptor(self) -> ProviderDescriptor:
        """Return deterministic provider metadata."""
        return ProviderDescriptor(
            name="example-tools",
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Expose one deterministic test tool."""
        return (
            ToolDefinition(
                name="echo",
                description="Return supplied text.",
                parameters=(
                    ToolParameter(
                        name="text",
                        parameter_type=ToolParameterType.STRING,
                        required=True,
                    ),
                ),
            ),
        )

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        """Return deterministic execution output."""
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content="executed",
        )


def test_tool_wire_values_are_stable() -> None:
    """Tool enums should expose stable serialized values."""
    assert ToolParameterType.STRING.value == "string"
    assert ToolParameterType.INTEGER.value == "integer"
    assert ToolParameterType.NUMBER.value == "number"
    assert ToolParameterType.BOOLEAN.value == "boolean"

    assert ToolExecutionStatus.SUCCESS.value == "success"
    assert ToolExecutionStatus.ERROR.value == "error"


def test_tool_parameter_is_immutable() -> None:
    """Tool parameter definitions should not mutate."""
    parameter = ToolParameter(
        name="text",
        parameter_type=ToolParameterType.STRING,
    )

    with pytest.raises(FrozenInstanceError):
        parameter.name = "changed"  # type: ignore[misc]


def test_tool_parameter_validates_metadata() -> None:
    """Tool parameters require usable names and descriptions."""
    with pytest.raises(
        ValueError,
        match="parameter name must not be empty",
    ):
        ToolParameter(
            name=" ",
            parameter_type=ToolParameterType.STRING,
        )

    with pytest.raises(
        ValueError,
        match="parameter description must not be empty",
    ):
        ToolParameter(
            name="text",
            parameter_type=ToolParameterType.STRING,
            description=" ",
        )


def test_tool_definition_validates_identity_and_parameters() -> None:
    """Tool definitions require usable and unique metadata."""
    with pytest.raises(
        ValueError,
        match="tool name must not be empty",
    ):
        ToolDefinition(
            name=" ",
            description="description",
        )

    with pytest.raises(
        ValueError,
        match="tool description must not be empty",
    ):
        ToolDefinition(
            name="tool",
            description=" ",
        )

    with pytest.raises(
        ValueError,
        match="tool parameter names must be unique",
    ):
        ToolDefinition(
            name="tool",
            description="description",
            parameters=(
                ToolParameter(
                    name="value",
                    parameter_type=ToolParameterType.STRING,
                ),
                ToolParameter(
                    name="value",
                    parameter_type=ToolParameterType.INTEGER,
                ),
            ),
        )


def test_tool_argument_validates_name_and_float_value() -> None:
    """Tool arguments require usable names and finite numbers."""
    with pytest.raises(
        ValueError,
        match="argument name must not be empty",
    ):
        ToolArgument(
            name=" ",
            value="value",
        )

    with pytest.raises(
        ValueError,
        match="tool argument float values must be finite",
    ):
        ToolArgument(
            name="number",
            value=float("nan"),
        )

    with pytest.raises(
        ValueError,
        match="tool argument float values must be finite",
    ):
        ToolArgument(
            name="number",
            value=float("inf"),
        )


def test_tool_invocation_validates_identity() -> None:
    """Tool invocations require call and tool identifiers."""
    with pytest.raises(
        ValueError,
        match="call_id must not be empty",
    ):
        ToolInvocation(
            call_id=" ",
            tool_name="echo",
        )

    with pytest.raises(
        ValueError,
        match="tool_name must not be empty",
    ):
        ToolInvocation(
            call_id="call-1",
            tool_name=" ",
        )


def test_tool_invocation_rejects_duplicate_arguments() -> None:
    """One invocation cannot provide an argument twice."""
    with pytest.raises(
        ValueError,
        match="tool argument names must be unique",
    ):
        ToolInvocation(
            call_id="call-1",
            tool_name="echo",
            arguments=(
                ToolArgument(
                    name="value",
                    value="first",
                ),
                ToolArgument(
                    name="value",
                    value="second",
                ),
            ),
        )


def test_tool_result_validates_identity() -> None:
    """Normalized tool results retain valid invocation identity."""
    with pytest.raises(
        ValueError,
        match="call_id must not be empty",
    ):
        ToolResult(
            call_id=" ",
            tool_name="echo",
            status=ToolExecutionStatus.ERROR,
            content="error",
        )

    with pytest.raises(
        ValueError,
        match="tool_name must not be empty",
    ):
        ToolResult(
            call_id="call-1",
            tool_name=" ",
            status=ToolExecutionStatus.SUCCESS,
            content="",
        )


def test_tool_provider_supports_structural_async_typing() -> None:
    """Concrete adapters should satisfy ToolProvider structurally."""
    provider: ToolProvider = ExampleToolProvider()

    assert isinstance(provider, ToolProvider)
    assert provider.descriptor.kind is ProviderKind.TOOL
    assert len(provider.definitions) == 1
    assert provider.definitions[0].name == "echo"

    invocation = ToolInvocation(
        call_id="call-1",
        tool_name="echo",
        arguments=(
            ToolArgument(
                name="text",
                value="hello",
            ),
        ),
    )

    result = asyncio.run(provider.execute(invocation))

    assert result.call_id == "call-1"
    assert result.tool_name == "echo"
    assert result.status is ToolExecutionStatus.SUCCESS
    assert result.content == "executed"
