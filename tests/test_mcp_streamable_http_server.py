"""Tests for the authenticated MCP Streamable HTTP adapter."""

from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient
from mcp.server import ServerRequestContext
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolRequestParams,
    TextContent,
)

from ai_engineering_agent_platform.api.mcp_server import (
    AuthenticatedMCPServerAdapter,
    MCPAccessPolicy,
    MCPToolScopeGrant,
    _tool_definition_to_mcp_tool,
)
from ai_engineering_agent_platform.contracts import (
    ProviderDescriptor,
    ProviderKind,
    ToolDefinition,
    ToolExecutionStatus,
    ToolInvocation,
    ToolParameter,
    ToolParameterType,
    ToolResult,
)
from ai_engineering_agent_platform.services import (
    OwnedMCPToolService,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolExecutionPolicy,
    ToolExecutionService,
    ToolRegistry,
)

RESOURCE = "http://testserver/mcp"


def _valid_wire_credential() -> str:
    """Return deterministic synthetic verifier input without secret literal."""
    return "".join(
        (
            "valid",
            "-",
            "credential",
        )
    )


def _synthetic_access_credential() -> str:
    """Return deterministic synthetic AccessToken material."""
    return "".join(
        (
            "synthetic",
            "-",
            "credential",
        )
    )


class FakeVerifier(TokenVerifier):
    """Synthetic bearer verifier for ASGI boundary tests."""

    async def verify_token(
        self,
        token: str,
    ) -> AccessToken | None:
        if token != _valid_wire_credential():
            return None

        return _token(
            scopes=(
                "mcp:tools",
                "tool:read",
            )
        )


class FakeToolProvider:
    """Synthetic local provider behind controlled execution."""

    def __init__(self) -> None:
        self.invocations: list[ToolInvocation] = []

    @property
    def descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            name="feature16-tools",
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return (
            ToolDefinition(
                name="read_status",
                description="Read synthetic status.",
                parameters=(
                    ToolParameter(
                        name="subject",
                        parameter_type=ToolParameterType.STRING,
                        required=True,
                        description="Synthetic subject.",
                    ),
                    ToolParameter(
                        name="verbose",
                        parameter_type=ToolParameterType.BOOLEAN,
                        required=False,
                    ),
                ),
            ),
            ToolDefinition(
                name="dangerous_change",
                description="Synthetic approval-required change.",
                parameters=(),
            ),
        )

    async def execute(
        self,
        invocation: ToolInvocation,
    ) -> ToolResult:
        self.invocations.append(invocation)

        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content="provider-green",
        )


def _token(
    *,
    scopes: tuple[str, ...],
    claims: dict[str, object] | None = None,
) -> AccessToken:
    return AccessToken(
        token=_synthetic_access_credential(),
        client_id="synthetic-client",
        scopes=list(scopes),
        resource=RESOURCE,
        subject="synthetic-user",
        claims=claims,
    )


def _policy() -> MCPAccessPolicy:
    return MCPAccessPolicy(
        grants=(
            MCPToolScopeGrant(
                scope="tool:read",
                tool_names=("read_status",),
            ),
            MCPToolScopeGrant(
                scope="tool:change",
                tool_names=("dangerous_change",),
            ),
        )
    )


def _stack() -> tuple[
    FakeToolProvider,
    OwnedMCPToolService,
]:
    provider = FakeToolProvider()

    registry = ToolRegistry(
        (provider,),
        policies=(
            ToolExecutionPolicy(
                tool_name="read_status",
                enabled=True,
                requires_approval=False,
            ),
            ToolExecutionPolicy(
                tool_name="dangerous_change",
                enabled=True,
                requires_approval=True,
            ),
        ),
    )

    execution = ToolExecutionService(registry)

    service = OwnedMCPToolService(
        registry=registry,
        tool_execution=execution,
        exposed_tool_names=(
            "read_status",
            "dangerous_change",
        ),
        call_id_factory=lambda: "internal-feature16-call",
    )

    return provider, service


def _context(
    request_id: str = "wire-request-1",
) -> ServerRequestContext:
    return cast(
        ServerRequestContext,
        SimpleNamespace(request_id=request_id),
    )


def _adapter(
    *,
    token: AccessToken,
) -> tuple[
    FakeToolProvider,
    AuthenticatedMCPServerAdapter,
]:
    provider, service = _stack()

    adapter = AuthenticatedMCPServerAdapter(
        tool_service=service,
        access_policy=_policy(),
        token_verifier=FakeVerifier(),
        issuer_url="https://auth.example.invalid/",
        resource_server_url=RESOURCE,
        host="testserver",
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[
                "testserver",
            ],
            allowed_origins=[],
        ),
        access_token_getter=lambda: token,
    )

    return provider, adapter


def test_scope_policy_derives_allowlist_only_from_verified_scopes() -> None:
    """Client-controlled claims never become local capability grants."""
    token = _token(
        scopes=(
            "mcp:tools",
            "tool:read",
        ),
        claims={
            "allowed_tool_names": [
                "dangerous_change",
            ],
        },
    )

    authorization = _policy().authorization_for(token)

    assert authorization.allowed_tool_names == ("read_status",)

    assert authorization.approval_grants == ()


def test_tool_schema_preserves_platform_scalar_contract() -> None:
    """Wire schema mirrors required/optional platform scalar semantics."""
    definition = FakeToolProvider().definitions[0]

    tool = _tool_definition_to_mcp_tool(definition)

    assert tool.name == "read_status"
    assert tool.description == "Read synthetic status."

    assert tool.input_schema == {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Synthetic subject.",
            },
            "verbose": {
                "type": [
                    "boolean",
                    "null",
                ],
            },
        },
        "additionalProperties": False,
        "required": [
            "subject",
        ],
    }


@pytest.mark.anyio
async def test_tools_list_uses_verified_scope_and_core_policy() -> None:
    """Scope grants cannot expose approval-required tools."""
    _, adapter = _adapter(
        token=_token(
            scopes=(
                "mcp:tools",
                "tool:read",
                "tool:change",
            )
        )
    )

    result = await adapter.handle_list_tools(
        _context(),
        None,
    )

    assert tuple(tool.name for tool in result.tools) == ("read_status",)


@pytest.mark.anyio
async def test_tools_call_executes_once_through_owned_service() -> None:
    """Authenticated allowed calls reach controlled execution exactly once."""
    provider, adapter = _adapter(
        token=_token(
            scopes=(
                "mcp:tools",
                "tool:read",
            )
        )
    )

    result = await adapter.handle_call_tool(
        _context("wire-request-77"),
        CallToolRequestParams(
            name="read_status",
            arguments={
                "subject": "feature16",
                "verbose": True,
            },
        ),
    )

    assert result.is_error is False

    assert len(result.content) == 1
    assert isinstance(
        result.content[0],
        TextContent,
    )
    assert result.content[0].text == "provider-green"

    assert len(provider.invocations) == 1

    invocation = provider.invocations[0]

    assert invocation.call_id == "internal-feature16-call"
    assert invocation.tool_name == "read_status"

    assert tuple(
        (
            argument.name,
            argument.value,
        )
        for argument in invocation.arguments
    ) == (
        (
            "subject",
            "feature16",
        ),
        (
            "verbose",
            True,
        ),
    )


@pytest.mark.anyio
async def test_tools_call_rejects_tool_not_granted_by_scope() -> None:
    """Transport authentication alone does not authorize every exposed tool."""
    provider, adapter = _adapter(
        token=_token(
            scopes=(
                "mcp:tools",
                "tool:read",
            )
        )
    )

    result = await adapter.handle_call_tool(
        _context(),
        CallToolRequestParams(
            name="dangerous_change",
            arguments={},
        ),
    )

    assert result.is_error is True

    assert len(result.content) == 1
    assert isinstance(
        result.content[0],
        TextContent,
    )
    assert result.content[0].text == "tool request rejected"

    assert provider.invocations == []


@pytest.mark.anyio
async def test_tools_call_rejects_composite_argument_before_execution() -> None:
    """Nested untrusted MCP values cannot cross the scalar tool boundary."""
    provider, adapter = _adapter(
        token=_token(
            scopes=(
                "mcp:tools",
                "tool:read",
            )
        )
    )

    result = await adapter.handle_call_tool(
        _context(),
        CallToolRequestParams(
            name="read_status",
            arguments={
                "subject": {
                    "nested": "not-supported",
                },
            },
        ),
    )

    assert result.is_error is True
    assert provider.invocations == []


def test_non_local_host_requires_explicit_transport_security() -> None:
    """Deployment hostnames cannot silently disable DNS-rebinding controls."""
    _, service = _stack()

    with pytest.raises(
        ValueError,
        match="requires explicit transport_security",
    ):
        AuthenticatedMCPServerAdapter(
            tool_service=service,
            access_policy=_policy(),
            token_verifier=FakeVerifier(),
            issuer_url="https://auth.example.invalid/",
            resource_server_url="https://mcp.example.invalid/mcp",
            host="mcp.example.invalid",
        )


def test_streamable_http_rejects_unauthenticated_request() -> None:
    """Bearer middleware rejects requests before MCP tool handlers run."""
    _, adapter = _adapter(
        token=_token(
            scopes=(
                "mcp:tools",
                "tool:read",
            )
        )
    )

    with TestClient(
        adapter.app,
        base_url="http://testserver",
    ) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": "unauthenticated",
                "method": "tools/list",
                "params": {},
            },
            headers={
                "Accept": ("application/json, text/event-stream"),
            },
        )

    assert response.status_code == 401
    assert "www-authenticate" in response.headers


def test_streamable_http_exposes_protected_resource_metadata() -> None:
    """Resource-server metadata remains available for OAuth discovery."""
    _, adapter = _adapter(
        token=_token(
            scopes=(
                "mcp:tools",
                "tool:read",
            )
        )
    )

    route_paths = tuple(
        getattr(
            route,
            "path",
            None,
        )
        for route in adapter.app.routes
    )

    assert "/mcp" in route_paths

    assert "/.well-known/oauth-protected-resource/mcp" in route_paths
