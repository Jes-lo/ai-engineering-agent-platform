"""End-to-end HTTP tests for MCP mounted in the FastAPI host."""

from fastapi.testclient import TestClient
from mcp.server.auth.provider import (
    AccessToken,
    TokenVerifier,
)
from mcp.server.transport_security import (
    TransportSecuritySettings,
)

from ai_engineering_agent_platform.api.mcp_server import (
    AuthenticatedMCPServerAdapter,
    MCPAccessPolicy,
    MCPToolScopeGrant,
)
from ai_engineering_agent_platform.app import create_app
from ai_engineering_agent_platform.config import Settings
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
from ai_engineering_agent_platform.runtime.mcp import (
    create_mcp_server_adapter,
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
PROTOCOL_VERSION = "2026-07-28"
ORIGIN = "http://testserver"


def _wire_credential() -> str:
    """Return deterministic synthetic bearer material."""
    return "-".join(
        (
            "feature16",
            "credential",
        )
    )


class IntegrationVerifier(TokenVerifier):
    """Synthetic verifier for real ASGI authentication tests."""

    async def verify_token(
        self,
        token: str,
    ) -> AccessToken | None:
        if token != _wire_credential():
            return None

        return AccessToken(
            token=_wire_credential(),
            client_id="feature16-client",
            scopes=[
                "mcp:tools",
                "tool:read",
            ],
            resource=RESOURCE,
            subject="feature16-user",
            claims={
                "allowed_tool_names": [
                    "dangerous_change",
                ],
            },
        )


class IntegrationToolProvider:
    """Synthetic provider reached only through controlled execution."""

    def __init__(self) -> None:
        self.invocations: list[ToolInvocation] = []

    @property
    def descriptor(
        self,
    ) -> ProviderDescriptor:
        return ProviderDescriptor(
            name="feature16-http-tools",
            kind=ProviderKind.TOOL,
        )

    @property
    def definitions(
        self,
    ) -> tuple[
        ToolDefinition,
        ...,
    ]:
        return (
            ToolDefinition(
                name="read_status",
                description=("Read synthetic integration status."),
                parameters=(
                    ToolParameter(
                        name="subject",
                        parameter_type=(ToolParameterType.STRING),
                        required=True,
                    ),
                ),
            ),
            ToolDefinition(
                name="dangerous_change",
                description=("Synthetic approval-required change."),
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
            content="http-provider-green",
        )


def _settings(
    *,
    enabled: bool = True,
) -> Settings:
    if not enabled:
        return Settings(
            environment="test",
            mcp_enabled=False,
        )

    return Settings(
        environment="test",
        mcp_enabled=True,
        mcp_host="testserver",
        mcp_issuer_url=("https://auth.example.invalid/"),
        mcp_resource_server_url=RESOURCE,
        mcp_required_scopes=("mcp:tools",),
        mcp_max_request_body_size=(1024 * 1024),
    )


def _composition() -> tuple[
    IntegrationToolProvider,
    Settings,
    AuthenticatedMCPServerAdapter,
]:
    provider = IntegrationToolProvider()

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
        call_id_factory=lambda: "feature16-internal-http-call",
    )

    access_policy = MCPAccessPolicy(
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

    settings = _settings()

    adapter = create_mcp_server_adapter(
        settings,
        tool_service=service,
        access_policy=access_policy,
        token_verifier=IntegrationVerifier(),
        transport_security=(
            TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[
                    "testserver",
                ],
                allowed_origins=[
                    ORIGIN,
                ],
            )
        ),
    )

    return (
        provider,
        settings,
        adapter,
    )


def _meta() -> dict[str, object]:
    return {
        ("io.modelcontextprotocol/protocolVersion"): PROTOCOL_VERSION,
        ("io.modelcontextprotocol/clientInfo"): {
            "name": "feature16-integration",
            "version": "1.0",
        },
        ("io.modelcontextprotocol/clientCapabilities"): {},
    }


def _headers(
    method: str,
    *,
    name: str | None = None,
    authenticated: bool = True,
) -> dict[str, str]:
    headers = {
        "Accept": ("application/json, text/event-stream"),
        "Content-Type": ("application/json"),
        "MCP-Protocol-Version": (PROTOCOL_VERSION),
        "Mcp-Method": method,
        "Origin": ORIGIN,
    }

    if name is not None:
        headers["Mcp-Name"] = name

    if authenticated:
        headers["Authorization"] = "Bearer " + _wire_credential()

    return headers


def test_app_rejects_mcp_configuration_mismatch() -> None:
    """MCP cannot be accidentally exposed or silently omitted."""
    _, _, adapter = _composition()

    try:
        create_app(
            _settings(),
        )
    except ValueError as exc:
        assert "no authenticated MCP adapter" in str(exc)
    else:
        raise AssertionError("enabled MCP without adapter must fail")

    try:
        create_app(
            _settings(enabled=False),
            mcp_adapter=adapter,
        )
    except ValueError as exc:
        assert "must not be mounted" in str(exc)
    else:
        raise AssertionError("disabled MCP with adapter must fail")


def test_host_keeps_health_and_protected_metadata_routes() -> None:
    """Mounting MCP preserves existing FastAPI and OAuth metadata routes."""
    _, settings, adapter = _composition()

    application = create_app(
        settings,
        mcp_adapter=adapter,
    )

    with TestClient(
        application,
        base_url="http://testserver",
    ) as client:
        health = client.get("/health")

        metadata = client.get(
            "/.well-known/oauth-protected-resource/mcp",
            headers={
                "Origin": ORIGIN,
            },
        )

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
    }

    assert metadata.status_code == 200


def test_host_mcp_rejects_missing_bearer_token() -> None:
    """Bearer middleware rejects unauthenticated MCP traffic."""
    provider, settings, adapter = _composition()

    application = create_app(
        settings,
        mcp_adapter=adapter,
    )

    with TestClient(
        application,
        base_url="http://testserver",
    ) as client:
        response = client.post(
            "/mcp",
            headers=_headers(
                "tools/list",
                authenticated=False,
            ),
            json={
                "jsonrpc": "2.0",
                "id": "unauth-list",
                "method": "tools/list",
                "params": {
                    "_meta": _meta(),
                },
            },
        )

    assert response.status_code == 401
    assert provider.invocations == []


def test_host_mcp_modern_tools_list_is_scope_filtered() -> None:
    """Modern tools/list crosses real auth and scope-policy boundaries."""
    provider, settings, adapter = _composition()

    application = create_app(
        settings,
        mcp_adapter=adapter,
    )

    with TestClient(
        application,
        base_url="http://testserver",
    ) as client:
        response = client.post(
            "/mcp",
            headers=_headers("tools/list"),
            json={
                "jsonrpc": "2.0",
                "id": "list-1",
                "method": "tools/list",
                "params": {
                    "_meta": _meta(),
                },
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "list-1"

    tools = payload["result"]["tools"]

    assert tuple(tool["name"] for tool in tools) == ("read_status",)

    assert provider.invocations == []


def test_host_mcp_modern_tools_call_executes_controlled_tool_once() -> None:
    """Real HTTP tools/call reaches controlled execution exactly once."""
    provider, settings, adapter = _composition()

    application = create_app(
        settings,
        mcp_adapter=adapter,
    )

    with TestClient(
        application,
        base_url="http://testserver",
    ) as client:
        response = client.post(
            "/mcp",
            headers=_headers(
                "tools/call",
                name="read_status",
            ),
            json={
                "jsonrpc": "2.0",
                "id": "call-1",
                "method": "tools/call",
                "params": {
                    "name": "read_status",
                    "arguments": {
                        "subject": "feature16",
                    },
                    "_meta": _meta(),
                },
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "call-1"

    result = payload["result"]

    assert result["isError"] is False

    assert result["content"] == [
        {
            "type": "text",
            "text": "http-provider-green",
        },
    ]

    assert len(provider.invocations) == 1

    invocation = provider.invocations[0]

    assert invocation.call_id == ("feature16-internal-http-call")

    assert invocation.call_id != ("call-1")

    assert invocation.tool_name == ("read_status")


def test_host_mcp_claim_cannot_escalate_to_approval_tool() -> None:
    """Malicious claims cannot grant approval-required capabilities."""
    provider, settings, adapter = _composition()

    application = create_app(
        settings,
        mcp_adapter=adapter,
    )

    with TestClient(
        application,
        base_url="http://testserver",
    ) as client:
        response = client.post(
            "/mcp",
            headers=_headers(
                "tools/call",
                name="dangerous_change",
            ),
            json={
                "jsonrpc": "2.0",
                "id": "call-dangerous",
                "method": "tools/call",
                "params": {
                    "name": "dangerous_change",
                    "arguments": {},
                    "_meta": _meta(),
                },
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["result"]["isError"] is True

    assert provider.invocations == []
