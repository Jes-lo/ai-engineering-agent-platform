"""Runtime composition for authenticated MCP Streamable HTTP."""

from mcp.server.auth.provider import TokenVerifier
from mcp.server.transport_security import TransportSecuritySettings

from ai_engineering_agent_platform.api.mcp_server import (
    AuthenticatedMCPServerAdapter,
    MCPAccessPolicy,
)
from ai_engineering_agent_platform.config import Settings
from ai_engineering_agent_platform.services import (
    OwnedMCPToolService,
)


def create_mcp_server_adapter(
    settings: Settings,
    *,
    tool_service: OwnedMCPToolService,
    access_policy: MCPAccessPolicy,
    token_verifier: TokenVerifier,
    transport_security: TransportSecuritySettings | None = None,
) -> AuthenticatedMCPServerAdapter:
    """Compose the MCP wire adapter from validated runtime settings."""
    if not settings.mcp_enabled:
        raise ValueError("MCP runtime cannot be created while MCP is disabled")

    issuer_url = settings.mcp_issuer_url
    resource_server_url = settings.mcp_resource_server_url

    if issuer_url is None:
        raise ValueError("mcp_issuer_url is required when MCP is enabled")

    if resource_server_url is None:
        raise ValueError("mcp_resource_server_url is required when MCP is enabled")

    return AuthenticatedMCPServerAdapter(
        tool_service=tool_service,
        access_policy=access_policy,
        token_verifier=token_verifier,
        issuer_url=issuer_url,
        resource_server_url=resource_server_url,
        required_scopes=settings.mcp_required_scopes,
        host=settings.mcp_host,
        max_request_body_size=(settings.mcp_max_request_body_size),
        transport_security=transport_security,
    )
