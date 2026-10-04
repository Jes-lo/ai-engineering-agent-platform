"""Authenticated MCP Streamable HTTP adapter for platform-owned tools."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from mcp.server import Server, ServerRequestContext
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

from ai_engineering_agent_platform.contracts import (
    MCPToolCallRequest,
    ToolArgument,
    ToolDefinition,
    ToolExecutionStatus,
)
from ai_engineering_agent_platform.contracts.tool import ToolArgumentValue
from ai_engineering_agent_platform.services import (
    MCPServerPolicyError,
    OwnedMCPToolService,
)
from ai_engineering_agent_platform.services.tool_execution import (
    ToolExecutionAuthorization,
    ToolExecutionControlError,
)

DEFAULT_MCP_MAX_REQUEST_BODY_SIZE = 1024 * 1024
DEFAULT_MCP_REQUIRED_SCOPES = ("mcp:tools",)
_LOCAL_HOSTS = frozenset(
    {
        "127.0.0.1",
        "localhost",
        "::1",
    }
)


class MCPWireError(Exception):
    """Base error for the authenticated MCP wire boundary."""


class MCPWireAuthorizationError(MCPWireError):
    """Raised when verified transport identity cannot authorize a request."""


class MCPWireProtocolError(MCPWireError):
    """Raised when MCP wire input cannot map safely to platform contracts."""


@dataclass(frozen=True, slots=True)
class MCPToolScopeGrant:
    """Platform-owned mapping from one verified OAuth scope to local tools."""

    scope: str
    tool_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate one explicit scope-to-capability mapping."""
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("scope must be a non-empty string")

        if not isinstance(self.tool_names, tuple):
            raise ValueError("tool_names must be a tuple")

        if not self.tool_names:
            raise ValueError("tool_names must not be empty")

        for tool_name in self.tool_names:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tool names must be non-empty strings")

        if len(self.tool_names) != len(set(self.tool_names)):
            raise ValueError("tool_names must be unique")


@dataclass(frozen=True, slots=True)
class MCPAccessPolicy:
    """Derive local execution authority only from verified token scopes."""

    grants: tuple[MCPToolScopeGrant, ...]

    def __post_init__(self) -> None:
        """Require deterministic unique platform-owned scope mappings."""
        if not isinstance(self.grants, tuple):
            raise ValueError("grants must be a tuple")

        if not self.grants:
            raise ValueError("grants must not be empty")

        if any(not isinstance(grant, MCPToolScopeGrant) for grant in self.grants):
            raise ValueError("grants must contain MCPToolScopeGrant values")

        scopes = tuple(grant.scope for grant in self.grants)

        if len(scopes) != len(set(scopes)):
            raise ValueError("scope mappings must be unique")

    def authorization_for(
        self,
        access_token: AccessToken,
    ) -> ToolExecutionAuthorization:
        """Build a no-approval allowlist from verified token scopes only."""
        if not isinstance(access_token, AccessToken):
            raise MCPWireAuthorizationError(
                "authenticated MCP access token is unavailable"
            )

        token_scopes = set(access_token.scopes)
        allowed: list[str] = []

        for grant in self.grants:
            if grant.scope not in token_scopes:
                continue

            for tool_name in grant.tool_names:
                if tool_name not in allowed:
                    allowed.append(tool_name)

        return ToolExecutionAuthorization(
            allowed_tool_names=tuple(allowed),
            approval_grants=(),
        )


def _tool_definition_to_mcp_tool(
    definition: ToolDefinition,
) -> Tool:
    """Map one platform-owned tool definition to exact MCP JSON Schema."""
    properties: dict[str, object] = {}
    required: list[str] = []

    for parameter in definition.parameters:
        json_type = parameter.parameter_type.value

        parameter_schema: dict[str, object]

        if parameter.required:
            parameter_schema = {
                "type": json_type,
            }
            required.append(parameter.name)
        else:
            parameter_schema = {
                "type": [
                    json_type,
                    "null",
                ],
            }

        if parameter.description is not None:
            parameter_schema["description"] = parameter.description

        properties[parameter.name] = parameter_schema

    input_schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }

    if required:
        input_schema["required"] = required

    return Tool(
        name=definition.name,
        description=definition.description,
        input_schema=input_schema,
    )


def _portable_argument_value(
    value: object,
) -> ToolArgumentValue:
    """Accept only scalar values supported by platform tool contracts."""
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not isfinite(value):
            raise MCPWireProtocolError("MCP tool arguments must contain finite numbers")

        return value

    raise MCPWireProtocolError("MCP tool arguments must contain portable scalar values")


def _arguments_from_mcp(
    arguments: dict[str, object] | None,
) -> tuple[ToolArgument, ...]:
    """Map untrusted MCP arguments without coercing scalar values."""
    if arguments is None:
        return ()

    if not isinstance(arguments, dict):
        raise MCPWireProtocolError("MCP tool arguments must be an object")

    mapped: list[ToolArgument] = []

    for name, value in arguments.items():
        if not isinstance(name, str) or not name.strip():
            raise MCPWireProtocolError("MCP argument names must be non-empty strings")

        mapped.append(
            ToolArgument(
                name=name,
                value=_portable_argument_value(value),
            )
        )

    return tuple(mapped)


def _request_id_from_context(
    context: ServerRequestContext,
) -> str:
    """Return one concrete protocol request identifier."""
    if context.request_id is None:
        raise MCPWireProtocolError("tools/call requires an MCP request identifier")

    request_id = str(context.request_id)

    if not request_id.strip():
        raise MCPWireProtocolError("MCP request identifier must not be empty")

    return request_id


AccessTokenGetter = Callable[[], AccessToken | None]


class AuthenticatedMCPServerAdapter:
    """Expose OwnedMCPToolService through authenticated Streamable HTTP."""

    def __init__(
        self,
        *,
        tool_service: OwnedMCPToolService,
        access_policy: MCPAccessPolicy,
        token_verifier: TokenVerifier,
        issuer_url: str,
        resource_server_url: str,
        required_scopes: tuple[str, ...] = DEFAULT_MCP_REQUIRED_SCOPES,
        host: str = "127.0.0.1",
        max_request_body_size: int = DEFAULT_MCP_MAX_REQUEST_BODY_SIZE,
        transport_security: TransportSecuritySettings | None = None,
        access_token_getter: AccessTokenGetter = get_access_token,
    ) -> None:
        """Create one fail-closed authenticated MCP ASGI adapter."""
        if not isinstance(tool_service, OwnedMCPToolService):
            raise ValueError("tool_service must be an OwnedMCPToolService")

        if not isinstance(access_policy, MCPAccessPolicy):
            raise ValueError("access_policy must be an MCPAccessPolicy")

        if not isinstance(required_scopes, tuple):
            raise ValueError("required_scopes must be a tuple")

        if not required_scopes:
            raise ValueError("required_scopes must not be empty")

        if any(
            not isinstance(scope, str) or not scope.strip() for scope in required_scopes
        ):
            raise ValueError("required scopes must be non-empty strings")

        if len(required_scopes) != len(set(required_scopes)):
            raise ValueError("required scopes must be unique")

        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")

        if host not in _LOCAL_HOSTS and transport_security is None:
            raise ValueError("non-local MCP host requires explicit transport_security")

        if (
            not isinstance(max_request_body_size, int)
            or isinstance(max_request_body_size, bool)
            or max_request_body_size <= 0
        ):
            raise ValueError("max_request_body_size must be a positive integer")

        self._tool_service = tool_service
        self._access_policy = access_policy
        self._access_token_getter = access_token_getter

        self._server = Server(
            "ai-engineering-agent-platform",
            on_list_tools=self.handle_list_tools,
            on_call_tool=self.handle_call_tool,
        )

        self._app = self._server.streamable_http_app(
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
            max_request_body_size=max_request_body_size,
            host=host,
            transport_security=transport_security,
            token_verifier=token_verifier,
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(issuer_url),
                resource_server_url=AnyHttpUrl(resource_server_url),
                required_scopes=list(required_scopes),
                validate_token_resource=True,
            ),
        )

    @property
    def app(self) -> Starlette:
        """Return the authenticated standalone MCP ASGI application."""
        return self._app

    @property
    def server(self) -> Server:
        """Return the low-level server for later host-lifespan integration."""
        return self._server

    def _current_authorization(
        self,
    ) -> ToolExecutionAuthorization:
        """Derive authority from the token already verified by middleware."""
        access_token = self._access_token_getter()

        if access_token is None:
            raise MCPWireAuthorizationError("verified MCP access token is unavailable")

        return self._access_policy.authorization_for(access_token)

    async def handle_list_tools(
        self,
        context: ServerRequestContext,
        params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        """Expose only tools authorized for the verified principal."""
        del context
        del params

        authorization = self._current_authorization()

        definitions = self._tool_service.list_tools(authorization=authorization)

        return ListToolsResult(
            tools=[
                _tool_definition_to_mcp_tool(definition) for definition in definitions
            ]
        )

    async def handle_call_tool(
        self,
        context: ServerRequestContext,
        params: CallToolRequestParams,
    ) -> CallToolResult:
        """Map one authenticated MCP tools/call into controlled execution."""
        try:
            authorization = self._current_authorization()

            request = MCPToolCallRequest(
                request_id=_request_id_from_context(context),
                tool_name=params.name,
                arguments=_arguments_from_mcp(params.arguments),
            )

            response = await self._tool_service.call_tool(
                request,
                authorization=authorization,
            )
        except (
            MCPServerPolicyError,
            MCPWireProtocolError,
            ToolExecutionControlError,
            ValueError,
        ):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="tool request rejected",
                    ),
                ],
                is_error=True,
            )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=response.content,
                ),
            ],
            is_error=(response.status is ToolExecutionStatus.ERROR),
        )
