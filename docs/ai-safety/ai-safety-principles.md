# AI Safety Principles

## Purpose

This document defines platform-level safety principles for AI-enabled
behavior within the AI Engineering & Agent Platform.

These principles complement traditional application and infrastructure
security controls.

## 1. Treat Model Output as Untrusted

Model-generated content must not automatically be treated as:

- executable code;
- authorization;
- trusted configuration;
- trusted database input;
- trusted tool parameters;
- factual truth.

Downstream systems must apply their own validation and authorization.

## 2. Treat Retrieved Content as Data

Documents, websites, database records, tool outputs, and other retrieved
content may contain malicious or misleading instructions.

Retrieved content must not automatically override platform policy or
trusted instructions.

## 3. Authorization Is Enforced Outside the Model

The model must not be the final authority for access control.

Authentication and authorization decisions must be enforced by
deterministic platform controls.

## 4. Tool Access Is Explicit

Agents must not automatically receive access to every available tool.

Tool availability should depend on:

- identity;
- permissions;
- execution context;
- risk;
- task requirements.

## 5. High-Impact Actions May Require Human Approval

Human approval should be required when an AI-driven operation has
sufficient impact or risk.

Approval must be enforced by platform logic.

A prompt saying "ask the user first" is not an enforceable approval
control.

## 6. Agent Execution Is Bounded

The platform now applies bounds at both the controlled single-turn layer and
the conversational loop layer.

Each `AgentTurnRequest` has a positive `max_steps`, with the absolute
single-turn ceiling `MAX_AGENT_TURN_STEPS = 8`.

`BoundedAgentLoopService` additionally enforces two global ceilings across the
complete conversation:

- `MAX_AGENT_LOOP_MODEL_TURNS = 8`;
- `MAX_AGENT_LOOP_TOOL_CALLS = 8`.

The tool-call budget is cumulative across model turns rather than reset for
each generation. Once the tool budget is exhausted, tools are removed from the
next controlled model turn. A model proposal that exceeds the remaining
budget fails before provider execution.

Bounded execution also requires capability control rather than prompt wording.
Every executable proposal continues through `ControlledAgentService`, so only
registered, enabled, explicitly authorized tools may become model-visible and
every execution retains the existing fresh-identity, preflight, approval, and
result-validation controls.

Tool results returned to the LLM are untrusted context. Typed
`LLMAssistantToolCallMessage` and `LLMToolResultMessage` values preserve
conversation structure, while transcript validation rejects orphaned,
incomplete, or mismatched tool-result sequences. Tool output cannot grant
authorization for a later action.

Approval-required execution pauses before provider side effects. Resume uses
the exact service-issued plan without regenerating the model decision.
Continuation state and replay protection remain process-local and in-memory;
they are not durable approval workflow state or authenticated human identity.

Sequential multi-tool execution is not atomic. Earlier successful side effects
are not rolled back if a later provider execution fails, and automatic retries
after execution starts remain intentionally absent.

Future hardening includes durable/distributed continuation state, authenticated human-approval lifecycle controls, elapsed-time/rate budgets, stronger audit/observability support, outbound remote MCP wire transport, MCP resources/prompts, and workflow capabilities. Those additions must preserve the same authority boundaries.

## 7. Minimize Sensitive Context

Only data required for a task should be exposed to:

- models;
- tools;
- workflows;
- telemetry;
- external providers.

Secrets and unrelated sensitive data should not be included in prompts
or retrieved context.

## 8. External AI Systems Are Trust Boundaries

Remote model providers, embedding providers, rerankers, MCP servers,
tools, and workflow systems must be treated as external trust
boundaries.

Connecting a service does not automatically authorize it to receive
all platform data.

The MCP boundary applies the same rule to both remote capabilities and
project-owned tool exposure: protocol data is not authority. Remote discovery
remains untrusted, explicitly bound, and subordinate to platform-owned tool
definitions and execution policy.

The concrete inbound Streamable HTTP server authenticates Bearer material
through an injected `TokenVerifier`, but authentication is only the first
gate. `MCPAccessPolicy` converts verified token scopes through platform-owned
mappings into execution authorization. Token claims cannot self-assert a tool
allowlist, and the wire layer cannot manufacture approval grants.

Project-owned MCP exposure remains capability-limited. Tools must be explicitly
exported, enabled, and authorized, while approval-required tools remain
excluded. External request identifiers do not become platform execution
identities, and tool execution still crosses `ToolExecutionService`.

MCP is disabled by default. Production issuer/resource URLs require HTTPS,
non-local hosts require explicit transport-security configuration, and request
body size is bounded. These controls reduce accidental exposure but do not
replace a production identity provider, TLS termination, rate limiting,
monitoring, or durable human approval.

Remote MCP results and protocol content remain untrusted context. They must not
override system policy, expand later tool authority, or be interpreted as
trusted instructions merely because they arrived over an authenticated
transport.

## 9. Evaluate AI Behavior

AI behavior should be evaluated systematically.

Evaluation should evolve to include:

- correctness where measurable;
- grounding;
- retrieval quality;
- citation quality;
- policy compliance;
- tool selection;
- tool execution;
- adversarial inputs;
- regressions.

Manual demonstrations alone are insufficient evidence of reliability.

## 10. Prefer Deterministic Controls

Where deterministic software can reliably enforce a safety or security
requirement, deterministic enforcement should be preferred over relying
solely on model instructions.

## 11. Fail Safely

When authorization, validation, policy evaluation, tool execution, or
approval state is uncertain, the system should prefer a safe failure
over silently performing a privileged action.

## 12. Preserve Auditability

Privileged AI-driven operations should eventually provide enough
structured information to determine:

- what initiated the action;
- which identity or context was involved;
- which agent or workflow executed;
- which tool was called;
- what authorization was applied;
- whether approval was required;
- what result occurred.

Sensitive content should not be indiscriminately stored merely for
auditability.

## 13. Maintain Human Control

Users and operators should retain meaningful control over consequential
AI-driven operations.

Automation should not remove human decision points solely for the
purpose of making a demonstration appear more autonomous.

## 14. Safety Evolves With Capability

Controls should be introduced when the corresponding risk becomes real.

The project should not claim that a safety control exists until it is
implemented and, where practical, tested.
