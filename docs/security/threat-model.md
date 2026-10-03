# Threat Model

## Status

Version: 0.2

This threat model describes the initial and evolving security assumptions
and threat categories for the AI Engineering & Agent Platform.

It is reviewed as additional executable platform capabilities and trust
boundaries are introduced.

## Security Objectives

The platform should protect:

- confidentiality of user and platform data;
- integrity of knowledge sources;
- integrity of AI execution;
- integrity of tools and workflows;
- availability of platform services;
- credentials and secrets;
- authorization boundaries;
- auditability of privileged actions;
- evaluation integrity;
- software supply-chain integrity.

## Trust Boundaries

Initial trust boundaries include:

1. user to platform;
2. Developer Console to API;
3. API to internal services;
4. platform to model provider;
5. platform to embedding provider;
6. platform to knowledge sources;
7. platform to database;
8. agent runtime to tools;
9. platform to MCP servers;
10. platform to workflow systems;
11. application to observability systems;
12. CI/CD to deployment environments;
13. repository to third-party dependencies.

Crossing a trust boundary must not implicitly establish unlimited trust.

## Protected Assets

Important assets include:

- source code;
- repository integrity;
- credentials;
- access tokens;
- API keys;
- private keys;
- user data;
- knowledge-base content;
- vector data;
- model configuration;
- prompts;
- agent definitions;
- tool definitions;
- workflow definitions;
- evaluation datasets;
- evaluation results;
- audit records;
- telemetry;
- infrastructure configuration.

## Threat Actors

Possible threat actors include:

- unauthenticated external users;
- authenticated users exceeding intended permissions;
- compromised user accounts;
- malicious knowledge-source authors;
- malicious document providers;
- malicious or compromised external services;
- compromised MCP servers;
- malicious dependencies;
- compromised CI/CD identities;
- compromised developer environments;
- automated abuse.

Accidental misuse and configuration mistakes are also considered threat
sources.

## AI Threat Categories

### Prompt Injection

Untrusted user input may attempt to alter intended model behavior.

Potential effects include:

- policy bypass;
- unauthorized tool use;
- disclosure attempts;
- instruction hierarchy manipulation.

Controls will evolve alongside model execution.

### Indirect Prompt Injection

Content retrieved from documents, websites, databases, tools, or other
external sources may contain instructions intended to manipulate the
model.

Retrieved content must be treated as data rather than automatically
trusted instructions.

### Knowledge-Base Poisoning

An attacker may attempt to introduce misleading, malicious, or
manipulated content into a knowledge base.

Planned controls include:

- controlled ingestion;
- provenance;
- authorization;
- source metadata;
- validation;
- auditability.

### Retrieval Manipulation

Attackers may attempt to influence retrieval ranking or metadata to
increase the probability that malicious content enters model context.

Retrieval behavior should eventually be covered by adversarial
evaluation.

### Unsafe Model Output

Model output may be:

- incorrect;
- misleading;
- malformed;
- unsafe;
- unauthorized;
- incompatible with downstream systems.

Model output must not automatically become a trusted command.

### Tool Abuse

AI-generated decisions may attempt to invoke tools outside the user's
intended purpose.

Tool access should eventually enforce:

- explicit registration;
- authorization;
- scoped credentials;
- validated inputs;
- bounded execution;
- timeout;
- risk classification;
- approval where required.

### Excessive Agency

An agent may perform more actions than necessary or enter repeated
execution loops.

Controls may include:

- maximum execution steps;
- maximum elapsed time;
- resource budgets;
- tool allowlists;
- explicit completion conditions;
- human approval.

### Human-Approval Bypass

Sensitive operations intended to require approval may be executed
through another path.

Approval requirements must be enforced by application policy rather than
only described in prompts.

### MCP Trust Failure

External MCP capabilities may provide malicious or misleading metadata,
tools, resources, or outputs.

Connecting to an MCP server must not implicitly authorize every
capability it exposes.

### Data Exfiltration

Model, tool, workflow, retrieval, or observability operations may expose
protected data to unintended systems.

Data flows must eventually have explicit destinations and authorization
boundaries.

## Application Threat Categories

### Authentication Bypass

Attackers may attempt to access protected resources without valid
identity.

### Authorization Bypass

Authenticated identities may attempt to access resources or operations
outside their permissions.

### Injection

Application inputs may target:

- databases;
- command execution;
- templates;
- structured query systems;
- downstream integrations.

Input handling must use safe interfaces and explicit validation.

### SSRF

Tools, ingestion components, or external-resource integrations may be
abused to access unintended network destinations.

Network-access capabilities must be constrained.

### Path Traversal

File ingestion and processing must prevent access outside approved
storage boundaries.

### Arbitrary Code Execution

AI output, document content, tool parameters, or workflow content must
not automatically become executable code.

Any future code-execution capability will require an explicit isolation
model.

## Data Threat Categories

### Cross-User Data Exposure

Retrieval or API behavior may return content owned by another user or
tenant.

Authorization must be enforced before retrieval results become model
context.

### Sensitive Data in Telemetry

Prompts, tool inputs, retrieved documents, or outputs may contain
sensitive information.

Telemetry must support redaction and data minimization.

### Dataset Contamination

Evaluation results may become misleading if evaluation datasets are
modified unintentionally or by an attacker.

Dataset provenance and versioning will be required as evaluation
capabilities mature.

## Supply-Chain Threat Categories

Potential risks include:

- vulnerable dependencies;
- dependency confusion;
- malicious packages;
- compromised container images;
- compromised build actions;
- unexpected model artifacts;
- model-license incompatibilities;
- tampered downloaded artifacts.

Relevant verification and scanning will be introduced when these
artifacts become part of the implementation.

## Secrets Threats

Secrets must not be:

- committed to Git;
- embedded in container images;
- included in example configuration;
- exposed in logs;
- sent to models without explicit need;
- included in evaluation datasets.

Runtime secrets should eventually use environment-appropriate
secret-management mechanisms.

## Availability Threats

Potential availability risks include:

- oversized requests;
- expensive model requests;
- excessive retrieval;
- unbounded agent loops;
- tool retry storms;
- workflow loops;
- queue exhaustion;
- database exhaustion;
- resource exhaustion.

Execution budgets and rate controls will be introduced alongside the
relevant capabilities.

## Initial Security Invariants

The following rules should remain true as the project evolves:

1. no secret belongs in source control;
2. model output is untrusted input to downstream systems;
3. retrieved content is not trusted instruction;
4. tool execution requires explicit platform control;
5. human approval cannot be implemented only through prompt wording;
6. external services do not automatically receive platform secrets;
7. authorization must precede access to protected data;
8. telemetry must not indiscriminately capture sensitive content;
9. third-party dependencies and models retain their own licensing and
   security considerations;
10. privileged AI actions must be auditable;
11. execution must eventually have bounded resource consumption;
12. planned security controls must become executable tests when the
    corresponding functionality is implemented.

## Current LLM Runtime Security Posture

The platform-to-model-provider trust boundary is now active for the Ollama
LLM adapter.

Current controls include:

- a default Ollama base URL restricted to loopback
  (`http://127.0.0.1:11434`);
- configuration validation requiring an absolute HTTP or HTTPS origin;
- rejection of embedded URL credentials;
- rejection of URL paths, query parameters, and fragments in the configured
  provider origin;
- a positive finite request timeout;
- provider-specific HTTP execution isolated behind the adapter boundary;
- normalized timeout and transport failures;
- normalized HTTP-status failures;
- provider response validation before conversion to platform domain models;
- rejection of malformed JSON and malformed response structures;
- rejection of non-empty Ollama `tool_calls` while platform LLM tool calling
  remains unsupported;
- preservation of underlying exception causes without copying remote response
  bodies into normalized platform-error messages;
- runtime-owned HTTP-client creation and cleanup;
- mocked-transport tests that do not require external network access.

The configured model endpoint is still a trust boundary. Configuration can
point to non-loopback HTTP or HTTPS origins, so operators are responsible for
selecting an intended endpoint. The current feature does not implement a
network-destination allowlist, provider authentication, certificate pinning,
automatic retries, or outbound network policy enforcement.

Model output remains untrusted data. Successful model generation does not
authorize tool execution, command execution, privileged actions, or access to
protected resources.

## Current Limitations

The repository now contains a concrete Ollama LLM adapter and runtime
composition capable of non-streaming model invocation through the existing
provider-neutral LLM contract.

The provider-neutral contracts themselves remain free of provider-specific
execution behavior. Ollama-specific mapping, HTTP execution, configuration,
error normalization, and client lifecycle are contained by adapter and
runtime layers.

The current LLM runtime does not implement:

- streaming model responses;
- LLM tool-calling semantics or tool execution;
- automatic retries;
- provider authentication;
- model routing;
- public model-generation API endpoints;
- per-user or per-tenant model authorization;
- AI-specific telemetry or model-operation tracing.

There is also no embedding-provider adapter, persistent database, production
vector-store integration, RAG pipeline, agent runtime, MCP integration, or
workflow runtime.

AI-specific attack surfaces associated with retrieval, tools, agents, MCP,
workflows, and observability therefore remain anticipatory. Controls for
those attack surfaces must become executable and testable as the
corresponding runtime capabilities are introduced.

Each future feature must update this threat model when it materially
changes:

- assets;
- trust boundaries;
- attack surface;
- data flows;
- privilege;
- external dependencies.

## Review Trigger

This threat model must be reviewed whenever the project introduces or
materially changes:

- authentication;
- authorization;
- model providers;
- knowledge ingestion;
- vector retrieval;
- agent execution;
- tools;
- MCP;
- workflow automation;
- human approval;
- external network access;
- persistent user data;
- observability exports;
- deployment infrastructure.
