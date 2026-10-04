# Threat Model

## Status

Version: 1.4

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

### Current Controlled Tool Execution Foundation

The platform now has an application-owned tool execution control layer.

Current controls include:

- tool providers must identify as tool providers;
- provider identities must be unique inside one registry;
- registered tool names must be globally unique;
- every registered tool requires explicit policy coverage;
- unknown policy entries fail closed;
- disabled tools cannot execute;
- every execution requires an explicit tool allowlist;
- approval-required tools require an exact structural grant bound to
  `call_id` and `tool_name`;
- required and unexpected arguments are validated before provider execution;
- portable scalar argument types are checked without coercion;
- provider results must retain the exact invocation `call_id` and
  `tool_name`;
- provider failures are propagated rather than silently retried;
- this layer provides no shell, filesystem, or network tool implementation.

The current approval grant is not a complete human-in-the-loop security
mechanism. It has no authenticated approver identity, persistence, signature,
expiry, revocation, or single-use consumption semantics.

Ollama `tool_calls` are now accepted only as untrusted proposals when the
request explicitly exposed the matching tool name. An unrequested tool name
fails closed. A parsed proposal remains separate from `ToolInvocation`,
`ToolExecutionAuthorization`, approval state, and `ToolExecutionService`, so
model output cannot directly acquire execution authority.

Bounded conversational looping and typed tool-result round trips are now
implemented through `BoundedAgentLoopService`. Durable/distributed continuation
state, richer authorization scopes, authenticated human-approval lifecycle
controls, MCP integration, workflows, and n8n integration remain future work.

### LLM Tool-Call Proposal Threat Boundary

LLM tool proposals are untrusted model output.

Current controls are:

- only tool definitions explicitly supplied in `LLMRequest.tools` are exposed
  to Ollama;
- request tool names must be unique;
- a response proposing a tool outside that request set fails closed;
- Ollama tool-call arguments must be JSON objects containing only portable
  scalar values;
- provider tool-call identifiers are optional metadata and do not become
  platform execution identifiers;
- proposals normalize to immutable `LLMToolCall` values;
- tool proposals require `FinishReason.TOOL_CALLS`;
- proposal parsing never invokes `ToolExecutionService`;
- Feature 11 registration, allowlist, approval, argument validation, and
  result-identity controls remain unchanged;
- grounded generation remains STOP-only and therefore rejects tool-call
  proposal responses.

The LLM proposal layer alone still does not execute tools or grant authority.
`ControlledAgentService` remains the execution bridge from accepted proposals
to controlled `ToolInvocation` values. `BoundedAgentLoopService` may now return
validated `ToolResult` content to the model through typed
`LLMToolResultMessage` values and explicit `MessageRole.TOOL` semantics, but a
tool result is untrusted model context rather than authorization. Any later
model proposal must pass through the controlled execution boundary again.
Durable/authenticated HITL state, MCP, workflows, n8n, and
shell/filesystem/network tools remain outside this feature.

### Controlled Agent Turn Orchestration Threat Boundary

The controlled single-turn agent surface introduces an execution bridge between
untrusted model proposals and registered tool providers.

Current controls are:

- caller-supplied `run_id` is correlation metadata only;
- the caller cannot inject arbitrary `LLMRequest.tools` into this orchestration
  surface;
- only registered, enabled, explicitly authorized tools become model-visible;
- proposals outside that exact exposure set fail closed;
- each turn has a caller-selected step budget under the absolute platform
  ceiling `MAX_AGENT_TURN_STEPS = 8`;
- provider-originating call IDs remain metadata and cannot become execution
  authority;
- the default execution call ID includes a fresh UUID4 nonce created by the
  platform;
- every planned invocation is preflighted before the first provider side
  effect;
- missing structural approvals pause the whole plan instead of partially
  executing it;
- a resumable plan must be an active continuation issued by the same service
  instance;
- successful resume consumes the continuation before provider execution,
  preventing same-instance replay;
- a resumed plan reuses the original model proposal and does not ask the model
  to regenerate a possibly different action.

Residual limitations are explicit:

- pending continuations exist only in process memory;
- a process restart invalidates that continuation instead of restoring it;
- replay protection is therefore process-local rather than durable;
- `ToolApprovalGrant` remains structural evidence, not authenticated human
  identity or approval provenance;
- sequential multi-tool execution is not transactional;
- a later provider failure cannot roll back an earlier successful side effect;
- automatic retry after tool execution starts is intentionally absent;
- the single-turn execution service itself does not persist or own multi-turn
  conversation history; Feature 14 composes it through a separate bounded loop;
- bounded conversational looping exists, but its pending continuation state is
  still process-local rather than durable or distributed;
- there is no MCP, workflow engine, n8n execution, shell tool, filesystem tool,
  or network tool in this orchestration layer.

### Bounded Conversational Agent Loop Threat Boundary

Feature 14 introduces repeated model/tool interaction, which increases the
risk of excessive agency, tool-output prompt injection, repeated side effects,
and budget-reset mistakes.

Current controls include:

- model-visible tools continue to come only from the registered, enabled, and
  explicitly authorized exposure set;
- every executable proposal still passes through `ControlledAgentService`;
- the loop itself has no direct `ToolExecutionService` execution path;
- the loop has a hard maximum of eight model turns;
- the loop has a separate hard maximum of eight tool calls;
- tool-call consumption is cumulative across the complete loop;
- when no tool budget remains, tools are removed from subsequent model
  exposure;
- an oversized proposed batch fails before provider execution;
- validated `ToolResult` values are typed as `LLMToolResultMessage` before
  becoming model context;
- transcript validation rejects orphaned, incomplete, tool-name-mismatched, or
  provider-call-ID-mismatched tool-result sequences;
- internal platform execution `call_id` values are not serialized into the
  Ollama tool-result message;
- tool output remains untrusted context and does not grant execution authority;
- approval-required plans still pause before execution;
- approval resume reuses the frozen model decision rather than regenerating it;
- loop continuations must be active values issued by the same service instance.

Residual limitations remain explicit:

- pending loop and approval state is process-local and in-memory;
- process restart loses pending continuation state;
- replay protection is not durable across processes or replicas;
- approval grants still do not prove authenticated human identity;
- sequential multi-tool execution is not transactional;
- a later provider failure cannot roll back an earlier side effect;
- automatic retry after execution begins is intentionally absent;
- there is no global elapsed-time budget or rate limiter in this feature;
- MCP, workflow execution, n8n, shell tools, filesystem tools, and network
  tools are not introduced by this loop.

### MCP Trust Failure

External MCP discovery, metadata, tool results, and future protocol messages are
untrusted input. Connecting or configuring an MCP server must not implicitly
authorize every capability it exposes.

The current transport-neutral foundation applies these controls:

- remote capabilities require explicit `MCPToolBinding` entries;
- an unbound discovered remote tool is not automatically registered or exposed;
- duplicate remote discovery names fail closed;
- the platform owns the local tool name and description;
- discovered remote parameter schemas must exactly match the pinned local
  parameter schema before the adapter is created;
- the configured client `server_name` must match the trust-policy label;
- remote discovery alone grants no `ToolExecutionAuthorization`;
- remote execution integrated into the application remains subject to the
  existing `ToolRegistry`, `ToolExecutionPolicy`,
  `ToolExecutionAuthorization`, and `ToolExecutionService` boundaries;
- the project-owned MCP service exposes only explicitly configured, enabled,
  authorized tools that do not require approval;
- an inbound MCP `request_id` is correlation data rather than platform
  execution identity;
- the project-owned service creates a separate internal call ID, rejects an
  external/internal identity collision, and does not return the internal call
  ID in its response;
- the project-owned service delegates execution through
  `ToolExecutionService` rather than invoking a provider directly;
- remote MCP execution has no automatic retry loop in this foundation.

Residual limitations are explicit:

- `server_name` is a configured label, not authenticated or cryptographically
  verified remote identity;
- `MCPToolProviderAdapter`, like every `ToolProvider`, has an internal
  `execute()` method, so architectural composition must continue to route
  application execution through the controlled service rather than treating
  a provider object itself as authorization;
- there is no JSON-RPC, stdio, Streamable HTTP, or other concrete MCP wire
  transport yet;
- authentication and connection/session lifecycle are not implemented;
- MCP resources and prompts are not implemented;
- transport-level timeout, response-size, rate-limit, and network-destination
  controls are not yet implemented;
- remote result content remains untrusted and may contain prompt injection,
  misleading content, or sensitive data;
- the transport-neutral service accepts a platform
  `ToolExecutionAuthorization` object from its trusted caller; a future remote
  transport must derive that object from authenticated platform identity and
  policy rather than accepting client-asserted permissions;
- approval-required tools are excluded because authenticated durable HITL
  approval is not yet available;
- no shell, filesystem, or network tool is introduced by this foundation.

MCP wire transports therefore remain a future security boundary requiring
authentication, identity binding, destination controls, bounded payloads,
timeouts, rate limits, error normalization, and protocol-level adversarial
testing before production exposure.

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

The current bounded conversational agent loop now enforces separate hard
model-turn and tool-call budgets. Elapsed-time budgets, rate controls,
distributed quotas, and workflow-loop controls remain future hardening.

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
    corresponding functionality is implemented;
13. citation provenance must be resolved from platform-owned retrieved evidence
    rather than model-authored source identifiers.

## Current Ollama Runtime Security Posture

The platform-to-model-provider and platform-to-embedding-provider trust
boundaries are now active through the Ollama LLM and embedding adapters.

Current controls include:

- a default Ollama base URL restricted to loopback
  (`http://127.0.0.1:11434`);
- configuration validation requiring an absolute HTTP or HTTPS origin;
- rejection of embedded URL credentials;
- rejection of URL paths, query parameters, and fragments in the configured
  provider origin;
- a positive finite request timeout;
- provider-specific HTTP execution isolated behind adapter boundaries;
- a shared HTTP-client factory with runtime-owned lifecycle management;
- separate LLM and embedding provider contracts;
- embedding requests sent with `truncate=false` so oversized inputs are
  rejected instead of silently truncated;
- embedding response cardinality, dimensionality, numeric type, finite-value,
  model, and token-accounting validation before domain use;
- normalized timeout and transport failures;
- normalized HTTP-status failures;
- provider response validation before conversion to platform domain models;
- rejection of malformed JSON and malformed response structures;
- requested-tool validation and inert normalization of non-empty Ollama
  `tool_calls`, with execution authority remaining outside the LLM adapter;
- preservation of underlying exception causes without copying remote response
  bodies into normalized platform-error messages;
- runtime-owned HTTP-client creation and cleanup;
- provider implementations that do not own or close injected clients;
- mocked-transport tests that do not require external network access.

The configured Ollama endpoint is still a trust boundary. Configuration can
point to non-loopback HTTP or HTTPS origins, so operators are responsible for
selecting an intended endpoint. LLM prompts and embedding inputs sent to a
non-local endpoint may cross an external trust boundary and must be treated
accordingly.

Embedding batches can aggregate multiple input texts into one provider
request. Sensitive, confidential, personal, or tenant-isolated content must
not be sent to an embedding provider without the corresponding authorization
and data-handling controls.

The current feature does not implement a network-destination allowlist,
provider authentication, certificate pinning, automatic retries, outbound
network policy enforcement, per-user provider authorization, or provider-side
data-retention controls.

Model output remains untrusted data. Successful model generation does not
authorize tool execution, command execution, privileged actions, or access to
protected resources.

## Current PostgreSQL Persistence Security Posture

The platform-to-database trust boundary is now active through the PostgreSQL
runtime and `PostgreSQLVectorStoreProvider`.

Current controls include:

- PostgreSQL exposed to the local host through loopback by default;
- PostgreSQL host and local authentication initialized with SCRAM-SHA-256;
- database bootstrap/migration credentials kept separate from application
  runtime credentials;
- an application role fixed to `ai_platform_runtime`;
- `NOSUPERUSER`, `NOCREATEDB`, `NOCREATEROLE`, `NOINHERIT`,
  `NOREPLICATION`, and `NOBYPASSRLS` on the runtime role;
- a bounded runtime-role connection limit;
- no database-level `CREATE` privilege for the runtime role;
- schema `USAGE` without schema `CREATE`;
- table privileges limited to operations required by the current vector
  adapter;
- schema and table ownership retained by the bootstrap/migration identity;
- PostgreSQL credentials excluded from source control and local development
  secrets stored outside the repository;
- placeholder-only repository configuration examples;
- Alembic migrations executed separately from the application runtime;
- parameterized Psycopg SQL for dynamic values;
- `Jsonb` adaptation for metadata rather than interpolated JSON SQL;
- database constraints enforcing collection dimensionality and record shape;
- collection uniqueness over `(namespace, space_id)`, with `NULL` namespaces
  compared using `UNIQUE NULLS NOT DISTINCT`;
- explicit vector-space identity kept separate from logical namespace;
- indexing and retrieval reject embedding-provider model substitution before
  vector persistence or vector search;
- pre-isolation collections migrate to `legacy-unidentified` instead of being
  silently associated with a currently configured embedding model;
- lossy downgrade is rejected when one namespace contains multiple vector
  spaces;
- exact L2 vector search without HNSW or IVFFlat indexes at this stage;
- normalized database failures at the provider boundary;
- runtime-owned connection-pool lifecycle;
- isolated PostgreSQL integration tests using synthetic data and ephemeral
  credentials;
- cleanup validation proving the isolated integration database returns to a
  pristine application-data state;
- validation proving the integration environment does not mutate the normal
  development database;
- a dedicated PostgreSQL integration CI job requiring no repository database
  secrets.

The local Docker configuration and external development secret files are
development mechanisms, not a production secret-management, backup, high
availability, encryption, or disaster-recovery design.

## Current RAG Evaluation Security Posture

Deterministic RAG evaluation is now an active application surface.

Current controls include:

- every evaluation dataset has an explicit identifier, version, and
  provenance reference;
- case identifiers and expected relevant chunk identifiers must be unique;
- evaluation output must match the case query and namespace;
- grounding evidence must remain a subset of retrieval evidence;
- grounding evidence must preserve the exact retrieved evidence objects;
- citations must remain bound to evidence present in the final grounding
  input;
- retrieval, grounding, citation, and answer-status signals remain separate
  instead of being hidden behind one opaque overall score;
- evaluation cases execute in deterministic dataset order;
- RAG/provider failures remain visible rather than being converted into
  successful evaluation results;
- the current evaluation layer does not load external datasets, fetch network
  resources, open filesystem dataset paths, or invoke an LLM-as-a-judge.

These controls protect structural evaluation integrity. They do not establish
that dataset labels are semantically correct, unbiased, uncontaminated, or
authorized for every use. Dataset provenance and review remain necessary.

The current metrics evaluate evidence identity alignment. They do not prove
semantic entailment, factual correctness, absence of hallucination, or model
truthfulness.

## Current Grounded Generation Security Posture

Grounded generation is now an active application surface over already validated
retrieval evidence.

Current controls include:

- `GroundedGenerationService` accepts only a `RetrievalResponse` or
  `RerankedRetrievalResponse` supplied by the caller;
- retrieved evidence is serialized into a deterministic context payload and
  remains explicitly treated as untrusted data;
- the grounding prompt instructs the model not to treat retrieved evidence as
  instructions, but prompt wording is not treated as a complete security
  boundary;
- model responses must identify the exact configured model;
- only a completed `STOP` generation is accepted as a normal grounded result;
- empty generations fail closed;
- normal answers must contain at least one canonical citation marker;
- citation identifiers use the canonical `C1`, `C2`, ... grammar;
- malformed citation-like markers fail closed;
- citation identifiers outside the supplied evidence set fail closed;
- duplicate model references resolve to one platform citation object;
- the exact `INSUFFICIENT_EVIDENCE` sentinel produces an explicit abstention;
- mixing the abstention sentinel with asserted answer text fails closed;
- source reference, document identity, offsets, metadata, vector diagnostics,
  and other provenance remain owned by the retrieved evidence object rather
  than being accepted from model-authored output.

These controls establish citation integrity and provenance binding. They do not
establish semantic entailment between every generated claim and its citation,
prove that retrieved evidence is correct, or eliminate direct or indirect
prompt injection.

The grounding service also does not perform retrieval authorization. Callers
must enforce user or tenant authorization before retrieved evidence becomes
model context.

Presentation-layer citation rendering, semantic entailment/factual-correctness
evaluation, and stronger prompt injection defenses remain future work.

## Current Knowledge Ingestion Security Posture

Bounded caller-supplied text ingestion is now an active application surface
through `KnowledgeSource`, `parse_knowledge_source`, and
`KnowledgeIngestionService`.

Current controls include:

- source content crosses the ingestion boundary as caller-supplied bytes;
- only `text/plain` and `text/markdown` are accepted;
- a configured positive source-size limit is enforced before decoding;
- source bytes are decoded as strict UTF-8;
- NUL-bearing text is rejected;
- `source_ref` remains opaque provenance and is not interpreted as a
  filesystem path or network location;
- the current ingestion implementation imports no filesystem, network, or
  subprocess execution capability;
- chunk output must remain bound to the parsed document identity, source
  reference, title, metadata, exact source offsets, and source text;
- duplicate chunk identities and non-contiguous chunk indexes fail closed;
- existing `IndexingService` embedding/vector validation remains authoritative
  after ingestion validation;
- source, parsing, chunking, embedding, or persistence failures are not
  converted into successful ingestion results.

The current ingestion boundary does not fetch URLs, open filesystem paths,
parse PDF/DOCX/HTML or other rich binary formats, perform automatic retries,
implement document replacement/reindex lifecycle orchestration, or provide
authentication or tenant authorization.

Knowledge content remains untrusted data after successful ingestion. Passing
format, provenance, and chunk-integrity validation does not establish that the
source is truthful, safe, authorized for every caller, or free from indirect
prompt injection.

## Current End-to-End RAG Orchestration Security Posture

End-to-end retrieval-to-generation composition is now an active application
surface through `RAGService`.

Current controls include:

- deterministic stage ordering from retrieval to optional reranking to grounded
  generation;
- empty retrieval produces an explicit abstention before reranker or LLM
  execution;
- reranking remains optional and uses only the already validated retrieval
  candidates;
- the original retrieval result is retained separately from the exact grounding
  input used for model generation;
- citation provenance remains bound to platform-owned retrieved evidence rather
  than model-authored source identifiers;
- existing retrieval, reranking, and grounded-generation validation remains
  authoritative at each stage;
- provider and domain failures propagate without being converted into a
  successful RAG response;
- the orchestration layer does not implement automatic retries that could hide
  stage failures or duplicate provider operations.

`RAGService` itself introduces no new model-provider, database, tool, workflow,
MCP, or observability backend. It composes existing application services and
therefore inherits their existing trust boundaries.

This orchestration does not provide authentication, tenant authorization,
row-level tenant isolation, semantic entailment verification, prompt-injection
elimination, context token budgeting, presentation-layer citation rendering,
automatic retries, workflow execution, or AI observability.

## Current Limitations

The repository now contains concrete Ollama LLM and embedding adapters with
runtime composition capable of non-streaming model invocation and batch
embedding generation through the existing provider-neutral contracts.

The provider-neutral contracts themselves remain free of provider-specific
execution behavior. Ollama-specific mapping, HTTP execution, configuration,
error normalization, and client lifecycle are contained by adapter and
runtime layers.

The current Ollama runtime does not implement:

- streaming model responses;
- durable/distributed conversational-agent continuation state and recovery;
- automatic retries;
- provider authentication;
- model routing;
- public model-generation API endpoints;
- per-user or per-tenant model authorization;
- AI-specific telemetry or model-operation tracing.

The repository has PostgreSQL + pgvector persistence and a concrete
vector-store provider supporting upsert, exact L2 nearest-neighbor query, and
scoped delete operations.

Feature 6 activates provider-neutral retrieval execution paths on top of that
persistence foundation:

- deterministic text chunking preserves exact source offsets and document
  provenance;
- indexing orchestration sends validated chunk text through the embedding
  provider and persists the resulting vectors through the vector-store
  provider;
- semantic retrieval embeds a query, performs vector search, and reconstructs
  validated retrieval evidence;
- retrieval provenance, source metadata, text, vector score, vector rank,
  source reference, and offsets are validated before evidence is returned;
- optional reranking uses the existing provider-neutral reranker contract while
  preserving the original vector score and rank separately from reranker
  output;
- empty retrieval results bypass reranking without invoking a reranker.

Feature 8 composes the retrieval, optional reranking, and grounded-generation
services into one end-to-end RAG execution path. Empty retrieval short-circuits
to an explicit abstention before reranking or LLM generation. Non-empty
retrieval preserves the original evidence while the exact retrieval or
reranked result used for grounding is retained separately.

Retrieved content remains untrusted data. Retrieval evidence must not be
interpreted as trusted model instruction merely because it passed vector
similarity, provenance reconstruction, or reranking.

The current retrieval foundation does not yet implement:

- filesystem or network knowledge-source acquisition;
- PDF, DOCX, HTML, or other rich-document parsing;
- document replacement/reindex lifecycle orchestration;
- a public retrieval API;
- a concrete reranker adapter or reranker runtime;
- presentation-layer citation rendering;
- semantic groundedness or entailment verification;
- per-user or per-tenant retrieval authorization;
- row-level tenant isolation;
- data-retention or deletion-policy orchestration;
- production backup and recovery;
- database high availability;
- repository-defined encryption-at-rest controls;
- approximate-nearest-neighbor indexes;
- adversarial retrieval evaluations;
- durable/distributed conversational agent runtime;
- concrete MCP wire transport, authentication, resources, and prompts;
- workflow runtime.

Bounded caller-supplied text ingestion, retrieval, provider-neutral
reranking, grounded generation, and end-to-end RAG orchestration are active
application surfaces rather than purely anticipatory surfaces.
Filesystem/network source acquisition, richer document parsing, public
retrieval exposure, tenant authorization, durable/distributed agent runtime,
concrete MCP wire transports, workflows, and AI observability remain
partially or wholly anticipatory and require additional executable controls
when introduced.

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
- durable/distributed conversational agent execution;
- tools;
- MCP;
- workflow automation;
- human approval;
- external network access;
- persistent user data;
- observability exports;
- deployment infrastructure.
