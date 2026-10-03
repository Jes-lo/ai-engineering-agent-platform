# Threat Model

## Status

Version: 0.8

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
- rejection of non-empty Ollama `tool_calls` while platform LLM tool calling
  remains unsupported;
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

Presentation-layer citation rendering, automated groundedness/citation-quality
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
- LLM tool-calling semantics or tool execution;
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
- agent runtime;
- MCP integration;
- workflow runtime.

Bounded caller-supplied text ingestion, retrieval, provider-neutral
reranking, grounded generation, and end-to-end RAG orchestration are active
application surfaces rather than purely anticipatory surfaces.
Filesystem/network source acquisition, richer document parsing, public
retrieval exposure, tenant authorization, agents, MCP, workflows, and AI
observability remain partially or wholly anticipatory and require additional
executable controls when introduced.

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
