# Architecture Overview

## Purpose

AI Engineering & Agent Platform is a self-hosted-oriented platform for
building, running, evaluating, securing, and observing AI-enabled
systems.

The project is designed as an AI platform rather than as a single
chatbot or application.

Its long-term purpose is to provide reusable platform capabilities that
can be consumed by independent applications, including the future
AI-Native Web & Mobile Application Platform.

## Architectural Goals

The architecture prioritizes:

- modularity;
- explicit contracts;
- provider replaceability;
- secure defaults;
- least privilege;
- bounded agent execution;
- testability;
- observable behavior;
- reproducible environments;
- incremental delivery;
- independent demonstration;
- responsible handling of third-party components and data.

## Current Implementation

The current repository implements foundational platform layers, concrete LLM
and embedding runtime integrations through Ollama, PostgreSQL + pgvector
persistence, bounded caller-supplied text ingestion, provider-neutral indexing,
semantic retrieval, optional reranking, grounded generation, end-to-end RAG
orchestration, controlled tool execution, bounded agent-turn orchestration, and
a bounded conversational agent loop with typed tool-result history.

Implemented capabilities currently include:

- a Python/FastAPI application runtime;
- liveness and readiness API endpoints;
- a provider-neutral base `Provider` contract;
- provider contracts for language models, embeddings, rerankers, vector
  stores, and tools;
- immutable provider-neutral request, response, and value objects;
- a domain-level provider exception hierarchy;
- a concrete `OllamaLLMProvider` adapter;
- a concrete `OllamaEmbeddingProvider` adapter;
- provider-specific mapping between platform contracts and Ollama chat and
  embedding payloads;
- non-streaming model generation through Ollama `/api/chat`;
- batch embedding generation through Ollama `/api/embed`;
- optional requested embedding dimensions with response-dimension validation;
- explicit embedding `truncate=false` requests to reject oversized inputs
  rather than silently truncating them;
- normalized finish reasons and token-usage metadata;
- explicit requested-tool validation for Ollama tool-call proposals;
- validated Ollama base-URL and timeout configuration;
- shared Ollama HTTP-client construction plus capability-specific runtime
  composition, provider construction, and deterministic cleanup;
- normalized timeout, transport, HTTP-status, JSON, and provider-response
  failures;
- mocked-transport adapter tests that do not require a running model server;
- an opt-in manual smoke-test procedure for a real local Ollama runtime;
- PostgreSQL configuration with an explicit asynchronous pool lifecycle;
- separate bootstrap/migration and least-privilege runtime database roles;
- an Alembic-managed `ai_platform` schema;
- pgvector-backed vector persistence;
- a concrete `PostgreSQLVectorStoreProvider` for upsert, exact L2 query, and
  scoped delete behavior;
- ordered JSONB metadata storage;
- deterministic text chunking with exact source offsets and provenance;
- a bounded `KnowledgeSource` ingestion boundary for caller-supplied bytes;
- explicit `text/plain` and `text/markdown` source allowlisting;
- strict UTF-8 decoding, source-size bounds, and NUL rejection before
  chunking;
- `KnowledgeIngestionService` orchestration from validated source bytes through
  an injected chunker into the existing `IndexingService`;
- validation that produced chunks remain bound to document identity, source
  reference, title, metadata, offsets, and source text;
- no filesystem or network source acquisition in the current ingestion
  foundation;
- provider-neutral indexing orchestration through `IndexingService`;
- provider-neutral semantic retrieval through `RetrievalService`;
- strict mapping from vector results into validated retrieval evidence;
- optional provider-neutral reranking through `RerankingService`;
- preservation of original vector scores and ranks when reranking;
- deterministic context assembly from `RetrievalResponse` or
  `RerankedRetrievalResponse`;
- provider-neutral grounded generation through `GroundedGenerationService`;
- fail-closed grounded-generation checks for configured model identity and a
  completed `STOP` finish reason;
- canonical citation-marker validation and resolution to already retrieved
  evidence;
- preservation of retrieval provenance outside model-authored output;
- explicit insufficient-evidence abstention;
- provider-neutral end-to-end RAG composition through `RAGService`;
- deterministic retrieval -> optional reranking -> grounded-generation stage
  ordering;
- no-evidence short-circuiting before reranker or LLM execution;
- preservation of original retrieval and effective grounding input through
  `RAGResult`;
- runtime composition that owns the database pool while the adapter remains
  lifecycle-independent;
- isolated live PostgreSQL integration validation with ephemeral credentials
  and Docker state;
- architectural dependency checks protecting the contracts and domain
  layers;
- CI/CD and software supply-chain validation, including PostgreSQL integration.

Provider-neutral contracts remain independent of Ollama-specific
implementation details. HTTP execution is contained by the adapter/runtime
boundary rather than spread throughout application code.

There is currently no public model-generation, embedding, or retrieval API
endpoint, streaming model generation, concrete reranker adapter,
durable/distributed conversational-agent runtime, concrete MCP wire transport,
authenticated MCP connection lifecycle, workflow runtime, or AI observability
backend.

PostgreSQL + pgvector persistence, deterministic chunking, indexing,
semantic retrieval, provider-neutral optional reranking, deterministic context
assembly, grounded generation, and end-to-end RAG orchestration are
implemented. Production concerns such as tenant-aware authorization,
backup/recovery, high availability,
encryption-at-rest policy, retention, and approximate-nearest-neighbor indexing
are not yet implemented by this repository.

## High-Level Architecture

The target architecture is conceptually organized as follows:

    AI Developer Console
            |
            v
        API Layer
            |
    +-------+-------+
    |       |       |
    v       v       v
  Models   RAG    Agents
    |       |       |
    |       |       +----------+
    |       |                  |
    v       v                  v
 Provider  Retrieval       Tool Registry
 Adapters  Pipeline            |
                               v
                         MCP / APIs / Tools
                               |
                               v
                        Approval / Policy

            |
            v

    Evaluation & Guardrails

            |
            v

    Telemetry & Observability

            |
            v

 Infrastructure / Data / Runtime

## Planned Platform Areas

### API Layer

The API layer will provide controlled access to platform capabilities.

Responsibilities are expected to include:

- request validation;
- authentication;
- authorization;
- resource access;
- platform APIs;
- streaming interfaces;
- audit context propagation.

### Model Gateway

The model gateway separates platform behavior from specific model providers.

The current implementation provides:

- a provider-neutral `LLMProvider` contract;
- an Ollama-specific adapter behind that contract;
- validated model-runtime configuration;
- non-streaming chat generation;
- generation controls for temperature and maximum output tokens;
- finish-reason normalization;
- token-usage normalization when supplied by the provider;
- normalized provider availability and execution errors;
- HTTP-client lifecycle composition outside the provider implementation.

The current dependency flow is:

    LLMRequest
        |
        v
    runtime composition
        |
        v
    OllamaLLMProvider
        |
        v
    Ollama request mapping
        |
        v
    POST /api/chat
        |
        v
    Ollama response mapping
        |
        v
    LLMResponse

Provider-specific behavior remains behind adapter and runtime boundaries.
Core contracts and domain models do not import Ollama or HTTPX2.

Remaining model-gateway capabilities include:

- streaming;
- additional model providers;
- richer model capability metadata;
- explicit retry policy;
- model routing;
- telemetry and AI observability;
- public authenticated and authorized generation APIs.

### Embeddings

Embedding generation is implemented as an independent provider capability.

The current implementation provides:

- a provider-neutral `EmbeddingProvider` contract;
- immutable `EmbeddingRequest`, `EmbeddingVector`, and `EmbeddingResponse`
  models;
- an Ollama-specific `OllamaEmbeddingProvider`;
- batch request mapping to Ollama `/api/embed`;
- optional requested dimensionality;
- strict validation of vector count, dimensions, numeric values, finite
  values, model identity, and token accounting;
- explicit `truncate=false` requests to prevent silent truncation;
- normalized timeout, transport, HTTP-status, malformed-JSON, and malformed
  response failures;
- shared Ollama HTTP-client construction with an embedding-specific runtime
  responsible for lifecycle cleanup;
- mocked transport tests that do not require a running Ollama instance;
- opt-in real-runtime validation against a separately installed local
  embedding model.

The current dependency flow is:

    EmbeddingRequest
        |
        v
    runtime composition
        |
        v
    OllamaEmbeddingProvider
        |
        v
    Ollama embedding request mapping
        |
        v
    POST /api/embed
        |
        v
    Ollama embedding response mapping
        |
        v
    EmbeddingResponse

Embedding implementations remain replaceable behind the provider contract.

Generated embeddings can be persisted through provider-neutral indexing
orchestration. `IndexingService` maps validated document chunks to an
`EmbeddingProvider`, verifies the embedding response, and then maps the
resulting vectors into `VectorStoreProvider` upserts.

`RetrievalService` performs the complementary semantic-query path: it embeds
one retrieval query, executes a provider-neutral vector query, and converts
the returned records into validated retrieval evidence.

Retrieval evidence preserves the original vector score and rank together with
chunk identity, document identity, source reference, source metadata, text,
title, and exact source offsets. Persisted retrieval provenance is validated
strictly rather than guessed or silently coerced.

`RerankingService` can optionally pass retrieved evidence through the existing
provider-neutral `RerankerProvider`. Reranking preserves the original vector
score and rank separately from the reranker score and final rank. No concrete
reranker adapter or reranker runtime is implemented yet.

The current RAG foundation now includes bounded ingestion of
caller-supplied UTF-8 `text/plain` and `text/markdown` payloads. Source
references remain opaque provenance; the ingestion layer does not read
filesystem paths or fetch network resources. Filesystem/network source
loaders, PDF/DOCX/HTML parsing, document replacement/reindex lifecycle
orchestration, a public retrieval API, tenant-aware retrieval authorization,
presentation-layer citation rendering, and semantic groundedness/entailment
verification remain outside the current implementation.

### Knowledge and RAG

The broader RAG subsystem evolves through the following target flow:

    source
      |
      v
    ingestion
      |
      v
    validation
      |
      v
    parsing
      |
      v
    chunking
      |
      v
    metadata enrichment
      |
      v
    embeddings
      |
      v
    indexing
      |
      v
    retrieval
      |
      v
    reranking
      |
      v
    context assembly
      |
      v
    model generation
      |
      v
    grounded answer + citations

Features 6 through 9 currently cover bounded caller-supplied text
ingestion, deterministic chunking, embeddings and indexing orchestration,
semantic retrieval, evidence reconstruction, provider-neutral optional
reranking, deterministic context assembly, grounded model generation, explicit
abstention, citation-marker resolution back to validated evidence, and
end-to-end retrieval-to-generation composition.

`RAGService` invokes `RetrievalService`, optionally applies
`RerankingService`, and supplies the resulting validated evidence to
`GroundedGenerationService`. Empty retrieval becomes an explicit abstention
without invoking a reranker or LLM. Successful executions preserve the
original retrieval result separately from the exact grounding input so vector
ranking diagnostics are not lost when reranking is enabled.

`GroundedGenerationService` intentionally accepts an already validated
`RetrievalResponse` or `RerankedRetrievalResponse` instead of invoking
retrieval itself. The service constructs a provider-neutral `LLMRequest`,
requires the configured model identity and a completed `STOP` response, then
resolves canonical markers such as `[[C1]]` only against the evidence presented
to the model.

The model does not supply authoritative source references, document
identifiers, source offsets, or retrieval metadata. Those values remain owned
by the validated evidence object retained by the platform.

These controls establish citation integrity and provenance binding. They do not
prove that every generated sentence is semantically entailed by its cited
evidence, that the evidence is true, or that indirect prompt injection has been
eliminated.

Filesystem/network source acquisition, richer document parsing,
document replacement/reindex lifecycle orchestration, presentation-layer
citation rendering, and automated semantic groundedness/citation-quality
evaluation remain future stages.

Retrieval and answer-grounding quality is now measurable through a
deterministic structural evaluation foundation rather than judged only
manually. The current metrics compare retrieved, grounding, and cited chunk
identities against explicit versioned dataset expectations. They do not infer
semantic entailment or factual correctness.

### Controlled Agent Turn Orchestration Boundary

The platform now includes a bounded single-turn orchestration layer through
`ControlledAgentService`.

The orchestration boundary:

- accepts a caller correlation `run_id` but does not treat it as execution
  authority;
- owns the model-visible tool set instead of accepting pre-populated
  `LLMRequest.tools` from the caller;
- resolves only tools that are registered, enabled by platform policy, and
  explicitly present in `ToolExecutionAuthorization`;
- accepts only `STOP` or `TOOL_CALLS` as valid turn outcomes;
- applies a caller-selected `max_steps` with a hard platform ceiling of
  `MAX_AGENT_TURN_STEPS = 8`;
- rechecks every proposed tool against the exact exposure set;
- creates a new platform-owned `ToolInvocation.call_id` using a fresh
  UUID4-backed nonce by default;
- never treats provider-originating `provider_call_id` metadata as execution
  identity or authority;
- preflights every planned invocation through
  `ToolExecutionService.validate()` before the first provider side effect;
- returns an `APPROVAL_REQUIRED` state when exact structural approval is
  missing;
- keeps only service-issued pending continuations eligible for `resume()`;
- can resume an approved frozen plan without asking the model to regenerate
  the decision;
- consumes the pending continuation before provider execution so a completed
  continuation cannot be replayed through the same service instance.

The current continuation registry is process-local and in-memory. It is not
durable across restarts, distributed across replicas, or an authenticated
human-in-the-loop approval system.

A planned batch executes sequentially after preflight. It is not
transactional. If one provider execution succeeds and a later execution fails,
the platform does not claim rollback or atomicity, and it deliberately does not
automatically replay the batch after execution has started.

`ControlledAgentService` deliberately remains a bounded single-turn execution
boundary. Feature 14 does not expand that service's execution authority;
`BoundedAgentLoopService` composes repeated controlled turns around it and
maintains typed tool-call/tool-result conversation history. Each later model
proposal therefore crosses the same Feature 13 execution controls again.

### Bounded Conversational Agent Loop Boundary

`BoundedAgentLoopService` provides the current multi-turn composition layer.

Its provider-neutral transcript uses:

- ordinary `LLMMessage` values for system, user, and plain assistant messages;
- `LLMAssistantToolCallMessage` for the model decision that proposed tools;
- `LLMToolResultMessage` for validated controlled tool results;
- explicit `MessageRole.TOOL` semantics for tool-result context.

`LLMRequest` validates the transcript before generation. Tool results cannot be
orphaned, cannot precede their assistant proposal, must match the proposed tool
name, and must preserve matching optional `provider_call_id` metadata.

Platform `ToolResult.call_id` remains available inside platform state for
execution/audit correlation but is intentionally not serialized into the
Ollama tool-result wire message. Provider call IDs remain transcript metadata,
not execution authorization.

The loop applies two independent caller-selected budgets under hard platform
ceilings:

- `MAX_AGENT_LOOP_MODEL_TURNS = 8`;
- `MAX_AGENT_LOOP_TOOL_CALLS = 8`.

Tool-call consumption is cumulative across model turns. When the remaining
tool budget reaches zero, tools are no longer exposed on the next controlled
turn. An oversized model-proposed batch fails before provider execution.

Every executable proposal still flows through `ControlledAgentService`; the
loop has no direct `ToolExecutionService` execution path. Approval-required
plans pause before execution, and resume reuses the exact previously issued
decision without asking the model to regenerate it.

Loop continuation state remains process-local and in-memory. The feature does
not claim durable recovery, distributed coordination, authenticated approver
identity, transactional tool batches, rollback, automatic execution retries,
MCP, workflows, n8n, or shell/filesystem/network tools.

### LLM Tool-Call Proposal Boundary

`LLMRequest` may now include a tuple of provider-neutral `ToolDefinition`
values. Tool names must be unique inside one request.

For Ollama, those definitions are mapped to `/api/chat` function-tool
schemas. A completed response containing `message.tool_calls` is treated as
untrusted provider output and is normalized only when every proposed tool
name belongs to the tool set explicitly exposed by that request.

Normalized proposals use `LLMToolCall`. They contain the proposed tool name,
portable scalar `ToolArgument` values, and an optional provider-originating
`provider_call_id`. That provider identifier is metadata only; it is not the
platform-controlled `ToolInvocation.call_id`.

Responses with one or more proposals use `FinishReason.TOOL_CALLS`. A
`TOOL_CALLS` response without proposals, or a response containing proposals
under another finish reason, is invalid at the provider-neutral contract
boundary.

The provider proposal layer itself deliberately performs no execution and
continues to return untrusted `LLMToolCall` values. Execution identity,
authorization, policy, approval, and provider execution remain outside the
LLM adapter. The separate `ControlledAgentService` is the only current bridge
from accepted proposals to controlled `ToolInvocation` values, and that bridge
applies explicit exposure, authorization, bounded-step, fresh-identity, and
preflight controls before execution.

### Tool Registry

Tools are explicit platform resources behind provider-neutral contracts.

The current controlled execution foundation registers `ToolProvider`
definitions through `ToolRegistry`. Registration fails closed unless:

- provider identities are unique;
- tool names are globally unique;
- every registered tool has one explicit `ToolExecutionPolicy`;
- policy entries do not refer to unknown tools.

Each current policy records whether a tool is enabled and whether execution
requires approval. Every execution also receives a
`ToolExecutionAuthorization` containing an explicit tool allowlist and
optional `ToolApprovalGrant` values.

Before a provider is invoked, `ToolExecutionService` validates authorization,
required approval, declared arguments, and portable scalar argument types.
Arguments are not coerced. After provider execution, result `call_id` and
`tool_name` identities must match the exact invocation.

The current approval grant is deliberately structural: it binds exact
`call_id` and `tool_name` values but does not yet represent authenticated
human identity, persistence, signatures, expiry, revocation, or single-use
consumption.

The controlled execution service itself remains model-agnostic.
`ToolExecutionService.validate()` provides the deterministic registration,
enabled-policy, allowlist, approval, and argument checks used by `execute()`
without calling the provider. `ControlledAgentService` composes that boundary
with untrusted `LLMToolCall` proposals, and `BoundedAgentLoopService` composes
multiple controlled turns plus validated tool-result history without granting
the model new execution authority. Durable/distributed agent state,
authenticated HITL approval, concrete MCP wire transports and authenticated
MCP connection lifecycle, workflow execution, n8n integration, shell tools,
filesystem tools, and network tools remain future capabilities.

### MCP

The project now has a transport-neutral MCP interoperability core in both
directions, without yet claiming wire-protocol compatibility.

For remote MCP consumption, `MCPClient` is the transport adapter contract.
`MCPTrustPolicy` contains explicit `MCPToolBinding` values that map a configured
remote tool name to a platform-owned local `ToolDefinition`. Discovery is
treated as untrusted metadata and does not itself create authorization.

`MCPToolProviderAdapter.discover()` fails closed when:

- the configured client `server_name` differs from the trust-policy label;
- a policy-bound remote tool is absent;
- discovery contains duplicate remote tool names;
- a discovered parameter schema differs from the pinned platform schema.

Unbound remotely discovered tools are ignored rather than automatically
registered. Remote descriptions are not adopted into the platform-owned tool
definition. The `server_name` comparison is configuration binding only; it is
not authenticated or cryptographic server identity.

After discovery, the adapter implements the normal `ToolProvider` contract.
It deliberately makes no authorization decision. Correct platform composition
registers it with `ToolRegistry` and executes it through the existing
`ToolExecutionService` boundary. The provider method remains an internal
adapter surface just like other `ToolProvider` implementations and must not be
treated as authorization authority.

For project-owned MCP exposure, `OwnedMCPToolService` is a transport-neutral
service core. Its exposure list is explicit. The explicit exposure set is
further filtered by enabled platform policy, the supplied execution
authorization, and the current prohibition on approval-required tools.

Inbound `MCPToolCallRequest.request_id` is external correlation data. The
service creates a separate internal `ToolInvocation.call_id`, rejects an
identity collision, delegates execution to `ToolExecutionService`, and returns
only the external request correlation plus normalized tool result fields.
Internal execution identity is not part of `MCPToolCallResponse`.

The current MCP core has no JSON-RPC codec, stdio process transport, Streamable
HTTP transport, authentication, connection/session lifecycle, resources,
prompts, retry loop, shell/filesystem/network tools, or transport-level
timeout/response-size/rate controls. Those belong to later adapter/runtime
features.

A future authenticated MCP transport must convert authenticated principal and
platform policy into `ToolExecutionAuthorization`; execution allowlists and
approval evidence must never be trusted merely because a remote client supplied
them. Approval-required tool export also remains blocked until the platform has
an authenticated and durable HITL lifecycle.

### Workflows and Automation

Workflows may combine deterministic application logic with AI-driven
steps.

Deterministic automation should remain deterministic where an LLM is
not needed.

Human approval should be introduced for actions where the impact
justifies confirmation.

### Deterministic RAG Evaluation Foundation

The platform now provides infrastructure-independent contracts for versioned
RAG evaluation datasets with explicit dataset identity, version, and
provenance reference.

`RAGEvaluationService` executes cases sequentially through an injected
`RAGRunner` and preserves each `RAGResult` alongside transparent metrics.

Current deterministic metrics are:

- retrieval precision and retrieval recall against expected relevant chunk identifiers;
- grounding precision and grounding recall over the final grounding input after optional reranking;
- citation precision and citation recall over the evidence actually cited;
- answer-status accuracy for expected answered versus abstained outcomes;
- macro-averaged dataset summaries with each metric kept independent.

Evaluation validates query/namespace alignment and verifies that grounding
evidence remains drawn from retrieval and citations remain bound to the exact
grounding evidence objects.

This foundation intentionally does not claim semantic entailment, factual
correctness, hallucination elimination, LLM-as-a-judge scoring, or external
dataset trust.

### Evaluation

Evaluation is a first-class platform capability.

Current and planned evaluation areas include:

- retrieval relevance;
- answer grounding;
- citation quality;
- tool selection;
- tool execution;
- agent completion;
- policy compliance;
- regression testing;
- latency;
- resource usage.

### Guardrails

Guardrails may operate before, during, and after AI execution.

They are expected to include a combination of:

- validation;
- authorization;
- policy checks;
- content constraints;
- execution limits;
- tool restrictions;
- human approval.

Guardrails must not rely exclusively on asking the same model to judge
its own behavior.

### Observability

Operational and AI observability will be related but distinct.

Operational observability will focus on:

- availability;
- latency;
- throughput;
- failures;
- saturation;
- infrastructure health.

AI observability will eventually include:

- model operations;
- retrieval operations;
- agent runs;
- tool calls;
- evaluation results;
- token or resource usage where available;
- quality signals.

Sensitive prompts, retrieved content, credentials, or personal
information must not be indiscriminately exported to telemetry systems.

## Data Architecture

PostgreSQL is the current persistence technology for the implemented vector
foundation. pgvector provides the vector column type and exact distance
operations while a provider-neutral application boundary keeps PostgreSQL
details out of the vector-store contract.

The implemented persistence schema currently contains:

- vector collections keyed by optional logical namespace plus explicit
  provider-neutral `space_id`;
- uniqueness over `(namespace, space_id)`, including `NULL` namespaces;
- one enforced embedding dimensionality per vector space;
- vector records scoped to a collection;
- arbitrary non-empty text record identifiers;
- pgvector embeddings;
- optional record text;
- ordered metadata represented as a JSONB array.

`namespace` and `space_id` intentionally represent different concerns.
`namespace` partitions application data; `space_id` identifies the vector
space in which embeddings are comparable. The indexing and retrieval services
default `space_id` to the configured embedding model and reject an embedding
provider response whose reported model differs from the requested model before
vector-store operations occur.

Migration `0002_embedding_space_isolation` assigns pre-existing collections to
the explicit `legacy-unidentified` space rather than guessing which embedding
model produced them. Moving legacy vectors into a known model space therefore
requires explicit reindexing or separately proven provenance.

A caller may use a stronger `space_id` containing revision or artifact
identity, but the current implementation does not itself resolve, verify, or
pin model revisions or digests.

The implementation deliberately does not yet contain document, chunk,
knowledge-source, agent, workflow, evaluation, or audit schemas.

Future persistent data categories may include:

- platform configuration;
- model metadata;
- knowledge-base metadata;
- documents;
- chunks;
- agent definitions;
- workflow definitions;
- tool registrations;
- evaluation datasets;
- evaluation results;
- audit metadata.

Vector retrieval storage remains replaceable behind the
`VectorStoreProvider` boundary.

## Deployment Evolution

The project will evolve incrementally.

Early development should favor a reproducible local environment.

The repository already includes a CI/CD and software supply-chain
validation baseline together with a containerized PostgreSQL development and
integration environment.

Later stages may introduce:

- Kubernetes;
- infrastructure as code;
- centralized observability;
- hardened runtime configurations.

Infrastructure should be introduced when it provides demonstrable value,
not solely to increase the number of technologies in the repository.

## Relationship to Other Portfolio Projects

This repository is independently usable and demonstrable.

It may reuse engineering knowledge from earlier portfolio projects but
must not depend on their repositories to function.

The future AI-Native Web & Mobile Application Platform may consume this
platform through versioned external interfaces.

Direct coupling to this repository's internal modules should not be
required by consuming products.

## Architecture Evolution

This document describes the architectural direction and not an assertion
that every described capability is currently implemented.

Implemented capabilities must always be distinguishable from planned
capabilities.

Significant architectural decisions must be recorded through ADRs.
