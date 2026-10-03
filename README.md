# AI Engineering & Agent Platform

A self-hosted AI engineering and agent platform built as a portfolio
project to demonstrate secure, observable, testable, and maintainable AI
platform engineering practices.

## Status

Early development.

The repository currently includes:

- repository governance, security, architecture, and engineering standards;
- a Python/FastAPI application foundation;
- liveness and readiness endpoints;
- provider-neutral contracts for language models, embeddings, rerankers,
  vector stores, and controlled tool execution;
- immutable provider-neutral request and response models;
- a domain-level provider exception hierarchy;
- concrete Ollama adapters implementing the language-model and embedding
  provider contracts;
- non-streaming Ollama chat request and response mapping;
- batch Ollama embedding request and response mapping through `/api/embed`;
- optional embedding-dimension requests with validated response dimensions;
- explicit `truncate=false` embedding requests to avoid silent input
  truncation;
- validated Ollama runtime configuration shared by LLM and embedding
  capabilities;
- shared HTTP-client construction with capability-specific provider runtime
  composition and explicit lifecycle ownership;
- normalized provider errors for timeout, transport, HTTP-status, malformed
  JSON, and malformed provider responses;
- isolated Ollama adapter tests using mocked HTTP transport;
- PostgreSQL 18 + pgvector 0.8.6 persistence through a pinned local
  development container image;
- Alembic-managed PostgreSQL schema migrations;
- separate bootstrap/migration and least-privilege application database
  identities;
- a Psycopg async connection-pool runtime with explicit lifecycle ownership;
- a concrete `PostgreSQLVectorStoreProvider` implementing vector upsert,
  exact L2 nearest-neighbor query, and scoped delete operations;
- provider-neutral higher-is-better vector scores, with PostgreSQL L2
  distance normalized as `1 / (1 + distance)`;
- ordered vector metadata persisted as JSONB;
- reproducible PostgreSQL/pgvector integration validation using an isolated,
  ephemeral Docker Compose project;
- deterministic character-window chunking with exact source offsets and
  preserved document provenance;
- bounded caller-supplied knowledge-source ingestion for UTF-8 `text/plain`
  and `text/markdown` payloads;
- explicit source-size, media-type, UTF-8, NUL, and chunk-provenance
  validation before indexing;
- no filesystem or network source acquisition in the current ingestion
  foundation;
- provider-neutral indexing orchestration from validated chunks through the
  embedding provider into the vector-store provider;
- provider-neutral semantic retrieval orchestration from query embedding
  through vector search into validated evidence;
- strict reconstruction of retrieval provenance, source metadata, text, rank,
  score, and citation-ready source offsets;
- optional provider-neutral reranking orchestration that preserves the
  original vector score and rank separately from reranker score and rank;
- deterministic grounded-context assembly over validated retrieval or reranked
  evidence;
- provider-neutral grounded generation through the existing `LLMProvider`
  contract;
- fail-closed grounded-generation validation for model identity, completion
  reason, citation grammar, and citation membership;
- canonical `C1`, `C2`, ... citation identifiers resolved only to evidence
  already present in the supplied retrieval result;
- explicit `INSUFFICIENT_EVIDENCE` abstention without fabricated provenance;
- provider-neutral end-to-end RAG orchestration through `RAGService`;
- deterministic retrieval -> optional reranking -> grounded-generation stage
  ordering;
- controlled no-evidence abstention without reranker or LLM execution;
- preservation of the original retrieval result and the exact grounding input
  used for generation through `RAGResult`;
- automated architectural dependency checks;
- CI/CD and software supply-chain validation, including a dedicated
  PostgreSQL integration job.

Current model execution is available through provider runtime boundaries.
LLM generation uses Ollama's non-streaming `/api/chat` endpoint, while
embedding generation uses `/api/embed`. The FastAPI application does not yet
expose public model-generation, embedding, or retrieval endpoints.

The provider-neutral LLM contract now supports explicit tool definitions and
inert tool-call proposals. `LLMRequest.tools` exposes unique
`ToolDefinition` values to a provider, while `LLMResponse.tool_calls`
contains immutable `LLMToolCall` proposals and uses
`FinishReason.TOOL_CALLS` when proposals are present. A proposal is
untrusted model output: it is not a `ToolInvocation`, does not carry platform
execution authority, and cannot execute a tool by itself.

## Controlled Tool Execution Foundation

The platform now includes a provider-neutral controlled tool execution
foundation built on the existing `ToolProvider`, `ToolDefinition`,
`ToolInvocation`, and `ToolResult` contracts.

`ToolRegistry` requires globally unique tool identities and explicit policy
coverage for every registered tool. `ToolExecutionService` then enforces:

- an explicit per-execution tool allowlist;
- enabled/disabled platform policy;
- structural approval evidence when policy requires it;
- exact `call_id` plus `tool_name` approval binding;
- strict required/unexpected argument validation;
- portable scalar argument type validation without coercion;
- exact result-to-invocation identity validation;
- provider failure propagation without implicit retries.

The current `ToolApprovalGrant` is structural approval evidence only. It is
not yet an identity-bound, persistent, signed, expiring, or single-use
human-in-the-loop approval system.

The provider adapter still produces inert `LLMToolCall` values and never
executes tools by itself. A separate `ControlledAgentService` may now expose
only registered, enabled, explicitly authorized tools to an LLM and convert
accepted proposals into controlled `ToolInvocation` values. Execution IDs are
created by the platform with a fresh UUID4-backed nonce by default;
`provider_call_id` remains untrusted metadata and never becomes execution
authority.

Before the first provider side effect, every planned invocation is checked
through the side-effect-free `ToolExecutionService.validate()` boundary. A
missing structural approval produces an `APPROVAL_REQUIRED` result instead of
executing the batch. `resume()` accepts only an active continuation previously
issued by the same service instance and consumes that continuation before
provider execution, providing in-memory replay protection without re-running
the LLM decision.

This is deliberately not a full conversational agent loop. Pending
continuations are in-memory and are lost on process restart. The tool batch is
sequential, not transactional, and a later provider failure does not roll back
an earlier side effect. Automatic retry after execution starts is intentionally
absent. Tool-result messages, `MessageRole.TOOL`, tool-result round trips to the
LLM, authenticated human approval lifecycle, MCP, workflows, n8n, shell tools,
filesystem tools, and network tools remain outside this feature.

Persistent vector storage is now complemented by provider-neutral retrieval,
grounded generation, and end-to-end RAG orchestration. Deterministic chunking,
indexing, semantic retrieval, validated evidence reconstruction, optional
reranking, deterministic context assembly, grounded model generation, explicit
abstention, citation-to-evidence resolution, and retrieval-to-generation
composition are implemented and covered by tests.

`GroundedGenerationService` consumes an already validated `RetrievalResponse`
or `RerankedRetrievalResponse`; it does not own retrieval execution. Model
citation markers such as `[[C1]]` are resolved by the platform back to the
retrieved evidence object, so the model does not author source references,
document identifiers, offsets, or other provenance metadata.

`RAGService` composes `RetrievalService`, optional `RerankingService`, and
`GroundedGenerationService` without moving provider ownership into the
orchestration layer. Empty retrieval is handled as a controlled abstention
before reranking or LLM execution. `RAGResult` preserves both the original
retrieval output and the exact retrieval or reranked evidence supplied to
grounded generation.

This establishes citation integrity and provenance binding, not automatic
semantic entailment or factual verification of every generated claim.

The current evaluation foundation adds versioned, provenance-aware RAG
evaluation datasets and deterministic execution through
`RAGEvaluationService`. It measures retrieval, final grounding-input, and
citation precision/recall against explicit expected chunk identifiers, plus
answer-status accuracy. These are structural evidence-alignment metrics; they
do not establish semantic entailment, factual correctness, or model
truthfulness.
Retrieved evidence also remains untrusted data and may contain indirect prompt
injection content.

The repository now implements a bounded caller-supplied text ingestion
foundation. It still does not implement filesystem or network source loaders,
PDF/DOCX/HTML parsing, document replacement/reindex lifecycle orchestration, a
public retrieval API, a concrete reranker adapter, tenant-aware retrieval
authorization, presentation-layer citation rendering, semantic groundedness
evaluation, executable tools, agents, MCP integrations, workflow execution, or
AI observability backends.

## Planned Capabilities

The platform is intended to evolve incrementally toward capabilities
including:

- additional local and remote model adapters;
- streaming and richer model-capability handling;
- additional embedding-provider adapters;
- additional knowledge-source loaders and richer document parsing (for example PDF, DOCX, and HTML);
- public retrieval API exposure;
- concrete local or remote reranker adapters;
- production database hardening, backup/recovery, and availability patterns;
- tenant-aware retrieval authorization and data lifecycle controls;
- presentation-layer citation rendering and automated semantic
  groundedness/citation-quality evaluation;
- durable/full conversational agent execution;
- full conversational agent orchestration that returns validated tool results
  to the model through an explicit tool-result message contract;
- MCP integrations and a project-owned MCP server;
- workflow automation;
- human-in-the-loop approval;
- guardrails;
- richer external evaluation datasets, semantic entailment/factual correctness evaluators, and optional judge-based evaluation;
- AI and application observability;
- OpenTelemetry;
- metrics and dashboards;
- an AI Developer Console.

Capabilities listed here describe project direction and are not considered
implemented until corresponding code, tests, documentation, and validation
are merged.

## Local Development

The project currently targets Python 3.13.15 and uses `uv` for dependency
and environment management.

Synchronize the environment:

    uv sync

Run the API locally:

    uv run uvicorn ai_engineering_agent_platform.app:app --host 127.0.0.1 --port 8000

Current system endpoints:

- `GET /health` - liveness;
- `GET /ready` - readiness.

Run local validation:

    uv run pytest
    uv run ruff format --check src tests
    uv run ruff check src tests
    uv run mypy src tests

### Local Ollama Runtime

Ollama is the first concrete model-runtime integration and currently backs
both LLM generation and embedding execution. Ollama itself and model weights
are not distributed by this repository.

Runtime configuration uses:

- `AI_PLATFORM_OLLAMA_BASE_URL`, defaulting to
  `http://127.0.0.1:11434`;
- `AI_PLATFORM_OLLAMA_REQUEST_TIMEOUT_SECONDS`, defaulting to `120`.

The configured base URL must be an absolute HTTP or HTTPS origin without
embedded credentials, path components, query parameters, or fragments.

The current LLM adapter supports non-streaming chat generation. The
embedding adapter supports batch `/api/embed` requests, optional requested
dimensions, response-shape validation, and explicit `truncate=false`
behavior.

A shared Ollama HTTP-client factory consumes the validated runtime settings.
Capability-specific runtimes own client cleanup, while
`OllamaLLMProvider` and `OllamaEmbeddingProvider` remain independent of
HTTP-client lifecycle management.

For opt-in real-runtime validation procedures, see
[Local Ollama Smoke Tests](docs/operations/local-ollama-smoke-test.md).

### PostgreSQL + pgvector Persistence

PostgreSQL persistence is implemented behind the provider-neutral
`VectorStoreProvider` contract. The concrete adapter uses pgvector for exact
L2 nearest-neighbor search while keeping PostgreSQL-specific behavior outside
the contract layer.

The local persistence foundation includes:

- PostgreSQL 18 with pgvector 0.8.6;
- an image pinned by digest in `compose.yaml`;
- loopback-only host-port publishing by default;
- SCRAM authentication;
- an `ai_platform_admin` bootstrap/migration identity;
- a separate `ai_platform_runtime` application identity;
- least-privilege grants for the runtime role;
- Alembic migrations;
- external database credentials rather than committed secrets;
- an async Psycopg pool owned by runtime composition;
- exact vector search without HNSW or IVFFlat indexes at this stage;
- explicit provider-neutral `space_id` on vector upsert, query, and delete;
- PostgreSQL collection identity scoped by `(namespace, space_id)`;
- fail-closed embedding-response model validation before indexing or retrieval
  reaches vector persistence.

`namespace` remains the logical data partition. `space_id` independently
identifies the vector/embedding space. Indexing and retrieval default that
identity to the configured embedding model, while callers may provide a
stricter identity when they control model revision or artifact provenance.

The repository does not currently verify or pin an embedding-model revision or
digest through `space_id`; providing such a stronger identifier remains the
caller's responsibility.

For local setup, migrations, credentials, operational boundaries, and
integration validation, see
[PostgreSQL + pgvector Operations](docs/operations/postgres-vector-store.md).

## Continuous Integration

GitHub Actions validates pull requests and pushes to `main`.

The current CI baseline includes:

- GitHub Actions workflow linting;
- locked Python environment synchronization;
- formatting;
- linting;
- strict static type checking;
- automated tests with warnings treated as errors;
- package build validation;
- isolated PostgreSQL + pgvector provisioning, migration, and live
  vector-provider integration;
- Git history and working-tree secret scanning;
- dependency vulnerability auditing;
- dependency-license inventory;
- CycloneDX SBOM generation and validation.

Supply-chain reports are generated during CI and retained as workflow
artifacts for a limited period.

## Architecture

Architecture decisions and supporting documentation are maintained under:

- `docs/adr/`
- `docs/architecture/`
- `docs/security/`
- `docs/ai-safety/`
- `docs/operations/`

## Engineering Standards

Development and feature completion standards are documented in:

- [Development Standards](docs/operations/development-standards.md)
- [Definition of Done](docs/operations/definition-of-done.md)
- [AI Safety Principles](docs/ai-safety/ai-safety-principles.md)

## Security

Security is treated as a first-class design requirement.

See:

- [SECURITY.md](SECURITY.md)
- [Threat Model](docs/security/threat-model.md)

## Third-Party Components

Third-party software, services, models, datasets, specifications, tools,
and trademarks remain subject to their respective licenses and terms.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Repository-specific material is provided for portfolio, demonstration,
and evaluation purposes.

Copyright (c) 2026 Jesus Lopez.

All Rights Reserved.

See [LICENSE](LICENSE).
