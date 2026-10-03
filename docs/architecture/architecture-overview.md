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
and embedding runtime integrations through Ollama, and a PostgreSQL + pgvector
persistence foundation behind the vector-store contract.

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
- explicit rejection of unsupported Ollama tool calls;
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

There is currently no public model-generation or embedding API endpoint,
streaming model generation, complete retrieval/RAG pipeline, agent runtime,
executable tool integration, MCP integration, workflow runtime, or AI
observability backend.

PostgreSQL + pgvector persistence and exact vector-store operations are
implemented, but production concerns such as tenant-aware authorization,
backup/recovery, high availability, encryption-at-rest policy, retention,
and approximate-nearest-neighbor indexing are not yet implemented by this
repository.

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

Generated embedding vectors can now be persisted explicitly through the
separate provider-neutral vector-store contract and PostgreSQL/pgvector
adapter. Embedding generation does not automatically persist data, and no
end-to-end ingestion, indexing pipeline, retrieval pipeline, reranking, or
RAG orchestration has been introduced yet.

### Knowledge and RAG

The RAG subsystem is expected to evolve through:

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

Retrieval quality must eventually be measurable through evaluations
rather than judged only manually.

### Agent Runtime

The agent runtime will coordinate controlled AI-driven execution.

Agent behavior may include:

- reasoning over context;
- selecting approved tools;
- invoking workflows;
- maintaining bounded execution state;
- requesting human approval;
- returning structured execution results.

Agents must not receive unrestricted platform permissions by default.

### Tool Registry

Tools will be explicit platform resources.

Each tool should eventually expose metadata such as:

- identifier;
- description;
- input contract;
- output contract;
- required permissions;
- timeout;
- risk classification;
- approval requirements.

Tool metadata must not automatically be considered trusted merely
because it was supplied by an external integration.

### MCP

The project is expected to both consume MCP capabilities and implement a
project-owned MCP service.

MCP-related behavior must remain behind explicit trust and authorization
boundaries.

External MCP servers must not automatically inherit access to platform
secrets, internal networks, user data, or privileged tools.

### Workflows and Automation

Workflows may combine deterministic application logic with AI-driven
steps.

Deterministic automation should remain deterministic where an LLM is
not needed.

Human approval should be introduced for actions where the impact
justifies confirmation.

### Evaluation

Evaluation will be a first-class platform capability.

Planned evaluation areas include:

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

- vector collections keyed by optional namespace;
- one enforced embedding dimensionality per collection;
- vector records scoped to a collection;
- arbitrary non-empty text record identifiers;
- pgvector embeddings;
- optional record text;
- ordered metadata represented as a JSONB array.

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
