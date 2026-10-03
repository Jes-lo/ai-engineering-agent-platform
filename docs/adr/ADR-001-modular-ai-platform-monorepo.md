# ADR-001: Modular AI Platform Monorepo

- Status: Accepted
- Date: 2026-10-02
- Decision owners: Repository owner
- Scope: Platform architecture

## Context

The AI Engineering & Agent Platform is intended to demonstrate the
engineering of a reusable AI platform rather than a single-purpose AI
application.

The platform is expected to evolve toward capabilities including:

- model-provider abstraction;
- embeddings;
- retrieval-augmented generation;
- retrieval and reranking;
- grounded responses and citations;
- agent execution;
- tool calling;
- MCP integrations;
- workflow automation;
- human approval;
- evaluations;
- guardrails;
- application and AI observability;
- an AI Developer Console.

The architecture must support incremental development while avoiding
unnecessary coupling to individual AI vendors, model providers,
databases, workflow engines, observability products, or external tools.

The repository must also remain understandable and demonstrable as an
independent portfolio project.

## Decision

The platform will be developed as a modular monorepo.

Repository components will be separated by clear architectural
boundaries while remaining in a single source-control repository during
the initial development stages.

Logical boundaries will be created for:

- API services;
- background workers;
- MCP services;
- AI provider adapters;
- embedding providers;
- retrieval and reranking;
- agent execution;
- tools;
- workflows;
- evaluations;
- guardrails;
- security;
- telemetry;
- infrastructure;
- the AI Developer Console.

Shared behavior must be exposed through explicit internal contracts
rather than direct dependency on provider-specific implementations.

## Provider Abstraction

Core platform capabilities must not depend directly on a specific
external model provider.

Replaceable interfaces will be used where practical for:

- language models;
- embedding models;
- rerankers;
- vector stores;
- tool providers;
- workflow execution;
- evaluation providers;
- telemetry backends.

Provider-specific behavior must remain behind adapters.

For example, application logic should depend conceptually on a model
provider interface rather than directly invoking one particular model
runtime throughout the codebase.

## Service Boundaries

A component will become an independently deployable service only when
there is a demonstrated operational or architectural reason.

Possible reasons include:

- independent scaling;
- separate security boundary;
- different runtime lifecycle;
- isolation of privileged behavior;
- asynchronous execution;
- reliability requirements;
- independent deployment requirements.

The project will not introduce distributed services solely to simulate
architectural complexity.

## Dependency Direction

Core domain behavior should not depend on infrastructure-specific
implementations.

The intended dependency direction is:

    interfaces / contracts
            |
            v
      domain services
            |
            v
        adapters
            |
            v
 external systems / infrastructure

External providers should be replaceable without rewriting unrelated
domain logic.

## Security

Privileged operations must have explicit authorization boundaries.

Agent execution and tool invocation must not implicitly inherit
unrestricted platform access.

Security controls will evolve alongside the capabilities that create
the corresponding risks.

Sensitive operations may require:

- explicit permission;
- scoped credentials;
- policy evaluation;
- execution limits;
- human approval;
- audit records.

## Observability

Observability is a platform capability rather than an afterthought.

Core operations should eventually support trace, metric, and structured
event correlation across:

- API requests;
- retrieval operations;
- model calls;
- agent runs;
- tool calls;
- workflows;
- evaluations.

Provider-specific telemetry must not become the only source of platform
observability.

## Data

Persistent application and AI metadata will use clearly defined data
ownership boundaries.

Vector-search implementation details must not leak unnecessarily into
unrelated application components.

Knowledge sources, evaluation datasets, model metadata, and operational
telemetry must remain logically distinguishable.

## Consequences

### Positive

- clear separation of concerns;
- easier testing;
- replaceable AI providers;
- reduced vendor coupling;
- incremental implementation;
- simpler local development than an early microservice architecture;
- easier portfolio demonstration;
- clearer security boundaries;
- independent evolution of platform capabilities.

### Negative

- additional interface design work;
- adapters may initially appear more complex than direct provider calls;
- monorepo boundaries require discipline;
- some abstractions may need revision as real implementations expose
  additional requirements.

## Alternatives Considered

### Provider-specific application

Directly integrating one model runtime throughout the application would
be simpler initially but would create unnecessary coupling and would
demonstrate less platform-engineering depth.

Rejected.

### Microservices from the beginning

Separating every capability into independently deployed services would
increase operational complexity before scaling and isolation
requirements are understood.

Rejected for the initial architecture.

### Single monolithic application without internal boundaries

This would reduce initial structure but would make RAG, agents, tools,
evaluation, security, and observability increasingly difficult to
evolve independently.

Rejected.

## Revisit Conditions

This decision should be revisited if:

- repository size significantly impairs development;
- independent deployment becomes operationally necessary;
- security boundaries require process isolation;
- specific workloads require independent scaling;
- ownership by separate engineering teams becomes relevant.

A future change must be documented by a new ADR rather than silently
rewriting this decision.
