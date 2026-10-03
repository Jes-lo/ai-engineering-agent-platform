# PostgreSQL + pgvector Operations

## Purpose

This document describes the current local-development and CI operating model
for the PostgreSQL + pgvector persistence foundation.

It covers the implemented Feature 5 foundation only. It is not a production
database architecture or a complete RAG operating procedure.

## Implemented Scope

The repository currently provides:

- PostgreSQL 18 for local and integration-test persistence;
- pgvector 0.8.6;
- a container image pinned by digest in `compose.yaml`;
- SCRAM-SHA-256 PostgreSQL authentication;
- separate bootstrap/migration and runtime identities;
- Alembic migrations;
- an async Psycopg runtime pool;
- `PostgreSQLVectorStoreProvider`;
- exact L2 vector queries;
- vector upsert and scoped delete operations;
- isolated live integration validation.

The repository does not yet provide:

- a production PostgreSQL deployment;
- managed secret storage;
- backup or point-in-time recovery configuration;
- high availability;
- replica topology;
- repository-defined encryption-at-rest policy;
- tenant row-level security;
- retention automation;
- HNSW or IVFFlat vector indexes;
- a complete retrieval or RAG pipeline.

## Trust and Identity Model

Two database identities have different responsibilities.

### Bootstrap / Migration Identity

`ai_platform_admin` is used for:

- database bootstrap;
- role provisioning;
- extension creation;
- schema migration;
- schema ownership.

Application runtime code must not use this identity.

### Application Runtime Identity

`ai_platform_runtime` is the application database identity.

The provisioning script configures it without:

- superuser;
- database creation;
- role creation;
- inheritance;
- replication;
- row-level-security bypass.

The runtime role receives only the database, schema, table, and sequence
privileges required by the current vector-store implementation.

## Secret Locations

Local development credentials are intentionally stored outside the repository.

Default bootstrap file:

    ~/.config/ai-engineering-agent-platform/postgres.env

Default runtime file:

    ~/.config/ai-engineering-agent-platform/runtime.env

The containing directory should use mode `700` and secret files mode `600`.

No real password belongs in `.env.example`, source files, documentation,
commits, CI workflow YAML, or test fixtures.

The repository `.env.example` contains placeholders only.

## Bootstrap File Shape

The bootstrap file contains:

    POSTGRES_USER=ai_platform_admin
    POSTGRES_DB=ai_platform
    POSTGRES_PASSWORD=<external-secret>

Do not copy a real value into source control.

## Runtime Configuration

Application runtime settings use:

    AI_PLATFORM_POSTGRES_HOST
    AI_PLATFORM_POSTGRES_PORT
    AI_PLATFORM_POSTGRES_DATABASE
    AI_PLATFORM_POSTGRES_USER
    AI_PLATFORM_POSTGRES_PASSWORD
    AI_PLATFORM_POSTGRES_SSLMODE
    AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS
    AI_PLATFORM_POSTGRES_POOL_MIN_SIZE
    AI_PLATFORM_POSTGRES_POOL_MAX_SIZE
    AI_PLATFORM_POSTGRES_POOL_TIMEOUT_SECONDS

`AI_PLATFORM_POSTGRES_USER` is expected to resolve to
`ai_platform_runtime`.

Runtime configuration must be injected into the process environment by the
deployment or local execution mechanism. The external `runtime.env` file is a
development convenience and is not automatically a production secret store.

## Start Local PostgreSQL

With the external bootstrap file present:

    docker compose up -d postgres

The normal host binding is loopback-only on port `5432`.

An alternate host port may be selected with:

    AI_PLATFORM_POSTGRES_HOST_PORT=<port> docker compose up -d postgres

## Provision Runtime Role

Provision the restricted runtime identity with:

    uv run python scripts/postgres/provision_runtime_role.py

The provisioner is designed to be repeatable. It creates the role when absent
and normalizes the managed role attributes and password when it already
exists.

## Apply Migrations

Apply the current migration head with:

    uv run python scripts/postgres/migrate.py upgrade head

Inspect the current revision with:

    uv run python scripts/postgres/migrate.py current

The initial migration creates:

- the pgvector extension;
- schema `ai_platform`;
- `vector_collections`;
- `vector_records`;
- runtime grants.

Role creation is intentionally not owned by Alembic because PostgreSQL roles
are cluster-level objects with separate credential lifecycle concerns.

## Downgrade

For an empty development database, migration reversibility can be exercised
with:

    uv run python scripts/postgres/migrate.py downgrade base

The downgrade removes Feature 5 schema objects and the vector extension.

The independently provisioned runtime role remains because it is not owned by
the migration lifecycle.

Do not use destructive migration commands against data that must be retained.

## Vector Storage Semantics

Collections are scoped by optional namespace.

The schema enforces:

- one collection per namespace, including one `NULL` default namespace;
- dimensions between 1 and 16000;
- one embedding dimensionality per collection;
- non-empty record identifiers;
- generated dimensions matching the stored pgvector value;
- ordered metadata represented as a JSONB array.

The same `record_id` may exist in different namespaces.

## Query Semantics

The PostgreSQL adapter currently performs exact L2 nearest-neighbor search
using pgvector's `<->` operator.

The provider-neutral contract exposes higher scores as better matches.

The PostgreSQL adapter maps L2 distance to:

    score = 1 / (1 + distance)

Therefore:

- exact match => `1.0`;
- increasing L2 distance => decreasing positive score.

No approximate-nearest-neighbor index is installed by Feature 5.

## Integration Validation

Run the isolated PostgreSQL integration suite with:

    ./scripts/ci/validate-postgres-integration.sh

The integration runner:

- allocates an isolated Docker Compose project;
- defaults to host port `55432`;
- generates synthetic credentials outside the repository;
- creates an ephemeral database volume;
- provisions the runtime role;
- applies Alembic migrations;
- executes the live vector-provider integration test;
- verifies final database invariants;
- removes the integration container and volume.

It must not mutate the normal development database.

## CI

GitHub Actions runs the same integration runner in the dedicated:

    PostgreSQL Integration

job.

The job does not require stored PostgreSQL repository secrets because its
database credentials are generated ephemerally for that CI run.

## Operational Boundaries

The local environment proves application behavior, migration correctness,
least-privilege database access, and integration reproducibility.

Before any production use, deployment-specific controls should address at
least:

- managed secrets;
- TLS policy appropriate to the deployment boundary;
- encryption at rest;
- backup and restore;
- recovery objectives;
- high availability;
- monitoring and alerting;
- capacity planning;
- connection limits;
- patching;
- retention;
- tenant isolation and authorization;
- audit requirements.
