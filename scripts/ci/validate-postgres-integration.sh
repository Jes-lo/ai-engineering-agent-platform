#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PORT="${AI_PLATFORM_POSTGRES_INTEGRATION_PORT:-55432}"
PROJECT="ai-platform-integration-$$"

CI_HOME="$(
  mktemp \
    -d \
    "${TMPDIR:-/tmp}/ai-platform-postgres-ci.XXXXXX"
)"

CONFIG_DIR="$CI_HOME/.config/ai-engineering-agent-platform"
BOOTSTRAP_ENV="$CONFIG_DIR/postgres.env"
RUNTIME_ENV="$CONFIG_DIR/runtime.env"

cleanup() {
  local exit_code=$?

  set +e

  HOME="$CI_HOME" \
  AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
  docker compose \
    -p "$PROJECT" \
    -f compose.yaml \
    down \
    -v \
    --remove-orphans \
    >/dev/null \
    2>&1

  rm -rf "$CI_HOME"

  exit "$exit_code"
}

trap cleanup EXIT

echo "===== PREREQUISITES ====="

docker version >/dev/null
docker compose version

uv lock --check
uv sync --locked

python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])

with socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM,
) as sock:
    try:
        sock.bind(
            (
                "127.0.0.1",
                port,
            )
        )
    except OSError as exc:
        raise SystemExit(
            f"FAIL: integration port {port} is unavailable: {exc}"
        ) from exc

print(
    f"PASS: integration host port {port} is available"
)
PY

echo "PASS: integration prerequisites"

echo
echo "===== EPHEMERAL EXTERNAL CREDENTIALS ====="

install \
  -d \
  -m 700 \
  "$CONFIG_DIR"

BOOTSTRAP_ENV="$BOOTSTRAP_ENV" \
RUNTIME_ENV="$RUNTIME_ENV" \
PORT="$PORT" \
python3 - <<'PY'
import os
import secrets
from pathlib import Path

bootstrap = Path(
    os.environ["BOOTSTRAP_ENV"]
)

runtime = Path(
    os.environ["RUNTIME_ENV"]
)

port = os.environ["PORT"]

bootstrap_password = secrets.token_hex(32)
runtime_password = secrets.token_hex(32)

bootstrap.write_text(
    (
        "POSTGRES_USER=ai_platform_admin\n"
        "POSTGRES_DB=ai_platform\n"
        f"POSTGRES_PASSWORD={bootstrap_password}\n"
    ),
    encoding="utf-8",
)

runtime.write_text(
    (
        "AI_PLATFORM_POSTGRES_HOST=127.0.0.1\n"
        f"AI_PLATFORM_POSTGRES_PORT={port}\n"
        "AI_PLATFORM_POSTGRES_DATABASE=ai_platform\n"
        "AI_PLATFORM_POSTGRES_USER=ai_platform_runtime\n"
        f"AI_PLATFORM_POSTGRES_PASSWORD={runtime_password}\n"
        "AI_PLATFORM_POSTGRES_SSLMODE=prefer\n"
        "AI_PLATFORM_POSTGRES_CONNECT_TIMEOUT_SECONDS=5\n"
        "AI_PLATFORM_POSTGRES_POOL_MIN_SIZE=1\n"
        "AI_PLATFORM_POSTGRES_POOL_MAX_SIZE=5\n"
        "AI_PLATFORM_POSTGRES_POOL_TIMEOUT_SECONDS=10\n"
    ),
    encoding="utf-8",
)

bootstrap.chmod(0o600)
runtime.chmod(0o600)

print(
    "PASS: generated isolated external credentials"
)
PY

[[ "$(stat -c '%a' "$CONFIG_DIR")" == "700" ]]
[[ "$(stat -c '%a' "$BOOTSTRAP_ENV")" == "600" ]]
[[ "$(stat -c '%a' "$RUNTIME_ENV")" == "600" ]]

echo "PASS: credential permissions"

echo
echo "===== START ISOLATED POSTGRESQL ====="

HOME="$CI_HOME" \
AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
docker compose \
  -p "$PROJECT" \
  -f compose.yaml \
  up \
  -d \
  postgres

CONTAINER_ID="$(
  HOME="$CI_HOME" \
  AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
  docker compose \
    -p "$PROJECT" \
    -f compose.yaml \
    ps \
    -q \
    postgres
)"

[[ -n "$CONTAINER_ID" ]] || {
  echo "FAIL: PostgreSQL container was not created"
  exit 1
}

HEALTH=""

for _ in {1..30}; do
  HEALTH="$(
    docker inspect \
      --format '{{.State.Health.Status}}' \
      "$CONTAINER_ID" \
      2>/dev/null \
      || true
  )"

  if [[ "$HEALTH" == "healthy" ]]; then
    break
  fi

  sleep 2
done

echo "postgres_health=$HEALTH"

[[ "$HEALTH" == "healthy" ]] || {
  echo "FAIL: PostgreSQL did not become healthy"

  HOME="$CI_HOME" \
  AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
  docker compose \
    -p "$PROJECT" \
    -f compose.yaml \
    logs \
    postgres

  exit 1
}

echo "PASS: isolated PostgreSQL healthy"

echo
echo "===== PROVISION RUNTIME ROLE ====="

uv run python \
  scripts/postgres/provision_runtime_role.py \
  --bootstrap-env "$BOOTSTRAP_ENV" \
  --runtime-env "$RUNTIME_ENV"

echo "PASS: runtime role provisioned"

echo
echo "===== MIGRATE TO HEAD ====="

uv run python \
  scripts/postgres/migrate.py \
  upgrade \
  head \
  --bootstrap-env "$BOOTSTRAP_ENV" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --sslmode prefer

echo "PASS: migration head applied"

echo
echo "===== LIVE PROVIDER INTEGRATION ====="

AI_PLATFORM_RUN_POSTGRES_INTEGRATION=1 \
AI_PLATFORM_POSTGRES_INTEGRATION_BOOTSTRAP_ENV="$BOOTSTRAP_ENV" \
AI_PLATFORM_POSTGRES_INTEGRATION_RUNTIME_ENV="$RUNTIME_ENV" \
uv run pytest \
  -W error \
  tests/integration/test_postgres_vector_live.py \
  tests/integration/test_postgres_workflow_persistence_live.py \
  -v

echo "PASS: live provider integration"

echo
echo "===== FINAL DATABASE INVARIANTS ====="

ALEMBIC_HEADS="$(
  uv run alembic heads
)"

ALEMBIC_HEAD_COUNT="$(
  printf '%s\n' "$ALEMBIC_HEADS"     | grep -c '(head)'
)"

if [[ "$ALEMBIC_HEAD_COUNT" -ne 1 ]]; then
  echo "FAIL: expected exactly one Alembic head"
  printf '%s\n' "$ALEMBIC_HEADS"
  exit 1
fi

EXPECTED_REVISION="$(
  printf '%s\n' "$ALEMBIC_HEADS"     | awk '/\(head\)$/ {print $1}'
)"

[[ -n "$EXPECTED_REVISION" ]] || {
  echo "FAIL: unable to resolve Alembic head revision"
  exit 1
}

echo "expected_revision=$EXPECTED_REVISION"

FINAL_STATE="$(
  HOME="$CI_HOME" \
  AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
  docker compose \
    -p "$PROJECT" \
    -f compose.yaml \
    exec \
    -T \
    postgres \
    sh -lc \
    'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F "|" -At' \
    <<'SQL'
SELECT
    (
        SELECT count(*)
        FROM pg_roles
        WHERE rolname = 'ai_platform_runtime'
    ),
    (
        SELECT count(*)
        FROM pg_extension
        WHERE extname = 'vector'
          AND extversion = '0.8.6'
    ),
    (
        SELECT count(*)
        FROM pg_tables
        WHERE schemaname = 'ai_platform'
    ),
    (
        SELECT count(*)
        FROM ai_platform.vector_collections
    ),
    (
        SELECT count(*)
        FROM ai_platform.vector_records
    ),
    (
        SELECT version_num
        FROM alembic_version
        LIMIT 1
    );
SQL
)"

echo "role|vector086|tables|collections|records|revision=$FINAL_STATE"

[[ "$FINAL_STATE" == "1|1|3|0|0|$EXPECTED_REVISION" ]] || {
  echo "FAIL: integration database final state is unexpected"
  exit 1
}

SEQUENCE_STATE="$(
  HOME="$CI_HOME" \
  AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
  docker compose \
    -p "$PROJECT" \
    -f compose.yaml \
    exec \
    -T \
    postgres \
    sh -lc \
    'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F "|" -At' \
    <<'SQL'
SELECT
    last_value,
    is_called
FROM ai_platform.vector_collections_collection_id_seq;
SQL
)"

echo "sequence=$SEQUENCE_STATE"

[[ "$SEQUENCE_STATE" == "1|f" ]] || {
  echo "FAIL: integration identity sequence not reset"
  exit 1
}

AUTH_STATE="$(
  HOME="$CI_HOME" \
  AI_PLATFORM_POSTGRES_HOST_PORT="$PORT" \
  docker compose \
    -p "$PROJECT" \
    -f compose.yaml \
    exec \
    -T \
    postgres \
    sh -lc \
    'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F "|" -At' \
    <<'SQL'
SELECT
    (
        SELECT count(*)
        FROM pg_hba_file_rules
        WHERE auth_method = 'trust'
    ),
    (
        SELECT count(*)
        FROM pg_hba_file_rules
        WHERE type LIKE 'host%'
          AND auth_method <> 'scram-sha-256'
    ),
    (
        SELECT count(*)
        FROM pg_hba_file_rules
        WHERE type = 'local'
          AND auth_method <> 'scram-sha-256'
    );
SQL
)"

echo "trust|host_non_scram|local_non_scram=$AUTH_STATE"

[[ "$AUTH_STATE" == "0|0|0" ]] || {
  echo "FAIL: integration DB authentication policy is not SCRAM-only"
  exit 1
}

echo "PASS: final database invariants"

echo
echo "POSTGRESQL INTEGRATION VALIDATION: PASS"
