#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "=================================================="
echo "AI ENGINEERING & AGENT PLATFORM — GOLDEN DEMO"
echo "=================================================="

echo
echo "===== PREREQUISITES ====="

command -v git >/dev/null 2>&1 || {
  echo "FAIL: git is required"
  exit 1
}

command -v uv >/dev/null 2>&1 || {
  echo "FAIL: uv is required"
  exit 1
}

command -v docker >/dev/null 2>&1 || {
  echo "FAIL: Docker is required"
  exit 1
}

git --version
uv --version
docker compose version

uv lock --check
uv sync --locked

echo
echo "===== STAGE 1: DETERMINISTIC RAG + GUARDRAILS ====="

uv run python \
  examples/golden_demo.py

echo
echo "===== STAGE 2: LIVE DURABLE POSTGRESQL CONTROL PLANE ====="

echo
echo "This stage validates:"
echo "- PostgreSQL + pgvector"
echo "- workflow checkpoint persistence"
echo "- workflow restart and resume"
echo "- durable approval state"
echo "- authenticated approval authorization"
echo "- durable agent continuations"
echo "- agent + ToolExecutionService control"
echo "- HITL pause / approval / restart / resume"
echo "- replay protection boundaries"

./scripts/ci/validate-postgres-integration.sh

echo
echo "=================================================="
echo "PROJECT 5 GOLDEN DEMO — PASS"
echo "=================================================="
echo "DETERMINISTIC_GROUNDED_RETRIEVAL=PASS"
echo "PLATFORM_OWNED_CITATIONS=PASS"
echo "DETERMINISTIC_GUARDRAILS=PASS"
echo "POSTGRESQL_LIVE=PASS"
echo "DURABLE_WORKFLOW_STATE=PASS"
echo "AUTHENTICATED_HITL=PASS"
echo "DURABLE_AGENT_CONTINUATION=PASS"
echo "CONTROLLED_TOOL_EXECUTION=PASS"
echo "RESTART_AND_RESUME=PASS"
echo "REPLAY_BOUNDARIES=PASS"
echo "EXACTLY_ONCE_EXTERNAL_EFFECTS=NOT_CLAIMED"
echo "LLM_BYTE_FOR_BYTE_REPRODUCIBILITY=NOT_CLAIMED"
echo "=================================================="
