#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

EXPECTED_PYTHON="3.13.15"

echo "===== LOCKFILE ====="

uv lock --check

echo "PASS: lockfile"

echo
echo "===== ENVIRONMENT ====="

uv sync --locked

PYTHON_VERSION="$(
  uv run python -c \
    'import sys; print(".".join(map(str, sys.version_info[:3])))'
)"

if [[ "$PYTHON_VERSION" != "$EXPECTED_PYTHON" ]]; then
  echo "FAIL: expected Python $EXPECTED_PYTHON"
  echo "found: $PYTHON_VERSION"
  exit 1
fi

echo "PASS: Python=$PYTHON_VERSION"

echo
echo "===== FORMAT ====="

uv run ruff format --check src tests
echo "PASS: format"

echo
echo "===== LINT ====="

uv run ruff check src tests
echo "PASS: lint"

echo
echo "===== TYPE CHECK ====="

uv run mypy src tests
echo "PASS: mypy"

echo
echo "===== TESTS ====="

uv run pytest -W error
echo "PASS: pytest"

echo
echo "===== BUILD ====="

rm -rf dist
uv build

WHEELS="$(find dist -maxdepth 1 -type f -name '*.whl' | wc -l)"
SDISTS="$(find dist -maxdepth 1 -type f -name '*.tar.gz' | wc -l)"

[[ "$WHEELS" -eq 1 ]] \
  || { echo "FAIL: expected one wheel"; exit 1; }

[[ "$SDISTS" -eq 1 ]] \
  || { echo "FAIL: expected one sdist"; exit 1; }

echo "PASS: build"

echo
echo "QUALITY VALIDATION: PASS"
