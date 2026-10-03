#!/usr/bin/env bash

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

GITLEAKS="$ROOT/.tools/gitleaks/8.30.1/gitleaks"

mkdir -p reports

echo "===== LOCKFILE ====="

uv lock --check
uv sync --locked

echo "PASS: locked environment"

echo
echo "===== GITLEAKS INSTALL ====="

./scripts/security/install-gitleaks.sh

echo
echo "===== GITLEAKS HISTORY ====="

"$GITLEAKS" \
  git \
  --no-banner \
  --redact \
  --log-opts="--all" \
  .

echo "PASS: Git history secret scan"

echo
echo "===== GITLEAKS SOURCE TREE ====="

"$GITLEAKS" \
  dir \
  --no-banner \
  --redact \
  .

echo "PASS: source secret scan"

echo
echo "===== VULNERABILITY AUDIT ====="

uvx \
  --from pip-audit==2.10.1 \
  pip-audit \
  --path .venv/lib/python3.13/site-packages \
  --skip-editable \
  --progress-spinner off

echo "PASS: dependency vulnerability audit"

echo
echo "===== LICENSE INVENTORY ====="

uvx \
  --from pip-licenses==5.5.5 \
  pip-licenses \
  --python .venv/bin/python \
  --from=mixed \
  --ignore-packages ai-engineering-agent-platform \
  --format=json \
  --output-file reports/licenses.json

uvx \
  --from pip-licenses==5.5.5 \
  pip-licenses \
  --python .venv/bin/python \
  --from=mixed \
  --ignore-packages ai-engineering-agent-platform \
  --fail-on="UNKNOWN"

echo "PASS: dependency licenses identified"

echo
echo "===== SBOM ====="

uvx \
  --from cyclonedx-bom==7.3.1 \
  cyclonedx-py \
  environment \
  --output-reproducible \
  --spec-version 1.6 \
  --output-format JSON \
  --output-file reports/sbom.cdx.json \
  .venv/bin/python

uv run python scripts/security/validate-sbom.py

echo
echo "SUPPLY-CHAIN VALIDATION: PASS"
