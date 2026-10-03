#!/usr/bin/env bash

set -euo pipefail

VERSION="8.30.1"

ROOT="$(git rev-parse --show-toplevel)"
TOOL_DIR="$ROOT/.tools/gitleaks/$VERSION"
BINARY="$TOOL_DIR/gitleaks"

case "$(uname -m)" in
  x86_64)
    ARCHIVE="gitleaks_${VERSION}_linux_x64.tar.gz"
    SHA256="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
    ;;
  aarch64|arm64)
    ARCHIVE="gitleaks_${VERSION}_linux_arm64.tar.gz"
    SHA256="e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080"
    ;;
  *)
    echo "FAIL: unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac

if [[ -x "$BINARY" ]]; then
  INSTALLED="$("$BINARY" version)"

  if [[ "$INSTALLED" == *"$VERSION"* ]]; then
    echo "PASS: Gitleaks $VERSION already installed"
    echo "$BINARY"
    exit 0
  fi
fi

mkdir -p "$TOOL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

URL="https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}/${ARCHIVE}"
ARCHIVE_PATH="$TMP_DIR/$ARCHIVE"

echo "Downloading Gitleaks $VERSION..."

curl \
  --fail \
  --silent \
  --show-error \
  --location \
  "$URL" \
  --output "$ARCHIVE_PATH"

echo "${SHA256}  ${ARCHIVE_PATH}" | sha256sum --check --status \
  || {
    echo "FAIL: Gitleaks checksum verification failed"
    exit 1
  }

tar \
  -xzf "$ARCHIVE_PATH" \
  -C "$TOOL_DIR" \
  gitleaks

chmod 0755 "$BINARY"

"$BINARY" version

echo "PASS: Gitleaks installed and checksum verified"
echo "$BINARY"
