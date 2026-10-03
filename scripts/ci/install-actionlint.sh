#!/usr/bin/env bash

set -euo pipefail

VERSION="1.7.12"

ROOT="$(git rev-parse --show-toplevel)"
TOOL_DIR="$ROOT/.tools/actionlint/$VERSION"
BINARY="$TOOL_DIR/actionlint"

case "$(uname -m)" in
  x86_64)
    ARCHIVE="actionlint_${VERSION}_linux_amd64.tar.gz"
    SHA256="8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
    ;;
  aarch64|arm64)
    ARCHIVE="actionlint_${VERSION}_linux_arm64.tar.gz"
    SHA256="325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6"
    ;;
  *)
    echo "FAIL: unsupported architecture: $(uname -m)"
    exit 1
    ;;
esac

if [[ -x "$BINARY" ]]; then
  INSTALLED="$("$BINARY" -version)"

  if [[ "$INSTALLED" == *"$VERSION"* ]]; then
    echo "PASS: actionlint $VERSION already installed"
    echo "$BINARY"
    exit 0
  fi
fi

mkdir -p "$TOOL_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

URL="https://github.com/rhysd/actionlint/releases/download/v${VERSION}/${ARCHIVE}"
ARCHIVE_PATH="$TMP_DIR/$ARCHIVE"

echo "Downloading actionlint $VERSION..."

curl \
  --fail \
  --silent \
  --show-error \
  --location \
  "$URL" \
  --output "$ARCHIVE_PATH"

echo "${SHA256}  ${ARCHIVE_PATH}" \
  | sha256sum --check --status \
  || {
    echo "FAIL: actionlint checksum verification failed"
    exit 1
  }

tar \
  -xzf "$ARCHIVE_PATH" \
  -C "$TOOL_DIR" \
  actionlint

chmod 0755 "$BINARY"

"$BINARY" -version

echo "PASS: actionlint installed and checksum verified"
echo "$BINARY"
