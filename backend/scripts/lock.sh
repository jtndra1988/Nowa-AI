#!/usr/bin/env bash
set -euo pipefail

# Ensures a clean, reproducible Poetry install + lock on CI or local.
if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found. Installing..."
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[poetry] Using $(poetry --version)"

# Create/refresh virtualenv and lock deps
poetry env use python3
poetry lock --no-update
poetry install --no-interaction --no-root

echo "poetry.lock generated and environment installed."
