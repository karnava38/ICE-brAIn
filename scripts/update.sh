#!/usr/bin/env bash
set -euo pipefail

echo "=== ICE-brAIn update ==="

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: local changes exist. Commit or stash them before updating."
  git status --short
  exit 1
fi

git pull --ff-only
python -m pip install -e ".[dev]"
pytest -q

echo
echo "Update complete."
git rev-parse --short HEAD
