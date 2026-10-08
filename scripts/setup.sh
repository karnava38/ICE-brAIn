#!/usr/bin/env bash
set -euo pipefail

echo "=== ICE-brAIn setup ==="

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi was not found."
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python was not found."
  exit 1
fi

python --version

echo "Installing ICE-brAIn and development dependencies..."
python -m pip install -e ".[dev]"

if [[ ! -f models/HockeyAI_model_weight.pt ]]; then
  echo "Downloading HockeyAI weights..."
  python scripts/download_hockeyai.py
else
  echo "HockeyAI weights already present; skipping download."
fi

echo
echo "Running tests..."
pytest -q

echo
echo "Setup complete."
