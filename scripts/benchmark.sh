#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-}"
if [[ -z "$VIDEO" ]]; then
  echo "Usage: ./scripts/benchmark.sh <video> [output_dir]"
  exit 2
fi

OUTPUT="${2:-data/processed/benchmark}"

if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: video not found: $VIDEO"
  exit 1
fi

if [[ ! -f models/HockeyAI_model_weight.pt ]]; then
  echo "ERROR: HockeyAI weights not found. Run ./scripts/setup.sh first."
  exit 1
fi

python scripts/benchmark_video.py \
  --video "$VIDEO" \
  --model models/HockeyAI_model_weight.pt \
  --output "$OUTPUT" \
  --device 0 \
  --stride 1
