#!/usr/bin/env bash
set -euo pipefail

echo "=== ICE-brAIn status ==="
echo
echo "Working directory: $(pwd)"
echo "Git:"
git rev-parse --short HEAD
git status --short
echo
echo "Python:"
python --version
echo
echo "PyTorch:"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"
echo
echo "FFmpeg:"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1
else
  echo "NOT INSTALLED"
fi
echo
echo "HockeyAI weights:"
if [[ -f models/HockeyAI_model_weight.pt ]]; then
  ls -lh models/HockeyAI_model_weight.pt
else
  echo "NOT DOWNLOADED"
fi
