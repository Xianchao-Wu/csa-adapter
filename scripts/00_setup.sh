#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Keep the CUDA PyTorch installation supplied by your NVIDIA container.
python -c 'import torch; print("Existing torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
python -m pip install -e '.[dev]'
python -m pytest -q
