#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_repo_env.sh"
# Keep the CUDA PyTorch installation supplied by your NVIDIA container.
python -c 'import torch; print("Existing torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'
python -m pip install -e '.[dev]'
python -m pytest -q
