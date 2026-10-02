#!/usr/bin/env bash
# Force every repository script to use the source tree that ships with this checkout.
# This prevents an older site-packages installation of csa-adapter from shadowing
# v0.2.1 when invoking `python -m csa_adapter...`.
set -euo pipefail
_CSA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CSA_REPO_ROOT="$(cd "${_CSA_SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${CSA_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CSA_REPO_ROOT}"
