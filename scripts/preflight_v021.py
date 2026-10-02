#!/usr/bin/env python3
"""Dependency-free preflight for repository/module resolution and v0.2.1 CLI flags."""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

repo = Path(os.environ.get("CSA_REPO_ROOT", Path(__file__).resolve().parents[1])).resolve()
expected_src = (repo / "src").resolve()
expected_train = expected_src / "csa_adapter" / "longform" / "train.py"
expected_eval = expected_src / "csa_adapter" / "longform" / "evaluate.py"

if not expected_train.is_file() or not expected_eval.is_file():
    raise SystemExit(f"[preflight] missing repository source files under {expected_src}")

# Determine which csa_adapter package Python would see first without importing it.
resolved_pkg = None
for entry in sys.path:
    base = Path(entry or ".").resolve()
    candidate = base / "csa_adapter" / "__init__.py"
    if candidate.is_file():
        resolved_pkg = candidate.parent.resolve()
        break
expected_pkg = expected_src / "csa_adapter"
if resolved_pkg != expected_pkg:
    raise SystemExit(
        "[preflight] WRONG csa_adapter resolved first:\n"
        f"  resolved: {resolved_pkg}\n"
        f"  expected: {expected_pkg}\n"
        "Ensure repository src/ is first on PYTHONPATH."
    )

# Parse source with stdlib AST so this check does not require transformers/torch.
def string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

train_strings = string_literals(expected_train)
eval_strings = string_literals(expected_eval)
required_train = {
    "--alpha-init",
    "--alpha-max",
    "--gate-bias-init",
    "--residual-ratio-cap",
    "--diagnostics-every",
    "--save-validation-checkpoints",
}
required_eval = {"--diagnostics", "--memory-mode", "--dense-reading"}
missing_train = sorted(required_train - train_strings)
missing_eval = sorted(required_eval - eval_strings)
if missing_train or missing_eval:
    raise SystemExit(
        f"[preflight] v0.2.1 CLI mismatch: missing train={missing_train}, eval={missing_eval}"
    )

version_file = expected_src / "csa_adapter" / "__init__.py"
version_text = version_file.read_text()
if '__version__ = "0.2.1"' not in version_text:
    raise SystemExit("[preflight] repository does not identify itself as csa-adapter v0.2.1")

print(f"[preflight] repo      : {repo}")
print(f"[preflight] python src: {expected_src}")
print(f"[preflight] package   : {resolved_pkg}")
print("[preflight] v0.2.1 CLI/source checks: OK")
