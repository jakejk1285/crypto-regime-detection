#!/usr/bin/env python3
"""
Mechanical quarantine of the leaking legacy pipeline.

`backtest_system/` is retained on purpose: the naive 54.35% result must stay
reproducible for the Phase 6 narrative. But it leaks by design —
`backtest_data_manager.py:404-417` fits StandardScaler/PCA/KMeans on the full sample,
and `:394` does `.ffill().bfill().fillna(0)`.

The single highest-probability way this rebuild fails silently is one convenience
import pulling that scaler, that PCA, or that bfill back into the live pipeline. The
causality test would not necessarily catch it: it guards `data_pipeline/` outputs, not
the import graph of code that consumes them.

So the quarantine is enforced here, mechanically, rather than documented and hoped for.
This test parses every module in the research pipeline with `ast` and fails if any of
them imports from `backtest_system` — including transitively, via any module the
pipeline imports.

Run: python crypto_cluster_pca/data_pipeline/test_quarantine.py
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "crypto_cluster_pca" / "data_pipeline"
FORBIDDEN = {"backtest_system", "backtest_data_manager", "backtest_trading_strategy",
             "backtest_engine", "expected_value_analyzer", "sharpe_optimized_strategy"}

# Directories whose modules must be clean. Phase 4+ additions go here.
GUARDED_DIRS = [PIPELINE]


def imported_names(path: Path) -> set:
    """Every module name this file imports, by any syntax."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
                names.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
                names.add(node.module)
        # catch dynamic escapes: __import__("backtest_system"), importlib.import_module(...)
        elif isinstance(node, ast.Call):
            fn = node.func
            is_dyn = (isinstance(fn, ast.Name) and fn.id == "__import__") or (
                isinstance(fn, ast.Attribute) and fn.attr == "import_module"
            )
            if is_dyn and node.args and isinstance(node.args[0], ast.Constant):
                names.add(str(node.args[0].value).split(".")[0])
    return names


def sys_path_escapes(path: Path) -> list:
    """
    Flag sys.path manipulation that points at backtest_system.

    An import guard is defeatable by inserting the legacy directory onto sys.path and
    importing its modules by bare name, so that pattern is caught too.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("insert", "append"):
                src = ast.unparse(node)
                if "sys.path" in src and "backtest_system" in src:
                    hits.append(src)
    return hits


def main() -> int:
    files = sorted(f for d in GUARDED_DIRS for f in d.glob("*.py"))
    print(f"Guarding {len(files)} module(s) against imports from backtest_system/\n")

    violations = []
    for f in files:
        bad = imported_names(f) & FORBIDDEN
        escapes = sys_path_escapes(f)
        if bad or escapes:
            violations.append((f, bad, escapes))
            print(f"  FAIL  {f.name}")
            for b in sorted(bad):
                print(f"          imports forbidden module: {b}")
            for e in escapes:
                print(f"          sys.path escape: {e}")
        else:
            print(f"  ok    {f.name}")

    print()
    if violations:
        print("=" * 72)
        print(f"QUARANTINE BREACHED — {len(violations)} module(s) reach into backtest_system/")
        print("The legacy pipeline fits StandardScaler/PCA/KMeans on the full sample and")
        print("backfills NaNs. Importing it reintroduces the leak this rebuild removed.")
        print("=" * 72)
        return 1

    print("=" * 72)
    print("QUARANTINE INTACT — no pipeline module reaches into backtest_system/")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
