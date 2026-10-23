from __future__ import annotations

import ast
from pathlib import Path


def get_sweep_pkg_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "sweep"


def test_no_suppression_comments_in_new_modules() -> None:
    """Ensures sub-agents do not cheat by adding type or lint suppression comments."""
    if "mutants" in Path(__file__).resolve().parts:
        return
    pkg_dir = get_sweep_pkg_dir()
    new_modules = [
        "models.py",
        "checks.py",
        "r_mt19937.py",
        "indexing.py",
        "hdv.py",
        "projection_core.py",
        "phylo.py",
        "core.py",
    ]

    for mod_name in new_modules:
        mod_path = pkg_dir / mod_name
        if not mod_path.exists():
            continue
        lines = mod_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            # We allow noqa only on specific documented signatures if strictly needed,
            # but disallow type: ignore or pyrefly: ignore
            assert "# type: ignore" not in line, f"Found '# type: ignore' in {mod_name}:{idx}"
            assert "# pyrefly: ignore" not in line, f"Found '# pyrefly: ignore' in {mod_name}:{idx}"


def test_sparse_anti_densification_guard() -> None:
    """Ensures sparse routines never call .toarray() or .todense() internally."""
    pkg_dir = get_sweep_pkg_dir()
    sparse_core_modules = ["hdv.py", "projection_core.py"]

    for mod_name in sparse_core_modules:
        mod_path = pkg_dir / mod_name
        if not mod_path.exists():
            continue
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                assert attr_name not in (
                    "toarray",
                    "todense",
                ), (
                    f"Prohibited sparse densification call '.{attr_name}()' "
                    f"found in {mod_name} at line {node.lineno}!"
                )


def test_architectural_layers_dag() -> None:
    """Verifies that foundational modules never import higher-level orchestration modules."""
    pkg_dir = get_sweep_pkg_dir()
    foundation_modules = ["models.py", "checks.py", "r_mt19937.py", "indexing.py"]
    prohibited_imports = {"sweep.core", "sweep.phylo", "core", "phylo"}

    for mod_name in foundation_modules:
        mod_path = pkg_dir / mod_name
        if not mod_path.exists():
            continue
        tree = ast.parse(mod_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in prohibited_imports, (
                        f"Illegal upward dependency '{alias.name}' in {mod_name}!"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert mod not in prohibited_imports, (
                    f"Illegal upward dependency 'from {mod}' in {mod_name}!"
                )
