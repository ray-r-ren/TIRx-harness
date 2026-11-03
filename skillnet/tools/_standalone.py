"""Make kernel .py copies load on their own: rewrite relative imports to absolute tirx_kernels imports.

Only parses (never executes) the code. Line numbers are kept: a multi-line relative import is
replaced by a one-line absolute import followed by blank lines.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Standalone:
    text: str
    standalone: bool
    rewritten: int = 0
    problems: list[str] = field(default_factory=list)


def _module_exists(root: Path, dotted: str) -> bool:
    """Whether `dotted` names a module or package file under `root` (the directory holding the top package)."""
    rel = Path(*dotted.split("."))
    return (root / rel).with_suffix(".py").is_file() or (root / rel / "__init__.py").is_file()


def make_standalone(text: str, package: str | None, packages_root: Path | None) -> Standalone:
    """Rewrite `from .x import y` in a file of `package` (e.g. 'tirx_kernels.gemm') to absolute imports.

    `packages_root` is the directory that contains the top-level package (site-packages); it is used to
    check that every rewritten target module exists. Returns the new text and whether it is standalone:
    it parses, no relative import remains, and every rewritten target exists.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return Standalone(text, False, 0, [f"does not parse: {e.msg} (line {e.lineno})"])
    rel = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0]
    if not rel:
        return Standalone(text, True)
    if not package:
        return Standalone(text, False, 0, ["relative imports but no package to resolve them against"])
    lines = text.splitlines(keepends=True)
    problems: list[str] = []
    parts = package.split(".")
    for node in sorted(rel, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        keep = len(parts) - (node.level - 1)
        if keep < 1:
            problems.append(f"line {node.lineno}: relative import goes above the top package")
            continue
        target = ".".join(parts[:keep] + ([node.module] if node.module else []))
        if packages_root is not None and not _module_exists(packages_root, target):
            problems.append(f"line {node.lineno}: module {target} not found")
        new = ast.unparse(ast.ImportFrom(module=target, names=node.names, level=0))
        first, last = node.lineno - 1, node.end_lineno - 1
        prefix = lines[first][: node.col_offset]
        tail = lines[last][node.end_col_offset:]
        eol = "" if tail.endswith("\n") else "\n" if last + 1 < len(lines) else ""
        replacement = prefix + new + tail + eol + "\n" * (last - first)
        lines[first:last + 1] = [replacement]
    out = "".join(lines)
    try:
        left = [n for n in ast.walk(ast.parse(out)) if isinstance(n, ast.ImportFrom) and n.level > 0]
    except SyntaxError as e:
        return Standalone(text, False, 0, [f"rewrite did not parse: {e.msg}"])
    if left:
        problems.append(f"{len(left)} relative import(s) remain")
        return Standalone(out, False, len(rel) - len(left), problems)
    return Standalone(out, not problems, len(rel), problems)


def package_of(rel_path: Path, top: str = "tirx_kernels") -> str:
    """Dotted package of a file at `rel_path` inside package `top` (e.g. gemm/a.py -> tirx_kernels.gemm)."""
    return ".".join([top, *rel_path.parts[:-1]])
