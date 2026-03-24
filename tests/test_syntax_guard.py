from __future__ import annotations

import ast
from pathlib import Path


ROOTS = [Path("reco_trading"), Path("tests")]


def test_python_sources_have_valid_syntax() -> None:
    failures: list[str] = []
    for root in ROOTS:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                failures.append(f"{path}:{exc.lineno}:{exc.offset} {exc.msg}")
    assert not failures, "\n".join(failures)
