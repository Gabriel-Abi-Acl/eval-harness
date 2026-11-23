"""Apoio dos testes. Monta exemplo e JSONL temporário sem passar pelo dataset versionado."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from eval_harness import Example

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = ROOT / "data" / "placeholder.jsonl"


def make_example(
    example_id: str = "ex-01",
    text: str = "pergunta",
    expected: str = "resposta",
    metadata: Mapping[str, Any] | None = None,
) -> Example:
    return Example(
        id=example_id,
        input=text,
        expected=expected,
        metadata=MappingProxyType(dict(metadata or {})),
    )


def write_jsonl(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str = "examples.jsonl",
) -> Path:
    path = directory / name
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
