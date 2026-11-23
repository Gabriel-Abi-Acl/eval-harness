"""Loader JSONL do exemplo rotulado.

Schema fechado: `id`, `input`, `expected` e `metadata` opcional. Uma linha ruim recusa
o arquivo inteiro, porque um dataset parcial mudaria o hash sem avisar.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from eval_harness.errors import DatasetError
from eval_harness.hashing import digest

SCHEMA_NAME = "eval_harness.example.v1"
_REQUIRED = ("id", "input", "expected")
_ALLOWED = frozenset({"id", "input", "expected", "metadata"})


@dataclass(frozen=True)
class Example:
    """Um item rotulado. `expected` é string; métrica de JSON interpreta esse texto."""

    id: str
    input: str
    expected: str
    metadata: Mapping[str, Any]

    def to_canonical(self) -> dict[str, Any]:
        """Forma que entra no hash. `metadata` ausente vira objeto vazio."""
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "metadata": _plain_metadata(self.metadata),
        }


@dataclass(frozen=True)
class Dataset:
    """Exemplos na ordem do arquivo. `content_hash` não inclui o caminho."""

    examples: tuple[Example, ...]
    source: str
    content_hash: str


def load_dataset(path: str | Path) -> Dataset:
    """Lê um JSONL UTF-8. Linha em branco, campo extra e id repetido são erro."""
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError("dataset is not valid UTF-8") from exc
    except OSError as exc:
        raise DatasetError(f"could not read dataset: {source}") from exc

    examples: list[Example] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(raw.splitlines(), start=1):
        example = _parse_line(line, number)
        if example.id in seen:
            first = seen[example.id]
            raise DatasetError(
                f"line {number}: duplicate id {example.id!r} (first at line {first})",
                line=number,
            )
        seen[example.id] = number
        examples.append(example)

    if not examples:
        raise DatasetError("dataset has no examples")

    payload = {
        "schema": SCHEMA_NAME,
        "examples": [example.to_canonical() for example in examples],
    }
    return Dataset(
        examples=tuple(examples),
        source=str(source),
        content_hash=digest(payload),
    )


def _parse_line(line: str, number: int) -> Example:
    if line.strip() == "":
        raise DatasetError(f"line {number}: blank line", line=number)
    try:
        payload = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"line {number}: invalid JSON", line=number) from exc
    except ValueError as exc:
        raise DatasetError(f"line {number}: {exc}", line=number) from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"line {number}: expected a JSON object", line=number)

    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        names = ", ".join(unknown)
        raise DatasetError(f"line {number}: unknown field: {names}", line=number)
    missing = [key for key in _REQUIRED if key not in payload]
    if missing:
        names = ", ".join(missing)
        raise DatasetError(f"line {number}: missing field: {names}", line=number)

    example_id = _required_string(payload["id"], number=number, field="id")
    if example_id != example_id.strip() or any(char.isspace() for char in example_id):
        raise DatasetError(
            f"line {number}: id must be a non-empty string without whitespace",
            line=number,
        )
    text = _required_string(payload["input"], number=number, field="input")
    expected = _required_string(payload["expected"], number=number, field="expected")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"line {number}: metadata must be an object", line=number)

    return Example(
        id=example_id,
        input=text,
        expected=expected,
        metadata=MappingProxyType(metadata),
    )


def _required_string(value: Any, *, number: int, field: str) -> str:
    # id vazio e não-string quebram o manifesto; input e expected vazios ainda são string.
    if not isinstance(value, str):
        raise DatasetError(f"line {number}: {field} must be a string", line=number)
    if field == "id" and value == "":
        raise DatasetError(
            f"line {number}: id must be a non-empty string without whitespace",
            line=number,
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key: {key}")
        seen[key] = value
    return seen


def _plain_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in sorted(metadata)}
