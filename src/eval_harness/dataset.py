"""Loader JSONL do exemplo rotulado e cartão do conjunto.

Schema fechado do exemplo: `id`, `input`, `expected` e `metadata` opcional. Uma linha
ruim recusa o arquivo inteiro, porque um dataset parcial mudaria o hash sem avisar.
O cartão (`card.json`) acrescenta versão, tarefa e checagens do gabarito.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from eval_harness.errors import DatasetError
from eval_harness.hashing import digest

SCHEMA_NAME = "eval_harness.example.v1"
CARD_SCHEMA_NAME = "eval_harness.dataset_card.v1"
_REQUIRED = ("id", "input", "expected")
_ALLOWED = frozenset({"id", "input", "expected", "metadata"})
_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_JSON_TYPES = frozenset({"string", "integer", "boolean"})
_CARD_REQUIRED = (
    "schema",
    "name",
    "version",
    "task_type",
    "metrics",
    "license_note",
    "construction_notes",
    "language",
    "tag_schema",
    "example_count",
)
# Nomes que o cartão declara. Métrica de conjunto não entra no runner por exemplo.
TASK_METRICS = MappingProxyType(
    {
        "short_answer_qa": ("exact_match", "normalized_match", "token_f1"),
        "classification": ("accuracy", "per_class_precision_recall", "macro_f1"),
        "json_field_extraction": ("json_field_accuracy",),
    }
)


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


@dataclass(frozen=True)
class DatasetCard:
    """Metadados versionados de um conjunto. Não entra no hash dos exemplos."""

    name: str
    version: str
    task_type: str
    metrics: tuple[str, ...]
    license_note: str
    construction_notes: str
    language: str
    tag_schema: Mapping[str, tuple[str, ...]]
    example_count: int
    label_set: tuple[str, ...] | None = None
    fields: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class CardedDataset:
    """Cartão mais os exemplos já validados contra ele."""

    card: DatasetCard
    dataset: Dataset


def load_carded_dataset(path: str | Path) -> CardedDataset:
    """Lê `card.json` e `examples.jsonl` de um diretório.

    O JSONL segue o schema do exemplo. O cartão recusa rótulo fora do conjunto,
    JSON de referência inválido, tag fora do esquema e contagem divergente.
    A linha reportada é a do JSONL: linha em branco já foi recusada antes.
    """
    directory = Path(path)
    if not directory.is_dir():
        raise DatasetError(f"dataset directory not found: {directory}")
    card = _load_card(directory / "card.json")
    dataset = load_dataset(directory / "examples.jsonl")
    _validate_card_examples(card, dataset)
    return CardedDataset(card=card, dataset=dataset)


def _load_card(path: Path) -> DatasetCard:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError("dataset card is not valid UTF-8") from exc
    except OSError as exc:
        raise DatasetError(f"could not read dataset card: {path}") from exc
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise DatasetError("dataset card is not valid JSON") from exc
    except ValueError as exc:
        raise DatasetError(f"dataset card: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetError("dataset card must be a JSON object")

    task_type = payload.get("task_type")
    allowed = set(_CARD_REQUIRED)
    required = list(_CARD_REQUIRED)
    if task_type == "classification":
        allowed.add("label_set")
        required.append("label_set")
    elif task_type == "json_field_extraction":
        allowed.add("fields")
        required.append("fields")

    unknown = sorted(set(payload) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise DatasetError(f"dataset card: unknown field: {names}")
    missing = [key for key in required if key not in payload]
    if missing:
        names = ", ".join(missing)
        raise DatasetError(f"dataset card: missing field: {names}")
    if payload["schema"] != CARD_SCHEMA_NAME:
        raise DatasetError(f"dataset card: schema must be {CARD_SCHEMA_NAME}")

    name = _plain_text(payload["name"], "name")
    version = _plain_text(payload["version"], "version")
    if _VERSION_RE.fullmatch(version) is None:
        raise DatasetError("dataset card: version must be major.minor.patch")
    language = _plain_text(payload["language"], "language")
    license_note = _plain_text(payload["license_note"], "license_note")
    construction_notes = _plain_text(payload["construction_notes"], "construction_notes")
    if not isinstance(task_type, str) or task_type not in TASK_METRICS:
        raise DatasetError("dataset card: unknown task_type")
    metrics = _metric_names(payload["metrics"], task_type)
    example_count = _example_count(payload["example_count"])
    tag_schema = _tag_schema(payload["tag_schema"])
    label_set = None
    fields = None
    if task_type == "classification":
        label_set = _string_tokens(payload["label_set"], field="label_set")
        if len(label_set) < 2:
            raise DatasetError("dataset card: label_set must contain at least two labels")
    if task_type == "json_field_extraction":
        fields = _field_schema(payload["fields"])

    return DatasetCard(
        name=name,
        version=version,
        task_type=task_type,
        metrics=metrics,
        license_note=license_note,
        construction_notes=construction_notes,
        language=language,
        tag_schema=MappingProxyType(tag_schema),
        example_count=example_count,
        label_set=label_set,
        fields=fields,
    )


def _validate_card_examples(card: DatasetCard, dataset: Dataset) -> None:
    for line, example in enumerate(dataset.examples, start=1):
        _validate_card_example(card, example, line)
    found = len(dataset.examples)
    if found != card.example_count:
        raise DatasetError(
            f"card example_count is {card.example_count} but examples.jsonl has {found}"
        )


def _validate_card_example(card: DatasetCard, example: Example, line: int) -> None:
    _bare_text(example.input, line=line, field="input")
    _bare_text(example.expected, line=line, field="expected")
    if set(example.metadata) != {"tags"}:
        raise DatasetError(f"line {line}: metadata must be an object with only tags", line=line)
    tags = example.metadata["tags"]
    if not isinstance(tags, dict):
        raise DatasetError(f"line {line}: tags must be an object", line=line)
    expected_keys = set(card.tag_schema)
    if set(tags) != expected_keys:
        names = ", ".join(sorted(expected_keys))
        raise DatasetError(f"line {line}: tags keys must be {names}", line=line)
    for key, allowed in card.tag_schema.items():
        value = tags[key]
        if not isinstance(value, str) or value not in allowed:
            choices = ", ".join(allowed)
            raise DatasetError(
                f"line {line}: tag {key} must be one of: {choices}",
                line=line,
            )
    if card.task_type == "classification":
        if card.label_set is None:
            raise DatasetError("classification card is missing label_set")
        if example.expected not in card.label_set:
            raise DatasetError(
                f"line {line}: label {example.expected!r} is not in the label set",
                line=line,
            )
    elif card.task_type == "json_field_extraction":
        if card.fields is None:
            raise DatasetError("json card is missing fields")
        _validate_json_reference(example.expected, card.fields, line)


def _validate_json_reference(
    text: str,
    fields: tuple[tuple[str, str], ...],
    line: int,
) -> None:
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"line {line}: reference is not valid JSON", line=line) from exc
    except ValueError as exc:
        raise DatasetError(f"line {line}: reference {exc}", line=line) from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"line {line}: reference must be a JSON object", line=line)
    names = [name for name, _type_name in fields]
    if list(payload) != names:
        raise DatasetError(f"line {line}: reference fields must be {names}", line=line)
    for name, type_name in fields:
        if not _json_type_ok(payload[name], type_name):
            raise DatasetError(f"line {line}: field {name} must be {type_name}", line=line)
    canonical = json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
    if text != canonical:
        raise DatasetError(f"line {line}: reference JSON must be canonical", line=line)


def _plain_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or value.strip() == "" or value != value.strip():
        raise DatasetError(f"dataset card: {field} must be a non-empty string")
    return value


def _example_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetError("dataset card: example_count must be a positive integer")
    return value


def _metric_names(value: Any, task_type: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetError("dataset card: metrics must be a list of strings")
    expected = TASK_METRICS[task_type]
    if tuple(value) != expected:
        raise DatasetError(f"dataset card: metrics do not match task_type {task_type!r}")
    return expected


def _tag_schema(value: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not value:
        raise DatasetError("dataset card: tag_schema must be a non-empty object")
    schema: dict[str, tuple[str, ...]] = {}
    for key, allowed in value.items():
        if not isinstance(key, str) or key == "" or any(char.isspace() for char in key):
            raise DatasetError(
                "dataset card: tag names must be non-empty strings without whitespace"
            )
        schema[key] = _string_tokens(allowed, field=f"tag_schema.{key}")
    return schema


def _string_tokens(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetError(f"dataset card: {field} must be a non-empty list")
    tokens: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "" or any(char.isspace() for char in item):
            raise DatasetError(
                f"dataset card: {field} must contain non-empty strings without whitespace"
            )
        if item in tokens:
            raise DatasetError(f"dataset card: duplicate value in {field}: {item}")
        tokens.append(item)
    return tuple(tokens)


def _field_schema(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or not value:
        raise DatasetError("dataset card: fields must be a non-empty object")
    fields: list[tuple[str, str]] = []
    for key, type_name in value.items():
        if not isinstance(key, str) or key == "" or any(char.isspace() for char in key):
            raise DatasetError(
                "dataset card: field names must be non-empty strings without whitespace"
            )
        if not isinstance(type_name, str) or type_name not in _JSON_TYPES:
            raise DatasetError(f"dataset card: unknown field type: {type_name!r}")
        fields.append((key, type_name))
    return tuple(fields)


def _bare_text(value: str, *, line: int, field: str) -> None:
    if value.strip() == "" or value != value.strip():
        raise DatasetError(
            f"line {line}: {field} must be a non-empty string without surrounding whitespace",
            line=line,
        )


def _json_type_ok(value: Any, type_name: str) -> bool:
    # bool é subclasse de int; true não passa como integer e 1 não passa como boolean.
    if type_name == "string":
        return isinstance(value, str) and value.strip() != "" and value == value.strip()
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return False
