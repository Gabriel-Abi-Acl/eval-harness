"""Conjuntos versionados e cartão. Contagens saem dos arquivos, sem nota de modelo."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from eval_harness import JsonFieldAccuracy, load_carded_dataset, score_classification
from eval_harness.dataset import TASK_METRICS
from eval_harness.errors import DatasetError
from tests.helpers import ROOT

EVAL = ROOT / "data" / "eval"
README = ROOT / "README.md"

_SHIPPED = (
    ("classificacao_tom", "tom", "classification", "Classificação de tom sintética"),
    ("extracao_pedido", "ped", "json_field_extraction", "Extração de pedido sintética"),
    ("qa_curta", "qa", "short_answer_qa", "QA curta sintética"),
)
_HASHES = {
    "classificacao_tom": "f7e93cac9c014f8241f47611b1604d03878f48ec391339766c1760536fe433bd",
    "extracao_pedido": "3b612a19ab4f1615027dfa9d23007f88cfd3c4013050fc4b44330d161418e506",
    "qa_curta": "559ae0295547a837273d5a8d10591e56e39d96bfc63901354885ebc22acb30b6",
}
_SCHEMA = "eval_harness.dataset_card.v1"


def _loaded() -> list[tuple[str, Any]]:
    return [(slug, load_carded_dataset(EVAL / slug)) for slug, *_rest in _SHIPPED]


def _tag_counts(carded: Any, tag_name: str) -> str:
    allowed = carded.card.tag_schema[tag_name]
    counts = Counter(example.metadata["tags"][tag_name] for example in carded.dataset.examples)
    return ", ".join(f"{value} {counts[value]}" for value in allowed)


def observed_count_rows() -> list[str]:
    """Linhas da tabela de contagem, na ordem dos conjuntos."""
    rows: list[str] = []
    for slug, carded in _loaded():
        difficulty = _tag_counts(carded, "difficulty")
        category = _tag_counts(carded, "category")
        count = carded.card.example_count
        rows.append(f"| {slug} | {count} | {difficulty} | {category} |")
    return rows


def observed_gold_rows() -> list[str]:
    """Distribuição do gabarito. QA curta não tem rótulo fechado."""
    rows: list[str] = []
    for slug, carded in _loaded():
        if carded.card.label_set is not None:
            counts = Counter(example.expected for example in carded.dataset.examples)
            rendered = ", ".join(f"{label} {counts[label]}" for label in carded.card.label_set)
            rows.append(f"| {slug} | {rendered} |")
        elif carded.card.task_type == "json_field_extraction":
            flags = Counter(
                json.loads(example.expected)["urgente"] for example in carded.dataset.examples
            )
            rendered = f"urgente true {flags[True]}, urgente false {flags[False]}"
            rows.append(f"| {slug} | {rendered} |")
    return rows


def observed_total_sentence() -> str:
    counts = [carded.card.example_count for _slug, carded in _loaded()]
    if len(set(counts)) == 1:
        return f"São {sum(counts)} exemplos no total, {counts[0]} em cada conjunto."
    rendered = ", ".join(str(count) for count in counts)
    return f"São {sum(counts)} exemplos no total ({rendered})."


def test_eval_directory_has_only_the_three_sets() -> None:
    directories = sorted(path.name for path in EVAL.iterdir() if path.is_dir())
    assert directories == [slug for slug, *_rest in _SHIPPED]
    files = sorted(path.relative_to(EVAL).as_posix() for path in EVAL.rglob("*") if path.is_file())
    assert files == [
        f"{slug}/{name}" for slug, *_rest in _SHIPPED for name in ("card.json", "examples.jsonl")
    ]


def test_shipped_cards_match_examples_and_hashes() -> None:
    by_slug = dict(_loaded())
    for slug, prefix, task_type, name in _SHIPPED:
        carded = by_slug[slug]
        card = carded.card
        examples = carded.dataset.examples
        assert card.name == name
        assert card.version == "1.0.0"
        assert card.language == "pt-BR"
        assert card.task_type == task_type
        assert card.metrics == TASK_METRICS[task_type]
        assert "sintéticos" in card.license_note
        assert "Uso livre" in card.license_note
        assert "autor" in card.construction_notes
        assert list(card.tag_schema) == ["difficulty", "category"]
        assert card.tag_schema["difficulty"] == ("fácil", "médio", "difícil")
        assert 15 <= card.example_count <= 20
        assert card.example_count == len(examples)
        assert [example.id for example in examples] == [
            f"{prefix}-{index:02d}" for index in range(1, len(examples) + 1)
        ]
        assert len({example.input for example in examples}) == len(examples)
        assert all(example.input != example.expected for example in examples)
        assert all(set(example.metadata) == {"tags"} for example in examples)
        for tag_name, allowed in card.tag_schema.items():
            seen = {example.metadata["tags"][tag_name] for example in examples}
            assert seen == set(allowed)
        assert carded.dataset.content_hash == _HASHES[slug]
        assert load_carded_dataset(EVAL / slug).dataset.content_hash == _HASHES[slug]


def test_qa_targets_string_metrics() -> None:
    carded = load_carded_dataset(EVAL / "qa_curta")
    assert carded.card.label_set is None
    assert carded.card.fields is None
    assert any(" " in example.expected for example in carded.dataset.examples)


def test_classification_labels_stay_inside_the_set() -> None:
    carded = load_carded_dataset(EVAL / "classificacao_tom")
    labels = carded.card.label_set
    assert labels == ("positivo", "negativo", "neutro")
    gold = [example.expected for example in carded.dataset.examples]
    assert set(gold) == set(labels)
    perfect = score_classification(gold, gold, labels)
    assert perfect.accuracy == 1.0
    assert perfect.macro_f1 == 1.0
    flipped = ["neutro" if label == "positivo" else label for label in gold]
    changed = score_classification(gold, flipped, labels)
    assert changed.accuracy < 1.0
    assert changed.macro_f1 < 1.0


def test_extraction_references_match_declared_fields() -> None:
    carded = load_carded_dataset(EVAL / "extracao_pedido")
    assert carded.card.fields == (
        ("produto", "string"),
        ("quantidade", "integer"),
        ("urgente", "boolean"),
    )
    metric = JsonFieldAccuracy()
    quantities: set[int] = set()
    flags: set[bool] = set()
    for example in carded.dataset.examples:
        payload = json.loads(example.expected)
        assert metric.score(example, example.expected) == 1.0
        assert payload["produto"] == payload["produto"].casefold()
        assert payload["quantidade"] >= 1
        quantities.add(payload["quantidade"])
        flags.add(payload["urgente"])
    assert len(quantities) >= 2
    assert flags == {True, False}
    first = carded.dataset.examples[0]
    flipped = first.expected.replace("false", "true", 1)
    assert metric.score(first, flipped) == pytest.approx(2 / 3)


def test_readme_reports_the_observed_counts() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## Datasets de exemplo" in text
    assert observed_total_sentence() in text
    for row in observed_count_rows():
        assert row in text
    for row in observed_gold_rows():
        assert row in text
    assert "Nenhum modelo foi executado" in text
    assert "docs/spec-pr2-datasets.md" in text


def test_valid_minimal_card_loads(tmp_path: Path) -> None:
    directory = _write(tmp_path, _qa_card(), [_qa_row()])
    carded = load_carded_dataset(directory)
    assert carded.card.example_count == 1
    assert carded.dataset.examples[0].id == "qa-01"
    assert carded.dataset.examples[0].metadata["tags"]["difficulty"] == "fácil"


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="dataset directory not found"):
        load_carded_dataset(tmp_path / "ausente")


def test_missing_card(tmp_path: Path) -> None:
    directory = tmp_path / "set"
    directory.mkdir()
    with pytest.raises(DatasetError, match="could not read dataset card"):
        load_carded_dataset(directory)


def test_missing_examples_file(tmp_path: Path) -> None:
    directory = tmp_path / "set"
    directory.mkdir()
    (directory / "card.json").write_text(
        json.dumps(_qa_card(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="could not read dataset"):
        load_carded_dataset(directory)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("[", "not valid JSON"),
        ("[]\n", "must be a JSON object"),
        (
            '{"schema": "eval_harness.dataset_card.v1", "schema": "outro"}\n',
            "duplicate key: schema",
        ),
    ],
)
def test_card_file_must_be_one_object(tmp_path: Path, raw: str, match: str) -> None:
    directory = tmp_path / "set"
    directory.mkdir()
    (directory / "card.json").write_text(raw, encoding="utf-8")
    (directory / "examples.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match=match):
        load_carded_dataset(directory)


def test_card_rejects_bad_utf8(tmp_path: Path) -> None:
    directory = tmp_path / "set"
    directory.mkdir()
    (directory / "card.json").write_bytes(b"\xff")
    with pytest.raises(DatasetError, match="not valid UTF-8"):
        load_carded_dataset(directory)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda card: card.__setitem__("note", "x"), "unknown field: note"),
        (lambda card: card.pop("version"), "missing field: version"),
        (lambda card: card.__setitem__("schema", "outro"), f"schema must be {_SCHEMA}"),
        (lambda card: card.__setitem__("version", "1.0"), "major.minor.patch"),
        (lambda card: card.__setitem__("task_type", "resumo"), "unknown task_type"),
        (
            lambda card: card.__setitem__(
                "metrics",
                ["token_f1", "exact_match", "normalized_match"],
            ),
            "metrics do not match task_type 'short_answer_qa'",
        ),
        (lambda card: card.__setitem__("metrics", "exact_match"), "metrics must be a list"),
        (lambda card: card.__setitem__("example_count", True), "positive integer"),
        (lambda card: card.__setitem__("example_count", 0), "positive integer"),
        (lambda card: card.__setitem__("label_set", ["a", "b"]), "unknown field: label_set"),
        (
            lambda card: card.__setitem__(
                "tag_schema",
                {"difficulty": ["fácil", "fácil"], "category": ["geografia"]},
            ),
            "duplicate value in tag_schema.difficulty",
        ),
        (lambda card: card.__setitem__("tag_schema", {}), "tag_schema must be a non-empty object"),
        (lambda card: card.__setitem__("name", "  "), "name must be a non-empty string"),
    ],
)
def test_card_schema_is_strict(tmp_path: Path, mutate: Any, match: str) -> None:
    card = _qa_card()
    mutate(card)
    directory = _write(tmp_path, card, [_qa_row()])
    with pytest.raises(DatasetError, match=match):
        load_carded_dataset(directory)


def test_example_count_must_match_the_file(tmp_path: Path) -> None:
    card = _qa_card()
    second = _qa_row()
    second["id"] = "qa-02"
    directory = _write(tmp_path, card, [_qa_row(), second])
    with pytest.raises(DatasetError, match="example_count is 1 but examples.jsonl has 2"):
        load_carded_dataset(directory)


def test_duplicate_id_rejects_the_directory(tmp_path: Path) -> None:
    card = _qa_card()
    card["example_count"] = 2
    directory = _write(tmp_path, card, [_qa_row(), _qa_row()])
    with pytest.raises(DatasetError, match=r"line 2: duplicate id 'qa-01'") as caught:
        load_carded_dataset(directory)
    assert caught.value.line == 2


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: row.__setitem__("input", "  "), "input must be a non-empty"),
        (lambda row: row.__setitem__("expected", ""), "expected must be a non-empty"),
        (
            lambda row: row.__setitem__(
                "metadata",
                {"tags": {"difficulty": "fácil"}, "origin": "x"},
            ),
            "metadata must be an object with only tags",
        ),
        (
            lambda row: row.__setitem__(
                "metadata",
                {"tags": {"difficulty": "easy", "category": "geografia"}},
            ),
            "tag difficulty must be one of",
        ),
        (lambda row: row.__setitem__("metadata", {"tags": []}), "tags must be an object"),
        (lambda row: row.__setitem__("metadata", {}), "only tags"),
    ],
)
def test_card_checks_each_example(tmp_path: Path, mutate: Any, match: str) -> None:
    row = _qa_row()
    mutate(row)
    directory = _write(tmp_path, _qa_card(), [row])
    with pytest.raises(DatasetError, match=match) as caught:
        load_carded_dataset(directory)
    assert caught.value.line == 1


def test_classification_card_requires_a_closed_label_set(tmp_path: Path) -> None:
    card = _classification_card()
    card.pop("label_set")
    directory = _write(tmp_path, card, [_classification_row()])
    with pytest.raises(DatasetError, match="missing field: label_set"):
        load_carded_dataset(directory)


def test_classification_label_must_belong_to_the_set(tmp_path: Path) -> None:
    row = _classification_row()
    row["expected"] = "talvez"
    directory = _write(tmp_path, _classification_card(), [row])
    with pytest.raises(DatasetError, match="label 'talvez' is not in the label set") as caught:
        load_carded_dataset(directory)
    assert caught.value.line == 1


def test_label_set_needs_two_distinct_labels(tmp_path: Path) -> None:
    card = _classification_card()
    card["label_set"] = ["positivo"]
    directory = _write(tmp_path, card, [_classification_row()])
    with pytest.raises(DatasetError, match="at least two labels"):
        load_carded_dataset(directory)


@pytest.mark.parametrize(
    ("expected", "match"),
    [
        ("não é json", "not valid JSON"),
        ("[]", "must be a JSON object"),
        (
            '{"produto": "caderno", "quantidade": 1.0, "urgente": false}',
            "field quantidade must be integer",
        ),
        (
            '{"produto": "caderno", "quantidade": 1, "urgente": 1}',
            "field urgente must be boolean",
        ),
        (
            '{"produto": "", "quantidade": 1, "urgente": false}',
            "field produto must be string",
        ),
        (
            '{"produto": "caderno", "quantidade": 1, "urgente": false, "extra": 1}',
            "reference fields must be",
        ),
        (
            '{"quantidade": 1, "produto": "caderno", "urgente": false}',
            "reference fields must be",
        ),
        (
            '{"produto":"caderno", "quantidade": 1, "urgente": false}',
            "reference JSON must be canonical",
        ),
        (
            '{"produto": "caderno", "produto": "outro", "quantidade": 1, "urgente": false}',
            "duplicate key: produto",
        ),
    ],
)
def test_json_references_are_strict(tmp_path: Path, expected: str, match: str) -> None:
    row = _json_row()
    row["expected"] = expected
    directory = _write(tmp_path, _json_card(), [row])
    with pytest.raises(DatasetError, match=match) as caught:
        load_carded_dataset(directory)
    assert caught.value.line == 1


def test_json_card_requires_fields(tmp_path: Path) -> None:
    card = _json_card()
    card.pop("fields")
    directory = _write(tmp_path, card, [_json_row()])
    with pytest.raises(DatasetError, match="missing field: fields"):
        load_carded_dataset(directory)


def test_unknown_json_field_type(tmp_path: Path) -> None:
    card = _json_card()
    card["fields"] = {"produto": "number"}
    directory = _write(tmp_path, card, [_json_row()])
    with pytest.raises(DatasetError, match="unknown field type: 'number'"):
        load_carded_dataset(directory)


def _write(directory: Path, card: Mapping[str, Any], rows: list[dict[str, Any]]) -> Path:
    target = directory / "set"
    target.mkdir()
    (target / "card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    (target / "examples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _qa_card() -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "name": "conjunto teste",
        "version": "1.0.0",
        "task_type": "short_answer_qa",
        "metrics": ["exact_match", "normalized_match", "token_f1"],
        "license_note": "sintético de teste",
        "construction_notes": "escrito por um autor para o teste",
        "language": "pt-BR",
        "tag_schema": {
            "difficulty": ["fácil", "médio"],
            "category": ["geografia"],
        },
        "example_count": 1,
    }


def _qa_row() -> dict[str, Any]:
    return {
        "id": "qa-01",
        "input": "Capital?",
        "expected": "Brasília",
        "metadata": {"tags": {"difficulty": "fácil", "category": "geografia"}},
    }


def _classification_card() -> dict[str, Any]:
    card = _qa_card()
    card["task_type"] = "classification"
    card["metrics"] = ["accuracy", "per_class_precision_recall", "macro_f1"]
    card["label_set"] = ["positivo", "negativo"]
    return card


def _classification_row() -> dict[str, Any]:
    row = _qa_row()
    row["expected"] = "positivo"
    return row


def _json_card() -> dict[str, Any]:
    card = _qa_card()
    card["task_type"] = "json_field_extraction"
    card["metrics"] = ["json_field_accuracy"]
    card["fields"] = {"produto": "string", "quantidade": "integer", "urgente": "boolean"}
    return card


def _json_row() -> dict[str, Any]:
    row = _qa_row()
    row["expected"] = '{"produto": "caderno", "quantidade": 1, "urgente": false}'
    return row
