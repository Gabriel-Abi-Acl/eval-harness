"""Contrato do loader: schema fechado e hash estável. Não mede sistema."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness import load_dataset
from eval_harness.errors import DatasetError
from eval_harness.hashing import digest
from tests.helpers import PLACEHOLDER, write_jsonl

_ROW = {"id": "ex-01", "input": "pergunta", "expected": "resposta"}


def test_placeholder_has_eight_labeled_lines() -> None:
    dataset = load_dataset(PLACEHOLDER)
    ids = [example.id for example in dataset.examples]
    assert ids == [f"ph-{index:02d}" for index in range(1, 9)]
    assert dataset.examples[0].metadata["origin"] == "placeholder"
    assert dataset.examples[1].metadata == {}
    assert all(example.expected != "" for example in dataset.examples)


def test_key_order_and_whitespace_do_not_change_hash(tmp_path: Path) -> None:
    compact = write_jsonl(tmp_path, [_ROW], name="compact.jsonl")
    spaced = tmp_path / "spaced.jsonl"
    spaced.write_text(
        '{ "expected" : "resposta", "input" : "pergunta", "id" : "ex-01" }\n',
        encoding="utf-8",
    )
    first = load_dataset(compact)
    second = load_dataset(spaced)
    assert first.content_hash == second.content_hash
    assert first.content_hash == digest(
        {
            "schema": "eval_harness.example.v1",
            "examples": [
                {
                    "id": "ex-01",
                    "input": "pergunta",
                    "expected": "resposta",
                    "metadata": {},
                }
            ],
        }
    )


def test_content_change_changes_hash(tmp_path: Path) -> None:
    original = load_dataset(write_jsonl(tmp_path, [_ROW], name="original.jsonl"))
    changed = load_dataset(
        write_jsonl(tmp_path, [{**_ROW, "expected": "outra"}], name="changed.jsonl")
    )
    assert original.content_hash != changed.content_hash


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "no examples"),
        ("\n", "line 1: blank line"),
        ('{"id": "ex-01", "input": "pergunta"}\n', "missing field: expected"),
        (
            '{"id": "ex-01", "input": "pergunta", "expected": "resposta", "note": "x"}\n',
            "unknown field: note",
        ),
        ('{"id": "ex-01", "input": "pergunta", "expected": 1}\n', "expected must be a string"),
        ('{"id": "", "input": "pergunta", "expected": "resposta"}\n', "id must be a non-empty"),
        ('{"id": "ex 01", "input": "pergunta", "expected": "resposta"}\n', "without whitespace"),
        (
            '{"id": "ex-01", "input": "pergunta", "expected": "resposta", "metadata": []}\n',
            "metadata must be an object",
        ),
        ("not-json\n", "line 1: invalid JSON"),
        ("[]\n", "expected a JSON object"),
    ],
)
def test_strict_schema_rejects_the_file(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(DatasetError, match=match) as caught:
        load_dataset(path)
    if "line 1:" in match or match.startswith("line"):
        assert caught.value.line == 1


def test_duplicate_id_reports_both_lines(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path, [_ROW, {**_ROW, "input": "outra"}])
    with pytest.raises(DatasetError, match=r"line 2: duplicate id 'ex-01'") as caught:
        load_dataset(path)
    assert caught.value.line == 2


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"id": "ex-01", "id": "ex-02", "input": "pergunta", "expected": "resposta"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="line 1: duplicate key: id") as caught:
        load_dataset(path)
    assert caught.value.line == 1


def test_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(DatasetError, match="not valid UTF-8"):
        load_dataset(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="could not read dataset"):
        load_dataset(tmp_path / "missing.jsonl")


def test_metadata_round_trip(tmp_path: Path) -> None:
    row = {**_ROW, "metadata": {"split": "dev", "n": 1}}
    dataset = load_dataset(write_jsonl(tmp_path, [row]))
    assert dict(dataset.examples[0].metadata) == {"split": "dev", "n": 1}
    assert dataset.examples[0].expected == "resposta"
