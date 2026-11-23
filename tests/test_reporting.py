"""Contrato do relatório: erro fora da média, média nula e tabela Markdown.

Os números daqui são de fakes e de sistemas de teste. Não são achado de modelo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval_harness import (
    Accuracy,
    ExactMatch,
    Example,
    SystemOutput,
    load_carded_dataset,
    run_eval,
)
from eval_harness.errors import ConfigError
from eval_harness.reporting import (
    build_comparison,
    render_comparison,
    render_report,
    write_comparison,
    write_run,
)

_SCHEMA = "eval_harness.dataset_card.v1"
_STAMP = "2026-01-01T00:00:00+00:00"


class _FailSome:
    name = "fail-some"
    params = {"mode": "fail-some"}

    def __init__(self, fail_ids: set[str]) -> None:
        self._fail_ids = fail_ids

    def __call__(self, example: Example) -> SystemOutput:
        if example.id in self._fail_ids:
            raise RuntimeError("boom\nsecret tail")
        return SystemOutput(output=example.expected)


class _BoomMetric:
    name = "boom"

    def score(self, example: Example, output: str) -> float:
        raise RuntimeError("nope\nsecret tail")


def test_system_errors_are_recorded_and_left_out_of_means(tmp_path: Path) -> None:
    directory = _write_qa(
        tmp_path,
        [_row("qa-01", "fácil"), _row("qa-02", "médio", text="Outra?", expected="Lisboa")],
    )
    carded = load_carded_dataset(directory)
    result = run_eval(
        carded.dataset,
        _FailSome({"qa-02"}),
        [ExactMatch()],
        seed=0,
    )
    out = tmp_path / "run"
    summary = write_run(out, carded.card, carded.dataset, result, created_at=_STAMP)
    assert summary["counts"] == {"examples": 2, "errors": 1, "scored": 1}
    assert summary["metrics"]["exact_match"] == {"mean": 1.0, "count": 1}
    assert summary["by_tag"]["difficulty"]["fácil"]["metrics"]["exact_match"]["mean"] == 1.0
    missed = summary["by_tag"]["difficulty"]["médio"]["metrics"]["exact_match"]
    assert missed == {"mean": None, "count": 0}
    records = (out / "records.jsonl").read_text(encoding="utf-8")
    assert "secret tail" not in records
    assert "secret tail" not in (out / "summary.json").read_text(encoding="utf-8")
    second = json.loads(records.splitlines()[1])
    assert second["error"] == "boom"
    assert second["output"] is None
    assert second["scores"] == {}


def test_all_errors_make_the_mean_null_and_skip_classification(tmp_path: Path) -> None:
    directory = _write_classification(tmp_path, [_class_row("tom-01"), _class_row("tom-02")])
    carded = load_carded_dataset(directory)
    result = run_eval(
        carded.dataset,
        _FailSome({"tom-01", "tom-02"}),
        [ExactMatch()],
        seed=0,
    )
    # O cartão pede accuracy; este teste força a métrica por exemplo e ainda assim
    # não calcula macro quando não houve saída.
    summary = write_run(
        tmp_path / "run",
        carded.card,
        carded.dataset,
        result,
        created_at=_STAMP,
    )
    assert summary["metrics"]["exact_match"]["mean"] is None
    assert summary["classification"] is None
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "| exact_match | n/a | 0 |" in report
    assert "A macro-F1 não foi calculada." in report


def test_metric_error_keeps_scores_already_recorded(tmp_path: Path) -> None:
    directory = _write_qa(tmp_path, [_row("qa-01", "fácil")])
    carded = load_carded_dataset(directory)
    result = run_eval(
        carded.dataset,
        _FailSome(set()),
        [ExactMatch(), _BoomMetric()],
        seed=0,
    )
    summary = write_run(
        tmp_path / "run",
        carded.card,
        carded.dataset,
        result,
        created_at=_STAMP,
    )
    assert summary["metrics"]["exact_match"] == {"mean": 1.0, "count": 1}
    assert summary["metrics"]["boom"] == {"mean": None, "count": 0}
    assert summary["counts"]["errors"] == 1
    record = json.loads((tmp_path / "run" / "records.jsonl").read_text(encoding="utf-8"))
    assert record["error"] == "boom: nope"
    assert record["scores"] == {"exact_match": 1.0}
    assert "secret tail" not in json.dumps(record)


def test_partial_classification_uses_only_scored_rows(tmp_path: Path) -> None:
    directory = _write_classification(
        tmp_path,
        [_class_row("tom-01"), _class_row("tom-02", expected="negativo")],
    )
    carded = load_carded_dataset(directory)
    result = run_eval(
        carded.dataset,
        _FailSome({"tom-02"}),
        [Accuracy()],
        seed=0,
    )
    summary = write_run(
        tmp_path / "run",
        carded.card,
        carded.dataset,
        result,
        created_at=_STAMP,
    )
    assert summary["metrics"]["accuracy"] == {"mean": 1.0, "count": 1}
    assert summary["classification"]["accuracy"] == 1.0
    assert summary["classification"]["macro_f1"] == pytest.approx(0.5)
    labels = [item["label"] for item in summary["classification"]["per_class"]]
    assert labels == ["positivo", "negativo"]


def test_report_table_renders_null_as_na_and_escapes_pipes() -> None:
    summary = _summary(system="oracle", seed=0, mean=None, dataset_hash="a")
    summary["by_tag"]["difficulty"]["a|b"] = summary["by_tag"]["difficulty"].pop("fácil")
    text = render_report(summary)
    assert text.startswith("# Relatório\n")
    assert "| métrica | média | n |" in text
    assert "| --- | ---: | ---: |" in text
    assert "| exact_match | n/a | 0 |" in text
    assert "| a\\|b |" in text
    assert "Números de sistema fake verificam o encanamento. Não são achado experimental." in text
    assert "O repositório não versiona este relatório." in text
    assert "## Classificação" not in text

    summary["manifest"]["system"] = "openai"
    live = render_report(summary)
    assert "Não são achado experimental." not in live
    assert "Este arquivo não é um resultado publicado." in live


def test_compare_delta_is_other_minus_baseline_and_refuses_hash_mismatch(tmp_path: Path) -> None:
    base = _summary(system="oracle", seed=0, mean=1.0, dataset_hash="same", tag_mean=0.5)
    other = _summary(system="echo", seed=0, mean=0.25, dataset_hash="same", tag_mean=0.0)
    comparison = build_comparison(
        [(tmp_path / "oracle", base), (tmp_path / "echo", other)],
        force=False,
    )
    assert comparison["baseline"] == "oracle@0"
    assert comparison["dataset_hashes_match"] is True
    assert comparison["forced"] is False
    row = comparison["metrics"][0]
    assert row["metric"] == "exact_match"
    assert row["values"] == {"oracle@0": 1.0, "echo@0": 0.25}
    assert row["deltas"] == {"echo@0": pytest.approx(-0.75)}
    tag = comparison["by_tag"]["difficulty"]["fácil"]["metrics"][0]
    assert tag["deltas"]["echo@0"] == pytest.approx(-0.5)
    assert "created_at" not in json.dumps(comparison)

    mismatch = _summary(system="echo", seed=1, mean=0.0, dataset_hash="other")
    with pytest.raises(ConfigError, match="No comparison was written"):
        build_comparison(
            [(tmp_path / "oracle", base), (tmp_path / "other", mismatch)],
            force=False,
        )
    forced = build_comparison(
        [(tmp_path / "oracle", base), (tmp_path / "other", mismatch)],
        force=True,
    )
    assert forced["dataset_hashes_match"] is False
    assert forced["dataset_hash"] is None
    assert forced["forced"] is True
    assert forced["metrics"][0]["deltas"]["echo@1"] == pytest.approx(-1.0)
    text = render_comparison(forced)
    assert "A comparação foi forçada" in text
    assert "| métrica | oracle@0 | echo@1 | delta echo@1 |" in text
    assert "| --- | ---: | ---: | ---: |" in text


def test_compare_disambiguates_labels_and_is_byte_stable(tmp_path: Path) -> None:
    base = _summary(system="oracle", seed=0, mean=1.0, dataset_hash="same")
    again = _summary(system="oracle", seed=0, mean=None, dataset_hash="same")
    comparison = build_comparison(
        [(tmp_path / "one", base), (tmp_path / "two", again)],
        force=False,
    )
    assert [run["label"] for run in comparison["runs"]] == ["oracle@0", "oracle@0#2"]
    assert comparison["metrics"][0]["deltas"]["oracle@0#2"] is None
    out = tmp_path / "cmp"
    write_comparison(out, comparison)
    first = (out / "comparison.json").read_text(encoding="utf-8")
    write_comparison(out, comparison)
    second = (out / "comparison.json").read_text(encoding="utf-8")
    assert first == second
    assert first == json.dumps(json.loads(first), ensure_ascii=False, separators=(",", ":")) + "\n"
    assert "created_at" not in first
    markdown = (out / "comparison.md").read_text(encoding="utf-8")
    assert "n/a" in markdown


def _summary(
    *,
    system: str,
    seed: int,
    mean: float | None,
    dataset_hash: str,
    tag_mean: float | None | object = ...,
) -> dict[str, Any]:
    if tag_mean is ...:
        tag_mean = mean
    return {
        "schema": "eval_harness.summary.v1",
        "dataset": {
            "name": "Mini",
            "version": "1.0.0",
            "task_type": "short_answer_qa",
            "language": "pt-BR",
        },
        "manifest": {
            "dataset_hash": dataset_hash,
            "config_hash": "config",
            "seed": seed,
            "versions": {"eval_harness": "0.1.0", "python": "3.11.0"},
            "system": system,
            "metrics": ["exact_match"],
            "example_count": 1,
            "created_at": _STAMP,
        },
        "counts": {"examples": 1, "errors": 0, "scored": 1},
        "metrics": {"exact_match": {"mean": mean, "count": 0 if mean is None else 1}},
        "by_tag": {
            "difficulty": {
                "fácil": {
                    "count": 1,
                    "errors": 0,
                    "metrics": {
                        "exact_match": {"mean": tag_mean, "count": 0 if tag_mean is None else 1}
                    },
                }
            }
        },
        "classification": None,
    }


def _write_qa(directory: Path, rows: list[dict[str, Any]]) -> Path:
    card = {
        "schema": _SCHEMA,
        "name": "conjunto teste",
        "version": "1.0.0",
        "task_type": "short_answer_qa",
        "metrics": ["exact_match", "normalized_match", "token_f1"],
        "license_note": "sintético de teste",
        "construction_notes": "escrito por um autor para o teste",
        "language": "pt-BR",
        "tag_schema": {"difficulty": ["fácil", "médio"], "category": ["geografia"]},
        "example_count": len(rows),
    }
    return _write_card(directory, card, rows)


def _write_classification(directory: Path, rows: list[dict[str, Any]]) -> Path:
    card = {
        "schema": _SCHEMA,
        "name": "classe teste",
        "version": "1.0.0",
        "task_type": "classification",
        "metrics": ["accuracy", "per_class_precision_recall", "macro_f1"],
        "license_note": "sintético de teste",
        "construction_notes": "escrito por um autor para o teste",
        "language": "pt-BR",
        "tag_schema": {"difficulty": ["fácil"], "category": ["produto"]},
        "example_count": len(rows),
        "label_set": ["positivo", "negativo"],
    }
    return _write_card(directory, card, rows)


def _write_card(directory: Path, card: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    target = directory / "set"
    target.mkdir()
    (target / "card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    (target / "examples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _row(
    example_id: str,
    difficulty: str,
    *,
    text: str = "Capital?",
    expected: str = "Brasília",
) -> dict[str, Any]:
    return {
        "id": example_id,
        "input": text,
        "expected": expected,
        "metadata": {"tags": {"difficulty": difficulty, "category": "geografia"}},
    }


def _class_row(example_id: str, *, expected: str = "positivo") -> dict[str, Any]:
    return {
        "id": example_id,
        "input": "A caneca nova é linda.",
        "expected": expected,
        "metadata": {"tags": {"difficulty": "fácil", "category": "produto"}},
    }
