"""Contrato do runner e do manifesto. O fake devolve o gabarito só para checar o encanamento."""

from __future__ import annotations

import math

import pytest

from eval_harness import (
    ExactMatch,
    Example,
    FakeSystem,
    NormalizedMatch,
    SystemOutput,
    TokenF1,
    Usage,
    load_dataset,
    run_eval,
)
from eval_harness.errors import ConfigError
from eval_harness.hashing import digest
from eval_harness.version import __version__
from tests.helpers import PLACEHOLDER, write_jsonl


class _Boom:
    name = "boom"
    params: dict[str, str] = {"mode": "boom"}

    def __call__(self, example: Example) -> SystemOutput:
        if example.id == "ph-02":
            raise RuntimeError("scripted failure\nsecret-looking tail")
        return SystemOutput(output="ok")


class _BoolMetric:
    name = "bad"

    def score(self, example: object, output: str) -> bool:
        return True


def _placeholder_run(seed: int = 0, *, metrics: list[object] | None = None):
    dataset = load_dataset(PLACEHOLDER)
    system = FakeSystem(
        "lookup",
        table={example.id: example.expected for example in dataset.examples},
    )
    chosen = metrics if metrics is not None else [ExactMatch(), NormalizedMatch(), TokenF1()]
    return run_eval(dataset, system, chosen, seed=seed)


def test_lookup_pipeline_records_output_scores_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_HARNESS_SEED", "9")
    result = _placeholder_run(seed=1)
    assert result.manifest.seed == 1
    assert result.manifest.example_count == 8
    assert result.manifest.system == "fake"
    assert result.manifest.metrics == ("exact_match", "normalized_match", "token_f1")
    assert result.manifest.versions["eval_harness"] == __version__
    assert result.manifest.versions["python"]
    assert len(result.results) == 8
    for example, row in zip(load_dataset(PLACEHOLDER).examples, result.results, strict=True):
        assert row.example_id == example.id
        assert row.output == example.expected
        assert row.error is None
        assert row.scores["exact_match"] == 1.0
        assert math.isfinite(row.latency_ms) and row.latency_ms >= 0.0


def test_same_inputs_repeat_manifest_and_scores() -> None:
    first = _placeholder_run(seed=7)
    second = _placeholder_run(seed=7)
    assert first.manifest.dataset_hash == second.manifest.dataset_hash
    assert first.manifest.config_hash == second.manifest.config_hash
    assert [row.output for row in first.results] == [row.output for row in second.results]
    first_scores = [dict(row.scores) for row in first.results]
    second_scores = [dict(row.scores) for row in second.results]
    assert first_scores == second_scores
    assert [row.error for row in first.results] == [row.error for row in second.results]


def test_seed_changes_config_hash_and_not_dataset_hash() -> None:
    low = _placeholder_run(seed=1)
    high = _placeholder_run(seed=2)
    assert low.manifest.dataset_hash == high.manifest.dataset_hash
    assert low.manifest.config_hash != high.manifest.config_hash


def test_config_hash_covers_seed_system_metrics_and_params() -> None:
    result = _placeholder_run(seed=3, metrics=[ExactMatch()])
    dataset = load_dataset(PLACEHOLDER)
    assert result.manifest.dataset_hash == dataset.content_hash
    assert result.manifest.config_hash == digest(
        {
            "seed": 3,
            "system": "fake",
            "metrics": ["exact_match"],
            "system_params": {
                "mode": "lookup",
                "table": {example.id: example.expected for example in dataset.examples},
            },
        }
    )


def test_metric_order_changes_config_hash() -> None:
    forward = _placeholder_run(metrics=[ExactMatch(), TokenF1()])
    backward = _placeholder_run(metrics=[TokenF1(), ExactMatch()])
    assert forward.manifest.config_hash != backward.manifest.config_hash


def test_system_error_does_not_stop_the_run() -> None:
    dataset = load_dataset(PLACEHOLDER)
    result = run_eval(dataset, _Boom(), [ExactMatch()], seed=0)
    by_id = {row.example_id: row for row in result.results}
    failed = by_id["ph-02"]
    assert failed.output is None
    assert failed.scores == {}
    assert failed.error == "scripted failure"
    assert failed.usage is None
    assert by_id["ph-01"].output == "ok"
    assert by_id["ph-01"].scores["exact_match"] == 0.0
    assert by_id["ph-03"].error is None


def test_metric_error_keeps_output_and_stops_later_metrics() -> None:
    dataset = load_dataset(PLACEHOLDER)
    system = FakeSystem("echo")
    result = run_eval(dataset, system, [ExactMatch(), _BoolMetric(), TokenF1()], seed=0)
    row = result.results[0]
    assert row.output == dataset.examples[0].input
    assert set(row.scores) == {"exact_match"}
    assert row.error is not None
    assert row.error.startswith("bad:")


def test_plain_callable_is_a_system(tmp_path) -> None:
    path = write_jsonl(
        tmp_path,
        [{"id": "ex-01", "input": "pergunta", "expected": "resposta"}],
    )
    dataset = load_dataset(path)

    def echo(example):
        return SystemOutput(output=example.input, usage=None)

    result = run_eval(dataset, echo, [], seed=0)
    assert result.manifest.system == "echo"
    assert result.manifest.metrics == ()
    assert result.results[0].output == "pergunta"
    assert result.results[0].scores == {}
    assert result.results[0].error is None


def test_scripted_short_tape_records_the_exhausted_example(tmp_path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            {"id": "ex-01", "input": "a", "expected": "um"},
            {"id": "ex-02", "input": "b", "expected": "dois"},
        ],
    )
    dataset = load_dataset(path)
    system = FakeSystem("scripted", script=["um"])
    result = run_eval(dataset, system, [ExactMatch()], seed=0)
    assert result.results[0].output == "um"
    assert result.results[0].scores["exact_match"] == 1.0
    assert result.results[1].output is None
    assert result.results[1].error == "scripted outputs are exhausted"


def test_secret_param_is_refused_without_echoing_the_value(tmp_path) -> None:
    path = write_jsonl(tmp_path, [{"id": "ex-01", "input": "a", "expected": "b"}])
    dataset = load_dataset(path)

    class _Leaky:
        name = "leaky"
        params = {"api_key": "super-secret-value"}

        def __call__(self, example):
            return SystemOutput(output="b")

    with pytest.raises(ConfigError, match="api_key") as caught:
        run_eval(dataset, _Leaky(), [], seed=0)
    assert "super-secret-value" not in str(caught.value)


@pytest.mark.parametrize("seed", [-1, True, "0", 1.5])
def test_seed_must_be_a_non_negative_integer(seed: object) -> None:
    dataset = load_dataset(PLACEHOLDER)
    with pytest.raises(ConfigError, match="seed must be"):
        run_eval(dataset, FakeSystem(), [], seed=seed)  # type: ignore[arg-type]


def test_duplicate_metric_name_is_rejected() -> None:
    dataset = load_dataset(PLACEHOLDER)
    with pytest.raises(ConfigError, match="duplicate metric name"):
        run_eval(dataset, FakeSystem(), [ExactMatch(), ExactMatch()], seed=0)


def test_usage_is_copied_onto_the_row() -> None:
    dataset = load_dataset(PLACEHOLDER)
    usage = Usage(input_tokens=4, output_tokens=1)
    result = run_eval(dataset, FakeSystem(usage=usage), [], seed=0)
    assert result.results[0].usage == usage
