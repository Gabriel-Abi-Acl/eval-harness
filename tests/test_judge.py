"""Contrato da interface de juiz. O dublê é local; não há chamada de modelo."""

from __future__ import annotations

import pytest

from eval_harness import JudgeMetric
from tests.helpers import make_example


class _ScriptedJudge:
    def __init__(self, value: object) -> None:
        self.value = value

    def judge(self, example: object, output: str) -> object:
        assert output == "candidato"
        return self.value


def test_judge_metric_accepts_a_unit_interval_score() -> None:
    metric = JudgeMetric(_ScriptedJudge(1))
    assert metric.name == "llm_judge"
    assert metric.score(make_example(), "candidato") == 1.0
    renamed = JudgeMetric(_ScriptedJudge(0.25), name="faithfulness")
    assert renamed.score(make_example(), "candidato") == 0.25


@pytest.mark.parametrize("value", [True, False, "yes", 1.1, -0.01, float("nan")])
def test_judge_metric_rejects_bool_and_out_of_range(value: object) -> None:
    metric = JudgeMetric(_ScriptedJudge(value))
    with pytest.raises((TypeError, ValueError)):
        metric.score(make_example(), "candidato")


def test_empty_judge_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        JudgeMetric(_ScriptedJudge(1), name="  ")
