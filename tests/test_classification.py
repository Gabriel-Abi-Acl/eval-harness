"""Acurácia e métricas de conjunto. Pares escritos no teste, sem modelo."""

from __future__ import annotations

import pytest

from eval_harness import Accuracy, FakeSystem, load_dataset, run_eval, score_classification
from tests.helpers import make_example, write_jsonl

_ACCURACY = Accuracy()


def test_accuracy_is_exact_label_equality() -> None:
    example = make_example(expected="positivo")
    assert _ACCURACY.name == "accuracy"
    assert _ACCURACY.score(example, "positivo") == 1.0
    assert _ACCURACY.score(example, "Positivo") == 0.0
    assert _ACCURACY.score(example, " positivo") == 0.0
    assert _ACCURACY.score(example, "negativo") == 0.0


def test_accuracy_rejects_non_string_output() -> None:
    with pytest.raises(TypeError, match="output must be a string"):
        _ACCURACY.score(make_example(), None)  # type: ignore[arg-type]


def test_accuracy_runs_as_a_per_example_metric(tmp_path) -> None:
    path = write_jsonl(
        tmp_path,
        [
            {"id": "ex-01", "input": "frase", "expected": "positivo"},
            {"id": "ex-02", "input": "outra", "expected": "negativo"},
        ],
    )
    dataset = load_dataset(path)
    system = FakeSystem("lookup", table={"ex-01": "positivo", "ex-02": "positivo"})
    result = run_eval(dataset, system, [_ACCURACY], seed=0)
    assert result.manifest.metrics == ("accuracy",)
    assert result.results[0].scores["accuracy"] == 1.0
    assert result.results[1].scores["accuracy"] == 0.0
    assert result.results[0].error is None


def test_perfect_predictions_score_one() -> None:
    gold = ["positivo", "negativo", "neutro"]
    score = score_classification(gold, list(gold), ["positivo", "negativo", "neutro"])
    assert score.accuracy == 1.0
    assert score.macro_f1 == 1.0
    assert [item.label for item in score.per_class] == ["positivo", "negativo", "neutro"]
    assert all(item.precision == 1.0 and item.recall == 1.0 for item in score.per_class)
    assert all(item.support == 1 for item in score.per_class)


def test_partial_overlap_sets_precision_recall_and_macro_f1() -> None:
    # gold:      gato, gato, cachorro, cachorro
    # predicted: gato, cachorro, cachorro, cachorro
    # gato: tp 1, fp 0, fn 1 -> precisão 1, recall 1/2, F1 2/3
    # cachorro: tp 2, fp 1, fn 0 -> precisão 2/3, recall 1, F1 4/5
    # macro-F1 = (2/3 + 4/5) / 2 = 11/15; acurácia = 3/4
    score = score_classification(
        ["gato", "gato", "cachorro", "cachorro"],
        ["gato", "cachorro", "cachorro", "cachorro"],
        ["gato", "cachorro"],
    )
    gato, cachorro = score.per_class
    assert score.accuracy == 0.75
    assert gato.precision == 1.0
    assert gato.recall == pytest.approx(1 / 2)
    assert gato.f1 == pytest.approx(2 / 3)
    assert gato.support == 2
    assert cachorro.precision == pytest.approx(2 / 3)
    assert cachorro.recall == 1.0
    assert cachorro.f1 == pytest.approx(4 / 5)
    assert cachorro.support == 2
    assert score.macro_f1 == pytest.approx(11 / 15)


def test_report_follows_the_given_label_order() -> None:
    score = score_classification(["b"], ["b"], ["b", "a"])
    assert [item.label for item in score.per_class] == ["b", "a"]


def test_unused_label_counts_as_zero_in_macro_f1() -> None:
    score = score_classification(["a", "a"], ["a", "a"], ["a", "b"])
    unused = score.per_class[1]
    assert score.accuracy == 1.0
    assert unused.support == 0
    assert unused.precision == 0.0
    assert unused.recall == 0.0
    assert unused.f1 == 0.0
    assert score.macro_f1 == 0.5


def test_prediction_outside_the_label_set_adds_no_class() -> None:
    score = score_classification(["sim"], ["talvez"], ["sim", "nao"])
    sim, nao = score.per_class
    assert score.accuracy == 0.0
    assert [item.label for item in score.per_class] == ["sim", "nao"]
    assert sim.support == 1
    assert sim.precision == 0.0
    assert sim.recall == 0.0
    assert nao.support == 0
    assert nao.precision == 0.0
    assert score.macro_f1 == 0.0


def test_false_positive_on_an_unused_label() -> None:
    score = score_classification(["nao"], ["sim"], ["sim", "nao"])
    sim, nao = score.per_class
    assert score.accuracy == 0.0
    assert sim.support == 0
    assert sim.precision == 0.0
    assert sim.recall == 0.0
    assert nao.support == 1
    assert nao.recall == 0.0
    assert score.macro_f1 == 0.0


def test_single_label_set_still_scores() -> None:
    score = score_classification(["sim"], ["sim"], ["sim"])
    assert score.accuracy == 1.0
    assert score.macro_f1 == 1.0
    assert score.per_class[0].support == 1


@pytest.mark.parametrize(
    ("gold", "predicted", "labels", "exc", "match"),
    [
        ("sim", ["sim"], ["sim"], TypeError, "sequences of strings"),
        (["sim"], "sim", ["sim"], TypeError, "sequences of strings"),
        (["sim"], ["sim"], "sim", TypeError, "labels must be a sequence of strings"),
        (["sim"], ["nao"], ["sim", "nao", "sim"], ValueError, "duplicate label: sim"),
        (["sim"], ["sim"], [], ValueError, "labels must not be empty"),
        (["sim"], ["sim"], ["si m"], ValueError, "without whitespace"),
        (["sim"], ["sim"], ["sim", 1], TypeError, "labels must be a sequence of strings"),
        (["sim"], ["nao", "sim"], ["sim"], ValueError, "same length"),
        ([], [], ["sim"], ValueError, "at least one example"),
        (["talvez"], ["sim"], ["sim"], ValueError, "not in the label set"),
        ([1], ["sim"], ["sim"], TypeError, "items must be strings"),
        (["sim"], [None], ["sim"], TypeError, "items must be strings"),
    ],
)
def test_classification_score_rejects_bad_input(
    gold: object,
    predicted: object,
    labels: object,
    exc: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exc, match=match):
        score_classification(gold, predicted, labels)  # type: ignore[arg-type]
