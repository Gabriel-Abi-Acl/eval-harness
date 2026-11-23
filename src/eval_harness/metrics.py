"""Métricas determinísticas. Não chamam modelo.

Match exato é igualdade de string. O normalizado só mexe em caixa e espaço; acento fica.
F1 de tokens usa essa mesma normalização e conta multiconjunto.
Acurácia de classe fechada também é igualdade de string. Precisão, recall e macro-F1
saem do conjunto inteiro, não de um exemplo.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from eval_harness.dataset import Example


class Metric(Protocol):
    """Nota de uma saída contra o exemplo. O nome entra no manifesto."""

    name: str

    def score(self, example: Example, output: str) -> float: ...


def normalize_text(value: str) -> str:
    """NFKC, casefold e espaço colapsado. Acento permanece."""
    if not isinstance(value, str):
        raise TypeError("text must be a string")
    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(folded.split())


class ExactMatch:
    """1 se `output` é igual a `expected`, senão 0. Caixa e espaço contam."""

    name = "exact_match"

    def score(self, example: Example, output: str) -> float:
        _require_text(output)
        return 1.0 if output == example.expected else 0.0


class NormalizedMatch:
    """1 se a forma normalizada é igual. Não é substring e não tira pontuação."""

    name = "normalized_match"

    def score(self, example: Example, output: str) -> float:
        _require_text(output)
        return 1.0 if normalize_text(output) == normalize_text(example.expected) else 0.0


class TokenF1:
    """F1 do multiconjunto de tokens separados por espaço, depois de normalizar.

    Os dois lados vazios valem 1. Um lado vazio vale 0.
    """

    name = "token_f1"

    def score(self, example: Example, output: str) -> float:
        _require_text(output)
        predicted = Counter(normalize_text(output).split())
        gold = Counter(normalize_text(example.expected).split())
        if not predicted and not gold:
            return 1.0
        if not predicted or not gold:
            return 0.0
        overlap = sum((predicted & gold).values())
        precision = overlap / sum(predicted.values())
        recall = overlap / sum(gold.values())
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


class JsonFieldAccuracy:
    """Fração dos campos do gabarito com o mesmo valor JSON no candidato.

    Os dois textos precisam ser objetos JSON. Campo extra no candidato não entra.
    Bool não equivale a 0 ou 1. `1` e `1.0` não equivalem. Texto que não é objeto vale 0.
    """

    name = "json_field_accuracy"

    def score(self, example: Example, output: str) -> float:
        _require_text(output)
        gold = _json_object(example.expected)
        predicted = _json_object(output)
        if gold is None or predicted is None:
            return 0.0
        if not gold:
            return 1.0 if not predicted else 0.0
        hits = 0
        for key, value in gold.items():
            if key in predicted and _same_json(predicted[key], value):
                hits += 1
        return hits / len(gold)


def _require_text(output: str) -> None:
    if not isinstance(output, str):
        raise TypeError("output must be a string")


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


class Accuracy:
    """1 se a saída é igual ao rótulo, senão 0. Caixa e espaço contam.

    A comparação é a de `exact_match`. O nome `accuracy` marca classe fechada.
    Precisão, recall e macro-F1 ficam em `score_classification`.
    """

    name = "accuracy"

    def score(self, example: Example, output: str) -> float:
        _require_text(output)
        return 1.0 if output == example.expected else 0.0


@dataclass(frozen=True)
class ClassScore:
    """Precisão, recall e F1 de uma classe. `support` é a contagem no gabarito."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationScore:
    """Acurácia do conjunto e uma linha por rótulo, na ordem recebida."""

    accuracy: float
    per_class: tuple[ClassScore, ...]
    macro_f1: float


def score_classification(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
) -> ClassificationScore:
    """Acurácia, precisão e recall por classe e macro-F1.

    `labels` é o conjunto fechado, na ordem do relatório. Classe sem exemplo entra
    com precisão, recall e F1 iguais a 0. Predição fora do conjunto não abre classe:
    erra a acurácia e conta só como falso negativo do gabarito. Denominador vazio vale 0.
    """
    allowed = _label_set(labels)
    pairs = _label_pairs(gold, predicted, allowed)
    tallies = {label: {"tp": 0, "fp": 0, "fn": 0} for label in allowed}
    correct = 0
    for gold_label, pred_label in pairs:
        if pred_label == gold_label:
            correct += 1
            tallies[gold_label]["tp"] += 1
            continue
        tallies[gold_label]["fn"] += 1
        if pred_label in tallies:
            tallies[pred_label]["fp"] += 1

    per_class: list[ClassScore] = []
    for label in allowed:
        tp = tallies[label]["tp"]
        fp = tallies[label]["fp"]
        fn = tallies[label]["fn"]
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        per_class.append(
            ClassScore(
                label=label,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                support=tp + fn,
            )
        )
    macro_f1 = sum(item.f1 for item in per_class) / len(per_class)
    return ClassificationScore(
        accuracy=correct / len(pairs),
        per_class=tuple(per_class),
        macro_f1=macro_f1,
    )


def _label_set(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a sequence of strings")
    checked: list[str] = []
    for label in labels:
        if not isinstance(label, str):
            raise TypeError("labels must be a sequence of strings")
        if label == "" or any(char.isspace() for char in label):
            raise ValueError("labels must be non-empty strings without whitespace")
        if label in checked:
            raise ValueError(f"duplicate label: {label}")
        checked.append(label)
    if not checked:
        raise ValueError("labels must not be empty")
    return tuple(checked)


def _label_pairs(
    gold: Sequence[str],
    predicted: Sequence[str],
    allowed: tuple[str, ...],
) -> list[tuple[str, str]]:
    if isinstance(gold, (str, bytes)) or not isinstance(gold, Sequence):
        raise TypeError("gold and predicted must be sequences of strings")
    if isinstance(predicted, (str, bytes)) or not isinstance(predicted, Sequence):
        raise TypeError("gold and predicted must be sequences of strings")
    gold_list = list(gold)
    predicted_list = list(predicted)
    if len(gold_list) != len(predicted_list):
        raise ValueError("gold and predicted must have the same length")
    if not gold_list:
        raise ValueError("classification score needs at least one example")
    allowed_set = set(allowed)
    pairs: list[tuple[str, str]] = []
    for gold_label, pred_label in zip(gold_list, predicted_list, strict=True):
        if not isinstance(gold_label, str) or not isinstance(pred_label, str):
            raise TypeError("gold and predicted items must be strings")
        if gold_label not in allowed_set:
            raise ValueError(f"gold label {gold_label!r} is not in the label set")
        pairs.append((gold_label, pred_label))
    return pairs


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _same_json(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_same_json(item, other) for item, other in zip(left, right, strict=True))
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    if isinstance(left, float) and isinstance(right, float):
        return left == right
    if isinstance(left, str) and isinstance(right, str):
        return left == right
    return left is None and right is None
