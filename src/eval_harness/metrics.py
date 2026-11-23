"""Métricas determinísticas. Não chamam modelo.

Match exato é igualdade de string. O normalizado só mexe em caixa e espaço; acento fica.
F1 de tokens usa essa mesma normalização e conta multiconjunto.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
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
