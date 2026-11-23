"""Interface de juiz. Este módulo não chama modelo.

O adaptador só aceita nota numérica em [0, 1]. A implementação do juiz fica com quem chama.
"""

from __future__ import annotations

import math
from typing import Protocol

from eval_harness.dataset import Example


class Judge(Protocol):
    """Devolve uma nota para um exemplo. Não há juiz de rede neste pacote."""

    def judge(self, example: Example, output: str) -> float: ...


class JudgeMetric:
    """Encaixa um `Judge` no protocolo de métrica.

    Bool não vale como nota: `True` não vira 1 por coerção. Fora de [0, 1] é erro.
    """

    def __init__(self, judge: Judge, *, name: str = "llm_judge") -> None:
        if not isinstance(name, str) or name.strip() == "":
            raise ValueError("metric name must be non-empty")
        self.name = name
        self._judge = judge

    def score(self, example: Example, output: str) -> float:
        value = self._judge.judge(example, output)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("judge score must be a number")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("judge score must be between 0 and 1")
        return score
