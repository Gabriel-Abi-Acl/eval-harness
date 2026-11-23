"""Sistema sob teste e o fake determinístico.

O protocolo é um callable: devolve texto e, se houver, uso de tokens. O fake não
sorteia texto e não preenche token quando o chamador não passou `usage`.
oracle, echo e scripted-noisy só montam esse fake para o CLI.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from eval_harness.dataset import Dataset, Example


class FakeScriptError(LookupError):
    """O fake não tem texto para este exemplo. Não inventa um."""


@dataclass(frozen=True)
class Usage:
    """Contagem reportada pelo sistema. Ausência fica `None`, não vira zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class SystemOutput:
    """Saída de uma chamada. `usage` é opcional."""

    output: str
    usage: Usage | None = None


class System(Protocol):
    """Callable. O runner não exige classe, só este retorno."""

    def __call__(self, example: Example) -> SystemOutput: ...


_MODES = frozenset({"echo", "lookup", "scripted"})


class FakeSystem:
    """Eco, tabela por id ou fita de respostas.

    `lookup` e `scripted` falham quando o roteiro acaba, em vez de inventar texto.
    A fita avança e não rebobina: outra execução precisa de outra instância.
    """

    def __init__(
        self,
        mode: str = "echo",
        *,
        table: Mapping[str, str] | None = None,
        script: Sequence[str] | None = None,
        usage: Usage | None = None,
        name: str = "fake",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown fake mode: {mode}")
        if mode == "lookup" and table is None:
            raise ValueError("lookup mode requires a table")
        if mode == "scripted" and script is None:
            raise ValueError("scripted mode requires a script")
        if mode != "lookup" and table is not None:
            raise ValueError("table is only valid in lookup mode")
        if mode != "scripted" and script is not None:
            raise ValueError("script is only valid in scripted mode")
        if name.strip() == "":
            raise ValueError("system name must be non-empty")
        self.mode = mode
        self.name = name
        self._usage = usage
        self._table = _string_table(table or {})
        self._script = _string_script(script or ())
        self._cursor = 0

    @property
    def params(self) -> dict[str, Any]:
        """Parâmetros públicos do manifesto. Não inclui cursor nem uso de tokens."""
        if self.mode == "echo":
            return {"mode": "echo"}
        if self.mode == "lookup":
            return {"mode": "lookup", "table": dict(sorted(self._table.items()))}
        return {"mode": "scripted", "script": list(self._script)}

    def __call__(self, example: Example) -> SystemOutput:
        return SystemOutput(output=self._text(example), usage=self._usage)

    def _text(self, example: Example) -> str:
        if self.mode == "echo":
            return example.input
        if self.mode == "lookup":
            try:
                return self._table[example.id]
            except KeyError:
                raise FakeScriptError(f"no scripted output for id {example.id}") from None
        if self._cursor >= len(self._script):
            raise FakeScriptError("scripted outputs are exhausted")
        text = self._script[self._cursor]
        self._cursor += 1
        return text


def _string_table(table: Mapping[str, str]) -> dict[str, str]:
    copied: dict[str, str] = {}
    for key, value in table.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("lookup table keys and values must be strings")
        copied[key] = value
    return copied


def build_fake_system(spec: str, dataset: Dataset, *, seed: int) -> FakeSystem:
    """Oracle, eco ou fita barulhenta. Os três são determinísticos.

    A fita consome o seed: outro seed muda a saída, o mesmo seed repete.
    Nenhum modo chama provedor nem inventa texto fora dessa regra.
    """
    _require_seed(seed)
    if spec == "oracle":
        table = {example.id: example.expected for example in dataset.examples}
        return FakeSystem("lookup", table=table, name="oracle")
    if spec == "echo":
        return FakeSystem("echo", name="echo")
    if spec == "scripted-noisy":
        script = [
            _noisy_text(example.expected, index, seed)
            for index, example in enumerate(dataset.examples)
        ]
        return FakeSystem("scripted", script=script, name="scripted-noisy")
    raise ValueError(f"unknown fake system: {spec}")


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be zero or a positive integer")


def _noisy_text(expected: str, index: int, seed: int) -> str:
    # Um terço repete o gabarito, um terço altera, um terço erra. A regra não sorteia.
    slot = (index + seed) % 3
    if slot == 0:
        return expected
    if slot == 1:
        return _partial_text(expected)
    return _miss_text(expected)


def _partial_text(expected: str) -> str:
    payload = _json_object(expected)
    if payload:
        key = next(reversed(payload))
        payload[key] = _wrong_json_value(payload[key])
        return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))
    parts = expected.split(" ")
    if len(parts) > 1:
        return " ".join(parts[:-1])
    return f"{expected} extra"


def _miss_text(expected: str) -> str:
    if _json_object(expected) is not None:
        return "{}"
    if expected != "incorreto":
        return "incorreto"
    return "outro"


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _wrong_json_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return f"{value}-x" if value else "x"
    if isinstance(value, float):
        return value + 1.0
    if value is None:
        return False
    return "x"


def _string_script(script: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(script, Sequence) or isinstance(script, (str, bytes)):
        raise ValueError("script must be a sequence of strings")
    copied: list[str] = []
    for item in script:
        if not isinstance(item, str):
            raise ValueError("script must be a sequence of strings")
        copied.append(item)
    return tuple(copied)
