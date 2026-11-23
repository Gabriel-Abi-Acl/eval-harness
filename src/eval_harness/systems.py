"""Sistema sob teste e o fake determinístico.

O protocolo é um callable: devolve texto e, se houver, uso de tokens. O fake não
sorteia texto e não preenche token quando o chamador não passou `usage`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from eval_harness.dataset import Example


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


def _string_script(script: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(script, Sequence) or isinstance(script, (str, bytes)):
        raise ValueError("script must be a sequence of strings")
    copied: list[str] = []
    for item in script:
        if not isinstance(item, str):
            raise ValueError("script must be a sequence of strings")
        copied.append(item)
    return tuple(copied)
