"""Executa um sistema sobre um dataset e grava o manifesto.

Latência cobre só a chamada do sistema. Erro de um exemplo não inventa nota e não
interrompe os outros. O seed entra no hash de configuração; o runner não sorteia sozinho.
"""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from eval_harness.config import public_params
from eval_harness.dataset import Dataset, Example
from eval_harness.errors import ConfigError
from eval_harness.hashing import digest
from eval_harness.metrics import Metric
from eval_harness.systems import System, SystemOutput, Usage
from eval_harness.version import __version__


@dataclass(frozen=True)
class ExampleResult:
    """Registro de um exemplo. `output` e `scores` ficam vazios quando o sistema falha."""

    example_id: str
    output: str | None
    scores: Mapping[str, float]
    latency_ms: float
    error: str | None
    usage: Usage | None


@dataclass(frozen=True)
class RunManifest:
    """Identidade da execução. Sem relógio de parede, para o hash poder repetir."""

    dataset_hash: str
    config_hash: str
    seed: int
    versions: Mapping[str, str]
    system: str
    metrics: tuple[str, ...]
    example_count: int


@dataclass(frozen=True)
class RunResult:
    """Manifesto mais uma linha por exemplo, na ordem do dataset."""

    manifest: RunManifest
    results: tuple[ExampleResult, ...]


def runtime_versions() -> dict[str, str]:
    """Python e versão do pacote. Não entra no hash de configuração."""
    return {
        "python": platform.python_version(),
        "eval_harness": __version__,
    }


def run_eval(
    dataset: Dataset,
    system: System,
    metrics: Sequence[Metric],
    *,
    seed: int,
) -> RunResult:
    """Roda `system` em cada exemplo e aplica as métricas na ordem recebida.

    O seed é argumento. `EVAL_HARNESS_SEED` não é lido aqui.
    """
    checked_seed = _seed(seed)
    metric_names = _metric_names(metrics)
    system_name = _system_name(system)
    params = public_params(_system_params(system))
    config = {
        "seed": checked_seed,
        "system": system_name,
        "metrics": list(metric_names),
        "system_params": params,
    }
    manifest = RunManifest(
        dataset_hash=dataset.content_hash,
        config_hash=digest(config),
        seed=checked_seed,
        versions=MappingProxyType(runtime_versions()),
        system=system_name,
        metrics=metric_names,
        example_count=len(dataset.examples),
    )
    results = tuple(_one(example, system, metrics) for example in dataset.examples)
    return RunResult(manifest=manifest, results=results)


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("seed must be an integer")
    if seed < 0:
        raise ConfigError("seed must be zero or positive")
    return seed


def _metric_names(metrics: Sequence[Metric]) -> tuple[str, ...]:
    names: list[str] = []
    for metric in metrics:
        name = getattr(metric, "name", None)
        if not isinstance(name, str) or name.strip() == "":
            raise ConfigError("metric name must be a non-empty string")
        if name in names:
            raise ConfigError(f"duplicate metric name: {name}")
        names.append(name)
    return tuple(names)


def _system_name(system: System) -> str:
    name = getattr(system, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    fallback = getattr(system, "__name__", None)
    if isinstance(fallback, str) and fallback.strip():
        return fallback
    return type(system).__name__


def _system_params(system: System) -> Mapping[str, Any]:
    params = getattr(system, "params", {})
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("system params must be a mapping")
    return params


def _one(example: Example, system: System, metrics: Sequence[Metric]) -> ExampleResult:
    started = time.perf_counter()
    try:
        produced = system(example)
    except Exception as exc:
        return _failed(example.id, _elapsed_ms(started), _error_text(exc))
    latency_ms = _elapsed_ms(started)
    if not isinstance(produced, SystemOutput) or not isinstance(produced.output, str):
        return _failed(example.id, latency_ms, "system returned no text output")

    scores: dict[str, float] = {}
    error: str | None = None
    for metric in metrics:
        try:
            scores[metric.name] = _finite_score(metric.score(example, produced.output))
        except Exception as exc:
            # Métrica seguinte não roda: a nota parcial não finge que o restante passou.
            error = f"{metric.name}: {_error_text(exc)}"
            break
    return ExampleResult(
        example_id=example.id,
        output=produced.output,
        scores=MappingProxyType(scores),
        latency_ms=latency_ms,
        error=error,
        usage=produced.usage,
    )


def _failed(example_id: str, latency_ms: float, error: str) -> ExampleResult:
    return ExampleResult(
        example_id=example_id,
        output=None,
        scores=MappingProxyType({}),
        latency_ms=latency_ms,
        error=error,
        usage=None,
    )


def _finite_score(score: Any) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("metric score must be a number")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("metric score must be finite")
    return value


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _error_text(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0]
