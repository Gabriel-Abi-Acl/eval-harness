"""CLI offline: `run` grava o relatório e `compare` calcula deltas.

Provedor ao vivo só segue com `--allow-live`, modelo e chave. Sem isso o
processo sai com código 2 e não abre requisição.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from eval_harness.config import Settings, load_settings
from eval_harness.dataset import Dataset, DatasetCard, load_carded_dataset
from eval_harness.errors import ConfigError, EvalHarnessError
from eval_harness.metrics import (
    Accuracy,
    ExactMatch,
    JsonFieldAccuracy,
    Metric,
    NormalizedMatch,
    TokenF1,
)
from eval_harness.reporting import (
    build_comparison,
    load_summary,
    render_comparison,
    write_comparison,
    write_run,
)
from eval_harness.runner import run_eval
from eval_harness.systems import System, build_fake_system

_FAKE_SYSTEMS = ("oracle", "echo", "scripted-noisy")
_LIVE_SYSTEM = "openai"
_SYSTEMS = (*_FAKE_SYSTEMS, _LIVE_SYSTEM)
_AGGREGATE_ONLY = frozenset({"per_class_precision_recall", "macro_f1"})
_FACTORIES: dict[str, type[Metric]] = {
    "exact_match": ExactMatch,
    "normalized_match": NormalizedMatch,
    "token_f1": TokenF1,
    "accuracy": Accuracy,
    "json_field_accuracy": JsonFieldAccuracy,
}


def main(argv: list[str] | None = None) -> int:
    """Devolve 0 em sucesso e 2 em uso inválido, dataset ruim ou trava ao vivo."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _run(args)
        else:
            _compare(args)
    except EvalHarnessError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-harness",
        description=(
            "Run an offline evaluation and write a local report. "
            "Fake systems check the pipeline; they are not a finding."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Score a carded dataset and write the local report.")
    run.add_argument("dataset", type=Path, help="Directory with card.json and examples.jsonl.")
    run.add_argument(
        "--system",
        default="oracle",
        choices=_SYSTEMS,
        help="Fake spec or openai. Default: oracle, which returns the reference.",
    )
    run.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Per-example metric. Repeat to override the card.",
    )
    run.add_argument("--seed", type=int, default=None, help="Overrides EVAL_HARNESS_SEED.")
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for records.jsonl, summary.json, and report.md.",
    )
    run.add_argument(
        "--allow-live",
        action="store_true",
        help="Required, with OPENAI_MODEL and OPENAI_API_KEY, before any live request.",
    )
    compare = sub.add_parser("compare", help="Delta of two or more run directories.")
    compare.add_argument("runs", nargs="+", type=Path, help="Directories written by run.")
    compare.add_argument(
        "--force",
        action="store_true",
        help="Compare runs whose dataset hashes differ.",
    )
    compare.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for comparison.md and comparison.json. Default: markdown on stdout.",
    )
    return parser


def _run(args: argparse.Namespace) -> None:
    seed = _seed(args.seed)
    carded = load_carded_dataset(args.dataset)
    metrics = _metrics(carded.card, args.metric)
    system = _system(args.system, carded.dataset, seed, allow_live=args.allow_live)
    result = run_eval(carded.dataset, system, metrics, seed=seed)
    directory = args.output
    if directory is None:
        slug = args.dataset.name or "dataset"
        directory = Path("results") / slug / args.system / f"seed-{seed}"
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    write_run(directory, carded.card, carded.dataset, result, created_at=created_at)
    print(directory)


def _compare(args: argparse.Namespace) -> None:
    if len(args.runs) < 2:
        raise ConfigError("compare needs at least two run directories")
    loaded = [(path, load_summary(path)) for path in args.runs]
    comparison = build_comparison(loaded, force=args.force)
    if args.output is None:
        print(render_comparison(comparison), end="")
        return
    write_comparison(args.output, comparison)
    print(args.output)


def _seed(explicit: int | None) -> int:
    # O runner não lê o ambiente. O flag ganha de EVAL_HARNESS_SEED.
    if explicit is None:
        return load_settings().seed
    if explicit < 0:
        raise ConfigError("seed must be zero or positive")
    return explicit


def _metrics(card: DatasetCard, override: list[str] | None) -> list[Metric]:
    if override is None:
        names = [name for name in card.metrics if name not in _AGGREGATE_ONLY]
    else:
        names = list(override)
        for name in names:
            if name in _AGGREGATE_ONLY:
                raise ConfigError(f"{name} is an aggregate and is not a per-example metric")
            if name not in _FACTORIES:
                raise ConfigError(f"unknown metric: {name}")
        if len(names) != len(set(names)):
            raise ConfigError("duplicate metric name")
    return [_FACTORIES[name]() for name in names]


def _system(spec: str, dataset: Dataset, seed: int, *, allow_live: bool) -> System:
    if spec != _LIVE_SYSTEM:
        return build_fake_system(spec, dataset, seed=seed)
    # A trava fica antes da importação e de qualquer chamada. Sem flag, sem rede.
    settings = load_settings()
    _require_live(allow_live, settings)
    from eval_harness.providers import OpenAISystem

    return OpenAISystem(model=settings.openai_model, api_key=settings.openai_api_key)


def _require_live(allow_live: bool, settings: Settings) -> None:
    if not allow_live:
        raise ConfigError("live providers require --allow-live. No request was sent.")
    if settings.openai_model == "":
        raise ConfigError("OPENAI_MODEL is empty. No request was sent.")
    if settings.openai_api_key == "":
        raise ConfigError("OPENAI_API_KEY is empty. No request was sent.")
