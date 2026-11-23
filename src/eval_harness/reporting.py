"""Relatório local de uma execução.

A média usa só o exemplo que tem nota naquela métrica. Se ninguém teve, a média
fica nula. O relógio entra só no manifesto do resumo, para o restante repetir.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval_harness.dataset import Dataset, DatasetCard, Example
from eval_harness.errors import ConfigError
from eval_harness.metrics import score_classification
from eval_harness.runner import ExampleResult, RunResult

SUMMARY_SCHEMA = "eval_harness.summary.v1"
RECORD_SCHEMA = "eval_harness.record.v1"
COMPARISON_SCHEMA = "eval_harness.comparison.v1"
_FAKE_SYSTEMS = frozenset({"oracle", "echo", "scripted-noisy"})


def summarize(
    card: DatasetCard,
    dataset: Dataset,
    result: RunResult,
    *,
    created_at: str,
) -> dict[str, Any]:
    """Agrega nota por métrica e por tag. `created_at` fica só no manifesto."""
    stamped = _created_at(created_at)
    paired = _pair(dataset, result)
    metric_names = list(result.manifest.metrics)
    rows = [row for _example, row in paired]
    errors = sum(1 for row in rows if row.error is not None)
    return {
        "schema": SUMMARY_SCHEMA,
        "dataset": {
            "name": card.name,
            "version": card.version,
            "task_type": card.task_type,
            "language": card.language,
        },
        "manifest": {
            "dataset_hash": result.manifest.dataset_hash,
            "config_hash": result.manifest.config_hash,
            "seed": result.manifest.seed,
            "versions": dict(result.manifest.versions),
            "system": result.manifest.system,
            "metrics": metric_names,
            "example_count": result.manifest.example_count,
            "created_at": stamped,
        },
        "counts": {
            "examples": len(rows),
            "errors": errors,
            "scored": len(rows) - errors,
        },
        "metrics": _means(rows, metric_names),
        "by_tag": _by_tag(card, paired, metric_names),
        "classification": _classification(card, paired, metric_names),
    }


def render_records(dataset: Dataset, result: RunResult) -> str:
    """Uma linha JSON por exemplo, na ordem do dataset. Latência não entra no resumo."""
    lines = [_json_line(_record(example, row)) for example, row in _pair(dataset, result)]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def render_report(summary: Mapping[str, Any]) -> str:
    """Tabela Markdown do resumo. `n/a` é média nula."""
    manifest = summary["manifest"]
    dataset = summary["dataset"]
    counts = summary["counts"]
    lines = [
        "# Relatório",
        "",
        *_table(
            ["campo", "valor"],
            [
                ["conjunto", _cell(dataset["name"])],
                ["versão", _cell(dataset["version"])],
                ["tarefa", _cell(dataset["task_type"])],
                ["sistema", _cell(manifest["system"])],
                ["seed", str(manifest["seed"])],
                ["exemplos", str(counts["examples"])],
                ["erros", str(counts["errors"])],
                ["dataset_hash", _cell(manifest["dataset_hash"])],
                ["config_hash", _cell(manifest["config_hash"])],
                ["criado_em", _cell(manifest["created_at"])],
            ],
        ),
        "",
        *_disclaimer(str(manifest["system"])),
        "",
        "## Métricas",
        "",
        (
            "A média usa só exemplos com nota nessa métrica. "
            "`n/a` significa que nenhum exemplo teve nota."
        ),
        "",
        *_metric_mean_table(summary["metrics"]),
        *_tag_sections(summary),
        *_classification_section(summary),
        "",
    ]
    return "\n".join(lines)


def write_run(
    directory: str | Path,
    card: DatasetCard,
    dataset: Dataset,
    result: RunResult,
    *,
    created_at: str,
) -> dict[str, Any]:
    """Grava `records.jsonl`, `summary.json` e `report.md`."""
    summary = summarize(card, dataset, result, created_at=created_at)
    target = Path(directory)
    _mkdir(target)
    _write(target / "records.jsonl", render_records(dataset, result))
    _write(target / "summary.json", _json_text(summary))
    _write(target / "report.md", render_report(summary))
    return summary


def load_summary(directory: str | Path) -> dict[str, Any]:
    """Lê o `summary.json` de um diretório de run. Não cria arquivo."""
    path = Path(directory)
    if not path.is_dir():
        raise ConfigError(f"run directory not found: {path}")
    file = path / "summary.json"
    try:
        raw = file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"summary.json is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"run directory has no summary.json: {path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"summary.json is not valid JSON: {path}") from exc
    _checked_summary(path, payload)
    if not isinstance(payload, dict):
        raise ConfigError(f"summary.json must be an object: {path}")
    return payload


def build_comparison(
    runs: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    force: bool,
) -> dict[str, Any]:
    """Delta por métrica e por tag. A base é o primeiro run.

    Hash de dataset diferente recusa, a menos que `force` seja verdadeiro.
    """
    if len(runs) < 2:
        raise ConfigError("compare needs at least two run directories")
    checked = [_checked_summary(Path(path), summary) for path, summary in runs]
    labels = _labels(checked)
    hashes = [item["manifest"]["dataset_hash"] for item in checked]
    match = all(value == hashes[0] for value in hashes)
    if not match and not force:
        detail = ", ".join(
            f"{label}={digest}" for label, digest in zip(labels, hashes, strict=True)
        )
        raise ConfigError(
            "dataset hash mismatch: "
            f"{detail}. Pass --force to compare anyway. No comparison was written."
        )
    names = _comparison_metric_names(checked)
    described = [
        {
            "label": label,
            "path": Path(path).as_posix(),
            "system": item["manifest"]["system"],
            "seed": item["manifest"]["seed"],
            "dataset_hash": item["manifest"]["dataset_hash"],
            "config_hash": item["manifest"]["config_hash"],
        }
        for (path, _summary), item, label in zip(runs, checked, labels, strict=True)
    ]
    return {
        "schema": COMPARISON_SCHEMA,
        "baseline": labels[0],
        "dataset_hashes_match": match,
        "dataset_hash": hashes[0] if match else None,
        "forced": bool(force and not match),
        "runs": described,
        "metrics": _metric_rows(checked, labels, names),
        "by_tag": _comparison_by_tag(checked, labels, names),
    }


def render_comparison(comparison: Mapping[str, Any]) -> str:
    """Tabela Markdown da comparação. Delta é o outro valor menos a base."""
    labels = [run["label"] for run in comparison["runs"]]
    lines = ["# Comparação", ""]
    if not comparison["dataset_hashes_match"]:
        lines.append(
            "Os hashes de dataset diferem. "
            "A comparação foi forçada e não descreve o mesmo conjunto."
        )
        lines.append("")
    lines.append(f"Base: {_cell(comparison['baseline'])}")
    shared = comparison["dataset_hash"]
    rendered_hash = "n/a" if shared is None else str(shared)
    lines.append(f"dataset_hash: {rendered_hash}")
    lines.append("")
    lines.extend(_metric_delta_table(comparison["metrics"], labels))
    lines.extend(_comparison_tag_sections(comparison, labels))
    lines.append("")
    return "\n".join(lines)


def write_comparison(directory: str | Path, comparison: Mapping[str, Any]) -> None:
    """Grava `comparison.json` e `comparison.md`. Não inclui relógio."""
    target = Path(directory)
    _mkdir(target)
    _write(target / "comparison.json", _json_text(comparison))
    _write(target / "comparison.md", render_comparison(comparison))


def _pair(dataset: Dataset, result: RunResult) -> list[tuple[Example, ExampleResult]]:
    if len(dataset.examples) != len(result.results):
        raise ConfigError("run results do not match the dataset")
    paired: list[tuple[Example, ExampleResult]] = []
    for example, row in zip(dataset.examples, result.results, strict=True):
        if row.example_id != example.id:
            raise ConfigError("run results do not match the dataset")
        paired.append((example, row))
    return paired


def _record(example: Example, row: ExampleResult) -> dict[str, Any]:
    usage = None
    if row.usage is not None:
        usage = {
            "input_tokens": row.usage.input_tokens,
            "output_tokens": row.usage.output_tokens,
        }
    return {
        "schema": RECORD_SCHEMA,
        "example_id": row.example_id,
        "output": row.output,
        "scores": {name: row.scores[name] for name in row.scores},
        "error": row.error,
        "latency_ms": row.latency_ms,
        "usage": usage,
        "tags": _tags(example),
    }


def _means(rows: Sequence[ExampleResult], metric_names: Sequence[str]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for name in metric_names:
        values = [row.scores[name] for row in rows if name in row.scores]
        aggregates[name] = {"mean": _mean(values), "count": len(values)}
    return aggregates


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _by_tag(
    card: DatasetCard,
    paired: Sequence[tuple[Example, ExampleResult]],
    metric_names: Sequence[str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ExampleResult]] = {}
    for example, row in paired:
        for key, value in _tags(example).items():
            grouped.setdefault((key, value), []).append(row)
    report: dict[str, Any] = {}
    for key in _tag_keys(card, paired):
        block: dict[str, Any] = {}
        for value in _tag_values(card, key, paired):
            rows = grouped.get((key, value), [])
            block[value] = {
                "count": len(rows),
                "errors": sum(1 for row in rows if row.error is not None),
                "metrics": _means(rows, metric_names),
            }
        report[key] = block
    return report


def _tag_keys(card: DatasetCard, paired: Sequence[tuple[Example, ExampleResult]]) -> list[str]:
    seen: list[str] = []
    for example, _row in paired:
        for key in _tags(example):
            if key not in seen:
                seen.append(key)
    ordered = [key for key in card.tag_schema if key in seen]
    ordered.extend(sorted(key for key in seen if key not in card.tag_schema))
    return ordered


def _tag_values(
    card: DatasetCard,
    key: str,
    paired: Sequence[tuple[Example, ExampleResult]],
) -> list[str]:
    seen: list[str] = []
    for example, _row in paired:
        value = _tags(example).get(key)
        if value is not None and value not in seen:
            seen.append(value)
    allowed = list(card.tag_schema.get(key, ()))
    ordered = [value for value in allowed if value in seen]
    ordered.extend(sorted(value for value in seen if value not in allowed))
    return ordered


def _tags(example: Example) -> dict[str, str]:
    raw = example.metadata.get("tags")
    if not isinstance(raw, Mapping):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            cleaned[key] = value
    return cleaned


def _classification(
    card: DatasetCard,
    paired: Sequence[tuple[Example, ExampleResult]],
    metric_names: Sequence[str],
) -> dict[str, Any] | None:
    # Sem `accuracy` na execução, a tabela de classe não entra. Não há nota inventada.
    if card.task_type != "classification" or "accuracy" not in metric_names:
        return None
    if card.label_set is None:
        raise ConfigError("classification card is missing label_set")
    gold: list[str] = []
    predicted: list[str] = []
    for example, row in paired:
        if row.output is None or "accuracy" not in row.scores:
            continue
        gold.append(example.expected)
        predicted.append(row.output)
    if not gold:
        return None
    score = score_classification(gold, predicted, card.label_set)
    return {
        "accuracy": score.accuracy,
        "macro_f1": score.macro_f1,
        "per_class": [
            {
                "label": item.label,
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
                "support": item.support,
            }
            for item in score.per_class
        ],
    }


def _created_at(value: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ConfigError("created_at must be a non-empty string")
    return value


def _checked_summary(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        raise ConfigError(f"summary.json must be an object: {path}")
    manifest = summary.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ConfigError(f"summary.json is missing a manifest: {path}")
    dataset_hash = manifest.get("dataset_hash")
    config_hash = manifest.get("config_hash")
    system = manifest.get("system")
    seed = manifest.get("seed")
    metrics = manifest.get("metrics")
    if not isinstance(dataset_hash, str) or dataset_hash == "":
        raise ConfigError(f"summary.json is missing dataset_hash: {path}")
    if not isinstance(config_hash, str) or config_hash == "":
        raise ConfigError(f"summary.json is missing config_hash: {path}")
    if not isinstance(system, str) or system == "":
        raise ConfigError(f"summary.json is missing system: {path}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError(f"summary.json is missing seed: {path}")
    if not isinstance(metrics, list) or any(not isinstance(item, str) for item in metrics):
        raise ConfigError(f"summary.json is missing metrics: {path}")
    metric_values = summary.get("metrics")
    by_tag = summary.get("by_tag")
    if not isinstance(metric_values, Mapping):
        raise ConfigError(f"summary.json is missing metrics: {path}")
    if not isinstance(by_tag, Mapping):
        raise ConfigError(f"summary.json is missing by_tag: {path}")
    return {
        "manifest": {
            "dataset_hash": dataset_hash,
            "config_hash": config_hash,
            "system": system,
            "seed": seed,
            "metrics": list(metrics),
        },
        "metrics": metric_values,
        "by_tag": by_tag,
        "classification": summary.get("classification"),
    }


def _labels(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[str, int] = {}
    labels: list[str] = []
    for summary in summaries:
        manifest = summary["manifest"]
        base = f"{manifest['system']}@{manifest['seed']}"
        seen[base] = seen.get(base, 0) + 1
        count = seen[base]
        labels.append(base if count == 1 else f"{base}#{count}")
    return labels


def _comparison_metric_names(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for summary in summaries:
        for name in summary["manifest"]["metrics"]:
            if name not in names:
                names.append(name)
    has_classification = any(
        isinstance(summary.get("classification"), Mapping) for summary in summaries
    )
    if has_classification and "macro_f1" not in names:
        names.append("macro_f1")
    return names


def _metric_rows(
    summaries: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in names:
        values = {
            label: _mean_of(summary, name) for label, summary in zip(labels, summaries, strict=True)
        }
        rows.append({"metric": name, "values": values, "deltas": _deltas(values, labels)})
    return rows


def _comparison_by_tag(
    summaries: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    global_names: Sequence[str],
) -> dict[str, Any]:
    keys: list[str] = []
    value_order: dict[str, list[str]] = {}
    for summary in summaries:
        for key, groups in summary["by_tag"].items():
            if not isinstance(groups, Mapping):
                continue
            if key not in value_order:
                keys.append(key)
                value_order[key] = []
            for value in groups:
                if value not in value_order[key]:
                    value_order[key].append(value)
    report: dict[str, Any] = {}
    for key in keys:
        block: dict[str, Any] = {}
        for value in value_order[key]:
            present = _present_tag_metrics(summaries, key, value)
            ordered = [name for name in global_names if name in present]
            ordered.extend(name for name in present if name not in ordered)
            rows = []
            for name in ordered:
                values = {
                    label: _tag_mean(summary, key, value, name)
                    for label, summary in zip(labels, summaries, strict=True)
                }
                rows.append({"metric": name, "values": values, "deltas": _deltas(values, labels)})
            block[value] = {"metrics": rows}
        report[key] = block
    return report


def _present_tag_metrics(
    summaries: Sequence[Mapping[str, Any]],
    key: str,
    value: str,
) -> list[str]:
    present: list[str] = []
    for summary in summaries:
        group = _tag_group(summary, key, value)
        if group is None:
            continue
        metrics = group.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for name in metrics:
            if name not in present:
                present.append(name)
    return present


def _deltas(values: Mapping[str, float | None], labels: Sequence[str]) -> dict[str, float | None]:
    baseline = values[labels[0]]
    return {label: _delta(baseline, values[label]) for label in labels[1:]}


def _delta(baseline: float | None, other: float | None) -> float | None:
    if baseline is None or other is None:
        return None
    return other - baseline


def _mean_of(summary: Mapping[str, Any], metric: str) -> float | None:
    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping) and metric in metrics:
        found = metrics[metric]
        if not isinstance(found, Mapping):
            return None
        return _as_float(found.get("mean"))
    if metric == "macro_f1":
        block = summary.get("classification")
        if isinstance(block, Mapping):
            return _as_float(block.get("macro_f1"))
    return None


def _tag_mean(summary: Mapping[str, Any], key: str, value: str, metric: str) -> float | None:
    group = _tag_group(summary, key, value)
    if group is None:
        return None
    metrics = group.get("metrics")
    if not isinstance(metrics, Mapping) or metric not in metrics:
        return None
    found = metrics[metric]
    if not isinstance(found, Mapping):
        return None
    return _as_float(found.get("mean"))


def _tag_group(summary: Mapping[str, Any], key: str, value: str) -> Mapping[str, Any] | None:
    groups = summary["by_tag"].get(key)
    if not isinstance(groups, Mapping):
        return None
    group = groups.get(value)
    if not isinstance(group, Mapping):
        return None
    return group


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _disclaimer(system: str) -> list[str]:
    if system in _FAKE_SYSTEMS:
        lead = "Números de sistema fake verificam o encanamento. Não são achado experimental."
    else:
        lead = "Execução local. Este arquivo não é um resultado publicado."
    parked = "O diretório de saída é ignorado pelo git. O repositório não versiona este relatório."
    return [lead, "", parked]


def _metric_mean_table(metrics: Mapping[str, Any]) -> list[str]:
    rows = [
        [_cell(name), _fmt(aggregate["mean"]), str(aggregate["count"])]
        for name, aggregate in metrics.items()
    ]
    return _table(["métrica", "média", "n"], rows)


def _tag_sections(summary: Mapping[str, Any]) -> list[str]:
    by_tag = summary["by_tag"]
    if not by_tag:
        return []
    metric_names = list(summary["metrics"])
    lines = ["", "## Tags"]
    for key, groups in by_tag.items():
        lines.extend(["", f"### {_cell(key)}", ""])
        headers = ["tag", "exemplos", "erros", *metric_names]
        rows = []
        for value, group in groups.items():
            row = [_cell(value), str(group["count"]), str(group["errors"])]
            for name in metric_names:
                row.append(_fmt(group["metrics"][name]["mean"]))
            rows.append(row)
        lines.extend(_table(headers, rows))
    return lines


def _classification_section(summary: Mapping[str, Any]) -> list[str]:
    if summary["dataset"]["task_type"] != "classification":
        return []
    lines = ["", "## Classificação", ""]
    block = summary["classification"]
    if block is None:
        lines.append("Sem saída com nota de acurácia. A macro-F1 não foi calculada.")
        return lines
    lines.append(f"acurácia: {_fmt(block['accuracy'])}")
    lines.append(f"macro_f1: {_fmt(block['macro_f1'])}")
    lines.append("")
    rows = [
        [
            _cell(item["label"]),
            _fmt(item["precision"]),
            _fmt(item["recall"]),
            _fmt(item["f1"]),
            str(item["support"]),
        ]
        for item in block["per_class"]
    ]
    lines.extend(_table(["classe", "precisão", "recall", "f1", "suporte"], rows))
    return lines


def _comparison_tag_sections(comparison: Mapping[str, Any], labels: Sequence[str]) -> list[str]:
    by_tag = comparison["by_tag"]
    if not by_tag:
        return []
    lines = ["", "## Tags"]
    for key, groups in by_tag.items():
        lines.extend(["", f"### {_cell(key)}"])
        for value, block in groups.items():
            lines.extend(["", f"#### {_cell(value)}", ""])
            lines.extend(_metric_delta_table(block["metrics"], labels))
    return lines


def _metric_delta_table(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> list[str]:
    others = list(labels[1:])
    headers = ["métrica", *labels, *[f"delta {label}" for label in others]]
    body: list[list[str]] = []
    for item in rows:
        values = item["values"]
        deltas = item["deltas"]
        row = [_cell(item["metric"]), _fmt(values.get(labels[0]))]
        row.extend(_fmt(values.get(label)) for label in others)
        row.extend(_fmt(deltas.get(label)) for label in others)
        body.append(row)
    return _table(headers, body)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    head = "| " + " | ".join(_cell(header) for header in headers) + " |"
    align = ["---", *["---:" for _header in headers[1:]]]
    separator = "| " + " | ".join(align) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return [head, separator, *body]


def _cell(value: object) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text.replace("|", "\\|")


def _fmt(value: object) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"


def _json_line(payload: Any) -> str:
    # Sem sort_keys: a ordem de inserção é a ordem do cartão e do manifesto.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _json_text(payload: Any) -> str:
    return _json_line(payload) + "\n"


def _mkdir(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"could not create output directory: {directory}") from exc


def _write(path: Path, text: str) -> None:
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ConfigError(f"could not write {path.name}: {path.parent}") from exc
