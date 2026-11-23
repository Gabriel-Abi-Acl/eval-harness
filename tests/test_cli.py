"""Contrato do CLI: run, compare, trava ao vivo e resumo reproduzível.

Sistemas fake só checam o encanamento. A suíte não envia requisição.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from eval_harness.cli import main
from eval_harness.reporting import SUMMARY_SCHEMA
from tests.helpers import ROOT

QA = ROOT / "data" / "eval" / "qa_curta"
TONE = ROOT / "data" / "eval" / "classificacao_tom"
EXTRACT = ROOT / "data" / "eval" / "extracao_pedido"
_SECRET = "unit-test-key-not-sent"
_PIPELINE = "Números de sistema fake verificam o encanamento. Não são achado experimental."


def test_run_writes_three_files_and_fakes_diverge(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    echo = tmp_path / "echo"
    noisy = tmp_path / "noisy"
    assert main(["run", str(QA), "--system", "oracle", "--seed", "0", "--output", str(oracle)]) == 0
    assert main(["run", str(QA), "--system", "echo", "--seed", "0", "--output", str(echo)]) == 0
    assert (
        main(["run", str(QA), "--system", "scripted-noisy", "--seed", "0", "--output", str(noisy)])
        == 0
    )
    for directory in (oracle, echo, noisy):
        assert (directory / "records.jsonl").is_file()
        assert (directory / "summary.json").is_file()
        assert (directory / "report.md").is_file()
        text = (directory / "summary.json").read_text(encoding="utf-8")
        assert json.loads(text)["schema"] == SUMMARY_SCHEMA
        assert text == _stable_json(text)
        assert _SECRET not in text

    oracle_summary = json.loads((oracle / "summary.json").read_text(encoding="utf-8"))
    echo_summary = json.loads((echo / "summary.json").read_text(encoding="utf-8"))
    noisy_summary = json.loads((noisy / "summary.json").read_text(encoding="utf-8"))
    assert oracle_summary["metrics"]["exact_match"]["mean"] == 1.0
    assert oracle_summary["metrics"]["normalized_match"]["mean"] == 1.0
    assert oracle_summary["metrics"]["token_f1"]["mean"] == 1.0
    assert oracle_summary["by_tag"]["difficulty"]["fácil"]["metrics"]["exact_match"]["mean"] == 1.0
    assert echo_summary["metrics"]["exact_match"]["mean"] == 0.0
    noisy_mean = noisy_summary["metrics"]["exact_match"]["mean"]
    assert noisy_mean is not None
    assert 0.0 < noisy_mean < 1.0
    report = (oracle / "report.md").read_text(encoding="utf-8")
    assert "| métrica | média | n |" in report
    assert "| --- | ---: | ---: |" in report
    assert report.index("### difficulty") < report.index("### category")
    assert report.index("| exact_match |") < report.index("| normalized_match |")
    assert report.index("| normalized_match |") < report.index("| token_f1 |")
    assert _PIPELINE in report


def test_same_seed_repeats_the_summary_except_the_manifest_clock(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    argv = ["run", str(QA), "--system", "scripted-noisy", "--seed", "3", "--output"]
    assert main([*argv, str(first)]) == 0
    assert main([*argv, str(second)]) == 0
    raw_first = (first / "summary.json").read_text(encoding="utf-8")
    raw_second = (second / "summary.json").read_text(encoding="utf-8")
    assert _neutralize(raw_first) == _neutralize(raw_second)
    assert _clock_paths(json.loads(raw_first)) == ["manifest.created_at"]
    assert "created_at" not in (first / "records.jsonl").read_text(encoding="utf-8")
    assert _records_without_latency(first) == _records_without_latency(second)

    shifted = tmp_path / "shifted"
    shifted_code = main(
        ["run", str(QA), "--system", "scripted-noisy", "--seed", "4", "--output", str(shifted)]
    )
    assert shifted_code == 0
    other = json.loads((shifted / "summary.json").read_text(encoding="utf-8"))
    base = json.loads(raw_first)
    assert other["manifest"]["config_hash"] != base["manifest"]["config_hash"]
    assert _records_without_latency(shifted) != _records_without_latency(first)


def test_compare_writes_deltas_and_refuses_a_different_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    oracle = _run(tmp_path / "oracle", QA, "oracle")
    echo = _run(tmp_path / "echo", QA, "echo")
    tone = _run(tmp_path / "tone", TONE, "oracle")
    compared = tmp_path / "cmp"
    assert main(["compare", str(oracle), str(echo), "--output", str(compared)]) == 0
    payload = json.loads((compared / "comparison.json").read_text(encoding="utf-8"))
    assert payload["baseline"] == "oracle@0"
    assert payload["dataset_hashes_match"] is True
    exact = next(item for item in payload["metrics"] if item["metric"] == "exact_match")
    assert exact["deltas"]["echo@0"] == pytest.approx(-1.0)
    tag = payload["by_tag"]["difficulty"]["fácil"]["metrics"]
    tag_exact = next(item for item in tag if item["metric"] == "exact_match")
    assert tag_exact["deltas"]["echo@0"] == pytest.approx(-1.0)
    markdown = (compared / "comparison.md").read_text(encoding="utf-8")
    assert "| métrica | oracle@0 | echo@0 | delta echo@0 |" in markdown
    assert markdown.index("### difficulty") < markdown.index("### category")
    assert "created_at" not in (compared / "comparison.json").read_text(encoding="utf-8")
    again = tmp_path / "cmp-again"
    assert main(["compare", str(oracle), str(echo), "--output", str(again)]) == 0
    assert (compared / "comparison.json").read_bytes() == (again / "comparison.json").read_bytes()

    blocked = tmp_path / "blocked"
    capsys.readouterr()
    code = main(["compare", str(oracle), str(tone), "--output", str(blocked)])
    captured = capsys.readouterr()
    assert code == 2
    assert "dataset hash mismatch" in captured.err
    assert "No comparison was written." in captured.err
    assert not blocked.exists()

    forced = tmp_path / "forced"
    assert main(["compare", str(oracle), str(tone), "--force", "--output", str(forced)]) == 0
    forced_payload = json.loads((forced / "comparison.json").read_text(encoding="utf-8"))
    assert forced_payload["dataset_hashes_match"] is False
    assert "A comparação foi forçada" in (forced / "comparison.md").read_text(encoding="utf-8")


def test_classification_and_extraction_reports_stay_on_the_fakes(tmp_path: Path) -> None:
    tone = _run(tmp_path / "tone", TONE, "oracle")
    echo = _run(tmp_path / "echo", TONE, "echo")
    summary = json.loads((tone / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["accuracy"]["mean"] == 1.0
    assert summary["classification"]["macro_f1"] == 1.0
    assert summary["classification"]["accuracy"] == 1.0
    assert sum(item["support"] for item in summary["classification"]["per_class"]) == 18
    report = (tone / "report.md").read_text(encoding="utf-8")
    assert "| classe | precisão | recall | f1 | suporte |" in report
    assert _PIPELINE in report
    echo_summary = json.loads((echo / "summary.json").read_text(encoding="utf-8"))
    assert echo_summary["metrics"]["accuracy"]["mean"] == 0.0

    compared = tmp_path / "cmp"
    assert main(["compare", str(tone), str(echo), "--output", str(compared)]) == 0
    payload = json.loads((compared / "comparison.json").read_text(encoding="utf-8"))
    macro = next(item for item in payload["metrics"] if item["metric"] == "macro_f1")
    assert macro["values"]["oracle@0"] == 1.0
    assert macro["deltas"]["echo@0"] != 0.0

    extracted = _run(tmp_path / "extract", EXTRACT, "oracle")
    noisy = _run(tmp_path / "extract-noisy", EXTRACT, "scripted-noisy")
    perfect = json.loads((extracted / "summary.json").read_text(encoding="utf-8"))
    shifted = json.loads((noisy / "summary.json").read_text(encoding="utf-8"))
    assert perfect["metrics"]["json_field_accuracy"]["mean"] == 1.0
    assert perfect["classification"] is None
    noisy_mean = shifted["metrics"]["json_field_accuracy"]["mean"]
    assert noisy_mean is not None
    assert 0.0 < noisy_mean < 1.0


def test_metric_override_seed_and_default_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "override"
    code = main(
        [
            "run",
            str(QA),
            "--system",
            "echo",
            "--metric",
            "exact_match",
            "--metric",
            "token_f1",
            "--seed",
            "2",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest"]["metrics"] == ["exact_match", "token_f1"]
    assert summary["manifest"]["seed"] == 2

    blocked = tmp_path / "blocked"
    capsys.readouterr()
    assert main(["run", str(QA), "--metric", "macro_f1", "--output", str(blocked)]) == 2
    assert "not a per-example metric" in capsys.readouterr().err
    assert not blocked.exists()
    assert main(["run", str(QA), "--metric", "bleu", "--output", str(blocked)]) == 2
    assert "unknown metric" in capsys.readouterr().err
    duplicate = main(
        [
            "run",
            str(QA),
            "--metric",
            "exact_match",
            "--metric",
            "exact_match",
            "--output",
            str(blocked),
        ]
    )
    assert duplicate == 2
    assert main(["run", str(QA), "--seed", "-1", "--output", str(blocked)]) == 2
    assert main(["compare", str(out)]) == 2
    assert "at least two" in capsys.readouterr().err

    monkeypatch.setenv("EVAL_HARNESS_SEED", "4")
    seeded = tmp_path / "from-env"
    assert main(["run", str(QA), "--system", "echo", "--output", str(seeded)]) == 0
    from_env = json.loads((seeded / "summary.json").read_text(encoding="utf-8"))
    assert from_env["manifest"]["seed"] == 4
    flagged = tmp_path / "flag"
    assert main(["run", str(QA), "--system", "echo", "--seed", "2", "--output", str(flagged)]) == 0
    flagged_summary = json.loads((flagged / "summary.json").read_text(encoding="utf-8"))
    assert flagged_summary["manifest"]["seed"] == 2

    monkeypatch.chdir(tmp_path)
    assert main(["run", str(QA), "--system", "oracle", "--seed", "0"]) == 0
    default = tmp_path / "results" / "qa_curta" / "oracle" / "seed-0" / "summary.json"
    assert default.is_file()


@pytest.mark.parametrize(
    ("extra", "env", "match"),
    [
        (
            [],
            {"OPENAI_API_KEY": _SECRET, "OPENAI_MODEL": "unit-test-model"},
            "require --allow-live",
        ),
        (
            ["--allow-live"],
            {"OPENAI_API_KEY": "", "OPENAI_MODEL": "unit-test-model"},
            "OPENAI_API_KEY is empty",
        ),
        (
            ["--allow-live"],
            {"OPENAI_API_KEY": _SECRET, "OPENAI_MODEL": ""},
            "OPENAI_MODEL is empty",
        ),
    ],
)
def test_live_gate_exits_2_without_a_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    env: dict[str, str],
    match: str,
) -> None:
    calls = {"init": 0, "net": 0}

    class _Guard:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["init"] += 1
            raise AssertionError("OpenAISystem was constructed")

    def _net(*args: object, **kwargs: object) -> None:
        calls["net"] += 1
        raise AssertionError("network")

    monkeypatch.setattr("eval_harness.providers.OpenAISystem", _Guard)
    monkeypatch.setattr(socket, "create_connection", _net)
    sys.modules.pop("openai", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    out = tmp_path / "live"
    code = main(["run", str(QA), "--system", "openai", "--output", str(out), *extra])
    captured = capsys.readouterr()
    assert code == 2
    assert match in captured.err
    assert "No request was sent." in captured.err
    assert _SECRET not in captured.err
    assert _SECRET not in captured.out
    assert calls == {"init": 0, "net": 0}
    assert "openai" not in sys.modules
    assert not out.exists()


def test_fake_run_does_not_leak_a_key_or_open_a_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _net(*args: object, **kwargs: object) -> None:
        raise AssertionError("network")

    monkeypatch.setattr(socket, "create_connection", _net)
    monkeypatch.setenv("OPENAI_API_KEY", _SECRET)
    monkeypatch.setenv("OPENAI_MODEL", "unit-test-model")
    out = tmp_path / "fake"
    assert main(["run", str(QA), "--system", "oracle", "--output", str(out)]) == 0
    captured = capsys.readouterr()
    blob = "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir() if path.is_file())
    assert _SECRET not in blob
    assert _SECRET not in captured.out
    assert _SECRET not in captured.err


def test_module_entrypoint_runs_the_same_cli(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.pop("OPENAI_API_KEY", None)
    env.pop("OPENAI_MODEL", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval_harness",
            "run",
            str(QA),
            "--system",
            "echo",
            "--seed",
            "0",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "report.md").is_file()
    assert _PIPELINE in (tmp_path / "report.md").read_text(encoding="utf-8")
    script = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'eval-harness = "eval_harness.cli:main"' in script


def test_readme_explains_the_cli_without_publishing_a_score() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "eval-harness run" in text
    assert "eval-harness compare" in text
    assert "--allow-live" in text
    assert "--force" in text
    assert _PIPELINE in text
    assert "docs/spec-pr3-cli.md" in text
    assert "records.jsonl" in text
    assert "summary.json" in text
    assert "report.md" in text
    assert "## Como plugar um sistema ou uma métrica" in text
    assert "Nenhum modelo foi executado" in text


def _run(directory: Path, dataset: Path, system: str) -> Path:
    code = main(
        ["run", str(dataset), "--system", system, "--seed", "0", "--output", str(directory)]
    )
    assert code == 0
    return directory


def _neutralize(text: str) -> str:
    payload = json.loads(text)
    payload["manifest"]["created_at"] = "TIMESTAMP"
    return _dump(payload)


def _stable_json(text: str) -> str:
    return _dump(json.loads(text))


def _dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _clock_paths(payload: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else key
            if key == "created_at":
                found.append(path)
            found.extend(_clock_paths(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_clock_paths(value, f"{prefix}[{index}]"))
    return found


def _records_without_latency(directory: Path) -> list[dict[str, object]]:
    rows = []
    for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.pop("latency_ms")
        rows.append(row)
    return rows
