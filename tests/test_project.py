"""Contrato do repositório: CI, exemplo de ambiente e versão única. Sem resultado de modelo."""

from __future__ import annotations

import tomllib

from eval_harness.version import __version__
from tests.helpers import ROOT


def test_env_example_has_empty_values() -> None:
    assignments: list[str] = []
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        assignments.append(key)
        assert value == ""
    assert assignments == ["EVAL_HARNESS_SEED", "OPENAI_API_KEY", "OPENAI_MODEL"]


def test_gitignore_covers_secrets_and_run_output() -> None:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert "results/" in lines


def test_ci_pins_setup_uv_and_runs_lint_and_tests() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "astral-sh/setup-uv@v10.3.3" in text
    assert "uv run ruff check ." in text
    assert "uv run ruff format --check ." in text
    assert "uv run pytest" in text


def test_package_version_matches_pyproject() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    assert project["project"]["version"] == __version__ == "0.1.0"
