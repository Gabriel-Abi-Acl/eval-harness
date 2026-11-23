"""Contrato do adaptador OpenAI. A suíte não abre rede e não grava a chave."""

from __future__ import annotations

import builtins
import subprocess
import sys

import pytest

from eval_harness import OpenAISystem, Usage, load_dataset, run_eval
from eval_harness.errors import ProviderConfigError, ProviderNotInstalled, ProviderResponseError
from tests.helpers import PLACEHOLDER, make_example


def test_importing_the_package_does_not_import_openai() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import eval_harness, sys; raise SystemExit('openai' in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_empty_credentials_fail_before_import() -> None:
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY is empty"):
        OpenAISystem(model="fixture-model", api_key="")
    with pytest.raises(ProviderConfigError, match="OPENAI_MODEL is empty"):
        OpenAISystem(model="", api_key="test-key")


def test_repr_and_params_omit_the_key() -> None:
    system = OpenAISystem(model="fixture-model", api_key="test-key")
    assert system.params == {"provider": "openai", "model": "fixture-model"}
    assert "test-key" not in repr(system)


def test_missing_extra_raises_without_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("blocked for the test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    system = OpenAISystem(model="fixture-model", api_key="test-key")
    with pytest.raises(ProviderNotInstalled, match="optional extra") as caught:
        system(make_example())
    assert "test-key" not in str(caught.value)


def test_live_call_imports_once_and_keeps_reported_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            created = calls.setdefault("create", [])
            assert isinstance(created, list)
            created.append(kwargs)
            return _Response("ok", prompt_tokens=3, completion_tokens=1)

    class _Client:
        def __init__(self, api_key: str) -> None:
            calls["api_key"] = api_key
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    class _Module:
        OpenAI = _Client

    monkeypatch.setitem(sys.modules, "openai", _Module)
    system = OpenAISystem(model="fixture-model", api_key="test-key")
    first = system(make_example(text="pergunta"))
    second = system(make_example(text="outra"))

    assert first.output == "ok"
    assert first.usage == Usage(input_tokens=3, output_tokens=1)
    assert second.output == "ok"
    assert calls["api_key"] == "test-key"
    created = calls["create"]
    assert isinstance(created, list)
    assert len(created) == 2
    assert created[0]["temperature"] == 0
    assert created[0]["model"] == "fixture-model"
    assert created[0]["messages"] == [{"role": "user", "content": "pergunta"}]
    assert "test-key" not in repr(system)


def test_missing_usage_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            return _Response("ok", prompt_tokens=None, completion_tokens=None)

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    class _Module:
        OpenAI = _Client

    monkeypatch.setitem(sys.modules, "openai", _Module)
    produced = OpenAISystem(model="fixture-model", api_key="test-key")(make_example())
    assert produced.output == "ok"
    assert produced.usage is None


def test_sdk_error_does_not_keep_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            raise RuntimeError("request failed for test-key")

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    class _Module:
        OpenAI = _Client

    monkeypatch.setitem(sys.modules, "openai", _Module)
    system = OpenAISystem(model="fixture-model", api_key="test-key")
    with pytest.raises(ProviderResponseError, match="openai request failed") as caught:
        system(make_example())
    assert "test-key" not in str(caught.value)


def test_non_text_content_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            return _Response(None, prompt_tokens=1, completion_tokens=1)

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    class _Module:
        OpenAI = _Client

    monkeypatch.setitem(sys.modules, "openai", _Module)
    system = OpenAISystem(model="fixture-model", api_key="test-key")
    with pytest.raises(ProviderResponseError, match="not text") as caught:
        system(make_example())
    assert "test-key" not in str(caught.value)


def test_runner_manifest_does_not_include_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Completions:
        def create(self, **kwargs: object) -> _Response:
            return _Response("texto", prompt_tokens=None, completion_tokens=None)

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.chat = type("_Chat", (), {"completions": _Completions()})()

    class _Module:
        OpenAI = _Client

    monkeypatch.setitem(sys.modules, "openai", _Module)
    system = OpenAISystem(model="fixture-model", api_key="test-key")
    result = run_eval(load_dataset(PLACEHOLDER), system, [], seed=0)
    blob = f"{result.manifest!r}\n{result.results!r}"
    assert "test-key" not in blob
    assert result.manifest.system == "openai"
    assert result.results[0].output == "texto"


class _Message:
    def __init__(self, content: object) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: object) -> None:
        self.message = _Message(content)


class _Usage:
    def __init__(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    def __init__(
        self,
        content: object,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        self.choices = [_Choice(content)]
        if prompt_tokens is None and completion_tokens is None:
            self.usage = None
        else:
            self.usage = _Usage(prompt_tokens, completion_tokens)
