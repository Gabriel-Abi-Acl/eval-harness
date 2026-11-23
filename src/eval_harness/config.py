"""Configuração lida do ambiente do processo.

Este módulo não abre `.env`. Chave vazia é válida aqui; o provedor recusa na hora de chamar.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from eval_harness.errors import ConfigError

_SECRET_PARAM_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "authorization",
    }
)


@dataclass(frozen=True)
class Settings:
    """Seed e credencial do extra OpenAI. O repr não mostra a chave."""

    seed: int = 0
    openai_api_key: str = ""
    openai_model: str = ""

    def __repr__(self) -> str:
        key_state = "set" if self.openai_api_key else "empty"
        return (
            f"Settings(seed={self.seed}, openai_api_key={key_state}, "
            f"openai_model={self.openai_model!r})"
        )


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Lê `EVAL_HARNESS_SEED`, `OPENAI_API_KEY` e `OPENAI_MODEL`.

    Seed ausente ou vazia vale 0. O runner não consulta o ambiente sozinho.
    """
    env = os.environ if environ is None else environ
    return Settings(
        seed=_seed(env.get("EVAL_HARNESS_SEED", "0")),
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        openai_model=env.get("OPENAI_MODEL", ""),
    )


def public_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copia parâmetros JSON-safe e recusa nome de segredo.

    O valor recusado não volta na mensagem. Sem `params`, o manifesto usa objeto vazio.
    """
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ConfigError("run config must be a JSON object")
    copied = _copy_public(params)
    if not isinstance(copied, dict):
        raise ConfigError("run config must be a JSON object")
    return copied


def _seed(raw: str) -> int:
    text = raw.strip()
    if text == "":
        text = "0"
    try:
        seed = int(text, 10)
    except ValueError as exc:
        raise ConfigError("EVAL_HARNESS_SEED must be an integer") from exc
    if seed < 0:
        raise ConfigError("EVAL_HARNESS_SEED must be zero or positive")
    return seed


def _copy_public(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.lower() in _SECRET_PARAM_KEYS:
        raise ConfigError(f"run config must not include {key}")
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError("run config must be finite JSON")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ConfigError("run config keys must be strings")
            copied[child_key] = _copy_public(child, key=child_key)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_public(item) for item in value]
    raise ConfigError("run config must be JSON-serializable")
