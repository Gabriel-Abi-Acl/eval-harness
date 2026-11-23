"""Contrato da configuração: ambiente explícito, chave fora do repr e fora do manifesto."""

from __future__ import annotations

import pytest

from eval_harness import load_settings
from eval_harness.config import public_params
from eval_harness.errors import ConfigError


def test_missing_environment_is_empty() -> None:
    settings = load_settings({})
    assert settings.seed == 0
    assert settings.openai_api_key == ""
    assert settings.openai_model == ""
    assert "empty" in repr(settings)


def test_blank_seed_is_zero_and_key_is_redacted() -> None:
    settings = load_settings(
        {
            "EVAL_HARNESS_SEED": "",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "fixture-model",
        }
    )
    assert settings.seed == 0
    assert settings.openai_model == "fixture-model"
    assert "test-key" not in repr(settings)
    assert "set" in repr(settings)


def test_seed_must_be_a_non_negative_integer() -> None:
    assert load_settings({"EVAL_HARNESS_SEED": "4"}).seed == 4
    with pytest.raises(ConfigError, match="integer"):
        load_settings({"EVAL_HARNESS_SEED": "nope"})
    with pytest.raises(ConfigError, match="zero or positive"):
        load_settings({"EVAL_HARNESS_SEED": "-1"})


def test_public_params_reject_secret_names_without_the_value() -> None:
    with pytest.raises(ConfigError, match="token") as caught:
        public_params({"mode": "echo", "nested": {"token": "super-secret-value"}})
    assert "super-secret-value" not in str(caught.value)


def test_public_params_reject_non_json() -> None:
    with pytest.raises(ConfigError, match="JSON-serializable"):
        public_params({"mode": object()})
    with pytest.raises(ConfigError, match="finite JSON"):
        public_params({"temperature": float("nan")})
