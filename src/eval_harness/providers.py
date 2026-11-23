"""Cliente OpenAI importado só na primeira chamada ao vivo.

Sem o extra `openai`, a chamada levanta `ProviderNotInstalled`. A suíte não usa rede.
Temperatura 0. O seed do manifesto não é enviado neste PR; execução real fica para o PR4.
"""

from __future__ import annotations

from typing import Any

from eval_harness.dataset import Example
from eval_harness.errors import ProviderConfigError, ProviderNotInstalled, ProviderResponseError
from eval_harness.systems import SystemOutput, Usage


class OpenAISystem:
    """Adaptador mínimo de chat completions. A chave não entra em `params` nem no repr."""

    name = "openai"

    def __init__(self, *, model: str, api_key: str) -> None:
        if api_key == "":
            raise ProviderConfigError("OPENAI_API_KEY is empty")
        if model == "":
            raise ProviderConfigError("OPENAI_MODEL is empty")
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    @property
    def params(self) -> dict[str, str]:
        return {"provider": "openai", "model": self._model}

    def __repr__(self) -> str:
        return f"OpenAISystem(model={self._model!r})"

    def __call__(self, example: Example) -> SystemOutput:
        try:
            return self._complete(example)
        except (ProviderNotInstalled, ProviderConfigError, ProviderResponseError):
            raise
        except Exception:
            # O SDK às vezes devolve a chave no texto do erro. O registro fica genérico.
            raise ProviderResponseError("openai request failed") from None

    def _complete(self, example: Example) -> SystemOutput:
        # Importação adiada: instalar o extra não é obrigatório para o fake nem para a suíte.
        client = self._client
        if client is None:
            client = _openai_client(self._api_key)
            self._client = client
        response = client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[{"role": "user", "content": example.input}],
        )
        return _parse_response(response)


def _openai_client(api_key: str) -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ProviderNotInstalled(
            "openai is not installed; sync the optional extra: uv sync --extra openai"
        ) from exc
    return openai.OpenAI(api_key=api_key)


def _parse_response(response: Any) -> SystemOutput:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ProviderResponseError("openai response has no message content") from exc
    if not isinstance(content, str):
        raise ProviderResponseError("openai response content is not text")
    return SystemOutput(output=content, usage=_usage(getattr(response, "usage", None)))


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    prompt = getattr(raw, "prompt_tokens", None)
    completion = getattr(raw, "completion_tokens", None)
    if prompt is None and completion is None:
        return None
    return Usage(input_tokens=_token_count(prompt), output_tokens=_token_count(completion))


def _token_count(value: Any) -> int | None:
    # Ausência fica nula. Bool é int em Python e não conta como token.
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderResponseError("openai token count is not an integer")
    if value < 0:
        raise ProviderResponseError("openai token count is negative")
    return value
