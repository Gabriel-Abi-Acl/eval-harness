"""Erros de fronteira do harness.

Mensagens em inglês e sem valor de chave. O loader preenche `line` com a linha do JSONL.
"""

from __future__ import annotations


class EvalHarnessError(Exception):
    """Base. Dataset, configuração e provedor herdam daqui."""


class DatasetError(EvalHarnessError):
    """JSONL recusado. O arquivo inteiro fica de fora; não há carga parcial."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        super().__init__(message)


class ConfigError(EvalHarnessError):
    """Seed, métrica ou parâmetro de execução inválido.

    Parâmetro com nome de segredo cai aqui. O valor não entra na mensagem.
    """


class ProviderConfigError(EvalHarnessError):
    """Modelo ou chave ausente. A mensagem não repete o valor."""


class ProviderNotInstalled(EvalHarnessError):
    """Extra opcional ausente. A importação só acontece na chamada ao vivo."""


class ProviderResponseError(EvalHarnessError):
    """Resposta do provedor sem texto utilizável."""
