"""Harness offline para avaliar sistemas com LLM.

A importação não chama rede e não carrega SDK de provedor. O cliente OpenAI entra
só na primeira chamada ao vivo, se o extra estiver instalado.
"""

from eval_harness.config import Settings, load_settings
from eval_harness.dataset import (
    CardedDataset,
    Dataset,
    DatasetCard,
    Example,
    load_carded_dataset,
    load_dataset,
)
from eval_harness.judge import Judge, JudgeMetric
from eval_harness.metrics import (
    Accuracy,
    ClassificationScore,
    ClassScore,
    ExactMatch,
    JsonFieldAccuracy,
    Metric,
    NormalizedMatch,
    TokenF1,
    normalize_text,
    score_classification,
)
from eval_harness.providers import OpenAISystem
from eval_harness.runner import ExampleResult, RunManifest, RunResult, run_eval
from eval_harness.systems import FakeSystem, System, SystemOutput, Usage
from eval_harness.version import __version__

__all__ = [
    "Accuracy",
    "CardedDataset",
    "ClassificationScore",
    "ClassScore",
    "Dataset",
    "DatasetCard",
    "ExactMatch",
    "Example",
    "ExampleResult",
    "FakeSystem",
    "JsonFieldAccuracy",
    "Judge",
    "JudgeMetric",
    "Metric",
    "NormalizedMatch",
    "OpenAISystem",
    "RunManifest",
    "RunResult",
    "Settings",
    "System",
    "SystemOutput",
    "TokenF1",
    "Usage",
    "__version__",
    "load_carded_dataset",
    "load_dataset",
    "load_settings",
    "normalize_text",
    "run_eval",
    "score_classification",
]
