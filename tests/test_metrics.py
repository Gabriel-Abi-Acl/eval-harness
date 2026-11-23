"""Contrato das métricas determinísticas. Gabarito sintético, sem modelo."""

from __future__ import annotations

import pytest

from eval_harness import ExactMatch, JsonFieldAccuracy, NormalizedMatch, TokenF1, normalize_text
from tests.helpers import make_example

_EXACT = ExactMatch()
_NORMAL = NormalizedMatch()
_F1 = TokenF1()
_JSON = JsonFieldAccuracy()


def test_exact_match_is_string_equality() -> None:
    example = make_example(expected="A")
    assert _EXACT.score(example, "A") == 1.0
    assert _EXACT.score(example, "a") == 0.0
    assert _EXACT.score(example, "A ") == 0.0
    assert _EXACT.score(make_example(expected=""), "") == 1.0


def test_normalized_match_folds_case_and_space_only() -> None:
    example = make_example(expected="Brasília")
    assert _NORMAL.score(example, "  brasília  ") == 1.0
    assert _NORMAL.score(example, "BRASILIA") == 0.0
    assert _NORMAL.score(example, "brasília.") == 0.0
    assert normalize_text("  Gato\tPreto ") == "gato preto"


def test_token_f1_is_multiset_overlap() -> None:
    example = make_example(expected="gato preto")
    assert _F1.score(example, "gato preto") == 1.0
    assert _F1.score(example, "preto gato") == 1.0
    assert _F1.score(example, "gato") == pytest.approx(2.0 / 3.0)
    assert _F1.score(make_example(expected="a"), "a a") == pytest.approx(2.0 / 3.0)
    assert _F1.score(example, "azul") == 0.0
    assert _F1.score(make_example(expected=""), "") == 1.0
    assert _F1.score(make_example(expected="a"), "") == 0.0
    assert _F1.score(make_example(expected="A B"), "a b") == 1.0


def test_json_field_accuracy_scores_gold_keys_only() -> None:
    example = make_example(expected='{"cidade": "Brasília", "n": 1, "ok": true}')
    full = '{"n": 1, "ok": true, "cidade": "Brasília", "extra": 0}'
    wrong_bool = '{"cidade": "Brasília", "n": 1, "ok": false}'
    float_one = '{"cidade": "Brasília", "n": 1.0, "ok": true}'
    bool_as_int = '{"cidade": "Brasília", "n": 1, "ok": 1}'
    assert _JSON.score(example, full) == 1.0
    assert _JSON.score(example, wrong_bool) == pytest.approx(2.0 / 3.0)
    assert _JSON.score(example, float_one) == pytest.approx(2.0 / 3.0)
    assert _JSON.score(example, bool_as_int) == pytest.approx(2.0 / 3.0)
    assert _JSON.score(example, "não é json") == 0.0
    assert _JSON.score(example, "[]") == 0.0
    assert _JSON.score(make_example(expected="{}"), "{}") == 1.0
    assert _JSON.score(make_example(expected="{}"), '{"a": 1}') == 0.0


def test_json_field_accuracy_compares_nested_values() -> None:
    example = make_example(expected='{"item": {"sku": "A", "qty": 2}}')
    assert _JSON.score(example, '{"item": {"qty": 2, "sku": "A"}}') == 1.0
    assert _JSON.score(example, '{"item": {"sku": "A"}}') == 0.0


def test_metric_rejects_non_string_output() -> None:
    example = make_example()
    with pytest.raises(TypeError, match="output must be a string"):
        _EXACT.score(example, None)  # type: ignore[arg-type]
