"""Contrato do fake: eco, tabela e fita. Nenhum modo inventa texto quando o roteiro acaba."""

from __future__ import annotations

import pytest

from eval_harness import FakeSystem, Usage
from eval_harness.systems import FakeScriptError
from tests.helpers import make_example


def test_echo_returns_the_input_and_optional_usage() -> None:
    usage = Usage(input_tokens=2, output_tokens=None)
    system = FakeSystem(usage=usage)
    produced = system(make_example(text="eco"))
    assert produced.output == "eco"
    assert produced.usage == usage
    assert system.params == {"mode": "echo"}


def test_lookup_returns_the_table_and_is_stable() -> None:
    system = FakeSystem("lookup", table={"ex-01": "Brasília", "ex-02": "4"})
    assert system(make_example()).output == "Brasília"
    assert system(make_example()).output == "Brasília"
    assert system.params["table"] == {"ex-01": "Brasília", "ex-02": "4"}


def test_lookup_missing_id_does_not_invent_text() -> None:
    system = FakeSystem("lookup", table={"ex-01": "ok"})
    with pytest.raises(FakeScriptError, match="no scripted output for id ex-02"):
        system(make_example(example_id="ex-02"))


def test_scripted_tape_does_not_rewind() -> None:
    first = FakeSystem("scripted", script=["um", "dois"])
    assert first(make_example()).output == "um"
    assert first(make_example(example_id="ex-02")).output == "dois"
    with pytest.raises(FakeScriptError, match="exhausted"):
        first(make_example(example_id="ex-03"))

    second = FakeSystem("scripted", script=["um", "dois"])
    assert [second(make_example(example_id=f"ex-0{index}")).output for index in (1, 2)] == [
        "um",
        "dois",
    ]


def test_fake_rejects_unknown_mode_and_mismatched_args() -> None:
    with pytest.raises(ValueError, match="unknown fake mode"):
        FakeSystem("random")
    with pytest.raises(ValueError, match="lookup mode requires"):
        FakeSystem("lookup")
    with pytest.raises(ValueError, match="only valid in lookup"):
        FakeSystem("echo", table={"ex-01": "x"})
    with pytest.raises(ValueError, match="sequence of strings"):
        FakeSystem("scripted", script="texto")  # type: ignore[arg-type]
