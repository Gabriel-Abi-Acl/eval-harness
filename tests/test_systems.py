"""Contrato do fake: eco, tabela e fita. Nenhum modo inventa texto quando o roteiro acaba.

oracle, echo e scripted-noisy só escolhem uma fita determinística. Não chamam provedor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_harness import FakeSystem, Usage, load_dataset
from eval_harness.systems import FakeScriptError, build_fake_system
from tests.helpers import PLACEHOLDER, make_example, write_jsonl


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


def _outputs(spec: str, dataset_path: Path, seed: int) -> list[str]:
    dataset = load_dataset(dataset_path)
    system = build_fake_system(spec, dataset, seed=seed)
    return [system(example).output for example in dataset.examples]


def test_named_fakes_repeat_and_diverge(tmp_path: Path) -> None:
    text_path = write_jsonl(
        tmp_path,
        [
            {"id": "a", "input": "in-a", "expected": "alpha beta"},
            {"id": "b", "input": "in-b", "expected": "Lisboa"},
            {"id": "c", "input": "in-c", "expected": "17"},
        ],
        name="text.jsonl",
    )
    gold = '{"produto": "caderno", "quantidade": 1, "urgente": false}'
    json_path = write_jsonl(
        tmp_path,
        [
            {"id": "d", "input": "in-d", "expected": gold},
            {"id": "e", "input": "in-e", "expected": gold},
            {"id": "f", "input": "in-f", "expected": gold},
        ],
        name="json.jsonl",
    )
    text = load_dataset(text_path)
    oracle = build_fake_system("oracle", text, seed=0)
    echo = build_fake_system("echo", text, seed=0)
    assert oracle.name == "oracle"
    assert echo.name == "echo"
    assert oracle(text.examples[0]).output == "alpha beta"
    assert echo(text.examples[0]).output == "in-a"

    assert _outputs("scripted-noisy", text_path, 0) == _outputs("scripted-noisy", text_path, 0)
    assert _outputs("scripted-noisy", text_path, 0) == [
        "alpha beta",
        "Lisboa extra",
        "incorreto",
    ]
    assert _outputs("scripted-noisy", text_path, 1) != _outputs("scripted-noisy", text_path, 0)
    flipped = '{"produto": "caderno", "quantidade": 1, "urgente": true}'
    assert _outputs("scripted-noisy", json_path, 0) == [gold, flipped, "{}"]


def test_named_fake_rejects_unknown_spec_and_bad_seed() -> None:
    dataset = load_dataset(PLACEHOLDER)
    with pytest.raises(ValueError, match="unknown fake system"):
        build_fake_system("random", dataset, seed=0)
    with pytest.raises(ValueError, match="seed must be zero"):
        build_fake_system("echo", dataset, seed=-1)


def test_fake_rejects_unknown_mode_and_mismatched_args() -> None:
    with pytest.raises(ValueError, match="unknown fake mode"):
        FakeSystem("random")
    with pytest.raises(ValueError, match="lookup mode requires"):
        FakeSystem("lookup")
    with pytest.raises(ValueError, match="only valid in lookup"):
        FakeSystem("echo", table={"ex-01": "x"})
    with pytest.raises(ValueError, match="sequence of strings"):
        FakeSystem("scripted", script="texto")  # type: ignore[arg-type]
