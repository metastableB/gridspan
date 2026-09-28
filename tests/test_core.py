"""Tests for gridspan.core."""

from __future__ import annotations

import pytest

from gridspan.core import expand, from_yaml, identity, to_argv


def test_scalar_only_gives_one_point():
    assert expand({"a": 1, "b": 2}) == [{"a": 1, "b": 2}]


def test_flat_grid_takes_the_product():
    out = expand({"a": [1, 2], "b": [3, 4]})
    assert {frozenset(d.items()) for d in out} == {
        frozenset({("a", 1), ("b", 3)}),
        frozenset({("a", 1), ("b", 4)}),
        frozenset({("a", 2), ("b", 3)}),
        frozenset({("a", 2), ("b", 4)}),
    }


def test_nested_keys_flatten_to_dotted():
    out = expand({"model": {"name": ["a", "b"]}, "runtime": {"bs": 16}})
    assert out == [
        {"model.name": "a", "runtime.bs": 16},
        {"model.name": "b", "runtime.bs": 16},
    ]


def test_set_of_sets_keeps_inner_lists_whole():
    out = expand({"name": [["A", "B"], "C", ["D"]]})
    assert out == [
        {"name": ["A", "B"]},
        {"name": "C"},
        {"name": ["D"]},
    ]


def test_singleton_dicts_travel_together():
    out = expand(
        {
            "sampling": [
                {"model": "gpt4", "tokens": 8192},
                {"model": "qwen", "tokens": 4096},
            ]
        }
    )
    assert out == [
        {"sampling": {"model": "gpt4", "tokens": 8192}},
        {"sampling": {"model": "qwen", "tokens": 4096}},
    ]


def test_empty_axis_is_an_error():
    with pytest.raises(ValueError, match="empty axis"):
        expand({"a": [1, 2], "b": []})


def test_identity_is_stable_across_key_order():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"runtime.bs": 16, "model.name": "x"}
    assert identity(a) == identity(b)


def test_identity_differs_when_a_value_differs():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"model.name": "x", "runtime.bs": 32}
    assert identity(a) != identity(b)


def test_identity_include_narrows_the_scope():
    a = {"model.name": "x", "runtime.bs": 16, "gridspan.id.include": ["model.name"]}
    b = {"model.name": "x", "runtime.bs": 32, "gridspan.id.include": ["model.name"]}
    assert identity(a) == identity(b)


def test_identity_exclude_drops_keys():
    a = {"model.name": "x", "runtime.bs": 16, "gridspan.id.exclude": ["runtime.bs"]}
    b = {"model.name": "x", "runtime.bs": 32, "gridspan.id.exclude": ["runtime.bs"]}
    assert identity(a) == identity(b)


def test_identity_ignores_its_own_reserved_keys():
    a = {"model.name": "x"}
    b = {"model.name": "x", "gridspan.id.hash": "deadbeef"}
    assert identity(a) == identity(b)


def test_expand_stamp_flag_fills_hash_per_point():
    out = expand({"a": [1, 2], "b": 3}, stamp=True)
    assert len(out) == 2
    for point in out:
        assert point["gridspan.id.hash"] == identity(point)
    assert out[0]["gridspan.id.hash"] != out[1]["gridspan.id.hash"]


def test_expand_passes_reserved_keys_through_whole():
    # A reserved key's list value must not be swept as an axis.
    out = expand({"a": [1, 2], "gridspan.id.exclude": ["a"]})
    assert len(out) == 2
    for point in out:
        assert point["gridspan.id.exclude"] == ["a"]


def test_expand_then_exclude_drops_a_key_from_identity():
    # The real user path: put exclude in the spec, expand, then dedup.
    spec = {"model": ["x", "y"], "retries": [3, 5], "gridspan.id.exclude": ["retries"]}
    out = expand(spec)
    assert len(out) == 4  # 2 models x 2 retries
    # retries is excluded from identity, so each model's two retry variants
    # collapse to one: 2 distinct identities, not 4.
    assert len({identity(p) for p in out}) == 2


def test_expand_then_include_narrows_identity():
    spec = {"model": ["x", "y"], "seed": [0, 1], "gridspan.id.include": ["model"]}
    out = expand(spec)
    assert len(out) == 4
    # only model counts for identity, so seed does not split it.
    assert len({identity(p) for p in out}) == 2


def test_from_yaml_reads_and_expands(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "model:\n  name:\n    - a\n    - b\nruntime:\n  bs: 16\n"
    )
    out = from_yaml(spec_file)
    assert out == [
        {"model.name": "a", "runtime.bs": 16},
        {"model.name": "b", "runtime.bs": 16},
    ]


def test_from_yaml_stamp_passthrough(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("a:\n  - 1\n  - 2\n")
    out = from_yaml(spec_file, stamp=True)
    for point in out:
        assert point["gridspan.id.hash"] == identity(point)


def test_to_argv_dashes_dotted_keys_by_default():
    argv = to_argv({"model.name": "gpt-4", "trainer.lr": 0.1})
    assert argv == ["--model-name", "gpt-4", "--trainer-lr", "0.1"]


def test_to_argv_translates_only_the_dotted_separator():
    # The "." is the flatten separator; an underscore in a key name is kept.
    assert to_argv({"runtime.num_gpus": 2}) == ["--runtime-num_gpus", "2"]


def test_to_argv_sep_controls_the_separator():
    cfg = {"model.name": "gpt-4"}
    assert to_argv(cfg, sep="_") == ["--model_name", "gpt-4"]
    assert to_argv(cfg, sep=".") == ["--model.name", "gpt-4"]


def test_to_argv_bool_true_is_a_bare_flag_false_is_skipped():
    argv = to_argv({"verbose": True, "dry_run": False, "n": 3})
    assert argv == ["--verbose", "--n", "3"]


def test_to_argv_skips_reserved_keys():
    argv = to_argv({"model": "x", "gridspan.id.hash": "deadbeef"})
    assert argv == ["--model", "x"]


def test_to_argv_key_override_remaps_one_flag():
    def key(k):
        return {"runtime.num_gpus": "--gpus"}.get(k, "--" + k.replace(".", "-"))

    argv = to_argv({"model.name": "gpt-4", "runtime.num_gpus": 2}, key=key)
    assert argv == ["--model-name", "gpt-4", "--gpus", "2"]


def test_to_argv_value_override_joins_a_list():
    argv = to_argv({"tools": ["a", "b"]}, value=lambda v: [",".join(v)])
    assert argv == ["--tools", "a,b"]
