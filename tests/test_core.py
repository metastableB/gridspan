"""Tests for gridspan.core."""

from __future__ import annotations

import subprocess
import sys
from datetime import date, datetime

import pytest

from gridspan.core import (
    GridSpan,
    GridSpec,
    from_dict,
    from_yaml,
    identity,
)


def expand(spec, key_sep="."):
    return from_dict(spec, key_sep=key_sep).expand()


def test_scalar_only_gives_one_point():
    assert expand({"a": 1, "b": 2}) == [{"a": 1, "b": 2}]


def test_expand_returns_gridspan():
    out = expand({"a": [1, 2]})
    assert isinstance(out, GridSpan)


@pytest.mark.parametrize("key_sep", [".", "/"])
def test_stamp_is_explicit_and_refreshes_hashes_in_place(key_sep):
    span = from_dict(
        {"model": {"name": ["x", "x"]}, "tools": [["search"]]},
        key_sep=key_sep,
    ).expand()
    hash_key = key_sep.join(("gridspan", "id", "hash"))
    assert all(hash_key not in cfg for cfg in span)
    point = span[0]
    assert span.stamp() is span
    assert span[0] is point
    assert len(span) == 2
    assert all(cfg[hash_key] == identity(cfg, key_sep=key_sep) for cfg in span)
    original_hash = point[hash_key]
    point["tools"].append("python")
    assert point[hash_key] == original_hash
    span.stamp()
    assert point[hash_key] == identity(point, key_sep=key_sep)
    assert point[hash_key] != original_hash


def test_stamp_accepts_an_empty_span():
    span = GridSpan()
    assert span.stamp() is span
    assert span == []


@pytest.mark.parametrize("constructor", [GridSpec, from_dict])
def test_gridspec_stores_separator_at_creation(constructor):
    spec = constructor(
        {
            "model": {"name": ["x", "y"]},
            "gridspan": {"id": {"include": ["model/name"]}},
        },
        key_sep="/",
    )
    assert spec.key_sep == "/"
    span = spec.expand().stamp()
    assert span.key_sep == "/"
    assert [point["model/name"] for point in span] == ["x", "y"]
    assert all("gridspan/id/hash" in point for point in span)


@pytest.mark.parametrize("constructor", [GridSpec, from_dict])
def test_gridspec_rejects_conflicting_selectors_at_creation(constructor):
    with pytest.raises(ValueError, match="mutually exclusive"):
        constructor(
            {
                "model": ["x", "y"],
                "gridspan": {"id": {"include": ["model"], "exclude": []}},
            }
        )


@pytest.mark.parametrize("constructor", [GridSpec, from_dict])
@pytest.mark.parametrize(
    "data, key_sep, error, message",
    [
        ([], ".", TypeError, "dictionary"),
        ({1: "x"}, ".", TypeError, "key must be a string"),
        ({"outer": {"": 1}}, ".", ValueError, "key names must not be empty"),
        ({"a/b": 1}, "/", ValueError, "key contains separator"),
        ({"model": "x"}, "::", ValueError, "single-character"),
    ],
)
def test_gridspec_checks_shape_and_separator_at_creation(
    constructor, data, key_sep, error, message
):
    with pytest.raises(error, match=message):
        constructor(data, key_sep=key_sep)


def test_from_yaml_validates_selectors_at_creation(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("model: [x, y]\ngridspan:\n  id:\n    include: [modle]\n")
    with pytest.raises(ValueError, match="unknown key 'modle'"):
        from_yaml(spec_file)


def test_gridspec_revalidates_after_nested_mutation():
    spec = from_dict({"model": ["x", "y"], "gridspan": {"id": {"include": ["model"]}}})
    spec["gridspan"]["id"]["exclude"] = []
    with pytest.raises(ValueError, match="mutually exclusive"):
        spec.expand()


def test_gridspec_revalidates_keys_after_nested_mutation():
    spec = from_dict({"model": {"name": "x"}}, key_sep="/")
    spec["model"]["invalid/name"] = "y"
    with pytest.raises(ValueError, match="key contains separator"):
        spec.expand()


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


def test_expand_allows_custom_key_separator():
    out = expand({"model": {"name": ["a", "b"]}, "runtime": {"bs": 16}}, key_sep="/")
    assert out == [
        {"model/name": "a", "runtime/bs": 16},
        {"model/name": "b", "runtime/bs": 16},
    ]


def test_expand_keeps_dot_in_literal_key_with_custom_separator():
    out = expand({"ke.eye": [1, 2], "runtime": {"bs": 16}}, key_sep="/")
    assert out == [
        {"ke.eye": 1, "runtime/bs": 16},
        {"ke.eye": 2, "runtime/bs": 16},
    ]


@pytest.mark.parametrize(
    "spec",
    [
        {"gridspan.note": [1, 2]},
        {"gridspan:note": 1},
        {"gridspan": {"id.exclude": ["x"]}, "x": 1},
        {"gridspan": {"id-include": ["x"]}, "x": 1},
    ],
)
def test_custom_separator_rejects_metadata_with_another_separator(spec):
    with pytest.raises(ValueError, match="with another separator"):
        from_dict(spec, key_sep="/")


@pytest.mark.parametrize("key", ["gridspanner", "gridspan_runs", "grid.span"])
def test_keys_that_only_resemble_gridspan_are_parameters(key):
    span = from_dict({key: [1, 2], "gridspan": {"idle": True}}, key_sep="/").expand()
    assert [point[key] for point in span] == [1, 2]


def test_custom_separator_preserves_metadata_meaning():
    span = from_dict(
        {"x": [1, 2], "gridspan": {"id": {"exclude": ["x"]}}}, key_sep="/"
    ).expand()
    assert span == [
        {"x": 1, "gridspan/id/exclude": ["x"]},
        {"x": 2, "gridspan/id/exclude": ["x"]},
    ]
    assert len(span.dedup()) == 1


@pytest.mark.parametrize("key_sep", [".", "/"])
@pytest.mark.parametrize("selector", ["include", "exclude"])
def test_metadata_separator_survives_stamping_and_chaining(key_sep, selector):
    selected_keys = ["model"] if selector == "include" else ["seed"]
    span = from_dict(
        {
            "model": ["x", "y"],
            "seed": [0, 1],
            "gridspan": {
                "id": {selector: selected_keys},
                "notes": ["first", "second"],
            },
        },
        key_sep=key_sep,
    ).expand().stamp()
    assert len(span) == 4
    out = span.apply(list, lambda cfgs: cfgs.dedup()).subsample(n=2).dedup()
    assert out.key_sep == key_sep
    assert len(out) == 2
    assert {point["model"] for point in out} == {"x", "y"}
    hash_key = key_sep.join(("gridspan", "id", "hash"))
    for point in out:
        assert point[hash_key] == identity(point, key_sep=key_sep)
    assert len(GridSpan(list(span), key_sep=key_sep).dedup()) == 2


def test_expand_rejects_non_string_keys():
    with pytest.raises(TypeError, match="key must be a string"):
        expand({1: "x"})


@pytest.mark.parametrize(
    "spec",
    [{"": 1}, {"": {"x": 1}}, {"section": {"": 1}}, {"gridspan": {"": 1}}],
)
def test_expand_rejects_empty_key_names(spec):
    with pytest.raises(ValueError, match="key names must not be empty"):
        expand(spec)


def test_expand_rejects_key_containing_active_separator():
    with pytest.raises(ValueError, match="key contains separator"):
        expand({"a/b": 1}, key_sep="/")


def test_expand_rejects_nested_key_containing_active_separator():
    with pytest.raises(ValueError, match="key contains separator"):
        expand({"outer": {"a/b": 1}}, key_sep="/")


def test_regression_default_separator_disallows_literal_dotted_keys():
    # Reversibility: with key_sep='.', a literal "a.b" key is ambiguous.
    with pytest.raises(ValueError, match="key contains separator"):
        expand({"a.b": 1})


def test_regression_custom_separator_disallows_flat_reserved_alias_key():
    # Metadata keys are path segments and cannot contain the active separator.
    with pytest.raises(ValueError, match="key contains separator"):
        expand({"gridspan/id/exclude": ["a"], "a": [1, 2]}, key_sep="/")


def test_expand_reserved_keys_use_nested_segments_with_default_separator():
    out = expand({"a": [1, 2], "gridspan": {"id": {"exclude": ["a"]}}})
    assert len(out) == 2
    assert all(point["gridspan.id.exclude"] == ["a"] for point in out)


@pytest.mark.parametrize("key_sep", ["", "::", "__", None, 1])
def test_gridspec_requires_a_single_character_separator(key_sep):
    with pytest.raises(ValueError, match="key_sep must be a single-character string"):
        GridSpec({"model": {"name": "x"}}, key_sep=key_sep)


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


@pytest.mark.parametrize("empty_value", [{}, []])
@pytest.mark.parametrize("key_sep", [".", "/"])
def test_expand_skips_empty_parameters_with_a_warning(empty_value, key_sep):
    spec = {
        "left": [1, 2],
        "outer": {"section": empty_value},
        "right": ["a", "b"],
    }
    with pytest.warns(UserWarning, match="skipping empty") as caught:
        out = expand(spec, key_sep=key_sep)
    assert len(caught) == 1
    assert repr(f"outer{key_sep}section") in str(caught[0].message)
    assert out == [
        {"left": 1, "right": "a"},
        {"left": 1, "right": "b"},
        {"left": 2, "right": "a"},
        {"left": 2, "right": "b"},
    ]


def test_expand_warns_when_all_parameters_are_empty():
    with pytest.warns(UserWarning, match="skipping empty") as caught:
        out = expand({"empty_dict": {}, "empty_list": []})
    assert len(caught) == 2
    assert out == [{}]


@pytest.mark.parametrize("call", [
    lambda spec: from_dict(spec).expand(),
    lambda spec: GridSpec(spec).expand(),
])
def test_empty_value_warning_points_at_the_callers_line(call):
    with pytest.warns(UserWarning, match="skipping empty") as caught:
        call({"a": [1], "b": []})
    assert caught[0].filename == __file__


def test_each_point_gets_its_own_copy_of_its_values():
    spec = {
        "layers": [[64, 64]],
        "optimizer": [{"name": "adam"}],
        "seed": [0, 1],
        "gridspan": {"id": {"exclude": ["seed"]}},
    }
    first, second = expand(spec)
    first["layers"].append(1)
    first["optimizer"]["name"] = "sgd"
    first["gridspan.id.exclude"].append("layers")
    assert second["layers"] == [64, 64]
    assert second["optimizer"] == {"name": "adam"}
    assert second["gridspan.id.exclude"] == ["seed"]
    assert spec["layers"] == [[64, 64]]


def test_expand_keeps_explicit_empty_values_and_metadata(recwarn):
    out = expand(
        {
            "dict_value": [{}],
            "list_value": [[]],
            "count": 0,
            "enabled": False,
            "text": "",
            "optional": None,
            "gridspan": {"id": {"include": []}, "notes": []},
        }
    )
    assert out == [
        {
            "dict_value": {},
            "list_value": [],
            "count": 0,
            "enabled": False,
            "text": "",
            "optional": None,
            "gridspan.id.include": [],
            "gridspan.notes": [],
        }
    ]
    assert not recwarn


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


def test_identity_rejects_include_and_exclude_together():
    cfg = {
        "model.name": "x",
        "runtime.bs": 16,
        "gridspan.id.include": ["model.name"],
        "gridspan.id.exclude": ["runtime.bs"],
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        identity(cfg)


@pytest.mark.parametrize("selector", ["include", "exclude"])
@pytest.mark.parametrize("key_sep", [".", "/"])
def test_gridspec_rejects_unknown_identity_keys(selector, key_sep):
    unknown_key = key_sep.join(("model", "nmae"))
    with pytest.raises(ValueError, match="unknown key") as error:
        from_dict(
            {
                "model": {"name": ["x", "y"]},
                "gridspan": {"id": {selector: [unknown_key]}},
            },
            key_sep=key_sep,
        )
    assert key_sep.join(("gridspan", "id", selector)) in str(error.value)
    assert unknown_key in str(error.value)


@pytest.mark.parametrize("selector", ["include", "exclude"])
@pytest.mark.parametrize("invalid_value", ["model", None, ["model", 1], {}, {"key": "model"}])
def test_identity_rejects_invalid_selector_types(selector, invalid_value):
    cfg = {"model": "x", f"gridspan.id.{selector}": invalid_value}
    with pytest.raises(TypeError, match="must be a list of strings"):
        identity(cfg)
    with pytest.raises(TypeError, match="must be a list of strings"):
        from_dict({"model": "x", "gridspan": {"id": {selector: invalid_value}}})


def test_identity_accepts_empty_selector_lists():
    assert identity({"model": "x", "gridspan.id.include": []}) == identity({})
    assert identity({"model": "x", "gridspan.id.exclude": []}) == identity(
        {"model": "x"}
    )


def test_identity_ignores_its_own_reserved_keys():
    a = {"model.name": "x"}
    b = {"model.name": "x", "gridspan.id.hash": "deadbeef"}
    assert identity(a) == identity(b)


def test_identity_ignores_all_gridspan_metadata_keys():
    a = {"model.name": "x"}
    b = {"model.name": "x", "gridspan.meta.note": "hello"}
    assert identity(a) == identity(b)


# Saved stores rely on these exact values. If this test fails, the hash format
# changed: bump the minor version and describe the change in the release notes.
@pytest.mark.parametrize(
    "cfg, key_sep, expected",
    [
        ({"lr": 0.1, "model": "gpt-4", "seed": 0}, ".",
         "356987f0cad7e999f2b5ac29cb4e898c"),
        ({"layers": [64, 64], "opt": {"name": "adam", "b": [0.9, 0.99]}}, ".",
         "d47cf7b28f3ebe7f298e8435b1052c6a"),
        ({"flag": True, "none": None, "text": "", "n": -3, "x": 1e-5}, ".",
         "92b2432211708a2e6bf9b984dd73129c"),
        ({"day": date(2026, 1, 1), "at": datetime(2026, 1, 1, 12, 30)}, ".",
         "ae0fa10b8fd81f571dbd57459965ae77"),
        ({"model": "x", "seed": 1, "gridspan.id.exclude": ["seed"]}, ".",
         "ad3d23c6eab84a431b33ecfa993c0697"),
        ({"model/name": "x", "gridspan/id/include": ["model/name"], "seed": 2}, "/",
         "e302b04f2effea7be0da00cbbf551b98"),
        ({}, ".", "99914b932bd37a50b983c5e7c90ae93b"),
    ],
)
def test_identity_hash_values_do_not_change(cfg, key_sep, expected):
    assert identity(cfg, key_sep=key_sep) == expected


def test_identity_of_a_set_is_the_same_in_every_process():
    code = (
        "from gridspan.core import identity; "
        "print(identity({'a': {'w', 'x', 'y', 'z', frozenset({1, 'b'})}}))"
    )
    hashes = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": str(seed)},
        ).stdout
        for seed in range(5)
    }
    assert len(hashes) == 1


@pytest.mark.parametrize(
    "value, message",
    [
        (object(), "object values have no stable hash"),
        ({1: "x", "b": 2}, "not supported between"),
    ],
)
def test_identity_names_the_key_it_cannot_hash(value, message):
    with pytest.raises(TypeError, match=f"cannot hash 'bad': .*{message}"):
        identity({"good": 1, "bad": value})


def test_stamp_fills_hash_per_point():
    out = expand({"a": [1, 2], "b": 3}).stamp()
    assert len(out) == 2
    for point in out:
        assert point["gridspan.id.hash"] == identity(point)
    assert out[0]["gridspan.id.hash"] != out[1]["gridspan.id.hash"]


def test_expand_passes_reserved_keys_through_whole():
    # A reserved key's list value must not be swept as an axis.
    out = expand({"a": [1, 2], "gridspan": {"id": {"exclude": ["a"]}}})
    assert len(out) == 2
    for point in out:
        assert point["gridspan.id.exclude"] == ["a"]


def test_expand_then_exclude_drops_a_key_from_identity():
    # The real user path: put exclude in the spec, expand, then dedup.
    spec = {
        "model": ["x", "y"],
        "retries": [3, 5],
        "gridspan": {"id": {"exclude": ["retries"]}},
    }
    out = expand(spec)
    assert len(out) == 4  # 2 models x 2 retries
    # retries is excluded from identity, so each model's two retry variants
    # collapse to one: 2 distinct identities, not 4.
    assert len({identity(p) for p in out}) == 2


def test_expand_then_include_narrows_identity():
    spec = {
        "model": ["x", "y"],
        "seed": [0, 1],
        "gridspan": {"id": {"include": ["model"]}},
    }
    out = expand(spec)
    assert len(out) == 4
    # only model counts for identity, so seed does not split it.
    assert len({identity(p) for p in out}) == 2


def test_expand_reserved_keys_work_with_custom_separator_nested_form():
    spec = {"model": ["x", "y"], "retries": [1, 2], "gridspan": {"id": {"exclude": ["retries"]}}}
    out = expand(spec, key_sep="/")
    assert len(out) == 4
    assert all("gridspan/id/exclude" in point for point in out)
    assert len(out.stamp().dedup()) == 2


def test_from_yaml_loads_a_spec_without_expanding(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "model:\n  name:\n    - a\n    - b\nruntime:\n  bs: 16\n"
    )
    # from_yaml returns the nested GridSpec, unexpanded.
    assert from_yaml(spec_file) == {"model": {"name": ["a", "b"]}, "runtime": {"bs": 16}}


@pytest.mark.parametrize("text", ["false", "0", "[]", "''", "text", "- [model, x]"])
def test_from_yaml_rejects_non_dictionary_roots(tmp_path, text):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(text)
    with pytest.raises(TypeError, match="YAML root must be a dictionary"):
        from_yaml(spec_file)


@pytest.mark.parametrize("text", ["", "# comment only\n", "null", "{}"])
def test_from_yaml_accepts_empty_or_null_documents(tmp_path, text):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(text)
    spec = from_yaml(spec_file)
    assert spec == {}
    assert spec.expand() == [{}]


def test_from_yaml_keeps_value_types_with_identity_string_fallback(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("value: [2026-01-01, '2026-01-01']\ncount: 2\n")
    spec = from_yaml(spec_file)
    assert spec["value"] == [date(2026, 1, 1), "2026-01-01"]
    assert spec["count"] == 2
    span = spec.expand()
    assert identity(span[0]) == identity(span[1])
    assert len(span.stamp().dedup()) == 1


def test_from_yaml_supports_chained_expand(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("a:\n  - 1\n  - 2\nb: 3\n")
    out = from_yaml(spec_file).expand()
    assert out == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_from_yaml_stores_custom_separator_without_expanding(tmp_path):
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(
        "model:\n  name: [x, y]\ngridspan:\n  id:\n    include: [model/name]\n"
    )
    spec = from_yaml(spec_file, key_sep="/")
    assert spec["model"] == {"name": ["x", "y"]}
    assert spec.key_sep == "/"
    span = spec.expand()
    assert span.key_sep == "/"
    assert [point["model/name"] for point in span] == ["x", "y"]


def test_from_dict_supports_chained_expand():
    out = from_dict({"a": [1, 2], "b": 3}).expand()
    assert out == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_gridspan_supports_chained_apply_style():
    span = from_dict({"a": [1, 1, 2], "b": [3]}).expand()
    out = span.apply(lambda xs: xs[:2]).apply(lambda xs: xs + [xs[-1]])
    assert isinstance(out, GridSpan)
    assert out == [{"a": 1, "b": 3}, {"a": 1, "b": 3}, {"a": 1, "b": 3}]


def test_gridspan_supports_chained_dedup_and_subsample():
    span = from_dict({"a": [1, 1, 2, 3], "b": [3]}).expand()
    out = span.stamp().dedup().subsample(n=2, seed=0)
    assert isinstance(out, GridSpan)
    assert len(out) == 2


def test_gridspan_subsample_zero_keeps_separator():
    span = from_dict({"model": {"name": ["x", "y"]}}, key_sep="/").expand()
    out = span.subsample(n=0)
    assert isinstance(out, GridSpan)
    assert out == []
    assert out.key_sep == "/"
    assert len(span) == 2


@pytest.mark.parametrize("count", [-1, 3])
def test_gridspan_subsample_uses_standard_sample_bounds(count):
    span = from_dict({"model": ["x", "y"]}).expand()
    with pytest.raises(ValueError):
        span.subsample(n=count)


@pytest.mark.parametrize("key_sep", [".", "/"])
def test_to_argv_keeps_the_span_separator_in_flags(key_sep):
    span = expand({"model": {"name": ["a", "b"]}, "runtime_bs": 2}, key_sep=key_sep)
    assert span.to_argv() == [
        [f"--model{key_sep}name", "a", "--runtime_bs", "2"],
        [f"--model{key_sep}name", "b", "--runtime_bs", "2"],
    ]


def test_to_argv_renders_every_value_uniformly():
    # No special cases: bools and everything else are just str(value).
    span = GridSpan([{"verbose": True, "dry_run": False, "n": 3}])
    assert span.to_argv() == [["--verbose", "True", "--dry_run", "False", "--n", "3"]]


@pytest.mark.parametrize("key_sep", [".", "/"])
def test_to_argv_skips_metadata(key_sep):
    span = expand(
        {
            "model": {"name": "x"},
            "gridspan": {"id": {"exclude": ["seed"]}, "note": "hi"},
            "seed": 1,
        },
        key_sep=key_sep,
    ).stamp()
    assert span.to_argv() == [[f"--model{key_sep}name", "x", "--seed", "1"]]


def test_to_argv_of_an_empty_span_is_empty():
    assert GridSpan().to_argv() == []
