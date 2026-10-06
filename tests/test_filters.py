"""Tests for GridSpan filters and stores."""

from __future__ import annotations

import random
from unittest.mock import Mock

import pytest

from gridspan import TextStore, from_dict
from gridspan.core import GridSpan, identity


def write_hashes(path, hashes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(h + "\n" for h in hashes))


def test_dedup_accepts_store_object(tmp_path):
    path = tmp_path / "done.txt"
    span = GridSpan([{"model": "x"}, {"model": "y"}])
    write_hashes(path, [identity(span[0])])
    store = TextStore(path)
    assert span.dedup(store=store) == [span[1]]
    assert store([identity(span[0]), "unknown"]) == {identity(span[0])}


def test_dedup_accepts_plain_function_as_store():
    def lookup(run_hashes):
        return [h for h in run_hashes if h == identity({"model": "x"})]

    span = GridSpan([{"model": "x"}, {"model": "y"}])
    assert span.dedup(store=lookup) == [span[1]]


@pytest.mark.parametrize("key_sep", [".", "/"])
def test_dedup_stamps_before_asking_the_store(key_sep):
    span = GridSpan([{"model": "x"}], key_sep=key_sep).stamp()
    span[0]["model"] = "z"
    span.append({"model": "y"})
    hash_key = key_sep.join(("gridspan", "id", "hash"))

    def lookup(run_hashes):
        assert all(cfg[hash_key] == identity(cfg, key_sep=key_sep) for cfg in span)
        assert run_hashes == [cfg[hash_key] for cfg in span]
        return {span[0][hash_key]}

    store = Mock(side_effect=lookup)
    out = span.dedup(store=store)
    assert out == [span[-1]]
    assert out.key_sep == key_sep
    store.assert_called_once()


def test_dedup_asks_store_once_for_unique_hashes():
    span = GridSpan([{"model": "x"}, {"model": "x"}, {"model": "y"}]).stamp()
    hashes = [span[0]["gridspan.id.hash"], span[2]["gridspan.id.hash"]]
    store = Mock(return_value={hashes[0]})
    assert span.dedup(store=store) == [span[-1]]
    store.assert_called_once_with(hashes)


def test_dedup_rejects_a_store_that_returns_a_string():
    span = GridSpan([{"model": "x"}])
    with pytest.raises(TypeError, match="collection of hashes"):
        span.dedup(store=lambda run_hashes: run_hashes[0])


def test_dedup_names_the_store_when_it_returns_none():
    span = GridSpan([{"model": "x"}])
    with pytest.raises(TypeError, match="store must return a collection of hashes, got NoneType"):
        span.dedup(store=lambda run_hashes: None)


def test_dedup_keeps_the_first_of_each_identity():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"runtime.bs": 16, "model.name": "x"}  # same identity, different order
    c = {"model.name": "y", "runtime.bs": 16}
    assert GridSpan([a, b, c]).dedup() == [a, c]


def test_dedup_collapses_a_grid_with_an_excluded_key():
    # An excluded key varies but must not create distinct runs.
    spec = {
        "model": ["x", "y"],
        "retries": [3, 5],
        "gridspan": {"id": {"exclude": ["retries"]}},
    }
    out = from_dict(spec).expand().dedup()
    assert len(out) == 2  # one per model; retries does not split identity
    assert [c["model"] for c in out] == ["x", "y"]


def test_dedup_drops_points_the_store_returns():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"model.name": "y", "runtime.bs": 16}
    span = GridSpan([a, b]).stamp()
    store = Mock(return_value={span[0]["gridspan.id.hash"]})
    assert span.dedup(store=store) == [b]


def test_dedup_ignores_returned_hashes_outside_the_span():
    span = GridSpan([{"model": "x"}]).stamp()
    store = Mock(return_value={"some-other-hash"})
    assert span.dedup(store=store) == [span[0]]


def test_dedup_with_store_still_removes_within_batch_repeats():
    a = {"model.name": "x", "runtime.bs": 16}
    a2 = {"runtime.bs": 16, "model.name": "x"}  # same identity as a
    b = {"model.name": "y", "runtime.bs": 16}

    span = GridSpan([a, a2, b]).stamp()
    store = Mock(return_value=set())

    assert span.dedup(store=store) == [a, b]


def test_dedup_error_policy_raises_on_a_within_batch_repeat():
    a = {"model.name": "x", "runtime.bs": 16}
    a2 = {"runtime.bs": 16, "model.name": "x"}  # same identity as a
    with pytest.raises(ValueError, match="duplicate run .*: it repeats"):
        GridSpan([a, a2]).dedup(policy="error")


def test_dedup_error_policy_raises_on_a_recorded_run(tmp_path):
    a = {"model.name": "x", "runtime.bs": 16}
    span = GridSpan([a]).stamp()
    path = tmp_path / "done.txt"
    write_hashes(path, [span[0]["gridspan.id.hash"]])
    store = TextStore(path)
    with pytest.raises(ValueError, match="duplicate run .*: the store returned it"):
        span.dedup(store=store, policy="error")


def test_dedup_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="unknown policy"):
        GridSpan([{"a": 1}]).dedup(policy="nope")


@pytest.mark.parametrize("stored_hash", [None, "", 123, "outdated"])
@pytest.mark.parametrize("key_sep", [".", "/"])
def test_dedup_replaces_old_stamps(stored_hash, key_sep):
    hash_key = key_sep.join(("gridspan", "id", "hash"))
    point = {"model": "x", hash_key: stored_hash}
    duplicate = {"model": "x"}
    assert GridSpan([point, duplicate], key_sep=key_sep).dedup() == [point]
    assert point[hash_key] == identity(point, key_sep=key_sep)
    assert duplicate[hash_key] == point[hash_key]


@pytest.mark.parametrize("key_sep", [".", "/"])
def test_dedup_refreshes_hashes_after_edits(key_sep):
    span = GridSpan([{"value": 1}, {"value": 2}], key_sep=key_sep).stamp()
    hash_key = key_sep.join(("gridspan", "id", "hash"))
    original_hash = span[1][hash_key]
    span[1]["value"] = 1
    assert span.dedup() == [span[0]]
    assert span[1][hash_key] != original_hash
    assert span[1][hash_key] == span[0][hash_key]


def test_dedup_validates_identity_before_asking_the_store():
    span = GridSpan([{"model": "x", "gridspan.id.include": ["modle"]}])
    store = Mock(return_value=set())
    with pytest.raises(ValueError, match="unknown key"):
        span.dedup(store=store)
    store.assert_not_called()


def test_dedup_propagates_store_errors():
    span = GridSpan([{"value": 1}]).stamp()
    store = Mock(side_effect=RuntimeError("store unavailable"))
    with pytest.raises(RuntimeError, match="store unavailable"):
        span.dedup(store=store)
    store.assert_called_once()


def test_text_store_starts_empty_without_creating_a_file(tmp_path):
    path = tmp_path / "done.txt"
    assert TextStore(path)(["any-hash"]) == set()
    assert not path.exists()


@pytest.mark.parametrize("key_sep", [".", "/"])
def test_text_store_resumes_listed_runs(tmp_path, key_sep):
    path = tmp_path / "state" / "done.txt"
    span = GridSpan([{"model": "x"}, {"model": "y"}], key_sep=key_sep).stamp()
    hash_key = key_sep.join(("gridspan", "id", "hash"))
    write_hashes(path, [span[0][hash_key]])
    fresh_span = GridSpan([{"model": "x"}, {"model": "y"}], key_sep=key_sep)
    assert fresh_span.dedup(store=TextStore(path)) == [span[1]]


def test_text_store_ignores_blank_lines_and_surrounding_spaces(tmp_path):
    path = tmp_path / "done.txt"
    path.write_text("\n  hash-a  \n\nhash-b")
    assert TextStore(path)(["hash-a", "hash-b", "hash-c"]) == {"hash-a", "hash-b"}


def test_text_store_reads_file_once(tmp_path):
    path = tmp_path / "done.txt"
    write_hashes(path, ["hash"])
    store = TextStore(path)
    assert store(["hash"]) == {"hash"}
    path.unlink()
    assert store(["hash"]) == {"hash"}


def test_text_store_line_cut_short_by_a_crash_skips_nothing_extra(tmp_path):
    path = tmp_path / "done.txt"
    path.write_text("hash-a\nhash-b-cut-sho")
    assert TextStore(path)(["hash-a", "hash-b-cut-short"]) == {"hash-a"}


def test_subsample_by_count():
    span = GridSpan({"i": i} for i in range(10))
    out = span.subsample(n=3, seed=0)
    assert len(out) == 3
    assert all(item in span for item in out)


@pytest.mark.parametrize("count", [0, 2, 3])
def test_subsample_matches_random_sample(count):
    span = GridSpan({"i": i} for i in range(3))
    assert span.subsample(n=count, seed=7) == random.Random(7).sample(span, k=count)


@pytest.mark.parametrize("count", [-1, 4])
def test_subsample_rejects_out_of_range_counts(count):
    with pytest.raises(ValueError):
        GridSpan({"i": i} for i in range(3)).subsample(n=count)


def test_subsample_preserves_duplicate_occurrences():
    assert GridSpan([{"i": 1}, {"i": 1}]).subsample(n=2) == [{"i": 1}, {"i": 1}]


def test_subsample_can_sample_zero_from_empty_span():
    assert GridSpan().subsample(n=0) == []


def test_apply_chains_filters():
    cfgs = GridSpan([{"i": 1}, {"i": 1}, {"i": 2}, {"i": 3}])
    out = cfgs.apply(lambda xs: xs.dedup(), lambda xs: xs[:2])
    assert out == [cfgs[0], cfgs[2]]
