"""Tests for gridspan.filters."""

from __future__ import annotations

from gridspan.core import identity
from gridspan.filters import apply, dedup, subsample


def test_dedup_keeps_the_first_of_each_identity():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"runtime.bs": 16, "model.name": "x"}  # same identity, different order
    c = {"model.name": "y", "runtime.bs": 16}
    assert dedup([a, b, c]) == [a, c]


def test_dedup_drops_configs_the_provider_already_finished():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"model.name": "y", "runtime.bs": 16}

    class Provider:
        def existing(self) -> dict[str, str]:
            return {identity(a): "FINISHED"}

    # a is already FINISHED, so only b survives.
    assert dedup([a, b], provider=Provider()) == [b]


def test_dedup_reruns_a_failed_or_interrupted_run_by_default():
    a = {"model.name": "x", "runtime.bs": 16}  # recorded but FAILED
    b = {"model.name": "y", "runtime.bs": 16}  # recorded but RUNNING

    class Provider:
        def existing(self) -> dict[str, str]:
            return {identity(a): "FAILED", identity(b): "RUNNING"}

    # default skip is ("FINISHED",), so both come back to be re-run.
    assert dedup([a, b], provider=Provider()) == [a, b]


def test_dedup_skip_can_include_failed():
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"model.name": "y", "runtime.bs": 16}

    class Provider:
        def existing(self) -> dict[str, str]:
            return {identity(a): "FAILED"}

    assert dedup([a, b], provider=Provider(), skip=("FINISHED", "FAILED")) == [b]


def test_dedup_with_provider_still_removes_within_batch_repeats():
    a = {"model.name": "x", "runtime.bs": 16}
    a2 = {"runtime.bs": 16, "model.name": "x"}  # same identity as a
    b = {"model.name": "y", "runtime.bs": 16}

    class Provider:
        def existing(self) -> dict[str, str]:
            return {}

    assert dedup([a, a2, b], provider=Provider()) == [a, b]


def test_subsample_by_count():
    cfgs = [{"i": i} for i in range(10)]
    out = subsample(cfgs, n=3, seed=0)
    assert len(out) == 3
    assert all(item in cfgs for item in out)


def test_subsample_by_fraction():
    cfgs = [{"i": i} for i in range(10)]
    out = subsample(cfgs, fraction=0.5, seed=0)
    assert len(out) == 5


def test_apply_chains_filters():
    cfgs = [{"i": 1}, {"i": 1}, {"i": 2}, {"i": 3}]
    out = apply(cfgs, dedup, lambda xs: xs[:2])
    assert out == [{"i": 1}, {"i": 2}]
