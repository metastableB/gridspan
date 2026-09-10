"""Filters over lists of RunCfgs — list -> list helpers to prune a grid.

Core is dict-in, dict-out. This layer stays at the same level: it just
gives you a few list -> list helpers to prune a grid before you run it.

    dedup       drop RunCfgs already recorded by a provider, and repeats
                within the batch.
    subsample   take a random subset by count or fraction.
    apply       chain filters left to right.

    StatusProvider: Protocol implementation to store run stats on disk.
"""

from __future__ import annotations

import random
from typing import Protocol

from gridspan.core import RunCfg, identity

# The run states a provider may report. FINISHED means done; the rest mean the
# work did not complete, so by default a re-run is allowed.
STATUSES = ("FINISHED", "FAILED", "QUEUED", "RUNNING")


class StatusProvider(Protocol):
    """A read view of runs already recorded in some store.

    Requires an implementation of the `existing` method that returns a mapping
    from identity hash to the run's status (one of STATUSES).
    """

    def existing(self) -> dict[str, str]: ...


def dedup(
    cfgs: list[RunCfg],
    provider: StatusProvider | None = None,
    skip: tuple[str, ...] = ("FINISHED",),
    policy: str = "skip",
) -> list[RunCfg]:
    """Drop repeats by identity, keeping the first of each.

    Two sources of a repeat are handled together:
      - Within the batch: the first cfg of each identity is kept and later
        copies dropped, whether or not a provider is given.
      - Already recorded: when a provider is given, a cfg is dropped if the
        store already holds its identity in a status listed in `skip`.

    skip: which recorded statuses count as "done, do not re-run". The default
        skips only FINISHED runs, so FAILED, QUEUED, and RUNNING (interrupted)
        runs are re-run. Pass more statuses to skip them too.
    policy: what to do when a cfg is a repeat. "skip" (default) drops it.
        "error" raises ValueError on the first repeat, for a run that must be
        entirely fresh.
    """
    if policy not in ("skip", "error"):
        raise ValueError(f"unknown policy {policy!r} (use 'skip' or 'error')")
    seen: set[str] = set()
    if provider is not None:
        seen.update(
            h for h, status in provider.existing().items() if status in skip
        )
    out: list[RunCfg] = []
    for cfg in cfgs:
        key = identity(cfg)
        if key in seen:
            if policy == "error":
                raise ValueError(f"duplicate run, identity already seen: {key}")
            continue
        seen.add(key)
        out.append(cfg)
    return out


def subsample(
    cfgs: list[RunCfg],
    n: int | None = None,
    fraction: float | None = None,
    seed: int = 0,
) -> list[RunCfg]:
    """Return a random subset of cfgs: either n points, or a fraction of them."""
    if n is None and fraction is None:
        return list(cfgs)
    total = len(cfgs)
    if n is not None:
        keep = n
    else:
        assert fraction is not None
        keep = max(1, round(total * fraction))
    return random.Random(seed).sample(cfgs, min(keep, total))


def apply(cfgs: list[RunCfg], *filters) -> list[RunCfg]:
    """Pipe cfgs through a chain of filters, in order."""
    for step in filters:
        cfgs = step(cfgs)
    return cfgs
