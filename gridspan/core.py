"""gridspan core"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

# A nested dict describing a whole sweep; each leaf is a set of options.
GridSpec = dict[str, Any]
# One flat dotted-key dict describing exactly one run.
RunCfg = dict[str, Any]


def _flatten(cfg: GridSpec, prefix: str = "") -> RunCfg:
    """Turn a nested dict into a flat dict with dotted keys. Dicts recurse; other values stop."""
    out: dict = {}
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full_key))
        else:
            out[full_key] = value
    return out


def expand(spec: GridSpec, stamp: bool = False) -> list[RunCfg]:
    """Return one RunCfg per point in the cartesian product over a GridSpec.

    1. Flatten the nested input to dotted keys; dict values are nesting, so
       they recurse. Everything else is a leaf.
    2. At each leaf, a list is an axis (a set of candidate values); anything
       else is a singleton value at that key.
    3. Take the cartesian product across every axis; return one flat dict
       per point, keyed by the same dotted keys as the flattened input.
    4. If `stamp` is True, fill in `gridspan.id.hash` on each output dict
       using identity().

    Args:
        spec: a nested dict-like where leaves are lists (axes) or scalars/lists-
              at-inner-position/dicts-at-inner-position (values).
        stamp: when True, each output RunCfg carries its own `gridspan.id.hash`.

    Returns:
        A list of RunCfgs, one per point in the cartesian product.
    """
    flat = _flatten(spec)
    keys = list(flat.keys())
    axes: list[list[Any]] = []
    for key in keys:
        value = flat[key]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"empty axis at {key!r}")
            axes.append(value)
        else:
            axes.append([value])
    points = [dict(zip(keys, combo)) for combo in itertools.product(*axes)]
    if stamp:
        for point in points:
            point["gridspan.id.hash"] = identity(point)
    return points


def identity(cfg: RunCfg) -> str:
    """Return a stable md5 hex of a RunCfg's identity subset.

    Matches the identity used by dmllib.dmlutil.mlflow so an existing
    corpus of runs stays queryable by the same hash.

    The identity subset is:
        - all keys in gridspan.id.include if set, else all keys;
        - minus everything in gridspan.id.exclude;
        - minus every reserved key (any 'gridspan.id.*').

    Values are JSON-serialized with sorted keys and default=str, so a Path
    or a tuple survives round-trip.
    """
    include = cfg.get("gridspan.id.include")
    exclude = set(cfg.get("gridspan.id.exclude") or [])
    keys = list(cfg.keys()) if include is None else list(include)
    scoped = {
        key: cfg[key]
        for key in keys
        if key in cfg
        and key not in exclude
        and not key.startswith("gridspan.id.")
    }
    payload = json.dumps(scoped, sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


def from_yaml(path: str | Path) -> list[RunCfg]:
    """Read a YAML GridSpec and expand it in one call."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        return expand(yaml.safe_load(handle) or {})
