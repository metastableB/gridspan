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
        if key.startswith("gridspan.id."):
            # Reserved keys configure identity; they are never swept as axes,
            # so their list values (e.g. an exclude list) pass through whole.
            axes.append([value])
        elif isinstance(value, list):
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


def from_yaml(path: str | Path, stamp: bool = False) -> list[RunCfg]:
    """Read a YAML GridSpec and expand it in one call.

    stamp is passed through to expand: when True, each output RunCfg carries
    its own gridspan.id.hash.
    """
    import yaml

    with open(path, encoding="utf-8") as handle:
        return expand(yaml.safe_load(handle) or {}, stamp=stamp)


def to_argv(cfg: RunCfg, sep: str = "-", key=None, value=None) -> list[str]:
    """Render a RunCfg as command-line tokens for an existing CLI script.

    For each key/value in cfg (reserved gridspan.id.* keys are skipped):
      - the flag comes from key(dotted_key). The default turns a dotted key
        into a dashed flag: "model.name" -> "--model-name", with sep choosing
        the separator ("-", "_", or "." to keep it dotted).
      - the value tokens come from value(v). The default renders each value as
        its str(). A bool is special-cased and handled here, not by value: True
        emits the bare flag (argparse store_true style) and False emits nothing.

    Override key or value to fit a target CLI whose flag names or value format
    differ. key overrides never see reserved keys; value overrides never see
    bools.

    Args:
        cfg: one RunCfg (a flat dotted-key dict).
        sep: separator the default key mapper substitutes for ".".
        key: optional dotted_key -> flag string.
        value: optional value -> list of token strings (not called for bools).

    Returns:
        A list of argv tokens, e.g. ["--model-name", "gpt-4", "--verbose"].
    """
    key = key or (lambda k: "--" + k.replace(".", sep))
    value = value or (lambda v: [str(v)])
    argv: list[str] = []
    for name, val in cfg.items():
        if name.startswith("gridspan.id."):
            continue
        flag = key(name)
        if isinstance(val, bool):
            if val:
                argv.append(flag)  # store_true: bare flag when True, nothing when False
            continue
        argv.append(flag)
        argv.extend(value(val))
    return argv
