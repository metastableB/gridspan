"""gridspan core"""

from __future__ import annotations

import copy
import datetime
import hashlib
import itertools
import json
import random
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

from gridspan.stores import Store

# One flat dictionary describing exactly one run.
RunCfg = dict[str, Any]


class GridSpan(list[RunCfg]):
    """Keep expanded configs and their key separator together."""

    def __init__(self, cfgs: Iterable[RunCfg] = (), *, key_sep: str = ".") -> None:
        super().__init__(cfgs)
        self.key_sep = key_sep

    def apply(
        self, *filters: Callable[[GridSpan], Iterable[RunCfg]]
    ) -> GridSpan:
        """Apply filter functions left-to-right and return a new GridSpan."""
        cfgs = GridSpan(self, key_sep=self.key_sep)
        for step in filters:
            cfgs = GridSpan(step(cfgs), key_sep=self.key_sep)
        return cfgs

    def stamp(self) -> GridSpan:
        """Write current identity hashes into each config and return this span.

        Configs stay editable. Call again after changes to refresh their hashes.
        """
        hash_key = self.key_sep.join(("gridspan", "id", "hash"))
        for cfg in self:
            cfg[hash_key] = identity(cfg, key_sep=self.key_sep)
        return self

    def dedup(
        self,
        store: Store | None = None,
        policy: Literal["skip", "error"] = "skip",
    ) -> GridSpan:
        """Refresh hashes and return a new GridSpan with duplicates removed.

        Every point is stamped in place, and the first point with each hash is
        kept. A store is called once with the unique hashes and returns the
        ones to skip; those points are removed too. policy="error" raises
        ValueError instead of removing a point.
        """
        if policy not in ("skip", "error"):
            raise ValueError(f"unknown policy {policy!r} (use 'skip' or 'error')")
        hash_key = self.key_sep.join(("gridspan", "id", "hash"))
        self.stamp()
        done: set[str] = set()
        if store is not None:
            skipped = store(list(dict.fromkeys(cfg[hash_key] for cfg in self)))
            if isinstance(skipped, str) or not isinstance(skipped, Iterable):
                raise TypeError(
                    "store must return a collection of hashes, "
                    f"got {type(skipped).__name__}"
                )
            done = set(skipped)
        seen: set[str] = set()
        out = GridSpan(key_sep=self.key_sep)
        for cfg in self:
            key = cfg[hash_key]
            if key in done or key in seen:
                if policy == "error":
                    reason = "the store returned it" if key in done else "it repeats"
                    raise ValueError(f"duplicate run {key}: {reason}")
                continue
            seen.add(key)
            out.append(cfg)
        return out

    def subsample(self, n: int, seed: int = 0) -> GridSpan:
        """Sample n configurations without replacement as a new GridSpan.

        Uses Python's random.sample, so n must be between zero and the span's
        length. seed makes sampling repeatable.
        """
        return GridSpan(random.Random(seed).sample(self, k=n), key_sep=self.key_sep)

    def to_argv(self) -> list[list[str]]:
        """Return command-line tokens for each config, in span order.

        Each setting becomes "--<key>" followed by str(value). Keys keep the
        span's separator, so "model.name" becomes "--model.name". gridspan
        metadata is skipped.
        """
        metadata_prefix = f"gridspan{self.key_sep}"
        commands = []
        for cfg in self:
            argv: list[str] = []
            for key, value in cfg.items():
                if not key.startswith(metadata_prefix):
                    argv += [f"--{key}", str(value)]
            commands.append(argv)
        return commands


class GridSpec(dict[str, Any]):
    """Validate a nested parameter dictionary and keep its key separator."""

    def __init__(
        self, data: dict[str, Any] | None = None, *, key_sep: str = "."
    ) -> None:
        if data is None:
            data = {}
        _validate_spec(data, key_sep)
        super().__init__(data)
        self.key_sep = key_sep

    def expand(self) -> GridSpan:
        """Revalidate the spec and expand it into one config per combination.

        Lists supply choices; metadata is not swept. Empty groups and choice
        lists are skipped with warnings. Each point gets its own copy of every
        value. Hashes are written later by stamp() or dedup().
        """
        skipped: list[str] = []
        flat = _validate_spec(self, self.key_sep, skipped)
        for message in skipped:
            warnings.warn(message, stacklevel=2)
        axes: list[list[Any]] = []
        metadata_prefix = f"gridspan{self.key_sep}"
        for key, value in flat.items():
            if key.startswith(metadata_prefix) or not isinstance(value, list):
                # Metadata lists, such as include/exclude, pass through whole.
                axes.append([value])
            else:
                axes.append(value)
        return GridSpan(
            (
                copy.deepcopy(dict(zip(flat, combo)))
                for combo in itertools.product(*axes)
            ),
            key_sep=self.key_sep,
        )


def _starts_with_word(key: str, word: str) -> bool:
    """Return True if key is word followed by punctuation, such as 'gridspan.x'."""
    rest = key[len(word):]
    return key.startswith(word) and rest != "" and not (rest[0].isalnum() or rest[0] == "_")


def _flatten(
    cfg: dict[str, Any],
    prefix: str = "",
    key_sep: str = ".",
    *,
    skipped: list[str],
) -> RunCfg:
    """Join nested dictionary keys with key_sep.

    Keys must be nonempty strings without key_sep. Recurse into dictionaries
    and copy other values as leaves. Skip empty groups and choice lists, and
    append a message for each one to skipped.
    """
    out: dict = {}
    metadata_prefix = f"gridspan{key_sep}"
    selector_keys = (
        key_sep.join(("gridspan", "id", "include")),
        key_sep.join(("gridspan", "id", "exclude")),
    )
    for key, value in cfg.items():
        if not isinstance(key, str):
            raise TypeError(f"key must be a string, got {type(key).__name__}")
        if not key:
            raise ValueError("key names must not be empty")
        if key_sep in key:
            raise ValueError(f"key contains separator {key_sep!r}: {key!r}")
        reserved = {"": "gridspan", "gridspan": "id"}.get(prefix)
        if reserved and _starts_with_word(key, reserved):
            raise ValueError(
                f"key {key!r} looks like {reserved!r} with another separator; "
                f"with key_sep {key_sep!r}, nest it under {reserved!r}"
            )
        full_key = f"{prefix}{key_sep}{key}" if prefix else key
        if isinstance(value, dict) and full_key not in selector_keys:
            if not value:
                skipped.append(f"skipping empty dictionary at {full_key!r}")
                continue
            for nested_key, nested_value in _flatten(
                value, full_key, key_sep=key_sep, skipped=skipped
            ).items():
                if nested_key in out:
                    raise ValueError(
                        f"key collision after flattening: {nested_key!r}"
                    )
                out[nested_key] = nested_value
        elif isinstance(value, list) and not value and not full_key.startswith(
            metadata_prefix
        ):
            skipped.append(f"skipping empty choice list at {full_key!r}")
        else:
            if full_key in out:
                raise ValueError(f"key collision after flattening: {full_key!r}")
            out[full_key] = value
    return out


def _validate_spec(
    spec: dict[str, Any], key_sep: str, skipped: list[str] | None = None
) -> RunCfg:
    """Check spec rules and return the flattened settings without expanding them.

    Messages for skipped empty values are appended to skipped when given.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"spec must be a dictionary, got {type(spec).__name__}")
    if not isinstance(key_sep, str) or len(key_sep) != 1:
        raise ValueError("key_sep must be a single-character string")
    flat = _flatten(
        spec, key_sep=key_sep, skipped=[] if skipped is None else skipped
    )
    _identity_keys(flat, key_sep=key_sep)
    return flat


def _identity_keys(cfg: RunCfg, *, key_sep: str) -> list[str]:
    """Validate include/exclude settings and return the keys used for identity."""
    include_key = key_sep.join(("gridspan", "id", "include"))
    exclude_key = key_sep.join(("gridspan", "id", "exclude"))
    has_include = include_key in cfg
    has_exclude = exclude_key in cfg
    if has_include and has_exclude:
        raise ValueError(
            f"{include_key} and {exclude_key} are mutually exclusive"
        )
    for selector_key in (include_key, exclude_key):
        if selector_key not in cfg:
            continue
        selected = cfg[selector_key]
        if not isinstance(selected, list) or not all(
            isinstance(key, str) for key in selected
        ):
            raise TypeError(f"{selector_key} must be a list of strings")
        for key in selected:
            if key not in cfg:
                raise ValueError(f"{selector_key} references unknown key {key!r}")
    include = cfg.get(include_key)
    exclude = set(cfg.get(exclude_key) or [])
    keys = list(cfg.keys()) if include is None else list(include)
    return [
        key
        for key in keys
        if key not in exclude
        and not key.startswith(f"gridspan{key_sep}")
    ]


def identity(cfg: RunCfg, *, key_sep: str = ".") -> str:
    """Compute a stable identity hash for one run config.

    key_sep must match expansion. The include or exclude setting under
    gridspan's id section selects keys. Metadata never contributes to the hash.
    Each selector must be a list of existing key names. Values must be JSON
    data, sets, dates, or times; other types raise TypeError.
    """
    scoped = {key: cfg[key] for key in _identity_keys(cfg, key_sep=key_sep)}
    try:
        payload = _dumps(scoped)
    except TypeError:
        for key, value in scoped.items():
            try:
                _dumps(value)
            except TypeError as error:
                raise TypeError(f"cannot hash {key!r}: {error}") from None
        raise
    # MD5 is used only as a fingerprint; this flag lets it run on FIPS systems.
    return hashlib.md5(payload.encode(), usedforsecurity=False).hexdigest()


def _dumps(value: Any) -> str:
    """Encode value as JSON text that is the same in every Python process."""
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    """Encode the non-JSON types whose text form is stable."""
    if isinstance(value, (set, frozenset)):
        # Set order varies between processes, so sort by each item's encoding.
        return sorted(value, key=_dumps)
    if isinstance(value, (datetime.date, datetime.time)):
        return str(value)
    raise TypeError(
        f"{type(value).__name__} values have no stable hash; use JSON data "
        "(str, int, float, bool, None, list, dict), a set, a date, or a time"
    )


def from_dict(data: dict[str, Any], *, key_sep: str = ".") -> GridSpec:
    """Validate a Python dictionary and create a GridSpec with key_sep."""
    return GridSpec(data, key_sep=key_sep)


def from_yaml(path: str | Path, *, key_sep: str = ".") -> GridSpec:
    """Load and validate a GridSpec from a YAML file.

    Empty or null documents produce an empty spec. Other non-dictionary roots
    raise TypeError. Values keep PyYAML's types; key_sep belongs to the spec.
    """
    import yaml

    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a dictionary, got {type(data).__name__}")
    return GridSpec(data, key_sep=key_sep)
