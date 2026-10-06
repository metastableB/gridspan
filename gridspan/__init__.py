"""Expand sweep specs into runnable configs.

This package turns a nested grid spec into one flat config per grid point.
from_dict and from_yaml create a GridSpec, which validates its input and keeps
key_sep. GridSpec.expand returns a GridSpan, a list of configs whose methods
stamp hashes, deduplicate, subsample, chain custom filter steps, and render
command-line arguments.
Pass a store to dedup to skip prior jobs: any callable that takes a list of
hashes and returns the ones to skip. TextStore reads them from a text file.

Main API:
    from_dict, from_yaml, GridSpec, GridSpan, Store, TextStore
"""

from gridspan.core import GridSpan, GridSpec, RunCfg, from_dict, from_yaml
from gridspan.stores import Store, TextStore

__version__ = "0.1.0"

__all__ = [
    "GridSpan",
    "GridSpec",
    "RunCfg",
    "Store",
    "TextStore",
    "from_dict",
    "from_yaml",
]
