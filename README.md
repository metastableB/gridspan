# gridspan

Expand a nested dict of parameter sets into a list of run configs, and prune
that list before you run it. The core is dict-in, dict-out and has one
dependency (PyYAML). Experiment-tracker integration (MLflow) is an optional
adapter.

## Install

```bash
git clone git@github.com:metastableB/gridspan.git
cd gridspan
pip install -e .            # core only
pip install -e '.[mlflow]'  # with the MLflow status provider
pip install -e '.[dev]'     # tests + lint + mlflow
```

## Quick start

A `GridSpec` is a nested dict. A list leaf is a swept axis; a scalar is
fixed. `expand` returns one flat `RunCfg` per point in the cartesian product,
with nested keys flattened to dotted keys.

```python
import gridspan

spec = {
    "model": {"name": ["gpt-4", "gpt-5"]},   # a swept axis (2 values)
    "runtime": {"max_tokens": 8192},          # fixed
}

for cfg in gridspan.expand(spec):
    # cfg is a flat dict: {"model.name": ..., "runtime.max_tokens": 8192}
    run_one(cfg["model.name"], cfg["runtime.max_tokens"])
```

Read the same spec from YAML instead:

```python
cfgs = gridspan.from_yaml("sweep.yaml")
```

Prune the batch before running — drop duplicates, take a sample:

```python
cfgs = gridspan.expand(spec)
cfgs = gridspan.dedup(cfgs)
cfgs = gridspan.subsample(cfgs, n=20, seed=0)
```

Skip runs a store already finished (resume). A `StatusProvider` reports which
run identities are recorded and in what state; `dedup` skips the finished
ones and lets failed or interrupted runs come back:

```python
from gridspan.providers import MlflowProvider

provider = MlflowProvider("my-experiment", tracking_uri="sqlite:///mlflow.db")
cfgs = gridspan.dedup(gridspan.expand(spec, stamp=True), provider=provider)
```

## Three ways a list leaf behaves

The whole grid rule is one sentence: at each leaf, a list is the axis; each
element is one value.

```python
gridspan.expand({"a": [1, 2], "b": 3})
# -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}]        # a plain sweep

gridspan.expand({"a": [[1, 2], 3]})
# -> [{"a": [1, 2]}, {"a": 3}]                    # a wrapped list is one value

gridspan.expand({"a": [{"x": 1}, {"x": 2}]})
# -> [{"a": {"x": 1}}, {"a": {"x": 2}}]           # dict values travel whole
```

To hold a plain list as one leaf value (do not sweep it), wrap it in another
list: `probes: [[a, b]]` is one axis with one point, and that point is the
list `[a, b]`.

## Reference

### Core

- `expand(spec: GridSpec, stamp: bool = False) -> list[RunCfg]`
  Compute the cartesian product over a nested dict of parameters:
  1. Flatten all nested dicts to a flat dict with `.`-separated keys.
  2. At each leaf, a list is a swept axis; anything else is a fixed value.
  3. Return one flat `RunCfg` per point in the product.
  4. If `stamp` is True, fill in `gridspan.id.hash` on each output config.

- `from_yaml(path) -> list[RunCfg]`
  Read a YAML `GridSpec` and expand it in one call.

### Filters (list -> list helpers)

- `dedup(cfgs, provider=None, skip=("FINISHED",), policy="skip")`
  Drop repeats by identity, keeping the first of each. With a
  `StatusProvider`, also drop any config the store already holds in a status
  listed in `skip` (default: only `FINISHED`, so failed and interrupted runs
  are re-run). `policy="error"` raises on the first repeat instead of dropping.

- `subsample(cfgs, n=None, fraction=None, seed=0)`
  A random subset: keep `n` points, or a `fraction` of them.

- `apply(cfgs, *filters)`
  Pipe a list through a chain of `list -> list` filters, in order.

### Types

- `GridSpec = dict[str, Any]` — the nested pre-expand dict. One `GridSpec`
  describes many runs.
- `RunCfg = dict[str, Any]` — a flat dotted-key dict. One `RunCfg` is exactly
  one run.

### Identity and reserved keys

Each config has a stable identity: an md5 over its identity subset. That hash
is what a tracker tags a run with and queries by, and what dedup and resume
are built on. Three reserved keys tune it:

- `gridspan.id.include` — a list of dotted keys the hash reads. When set, only
  these keys count; otherwise all non-reserved keys count.
- `gridspan.id.exclude` — a list of dotted keys the hash skips.
- `gridspan.id.hash` — filled in by `expand(spec, stamp=True)`; read by dedup
  and by tracker adapters when checking for existing runs.

### Status providers

`StatusProvider` is the seam to a run store. One method,
`existing() -> dict[str, str]`, returns a map from identity hash to run
status. `dedup` uses it to skip finished work. Adapters live in
`gridspan.providers` and import their backend lazily, so importing the package
is safe without the backend installed.

- `MlflowProvider(experiment, tracking_uri=None, hash_tag="gridspan.id.hash",
  search_experiments=None)` — reads run identities and statuses from one or
  more MLflow experiments.

## Design

Principles, open questions, and todos live in the notes vault at
`Notes-Personal/gridspan/design.md`.
