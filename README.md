# gridspan

A simple python library that takes a grid of parameters specified in a YAML and
generates individual grid points as a list of dicts. Useful when you want to
quickly setup massive experimental sweeps.

## Install

```bash
git clone git@github.com:metastableB/gridspan.git
cd gridspan
pip install -e .            # core only
```

## Quick start

Gridspan is designed to work with configurations/parameters specified as nested
dict-likes. Conceptually, Gridspan take such a dict,
1. flattens it, 
2. treats the values corresponding to each flattened key as sets, and
3. computes a set product to give a list of dicts. 


Consider this example grid configuration for some hypothetical script,
```yaml
# sweep_oom.yaml
model:
  name:
    - gpt-4
    - gpt-5               # a set of 2
runtime:
  max_tokens: 8192        # a singleton
  num_gpus: [1, 2, 4, 8]
```

To convert this to a grid using gridspan, we do
```python
import gridspan

for cfg in gridspan.from_yaml("sweep_oom.yaml"):
    # cfg is one flat dict:
    #   {
    #       "model.name": "gpt-4",
    #       "runtime.max_tokens": 8192,
    #       "runtime.num_gpus": 1,
    #   }
    run_one(cfg["model.name"], cfg["runtime.max_tokens"])
```

You can apply filters to eliminate invalid points in the grid using the `apply`
function (or just processing the gird yourself).

By default `gridspan` does not remove duplicates, but this can be enabled with
the `dedup` function.

```python
cfgs = gridspan.from_yaml("sweep_oom.yaml")
cfgs = gridspan.dedup(cfgs)
cfgs = gridspan.subsample(cfgs, n=20, seed=0)
```

Two configurations are considered identical if the hashes of their
configurations are identical. To include or exclude a key from this hashing, use
the `gridspan.id.include` and `gridspan.id.exclude` grids to specify a list of
included and excluded keys. Note, the values are not treated as sets.

Deduplication can also be extended to retry failed jobs, or skip finished jobs.
When a grid is large, failures are inevitable. For such cases, `dedup` reads a
store of past runs through a `StatusProvider` and drops the ones it already
finished. We ship one for MLflow:

```python
from gridspan.providers import MlflowProvider

provider = MlflowProvider("my-experiment", tracking_uri="sqlite:///mlflow.db")
cfgs = gridspan.dedup(gridspan.from_yaml("sweep_oom.yaml"), provider=provider)
```

## Examples 

**1. A nested dict — singleton vs set, and dotted keys.**

```yaml
# spec.yaml
model:
  name: ["gpt-4", "gpt-5"]
seed: 0
```

```python
gridspan.from_yaml("spec.yaml")
# [
#     {"model.name": "gpt-4", "seed": 0},
#     {"model.name": "gpt-5", "seed": 0},
# ]
```

**2. A value that is itself a list — wrap it so it stays whole.**

```yaml
# spec.yaml
tools:
  - [search, python]
  - [search]
```

```python
gridspan.from_yaml("spec.yaml")
# [
#     {"tools": ["search", "python"]},
#     {"tools": ["search"]},
# ]
```

**3. Parameters that don't matter for run-uniqueness — exclude them.**

Say we ran the following spec.

```yaml
# spec.yaml
model:
  - x
  - y
retries: 3
gridspan.id.exclude:          # TODO: Simplify this
  - retries
```

```python
gridspan.dedup(gridspan.from_yaml("spec.yaml"))
# 2 configs, one per model. retries is out of the identity, so its
# value does not affect the hash.
```

Since retries was excluded from the spec, we can increase retries and the runs
still produce the same hash. This can be useful, if part of the previous grid
failed and a certain retry value and you want to expand.

**4. Skip runs an MLflow experiment already finished.**

```yaml
# spec.yaml
model:
  - x
  - y
  - z
```

```python
from gridspan.providers import MlflowProvider

provider = MlflowProvider("my-experiment", tracking_uri="sqlite:///mlflow.db")
cfgs = gridspan.dedup(gridspan.from_yaml("spec.yaml"), provider=provider)
# any model already FINISHED in the experiment is dropped; failed or
# unfinished ones come back.
```

**5. Sample the grid, then prune with your own filter.**

<!-- TODO: the apply(...) chain reads awkwardly. Revisit the filter syntax and
     language — a fluent, Ray-like .map(...).filter(...) chain would read better.
     Or just drop apply all-together -->

```yaml
# spec.yaml
model:
  - x
  - y
batch:
  - 16
  - 32
  - 64
```

```python
def small_only(cfgs):
    return [c for c in cfgs if c["batch"] <= 32]

cfgs = gridspan.apply(
    gridspan.from_yaml("spec.yaml"),
    small_only,                                       # your own filter
    lambda cs: gridspan.subsample(cs, n=2, seed=0),   # then a random 2
)
```

