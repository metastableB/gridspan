# gridspan

A simple python library that takes a grid of parameters specified in a YAML
and generates the grid points as a list of dicts.

## Install

```bash
git clone git@github.com:metastableB/gridspan.git
cd gridspan
pip install -e .            # core only
```

## Quick start

At the core, gridspan takes a nested dict, 
1. flattens it, 
2. treats the values as sets (for each key), and
3. computes a set product to give a list of dicts. 


```python
import gridspan

spec = {
    "model": {"name": ["gpt-4", "gpt-5"]},   # a set of 2
    "runtime": {"max_tokens": 8192},          # a singleton
}

for cfg in gridspan.expand(spec):
    # cfg is one flat dict: {"model.name": ..., "runtime.max_tokens": 8192}
    run_one(cfg["model.name"], cfg["runtime.max_tokens"])
```

Read the same spec from a YAML file instead:
```yaml
model:
    name: ["gpt-4", "gpt-5"]
runtime:
    max_tokens: 8192
```
then, 
```python
cfgs = gridspan.from_yaml("sweep.yaml")
```

You can apply filters to eliminate invalid points in the grid using the `apply`
function (or just processing the gird yourself).

By default `gridspan` does not deduplicate. However, `dedup()` can be used to
deduplicate configurations.
```python
cfgs = gridspan.expand(spec)
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
cfgs = gridspan.dedup(gridspan.expand(spec, stamp=True), provider=provider)
```

## Examples 

**1. A nested dict — singleton vs set, and dotted keys.**

```python
spec = {"model": {"name": ["gpt-4", "gpt-5"]}, "seed": 0}
gridspan.expand(spec)
# [{"model.name": "gpt-4", "seed": 0},
#  {"model.name": "gpt-5", "seed": 0}]
```

**2. A value that is itself a list — wrap it so it stays whole.**

```python
spec = {"tools": [["search", "python"], ["search"]]}
gridspan.expand(spec)
# [{"tools": ["search", "python"]},
#  {"tools": ["search"]}]
```

**3. Parameters that don't matter for run-uniqueness — exclude them.**

Say we ran the following spec. 
```python
spec = {
    "model": ["x", "y"],
    "retries": 3,                  
    "gridspan.id.exclude": ["retries"],
}
gridspan.dedup(gridspan.expand(spec))
# retries is out of the identity, so the two retry variants per model
# collapse: 2 configs, not 4.
```
Since retries was excluded from the spec, we can increase retries and the runs
still produce the same hash. This can be useful, if part of the previous grid
failed and a certain retry value and you want to expand.

**4. Skip runs an MLflow experiment already finished.**

```python
from gridspan.providers import MlflowProvider

spec = {"model": ["x", "y", "z"]}
provider = MlflowProvider("my-experiment", tracking_uri="sqlite:///mlflow.db")
cfgs = gridspan.dedup(gridspan.expand(spec, stamp=True), provider=provider)
# any model already FINISHED in the experiment is dropped; failed or
# unfinished ones come back.
```

**5. Sample the grid, then prune with your own filter.**

<!-- TODO: the apply(...) chain reads awkwardly. Revisit the filter syntax and
     language — a fluent, Ray-like .map(...).filter(...) chain would read better. -->

```python
def small_only(cfgs):
    return [c for c in cfgs if c["batch"] <= 32]

spec = {"model": ["x", "y"], "batch": [16, 32, 64]}
cfgs = gridspan.apply(
    gridspan.expand(spec),
    small_only,                                       # your own filter
    lambda cs: gridspan.subsample(cs, n=2, seed=0),   # then a random 2
)
```

