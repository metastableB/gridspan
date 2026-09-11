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

1. Simple nested dict <- show dot dict, sigleton vs List/set
2. Simple nested dict with list as value
3. Simple nested dict with some paraters that don't matter for run-uniqueness
4. Simple nested dict with mlflow provider
5. 
