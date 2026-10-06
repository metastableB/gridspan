# gridspan

`gridspan` is a small Python library for parameter sweeps.  It takes a nested
parameter grid and returns concrete grid points as a list of flat dictionaries,
after performing optional filtering and deduplication.

## Install

```bash
git clone https://github.com/metastableB/gridspan.git
cd gridspan
pip install -e .
```

## Quick Start 

For illustration purposes, we will use `gridspan` to sweep over the `curl`
command line tool. First define a sweep file:

```yaml
# curl_sweep.yaml
max-time: [2, 5]
retry: [0, 3]
url: ["https://example.com"]
```

Then expand the grid and render each config to CLI args:

```python
import gridspan as gs

configs = gs.from_yaml("curl_sweep.yaml").expand()

for argv in configs.to_argv():
  print(["curl", *argv])
```

```text
# Output
['curl', '--max-time', '2', '--retry', '0', '--url', 'https://example.com']
['curl', '--max-time', '2', '--retry', '3', '--url', 'https://example.com']
['curl', '--max-time', '5', '--retry', '0', '--url', 'https://example.com']
['curl', '--max-time', '5', '--retry', '3', '--url', 'https://example.com']
```

## Deeper Usage

The behavior of `gridspan` is defined by these rules.

1. `GridSpec`: A spec is a nested dictionary, with the flattened key specifying
the argument key and the value specifying the set of values to sweep over. 

2. `GridSpan`: A spec expansion is a simple cartesian product of all sweep axes,
that gives the span.  Each output point is a flat dictionary with path keys.


**Example:** The following `spec.yaml`,

```yaml
# spec.yaml
model:
  name: [gpt-4, gpt-5]
runtime:
  max_tokens: [8192, 32364]
  batch: [1, 32]
```
Becomes,
```python
gs.from_yaml("spec.yaml").expand()
# [
#   {"model.name": "gpt-4", "runtime.max_tokens": 8192, 'runtime.batch': 1},
#   {"model.name": "gpt-4", "runtime.max_tokens": 8192, 'runtime.batch': 32},
#   ...
# ]
```

A list is always a set of choices, and each grid point gets its own copy of
its values. To use a list or a dictionary as one value, wrap it in a list:

```yaml
layers: [[64, 64], [128]]               # two choices: [64, 64] and [128]
optimizer: [{name: adam, lr: 0.001}]    # one choice: the whole dictionary
```


### Optional post-expansion transformations/filters 

We provide some common post-expansion operations. Custom operations can be
applied through `apply`.

- `subsample(n, seed=0)` samples exactly `n` items without replacement.
- `stamp` writes or refreshes each point's identity hash.
- `dedup` refreshes hashes and removes duplicates, optionally checking a store for prior run status.
- `apply` composes custom transforms left-to-right.
- `to_argv` renders each point's non-metadata keys as `--key value` pairs.
  Keys keep the spec's separator, for example `--model.name`.

`subsample` wraps Python's `random.sample`. A count of zero returns an empty
span; negative counts or counts larger than the span raise `ValueError`.
For a fraction, compute the count explicitly, for example
`span.subsample(n=round(len(span) * 0.25), seed=0)`.

`GridSpan.apply` is chainable. Each call receives the current list of configs
and returns the next list.

```python
span = gs.from_yaml("spec.yaml").expand()

def keep_small_batch_for_large_max_tokens(cfgs):
  exclude = lambda c: c["runtime.batch"] == 32 and c["runtime.max_tokens"] == 32364
  return [c for c in cfgs if not exclude (c)]

span = span.apply(keep_small_batch_for_large_max_tokens).dedup()
```
### Optional reserved keywords, metadata, resume, deduplication and identity semantics

Oftentimes, due to run failures or spec modification, we often have to rerun the
grid points in a certain span. To differentiate new grid points from previously
existing grid points that might have already finished their runs,
1. we compute an `id` for each grid point, and
2. we optionally check a store for jobs that have already finished.

`gridspan.*`: The top level `gridspan` key is reserved for metadata.

- `gridspan.*`: Metadata is carried into each grid point, but is not swept or
included in its identity hash.

- `gridspan.id.hash`: `.dedup()` and `.stamp()` write the hash of the point's selected
non-metadata key/value pairs. `gridspan.id.exclude` or `gridspan.id.include`
can select which keys contribute to the hash.

`.dedup()` automatically refreshes hashes before comparing points or checking
a store. Use `.stamp()` when hashes need to be explicitly computed.

Points remain editable and may be shared between spans. After further parameter
changes, call `.dedup()` or `.stamp()` again before recording a job's status.

Use only one of `include` or `exclude`, as a list of existing key names. Unknown
names and conflicting include/exclude settings raise `ValueError` when the
`GridSpec` is created. A selector that is not a list of strings raises `TypeError`.

```yaml
model: [x, y]
retries: [3, 5]
gridspan:
  id:
    exclude: [retries]
```
Consequence: changing `retries` does not change run identity in this spec.

`GridSpec` requires nonempty string key names.  Key names  cannot contain that
separator. 

```python
spec = gs.from_yaml("spec.yaml", key_sep="/")  # default key_sep = "."
span = spec.expand()
```

Note: `key_sep` applies to metadata too. With `key_sep="/"`, nested `gridspan`,
`id`, and `exclude` keys become `gridspan/id/exclude`. Metadata written with
another separator, such as a root key `gridspan.note` or a key `id.exclude`
under `gridspan`, raises `ValueError` when the `GridSpec` is created. The
include/exclude lists refer to the resulting flattened parameter names.

`GridSpan` keeps its separator through filters and uses it in `to_argv`.

**Available store providers**

- `TextStore`: skips hashes listed in a text file.
- `MlflowStore`: An optional MLflow integration in `gridspan.extras`.
  It is opinionated; its rules are listed below.

An optional store helps keep track of prior runs for detecting which jobs can
resume, which should be skipped, which have already finished and should be
removed/deduplicated etc.  A store is any function or callable object that takes
a list of grid-point hashes and returns the ones to skip. 

```python
def store(run_hashes):
  return my_database.finished_among(run_hashes)
```

Stores can be used as part of `dedup` to filter the current span,
```python
pending = span.dedup(store=store)
```

`gridspan` calls the store once with the span's hashes, and drops the points
whose hashes it returns. What counts as "done" is up to the store; gridspan has
no notion of job status. Without a store, `.dedup()` only removes duplicates
within the span.

Stores only read. gridspan never writes to them; your job runner records
finished jobs in whatever system it already uses. Each grid point carries its
hash in `gridspan.id.hash`, which is the key to record it under.

**Available Stores:** `TextStore`, `MlflowStore`

- `TextStore`: `TextStore` reads a text file with one hash per line. Every
listed hash is skipped:

```text
3f2a...
9b1c...
```

```python
store = gs.TextStore("done.txt")
span = gs.from_yaml("curl_sweep.yaml").expand()
pending = span.dedup(store=store)
```

- `MlflowStore` skips points whose MLflow run reached one of the statuses you
list, compared exactly:

```python
from gridspan.extras import MlflowStore

store = MlflowStore("my-experiment", tracking_uri="sqlite:///mlflow.db", skip={"FINISHED"})
span = span.dedup(store=store)
```

`MlflowStore` is opinionated. Write your own store if these rules do not fit:

1. A run belongs to a grid point when its `gridspan.id.hash` tag (set by
   `hash_tag`) equals the point's hash.
2. It searches the named experiment and any `search_experiments`. Deleted
   runs are ignored.
3. If several runs share a hash, the newest run's status is used.
4. A point is skipped when that status is in `skip`.



## Limitations

1. `key-value`: `to_argv` renders non-metadata settings as `--key value` pairs.
It does not render bare switches like `--verbose` or positional arguments
like `cp SRC DST`. These forms can be produced by post-processing each
generated command.

Example:
Define the following spec,
```yaml
# cp-example.yaml
src: ["/my/path/A", "/my/path/B"]
dst: ["/fast/dest/X", "/fastish/dest/Y"]
verbose: [True, False]
```
Then,
```python
import gridspan as gs

span = gs.from_yaml("cp-example.yaml").expand()
for argv in span.to_argv():
  print(["cp", *argv])
# ['cp', '--src', '/my/path/A', '--dst', '/fast/dest/X', '--verbose', 'True']
# ['cp', '--src', '/my/path/A', '--dst', '/fast/dest/X', '--verbose', 'False']
# ...
```

For this example, remove the `--src` and `--dst` labels while keeping their
values in order. Keep `--verbose` only when its value is `True`:

```python
def post_process(command):
  command.remove("--src")
  command.remove("--dst")
  verbose_index = command.index("--verbose")
  verbose_value = command.pop(verbose_index + 1)
  if verbose_value == "False":
    command.pop(verbose_index)
  return command

for argv in span.to_argv():
  command = ["cp", *argv]
  print(post_process(command))
```

```text
['cp', '/my/path/A', '/fast/dest/X', '--verbose']
['cp', '/my/path/A', '/fast/dest/X']
...
```

2. `empty parameters`: Empty dictionary groups (`section: {}`) and empty
choice lists (`section: []`) are skipped during expansion with a `UserWarning`
naming the path.
The remaining parameters expand normally. If all parameters are skipped,
the result is `[{}]`, as for an empty spec.

To use an empty container as a value, wrap it as one choice: `[{}]` or `[[]]`.
Values such as `0`, `False`, `None`, and `""` are not skipped. Lists under
`gridspan` are metadata and stay unchanged, including an empty `include` list.

3. `YAML input and types`: The YAML root must be a dictionary. Empty or null
documents produce an empty spec. Other root types raise `TypeError`.
Parameter values keep the types returned by PyYAML; gridspan does not cast
them. Quote a YAML value when it must be a string.

Hashing encodes the selected values as JSON with sorted keys. Sets are sorted,
and dates and times are written as text, so an unquoted YAML date and its
matching quoted string count as the same identity. Any other type raises
`TypeError` naming the key, because its text form may differ between runs.
`to_argv` uses `str(value)` for every command-line value.

4. `Hash stability`: Migration might be required while we iron out bugs, and
 move towards stability. For now, keep each finished job's config, then rebuild
 the store with the new version: `gs.GridSpan(finished_configs).stamp()` gives
 the new hashes.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
