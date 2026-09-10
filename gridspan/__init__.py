"""gridspan — expand a nested dict of parameter sets into a list of run configs.

Typical workflow:

    # 1. Write a GridSpec (a nested dict, or read one from YAML).
    #    List leaves are swept axes; scalars stay fixed.
        spec = {
            "model": {"name": ["gpt-4", "gpt-5"]},
            "runtime": {"max_tokens": 8192},
        }

    # 2. Expand into one RunCfg per point in the cartesian product.
    #    A RunCfg is a flat dotted-key dict.
        for cfg in gridspan.expand(spec):
            run_one(cfg)                      # cfg["model.name"], cfg["runtime.max_tokens"]

    # For a typed run config, define a project-side dataclass with its own
    # from_dict, then convert per point at the top of the loop:
    #
    #     for point in gridspan.expand(spec):
    #         run_one(MyRunCfg.from_dict(point))

    # 3. For dedup / resume across runs, stamp each point with an identity hash:
        for cfg in gridspan.expand(spec, stamp=True):
            run_one(cfg)                      # cfg["gridspan.id.hash"] is stable

    # 4. Prune a batch with the convention-layer filters:
    cfgs = gridspan.expand(spec)
    cfgs = gridspan.dedup(cfgs)
    cfgs = gridspan.subsample(cfgs, n=20, seed=0)

Core:
    expand(spec: GridSpec, stamp: bool = False) -> list[RunCfg]
        Compute the cartesian product over a nested dict of parameters.
        The algorithmic procedure is as follows:
        1. Flatten all nested dicts into a flat dict using `.` separator.
        2. Compute set products of all sets to make grid.
        3. Return a list of RunCfgs for each point in the grid.


        Note: 
            - To hold a plain list as one leaf value, wrap it in another list.
            [[1, 2]] is one axis with one point, and that point is [1, 2].

            - If `stamp` is True, fill in `gridspan.id.hash` for each output
            RunCfg. Keys and values to compute the grid hash can be specified
            via  `gridspan.id.exclude` keys and `gridspan.id.include` keys.
            The hash is computed as:
            ```
            # include = None -> include = All
            # exclude = None -> exclude = empty set
            include = include_keys if include_keys else set(cfg.keys())
            exclude = exclude_keys if exclude_keys else set()
            id_keys = (include - exclude) - {k for k in cfg if k.startswith("gridspan.id.")}
            payload = json.dumps({k: cfg[k] for k in sorted(id_keys)},
                                 sort_keys=True, default=str)
            id_hash = hashlib.md5(payload.encode()).hexdigest()
            ```

        Returns:
            - Returns a list of RunCfgs for each point in the grid, with an
            optional `gridspan.id.hash` key.

        Examples:
          {"a": [1, 2], "b": 3}
              -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}]
          {"a": [[1, 2], 3]}
              -> [{"a": [1, 2]}, {"a": 3}]
          {"a": [{"x": 1}, {"x": 2}]}
              -> [{"a": {"x": 1}}, {"a": {"x": 2}}]

    from_yaml(path: str | Path) -> list[RunCfg]
        Read a YAML GridSpec and expand it in one call.

Filters (list -> list helpers over batches of RunCfgs):

    dedup(cfgs: list[RunCfg], provider=None) -> list[RunCfg]
        Drop repeats by identity, keeping the first of each. With a
        StatusProvider, also drop any cfg a store already holds.

    subsample(cfgs, n=None, fraction=None, seed=0) -> list[RunCfg]
        Random subset: keep n points, or a fraction of them.

    apply(cfgs, *filters) -> list[RunCfg]
        Pipe cfgs through a chain of filters (each list -> list), in order.

Types:

    GridSpec = dict[str, Any]
        The nested pre-expand dict. One GridSpec describes many runs.

    RunCfg = dict[str, Any]
        A flat dotted-key dict. One RunCfg is exactly one run.

Protocols:

    StatusProvider (Protocol): A way for storing run status across runs.
        existing(spec: GridSpec) -> dict[str, str]

Reserved keys the core algorithm honors:

    gridspan.id.include    a list of dotted keys the identity hash reads.
                           When set, only these keys count; otherwise all
                           non-reserved keys count.
    gridspan.id.exclude    a list of dotted keys the identity hash skips.
    gridspan.id.hash       filled in by expand(spec, stamp=True); read by
                           dedup and by tracker adapters when checking for
                           existing runs.
"""

from gridspan.core import GridSpec, RunCfg, expand, from_yaml
from gridspan.filters import StatusProvider, apply, dedup, subsample

__all__ = [
    "GridSpec",
    "RunCfg",
    "StatusProvider",
    "apply",
    "dedup",
    "expand",
    "from_yaml",
    "subsample",
]
