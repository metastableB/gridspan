"""Skip grid points whose runs in MLflow reached chosen statuses.

This store is opinionated. Write your own store if its rules do not fit:

1. A run belongs to a grid point when its hash tag (default
   `gridspan.id.hash`) equals the point's hash.
2. It searches the named experiment and any `search_experiments`.
   Deleted runs are ignored.
3. If several runs share a hash, the newest run's status is used.
4. A point is skipped when that status is in `skip`, compared exactly.

Import is lazy, so gridspan core does not need MLflow.

    from gridspan.extras import MlflowStore
    store = MlflowStore("my-experiment", skip={"FINISHED"})
    span = span.dedup(store=store)
"""

from __future__ import annotations

from collections.abc import Collection


# TODO: Review cross-experiment lookup and identity-tag assumptions.
# TODO: Query only the requested hashes instead of reading every tagged run.
class MlflowStore:
    """Return the hashes whose MLflow runs have a status listed in skip.

    The store looks for identity hashes in run tags across one or more
    experiment names. Statuses are compared exactly.
    """

    def __init__(
        self,
        experiment: str,
        tracking_uri: str | None = None,
        hash_tag: str = "gridspan.id.hash",
        search_experiments: list[str] | None = None,
        *,
        skip: Collection[str],
    ):
        if isinstance(skip, str):
            raise TypeError("skip must be a collection of statuses, such as {'FINISHED'}")
        self.experiment = experiment
        self.tracking_uri = tracking_uri
        self.hash_tag = hash_tag
        self.skip = set(skip)
        names = [experiment]
        if search_experiments:
            names += [n for n in search_experiments if n != experiment]
        self.experiment_names = names

    def __call__(self, run_hashes: list[str]) -> set[str]:
        """Return the hashes in run_hashes whose MLflow status is in skip."""
        statuses = self._all_statuses()
        return {h for h in run_hashes if statuses.get(h) in self.skip}

    def _all_statuses(self) -> dict[str, str]:
        """Return a mapping of identity hash to MLflow run status.

        Missing experiments are skipped. Large results are read page by page.
        """
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)

        experiment_ids = []
        for name in self.experiment_names:
            exp = client.get_experiment_by_name(name)
            if exp is not None:
                experiment_ids.append(exp.experiment_id)
        if not experiment_ids:
            return {}

        out: dict[str, str] = {}
        filter_string = f"tags.`{self.hash_tag}` != ''"
        page_token = None
        while True:
            page = client.search_runs(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=1000,
                order_by=["attributes.start_time DESC"],
                page_token=page_token,
            )
            for run in page:
                hash_value = run.data.tags.get(self.hash_tag)
                if hash_value:
                    # Runs arrive newest first, so the first status seen wins.
                    out.setdefault(hash_value, run.info.status)
            page_token = getattr(page, "token", None)
            if not page_token:
                return out