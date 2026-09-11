"""MLflow-backed StatusProvider.

Reads which run identities an MLflow experiment already holds, so dedup can
skip finished work. One run's identity is stored as a tag (default
`gridspan.id.hash`); its status is the MLflow run status (FINISHED, FAILED,
RUNNING, ...). Import is lazy, so gridspan core does not need MLflow.

    from gridspan.providers import MlflowProvider
    provider = MlflowProvider("my-experiment", tracking_uri="sqlite:///mlflow.db")
    cfgs = gridspan.dedup(cfgs, provider=provider)
"""

from __future__ import annotations


class MlflowProvider:
    """Read mlflow runs from an experiment and provide status for dedupe.

    Args:
        experiment: the MLflow experiment name to read runs from.
        tracking_uri: the MLflow tracking uri. When None, MLflow's own default
            is used (the MLFLOW_TRACKING_URI env var, else a local ./mlruns).
        hash_tag: the run tag holding a run's identity hash.
        search_experiments: extra experiment names to also read, for when the
            same identity may have run under a previous experiment name. The
            main experiment is always included.
    """

    def __init__(
        self,
        experiment: str,
        tracking_uri: str | None = None,
        hash_tag: str = "gridspan.id.hash",
        search_experiments: list[str] | None = None,
    ):
        self.experiment = experiment
        self.tracking_uri = tracking_uri
        self.hash_tag = hash_tag
        names = [experiment]
        if search_experiments:
            names += [n for n in search_experiments if n != experiment]
        self.experiment_names = names

    def existing(self) -> dict[str, str]:
        """Return {identity_hash: status} for every recorded run carrying the tag.

        Pages through the matching runs so a large experiment is fully read.
        Experiments that do not exist yet are skipped, so a first run against a
        fresh experiment returns an empty map.
        """
        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.tracking.MlflowClient()

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
                page_token=page_token,
            )
            for run in page:
                hash_value = run.data.tags.get(self.hash_tag)
                if hash_value:
                    out[hash_value] = run.info.status
            page_token = getattr(page, "token", None)
            if not page_token:
                return out
