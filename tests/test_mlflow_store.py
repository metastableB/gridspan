"""Integration tests for MLflow status lookup.

These log real runs into a temporary local MLflow store (sqlite backend) and
read them back through MlflowStore, so the tag query and pagination are
exercised end to end. Skipped when mlflow is not installed.
"""

from __future__ import annotations

import time

import pytest

from gridspan.core import GridSpan, identity
from gridspan.extras import MlflowStore

mlflow = pytest.importorskip("mlflow")


def _log_run(experiment: str, hash_value: str, status: str) -> None:
    """Log one run carrying the identity hash tag and end it in `status`."""
    mlflow.set_experiment(experiment)
    mlflow.start_run()
    mlflow.set_tag("gridspan.id.hash", hash_value)
    # End with the wanted status directly; a `with` block would override it
    # with FINISHED on exit.
    mlflow.end_run(status=status)


@pytest.fixture
def tracking_uri(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    return uri


def test_store_returns_hashes_with_chosen_statuses(tracking_uri):
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"model.name": "y", "runtime.bs": 16}
    _log_run("exp", identity(a), "FINISHED")
    _log_run("exp", identity(b), "FAILED")
    hashes = [identity(a), identity(b), "never-run"]

    finished = MlflowStore("exp", tracking_uri=tracking_uri, skip={"FINISHED"})
    assert finished(hashes) == {identity(a)}
    either = MlflowStore(
        "exp", tracking_uri=tracking_uri, skip={"FINISHED", "FAILED"}
    )
    assert either(hashes) == {identity(a), identity(b)}


def test_store_skips_nothing_for_a_fresh_experiment(tracking_uri):
    store = MlflowStore("never-created", tracking_uri=tracking_uri, skip={"FINISHED"})
    assert store(["any-hash"]) == set()


def test_store_rejects_a_bare_string_for_skip():
    with pytest.raises(TypeError, match="collection of statuses"):
        MlflowStore("exp", skip="FINISHED")


def test_store_leaves_the_global_tracking_uri_unchanged(tracking_uri, tmp_path):
    _log_run("exp", "hash", "FINISHED")
    other = f"sqlite:///{tmp_path / 'other.db'}"
    mlflow.set_tracking_uri(other)
    store = MlflowStore("exp", tracking_uri=tracking_uri, skip={"FINISHED"})
    assert store(["hash"]) == {"hash"}
    assert mlflow.get_tracking_uri() == other


def test_dedup_skips_finished_reruns_failed(tracking_uri):
    a = {"model.name": "x", "runtime.bs": 16}  # will be FINISHED
    b = {"model.name": "y", "runtime.bs": 16}  # will be FAILED
    c = {"model.name": "z", "runtime.bs": 16}  # never run
    _log_run("exp", identity(a), "FINISHED")
    _log_run("exp", identity(b), "FAILED")

    store = MlflowStore("exp", tracking_uri=tracking_uri, skip={"FINISHED"})
    assert GridSpan([a, b, c]).dedup(store=store) == [b, c]


@pytest.mark.parametrize(
    "statuses, skipped",
    [(["FAILED", "FINISHED"], {"hash"}), (["FINISHED", "FAILED"], set())],
)
def test_store_uses_the_newest_run_for_a_hash(tracking_uri, statuses, skipped):
    for status in statuses:
        _log_run("exp", "hash", status)
        time.sleep(0.01)  # distinct start times
    store = MlflowStore("exp", tracking_uri=tracking_uri, skip={"FINISHED"})
    assert store(["hash"]) == skipped
