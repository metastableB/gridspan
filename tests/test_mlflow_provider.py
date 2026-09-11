"""Integration tests for the MLflow-backed StatusProvider.

These log real runs into a temporary local MLflow store (sqlite backend) and
read them back through MlflowProvider, so the tag query and pagination are
exercised end to end. Skipped when mlflow is not installed.
"""

from __future__ import annotations

import pytest

from gridspan.core import identity
from gridspan.filters import dedup
from gridspan.providers import MlflowProvider

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


def test_existing_reads_logged_hashes(tracking_uri):
    a = {"model.name": "x", "runtime.bs": 16}
    b = {"model.name": "y", "runtime.bs": 16}
    _log_run("exp", identity(a), "FINISHED")
    _log_run("exp", identity(b), "FAILED")

    provider = MlflowProvider("exp", tracking_uri=tracking_uri)
    got = provider.existing()
    assert got == {identity(a): "FINISHED", identity(b): "FAILED"}


def test_existing_is_empty_for_a_fresh_experiment(tracking_uri):
    provider = MlflowProvider("never-created", tracking_uri=tracking_uri)
    assert provider.existing() == {}


def test_dedup_skips_finished_reruns_failed(tracking_uri):
    a = {"model.name": "x", "runtime.bs": 16}  # will be FINISHED
    b = {"model.name": "y", "runtime.bs": 16}  # will be FAILED
    c = {"model.name": "z", "runtime.bs": 16}  # never run
    _log_run("exp", identity(a), "FINISHED")
    _log_run("exp", identity(b), "FAILED")

    provider = MlflowProvider("exp", tracking_uri=tracking_uri)
    # default skip=("FINISHED",): a is skipped, b (failed) and c (new) remain.
    assert dedup([a, b, c], provider=provider) == [b, c]
