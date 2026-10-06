"""Expose optional integrations kept separate from the core API."""

from gridspan.extras.mlflow import MlflowStore

__all__ = ["MlflowStore"]