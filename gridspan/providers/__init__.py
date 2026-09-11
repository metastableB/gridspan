"""Backend adapters that satisfy the StatusProvider protocol.

Each adapter reads which run identities a store already holds, so dedup can
skip finished work. Adapters live here because they pull in optional, heavy
dependencies (for example MLflow); gridspan core never imports them. Each
adapter imports its own dependency lazily, so importing this package is safe
even when that dependency is absent.
"""

from gridspan.providers.mlflow import MlflowProvider

__all__ = ["MlflowProvider"]
