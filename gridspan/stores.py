"""Look up which grid points to skip, from a local file."""

from __future__ import annotations

from collections.abc import Callable, Collection
from pathlib import Path

# Given run hashes, return the ones to skip.
Store = Callable[[list[str]], Collection[str]]


class TextStore:
    """Skip grid points whose hashes are listed in a text file, one per line.

    The file is read into memory once, on first use. Blank lines are ignored,
    and a missing file skips nothing. gridspan never writes this file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._hashes: set[str] | None = None

    def __call__(self, run_hashes: list[str]) -> set[str]:
        """Return the hashes in run_hashes that are listed in the file."""
        if self._hashes is None:
            self._hashes = self._read()
        return {h for h in run_hashes if h in self._hashes}

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open(encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}