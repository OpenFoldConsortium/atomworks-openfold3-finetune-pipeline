"""Keep tests from leaking os.environ changes into later tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def restored_environ() -> Iterator[None]:
    """Restore os.environ on exit, undoing every change made inside, including by code under test."""
    with patch.dict(os.environ):
        yield
