from __future__ import annotations

import contextlib

import pytest

from sweep.fas2sweep import fas2sweep
from sweep.orthbase import orthbase


def test_legacy_functions_issue_deprecation_warnings() -> None:
    with pytest.deprecated_call(), contextlib.suppress(Exception):
        orthbase(10, 2)

    with pytest.deprecated_call(), contextlib.suppress(Exception):
        fas2sweep(None, None)
