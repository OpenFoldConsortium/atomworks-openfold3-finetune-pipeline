"""Tests leave os.environ as they found it, including the user's $CCD_PATH and $OPENFOLD_CACHE."""

from __future__ import annotations

import os

from environ import restored_environ

ADDED = "AW_OF3_TEST_ADDED"
REMOVED = "AW_OF3_TEST_REMOVED"
CHANGED = "AW_OF3_TEST_CHANGED"


def test_restored_environ_undoes_direct_writes():
    os.environ[REMOVED] = "kept"
    os.environ[CHANGED] = "original"
    try:
        with restored_environ():
            os.environ[ADDED] = "leak"
            del os.environ[REMOVED]
            os.environ[CHANGED] = "leak"

        assert ADDED not in os.environ
        assert os.environ[REMOVED] == "kept"
        assert os.environ[CHANGED] == "original"
    finally:
        os.environ.pop(REMOVED, None)
        os.environ.pop(CHANGED, None)


def test_restored_environ_restores_after_an_exception():
    try:
        with restored_environ():
            os.environ[ADDED] = "leak"
            raise RuntimeError
    except RuntimeError:
        pass

    assert ADDED not in os.environ


def test_tests_see_no_user_ccd_settings():
    # isolate_ccd_settings: the user's values are hidden inside each test and restored after it.
    assert "CCD_PATH" not in os.environ
    assert "openfold_cache" in os.environ["OPENFOLD_CACHE"]
