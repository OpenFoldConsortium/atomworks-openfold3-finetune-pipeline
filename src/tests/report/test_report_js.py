"""Runs the report's Node unit tests so pytest covers the JS helpers."""

import shutil
import subprocess
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parents[3] / "src" / "aw_of3_finetune" / "report" / "assets" / "test"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_report_js_unit_tests_pass():
    """All `node --test` tests under the report assets pass."""
    test_files = sorted(TEST_DIR.glob("*.test.mjs"))
    assert test_files, f"no *.test.mjs files under {TEST_DIR}"
    result = subprocess.run(["node", "--test", *map(str, test_files)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
