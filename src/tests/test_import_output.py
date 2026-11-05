"""What users see on stderr when a pipeline command starts."""

import os
import subprocess
import sys


def test_unset_mirror_variables_print_nothing_at_import():
    """The pipeline takes these paths as options, so atomworks' notice about them is noise."""
    env = {k: v for k, v in os.environ.items() if k not in ("CCD_MIRROR_PATH", "PDB_MIRROR_PATH")}

    result = subprocess.run(
        [sys.executable, "-c", "import aw_of3_finetune.cli"],
        env=env, capture_output=True, text=True, check=True,
    )

    assert "MIRROR_PATH" not in result.stderr
