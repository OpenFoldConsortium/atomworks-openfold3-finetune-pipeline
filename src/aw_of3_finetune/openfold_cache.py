"""Where the pipeline finds the CCD: a path stored in OpenFold3's cache directory.

OpenFold3 keeps per-user state in ``$OPENFOLD_CACHE`` (default ``~/.openfold3``); its
``setup_openfold`` writes a one-line ``ckpt_root`` file there naming the weights
directory. ``setup-finetuning`` writes a ``ccd_path`` file beside it the same way, so
commands that need the CCD can find it without ``--ccd-path``.
"""

from __future__ import annotations

import os
from pathlib import Path

# OpenFold3's own default (openfold3.entry_points.parameters.DEFAULT_CACHE_PATH); kept
# as a literal so resolving a path never imports openfold3's entry points.
DEFAULT_OPENFOLD_CACHE = Path("~/.openfold3")
CCD_PATH_FILENAME = "ccd_path"
# The file setup_openfold writes, naming the weights directory (openfold3.entry_points.parameters).
CHECKPOINT_ROOT_FILENAME = "ckpt_root"
CCD_ENV_VAR = "CCD_PATH"

# OpenFold3's copy of the CCD. Use it rather than wwPDB's latest, whose ligand charges have changed.
CCD_S3_BUCKET = "openfold3-data"
CCD_S3_KEY = "chemical_component_dictionary.cif"

SETUP_HINT = "Run `pixi run setup-finetuning` once to download the CCD and store its path, or pass --ccd-path."


def openfold_cache_dir() -> Path:
    """Return OpenFold3's cache directory: ``$OPENFOLD_CACHE``, else ``~/.openfold3``."""
    return Path(os.environ.get("OPENFOLD_CACHE") or DEFAULT_OPENFOLD_CACHE).expanduser()


def stored_ccd_path_file() -> Path:
    """Return the file that holds the stored CCD path."""
    return openfold_cache_dir() / CCD_PATH_FILENAME


def read_stored_ccd_path() -> Path | None:
    """Return the CCD path stored by setup-finetuning, or None if none is stored."""
    record = stored_ccd_path_file()
    if not record.is_file():
        return None
    text = record.read_text().strip()
    return Path(text).expanduser() if text else None


def write_stored_ccd_path(ccd_path: Path) -> Path:
    """Store ``ccd_path`` (made absolute) in the cache directory.

    Args:
        ccd_path: The CCD mmCIF to record.

    Returns:
        The file the path was written to.
    """
    record = stored_ccd_path_file()
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(f"{ccd_path.expanduser().resolve()}\n")
    return record


def resolve_ccd_path(ccd_path: Path | None) -> tuple[Path | None, str]:
    """Pick the CCD: ``ccd_path`` if given, else ``$CCD_PATH``, else the stored path.

    Args:
        ccd_path: The --ccd-path value, or None.

    Returns:
        The chosen path (None when nothing is set) and where it came from, for messages.
    """
    if ccd_path is not None:
        return Path(ccd_path).expanduser(), "--ccd-path"
    env = os.environ.get(CCD_ENV_VAR)
    if env:
        return Path(env).expanduser(), f"${CCD_ENV_VAR}"
    stored = read_stored_ccd_path()
    if stored is not None:
        return stored, str(stored_ccd_path_file())
    return None, ""
