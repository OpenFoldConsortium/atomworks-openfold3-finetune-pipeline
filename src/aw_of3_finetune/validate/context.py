"""Validation context: dataset locations plus cached accessors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openfold3.core.data.io.dataset_cache import read_datacache

from aw_of3_finetune.constants import CLEANED_STRUCTURES_DIRNAME, COMPRESSED_STRUCTURES_DIRNAME


@dataclass
class ValidationContext:
    """Paths and options for one validation run, with lazily cached loaders."""

    output_dir: Path
    cif_dir: Path
    msa_dir: Path | None = None
    template_cache_dir: Path | None = None
    template_structure_dir: Path | None = None
    ccd_path: Path | None = None
    expect_ligands: bool = False
    cache_path: Path | None = None
    npz_dir: Path | None = None
    _dataset_cache: object = field(default=None, init=False, repr=False)
    _npz: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Default ``cache_path`` and ``npz_dir`` to their locations under ``output_dir``."""
        if self.cache_path is None:
            self.cache_path = self.output_dir / "dataset_cache.json"
        if self.npz_dir is None:
            self.npz_dir = self.output_dir / COMPRESSED_STRUCTURES_DIRNAME

    @property
    def dataset_cache(self):
        """OF3 ``ClusteredDatasetCache`` read from ``cache_path``."""
        if self._dataset_cache is None:
            self._dataset_cache = read_datacache(self.cache_path)
        return self._dataset_cache

    def entries(self) -> list[str]:
        """Sorted entry IDs in the dataset cache."""
        return sorted(self.dataset_cache.structure_data)

    def npz(self, entry: str) -> dict[str, np.ndarray]:
        """Load ``<npz_dir>/<id>/<id>.npz`` as a dict of arrays (cached).

        Args:
            entry: Entry ID.

        Raises:
            FileNotFoundError: If the NPZ file is missing.
        """
        if entry not in self._npz:
            path = self.npz_dir / entry / f"{entry}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                self._npz[entry] = {k: data[k] for k in data.files}
        return self._npz[entry]

    def augmented_cif(self, entry: str) -> Path:
        """Path to ``cleaned_structures/<id>.cif``."""
        return self.output_dir / CLEANED_STRUCTURES_DIRNAME / f"{entry}.cif"

    def source_cif(self, entry: str) -> Path:
        """Path to the input CIF in ``cif_dir`` whose stem matches case-insensitively.

        Raises:
            FileNotFoundError: If no CIF in ``cif_dir`` matches.
        """
        for path in sorted(Path(self.cif_dir).glob("*.cif")):
            if path.stem.lower() == entry.lower():
                return path
        raise FileNotFoundError(f"no source CIF for {entry} in {self.cif_dir}")

    def sdf(self, mol_id: str) -> Path:
        """Path to ``reference_mols/<mol_id>.sdf``."""
        return self.output_dir / "reference_mols" / f"{mol_id}.sdf"
