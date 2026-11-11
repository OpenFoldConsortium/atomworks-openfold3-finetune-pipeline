"""What the validity checks read: one pipeline output dir, its input CIFs and the CCD."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz

from aw_of3_finetune.constants import COMPRESSED_STRUCTURES_DIRNAME, PREP_DROPPED_FILE
from aw_of3_finetune.steplog import StepLog


@dataclass(frozen=True)
class ValidityInputs:
    """Paths of a pipeline run: its output dir, its input CIF dir, the CCD file and the cache.

    cache_path defaults to out_dir/dataset_cache.json (--cache-output). The NPZ files are always read from
    out_dir/of3_compressed_structures.
    """

    out_dir: Path
    cif_dir: Path
    ccd_path: Path
    cache_path: Path | None = None
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def npz_dir(self) -> Path:
        """Where the pipeline wrote the NPZ files: out_dir/of3_compressed_structures."""
        return self.out_dir / COMPRESSED_STRUCTURES_DIRNAME

    def manifest_ids(self) -> list[str]:
        """Every structure id of the dataset, in manifest order.

        Returns:
            The ids from out_dir/manifest.json.
        """
        return json.loads((self.out_dir / "manifest.json").read_text())["structures"]

    def entries(self) -> list[str]:
        """Structure ids to check, in manifest order (a report on a subset narrows these).

        Returns:
            The ids from out_dir/manifest.json.
        """
        return self.manifest_ids()

    def final_atoms(self, pdb_id: str) -> AtomArray:
        """The pipeline's final atoms of an entry.

        Args:
            pdb_id: Structure id.

        Returns:
            The atoms from <npz_dir>/<id>/<id>.npz.
        """
        return read_atomarray_from_npz(self.npz_dir / pdb_id / f"{pdb_id}.npz")

    def step_log(self, pdb_id: str) -> StepLog | None:
        """The cleanup step log of an entry.

        Args:
            pdb_id: Structure id.

        Returns:
            The log from cleanup_steps/<id>.json, or None if there is none.
        """
        path = self.out_dir / "cleanup_steps" / f"{pdb_id}.json"
        return StepLog.from_json(path.read_text()) if path.exists() else None

    def input_cifs(self) -> dict[str, Path]:
        """The input CIFs by structure id (the lowercased stem, as finetune-prep names ids), read once.

        Returns:
            {id: cif_dir/<stem>.cif} for every CIF in cif_dir.
        """
        if "input_cifs" not in self._cache:
            self._cache["input_cifs"] = {path.stem.lower(): path for path in sorted(self.cif_dir.glob("*.cif"))}
        return self._cache["input_cifs"]

    def input_cif(self, pdb_id: str) -> Path:
        """Path of an entry's input CIF, whatever the case of its file name.

        Args:
            pdb_id: Structure id.

        Returns:
            The input CIF whose lowercased stem is the id; cif_dir/<id>.cif if there is none.
        """
        return self.input_cifs().get(pdb_id.lower(), self.cif_dir / f"{pdb_id}.cif")

    def cache_entry(self, pdb_id: str) -> dict:
        """An entry's record in the pipeline's dataset cache.

        Args:
            pdb_id: Structure id.

        Returns:
            structure_data[<id>] of cache_path (default out_dir/dataset_cache.json).
        """
        if "dataset_cache" not in self._cache:
            path = self.cache_path or self.out_dir / "dataset_cache.json"
            self._cache["dataset_cache"] = json.loads(path.read_text())
        return self._cache["dataset_cache"]["structure_data"][pdb_id]

    def validation_report(self) -> dict:
        """The pipeline's validate-stage report, read once.

        Returns:
            out_dir/validation.json (entries_in, entries_out, findings, dropped, ...); empty if the file is missing
            or older than manifest.json (left over from a run whose validate stage was skipped).
        """
        if "validation" not in self._cache:
            path, manifest = self.out_dir / "validation.json", self.out_dir / "manifest.json"
            stale = path.exists() and manifest.exists() and path.stat().st_mtime < manifest.stat().st_mtime
            self._cache["validation"] = json.loads(path.read_text()) if path.exists() and not stale else {}
        return self._cache["validation"]

    def validation_findings(self, pdb_id: str | None) -> list[dict]:
        """An entry's findings from the pipeline's validate stage.

        Args:
            pdb_id: Structure id; None for the dataset-level findings.

        Returns:
            The findings of out_dir/validation.json for the entry (keys check, severity, entry,
            message, detail); empty if the file is missing.
        """
        return [finding for finding in self.validation_report().get("findings", []) if finding["entry"] == pdb_id]

    def prep_drop_reasons(self) -> dict[str, str]:
        """Why finetune-prep dropped each structure it dropped.

        Returns:
            {id: reason} from out_dir/prep_dropped.json; empty if the file is missing (older outputs).
        """
        path = self.out_dir / PREP_DROPPED_FILE
        return {item["id"]: item["reason"] for item in json.loads(path.read_text())} if path.exists() else {}

    def ccd(self) -> CIFFile:
        """The chemical component dictionary, read once.

        Returns:
            The parsed CCD file.
        """
        if "ccd" not in self._cache:
            self._cache["ccd"] = CIFFile.read(self.ccd_path)
        return self._cache["ccd"]
