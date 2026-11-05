"""Thin wrappers around atomworks.io's parse/write entry points (atomworks >= 3.0).

parse() takes a ParseConfig and to_cif_buffer() a CIFWriteConfig (flat keyword arguments
and ``filename=`` are deprecated in 3.0). This module pins the settings this pipeline
uses so every caller parses and writes the same way.
"""

import io
import logging
from os import PathLike
from pathlib import Path
from typing import Any

import atomworks.io as aw_io
import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_buffer
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from aw_of3_finetune.altloc import select_highest_occupancy_altlocs
from aw_of3_finetune.cleanup import CleanupSettings
from aw_of3_finetune.steplog import StepRecorder, read_input_atoms

logger = logging.getLogger(__name__)

# Seed for atomworks' random_clash_aware altloc choice, so every run picks the same conformers.
ALTLOC_SEED = 0
_ATOMWORKS_ALTLOC = {"clash_aware": "random_clash_aware", "first": "first"}

STANDARD_PARSE_KWARGS: dict[str, Any] = {
    "add_missing_atoms": True,
    # Waters and MSE are left to cleanup.clean_structure (OF3's functions), which logs them.
    "remove_waters": False,
    "convert_mse_to_met": False,
    "remove_ccds": (),
    "hydrogen_policy": "remove",
    "build_assembly": ["1"],
    "model": 1,
}


def _heavy_atoms(cif: CIFFile) -> AtomArray:
    """Model-1 heavy atoms, all altlocs, as in the step-log input table."""
    buffer = io.StringIO()
    cif.write(buffer)
    buffer.seek(0)
    return read_input_atoms(buffer)


def _parse_highest_occupancy(cif_path: PathLike, recorder: StepRecorder | None) -> dict:
    """Parse after filtering the CIF to each residue's highest-occupancy conformer (OF3's choice)."""
    cif = CIFFile.read(cif_path)
    n_residues = select_highest_occupancy_altlocs(cif)
    if recorder is not None:
        recorder.record(
            "altloc_highest_occupancy",
            "Keep each residue's highest-occupancy alternate conformer.",
            lambda: _heavy_atoms(cif),
        )
    if not n_residues:
        return aw_io.parse(str(cif_path), config=ParseConfig(**STANDARD_PARSE_KWARGS))
    logger.info(
        f"{Path(cif_path).stem.lower()}: altloc kept the highest-occupancy conformer for {n_residues} residue(s)"
    )
    buffer = io.StringIO()
    cif.write(buffer)
    buffer.seek(0)
    return aw_io.parse(buffer, config=ParseConfig(**STANDARD_PARSE_KWARGS, file_type="cif"))


def parse_structure(
    cif_path: PathLike,
    settings: CleanupSettings | None = None,
    recorder: StepRecorder | None = None,
) -> dict:
    """Parse an mmCIF with this pipeline's standard AtomWorks settings.

    Alternate conformers follow ``settings.altloc``: atomworks' first letter per chain by default;
    clash_aware falls back to that when atomworks finds no clash-free choice.

    Args:
        cif_path: mmCIF file to parse.
        settings: Cleanup switches; None means the defaults.
        recorder: Step recorder; None records nothing.

    Returns:
        atomworks.io.parse's result dict (``"asym_unit"``, ``"assemblies"``, ...).
    """
    settings = settings or CleanupSettings()
    if recorder is not None:
        recorder.record(
            "input",
            "Input atoms: model 1, heavy atoms, all altlocs.",
            lambda: read_input_atoms(Path(cif_path)),
        )
    if settings.altloc == "highest_occupancy":
        result = _parse_highest_occupancy(cif_path, recorder)
    else:
        config = ParseConfig(
            **STANDARD_PARSE_KWARGS, altloc=_ATOMWORKS_ALTLOC[settings.altloc], altloc_seed=ALTLOC_SEED
        )
        try:
            result = aw_io.parse(str(cif_path), config=config)
        except ValueError as error:
            if settings.altloc != "clash_aware":
                raise
            logger.warning(
                f"{Path(cif_path).stem.lower()}: clash-aware altloc choice failed ({error}); "
                "used the first altloc letter per chain instead"
            )
            config = ParseConfig(**STANDARD_PARSE_KWARGS, altloc="first")
            result = aw_io.parse(str(cif_path), config=config)
    if recorder is not None:
        recorder.record(
            "atomworks_parse",
            "AtomWorks parse: alternate conformers chosen, hydrogens removed, missing atoms added, names standardised.",
            result["asym_unit"][0],
        )
        recorder.record(
            "build_assembly_1",
            "Build biological assembly 1 (first copy).",
            result["assemblies"]["1"][0],
        )
    return result


def write_cif_buffer(
    atom_array: AtomArray,
    *,
    id: str,
    extra_categories: dict[str, dict[str, Any]],
    as_bcif: bool = False,
):
    """Serialize an AtomArray, plus extra CIF categories, to a CIF/BCIF buffer.

    Args:
        atom_array: Structure to write.
        id: Entry ID written as the data block name.
        extra_categories: {category: {column: values}} written alongside the atoms.
        as_bcif: Write BinaryCIF instead of text CIF.

    Returns:
        The buffer from atomworks' to_cif_buffer.
    """
    config = CIFWriteConfig(
        id=id,
        extra_categories=extra_categories,
        # OF3's parse_mmcif() needs chem_comp.id/type (bare KeyError: 'chem_comp'
        # otherwise); we write it via extra_categories (build_chem_comp_category).
        # Switching to True would change the written CIFs -- a deliberate, separate change.
        include_chem_comp=False,
        # Without it, bioassembly symmetry copies of one entity (e.g. 2q2k) raise
        # "Ambiguous chain annotations detected"; writes "A_1", "A_2". No-op otherwise.
        chain_disambiguation="chain_iid",
    )
    return to_cif_buffer(atom_array, config=config, as_bcif=as_bcif)


def build_chem_comp_category(atom_array: AtomArray, ccd: CIFFile) -> dict[str, list[str]]:
    """Build a minimal ``chem_comp`` category (id + type per unique residue) from the CCD.

    Passed via ``extra_categories``; see write_cif_buffer's include_chem_comp comment.

    Args:
        atom_array: Structure whose residue names to describe.
        ccd: The Chemical Component Dictionary; residues missing from it are skipped.

    Returns:
        ``{"id": [...], "type": [...]}``, sorted by residue name.
    """
    res_names = sorted(set(np.unique(atom_array.res_name)))
    ids: list[str] = []
    types: list[str] = []
    for res_name in res_names:
        if res_name not in ccd:
            continue
        ids.append(res_name)
        types.append(str(ccd[res_name]["chem_comp"]["type"].as_array()[0]))
    return {"id": ids, "type": types}
