"""Our atomworks-based cleanup must match OF3's cleanup_structure_of3 atom for atom."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.pipelines.preprocessing.structure import cleanup_structure_of3
from openfold3.core.data.primitives.structure.metadata import get_cif_block

from aw_of3_finetune.cleanup import CleanupSettings
from aw_of3_finetune.pipeline import parse_augmented_cif_for_training, prepare_augmented_cif

FIXTURES = Path(__file__).parent / "fixtures" / "mmcifs"
PDB_IDS = ["6hj2", "7axa", "1a3n", "1fdl", "1UBI", "1UBQ", "2crb", "2q2k", "3u8v", "4zey", "7riv"]


def _summary(atom_array: AtomArray) -> dict:
    """Summarize an AtomArray for order-independent comparison.

    Args:
        atom_array: Structure to summarize.

    Returns:
        Atom count, residue Counter, bond Counter and number of distinct chain IDs.
    """
    aa = atom_array
    boundaries = np.nonzero((aa.res_id[1:] != aa.res_id[:-1]) | (aa.chain_id[1:] != aa.chain_id[:-1]))[0] + 1
    starts = np.r_[0, boundaries, len(aa)]
    residues = Counter(
        (aa.res_name[s], tuple(aa.atom_name[s:e]), tuple(np.isnan(aa.coord[s:e]).any(axis=1)))
        for s, e in zip(starts[:-1], starts[1:])
    )
    bonds = Counter(
        tuple(sorted([(aa.res_name[i], aa.atom_name[i]), (aa.res_name[j], aa.atom_name[j])])) + (int(t),)
        for i, j, t in aa.bonds.as_array()
    )
    return {
        "n_atoms": len(aa),
        "residues": residues,
        "bonds": bonds,
        "n_chains": len(set(aa.chain_id)),
    }


def _assert_coords_match(ours: AtomArray, reference: AtomArray, tol: float = 0.02) -> None:
    """Check both arrays hold the same atoms at the same positions, up to ``tol`` Angstrom.

    Atoms are matched on (res_name, res_id, atom_name); unresolved atoms (NaN) must agree
    in count per key. Symmetry copies may differ by CIF rounding, hence the tolerance.

    Args:
        ours: Our cleaned structure.
        reference: OF3's cleaned structure.
        tol: Largest allowed distance between matched atoms, in Angstrom.
    """
    key_ids: dict[tuple, int] = {}

    def points(aa: AtomArray) -> tuple[np.ndarray, Counter]:
        keys = [key_ids.setdefault(k, len(key_ids)) for k in zip(aa.res_name, aa.res_id, aa.atom_name, strict=True)]
        resolved = ~np.isnan(aa.coord).any(axis=1)
        # A fourth coordinate far apart per key keeps atoms of different keys from matching.
        pts = np.c_[aa.coord[resolved], 1000.0 * np.asarray(keys)[resolved]]
        return pts, Counter(np.asarray(keys)[~resolved])

    our_pts, our_nan = points(ours)
    ref_pts, ref_nan = points(reference)
    assert our_nan == ref_nan
    assert len(our_pts) == len(ref_pts)
    assert cKDTree(ref_pts).query(our_pts)[0].max() <= tol
    assert cKDTree(our_pts).query(ref_pts)[0].max() <= tol


@pytest.mark.parametrize("pdb_id", PDB_IDS)
def test_matches_of3_cleanup(pdb_id: str, ccd_path: Path, tmp_path: Path):
    """Run both pipelines on a fixture CIF; atoms, bonds and coordinates must agree.

    Alternate conformers are chosen as OF3 does (highest occupancy), so only cleanup differs.
    """
    src = FIXTURES / f"{pdb_id}.cif"
    ccd = CIFFile.read(ccd_path)

    settings = CleanupSettings(altloc="highest_occupancy")
    prepare_augmented_cif(src, pdb_id, tmp_path / f"{pdb_id}.cif", ccd=ccd, cleanup_settings=settings)
    ours = parse_augmented_cif_for_training(tmp_path / f"{pdb_id}.cif").atom_array

    parsed = parse_mmcif(src, expand_bioassembly=True, include_bonds=True, renumber_chain_ids=True)
    reference = cleanup_structure_of3(parsed.atom_array, get_cif_block(parsed.structure_file), ccd)

    # Same atom count, residues, bonds and chain count as OF3, ignoring order.
    assert _summary(ours) == _summary(reference)

    # Same atoms at the same positions (within 0.02 A) as OF3.
    _assert_coords_match(ours, reference)
