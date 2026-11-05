"""clean_structure's OF3 filters, each checked against OF3's own function.

Every test edits an atomworks-parsed array (or a temp copy of a fixture CIF, when the
source CIF is what matters), runs clean_structure, and compares the resolved atoms that
survive with what the OF3 function (plus OF3's terminal-atom removal) leaves of the same
resolved atoms.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, BondType
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.primitives.structure.cleanup import (
    convert_MSE_to_MET,
    prefilter_bonds,
    remove_chains_with_CA_gaps,
    remove_clashing_chains,
    remove_crystallization_aids,
    remove_fully_unknown_polymers,
    remove_non_CCD_atoms,
    remove_small_polymers,
    remove_std_residue_terminal_atoms,
    remove_waters,
)
from openfold3.core.data.primitives.structure.labels import assign_molecule_type_ids

from assertions import write_emptiable_minimal_3aa
from aw_of3_finetune.cleanup import _BOND_FILTER_ARGS, EmptyStructureError, Removal, clean_structure
from aw_of3_finetune.parsing import parse_structure

FIXTURES = Path(__file__).parent / "fixtures" / "mmcifs"


@pytest.fixture(scope="module")
def ccd(ccd_path: Path) -> CIFFile:
    """The parsed CCD, read once per module."""
    return CIFFile.read(ccd_path)


def _parse(src: Path) -> AtomArray:
    """Parse a CIF with the pipeline's atomworks settings (first assembly copy)."""
    return parse_structure(src)["assemblies"]["1"][0]


def _edited_cif(src: Path, tmp_path: Path, edits: dict[str, str]) -> Path:
    """Write a copy of a CIF with text substitutions applied.

    Args:
        src: CIF to copy.
        tmp_path: Directory to write into.
        edits: ``{old text: new text}`` replacements.

    Returns:
        Path of the edited copy.
    """
    text = src.read_text()
    for old, new in edits.items():
        assert old in text, old
        text = text.replace(old, new)
    out = tmp_path / src.name
    out.write_text(text)
    return out


def _with_new_chain(arr: AtomArray, mask: np.ndarray, chain_iid: str, shift: float = 0.0) -> AtomArray:
    """Append a copy of the masked atoms as a new chain.

    Args:
        arr: Structure to extend.
        mask: Atoms to copy.
        chain_iid: chain_iid (and chain_id) of the copy.
        shift: Translation of the copy along x, in Angstrom.

    Returns:
        ``arr`` followed by the new chain.
    """
    copy = arr[mask].copy()
    copy.coord = copy.coord + np.array([shift, 0.0, 0.0])
    for name in ("chain_iid", "chain_id"):
        copy.set_annotation(name, np.full(len(copy), chain_iid))
    return arr + copy


def _resolved_keys(arr: AtomArray) -> set[tuple]:
    """(chain_iid, res_id, atom_name) of every resolved atom."""
    resolved = arr[~np.isnan(arr.coord).any(axis=1)]
    return set(zip(resolved.chain_iid, resolved.res_id, resolved.atom_name, strict=True))


def _of3_reference(arr: AtomArray, source: Path, step, **kwargs) -> tuple[set, set]:
    """Apply an OF3 function to the resolved atoms of ``arr``.

    Args:
        arr: Atomworks array (before clean_structure).
        source: CIF read for molecule types.
        step: OF3 cleanup function taking an AtomArray.
        **kwargs: Extra arguments for ``step``.

    Returns:
        ``(removed keys, expected surviving keys)`` after OF3's terminal-atom removal.
    """
    ref = arr.copy()
    convert_MSE_to_MET(ref)
    ref = remove_waters(ref)
    assign_molecule_type_ids(ref, CIFFile.read(source))
    ref.chain_id = ref.chain_iid
    resolved = ref[~np.isnan(ref.coord).any(axis=1)]
    kept = step(resolved, **kwargs)
    removed = _resolved_keys(resolved) - _resolved_keys(kept)
    return removed, _resolved_keys(remove_std_residue_terminal_atoms(kept))


def _assert_matches_of3(arr: AtomArray, source: Path, clean_ccd: CIFFile | None, step, **kwargs) -> tuple[AtomArray, list[Removal], set]:
    """Run clean_structure and check it kept exactly what OF3's function keeps.

    Returns:
        ``(cleaned array, removals, keys OF3 removed)``.
    """
    removed, expected = _of3_reference(arr, source, step, **kwargs)
    assert removed, "test input must trigger the OF3 step"
    cleaned, removals = clean_structure(arr, source, "test", clean_ccd)
    assert _resolved_keys(cleaned) == expected
    return cleaned, removals, removed


def _removal(removals: list[Removal], step: str) -> Removal:
    """The single Removal recorded for ``step``."""
    (match,) = [r for r in removals if r.step == step]
    return match


def test_aids_removed_for_xray():
    """For an X-ray structure, HEPES (EPE) is removed exactly as OF3's remove_crystallization_aids does."""
    src = FIXTURES / "2q2k.cif"
    arr = _parse(src)
    assert "EPE" in arr.res_name

    cleaned, removals, _ = _assert_matches_of3(arr, src, None, remove_crystallization_aids)

    # HEPES is gone from the cleaned structure.
    actual_residue_names = set(cleaned.res_name)
    assert "EPE" not in actual_residue_names

    # The aid removal was recorded with a nonzero atom count.
    actual_removed_atoms = _removal(removals, "remove_crystallization_aids").n_atoms
    assert actual_removed_atoms > 0


def test_aids_kept_for_non_xray(tmp_path: Path):
    """OF3 removes crystallization aids only for X-ray structures; an NMR 2q2k keeps HEPES (EPE)."""
    src = _edited_cif(FIXTURES / "2q2k.cif", tmp_path, {"'X-RAY DIFFRACTION'": "'SOLUTION NMR'"})
    arr = _parse(src)

    cleaned, removals = clean_structure(arr, src, "test", None)

    # HEPES is still in the cleaned structure.
    actual_residue_names = set(cleaned.res_name)
    assert "EPE" in actual_residue_names

    # No crystallization-aid removal was recorded.
    actual_steps = [removal.step for removal in removals]
    assert "remove_crystallization_aids" not in actual_steps


def test_small_polymer_removed(caplog: pytest.LogCaptureFixture):
    """A 3-residue chain is removed like OF3's remove_small_polymers, recorded and logged."""
    src = FIXTURES / "3u8v.cif"
    arr = _parse(src)
    first_three = np.unique(arr.res_id[arr.chain_iid == "A_1"])[:3]
    arr = _with_new_chain(arr, (arr.chain_iid == "A_1") & np.isin(arr.res_id, first_three), "Z_1", shift=100.0)

    with caplog.at_level(logging.INFO, logger="aw_of3_finetune.cleanup"):
        cleaned, removals, _ = _assert_matches_of3(arr, src, None, remove_small_polymers, max_residues=3)

    # The added short chain is gone from the cleaned structure.
    actual_chain_iids = set(cleaned.chain_iid)
    assert "Z_1" not in actual_chain_iids

    # The removal record names exactly that chain.
    actual_removed_chains = _removal(removals, "remove_small_polymers").chain_iids
    assert actual_removed_chains == ("Z_1",)

    # The log line names the structure id and the removed chain.
    (message,) = [record.getMessage() for record in caplog.records if "remove_small_polymers" in record.getMessage()]
    assert "test" in message and "('Z_1',)" in message


def test_fully_unknown_polymer_removed(tmp_path: Path):
    """A protein chain made only of UNK residues is removed like OF3's remove_fully_unknown_polymers."""
    # UNK is not in 3u8v's chem_comp; declare it so OF3 types the chain as protein.
    src = _edited_cif(
        FIXTURES / "3u8v.cif",
        tmp_path,
        {"_chem_comp.formula_weight \n": "_chem_comp.formula_weight \nUNK 'L-peptide linking' y UNKNOWN ? ? ?\n"},
    )
    arr = _parse(FIXTURES / "3u8v.cif")
    arr.res_name[arr.chain_iid == "A_1"] = "UNK"

    cleaned, removals, _ = _assert_matches_of3(arr, src, None, remove_fully_unknown_polymers)

    # The all-UNK chain is gone from the cleaned structure.
    actual_chain_iids = set(cleaned.chain_iid)
    assert "A_1" not in actual_chain_iids

    # The removal record names exactly that chain.
    actual_removed_chains = _removal(removals, "remove_fully_unknown_polymers").chain_iids
    assert actual_removed_chains == ("A_1",)


def test_clashing_chain_removed():
    """A duplicate of ligand P06 placed on top of itself makes one of the two clash-removed, like OF3."""
    src = FIXTURES / "7riv.cif"
    arr = _parse(src)
    arr = _with_new_chain(arr, arr.res_name == "P06", "Z_1")

    cleaned, removals, removed = _assert_matches_of3(arr, src, None, remove_clashing_chains, clash_distance=1.7, clash_percentage=0.3)

    # Exactly one chain lost the clash.
    (loser,) = {key[0] for key in removed}

    # The removal record names that loser.
    actual_removed_chains = _removal(removals, "remove_clashing_chains").chain_iids
    assert actual_removed_chains == (loser,)

    # The other clashing chain survives.
    expected_survivors = {"B_1", "Z_1"} - {loser}
    assert expected_survivors <= set(cleaned.chain_iid)


def _with_non_ccd_atom(arr: AtomArray) -> AtomArray:
    """Rename the first CB of a standard residue to XX1."""
    arr = arr.copy()
    arr.atom_name[np.nonzero(arr.atom_name == "CB")[0][0]] = "XX1"
    return arr


def test_non_ccd_atom_removed(ccd: CIFFile):
    """An atom name absent from the CCD (XX1) is removed like OF3's remove_non_CCD_atoms."""
    src = FIXTURES / "3u8v.cif"
    arr = _with_non_ccd_atom(_parse(src))

    cleaned, removals, removed = _assert_matches_of3(arr, src, ccd, remove_non_CCD_atoms, ccd=ccd)

    # The renamed atom is gone from the cleaned structure.
    actual_atom_names = set(cleaned.atom_name)
    assert "XX1" not in actual_atom_names

    # Exactly one atom was removed, and the record counts it.
    actual_removed_atoms = _removal(removals, "remove_non_CCD_atoms").n_atoms
    assert actual_removed_atoms == len(removed) == 1


@pytest.mark.filterwarnings("ignore:Residue 'AAJ' is not in the CCD")
def test_residue_missing_from_ccd_is_kept_and_logged(ccd: CIFFile, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A ligand whose name isn't in the CCD (custom ligands) is kept, not a crash.

    OF3's remove_non_CCD_atoms looks every residue up in the CCD and raises KeyError
    otherwise; OF3's own PDB data never hits this, user ligands do.
    """
    name = "AAJ"
    assert name not in ccd, "setup: test name must not be a CCD component"
    src = _edited_cif(FIXTURES / "7riv.cif", tmp_path, {"P06": name})
    arr = _parse(src)
    is_resolved = ~np.isnan(arr.coord).any(axis=1)
    expected_ligand_atoms = int(((arr.res_name == name) & is_resolved).sum())

    with caplog.at_level(logging.INFO, logger="aw_of3_finetune.cleanup"):
        cleaned, removals = clean_structure(arr, src, "7riv", ccd)

    # Every resolved atom of the custom ligand survives cleanup.
    actual_ligand_atoms = int((cleaned.res_name == name).sum())
    assert actual_ligand_atoms == expected_ligand_atoms

    # The non-CCD-atom filter removed nothing.
    actual_steps = [removal.step for removal in removals]
    assert "remove_non_CCD_atoms" not in actual_steps

    # The skipped residue is named in the log.
    expected_log = f"7riv: remove_non_CCD_atoms skipped residue(s) not in the CCD: {name}"
    assert expected_log in caplog.messages


def test_clean_structure_without_ccd_skips_ccd_steps():
    """With no CCD supplied, the non-CCD-atom filter is skipped and XX1 survives."""
    src = FIXTURES / "3u8v.cif"

    cleaned, removals = clean_structure(_with_non_ccd_atom(_parse(src)), src, "test", None)

    # The renamed atom is still present.
    actual_atom_names = set(cleaned.atom_name)
    assert "XX1" in actual_atom_names

    # No non-CCD-atom removal was recorded.
    actual_steps = [removal.step for removal in removals]
    assert "remove_non_CCD_atoms" not in actual_steps


def test_ca_gap_chain_removed():
    """A chain whose second half is shifted 20 A (a CA gap) is removed like OF3's remove_chains_with_CA_gaps."""
    src = FIXTURES / "2crb.cif"
    arr = _parse(src)
    arr = _with_new_chain(arr, arr.chain_iid == "A_1", "Z_1", shift=100.0)
    second_half = (arr.chain_iid == "A_1") & (arr.res_id > np.median(arr.res_id))
    arr.coord[second_half, 0] += 20.0

    cleaned, removals, _ = _assert_matches_of3(arr, src, None, remove_chains_with_CA_gaps, distance_threshold=10.0)

    # The gapped chain is removed; the intact copy Z_1 stays.
    actual_chain_iids = set(cleaned.chain_iid)
    assert "A_1" not in actual_chain_iids
    assert "Z_1" in actual_chain_iids

    # The removal record names exactly the gapped chain.
    actual_removed_chains = _removal(removals, "remove_chains_with_CA_gaps").chain_iids
    assert actual_removed_chains == ("A_1",)


def test_long_bonds_dropped():
    """A bond between two distant CA atoms is dropped while every atom is kept, as with OF3's prefilter_bonds."""
    src = FIXTURES / "3u8v.cif"
    arr = _parse(src)
    ca = np.nonzero((arr.chain_iid == "A_1") & (arr.atom_name == "CA"))[0][:2]
    assert np.linalg.norm(arr.coord[ca[0]] - arr.coord[ca[1]]) > 2.4
    arr.bonds.add_bond(int(ca[0]), int(ca[1]), BondType.SINGLE)

    cleaned, _ = clean_structure(arr, src, "test", None)

    # OF3's bond filter removes no atoms, so the surviving atoms match its reference.
    removed, expected_keys = _of3_reference(arr, src, prefilter_bonds, **_BOND_FILTER_ARGS)
    assert not removed
    assert _resolved_keys(cleaned) == expected_keys

    # The artificial long bond is no longer present in the cleaned structure.
    index = {key: i for i, key in enumerate(zip(cleaned.chain_iid, cleaned.res_id, cleaned.atom_name, strict=True))}
    first_ca, second_ca = (index[("A_1", arr.res_id[i], "CA")] for i in ca)
    actual_bonded_to_first = set(cleaned.bonds.get_bonds(first_ca)[0])
    assert second_ca not in actual_bonded_to_first


def _bonded(arr: AtomArray, first: tuple, second: tuple) -> bool:
    """Whether the atoms keyed (chain_iid, res_id, atom_name) are bonded in arr."""
    index = {key: i for i, key in enumerate(zip(arr.chain_iid, arr.res_id, arr.atom_name, strict=True))}
    return index[second] in set(arr.bonds.get_bonds(index[first])[0])


def test_stretched_link_to_nonstandard_residue_is_kept():
    """3ogo's chromophore CRO 76 stays linked to LEU 75 although the deposited link is 2.40 A (cutoff 2.4)."""
    src = FIXTURES / "3ogo.cif"

    cleaned, _ = clean_structure(_parse(src), src, "test", None)

    leu75_c, cro76_n1 = ("A_1", 75, "C"), ("A_1", 76, "N1")
    assert _bonded(cleaned, leu75_c, cro76_n1)


def test_stretched_link_between_standard_residues_is_dropped():
    """A stretched link between two standard residues is still removed: it is more likely a chain break."""
    src = FIXTURES / "3u8v.cif"
    arr = _parse(src)
    chain = arr.chain_iid == "A_1"
    first_res, second_res = np.unique(arr.res_id[chain])[:2]
    c_atom, n_atom = ("A_1", first_res, "C"), ("A_1", second_res, "N")
    c_index = np.flatnonzero(chain & (arr.res_id == first_res) & (arr.atom_name == "C"))[0]
    n_index = np.flatnonzero(chain & (arr.res_id == second_res) & (arr.atom_name == "N"))[0]
    direction = arr.coord[n_index] - arr.coord[c_index]
    arr.coord[chain & (arr.res_id == second_res)] += 1.5 * direction / np.linalg.norm(direction)
    stretched_length = np.linalg.norm(arr.coord[n_index] - arr.coord[c_index])
    assert stretched_length > 2.4
    assert _bonded(arr, c_atom, n_atom)

    cleaned, _ = clean_structure(arr, src, "test", None)

    assert not _bonded(cleaned, c_atom, n_atom)


def test_filters_treat_symmetry_copies_separately():
    """Two superimposed symmetry copies (A_1, A_2) clash: exactly one is removed, not both."""
    src = FIXTURES / "2q2k.cif"
    arr = _parse(src)
    arr = arr[arr.res_name != "EPE"]  # aids are a separate step; keep the reference single-step
    first, second = arr.chain_iid == "A_1", arr.chain_iid == "A_2"
    arr.coord[second] = arr.coord[first]

    cleaned, removals, removed = _assert_matches_of3(arr, src, None, remove_clashing_chains, clash_distance=1.7, clash_percentage=0.3)

    # Exactly one of the two copies survives.
    actual_survivors = {"A_1", "A_2"} & set(cleaned.chain_iid)
    assert len(actual_survivors) == 1

    # OF3 removed the other copy.
    expected_removed_chains = {"A_1", "A_2"} - actual_survivors
    assert {key[0] for key in removed} == expected_removed_chains

    # The removal record lists a single chain.
    actual_removed_chains = _removal(removals, "remove_clashing_chains").chain_iids
    assert len(actual_removed_chains) == 1


def test_chain_with_no_resolved_atoms_left_is_dropped(ccd: CIFFile):
    """A ligand whose atoms all become non-CCD or unresolved is removed as a whole chain."""
    src = FIXTURES / "7riv.cif"
    arr = _parse(src)
    ligand = np.nonzero(arr.chain_iid == "B_1")[0]
    arr.coord[ligand[-2:]] = np.nan
    arr.atom_name[ligand] = [f"X{i}" for i in range(len(ligand))]

    cleaned, removals, _ = _assert_matches_of3(arr, src, ccd, remove_non_CCD_atoms, ccd=ccd)

    # The emptied ligand chain is gone.
    actual_chain_iids = set(cleaned.chain_iid)
    assert "B_1" not in actual_chain_iids

    # The removal record names the chain and counts all its atoms.
    removal = _removal(removals, "remove_non_CCD_atoms")
    assert removal.chain_iids == ("B_1",)
    assert removal.n_atoms == len(ligand)


def test_missing_chem_comp_is_a_clear_error(tmp_path: Path):
    """A source CIF without _chem_comp raises a ValueError naming the structure."""
    src = _edited_cif(FIXTURES / "3u8v.cif", tmp_path, {"_chem_comp.": "_chemcomp."})
    arr = _parse(FIXTURES / "3u8v.cif")

    with pytest.raises(ValueError, match="3u8v"):
        clean_structure(arr, src, "3u8v", None)


def test_missing_exptl_is_a_clear_error(tmp_path: Path):
    """A source CIF without _exptl.method raises a ValueError naming the structure and the field."""
    text = (FIXTURES / "3u8v.cif").read_text()
    src = tmp_path / "3u8v.cif"
    src.write_text(text.replace("_exptl.", "_exptlx."))
    arr = _parse(FIXTURES / "3u8v.cif")

    with pytest.raises(ValueError, match="3u8v.*_exptl.method"):
        clean_structure(arr, src, "3u8v", None)


def test_empty_structure_raises(tmp_path: Path):
    """A structure with nothing left after cleanup raises EmptyStructureError carrying the removals."""
    src = write_emptiable_minimal_3aa(tmp_path)
    arr = _parse(src)

    with pytest.raises(EmptyStructureError) as excinfo:
        clean_structure(arr, src, "minimal", None)

    # The error carries the structure id.
    assert excinfo.value.pdb_id == "minimal"

    # The only step that removed anything was the small-polymer filter.
    actual_steps = [removal.step for removal in excinfo.value.removals]
    assert actual_steps == ["remove_small_polymers"]


def test_waters_removed_and_logged(caplog: pytest.LogCaptureFixture):
    """Waters are removed by cleanup (OF3's remove_waters), as a logged Removal."""
    src = FIXTURES / "7riv.cif"
    arr = _parse(src)
    n_water = int(np.isin(arr.res_name, ["HOH", "DOD"]).sum())
    assert n_water > 0, "precondition: the parse keeps waters for cleanup to remove"

    with caplog.at_level(logging.INFO, logger="aw_of3_finetune.cleanup"):
        cleaned, removals = clean_structure(arr, src, "7riv", None)

    # No water residue is left.
    assert not np.isin(cleaned.res_name, ["HOH", "DOD"]).any()

    # The removal record counts every water atom.
    actual_removed_atoms = _removal(removals, "remove_waters").n_atoms
    assert actual_removed_atoms == n_water

    # The log line reports the removal and its atom count.
    assert any(
        message.startswith("7riv: remove_waters removed") and f"({n_water} atoms)" in message
        for message in caplog.messages
    )


def test_mse_converted_and_logged(caplog: pytest.LogCaptureFixture):
    """MSE becomes MET in cleanup (OF3's convert_MSE_to_MET) and the count is logged."""
    src = FIXTURES / "4zey.cif"
    arr = _parse(src)
    mse_keys = set(zip(arr.chain_iid[arr.res_name == "MSE"], arr.res_id[arr.res_name == "MSE"]))
    assert mse_keys, "precondition: the parse keeps MSE for cleanup to convert"

    with caplog.at_level(logging.INFO, logger="aw_of3_finetune.cleanup"):
        cleaned, _ = clean_structure(arr, src, "4zey", None)

    # No MSE residue is left.
    assert not (cleaned.res_name == "MSE").any()

    # The log reports how many residues were converted.
    expected_log = f"4zey: convert_MSE_to_MET converted {len(mse_keys)} residue(s)"
    assert expected_log in caplog.messages


def test_summary_line_logged(caplog: pytest.LogCaptureFixture):
    """One line per structure reports the formatting steps: unresolved and terminal atoms."""
    src = FIXTURES / "7riv.cif"
    arr = _parse(src)

    with caplog.at_level(logging.INFO, logger="aw_of3_finetune.cleanup"):
        cleaned, _ = clean_structure(arr, src, "7riv", None)

    n_nan = int(np.isnan(cleaned.coord).any(axis=1).sum())
    (summary,) = [message for message in caplog.messages if message.startswith("7riv: cleanup done")]

    # The summary reports the final atom count.
    assert f"{len(cleaned)} atoms" in summary

    # The summary reports the unresolved-atom count.
    assert f"{n_nan} unresolved (NaN coordinates)" in summary

    # The summary mentions the terminal-atom removal.
    assert "terminal atom(s) removed" in summary
