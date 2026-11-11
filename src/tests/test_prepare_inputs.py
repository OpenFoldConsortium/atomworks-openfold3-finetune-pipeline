"""Tests for prepare_structure: one in-house structure to a pipeline-ready mmCIF."""

import io
import logging
from datetime import date
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.component import mol_from_atomarray
from rdkit import Chem

from aw_of3_finetune.ligand_inputs import LigandMatchError, LigandNamer, LigandRow, ccd_codes
from aw_of3_finetune.parsing import parse_structure
from aw_of3_finetune.pipeline import _read_release_date, _read_resolution
from aw_of3_finetune.prepare_inputs import (
    DEFAULT_RELEASE_DATE,
    _construct_positions,
    _rename_in_definition_order,
    prepare_structure,
)
from aw_of3_finetune.validate import run_validation
from ligand_helpers import (
    ACID_ATOM_NAMES,
    ACID_SMILES,
    INDOLE_SMILES,
    PXR_CONSTRUCT,
    PXR_CONSTRUCTS,
    X00242_DEPOSITED_H_PDB,
    X00242_PDB,
    X00242_RESTRAINT,
    X00358_ALTLOC_PDB,
    X00358_PDB,
    X00358_RESTRAINT,
    definition_smiles_from_block,
    reference_smiles_by_chain,
)
from validate_helpers import build_dataset, context

ACID_ROW = LigandRow("x00242-1", "LIG", ACID_SMILES, X00242_RESTRAINT)
INDOLE_ROW = LigandRow("x00358-1", "LIG", INDOLE_SMILES, X00358_RESTRAINT)


@pytest.fixture(scope="module")
def ccd(ccd_path: Path) -> CIFFile:
    """The parsed CCD, read once per module."""
    return CIFFile.read(ccd_path)


@pytest.fixture(scope="module")
def codes(ccd: CIFFile) -> frozenset[str]:
    """CCD component ids, computed once per module."""
    return ccd_codes(ccd)


@pytest.fixture
def namer(codes: frozenset[str]) -> LigandNamer:
    """A fresh namer per test."""
    return LigandNamer(codes)


def _block(cif_text: str) -> pdbx.CIFBlock:
    return CIFFile.read(io.StringIO(cif_text)).block


def _parse_written(cif_text: str, tmp_path: Path, name: str) -> struc.AtomArray:
    """Write the CIF and parse it the way finetune-prep does (asym unit, model 1)."""
    out = tmp_path / f"{name}.cif"
    out.write_text(cif_text)
    return parse_structure(out)["asym_unit"][0]


def _residue_atom_names(atoms: struc.AtomArray, res_name: str) -> dict[int, list[str]]:
    """{res_id: atom names in order} for every residue called res_name."""
    ligand = atoms[atoms.res_name == res_name]
    return {
        int(residue.res_id[0]): [str(name) for name in residue.atom_name]
        for residue in struc.residue_iter(ligand)
    }


def _cif_input_with_release_date(tmp_path: Path, revision_date: str) -> Path:
    """x00242-1 as an mmCIF input that carries pdbx_audit_revision_history."""
    atoms = PDBFile.read(X00242_PDB).get_structure(model=1)
    cif = CIFFile()
    pdbx.set_structure(cif, atoms, data_block="x00242-1")
    cif.block["pdbx_audit_revision_history"] = pdbx.CIFCategory(
        {"ordinal": ["1"], "revision_date": [revision_date]}
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "x00242-1.cif"
    cif.write(path)
    return path


def test_carboxylic_acid_pdb_round_trips_through_finetune_prep_parse(tmp_path, namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual_atoms = _parse_written(prepared.cif_text, tmp_path, "x00242-1")
    code = prepared.ligand_codes["LIG"]
    actual_ligand = actual_atoms[actual_atoms.res_name == code]
    expected_names = ACID_ATOM_NAMES
    assert list(actual_ligand.atom_name) == expected_names
    assert "LIG" not in set(actual_atoms.res_name)


def test_written_cif_defines_the_ligand_inline(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    block = _block(prepared.cif_text)
    code = prepared.ligand_codes["LIG"]
    actual_defined = set(block["chem_comp_atom"]["comp_id"].as_array())
    expected_defined = {code}
    assert actual_defined == expected_defined
    actual_smiles = definition_smiles_from_block(block, code)
    expected_smiles = Chem.CanonSmiles(ACID_SMILES)
    assert actual_smiles == expected_smiles


def test_written_cif_types_the_ligand_as_non_polymer(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    chem_comp = _block(prepared.cif_text)["chem_comp"]
    types = dict(zip(chem_comp["id"].as_array(), chem_comp["type"].as_array(), strict=True))
    actual_type = types[prepared.ligand_codes["LIG"]]
    assert actual_type == "NON-POLYMER"


def test_of3_builds_the_indole_reference_molecule_from_the_written_bonds(tmp_path, namer, ccd):
    # finetune-prep builds a non-CCD ligand's reference molecule with OF3's mol_from_atomarray,
    # which turns aromatic-flagged bonds into RDKit AROMATIC and cannot kekulize an [nH] ring.
    prepared = prepare_structure(X00358_PDB, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)
    out = tmp_path / "x00358-1.cif"
    out.write_text(prepared.cif_text)

    atoms = parse_mmcif(out).atom_array
    is_ligand = atoms.res_name == prepared.ligand_codes["LIG"]
    first_copy = atoms[is_ligand & (atoms.chain_id == atoms.chain_id[is_ligand][0])]
    actual_smiles = Chem.MolToSmiles(mol_from_atomarray(first_copy))
    expected_smiles = Chem.CanonSmiles(INDOLE_SMILES)
    assert actual_smiles == expected_smiles


def test_ring_bonds_are_written_as_kekule_orders(namer, ccd):
    prepared = prepare_structure(X00358_PDB, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    bonds = _block(prepared.cif_text)["chem_comp_bond"]
    actual_orders = list(bonds["value_order"].as_array())
    expected_double = 5  # indole's Kekulé structure (3 benzene + 1 pyrrole) and the C=O
    assert actual_orders.count("DOUB") == expected_double
    actual_flags = set(bonds["pdbx_aromatic_flag"].as_array())
    assert actual_flags == {"N"}


def test_release_date_resolution_and_method_are_written(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    block = _block(prepared.cif_text)
    assert _read_release_date(block) == date(2026, 1, 1)
    assert prepared.release_date_defaulted is True
    actual_resolution = _read_resolution(block)
    expected_resolution = 1.96  # REMARK 3 RESOLUTION RANGE HIGH
    assert actual_resolution == expected_resolution
    actual_method = block["exptl"]["method"].as_item()
    assert actual_method == "X-RAY DIFFRACTION"


def test_input_release_date_is_kept(tmp_path, namer, ccd):
    cif_input = _cif_input_with_release_date(tmp_path / "input", "2021-05-04")

    prepared = prepare_structure(cif_input, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual_date = _read_release_date(_block(prepared.cif_text))
    expected_date = date(2021, 5, 4)
    assert actual_date == expected_date
    assert prepared.release_date_defaulted is False


def test_written_cif_has_entity_poly_and_struct_asym(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    block = _block(prepared.cif_text)
    actual_poly_types = list(block["entity_poly"]["type"].as_array())
    expected_poly_types = ["polypeptide(L)"]
    assert actual_poly_types == expected_poly_types
    actual_asym_ids = set(block["struct_asym"]["id"].as_array())
    expected_asym_ids = set(block["atom_site"]["label_asym_id"].as_array())
    assert actual_asym_ids == expected_asym_ids


def test_two_copies_in_one_chain_share_one_code(tmp_path, namer, ccd):
    prepared = prepare_structure(X00358_PDB, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual_atoms = _parse_written(prepared.cif_text, tmp_path, "x00358-1")
    code = prepared.ligand_codes["LIG"]
    actual_copies = _residue_atom_names(actual_atoms, code)
    expected_copy_count = 2  # res 701 and 801
    assert len(actual_copies) == expected_copy_count
    assert "LIG" not in set(actual_atoms.res_name)


def test_altloc_ligand_atoms_keep_their_names(tmp_path, codes, ccd):
    plain = prepare_structure(
        X00358_PDB, [INDOLE_ROW], LigandNamer(codes), ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS
    )
    altloc = prepare_structure(
        X00358_ALTLOC_PDB, [INDOLE_ROW], LigandNamer(codes), ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS
    )

    expected_code = plain.ligand_codes["LIG"]
    actual_code = altloc.ligand_codes["LIG"]
    assert actual_code == expected_code
    plain_atoms = _parse_written(plain.cif_text, tmp_path, "plain")
    altloc_atoms = _parse_written(altloc.cif_text, tmp_path, "altloc")
    expected_names = _residue_atom_names(plain_atoms, expected_code)
    actual_names = _residue_atom_names(altloc_atoms, actual_code)
    assert actual_names == expected_names


def test_deposited_ligand_hydrogens_give_the_heavy_atom_definition(codes, ccd):
    heavy = prepare_structure(X00242_PDB, [ACID_ROW], LigandNamer(codes), ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)
    with_h = prepare_structure(
        X00242_DEPOSITED_H_PDB, [ACID_ROW], LigandNamer(codes), ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS
    )

    expected_code = heavy.ligand_codes["LIG"]
    actual_code = with_h.ligand_codes["LIG"]
    assert actual_code == expected_code
    expected_smiles = definition_smiles_from_block(_block(heavy.cif_text), expected_code)
    actual_smiles = definition_smiles_from_block(_block(with_h.cif_text), actual_code)
    assert actual_smiles == expected_smiles


def test_namer_records_the_structure(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    code = prepared.ligand_codes["LIG"]
    actual_structures = namer.records()[code]["structures"]
    expected_structures = ["x00242-1"]
    assert actual_structures == expected_structures


def test_unlisted_placeholder_raises(namer, ccd):
    with pytest.raises(ValueError, match="LIG"):
        prepare_structure(X00242_PDB, [], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)


def test_row_for_absent_residue_raises(namer, ccd):
    absent_row = LigandRow("x00242-1", "XYZ", ACID_SMILES, None)

    with pytest.raises(ValueError, match="XYZ"):
        prepare_structure(X00242_PDB, [ACID_ROW, absent_row], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)


def test_smiles_that_does_not_match_raises(namer, ccd):
    wrong_row = LigandRow("x00242-1", "LIG", INDOLE_SMILES, None)

    with pytest.raises(LigandMatchError):
        prepare_structure(X00242_PDB, [wrong_row], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)


def test_copies_that_differ_raise(tmp_path, namer, ccd):
    # Rename one atom of the second copy (res 801) so its definition differs from res 701's.
    lines = X00358_PDB.read_text().splitlines()
    edited = [
        line.replace(" C10 LIG A 801", " C99 LIG A 801") for line in lines
    ]
    path = tmp_path / "x00358-1.pdb"
    path.write_text("\n".join(edited) + "\n")
    row = LigandRow("x00358-1", "LIG", INDOLE_SMILES, None)

    with pytest.raises(ValueError, match="copies"):
        prepare_structure(path, [row], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)


def _label_asym_ids_by_res_id(cif_text: str, res_name: str) -> dict[int, str]:
    """{auth_seq_id: label_asym_id} of the residues called res_name in a written CIF."""
    atom_site = _block(cif_text)["atom_site"]
    is_residue = atom_site["label_comp_id"].as_array() == res_name
    res_ids = atom_site["auth_seq_id"].as_array(int)[is_residue]
    asym_ids = atom_site["label_asym_id"].as_array()[is_residue]
    return {int(res_id): str(asym_id) for res_id, asym_id in zip(res_ids, asym_ids, strict=True)}


def test_each_ligand_copy_is_its_own_chain(namer, ccd):
    # finetune-prep builds a single-residue ligand's reference molecule from its own chain;
    # two copies in one chain would be read as one two-residue ligand.
    prepared = prepare_structure(X00358_PDB, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    asym_ids = _label_asym_ids_by_res_id(prepared.cif_text, prepared.ligand_codes["LIG"])
    actual_chain_count = len(set(asym_ids.values()))
    expected_chain_count = 2
    assert actual_chain_count == expected_chain_count


def test_ligand_copies_out_of_residue_number_order_are_read(tmp_path, namer, ccd):
    # Refinement outputs list hetero residues in any order (x00242-1's raw file: DMS 801, LIG 701).
    lines = X00358_PDB.read_text().splitlines()
    copy_801 = [line for line in lines if line.startswith("HETATM") and line[22:26] == " 801"]
    others = [line for line in lines if line not in copy_801]
    first_hetatm = next(i for i, line in enumerate(others) if line.startswith("HETATM"))
    reordered = others[:first_hetatm] + copy_801 + others[first_hetatm:]
    path = tmp_path / "x00358-1.pdb"
    path.write_text("\n".join(reordered) + "\n")

    prepared = prepare_structure(path, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    code = prepared.ligand_codes["LIG"]
    actual_res_ids = set(_label_asym_ids_by_res_id(prepared.cif_text, code))
    expected_res_ids = {701, 801}
    assert actual_res_ids == expected_res_ids


def test_structure_without_ligands_is_written(tmp_path, namer, ccd):
    protein_only = [
        line for line in X00242_PDB.read_text().splitlines() if not line.startswith("HETATM")
    ]
    path = tmp_path / "protein_only.pdb"
    path.write_text("\n".join(protein_only) + "\n")

    prepared = prepare_structure(path, [], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    assert prepared.ligand_codes == {}
    assert "chem_comp_atom" not in _block(prepared.cif_text)


def test_ligand_copies_keep_the_author_chain(namer, ccd):
    prepared = prepare_structure(X00358_PDB, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    atom_site = _block(prepared.cif_text)["atom_site"]
    is_ligand = atom_site["label_comp_id"].as_array() == prepared.ligand_codes["LIG"]
    actual_auth_chains = set(atom_site["auth_asym_id"].as_array()[is_ligand])
    expected_auth_chains = {"A"}  # both copies are LIG A 701 / A 801 in the input
    assert actual_auth_chains == expected_auth_chains


def _reference_smiles_by_chain(
    prepared, tmp_path: Path, ccd: CIFFile
) -> dict[str, tuple[str, str]]:
    source = tmp_path / "source.cif"
    source.write_text(prepared.cif_text)
    return reference_smiles_by_chain(source, set(prepared.ligand_codes.values()), tmp_path, ccd)


def test_finetune_prep_builds_the_acid_reference_molecule(tmp_path, namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual = _reference_smiles_by_chain(prepared, tmp_path, ccd)

    expected = {"2": (prepared.ligand_codes["LIG"], Chem.CanonSmiles(ACID_SMILES))}
    assert actual == expected


def test_finetune_prep_builds_one_indole_reference_molecule_for_both_copies(
    tmp_path, namer, ccd
):
    prepared = prepare_structure(X00358_PDB, [INDOLE_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual = _reference_smiles_by_chain(prepared, tmp_path, ccd)

    expected_entry = (prepared.ligand_codes["LIG"], Chem.CanonSmiles(INDOLE_SMILES))
    expected = {"2": expected_entry, "3": expected_entry}
    assert actual == expected


def test_parsed_ligand_atom_missing_from_the_definition_is_named():
    atoms = struc.AtomArray(2)
    atoms.chain_id[:] = "B"
    atoms.res_id[:] = 701
    atoms.res_name[:] = "LIG"
    atoms.atom_name = ["C1", "C99"]

    with pytest.raises(ValueError, match="LIG B 701.*C99"):
        _rename_in_definition_order(atoms, "LIG", "L0000", ["C1", "C2"])


def test_loader_chain_split_warning_is_not_logged(caplog, namer, ccd):
    with caplog.at_level(logging.WARNING):
        prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    split_messages = [r for r in caplog.records if "non-polymer residues" in r.getMessage()]
    assert split_messages == []


def _polymer_rows(block: pdbx.CIFBlock) -> dict[str, np.ndarray]:
    """label_seq_id, auth_seq_id and label_comp_id per residue of the polymer's atom_site rows."""
    atom_site = block["atom_site"]
    is_polymer = atom_site["group_PDB"].as_array(str) == "ATOM"
    label_ids = atom_site["label_seq_id"].as_array(int, -1)[is_polymer]
    auth_ids = atom_site["auth_seq_id"].as_array(int)[is_polymer]
    names = atom_site["label_comp_id"].as_array(str)[is_polymer]
    starts = np.flatnonzero(np.r_[True, auth_ids[1:] != auth_ids[:-1]])
    return {"label": label_ids[starts], "auth": auth_ids[starts], "name": names[starts]}


def _author_residues(path: Path) -> list[tuple[int, str]]:
    """(author number, name) of each polymer residue in a .pdb, in file order."""
    atoms = PDBFile.read(path).get_structure(model=1)
    atoms = atoms[~atoms.hetero]
    starts = struc.get_residue_starts(atoms)
    return list(zip(atoms.res_id[starts].tolist(), atoms.res_name[starts].tolist(), strict=True))


AUTHOR_TO_CONSTRUCT_OFFSET = 141  # PXR residue 142 is construct position 1


def test_polymer_residues_are_numbered_by_construct_position(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    actual_label_ids = _polymer_rows(_block(prepared.cif_text))["label"].tolist()

    author_numbers = [number for number, _ in _author_residues(X00242_PDB)]
    expected_label_ids = [number - AUTHOR_TO_CONSTRUCT_OFFSET for number in author_numbers]
    assert actual_label_ids == expected_label_ids


def test_entity_poly_holds_the_whole_construct(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)
    entity_poly = _block(prepared.cif_text)["entity_poly"]

    actual_sequence = entity_poly["pdbx_seq_one_letter_code_can"].as_array(str)[0].replace("\n", "")

    assert actual_sequence == PXR_CONSTRUCT


def test_gaps_without_a_construct_raise(namer, ccd):
    with pytest.raises(ValueError, match="construct"):
        prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE)


def test_gaps_with_a_construct_that_does_not_fit_raise(namer, ccd):
    with pytest.raises(ValueError, match="construct"):
        prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, ("MKVLAAGG",))


def test_chain_without_gaps_is_numbered_from_one_without_a_construct(tmp_path, namer, ccd):
    atoms = PDBFile.read(X00242_PDB).get_structure(model=1)
    keep = (atoms.res_id >= 239) & (atoms.res_id <= 248) | (atoms.res_name == "LIG")
    pdb = PDBFile()
    pdb.set_structure(atoms[keep])
    path = tmp_path / "x00242-1.pdb"
    pdb.write(path)

    prepared = prepare_structure(path, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE)

    actual_label_ids = _polymer_rows(_block(prepared.cif_text))["label"].tolist()
    expected_label_ids = list(range(1, 11))
    assert actual_label_ids == expected_label_ids


def test_segments_keep_their_order_across_a_numbering_jump():
    construct = "AAAACDEFGGGGGGKLMNP"
    letters = "CDEF" + "KLMN"
    author_numbers = [10, 11, 12, 13, 682, 683, 684, 685]

    actual = _construct_positions(letters, author_numbers, construct)

    expected = [4, 5, 6, 7, 14, 15, 16, 17]
    assert actual == expected


def test_short_segment_is_placed_by_its_author_number():
    construct = "AKAAAAKAAAACDE"
    letters = "K" + "CDE"
    author_numbers = [7, 12, 13, 14]

    actual = _construct_positions(letters, author_numbers, construct)

    expected = [6, 11, 12, 13]
    assert actual == expected


def test_residues_that_do_not_fit_the_construct_have_no_positions():
    actual = _construct_positions("CDEF" + "KLMN", [10, 11, 12, 13, 20, 21, 22, 23], "KLMNCDEF")

    assert actual is None


def test_author_residue_numbers_stay_in_auth_seq_id(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    rows = _polymer_rows(_block(prepared.cif_text))
    actual = list(zip(rows["auth"].tolist(), rows["name"].tolist(), strict=True))

    expected = _author_residues(X00242_PDB)
    assert actual == expected


def test_entity_poly_seq_numbers_the_whole_construct(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)
    block = _block(prepared.cif_text)

    actual_numbers = block["entity_poly_seq"]["num"].as_array(int).tolist()

    expected_numbers = list(range(1, len(PXR_CONSTRUCT) + 1))
    assert actual_numbers == expected_numbers


def test_cif_input_with_author_numbers_is_numbered_by_construct_position(tmp_path, namer, ccd):
    path = _cif_input_with_release_date(tmp_path / "input", "2020-02-03")

    prepared = prepare_structure(path, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    rows = _polymer_rows(_block(prepared.cif_text))
    author_numbers = [number for number, _ in _author_residues(X00242_PDB)]
    expected_label_ids = [number - AUTHOR_TO_CONSTRUCT_OFFSET for number in author_numbers]
    assert rows["label"].tolist() == expected_label_ids
    assert rows["auth"].tolist() == author_numbers


def test_no_atom_is_added_to_the_input(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    atom_site = _block(prepared.cif_text)["atom_site"]
    actual_polymer_atoms = int((atom_site["group_PDB"].as_array(str) == "ATOM").sum())

    input_atoms = PDBFile.read(X00242_PDB).get_structure(model=1)
    input_polymer_heavy_atoms = input_atoms[~input_atoms.hetero & (input_atoms.element != "H")]
    expected_polymer_atoms = input_polymer_heavy_atoms.array_length()
    assert actual_polymer_atoms == expected_polymer_atoms


def test_every_written_atom_has_coordinates(namer, ccd):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)

    atom_site = _block(prepared.cif_text)["atom_site"]

    n_unresolved = 0
    for axis in ("Cartn_x", "Cartn_y", "Cartn_z"):
        n_unresolved += int(np.isnan(atom_site[axis].as_array(float)).sum())
    assert n_unresolved == 0


@pytest.mark.integration
def test_prepared_pdb_loads_through_of3(tmp_path, namer, ccd, ccd_path):
    prepared = prepare_structure(X00242_PDB, [ACID_ROW], namer, ccd, DEFAULT_RELEASE_DATE, PXR_CONSTRUCTS)
    source = tmp_path / "x00242-1.cif"
    source.write_text(prepared.cif_text)

    built = build_dataset(tmp_path / "built", [source], ccd_path)
    report = run_validation(context(built), ["V3", "V7", "V21"])

    actual_errors = [f"{f.check}: {f.message}" for f in report.errors]
    assert actual_errors == []
    final = CIFFile.read(built.output_dir / "cleaned_structures" / "x00242-1.cif").block
    actual_sequence = final["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array(str)[0].replace("\n", "")
    assert actual_sequence == PXR_CONSTRUCT
