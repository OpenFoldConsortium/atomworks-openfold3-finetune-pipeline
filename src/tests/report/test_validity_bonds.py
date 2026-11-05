import json
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import Atom, BondList, array
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz

from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.origin import classify_origins
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.report.validity.bonds import check_bond_lengths, check_residue_bonds

PDB_ID = "tiny"
SINGLE, DOUBLE = 1, 2
TINY_CCD = """data_XXX
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.pdbx_leaving_atom_flag
XXX A N
XXX B N
XXX C Y
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
XXX A B SING N
XXX B C DOUB N
data_YYY
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.pdbx_leaving_atom_flag
YYY A N
YYY B N
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
YYY A B DELO N
"""


def write_tiny_run(tmp_path, res_name, names, xs, bonds, chains=None):
    """A one-residue run: atoms at the given x (None = unresolved) joined by (i, j, type) bonds.

    chains: chain id per atom (default all "A"); each chain is its own residue.
    """
    chains = chains or ["A"] * len(names)
    atoms = [
        Atom([np.nan if x is None else x, 0.0, 0.0], chain_id=chain, res_id=1, res_name=res_name,
             atom_name=name, element="C", label_asym_id=chain)
        for name, x, chain in zip(names, xs, chains)
    ]
    structure = array(atoms)
    structure.bonds = BondList(len(atoms), np.array(bonds, dtype=int).reshape(-1, 3))
    out_dir = tmp_path / "out"
    (out_dir / "structure_files" / PDB_ID).mkdir(parents=True)
    (out_dir / "manifest.json").write_text(json.dumps({"structures": [PDB_ID]}))
    write_atomarray_to_npz(structure, out_dir / "structure_files" / PDB_ID / f"{PDB_ID}.npz")
    (tmp_path / "ccd.cif").write_text(TINY_CCD)
    return ValidityInputs(out_dir, tmp_path / "cifs", tmp_path / "ccd.cif")


def test_standard_residue_matches_ccd(built_7riv, ccd_path):
    inputs = ValidityInputs(built_7riv.output_dir, built_7riv.cif_dir, ccd_path)

    actual = check_residue_bonds(inputs, "7riv")

    expected_status = Status.MATCH
    assert [f.status for f in actual] == [expected_status], actual[0].detail


def test_flipped_bond_order_is_reported(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, 3.0], [(0, 1, DOUBLE), (1, 2, DOUBLE)])

    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.status == Status.UNEXPLAINED
    assert actual.detail == "XXX: order A-B 2 vs 1"
    expected_row = {"chain": "A", "res_name": "XXX", "res_id": 1, "issue": "ccd_bonds", "extra": [], "missing": [],
                    "order_differs": ["A-B"]}
    assert actual.data["rows"] == [expected_row]


def test_extra_and_missing_bonds_are_reported(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, 3.0], [(0, 2, SINGLE), (1, 2, DOUBLE)])

    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.detail == "XXX: extra bond A-C; XXX: missing bond A-B"


def test_residue_in_neither_ccd_nor_input_is_reported_not_raised(tmp_path):
    inputs = write_tiny_run(tmp_path, "QQQ", ["A", "B"], [0.0, 1.5], [(0, 1, SINGLE)])

    _input_cif_without_bonds(tmp_path)
    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.status == Status.UNEXPLAINED
    assert actual.detail == "QQQ: not in CCD"


def test_leaving_atom_absence_is_not_a_missing_bond(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B"], [0.0, 1.5], [(0, 1, SINGLE)])

    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.status == Status.MATCH


def test_overlong_bond_is_reported(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, 4.5], [(0, 1, SINGLE), (1, 2, DOUBLE)])

    actual = check_bond_lengths(inputs, PDB_ID)[0]

    assert actual.status == Status.UNEXPLAINED
    assert actual.detail == "A:XXX1:B-A:XXX1:C 3.000 Å, too long (max 2.4 Å)"
    expected_rows = [{"chain": "A", "label_asym_id": "A", "res_name": "XXX", "res_id": 1, "atom_1": "B",
                      "atom_2": "C", "distance": 3.0}]
    assert actual.data["rows"] == expected_rows


def test_bond_to_unresolved_atom_is_not_measured(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, None], [(0, 1, SINGLE), (1, 2, DOUBLE)])

    actual = check_bond_lengths(inputs, PDB_ID)[0]

    assert actual.status == Status.MATCH


def test_each_offending_bond_is_its_own_row(tmp_path):
    names, xs = ["A", "B", "C"], [0.0, 3.0, 6.0]
    inputs = write_tiny_run(tmp_path, "XXX", names, xs, [(0, 1, SINGLE), (1, 2, DOUBLE)])

    actual = check_bond_lengths(inputs, PDB_ID)[0]

    expected_pairs = [("A", "B"), ("B", "C")]
    actual_pairs = [(row["atom_1"], row["atom_2"]) for row in actual.data["rows"]]
    assert actual_pairs == expected_pairs


def test_residue_in_neither_ccd_nor_input_is_one_row_and_one_part_per_name(tmp_path):
    inputs = write_tiny_run(tmp_path, "QQQ", ["A", "B", "A", "B"], [0.0, 1.5, 10.0, 11.5],
                            [(0, 1, SINGLE), (2, 3, SINGLE)], chains=["A", "A", "B", "B"])
    _input_cif_without_bonds(tmp_path)
    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.detail == "QQQ: not in CCD"
    expected_rows = [{"res_name": "QQQ", "chains": ["A", "B"], "issue": "not_in_ccd"}]
    assert actual.data["rows"] == expected_rows


def test_unknown_ccd_bond_order_is_any(tmp_path):
    inputs = write_tiny_run(tmp_path, "YYY", ["A", "B"], [0.0, 1.5], [(0, 1, DOUBLE)])

    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.status == Status.MATCH, actual.detail


def test_short_bond_says_too_short(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B"], [0.0, 0.5], [(0, 1, SINGLE)])

    actual = check_bond_lengths(inputs, PDB_ID)[0]

    assert actual.detail == "A:XXX1:A-A:XXX1:B 0.500 Å, too short (min 0.9 Å)"


def _input_cif_without_bonds(tmp_path):
    """cifs/tiny.cif with no chem_comp_bond."""
    (tmp_path / "cifs").mkdir(exist_ok=True)
    (tmp_path / "cifs" / f"{PDB_ID}.cif").write_text("data_tiny\n_entry.id tiny\n")


def _input_cif_with_bonds(tmp_path, rows):
    """cifs/tiny.cif whose chem_comp_bond has the given (comp, atom_1, atom_2, order, aromatic) rows."""
    (tmp_path / "cifs").mkdir(exist_ok=True)
    header = ["data_tiny", "loop_"] + [f"_chem_comp_bond.{c}" for c in
                                       ("comp_id", "atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag")]
    (tmp_path / "cifs" / f"{PDB_ID}.cif").write_text("\n".join(header + [" ".join(row) for row in rows]) + "\n")


def test_residue_defined_only_by_the_input_cif_matches_when_bonds_agree(tmp_path):
    """prepare-inputs: a custom ligand is not in the CCD; its definition is in the input CIF."""
    inputs = write_tiny_run(tmp_path, "QQQ", ["A", "B"], [0.0, 1.5], [(0, 1, SINGLE)])
    _input_cif_with_bonds(tmp_path, [("QQQ", "A", "B", "SING", "N")])

    actual = check_residue_bonds(inputs, PDB_ID)[0]

    assert actual.status == Status.MATCH


def test_residue_defined_only_by_the_input_cif_differs_from_it_is_processing(tmp_path):
    inputs = write_tiny_run(tmp_path, "QQQ", ["A", "B"], [0.0, 1.5], [(0, 1, SINGLE)])
    _input_cif_with_bonds(tmp_path, [("QQQ", "A", "B", "DOUB", "N")])

    [actual] = classify_origins(inputs, check_residue_bonds(inputs, PDB_ID))

    assert actual.detail == "QQQ: order A-B 1 vs 2"
    assert actual.data["origin"] == "processing"


def test_order_that_the_input_cif_defines_is_input(tmp_path):
    """OpenBind: the input CIF's own ligand definition differs from the supplied CCD; the structure follows the input."""
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, 3.0], [(0, 1, DOUBLE), (1, 2, DOUBLE)])
    _input_cif_with_bonds(tmp_path, [("XXX", "A", "B", "DOUB", "N"), ("XXX", "B", "C", "DOUB", "N")])

    [actual] = classify_origins(inputs, check_residue_bonds(inputs, PDB_ID))

    assert actual.data["origin"] == "input"
    assert actual.status == Status.EXPECTED


def test_order_without_an_input_definition_is_processing(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, 3.0], [(0, 1, DOUBLE), (1, 2, DOUBLE)])
    _input_cif_without_bonds(tmp_path)

    [actual] = classify_origins(inputs, check_residue_bonds(inputs, PDB_ID))

    assert actual.data["origin"] == "processing"


def test_order_that_differs_from_the_input_definition_is_processing(tmp_path):
    inputs = write_tiny_run(tmp_path, "XXX", ["A", "B", "C"], [0.0, 1.5, 3.0], [(0, 1, DOUBLE), (1, 2, DOUBLE)])
    _input_cif_with_bonds(tmp_path, [("XXX", "A", "B", "SING", "N"), ("XXX", "B", "C", "DOUB", "N")])

    [actual] = classify_origins(inputs, check_residue_bonds(inputs, PDB_ID))

    assert actual.data["origin"] == "processing"
