import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import Atom, BondList, array
from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz, write_atomarray_to_npz

from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.validity import VALIDITY_CHECKS, ValidityInputs
from aw_of3_finetune.steplog import StepLog, StepRecord
from validate_helpers import build_dataset

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "src/tests/fixtures/mmcifs"
CHECK = "validity.covalent_links"
PDB_ID = "tiny"

ATOM_SITE_HEADER = """data_tiny
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""
STRUCT_CONN_HEADER = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_ptnr1_label_alt_id
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_ptnr2_label_alt_id
_struct_conn.ptnr2_symmetry
"""
# Input atoms: (asym, comp, seq_id, atom, element, x). Two cysteines of chain A 4 residues apart, a water.
DISULFIDE_ATOMS = [
    ("A", "CYS", "1", "SG", "S", 0.0),
    ("A", "CYS", "5", "SG", "S", 2.0),
    ("A", "ALA", "6", "CA", "C", 10.0),
    ("B", "HOH", ".", "O", "O", 20.0),
]
DISULFIDE = "disulf1 disulf A CYS 1 SG ? 1_555 A CYS 5 SG ? 1_555"


def write_run(tmp_path, atoms, conns, kept, bonds=(), molecule_types=None, removed=None):
    """Write a tiny run: input CIF with struct_conn rows, final NPZ of the kept input atoms, step log, cache.

    kept: indices of input atoms in the final structure; bonds: pairs of input indices bonded in the final.
    removed: {input index: step name} written into the step log.
    """
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    rows = [f"{'HETATM' if comp == 'HOH' else 'ATOM'} {i + 1} {element} {name} . {comp} {asym} 1 {seq} ? {x} 0.0 0.0 "
            f"1.0 10.0 {i + 1} {asym} 1" for i, (asym, comp, seq, name, element, x) in enumerate(atoms)]
    text = ATOM_SITE_HEADER + "\n".join(rows) + "\n"
    if conns:
        text += STRUCT_CONN_HEADER + "\n".join(conns) + "\n"
    (cif_dir / f"{PDB_ID}.cif").write_text(text)

    out_dir = tmp_path / "out"
    (out_dir / "structure_files" / PDB_ID).mkdir(parents=True)
    (out_dir / "cleanup_steps").mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({"structures": [PDB_ID]}))
    final = [Atom([atoms[i][5], 0.0, 0.0], chain_id=atoms[i][0], res_id=int(atoms[i][2]) if atoms[i][2] != "." else 1,
                  res_name=atoms[i][1], atom_name=atoms[i][3], element=atoms[i][4], label_asym_id=atoms[i][0])
             for i in kept]
    final_array = array(final)
    position = {index: p for p, index in enumerate(kept)}
    final_array.bonds = BondList(len(final), np.array([[position[a], position[b], 1] for a, b in bonds], dtype=int).reshape(-1, 3))
    write_atomarray_to_npz(final_array, out_dir / "structure_files" / PDB_ID / f"{PDB_ID}.npz")
    chains = {c: {"molecule_type": t} for c, t in (molecule_types or {"A": "PROTEIN", "B": "LIGAND"}).items()}
    (out_dir / "dataset_cache.json").write_text(json.dumps({"structure_data": {PDB_ID: {"chains": chains}}}))
    removed = removed or {}
    steps = [StepRecord(name="input", phase="parse", description="", resolved_atoms=len(atoms))]
    for index, name in removed.items():
        steps.append(StepRecord(name=name, phase="cleanup", description="", resolved_atoms=len(atoms) - 1,
                                removed_input=[(index, index)]))
    log = StepLog(pdb_id=PDB_ID, input_atoms=len(atoms), steps=steps)
    (out_dir / "cleanup_steps" / f"{PDB_ID}.json").write_text(log.to_json())
    return ValidityInputs(out_dir, cif_dir, tmp_path / "ccd.cif")


def run_check(inputs):
    """The single finding of validity.covalent_links."""
    [finding] = VALIDITY_CHECKS[CHECK](inputs, PDB_ID)
    return finding


@pytest.fixture(scope="module")
def built_7axf(tmp_path_factory, ccd_path):
    """Real finetune-prep output of the 7axf fixture, with NPZ."""
    return build_dataset(tmp_path_factory.mktemp("links7axf"), [FIXTURES / "7axf.cif"], ccd_path)


def test_7axf_covalent_drug_link_is_present(built_7axf, ccd_path):
    inputs = ValidityInputs(built_7axf.output_dir, built_7axf.cif_dir, ccd_path)

    finding = VALIDITY_CHECKS[CHECK](inputs, "7axf")[0]

    rows = finding.data["rows"]
    actual_states = [(row["conn_id"], row["conn_type"], row["copy"], row["state"], row["label"]) for row in rows]
    expected_states = [("covale1", "covale", "A", "present", "same")]
    assert actual_states == expected_states
    actual_atoms = (rows[0]["atom_1"].split(":", 1)[1], rows[0]["atom_2"].split(":")[0], rows[0]["atom_2"].split(":")[-1])
    expected_atoms = ("CYS89:SG", "C", "C6")
    assert actual_atoms == expected_atoms
    expected_distance = 1.75
    assert rows[0]["input_distance"] == expected_distance
    assert finding.status == Status.MATCH
    expected_links = 1
    assert finding.data["links"] == expected_links


def test_dropped_bond_is_missing(built_7axf, tmp_path, ccd_path):
    copied = tmp_path / "copy"
    shutil.copytree(built_7axf.output_dir, copied)
    npz = copied / "structure_files" / "7axf" / "7axf.npz"
    atoms = read_atomarray_from_npz(npz)
    sg = np.flatnonzero((atoms.label_asym_id == "A") & (atoms.res_id == 89) & (atoms.atom_name == "SG"))[0]
    c6 = np.flatnonzero((atoms.label_asym_id == "C") & (atoms.atom_name == "C6"))[0]
    bonds = atoms.bonds.as_array()
    keep = ~(((bonds[:, 0] == sg) & (bonds[:, 1] == c6)) | ((bonds[:, 0] == c6) & (bonds[:, 1] == sg)))
    assert not keep.all()
    atoms.bonds = BondList(len(atoms), bonds[keep])
    write_atomarray_to_npz(atoms, npz)
    inputs = ValidityInputs(copied, built_7axf.cif_dir, ccd_path)

    finding = VALIDITY_CHECKS[CHECK](inputs, "7axf")[0]

    [row] = finding.data["rows"]
    assert (row["state"], row["label"]) == ("missing", "unexplained")
    assert finding.status == Status.UNEXPLAINED
    expected_detail = f"{row['atom_1']}—{row['atom_2']}: missing"
    assert finding.detail == expected_detail
    expected_prefix = "A:CYS89:SG—C:S6T"
    assert finding.detail.startswith(expected_prefix)


def test_intra_chain_disulfide_is_dropped_by_design(tmp_path):
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [DISULFIDE], kept=[0, 1, 2, 3])

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    expected_row = ("dropped_by_design", "expected", "OF3 prefilter: polymer link between non-consecutive residues", 2.0)
    actual_row = (row["state"], row["label"], row["reason"], row["input_distance"])
    assert actual_row == expected_row
    assert finding.status == Status.MATCH


def test_kept_disulfide_is_present(tmp_path):
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [DISULFIDE], kept=[0, 1, 2, 3], bonds=[(0, 1)])

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    expected_row = ("present", "same", "A", "A:CYS1:SG", "A:CYS5:SG")
    actual_row = (row["state"], row["label"], row["copy"], row["atom_1"], row["atom_2"])
    assert actual_row == expected_row


def test_long_link_is_dropped_by_design(tmp_path):
    atoms = [("A", "CYS", "1", "SG", "S", 0.0), ("B", "LIG", ".", "C1", "C", 3.0)]
    conn = "covale1 covale A CYS 1 SG ? 1_555 B LIG . C1 ? 1_555"
    inputs = write_run(tmp_path, atoms, [conn], kept=[0, 1])

    [row] = run_check(inputs).data["rows"]

    expected_row = ("dropped_by_design", "OF3 prefilter: longer than 2.4 Å")
    assert (row["state"], row["reason"]) == expected_row


def test_link_between_polymer_chains_is_dropped_by_design(tmp_path):
    atoms = [("A", "CYS", "1", "SG", "S", 0.0), ("B", "CYS", "1", "SG", "S", 2.0)]
    conn = "disulf1 disulf A CYS 1 SG ? 1_555 B CYS 1 SG ? 1_555"
    inputs = write_run(tmp_path, atoms, [conn], kept=[0, 1], molecule_types={"A": "PROTEIN", "B": "PROTEIN"})

    [row] = run_check(inputs).data["rows"]

    expected_row = ("dropped_by_design", "OF3 prefilter: polymer–polymer link between chains")
    assert (row["state"], row["reason"]) == expected_row


def test_unbonded_short_ligand_link_is_missing(tmp_path):
    atoms = [("A", "CYS", "1", "SG", "S", 0.0), ("B", "LIG", ".", "C1", "C", 1.8)]
    conn = "covale1 covale A CYS 1 SG ? 1_555 B LIG . C1 ? 1_555"
    inputs = write_run(tmp_path, atoms, [conn], kept=[0, 1])

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    assert row["state"] == "missing"
    assert finding.status == Status.UNEXPLAINED
    expected_detail = "A:CYS1:SG—B:LIG1:C1: missing"
    assert finding.detail == expected_detail


def test_symmetry_link_is_not_checked(tmp_path):
    conn = "covale1 covale A CYS 1 SG ? 1_555 A CYS 5 SG ? 3_655"
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [conn], kept=[0, 1, 2, 3])

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    expected_row = ("symmetry_link", "same", "link to a crystal-symmetry mate; not checked")
    assert (row["state"], row["label"], row["reason"]) == expected_row
    assert finding.status == Status.MATCH


def test_partner_removed_names_the_step(tmp_path):
    conn = "covale1 covale A CYS 1 SG ? 1_555 B HOH . O ? 1_555"
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [conn], kept=[0, 1, 2], removed={3: "remove_waters"})

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    expected_row = ("partner_removed", "same", "partner removed by remove_waters")
    assert (row["state"], row["label"], row["reason"]) == expected_row
    assert finding.status == Status.MATCH


def test_partner_removed_without_a_logged_step_says_cleanup(tmp_path):
    conn = "covale1 covale A CYS 1 SG ? 1_555 B HOH . O ? 1_555"
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [conn], kept=[0, 1, 2])

    [row] = run_check(inputs).data["rows"]

    expected_reason = "partner removed by cleanup"
    assert row["reason"] == expected_reason


def test_metal_and_hydrogen_bond_links_are_ignored(tmp_path):
    conns = ["metalc1 metalc A CYS 1 SG ? 1_555 A CYS 5 SG ? 1_555", "hydrog1 hydrog A CYS 1 SG ? 1_555 A CYS 5 SG ? 1_555"]
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, conns, kept=[0, 1, 2, 3])

    finding = run_check(inputs)

    expected_links = 0
    assert finding.data["links"] == expected_links


def test_entry_without_links_matches(tmp_path):
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [], kept=[0, 1, 2, 3])

    finding = run_check(inputs)

    expected_data = {"links": 0, "unmapped": 0, "rows": []}
    assert finding.status == Status.MATCH
    assert finding.data == expected_data


def test_link_inside_one_residue_is_not_dropped_by_design(tmp_path):
    atoms = [("A", "CYS", "1", "SG", "S", 0.0), ("A", "CYS", "1", "CB", "C", 1.8)]
    conn = "covale1 covale A CYS 1 SG ? 1_555 A CYS 1 CB ? 1_555"
    inputs = write_run(tmp_path, atoms, [conn], kept=[0, 1])

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    assert row["state"] == "missing"
    assert finding.status == Status.UNEXPLAINED


def test_unmapped_link_is_unexplained(tmp_path):
    conn = "covale1 covale A CYS 1 SG ? 1_555 A CYS 5 XX ? 1_555"
    inputs = write_run(tmp_path, DISULFIDE_ATOMS, [conn], kept=[0, 1, 2, 3])

    finding = run_check(inputs)

    [row] = finding.data["rows"]
    expected_row = ("unmapped", "unexplained", "partner atom not in the input atom table", None)
    assert (row["state"], row["label"], row["reason"], row["input_distance"]) == expected_row
    assert finding.status == Status.UNEXPLAINED
    expected_detail = "A:CYS1:SG—A:CYS5:XX: unmapped"
    assert finding.detail == expected_detail
    assert finding.data["unmapped"] == 1


OPER_HEADER = """#
loop_
_pdbx_struct_oper_list.id
_pdbx_struct_oper_list.matrix[1][1]
_pdbx_struct_oper_list.matrix[1][2]
_pdbx_struct_oper_list.matrix[1][3]
_pdbx_struct_oper_list.vector[1]
_pdbx_struct_oper_list.matrix[2][1]
_pdbx_struct_oper_list.matrix[2][2]
_pdbx_struct_oper_list.matrix[2][3]
_pdbx_struct_oper_list.vector[2]
_pdbx_struct_oper_list.matrix[3][1]
_pdbx_struct_oper_list.matrix[3][2]
_pdbx_struct_oper_list.matrix[3][3]
_pdbx_struct_oper_list.vector[3]
1 1 0 0 0 0 1 0 0 0 0 1 0
2 1 0 0 100 0 1 0 0 0 0 1 0
"""


def test_operator_copy_link_is_checked_per_copy(tmp_path):
    # Chain A (two cysteines, disulfide) has an identity copy A_1 (bonded) and an operator copy A_2 (shifted 100 Å, not bonded).
    inputs = write_run(tmp_path, DISULFIDE_ATOMS[:2], [DISULFIDE], kept=[0, 1], bonds=[(0, 1)])
    cif = inputs.input_cif(PDB_ID)
    cif.write_text(cif.read_text() + OPER_HEADER)
    npz = inputs.out_dir / "structure_files" / PDB_ID / f"{PDB_ID}.npz"
    first = read_atomarray_from_npz(npz)
    first.label_asym_id = np.array(["A_1"] * len(first))
    first.chain_id = np.array(["1"] * len(first))
    second = first.copy()
    second.coord = second.coord + np.array([100.0, 0.0, 0.0])
    second.label_asym_id = np.array(["A_2"] * len(second))
    second.chain_id = np.array(["2"] * len(second))
    both = first + second
    both.bonds = BondList(len(both), np.array([[0, 1, 1]]))
    write_atomarray_to_npz(both, npz)
    chains = {"1": {"molecule_type": "PROTEIN"}, "2": {"molecule_type": "PROTEIN"}}
    (inputs.out_dir / "dataset_cache.json").write_text(json.dumps({"structure_data": {PDB_ID: {"chains": chains}}}))

    finding = run_check(inputs)

    actual_rows = [(row["copy"], row["state"]) for row in finding.data["rows"]]
    expected_rows = [("A_1", "present"), ("A_2", "dropped_by_design")]
    assert actual_rows == expected_rows
    assert finding.data["rows"][1]["reason"] == "OF3 prefilter: polymer link between non-consecutive residues"
