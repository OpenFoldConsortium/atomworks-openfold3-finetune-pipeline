import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from biotite.structure import Atom, BondList, array
from biotite.structure.io import pdbx
from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz, write_atomarray_to_npz

from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.origin import classify_origins
from aw_of3_finetune.report.validity import VALIDITY_CHECKS, ValidityInputs
from aw_of3_finetune.report.validity.ligands import (
    COULD_NOT_RUN,
    LIGAND_CHEMISTRY,
    LIGAND_CONTACTS,
    LIGAND_GEOMETRY,
    LIGAND_IDENTITY,
    _BOND_TYPES,
    _CONTACT_TESTS,
    _rows,
    check_ligand_chemistry,
    check_ligand_contacts,
    check_ligand_geometry,
    check_ligand_identity,
    describe_row,
)
from validate_helpers import build_dataset

REPO = Path(__file__).resolve().parents[3]
PDB_ID = "tiny"
SINGLE, DOUBLE = 1, 2
TINY_CCD = """data_LIG
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
LIG C1 C 0 N
LIG C2 C 0 N
LIG C3 C 0 N
LIG C4 C 0 Y
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
LIG C1 C2 SING N
LIG C2 C3 SING N
LIG C3 C4 SING N
data_CHI
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
CHI C1 C 0 N 0.0 0.0 0.0
CHI F1 F 0 N 0.8 0.8 0.8
CHI CL1 CL 0 N 1.0 -1.0 -1.0
CHI BR1 BR 0 N -1.0 1.0 -1.0
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
CHI C1 F1 SING N
CHI C1 CL1 SING N
CHI C1 BR1 SING N
data_THI
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
THI O1 O 0 N
THI C1 C 0 N
THI HS H 0 Y
THI H1 H 0 N
THI H2 H 0 N
THI H3 H 0 N
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
THI O1 C1 SING N
THI O1 HS SING N
THI C1 H1 SING N
THI C1 H2 SING N
THI C1 H3 SING N
"""

def _structure(residues, bonds):
    """Atoms of (chain, res_name, res_id, atom_name, element, x) tuples joined by the bonds."""
    atoms = [
        Atom([np.nan] * 3 if x is None else (x, 0.0, 0.0) if np.isscalar(x) else x, chain_id=chain, res_id=res_id, res_name=res_name,
             atom_name=name, element=element, label_asym_id=chain, hetero=chain == "L")
        for chain, res_name, res_id, name, element, x in residues
    ]
    structure = array(atoms)
    structure.bonds = BondList(len(atoms), np.array([(b[0], b[1], b[2] if len(b) > 2 else SINGLE) for b in bonds], dtype=int).reshape(-1, 3))
    return structure


def write_tiny_run(tmp_path, residues, bonds, input_residues=None):
    """A run with chain ids "P" (protein) and "L" (ligand), and an input CIF cifs/tiny.cif.

    residues: (chain, res_name, res_id, atom_name, element, x), x None = unresolved, a float = (x, 0, 0) or an
    (x, y, z) tuple. bonds: (i, j) index pairs, or (i, j, biotite bond type). input_residues: the input CIF's
    atoms in the same form (default: the resolved residues).
    """
    structure = _structure(residues, bonds)
    out_dir = tmp_path / "out"
    (out_dir / "structure_files" / PDB_ID).mkdir(parents=True)
    (out_dir / "manifest.json").write_text(json.dumps({"structures": [PDB_ID]}))
    write_atomarray_to_npz(structure, out_dir / "structure_files" / PDB_ID / f"{PDB_ID}.npz")
    input_structure = structure if input_residues is None else _structure(input_residues, [])
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, input_structure[~np.isnan(input_structure.coord).any(axis=1)])
    (tmp_path / "cifs").mkdir()
    cif.write(tmp_path / "cifs" / f"{PDB_ID}.cif")
    chains = {"P": {"molecule_type": "PROTEIN"}, "L": {"molecule_type": "LIGAND"}}
    (out_dir / "dataset_cache.json").write_text(json.dumps({"structure_data": {PDB_ID: {"chains": chains}}}))
    (tmp_path / "ccd.cif").write_text(TINY_CCD)
    return ValidityInputs(out_dir, tmp_path / "cifs", tmp_path / "ccd.cif")


@pytest.fixture(scope="module")
def built(tmp_path_factory, ccd_path):
    """Real finetune-prep output of the 7axf, 7riv and 1a3n fixtures, with NPZ."""
    cifs = [REPO / "src/tests/fixtures/mmcifs/7axf.cif", REPO / "src/tests/fixtures/mmcifs/7riv.cif",
            REPO / "src/tests/fixtures/mmcifs/1a3n.cif"]
    built = build_dataset(tmp_path_factory.mktemp("ligands"), cifs, ccd_path)
    return ValidityInputs(built.output_dir, built.cif_dir, ccd_path)


def test_checks_are_registered():
    expected_names = {LIGAND_CHEMISTRY, LIGAND_IDENTITY, LIGAND_GEOMETRY, LIGAND_CONTACTS}
    actual_names = expected_names & set(VALIDITY_CHECKS)
    assert actual_names == expected_names


def test_7axf_covalent_s6t_passes_sanitization(built):
    """The covalent S6T no longer gets its leaving CL back, so it sanitizes (was bug 2)."""
    actual = [f for f in check_ligand_chemistry(built, "7axf") if "S6T chain 3" in f.detail]

    expected_statuses = [Status.MATCH]
    actual_statuses = [f.status for f in actual]
    assert actual_statuses == expected_statuses


def test_7riv_p06_passes_all_ligand_checks(built):
    checks = [check_ligand_chemistry, check_ligand_identity, check_ligand_geometry, check_ligand_contacts]

    actual = [f for check in checks for f in check(built, "7riv")]

    expected_statuses = [Status.MATCH] * 4
    assert [f.status for f in actual] == expected_statuses, [f.detail for f in actual]


def test_covalent_partner_is_not_a_clash(tmp_path):
    residues = [
        ("L", "LIG", 1, "C1", "C", 0.0),
        ("L", "LIG", 1, "C2", "C", 1.5),
        ("P", "CYS", 1, "SG", "S", -1.8),
        ("P", "GLY", 2, "CA", "C", 4.5),
    ]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (0, 2)])

    actual = check_ligand_contacts(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.MATCH], actual[0].detail


def test_1a3n_heme_passes_all_ligand_checks(built):
    """HEM bound to histidine: iron's dative bonds, square-planar N-Fe-N angles and Fe-N coordination all pass."""
    checks = [check_ligand_chemistry, check_ligand_identity, check_ligand_geometry, check_ligand_contacts]

    actual = [f for check in checks for f in check(built, "1a3n") if f.detail.startswith("HEM chain 5")]

    expected_statuses = [Status.MATCH] * 4
    assert [f.status for f in actual] == expected_statuses, [f.detail for f in actual]


@pytest.mark.parametrize("unknown", ["?", "."])
def test_unknown_ccd_charge_counts_as_neutral(tmp_path, unknown):
    """wwPDB's CCD leaves some formal charges unknown, such as heme's iron; the checks still run."""
    residues = [("L", "LIG", 1, "C1", "C", 0.0), ("L", "LIG", 1, "C2", "C", 1.5), ("L", "LIG", 1, "C3", "C", 3.0)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2)])
    ccd_file = tmp_path / "ccd.cif"
    ccd_file.write_text(ccd_file.read_text().replace("LIG C1 C 0 N", f"LIG C1 C {unknown} N"))

    actual = check_ligand_chemistry(inputs, PDB_ID)

    expected_statuses = [Status.MATCH]
    assert [f.status for f in actual] == expected_statuses, [f.detail for f in actual]


def test_ccd_oxygen_charged_but_still_protonated_is_neutral(tmp_path):
    """wwPDB's 2026 HEM gives O2A charge -1 and keeps its H2A; an O with two bonds cannot be O-."""
    residues = [("L", "LIG", 1, "C1", "C", 0.0), ("L", "LIG", 1, "C2", "C", 1.5), ("L", "LIG", 1, "C3", "C", 3.0),
                ("L", "LIG", 1, "O1", "O", 4.4)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2), (2, 3)])
    ccd_file = tmp_path / "ccd.cif"
    text = ccd_file.read_text()
    text = text.replace("LIG C4 C 0 Y\n", "LIG C4 C 0 Y\nLIG O1 O -1 N\nLIG H1 H 0 N\n", 1)
    text = text.replace("LIG C3 C4 SING N\n", "LIG C3 C4 SING N\nLIG C3 O1 SING N\nLIG O1 H1 SING N\n", 1)
    ccd_file.write_text(text)

    actual = check_ligand_chemistry(inputs, PDB_ID)

    expected_statuses = [Status.MATCH]
    assert [f.status for f in actual] == expected_statuses, [f.detail for f in actual]


def test_coordinated_metal_ion_is_not_a_clash(tmp_path):
    """A zinc ion 2.0 Å from a histidine nitrogen is coordination: judged on covalent radii, it passes."""
    residues = [
        ("L", "ZN", 1, "ZN", "ZN", 0.0),
        ("P", "HIS", 1, "NE2", "N", 2.0),
        ("P", "GLY", 2, "CA", "C", 5.0),
    ]
    inputs = write_tiny_run(tmp_path, residues, [])

    actual = check_ligand_contacts(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.MATCH], actual[0].detail


def test_non_metal_atom_at_coordination_distance_still_clashes(tmp_path):
    """The same 2.0 Å from a carbon is a clash: only metal atoms get covalent radii."""
    residues = [
        ("L", "LIG", 1, "C1", "C", 0.0),
        ("L", "LIG", 1, "C2", "C", -1.5),
        ("P", "HIS", 1, "NE2", "N", 2.0),
        ("P", "GLY", 2, "CA", "C", 5.0),
    ]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1)])

    actual = check_ligand_contacts(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.UNEXPLAINED]


def test_ligand_with_unresolved_atoms_skips_identity(tmp_path):
    residues = [
        ("L", "LIG", 1, "C1", "C", 0.0),
        ("L", "LIG", 1, "C2", "C", 1.5),
        ("L", "LIG", 1, "C3", "C", None),
    ]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2)])

    actual = check_ligand_identity(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.MATCH]
    assert actual[0].data["skipped"] == "unresolved atoms"


def test_unresolved_leaving_atom_does_not_skip_identity(tmp_path):
    residues = [
        ("L", "LIG", 1, "C1", "C", 0.0),
        ("L", "LIG", 1, "C2", "C", 1.5),
        ("L", "LIG", 1, "C3", "C", 3.0),
        ("L", "LIG", 1, "C4", "C", None),
    ]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2), (2, 3)])

    actual = check_ligand_identity(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.MATCH]
    assert "skipped" not in actual[0].data


def test_ligand_not_in_ccd_is_reported_not_raised(tmp_path):
    residues = [("L", "ZZZ", 1, "C1", "C", 0.0), ("L", "ZZZ", 1, "C2", "C", 1.5)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1)])

    actual = check_ligand_identity(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.UNEXPLAINED]
    assert actual[0].detail == "ZZZ chain L: not in CCD"
    assert len(check_ligand_chemistry(inputs, PDB_ID)) == 1


@pytest.mark.parametrize("bonds", [[(0, 1), (1, 2, DOUBLE)], [(0, 1), (1, 2), (0, 2)]], ids=["bond order", "extra bond"])
def test_wrong_bonds_fail_identity(tmp_path, bonds):
    residues = [
        ("L", "LIG", 1, "C1", "C", 0.0),
        ("L", "LIG", 1, "C2", "C", 1.5),
        ("L", "LIG", 1, "C3", "C", 3.0),
        ("L", "LIG", 1, "C4", "C", None),
    ]
    inputs = write_tiny_run(tmp_path, residues, bonds)

    actual = check_ligand_identity(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.UNEXPLAINED]


CHIRAL = [("L", "CHI", 1, "C1", "C", (0.0, 0.0, 0.0)), ("L", "CHI", 1, "F1", "F", (0.8, 0.8, 0.8)),
          ("L", "CHI", 1, "CL1", "CL", (1.0, -1.0, -1.0)), ("L", "CHI", 1, "BR1", "BR", (-1.0, 1.0, -1.0))]


def test_same_stereocenter_passes_identity(tmp_path):
    inputs = write_tiny_run(tmp_path, CHIRAL, [(0, 1), (0, 2), (0, 3)])

    actual = check_ligand_identity(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.MATCH], actual[0].detail
    assert "stereo_skipped" not in actual[0].data


def test_mirrored_stereocenter_fails_identity(tmp_path):
    mirrored = [(*atom[:5], (x, y, -z)) for *atom, (x, y, z) in [(*a[:5], a[5]) for a in CHIRAL]]
    inputs = write_tiny_run(tmp_path, mirrored, [(0, 1), (0, 2), (0, 3)])

    actual = check_ligand_identity(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.UNEXPLAINED]
    assert "stereo" in actual[0].detail


def test_ideal_coordinates_missing_skips_stereo(tmp_path):
    residues = [("L", "LIG", 1, "C1", "C", 0.0), ("L", "LIG", 1, "C2", "C", 1.5), ("L", "LIG", 1, "C3", "C", 3.0)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2)])

    actual = check_ligand_identity(inputs, PDB_ID)

    assert actual[0].data["stereo_skipped"] is True


def test_leaving_hydrogen_replaced_by_covalent_bond_sanitizes(tmp_path):
    residues = [("L", "THI", 1, "O1", "O", 0.0), ("L", "THI", 1, "C1", "C", 1.8), ("P", "ALA", 1, "CB", "C", -1.8)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (0, 2)])

    actual = check_ligand_chemistry(inputs, PDB_ID)

    assert [f.status for f in actual] == [Status.MATCH], actual[0].detail


def test_biotite_bond_types_map_without_error():
    expected_types = {4, 7, 8, 9}
    assert expected_types <= set(_BOND_TYPES)


def test_nan_value_becomes_none():
    results = {"no_volume_clash": True}

    actual = _rows("L", "LIG", results, _CONTACT_TESTS, {"no_volume_clash": (float("nan"), 0.075)})

    expected_row = {"chain": "L", "res_name": "LIG", "test": "volume overlap with protein", "passed": True,
                    "value": None, "threshold": 0.075}
    assert actual == [expected_row]


def test_pd_na_pass_flag_is_could_not_run():
    """A pd.NA flag reads as could-not-run instead of raising in bool()."""
    results = {"no_volume_clash": pd.NA}

    actual = _rows("L", "LIG", results, _CONTACT_TESTS)

    expected_row = {"chain": "L", "res_name": "LIG", "test": "volume overlap with protein", "passed": False,
                    "value": None, "threshold": None, "label": "unexplained", "reason": "test could not run"}
    assert actual == [expected_row]


def test_chemistry_has_no_inchi_convertible_test(tmp_path):
    """InChI conversion is not a chemistry test: RDKit cannot write InChI for dative bonds (rdkit#6853)."""
    residues = [("L", "LIG", 1, "C1", "C", 0.0), ("L", "LIG", 1, "C2", "C", 1.5), ("L", "LIG", 1, "C3", "C", 3.0)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2)])

    [actual] = check_ligand_chemistry(inputs, PDB_ID)

    expected_tests = ["sanitization", "all atoms connected"]
    assert [row["test"] for row in actual.data["rows"]] == expected_tests
    assert actual.status == Status.MATCH, actual.detail


def test_geometry_rows_carry_measured_values_and_thresholds(tmp_path):
    residues = [("L", "LIG", 1, "C1", "C", (0.0, 0.0, 0.0)), ("L", "LIG", 1, "C2", "C", (1.5, 0.0, 0.0)),
                ("L", "LIG", 1, "C3", "C", (2.0, 1.4, 0.0))]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2)])

    [actual] = check_ligand_geometry(inputs, PDB_ID)

    rows = {row["test"]: row for row in actual.data["rows"]}
    expected_thresholds = {
        "bond angles": 1.25,
        "internal clash": 0.7,
        "aromatic ring flatness": 0.25,
        "non-aromatic ring non-flatness": 0.05,
        "double bond flatness": 0.25,
    }
    actual_thresholds = {test: rows[test]["threshold"] for test in expected_thresholds}
    assert actual_thresholds == expected_thresholds
    assert rows["bond lengths"]["threshold"] in (0.75, 1.25)
    assert isinstance(rows["bond lengths"]["value"], float)
    assert isinstance(rows["bond angles"]["value"], float)
    assert rows["aromatic ring flatness"]["value"] is None


@pytest.mark.parametrize("flag", [float("nan"), None])
def test_nan_pass_flag_is_not_passed(tmp_path, monkeypatch, flag):
    residues = [("L", "LIG", 1, "C1", "C", (0.0, 0.0, 0.0)), ("L", "LIG", 1, "C2", "C", (1.5, 0.0, 0.0)),
                ("L", "LIG", 1, "C3", "C", (2.0, 1.4, 0.0))]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2)])
    unrunnable = {"results": {"flatness_passes": flag, "max_distance": 0.1}}
    monkeypatch.setattr("aw_of3_finetune.report.validity.ligands.check_flatness", lambda *args, **kwargs: unrunnable)

    [finding] = check_ligand_geometry(inputs, PDB_ID)

    actual_row = next(row for row in finding.data["rows"] if row["test"] == "aromatic ring flatness")
    expected_row = {"chain": "L", "res_name": "LIG", "test": "aromatic ring flatness", "passed": False,
                    "value": None, "threshold": 0.25, "label": "unexplained", "reason": "test could not run"}
    assert actual_row == expected_row
    assert finding.status == Status.UNEXPLAINED
    expected_detail = "LIG chain L: aromatic ring flatness could not run; non-aromatic ring non-flatness could not run; double bond flatness could not run"
    assert finding.detail == expected_detail


def _swapped(atoms, name_1, name_2):
    """The atoms with the coordinates of the two named atoms swapped."""
    xyz = {atom[3]: atom[5] for atom in atoms}
    swap = {name_1: xyz[name_2], name_2: xyz[name_1]}
    return [(*atom[:5], swap.get(atom[3], atom[5])) for atom in atoms]


def _mirrored(atoms):
    """The atoms reflected through the xy plane."""
    return [(*atom[:5], (x, y, -z)) for *atom, (x, y, z) in [(*a[:5], a[5]) for a in atoms]]


CHIRAL_BONDS = [(0, 1), (0, 2), (0, 3)]


def test_identity_input_rows_pass_when_unchanged(tmp_path):
    inputs = write_tiny_run(tmp_path, CHIRAL, CHIRAL_BONDS)

    [finding] = check_ligand_identity(inputs, PDB_ID)

    rows = finding.data["rows"]
    actual_input_tests = [r["test"] for r in rows if r["reference"] == "input"]
    actual_failed = [r for r in rows if r["reference"] == "input" and r["passed"] is False]
    expected_input_tests = ["input double bond stereo", "input tetrahedral stereo", "input stereo"]
    assert actual_input_tests == expected_input_tests
    assert actual_failed == []


def test_identity_swapped_atom_names_fail_input_stereo(tmp_path):
    """The final structure has F1 and CL1 swapped vs the input: a processing failure."""
    inputs = write_tiny_run(tmp_path, _swapped(CHIRAL, "F1", "CL1"), CHIRAL_BONDS, input_residues=CHIRAL)

    findings = check_ligand_identity(inputs, PDB_ID)
    [classified] = classify_origins(inputs, findings)

    actual_row = next(r for r in findings[0].data["rows"] if r["test"] == "input tetrahedral stereo")
    assert actual_row["passed"] is False
    assert actual_row["reference"] == "input"
    assert classified.data["origin"] == "processing"


def test_identity_ccd_only_difference_is_input(tmp_path):
    """The input itself has the CCD's mirror image and processing kept it: only CCD rows fail."""
    inputs = write_tiny_run(tmp_path, _mirrored(CHIRAL), CHIRAL_BONDS)

    findings = check_ligand_identity(inputs, PDB_ID)
    [classified] = classify_origins(inputs, findings)

    actual_failed_references = {r["reference"] for r in findings[0].data["rows"] if r["passed"] is not True}
    expected_failed_references = {"ccd"}
    assert actual_failed_references == expected_failed_references
    assert classified.data["origin"] == "input"


def test_identity_without_input_match_could_not_run(tmp_path):
    """An atom absent from the input CIF: one could-not-run input row."""
    inputs = write_tiny_run(tmp_path, CHIRAL, CHIRAL_BONDS, input_residues=CHIRAL[:3])

    [finding] = check_ligand_identity(inputs, PDB_ID)

    actual_input_rows = [r for r in finding.data["rows"] if r["reference"] == "input"]
    expected_input_rows = [{"chain": "L", "res_name": "CHI", "reference": "input", "test": "input match", "passed": None,
                            "value": None, "threshold": None, "label": "unexplained", "reason": COULD_NOT_RUN}]
    assert actual_input_rows == expected_input_rows


def test_identity_input_stereo_crash_could_not_run(tmp_path):
    """Collinear atoms in a ring make PoseBusters' stereo perception raise: one could-not-run input row."""
    residues = [("L", "LIG", 1, "C1", "C", 0.0), ("L", "LIG", 1, "C2", "C", 1.5), ("L", "LIG", 1, "C3", "C", 3.0)]
    inputs = write_tiny_run(tmp_path, residues, [(0, 1), (1, 2), (0, 2)])

    [finding] = check_ligand_identity(inputs, PDB_ID)

    actual_input_rows = [r for r in finding.data["rows"] if r["reference"] == "input"]
    expected_input_rows = [{"chain": "L", "res_name": "LIG", "reference": "input", "test": "input stereo", "passed": None,
                            "value": None, "threshold": None, "label": "unexplained", "reason": COULD_NOT_RUN}]
    assert actual_input_rows == expected_input_rows


def test_identity_ccd_only_difference_with_unmatched_input_is_input(tmp_path):
    """A CCD mirror image whose input match could not run (an atom missing from the input) stays input, not red."""
    mirrored = _mirrored(CHIRAL)
    inputs = write_tiny_run(tmp_path, mirrored, CHIRAL_BONDS, input_residues=mirrored[:3])

    findings = check_ligand_identity(inputs, PDB_ID)
    [classified] = classify_origins(inputs, findings)

    expected_origin = "input"
    actual_origin = classified.data["origin"]
    assert actual_origin == expected_origin


OPERATORS = """#
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


def test_identity_assembly_copy_uses_its_own_input_residue(tmp_path):
    """Input chain L has two CHI residues (res 1 mirrored, far away; res 2 kept). The final keeps res 2 as
    copy L_1 and an operator copy L_2 shifted 100 Å; L_2's input atoms are res 2's, never res 1's same names."""
    far_mirrored = [(chain, name, 1, atom, element, (x + 50.0, y, -z))
                    for chain, name, _, atom, element, (x, y, z) in CHIRAL]
    kept = [(chain, name, 2, atom, element, xyz) for chain, name, _, atom, element, xyz in CHIRAL]
    inputs = write_tiny_run(tmp_path, kept, CHIRAL_BONDS, input_residues=far_mirrored + kept)
    cif = inputs.input_cif(PDB_ID)
    cif.write_text(cif.read_text() + OPERATORS)
    npz = inputs.out_dir / "structure_files" / PDB_ID / f"{PDB_ID}.npz"
    first = read_atomarray_from_npz(npz)
    first.label_asym_id = np.array(["L_1"] * len(first))
    second = first.copy()
    second.coord = second.coord + np.array([100.0, 0.0, 0.0])
    second.label_asym_id = np.array(["L_2"] * len(second))
    second.chain_id = np.array(["M"] * len(second))
    both = first + second
    write_atomarray_to_npz(both, npz)
    chains = {"L": {"molecule_type": "LIGAND"}, "M": {"molecule_type": "LIGAND"}}
    (inputs.out_dir / "dataset_cache.json").write_text(json.dumps({"structure_data": {PDB_ID: {"chains": chains}}}))

    findings = check_ligand_identity(inputs, PDB_ID)

    copy_finding = next(f for f in findings if "chain M" in f.detail)
    actual_failed_input_rows = [r["test"] for r in copy_finding.data["rows"]
                                if r["reference"] == "input" and r["passed"] is not True]
    expected_failed_input_rows = []
    assert actual_failed_input_rows == expected_failed_input_rows


def _failed_row(test, value, threshold, **extra):
    """A failed ligand row of chain L."""
    return {"chain": "L", "res_name": "LIG", "test": test, "passed": False, "value": value, "threshold": threshold} | extra


def test_long_ligand_bond_is_described():
    actual = describe_row(_failed_row("bond lengths", 1.554, 1.25))

    assert actual == "a bond too long (1.55× its upper bound, limit 1.25×)"


def test_short_ligand_bond_is_described():
    actual = describe_row(_failed_row("bond lengths", 0.53, 0.75))

    assert actual == "a bond too short (0.53× its lower bound, limit 0.75×)"


def test_angle_is_described():
    actual = describe_row(_failed_row("bond angles", 1.316, 1.25))

    assert actual == "an angle outside its range (1.32× the bound, limit 1.25×)"


def test_protein_clash_is_described():
    actual = describe_row(_failed_row("minimum distance to protein", 0.654, 0.75))

    assert actual == "too close to the protein (0.65× the van der Waals distance, limit 0.75×)"


def test_identity_row_names_its_reference():
    actual = describe_row(_failed_row("formula", None, None, reference="input"))

    assert actual == "formula differs from the input ligand"


def test_row_without_value_keeps_the_test_name():
    actual = describe_row(_failed_row("aromatic ring flatness", None, 0.25))

    assert actual == "aromatic ring not flat"


def test_could_not_run_row_says_so():
    actual = describe_row(_failed_row("bond angles", None, None, reason=COULD_NOT_RUN))

    assert actual == "bond angles could not run"
