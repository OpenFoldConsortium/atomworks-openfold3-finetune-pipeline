"""Tests for the ligand checks V4, V8-V12, V18, V19."""

import json
import re
import shutil

import numpy as np

from aw_of3_finetune.validate import CHECKS, run_validation
from aw_of3_finetune.validate.findings import Severity
from validate_helpers import context, mutable_copy

LIGAND_CHECKS = ["V4", "V8", "V9", "V10", "V11", "V12", "V18", "V19"]


def _rewrite_npz(built, entry, edit):
    """Load an entry's NPZ arrays, apply `edit(arrays)` in place, and save them back."""
    path = built.output_dir / "structure_files" / entry / f"{entry}.npz"
    with np.load(path, allow_pickle=True) as data:
        arrays = {name: data[name] for name in data.files}
    edit(arrays)
    np.savez(path, **arrays)


def _errors(report, check, entry):
    """The error findings of one check on one entry."""
    return [f for f in report.errors if f.check == check and f.entry == entry]


def _p06_rows(arrays):
    """Atom indices of the P06 ligand (7riv's only ligand) in the NPZ arrays."""
    return np.flatnonzero(arrays["res_name"] == "P06")


def test_ligand_checks_are_registered():
    """Every ligand check id is registered, and V9 and V10 share one implementation."""
    registered = {check for check in LIGAND_CHECKS if check in CHECKS}
    assert registered == set(LIGAND_CHECKS)
    assert CHECKS["V9"] is CHECKS["V10"]


def test_ligand_checks_pass_on_clean_build(built_dataset):
    """A clean build has no errors, and only V4 (no-ligand note) produces findings."""
    report = run_validation(context(built_dataset), LIGAND_CHECKS)

    assert report.errors == [], [f.message for f in report.errors]

    findings_from_other_checks = [f for f in report.findings if f.check != "V4"]
    assert findings_from_other_checks == []


def test_v9_v10_do_not_duplicate_findings(built_dataset, tmp_path):
    """Running V9 and V10 together reports one unknown atom name once, under V10 only."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def rename(arrays):
        arrays["atom_name"][_p06_rows(arrays)[0]] = "ZZ9"

    _rewrite_npz(built, "7riv", rename)

    report = run_validation(context(built), ["V9", "V10"])

    assert [f.check for f in report.errors] == ["V10"]


def test_v4_warns_only_when_expected(built_dataset):
    """V4 warns about entries without ligands only when --expect-ligands is set."""
    quiet_report = run_validation(context(built_dataset), ["V4"])
    loud_report = run_validation(context(built_dataset, expect_ligands=True), ["V4"])

    # Without the flag, 2crb gets no finding.
    quiet_2crb_findings = [f for f in quiet_report.findings if f.entry == "2crb"]
    assert quiet_2crb_findings == []

    # With the flag, the three entries without a ligand get a warning.
    v4_warnings = [f for f in loud_report.warnings if f.check == "V4"]
    assert {f.entry for f in v4_warnings} == {"2crb", "2q2k", "3u8v"}
    assert "no ligand" in v4_warnings[0].message

    # Warnings only; nothing fails.
    assert loud_report.errors == []


def test_v8_detects_missing_sdf(built_dataset, tmp_path):
    """V8 flags 7riv when its ligand's reference SDF (P06.sdf) is deleted."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    (built.output_dir / "reference_mols" / "P06.sdf").unlink()

    report = run_validation(context(built), ["V8"])

    assert report.failing_entries == {"7riv"}
    assert "P06.sdf" in _errors(report, "V8", "7riv")[0].message


def test_v8_per_residue_lookup(built_dataset_with_waters, tmp_path):
    """Waters (kept via remove_waters: false) and ions resolve per residue."""
    cache = json.loads((built_dataset_with_waters.output_dir / "dataset_cache.json").read_text())
    chains = cache["structure_data"]["3u8v"]["chains"]

    # Precondition: some 3u8v ligand chains (water, ion) have no single reference mol,
    # so V8 has to look up each residue separately.
    assert any(
        chain["molecule_type"] == "LIGAND" and chain["reference_mol_id"] is None
        for chain in chains.values()
    )

    # With all reference SDFs present there is no error.
    clean_report = run_validation(context(built_dataset_with_waters), ["V8"])
    assert clean_report.errors == []

    # Deleting the water (HOH) or the nickel ion (NI) SDF is flagged on 3u8v by name.
    for mol_id in ("HOH", "NI"):
        built = mutable_copy(built_dataset_with_waters, tmp_path / mol_id)
        (built.output_dir / "reference_mols" / f"{mol_id}.sdf").unlink()

        report = run_validation(context(built), ["V8"])

        assert f"{mol_id}.sdf" in _errors(report, "V8", "3u8v")[0].message


def test_v10_detects_unknown_atom_name(built_dataset, tmp_path):
    """V10 flags a ligand atom name (ZZ9) that is not in the reference SDF."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def rename(arrays):
        arrays["atom_name"][_p06_rows(arrays)[0]] = "ZZ9"

    _rewrite_npz(built, "7riv", rename)

    report = run_validation(context(built), ["V9", "V10"])

    assert report.failing_entries == {"7riv"}
    assert "ZZ9" in _errors(report, "V10", "7riv")[0].message


def test_v9_detects_element_change(built_dataset, tmp_path):
    """V9 flags a ligand atom whose element differs from the reference, and V10 stays quiet."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def change(arrays):
        row = _p06_rows(arrays)[0]
        arrays["element"][row] = "N" if arrays["element"][row] != "N" else "O"

    _rewrite_npz(built, "7riv", change)

    report = run_validation(context(built), ["V9", "V10"])

    assert report.failing_entries == {"7riv"}
    assert "element" in _errors(report, "V9", "7riv")[0].message

    # The atom name is unchanged, so V10 has nothing to report.
    assert _errors(report, "V10", "7riv") == []


def _duplicate_entry(built, src, dst):
    """Copy entry `src` (CIF, NPZ and cache metadata) to a new entry `dst`."""
    out = built.output_dir
    shutil.copy(out / "structures" / f"{src}.cif", out / "structures" / f"{dst}.cif")
    (out / "structure_files" / dst).mkdir()
    shutil.copy(
        out / "structure_files" / src / f"{src}.npz",
        out / "structure_files" / dst / f"{dst}.npz",
    )
    path = out / "dataset_cache.json"
    data = json.loads(path.read_text())
    data["structure_data"][dst] = data["structure_data"][src]
    path.write_text(json.dumps(data, indent=4))


def test_v11_detects_two_chemistries(built_dataset, tmp_path):
    """V11 flags both entries when two copies of P06 disagree on an element."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    # Precondition: the unmodified dataset is consistent.
    assert run_validation(context(built), ["V11"]).errors == []

    # Duplicate 7riv as 7riw, then change one P06 atom's element in 7riw.
    _duplicate_entry(built, "7riv", "7riw")

    def change(arrays):
        row = _p06_rows(arrays)[0]
        arrays["element"][row] = "N" if arrays["element"][row] != "N" else "O"

    _rewrite_npz(built, "7riw", change)

    report = run_validation(context(built), ["V11"])

    assert report.failing_entries == {"7riv", "7riw"}

    # The finding names the molecule, the property that differs, and both entries.
    finding = _errors(report, "V11", "7riw")[0]
    assert "P06" in finding.message
    assert "element" in finding.message
    assert finding.detail["mol_id"] == "P06"
    assert finding.detail["entries"] == ["7riv", "7riw"]


def test_v11_detects_bond_disagreement(built_dataset, tmp_path):
    """V11 flags both entries when one copy of P06 is missing a bond."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _duplicate_entry(built, "7riv", "7riw")

    def drop_bond(arrays):
        ligand_rows = set(_p06_rows(arrays).tolist())
        bonds = arrays["bonds"]
        bond_index = next(
            i
            for i, (atom_a, atom_b, _) in enumerate(bonds.tolist())
            if atom_a in ligand_rows and atom_b in ligand_rows
        )
        arrays["bonds"] = np.delete(bonds, bond_index, axis=0)

    _rewrite_npz(built, "7riw", drop_bond)

    report = run_validation(context(built), ["V11"])

    assert report.failing_entries == {"7riv", "7riw"}
    assert "bond" in _errors(report, "V11", "7riw")[0].message


def test_v11_detects_disjoint_atom_names(built_dataset, tmp_path):
    """V11 flags both entries when one copy of P06 renames every atom."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _duplicate_entry(built, "7riv", "7riw")

    def rename(arrays):
        rows = _p06_rows(arrays)
        arrays["atom_name"] = arrays["atom_name"].astype("<U8")
        arrays["atom_name"][rows] = [f"Z{name}" for name in arrays["atom_name"][rows]]

    _rewrite_npz(built, "7riw", rename)

    report = run_validation(context(built), ["V11"])

    assert report.failing_entries == {"7riv", "7riw"}
    assert "no atom names in common" in _errors(report, "V11", "7riw")[0].message


def test_v11_allows_partial_ligand(built_dataset, tmp_path):
    """A copy of P06 missing its last atom (partially resolved ligand) is not a conflict."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _duplicate_entry(built, "7riv", "7riw")

    def drop_atom(arrays):
        n_atoms = len(arrays["atom_name"])
        dropped_atom = _p06_rows(arrays)[-1]
        keep = np.arange(n_atoms) != dropped_atom

        # Re-index the surviving bonds after the atom is removed.
        new_index = np.cumsum(keep) - 1
        bonds = arrays["bonds"]
        bonds = bonds[keep[bonds[:, 0]] & keep[bonds[:, 1]]]
        bonds[:, 0], bonds[:, 1] = new_index[bonds[:, 0]], new_index[bonds[:, 1]]

        # Drop the atom from every per-atom array.
        for key, value in list(arrays.items()):
            if key != "bonds" and getattr(value, "ndim", 0) >= 1 and len(value) == n_atoms:
                arrays[key] = value[keep]
        arrays["bonds"] = bonds

    _rewrite_npz(built, "7riw", drop_atom)

    report = run_validation(context(built), ["V11"])

    assert report.errors == []


def test_v12_detects_unsanitizable_sdf(built_dataset, tmp_path):
    """V12 flags a reference SDF that RDKit cannot sanitize (a carbon with five bonds)."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    (built.output_dir / "reference_mols" / "P06.sdf").write_text(
        "\n     RDKit          3D\n\n"
        "  6  5  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "   -1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000   -1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000    0.0000    1.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0\n  1  3  1  0\n  1  4  1  0\n  1  5  1  0\n  1  6  1  0\n"
        "M  END\n$$$$\n"
    )

    report = run_validation(context(built), ["V12"])

    assert report.failing_entries == {"7riv"}
    assert "P06" in _errors(report, "V12", "7riv")[0].message


def test_v18_detects_ligand_in_polymer(built_dataset, tmp_path):
    """V18 flags a ligand relabeled into a protein chain."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def relabel(arrays):
        first_protein_atom = np.flatnonzero(arrays["molecule_type_id"] == 0)[0]
        protein_chain = arrays["chain_id"][first_protein_atom]
        arrays["chain_id"][_p06_rows(arrays)] = protein_chain

    _rewrite_npz(built, "7riv", relabel)

    report = run_validation(context(built), ["V18"])

    assert report.failing_entries == {"7riv"}
    assert "P06" in _errors(report, "V18", "7riv")[0].message


def test_v19_detects_merged_copies(built_dataset, tmp_path):
    """V19 flags two ligand copies merged into one chain (a multi-atom ligand chain)."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def merge(arrays):
        arrays["chain_id"][arrays["chain_id"] == "6"] = "5"

    _rewrite_npz(built, "1a3n", merge)

    report = run_validation(context(built), ["V19"])

    assert report.failing_entries == {"1a3n"}
    assert "multi-atom" in _errors(report, "V19", "1a3n")[0].message


def test_v19_detects_bond_between_ligand_chains(built_dataset, tmp_path):
    """V19 flags a bond added between two different ligand chains."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def bridge(arrays):
        first_atom = int(np.flatnonzero(arrays["chain_id"] == "5")[0])
        second_atom = int(np.flatnonzero(arrays["chain_id"] == "6")[0])
        new_bond = np.array([[first_atom, second_atom, 1]], dtype=arrays["bonds"].dtype)
        arrays["bonds"] = np.vstack([arrays["bonds"], new_bond])

    _rewrite_npz(built, "1a3n", bridge)

    report = run_validation(context(built), ["V19"])

    assert report.failing_entries == {"1a3n"}
    assert "between ligand chains" in _errors(report, "V19", "1a3n")[0].message


def test_v19_allows_water_chain(built_dataset_with_waters):
    """Waters (kept via remove_waters: false): one chain of many HOH is not merged copies."""
    npz = context(built_dataset_with_waters).npz("1a3n")
    water = npz["res_name"] == "HOH"
    water_chain = npz["chain_id"][water][0]

    # Precondition: the water chain holds more than one water residue.
    water_residue_ids = set(npz["res_id"][water & (npz["chain_id"] == water_chain)].tolist())
    assert len(water_residue_ids) > 1

    report = run_validation(context(built_dataset_with_waters), ["V19"])

    assert report.findings == []


def test_ligand_severities(built_dataset):
    """V4 findings are warnings, never errors."""
    report = run_validation(context(built_dataset, expect_ligands=True), LIGAND_CHECKS)

    actual_v4_severities = {f.severity for f in report.findings if f.check == "V4"}
    assert actual_v4_severities <= {Severity.WARNING}


def test_v9_v10_per_entry_crash_reported_under_each_requested_id(built_dataset, tmp_path):
    """A corrupt NPZ is reported as an error under whichever of V9 or V10 was requested."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    (built.output_dir / "structure_files" / "7riv" / "7riv.npz").write_bytes(b"junk")

    for check in ("V9", "V10"):
        report = run_validation(context(built), [check])
        assert _errors(report, check, "7riv"), check


def test_v10_detects_sdf_without_atom_names(built_dataset, tmp_path):
    """V10 flags a reference SDF that has no atom-name annotation to compare against."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    path = built.output_dir / "reference_mols" / "P06.sdf"

    # Strip the atom_annot_atom_name property block from the SDF.
    text = re.sub(r"> +<atom_annot_atom_name>[^\n]*\n[^\n]*\n\n", "", path.read_text())
    assert "atom_annot_atom_name" not in text
    path.write_text(text)

    report = run_validation(context(built), ["V10"])

    assert report.failing_entries == {"7riv"}

    # The message names the SDF file and says the atom names are missing.
    message = _errors(report, "V10", "7riv")[0].message
    assert "P06.sdf" in message
    assert "atom names" in message
