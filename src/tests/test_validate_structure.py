"""Tests for the structure checks V1-V3 and V5-V7."""

import json

import numpy as np
from biotite.structure.io.pdbx import CIFCategory, CIFColumn, CIFFile

from aw_of3_finetune.validate import CHECKS, run_validation
from aw_of3_finetune.validate.findings import Severity
from validate_helpers import context, mutable_copy

STRUCTURE_CHECKS = ["V1", "V2", "V3", "V5", "V6", "V7"]


def _rewrite_npz(built, entry, edit):
    """Load an entry's NPZ arrays, apply `edit(arrays)` in place, and save them back."""
    path = built.output_dir / "of3_compressed_structures" / entry / f"{entry}.npz"
    with np.load(path, allow_pickle=True) as data:
        arrays = {name: data[name] for name in data.files}
    edit(arrays)
    np.savez(path, **arrays)


def _errors(report, check, entry):
    """The error findings of one check on one entry."""
    return [f for f in report.errors if f.check == check and f.entry == entry]


def test_structure_checks_are_registered():
    """Every structure check id is present in the CHECKS registry."""
    registered = {check for check in STRUCTURE_CHECKS if check in CHECKS}
    assert registered == set(STRUCTURE_CHECKS)


def test_structure_checks_pass_on_clean_build(built_dataset):
    """A clean build has no errors; the only V1 finding is 2q2k's dropped crystallization aid."""
    report = run_validation(context(built_dataset), STRUCTURE_CHECKS)

    # No check reports an error on the untouched dataset.
    assert report.errors == [], [f.message for f in report.errors]

    # 2q2k lost its HEPES to cleanup, which V1 records as INFO.
    v1_findings = [f for f in report.findings if f.check == "V1" and f.entry == "2q2k"]
    actual_v1 = [(f.severity, f.detail["reason"]) for f in v1_findings]
    assert actual_v1 == [(Severity.INFO, "crystallization_aid")]


def test_v2_detects_sequence_change(built_dataset, tmp_path):
    """V2 flags an entry whose cleaned CIF has a different sequence than the source."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    path = built.output_dir / "cleaned_structures" / "3u8v.cif"
    text = path.read_text()
    assert text.count("SGHTAHV") == 1

    # Mutate one residue in the cleaned CIF (S -> A).
    path.write_text(text.replace("SGHTAHV", "AGHTAHV"))

    report = run_validation(context(built), ["V2"])

    assert report.failing_entries == {"3u8v"}


def test_v3_detects_moved_atom(built_dataset, tmp_path):
    """V3 flags an atom moved by 0.01 A in the NPZ and reports which atom."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def move(arrays):
        arrays["coord"][5, 0] += 0.01

    _rewrite_npz(built, "3u8v", move)

    report = run_validation(context(built), ["V3"])

    assert report.failing_entries == {"3u8v"}

    # The error detail identifies the offending atom.
    assert "atom" in _errors(report, "V3", "3u8v")[0].detail


def test_v3_accepts_symmetry_copy(built_dataset):
    """2q2k's bioassembly copies are accepted and noted as symmetry copies."""
    report = run_validation(context(built_dataset), ["V3"])

    # No V3 error on 2q2k.
    assert _errors(report, "V3", "2q2k") == []

    # A finding mentions the symmetry copy.
    messages_2q2k = [f.message for f in report.findings if f.entry == "2q2k"]
    assert any("symmetry copy" in message for message in messages_2q2k)


def test_v3_reports_atom_name_swap_as_info(built_dataset):
    """Arg 86 in 3u8v has swapped symmetric atom names; V3 reports it once, as INFO."""
    report = run_validation(context(built_dataset), ["V3"])

    swaps = [f for f in report.findings if f.message == "expected (atom-name swap)"]
    arg86_swaps = [f for f in swaps if f.entry == "3u8v" and f.detail["res_id"] == 86]

    assert len(arg86_swaps) == 1
    assert arg86_swaps[0].severity is Severity.INFO
    assert arg86_swaps[0].detail == {"chain": "A", "res_id": 86, "res_name": "ARG"}


def test_v3_rejects_non_partner_name_swap(built_dataset, tmp_path):
    """Renaming PHE CD1 to CE2 is not a symmetric partner swap, so V3 fails the entry."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def rename(arrays):
        phe_cd1 = (arrays["res_name"] == "PHE") & (arrays["atom_name"] == "CD1")
        arrays["atom_name"][int(np.flatnonzero(phe_cd1)[0])] = "CE2"

    _rewrite_npz(built, "7riv", rename)

    report = run_validation(context(built), ["V3"])

    assert report.failing_entries == {"7riv"}


def test_v3_rejects_moved_atom_in_symmetry_copy(built_dataset, tmp_path):
    """Moving one atom in copy B_2 of 2q2k breaks the symmetry, so V3 fails the entry."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def move(arrays):
        resolved = ~np.isnan(arrays["coord"]).any(axis=1)
        copy_b2_resolved = np.flatnonzero((arrays["label_asym_id"] == "B_2") & resolved)
        arrays["coord"][int(copy_b2_resolved[10]), 0] += 0.01

    _rewrite_npz(built, "2q2k", move)

    report = run_validation(context(built), ["V3"])

    assert report.failing_entries == {"2q2k"}


def test_v3_accepts_any_altloc(built_dataset, tmp_path):
    """With two altlocs (A, B) in the source CIF, V3 accepts coordinates matching either one."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    path = built.cif_dir / "3u8v.cif"
    cif = CIFFile.read(path)
    site = cif.block["atom_site"]
    columns = {name: list(site[name].as_array(str)) for name in site.keys()}

    # Duplicate the first ATOM row as altloc B, and shift altloc A's x by 0.5 A.
    atom_row = next(i for i, group in enumerate(columns["group_PDB"]) if group == "ATOM")
    for values in columns.values():
        values.append(values[atom_row])
    columns["label_alt_id"][-1] = "B"
    columns["label_alt_id"][atom_row] = "A"
    columns["Cartn_x"][atom_row] = f"{float(columns['Cartn_x'][atom_row]) + 0.5:.3f}"
    cif.block["atom_site"] = CIFCategory(
        {name: CIFColumn(np.array(values)) for name, values in columns.items()}
    )
    cif.write(path)

    report = run_validation(context(built), ["V3"])

    assert _errors(report, "V3", "3u8v") == [], [f.message for f in report.errors]


def test_v5_allows_unresolved_atoms(built_dataset):
    """Unresolved atoms carry NaN coordinates by design (OF3 cleanup); V5 accepts them."""
    coord = context(built_dataset).npz("3u8v")["coord"]
    assert np.isnan(coord).any(), "precondition: 3u8v has unresolved atoms"

    report = run_validation(context(built_dataset), ["V5"])

    assert report.errors == []


def test_v5_detects_inf(built_dataset, tmp_path):
    """V5 flags an infinite coordinate."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def poison(arrays):
        arrays["coord"][0, 0] = np.inf

    _rewrite_npz(built, "3u8v", poison)

    report = run_validation(context(built), ["V5"])

    assert report.failing_entries == {"3u8v"}


def test_v5_detects_chain_without_resolved_atoms(built_dataset, tmp_path):
    """A chain whose every atom is NaN can't be trained on; cleanup should have dropped it."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def blank_first_chain(arrays):
        first = arrays["chain_id"] == arrays["chain_id"][0]
        arrays["coord"][first] = np.nan

    _rewrite_npz(built, "3u8v", blank_first_chain)

    report = run_validation(context(built), ["V5"])

    # Exactly one V5 error, and it says why.
    (error,) = _errors(report, "V5", "3u8v")
    assert "no resolved atoms" in error.message


def test_v6_detects_metadata_mismatch(built_dataset, tmp_path):
    """V6 flags an entry whose cache metadata is missing a chain that the NPZ has."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    path = built.output_dir / "dataset_cache.json"
    data = json.loads(path.read_text())

    # Remove chain "2" of 3u8v from the cache metadata only.
    del data["structure_data"]["3u8v"]["chains"]["2"]
    path.write_text(json.dumps(data, indent=4))

    report = run_validation(context(built), ["V6"])

    assert report.failing_entries == {"3u8v"}


def test_v7_detects_author_numbering(built_dataset, tmp_path):
    """V7 flags residue ids shifted off the label numbering (as author numbering would be)."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def shift(arrays):
        arrays["res_id"] = np.where(
            arrays["chain_id"] == "1", arrays["res_id"] + 141, arrays["res_id"]
        )

    _rewrite_npz(built, "3u8v", shift)

    report = run_validation(context(built), ["V7"])

    assert report.failing_entries == {"3u8v"}
