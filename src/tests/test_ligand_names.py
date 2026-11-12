"""A ligand name outside the CCD must mean one molecule across the dataset."""

import re
import shutil
from pathlib import Path

import numpy as np
from click.testing import CliRunner

from aw_of3_finetune.cli import finetune_pipeline, finetune_prep
from aw_of3_finetune.validate import run_validation
from conftest import FIXTURES_MMCIFS
from test_validate_ligands import _duplicate_entry, _p06_rows, _rewrite_npz
from validate_helpers import context, mutable_copy

NON_CCD_NAME = "AAJ"
LAST_ATOM = "O55"  # the last atom of 7riv's ligand
LABEL_ATOM_ID_COLUMN = 3
CARTN_X_COLUMN = 10


def _renamed(source: Path, target: Path, old: str) -> Path:
    """Copy a CIF, replacing the whole-word residue name ``old`` with the non-CCD name."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(re.sub(rf"\b{old}\b", NON_CCD_NAME, source.read_text()))
    return target


def _edit_ligand_rows(path: Path, edit) -> None:
    """Rewrite each atom_site row of the non-CCD ligand; ``edit`` returns the new fields or None."""
    lines = []
    for line in path.read_text().splitlines(keepends=True):
        fields = line.split()
        if line.startswith("HETATM") and NON_CCD_NAME in fields:
            fields = edit(fields)
            if fields is None:
                continue
            line = " ".join(fields) + "\n"
        lines.append(line)
    path.write_text("".join(lines))


def _prep(cif_dir: Path, output_dir: Path, ccd_path: Path):
    return CliRunner().invoke(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path)],
    )


def _two_copies_of_7riv(cif_dir: Path) -> tuple[Path, Path]:
    """Two structures holding P06 under the non-CCD name, 7riv first in file order."""
    first = _renamed(FIXTURES_MMCIFS / "7riv.cif", cif_dir / "7riv.cif", "P06")
    second = cif_dir / "7rix.cif"
    shutil.copy(first, second)
    return first, second


def test_two_molecules_under_one_name_stop_prep_naming_both(tmp_path, ccd_path):
    cif_dir = tmp_path / "cifs"
    _renamed(FIXTURES_MMCIFS / "7riv.cif", cif_dir / "7riv.cif", "P06")
    _renamed(FIXTURES_MMCIFS / "1a3n.cif", cif_dir / "1a3n.cif", "HEM")

    result = _prep(cif_dir, tmp_path / "out", ccd_path)

    assert result.exit_code == 1
    assert NON_CCD_NAME in result.output
    assert "1a3n" in result.output
    assert "7riv" in result.output


def test_fragment_of_a_larger_ligand_stops_prep(tmp_path, ccd_path):
    """A smaller molecule with the larger one's atom names passes V10 and V11's shared-atom compare."""
    cif_dir = tmp_path / "cifs"
    _, fragment = _two_copies_of_7riv(cif_dir)
    lines = fragment.read_text().splitlines(keepends=True)
    fragment.write_text("".join(line for line in lines if not re.search(rf"\b{LAST_ATOM}\b", line)))

    result = _prep(cif_dir, tmp_path / "out", ccd_path)

    assert result.exit_code == 1
    assert NON_CCD_NAME in result.output


def test_same_molecule_with_different_atom_names_stops_prep(tmp_path, ccd_path):
    cif_dir = tmp_path / "cifs"
    _, renamed_atom = _two_copies_of_7riv(cif_dir)
    renamed_atom.write_text(re.sub(rf"\b{LAST_ATOM}\b", "X55", renamed_atom.read_text()))

    result = _prep(cif_dir, tmp_path / "out", ccd_path)

    assert result.exit_code == 1
    assert "X55" in result.output


def test_unresolved_atom_is_fine_when_the_cif_defines_the_ligand(tmp_path, ccd_path):
    """The definition restores the atom, so both copies are the same molecule."""
    cif_dir = tmp_path / "cifs"
    _, unresolved = _two_copies_of_7riv(cif_dir)
    _edit_ligand_rows(unresolved, lambda f: None if f[LABEL_ATOM_ID_COLUMN] == LAST_ATOM else f)

    result = _prep(cif_dir, tmp_path / "out", ccd_path)

    assert result.exit_code == 0, result.output


def test_unresolved_atom_stops_prep_when_the_cif_does_not_define_the_ligand(tmp_path, ccd_path):
    """Without a definition the two copies cannot be told from a fragment; the error says so."""
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    source = (FIXTURES_MMCIFS / "7riv.cif").read_text()
    first = cif_dir / "7riv.cif"
    first.write_text(source.replace(" P06 ", f" {NON_CCD_NAME} "))  # chem_comp rows keep P06
    unresolved = cif_dir / "7rix.cif"
    shutil.copy(first, unresolved)
    _edit_ligand_rows(unresolved, lambda f: None if f[LABEL_ATOM_ID_COLUMN] == LAST_ATOM else f)

    result = _prep(cif_dir, tmp_path / "out", ccd_path)

    assert result.exit_code == 1
    assert "define" in result.output


def test_same_molecule_in_a_different_pose_is_fine(tmp_path, ccd_path):
    cif_dir = tmp_path / "cifs"
    _, moved = _two_copies_of_7riv(cif_dir)

    def shift_x(fields):
        fields[CARTN_X_COLUMN] = f"{float(fields[CARTN_X_COLUMN]) + 1.5:.3f}"
        return fields

    _edit_ligand_rows(moved, shift_x)

    result = _prep(cif_dir, tmp_path / "out", ccd_path)

    assert result.exit_code == 0, result.output


def test_full_pipeline_stops_before_the_msa_stage(tmp_path, ccd_path):
    cif_dir = tmp_path / "cifs"
    _renamed(FIXTURES_MMCIFS / "7riv.cif", cif_dir / "7riv.cif", "P06")
    _renamed(FIXTURES_MMCIFS / "1a3n.cif", cif_dir / "1a3n.cif", "HEM")

    result = CliRunner().invoke(
        finetune_pipeline,
        ["--cif-dir", str(cif_dir), "--output-dir", str(tmp_path / "out"),
         "--organized-msa-dir", str(tmp_path / "msas"), "--ccd-path", str(ccd_path), "--name", "t"],
    )

    assert result.exit_code == 1
    assert "Stage 2/10" not in result.output


def _rename_ligand(arrays):
    """Give the P06 rows of an entry's NPZ the non-CCD name."""
    rows = _p06_rows(arrays)
    arrays["res_name"] = arrays["res_name"].astype("<U8")
    arrays["res_name"][rows] = NON_CCD_NAME


def _drop_last_ligand_atom(arrays):
    """Remove the last non-CCD ligand atom from an entry's NPZ."""
    n_atoms = len(arrays["atom_name"])
    dropped_atom = np.flatnonzero(arrays["res_name"] == NON_CCD_NAME)[-1]
    keep = np.arange(n_atoms) != dropped_atom
    new_index = np.cumsum(keep) - 1
    bonds = arrays["bonds"]
    bonds = bonds[keep[bonds[:, 0]] & keep[bonds[:, 1]]]
    bonds[:, 0], bonds[:, 1] = new_index[bonds[:, 0]], new_index[bonds[:, 1]]
    for key, value in list(arrays.items()):
        if key != "bonds" and getattr(value, "ndim", 0) >= 1 and len(value) == n_atoms:
            arrays[key] = value[keep]
    arrays["bonds"] = bonds


def _built_with_fragment(built_dataset, tmp_path):
    """A built dataset where 7riw holds a one-atom-smaller ligand under 7riv's non-CCD name."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _duplicate_entry(built, "7riv", "7riw")
    _rewrite_npz(built, "7riv", _rename_ligand)
    _rewrite_npz(built, "7riw", _rename_ligand)
    _rewrite_npz(built, "7riw", _drop_last_ligand_atom)
    cache_path = built.output_dir / "dataset_cache.json"
    cache_path.write_text(cache_path.read_text().replace('"P06"', f'"{NON_CCD_NAME}"'))
    return built


def test_v11_rejects_a_fragment_under_a_non_ccd_name(built_dataset, tmp_path, ccd_path):
    built = _built_with_fragment(built_dataset, tmp_path)

    report = run_validation(context(built, ccd_path=ccd_path), ["V11"])

    expected_entries = {"7riv", "7riw"}
    assert report.failing_entries == expected_entries
    assert NON_CCD_NAME in report.errors[0].message


def test_v11_keeps_the_tolerant_compare_without_a_ccd(built_dataset, tmp_path):
    built = _built_with_fragment(built_dataset, tmp_path)

    report = run_validation(context(built), ["V11"])

    assert report.errors == []
