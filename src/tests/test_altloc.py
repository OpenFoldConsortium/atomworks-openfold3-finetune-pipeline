"""Alternate-location (altloc) choice.

Default "first": atomworks' default, the first letter per chain (cheapest). Option
"clash_aware": atomworks' altloc="random_clash_aware" with a fixed seed, one conformer per
residue picked so the kept conformers do not clash. Option "highest_occupancy" reproduces
OpenFold3 (biotite's altloc="occupancy": per residue, the lettered conformer with the highest
summed occupancy, ties to the first letter), by filtering the source CIF before atomworks parses it.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
import atomworks.io as aw_io
from atomworks.io.config import ParseConfig
from biotite.structure.io.pdbx import CIFFile

from aw_of3_finetune.altloc import select_highest_occupancy_altlocs
from aw_of3_finetune.cleanup import CleanupSettings
import aw_of3_finetune.parsing
from aw_of3_finetune.parsing import ALTLOC_SEED, STANDARD_PARSE_KWARGS, parse_structure

FIXTURES = Path(__file__).parent / "fixtures" / "mmcifs"


def _residue_rows(cif: CIFFile, comp: str, label_seq_id: str) -> dict[str, np.ndarray]:
    """atom_site columns for one residue (all chains), keyed by column name."""
    site = cif.block["atom_site"]
    comps = site["label_comp_id"].as_array(str)
    seq_ids = site["label_seq_id"].as_array(str)
    rows = (comps == comp) & (seq_ids == label_seq_id)
    return {
        "alt": site["label_alt_id"].as_array(str)[rows],
        "occupancy": site["occupancy"].as_array(float)[rows],
        "atom": site["label_atom_id"].as_array(str)[rows],
    }


def test_keeps_the_higher_occupancy_conformer():
    """6hj2 SER 49 (label numbering; author 167) has conformers A (0.44) and B (0.56):
    only B's atoms remain, unlabelled."""
    cif = CIFFile.read(FIXTURES / "6hj2.cif")
    before = _residue_rows(cif, "SER", "49")
    expected_occupancies = sorted(before["occupancy"][(before["alt"] == "B") | (before["alt"] == ".")])

    n_residues = select_highest_occupancy_altlocs(cif)

    after = _residue_rows(cif, "SER", "49")
    # Only B's rows (and atoms that had no altloc) are left ...
    assert sorted(after["occupancy"]) == expected_occupancies
    # ... and no row carries an altloc letter any more.
    assert set(after["alt"]) == {"."}
    # 6hj2 has five residues with alternate conformers.
    assert n_residues == 5


def test_tie_goes_to_the_first_letter(tmp_path):
    """With equal summed occupancy, conformer A is kept (biotite's rule)."""
    src = (FIXTURES / "6hj2.cif").read_text()
    cif_path = tmp_path / "6hj2.cif"
    # Make SER 49's two conformers equally occupied.
    cif_path.write_text(src.replace(" 0.44 ", " 0.50 ").replace(" 0.56 ", " 0.50 "))
    cif = CIFFile.read(cif_path)
    before = _residue_rows(cif, "SER", "49")
    expected_occupancies = sorted(before["occupancy"][(before["alt"] == "A") | (before["alt"] == ".")])
    a_atom_count = int((before["alt"] == "A").sum())

    select_highest_occupancy_altlocs(cif)

    after = _residue_rows(cif, "SER", "49")
    assert sorted(after["occupancy"]) == expected_occupancies
    assert int((before["alt"] == ".").sum()) + a_atom_count == len(after["atom"])


def test_structure_without_altlocs_is_unchanged():
    """2crb has no alternate conformers: nothing is filtered."""
    cif = CIFFile.read(FIXTURES / "2crb.cif")
    n_atoms_before = len(cif.block["atom_site"]["id"].as_array(str))

    n_residues = select_highest_occupancy_altlocs(cif)

    n_atoms_after = len(cif.block["atom_site"]["id"].as_array(str))
    assert n_residues == 0
    assert n_atoms_after == n_atoms_before


def test_highest_occupancy_keeps_conformer_b():
    """altloc="highest_occupancy" picks SER 49's B conformer (highest occupancy), not atomworks' A."""
    raw = CIFFile.read(FIXTURES / "6hj2.cif")
    b_rows = _residue_rows(raw, "SER", "49")
    site = raw.block["atom_site"]
    is_ser49_b = (
        (site["label_comp_id"].as_array(str) == "SER")
        & (site["label_seq_id"].as_array(str) == "49")
        & (site["label_alt_id"].as_array(str) == "B")
    )
    og_b = site["Cartn_x"].as_array(float)[is_ser49_b & (site["label_atom_id"].as_array(str) == "OG")][0]
    assert "B" in b_rows["alt"]

    arr = parse_structure(FIXTURES / "6hj2.cif", CleanupSettings(altloc="highest_occupancy"))["assemblies"]["1"][0]

    actual_og_x = arr.coord[(arr.res_name == "SER") & (arr.res_id == 49) & (arr.atom_name == "OG")][0][0]
    assert actual_og_x == pytest.approx(og_b, abs=1e-3)


def test_first_is_atomworks_first_letter():
    """altloc="first" is atomworks' default choice, the first letter per chain: conformer A."""
    site = CIFFile.read(FIXTURES / "6hj2.cif").block["atom_site"]
    is_ser49_og_a = (
        (site["label_comp_id"].as_array(str) == "SER")
        & (site["label_seq_id"].as_array(str) == "49")
        & (site["label_alt_id"].as_array(str) == "A")
        & (site["label_atom_id"].as_array(str) == "OG")
    )
    og_a = site["Cartn_x"].as_array(float)[is_ser49_og_a][0]

    settings = CleanupSettings(altloc="first")
    arr = parse_structure(FIXTURES / "6hj2.cif", settings)["assemblies"]["1"][0]

    actual_og_x = arr.coord[(arr.res_name == "SER") & (arr.res_id == 49) & (arr.atom_name == "OG")][0][0]
    assert actual_og_x == pytest.approx(og_a, abs=1e-3)


def test_choice_is_logged(caplog: pytest.LogCaptureFixture):
    """One line per structure says how many residues had a conformer chosen."""
    with caplog.at_level(logging.INFO, logger="aw_of3_finetune.parsing"):
        parse_structure(FIXTURES / "6hj2.cif", CleanupSettings(altloc="highest_occupancy"))

    expected = "6hj2: altloc kept the highest-occupancy conformer for 5 residue(s)"
    assert expected in caplog.messages


def _coords(arr) -> np.ndarray:
    return np.nan_to_num(arr.coord, nan=1e9)


def test_default_is_atomworks_first_letter_choice():
    """By default the parse equals atomworks' own parse with altloc="first"."""
    config = ParseConfig(**STANDARD_PARSE_KWARGS, altloc="first")
    expected_arr = aw_io.parse(str(FIXTURES / "6hj2.cif"), config=config)["assemblies"]["1"][0]

    actual_arr = parse_structure(FIXTURES / "6hj2.cif")["assemblies"]["1"][0]

    assert CleanupSettings().altloc == "first"
    np.testing.assert_array_equal(_coords(actual_arr), _coords(expected_arr))


def test_clash_aware_is_atomworks_seeded_choice():
    """altloc="clash_aware" equals atomworks' random_clash_aware parse with the fixed seed."""
    config = ParseConfig(**STANDARD_PARSE_KWARGS, altloc="random_clash_aware", altloc_seed=ALTLOC_SEED)
    expected_arr = aw_io.parse(str(FIXTURES / "6hj2.cif"), config=config)["assemblies"]["1"][0]

    actual_arr = parse_structure(FIXTURES / "6hj2.cif", CleanupSettings(altloc="clash_aware"))["assemblies"]["1"][0]

    np.testing.assert_array_equal(_coords(actual_arr), _coords(expected_arr))


def test_clash_aware_choice_is_deterministic():
    """Two parses of the same CIF choose the same conformers (the seed is fixed)."""
    settings = CleanupSettings(altloc="clash_aware")
    first_arr = parse_structure(FIXTURES / "7axa.cif", settings)["assemblies"]["1"][0]
    second_arr = parse_structure(FIXTURES / "7axa.cif", settings)["assemblies"]["1"][0]

    np.testing.assert_array_equal(_coords(first_arr), _coords(second_arr))


def test_clash_aware_failure_falls_back_to_first(monkeypatch, caplog: pytest.LogCaptureFixture):
    """When atomworks finds no clash-free choice it raises; the parse then uses the default, first."""
    real_parse = aw_io.parse

    def parse_raising_on_clash_aware(source, config):
        if config.altloc == "random_clash_aware":
            raise ValueError("No fully clash-free altloc assignment found for 2 residues.")
        return real_parse(source, config=config)

    monkeypatch.setattr(aw_of3_finetune.parsing.aw_io, "parse", parse_raising_on_clash_aware)
    expected_arr = parse_structure(FIXTURES / "6hj2.cif", CleanupSettings(altloc="first"))["assemblies"]["1"][0]

    with caplog.at_level(logging.WARNING, logger="aw_of3_finetune.parsing"):
        actual_arr = parse_structure(FIXTURES / "6hj2.cif", CleanupSettings(altloc="clash_aware"))["assemblies"]["1"][0]

    np.testing.assert_array_equal(_coords(actual_arr), _coords(expected_arr))
    expected_message = (
        "6hj2: clash-aware altloc choice failed (No fully clash-free altloc assignment found for 2 residues.); "
        "used the first altloc letter per chain instead"
    )
    assert expected_message in caplog.messages
