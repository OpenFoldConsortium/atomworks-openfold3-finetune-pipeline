from pathlib import Path

import pytest
from biotite.structure.io.pdbx import CIFFile

from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.validity import VALIDITY_CHECKS, ValidityInputs
from aw_of3_finetune.report.validity.tracks import _chain_info, input_tracks
from report.test_validity_audit import PDB_ID, final_atom, input_as_final, step_log, write_run
from validate_helpers import build_dataset

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "src/tests/fixtures/mmcifs"
CHECK = "validity.chain_placement"


@pytest.fixture(scope="module")
def built(tmp_path_factory, ccd_path):
    """Real finetune-prep output of the 4x1f and 7axa fixtures, with NPZ."""
    paths = [FIXTURES / "4x1f.cif", FIXTURES / "7axa.cif"]
    return build_dataset(tmp_path_factory.mktemp("tracks"), paths, ccd_path)


def states(chain):
    """The set of residue states of a track chain."""
    return {residue[2] for residue in chain["res"]}


def test_identity_entry_places_every_residue_ok(tmp_path):
    inputs = write_run(tmp_path, input_as_final(), step_log())

    actual = input_tracks(inputs, PDB_ID)

    [ours] = actual["ours"]
    [ref] = actual["ref"]
    assert ours["res"] == [[1, "ALA", "ok"]]
    assert ours["label"] == "A"
    assert ours["paired"] == "A"
    assert ref["res"] == [[1, "ALA", "ok"]]
    assert ref["paired"] == [ours["id"]]
    [finding] = VALIDITY_CHECKS[CHECK](inputs, PDB_ID)
    assert finding.status == Status.MATCH


def test_assembly_copy_residues_are_labelled(built, ccd_path):
    inputs = ValidityInputs(built.output_dir, built.cif_dir, ccd_path)

    actual = input_tracks(inputs, "4x1f")

    by_label = {chain["label"]: chain for chain in actual["ours"]}
    assert states(by_label["A_1"]) <= {"ok", "unresolved"}
    assert "assembly_copy" in states(by_label["A_2"])
    assert states(by_label["A_2"]) <= {"assembly_copy", "unresolved"}
    [finding] = VALIDITY_CHECKS[CHECK](inputs, "4x1f")
    assert finding.status == Status.MATCH


def test_removed_chain_is_marked_removed(built, ccd_path):
    inputs = ValidityInputs(built.output_dir, built.cif_dir, ccd_path)

    actual = input_tracks(inputs, "7axa")

    water_chains = [chain for chain in actual["ref"] if {r[1] for r in chain["res"]} == {"HOH"}]
    assert water_chains
    for chain in water_chains:
        assert states(chain) == {"removed"}
        assert chain["paired"] == []


def test_shifted_residue_is_wrong_residue(tmp_path):
    shifted = [final_atom(res, name, element, x) for res, name, element, x in [("ALA", "N", "N", 0.0)]]
    shifted[0].res_id = 7
    rest = input_as_final()[1:]
    inputs = write_run(tmp_path, shifted + rest, step_log())

    actual = input_tracks(inputs, PDB_ID)

    states_by_residue = {residue[0]: residue[2] for residue in actual["ours"][0]["res"]}
    assert states_by_residue[7] == "wrong_residue"
    [finding] = VALIDITY_CHECKS[CHECK](inputs, PDB_ID)
    assert finding.status == Status.UNEXPLAINED


def test_chain_types_without_entity_poly_type_come_from_residue_names(cif_dna_complex, tmp_path):
    """OpenBind CIFs have no entity_poly.type; the protein and DNA chains are still typed."""
    stripped = tmp_path / "2q2k.cif"
    cif = CIFFile.read(cif_dna_complex)
    del cif.block["entity_poly"]["type"]
    cif.write(stripped)
    expected_types = {asym: chain["type"] for asym, chain in _chain_info(cif_dna_complex)[0].items()}

    actual_types = {asym: chain["type"] for asym, chain in _chain_info(stripped)[0].items()}

    assert actual_types == expected_types
