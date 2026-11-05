import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from biotite.structure.io import pdbx

from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.report.validity.steps3d import step_view_data
from aw_of3_finetune.report.validity.tracks import check_chain_placement, input_tracks
from aw_of3_finetune.steplog import CreatedGroup, StepLog, StepRecord, read_input_atoms
from report.test_validity_audit import PDB_ID, assembly_log, final_atom, input_as_final, shifted_copy, write_run
from validate_helpers import build_dataset

REPO = Path(__file__).resolve().parents[3]


def multi_phase_log(removed=()):
    """Input atoms in four phases; the cleanup step removes the given input indices."""
    ranges = [(i, i) for i in removed]
    n_final = 4 - len(removed)
    steps = [
        StepRecord(name="input", phase="parse", description="", resolved_atoms=4),
        StepRecord(name="atomworks_parse", phase="parse", description="", resolved_atoms=4),
        StepRecord(name="build_assembly_1", phase="assembly", description="", resolved_atoms=4),
        StepRecord(name="remove_waters", phase="cleanup", description="water", resolved_atoms=n_final,
                   removed_input=ranges),
        StepRecord(name="remove_terminal_atoms", phase="final", description="", resolved_atoms=n_final),
    ]
    return StepLog(pdb_id=PDB_ID, input_atoms=4, steps=steps)


@pytest.fixture(scope="module")
def built_4x1f(tmp_path_factory, ccd_path):
    """Real finetune-prep output of the 4x1f fixture, with NPZ."""
    return build_dataset(tmp_path_factory.mktemp("steps3d"), [REPO / "src/tests/fixtures/mmcifs/4x1f.cif"], ccd_path)


def test_phases_group_steps_as_in_prototype(tmp_path):
    inputs = write_run(tmp_path, input_as_final(), multi_phase_log())

    actual = step_view_data(inputs, PDB_ID)

    expected_phases = [
        {"name": "Parse", "steps": ["input", "atomworks_parse"]},
        {"name": "Assembly", "steps": ["build_assembly_1"]},
        {"name": "Cleanup filters", "steps": ["remove_waters"]},
        {"name": "Final trim", "steps": ["remove_terminal_atoms"]},
    ]
    assert actual["phases"] == expected_phases
    assert [step["name"] for step in actual["steps"]] == [s for p in expected_phases for s in p["steps"]]
    assert actual["steps"][3]["phase"] == "Cleanup filters"
    assert actual["steps"][3]["description"] == "water"
    assert actual["before_after_only"] is False
    assert json.dumps(actual)


def test_removed_at_marks_later_removals(tmp_path):
    inputs = write_run(tmp_path, input_as_final()[:3], multi_phase_log(removed=[3]))

    actual = step_view_data(inputs, PDB_ID)["input"]

    assert actual["removed_at"] == [-1, -1, -1, 3]
    assert "will_be_removed" not in actual
    assert actual["chain"] == [0, 0, 0, -1]
    assert actual["atom_name"] == ["N", "CA", "C", "O"]
    assert actual["xyz"] == [0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0, 0.0, 4.5, 0.0, 0.0]


def test_chains_list_final_chains_in_order(tmp_path):
    inputs = write_run(tmp_path, input_as_final(), multi_phase_log())

    actual = step_view_data(inputs, PDB_ID)["chains"]

    assert actual == [{"id": "A", "res_name": "ALA", "hetero": False}]


def test_no_log_gives_before_after_only(tmp_path):
    inputs = write_run(tmp_path, input_as_final()[:3], None)

    actual = step_view_data(inputs, PDB_ID)

    assert actual["before_after_only"] is True
    assert actual["steps"] is None
    assert actual["phases"] == []
    assert actual["input"]["chain"] == [0, 0, 0, -1]
    assert actual["input"]["removed_at"] == [-1, -1, -1, -1]
    assert actual["created"]["created_at"] == []


def test_created_then_removed_copy_has_coordinates(built_4x1f, ccd_path):
    inputs = ValidityInputs(built_4x1f.output_dir, built_4x1f.cif_dir, ccd_path)

    actual = step_view_data(inputs, "4x1f")

    steps = [step["name"] for step in actual["steps"]]
    waters_step = steps.index("remove_waters")
    created = actual["created"]
    in_waters = [i for i, removed in enumerate(created["removed_at"]) if removed == waters_step]
    assert len(in_waters) == 170
    block = pdbx.CIFFile.read(built_4x1f.cif_dir / "4x1f.cif").block
    oper = block["pdbx_struct_oper_list"]
    row = oper["id"].as_array(str).tolist().index("2")
    matrix = np.array([[oper[f"matrix[{i}][{j}]"].as_array(float)[row] for j in (1, 2, 3)] for i in (1, 2, 3)])
    vector = np.array([oper[f"vector[{i}]"].as_array(float)[row] for i in (1, 2, 3)])
    source = read_input_atoms(built_4x1f.cif_dir / "4x1f.cif").coord[2196:2366]
    expected_xyz = source @ matrix.T + vector
    actual_xyz = np.array(created["xyz"]).reshape(-1, 3)[in_waters]
    assert np.abs(actual_xyz - expected_xyz).max() <= 0.01
    assert set(created["res_name"][i] for i in in_waters) == {"HOH"}
    assert all(created["chain"][i] == -1 for i in in_waters)


def test_large_entry_file_size(tmp_path, ccd_path):
    from aw_of3_finetune.report.validity import ValidityInputs
    built = build_dataset(tmp_path, [REPO / "src/tests/fixtures/mmcifs/1a3n.cif"], ccd_path)
    inputs = ValidityInputs(built.output_dir, built.cif_dir, ccd_path)

    actual = len(json.dumps(step_view_data(inputs, "1a3n"), separators=(",", ":")))

    assert actual < 1_000_000


def test_input_columns_carry_label_asym_id_and_renamed_names(tmp_path):
    final = input_as_final()
    final[1] = final_atom("ALA", "CB", "C", 1.5)
    log = multi_phase_log()
    log.steps[3].renamed = [(1, "ALA:CA", "ALA:CB")]
    inputs = write_run(tmp_path, final, log)

    actual = step_view_data(inputs, PDB_ID)["input"]

    assert actual["label_asym_id"] == ["A", "A", "A", "A"]
    expected_atom_names = ["N", "CB", "C", "O"]
    assert actual["renamed_atom_name"] == expected_atom_names
    assert actual["renamed_res_name"] == ["ALA"] * 4


def test_copy_atoms_without_a_log_are_created_rows(tmp_path):
    final = input_as_final("A_1") + shifted_copy("A_2", 10.0)
    inputs = write_run(tmp_path, final, None, translations={"1": 0.0, "2": 10.0})

    actual = step_view_data(inputs, PDB_ID)

    created = actual["created"]
    expected_chain = [1, 1, 1, 1]
    assert created["chain"] == expected_chain
    assert created["created_at"] == [None] * 4
    assert created["removed_at"] == [-1] * 4
    assert created["atom_name"] == ["N", "CA", "C", "O"]
    assert created["xyz"][:3] == [10.0, 0.0, 0.0]


def test_unlogged_final_atom_is_created_at_build_assembly(tmp_path):
    final = input_as_final("A_1") + shifted_copy("A_2", 10.0) + [final_atom("ALA", "N", "N", 77.0, "A_3")]
    inputs = write_run(tmp_path, final, assembly_log(extra_final=1), translations={"1": 0.0, "2": 10.0})

    actual = step_view_data(inputs, PDB_ID)["created"]

    expected_build_step = 1
    assert actual["created_at"] == [expected_build_step] * 5
    assert actual["chain"] == [1, 1, 1, 1, 2]
    assert actual["xyz"][-3:] == [77.0, 0.0, 0.0]


def test_skipped_groups_are_recorded(tmp_path):
    log = assembly_log()
    log.steps[1].created[0].transformation_id = "9"
    orphan = CreatedGroup(transformation_id="7", source_input=[(0, 1)], n_atoms=2)
    log.steps.append(StepRecord(name="remove_waters", phase="cleanup", description="", resolved_atoms=6,
                                removed_created=[orphan]))
    inputs = write_run(tmp_path, input_as_final("A_1") + shifted_copy("A_2", 10.0), log,
                       translations={"1": 0.0, "2": 10.0})

    actual = step_view_data(inputs, PDB_ID)["skipped_groups"]

    expected_skipped = [
        {"transformation_id": "7", "reason": "removed_created group has no created group"},
        {"transformation_id": "9", "reason": "no pdbx_struct_oper_list operator"},
    ]
    assert sorted(actual, key=lambda group: group["transformation_id"]) == expected_skipped


@pytest.fixture(scope="module")
def built_4x1f_without_log(built_4x1f, tmp_path_factory, ccd_path):
    """A copy of the 4x1f output with its cleanup step log deleted."""
    output_dir = tmp_path_factory.mktemp("nolog") / "output"
    shutil.copytree(built_4x1f.output_dir, output_dir)
    (output_dir / "cleanup_steps" / "4x1f.json").unlink()
    return ValidityInputs(output_dir, built_4x1f.cif_dir, ccd_path)


def test_4x1f_without_log_still_draws_copies_and_places_chains(built_4x1f_without_log):
    inputs = built_4x1f_without_log

    actual = step_view_data(inputs, "4x1f")

    copy_chains = [index for index, chain in enumerate(actual["chains"]) if not chain["id"].endswith("_1")]
    actual_copy_rows = [chain for chain in actual["created"]["chain"] if chain in copy_chains]
    final = inputs.final_atoms("4x1f")
    in_copies = ~np.char.endswith(final.label_asym_id.astype(str), "_1") & ~np.isnan(final.coord).any(axis=1)
    expected_copy_rows = int(in_copies.sum())
    assert expected_copy_rows > 0
    assert len(actual_copy_rows) == expected_copy_rows
    assert set(actual["created"]["created_at"]) == {None}
    tracks = input_tracks(inputs, "4x1f")
    by_label = {chain["label"]: chain for chain in tracks["ours"]}
    assert "assembly_copy" in {residue[2] for residue in by_label["A_2"]["res"]}
    [finding] = check_chain_placement(inputs, "4x1f")
    assert finding.status == Status.MATCH, finding.detail
