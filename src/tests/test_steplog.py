"""Tests for the parse/cleanup step log, using small synthetic atom arrays."""

import logging

import numpy as np
from biotite.structure import Atom, array

from aw_of3_finetune.steplog import CreatedGroup, StepLog, StepRecord, StepRecorder


def make_atoms(specs):
    """Build an AtomArray from (chain_iid, transformation_id, res_name, atom_name, x) tuples.

    Args:
        specs: One tuple per atom; x is the x coordinate (NaN marks an unresolved atom).

    Returns:
        AtomArray with chain_iid and transformation_id annotations.
    """
    atoms = [
        Atom(
            [x, 0.0, 0.0],
            chain_id=chain_iid.split("_")[0],
            res_id=1,
            res_name=res_name,
            atom_name=atom_name,
            element="C",
        )
        for chain_iid, _, res_name, atom_name, x in specs
    ]
    atom_array = array(atoms)
    atom_array.set_annotation("chain_iid", np.array([s[0] for s in specs]))
    atom_array.set_annotation("transformation_id", np.array([s[1] for s in specs]))
    return atom_array


def plain(n, res_name="ALA", chain_iid="A_1", transformation_id="1", start=0):
    return make_atoms(
        [(chain_iid, transformation_id, res_name, f"C{i}", float(i)) for i in range(start, start + n)]
    )


def test_removed_atoms_are_attributed_to_the_step_that_dropped_them():
    input_atoms = plain(5)
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("remove_waters", "drop", input_atoms[:3])
    expected_removed = [(3, 4)]
    actual_removed = recorder.log().steps[1].removed_input
    assert actual_removed == expected_removed


def test_assembly_copy_is_a_created_group_with_its_source():
    input_atoms = plain(5)
    copy = plain(3, chain_iid="A_2", transformation_id="2")
    copy.coord = input_atoms[:3].coord + np.array([10.0, 0.0, 0.0])
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("build_assembly_1", "assembly", input_atoms[:3] + copy)
    expected_created = [CreatedGroup(transformation_id="2", source_input=[(0, 2)], n_atoms=3)]
    actual_created = recorder.log().steps[1].created
    assert actual_created == expected_created


def _log_with_copy():
    input_atoms = plain(5)
    copy = plain(3, chain_iid="A_2", transformation_id="2")
    copy.coord = input_atoms[:3].coord + np.array([10.0, 0.0, 0.0])
    assembly = input_atoms[:3] + copy
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("build_assembly_1", "assembly", assembly)
    return recorder, input_atoms, assembly


def test_removing_a_copy_lands_in_removed_created():
    recorder, input_atoms, _ = _log_with_copy()
    recorder.record("remove_waters", "drop", input_atoms[:3])
    actual_step = recorder.log().steps[2]
    assert actual_step.removed_created[0].n_atoms == 3
    assert actual_step.removed_input == []


def test_rename_is_recorded_with_input_index():
    input_atoms = make_atoms(
        [("A_1", "1", "ALA", "CA", 0.0), ("A_1", "1", "MSE", "SE", 1.0)]
    )
    renamed = input_atoms.copy()
    renamed.res_name[1] = "MET"
    renamed.atom_name[1] = "SD"
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("convert_MSE_to_MET", "rename", renamed)
    expected_renamed = [(1, "MSE:SE", "MET:SD")]
    actual_renamed = recorder.log().steps[1].renamed
    assert actual_renamed == expected_renamed


def test_unresolved_atoms_are_counted_not_matched():
    input_atoms = plain(3)
    nan_atoms = plain(2, start=10)
    nan_atoms.coord[:] = np.nan
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("atomworks_parse", "parse", input_atoms + nan_atoms)
    actual_step = recorder.log().steps[1]
    assert actual_step.unresolved_added == 2
    assert actual_step.removed_input == []
    assert actual_step.created == []


def test_duplicate_coordinates_remove_highest_index_first():
    input_atoms = make_atoms(
        [
            ("A_1", "1", "ALA", "C0", 0.0),
            ("A_1", "1", "ALA", "C1", 1.0),
            ("A_1", "1", "ALA", "C2", 2.0),
            ("A_1", "1", "ALA", "C3", 2.0),
        ]
    )
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("remove_waters", "drop", input_atoms[:3])
    expected_removed = [(3, 3)]
    actual_removed = recorder.log().steps[1].removed_input
    assert actual_removed == expected_removed


def test_every_input_atom_is_kept_or_attributed():
    recorder, input_atoms, assembly = _log_with_copy()
    recorder.record("remove_waters", "drop", assembly[[0, 1, 3, 4, 5]])
    recorder.record("remove_small_polymers", "drop", input_atoms[:1])
    actual_unattributed = recorder.log().unattributed_input()
    assert actual_unattributed == []


def test_json_round_trip():
    recorder, input_atoms, _ = _log_with_copy()
    recorder.record("remove_waters", "drop", input_atoms[:3])
    expected_log = recorder.log()
    actual_log = StepLog.from_json(expected_log.to_json())
    assert actual_log == expected_log


def test_chain_dropped_with_unresolved_atoms_counts_them():
    resolved = plain(2, chain_iid="A_1")
    unresolved = plain(3, chain_iid="B_1", start=10)
    unresolved.coord[:] = np.nan
    chain_b = plain(2, chain_iid="B_1", start=5)
    input_atoms = resolved + chain_b
    with_nan = resolved + chain_b + unresolved
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("atomworks_parse", "parse", with_nan)
    recorder.record("remove_small_polymers", "drop", resolved)
    actual_step = recorder.log().steps[2]
    assert actual_step.removed_input == [(2, 3)]
    assert actual_step.unresolved_removed == 3


def _shifted_copy(atoms, chain_iid, transformation_id, shift):
    """Copy atoms as an assembly copy with x shifted; shift 0 keeps the coordinates."""
    copy = atoms.copy()
    copy.chain_iid[:] = chain_iid
    copy.transformation_id[:] = transformation_id
    copy.coord = copy.coord + np.array([shift, 0.0, 0.0])
    return copy


def test_copy_atom_on_a_special_position_is_created():
    input_atoms = plain(3)
    copy = _shifted_copy(input_atoms, "A_2", "2", 10.0)
    copy.coord[0] = input_atoms.coord[0]
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("build_assembly_1", "assembly", input_atoms + copy)

    log = recorder.log()
    expected_created = [CreatedGroup(transformation_id="2", source_input=[(0, 2)], n_atoms=3)]
    actual_step = log.steps[1]
    assert actual_step.created == expected_created
    assert actual_step.removed_input == []
    assert log.balance_errors() == []


def test_removed_original_is_not_kept_by_a_coinciding_copy():
    input_atoms = plain(3, chain_iid="B_1")
    copy = _shifted_copy(input_atoms, "B_2", "2", 10.0)
    copy.coord[0] = input_atoms.coord[0]
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("build_assembly_1", "assembly", input_atoms + copy)
    recorder.record("remove_clashing_chains", "drop B_1", copy)

    log = recorder.log()
    expected_removed = [(0, 2)]
    actual_removed = log.steps[2].removed_input
    assert actual_removed == expected_removed
    assert log.kept_input == []
    assert log.balance_errors() == []


def test_non_identity_first_copy_replaces_the_input_chain():
    input_atoms = plain(3)
    copy = _shifted_copy(input_atoms, "A_2", "2", 10.0)
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("build_assembly_1", "assembly", copy)

    log = recorder.log()
    expected_removed = [(0, 2)]
    expected_created = [CreatedGroup(transformation_id="2", source_input=[(0, 2)], n_atoms=3)]
    actual_step = log.steps[1]
    assert actual_step.removed_input == expected_removed
    assert actual_step.created == expected_created
    assert log.balance_errors() == []


def test_assembly_source_is_found_by_atom_identity_not_chain_size():
    chain_a = plain(3, chain_iid="A_1")
    chain_b = plain(3, chain_iid="B_1", start=5)
    input_atoms = chain_a + chain_b
    copy_b = _shifted_copy(chain_b, "B_2", "2", 100.0)
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    recorder.record("build_assembly_1", "assembly", input_atoms + copy_b)

    expected_created = [CreatedGroup(transformation_id="2", source_input=[(3, 5)], n_atoms=3)]
    actual_created = recorder.log().steps[1].created
    assert actual_created == expected_created


def test_balance_errors_reports_a_step_whose_counts_do_not_add_up():
    log = StepLog(
        pdb_id="1abc",
        input_atoms=3,
        steps=[
            StepRecord(name="input", phase="parse", description="", resolved_atoms=3),
            StepRecord(name="remove_waters", phase="cleanup", description="", resolved_atoms=2),
        ],
    )
    actual_errors = log.balance_errors()
    assert len(actual_errors) == 1
    assert "remove_waters" in actual_errors[0]


def test_default_recorder_keeps_no_snapshots():
    input_atoms = plain(3)
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    actual_snapshots = recorder.snapshots()
    assert actual_snapshots == []


def test_recorder_keeps_snapshots_when_asked():
    input_atoms = plain(3)
    recorder = StepRecorder(input_atoms, "1abc", keep_snapshots=True)
    recorder.record("input", "input", input_atoms)
    actual_names = [name for name, _ in recorder.snapshots()]
    assert actual_names == ["input"]


def test_failed_record_warns_and_stops_recording(caplog):
    input_atoms = plain(3)
    recorder = StepRecorder(input_atoms, "1abc")
    recorder.record("input", "input", input_atoms)
    with caplog.at_level(logging.WARNING):
        recorder.record("not_a_step", "unknown", input_atoms)
    recorder.record("remove_waters", "drop", input_atoms[:1])

    assert recorder.failed
    assert len(recorder.log().steps) == 1
    assert "not_a_step" in caplog.text


def test_record_accepts_a_function_that_fails(caplog):
    def unreadable():
        raise KeyError("pdbx_PDB_model_num")

    recorder = StepRecorder(plain(3), "1abc")
    with caplog.at_level(logging.WARNING):
        recorder.record("input", "input", unreadable)

    assert recorder.failed
    assert "pdbx_PDB_model_num" in caplog.text
