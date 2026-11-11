import json

import numpy as np
import pytest
from biotite.structure import Atom, array
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz

from aw_of3_finetune.report.findings import Status
from aw_of3_finetune.report.validity import VALIDITY_CHECKS, ValidityInputs
from aw_of3_finetune.report.validity.audit import audit_entry
from aw_of3_finetune.steplog import CreatedGroup, StepLog, StepRecord

PDB_ID = "tiny"
# Input atoms (res_name, atom_name, element, x); index = position in this list.
INPUT_ATOMS = [("ALA", "N", "N", 0.0), ("ALA", "CA", "C", 1.5), ("ALA", "C", "C", 3.0), ("ALA", "O", "O", 4.5)]
CIF_HEADER = """data_tiny
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


def final_atom(res_name, atom_name, element, x, label_asym_id="A"):
    """One final-structure atom of the given chain copy."""
    return Atom([x, 0.0, 0.0], chain_id="1", res_id=1, res_name=res_name, atom_name=atom_name,
                element=element, label_asym_id=label_asym_id)


def input_as_final(label_asym_id="A"):
    """All input atoms as final atoms of one chain copy."""
    return [final_atom(res, name, element, x, label_asym_id) for res, name, element, x in INPUT_ATOMS]


def step_log(removed=(), n_final=4):
    """A balanced log: the input step, then one step removing the given input indices."""
    ranges = [(i, i) for i in removed]
    steps = [
        StepRecord(name="input", phase="parse", description="", resolved_atoms=4),
        StepRecord(name="remove_waters", phase="cleanup", description="", resolved_atoms=n_final,
                   removed_input=ranges),
    ]
    return StepLog(pdb_id=PDB_ID, input_atoms=4, steps=steps)


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
"""


def write_run(tmp_path, final_atoms, log, translations=None):
    """Write the input CIF, the final NPZ and (unless None) the step log; return the inputs.

    translations: {operator id: x shift} written as pdbx_struct_oper_list (pure translations along x).
    """
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    rows = [
        f"ATOM {i + 1} {element} {name} . {res} A 1 1 ? {x} 0.0 0.0 1.0 10.0 1 A 1"
        for i, (res, name, element, x) in enumerate(INPUT_ATOMS)
    ]
    text = CIF_HEADER + "\n".join(rows) + "\n"
    if translations:
        opers = [f"{oper_id} 1 0 0 {shift} 0 1 0 0 0 0 1 0" for oper_id, shift in translations.items()]
        text += OPER_HEADER + "\n".join(opers) + "\n"
    (cif_dir / f"{PDB_ID}.cif").write_text(text)
    out_dir = tmp_path / "out"
    (out_dir / "of3_compressed_structures" / PDB_ID).mkdir(parents=True)
    (out_dir / "cleanup_steps").mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({"structures": [PDB_ID]}))
    write_atomarray_to_npz(array(final_atoms), out_dir / "of3_compressed_structures" / PDB_ID / f"{PDB_ID}.npz")
    if log is not None:
        (out_dir / "cleanup_steps" / f"{PDB_ID}.json").write_text(log.to_json())
    return ValidityInputs(out_dir, cif_dir, tmp_path / "ccd.cif")


def test_consistent_log_passes(tmp_path):
    inputs = write_run(tmp_path, input_as_final()[:3], step_log(removed=[3], n_final=3))

    actual = audit_entry(inputs, PDB_ID)

    assert actual.kept == [0, 1, 2]
    assert actual.removed_by_step == {"remove_waters": [3]}
    assert actual.unaccounted == []
    assert actual.double_counted == []
    assert actual.renamed_unlogged == []
    assert actual.balance_errors == []
    assert actual.has_log
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.status == Status.MATCH
    assert json.dumps(finding.data)


def test_atom_missing_from_final_and_log_is_unaccounted(tmp_path):
    inputs = write_run(tmp_path, input_as_final()[:3], step_log(n_final=4))

    actual = audit_entry(inputs, PDB_ID)

    assert actual.unaccounted == [3]
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.status == Status.UNEXPLAINED
    expected_detail = "unaccounted 1; final_count_mismatch (final 3 resolved; kept 3 + copies 0 = 3; log 4)"
    assert finding.detail == expected_detail


def test_log_claims_removal_of_kept_atom_is_double_counted(tmp_path):
    inputs = write_run(tmp_path, input_as_final(), step_log(removed=[3], n_final=3))

    actual = audit_entry(inputs, PDB_ID)

    assert actual.double_counted == [3]
    assert actual.unaccounted == []
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    expected_detail = "double_counted 1; final_count_mismatch (final 4 resolved; kept 4 + copies 0 = 4; log 3)"
    assert finding.detail == expected_detail


def test_unlogged_rename_is_reported(tmp_path):
    final = input_as_final()
    final[1] = final_atom("ALA", "CB", "C", 1.5)
    inputs = write_run(tmp_path, final, step_log())

    actual = audit_entry(inputs, PDB_ID)

    assert actual.renamed_unlogged == [(1, "ALA:CA", "ALA:CB")]
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.detail == "renamed_unlogged 1"


def test_logged_rename_is_not_reported(tmp_path):
    final = input_as_final()
    final[1] = final_atom("ALA", "CB", "C", 1.5)
    log = step_log()
    log.steps[1].renamed = [(1, "ALA:CA", "ALA:CB")]
    inputs = write_run(tmp_path, final, log)

    actual = audit_entry(inputs, PDB_ID)

    assert actual.renamed_unlogged == []


def test_unbalanced_step_is_reported(tmp_path):
    inputs = write_run(tmp_path, input_as_final(), step_log(n_final=2))

    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)

    assert finding.status == Status.UNEXPLAINED
    expected_detail = (
        "final_count_mismatch (final 4 resolved; kept 4 + copies 0 = 4; log 2); unbalanced steps: remove_waters"
    )
    assert finding.detail == expected_detail


def test_missing_log_is_reported(tmp_path):
    inputs = write_run(tmp_path, input_as_final(), None)

    actual = audit_entry(inputs, PDB_ID)

    assert not actual.has_log
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.status == Status.UNEXPLAINED
    assert finding.detail == "no step log"


def test_assembly_copy_is_not_counted_as_kept_input(tmp_path):
    # Copy A_1 is the identity copy (atoms 0-2). Copy A_2 holds an atom at input atom 3's
    # coordinates next to a moved atom, so it is a transformed copy and keeps nothing.
    copy_two = [final_atom("ALA", "O", "O", 4.5, "A_2"), final_atom("ALA", "N", "N", 100.0, "A_2")]
    final = input_as_final("A_1")[:3] + copy_two
    inputs = write_run(tmp_path, final, step_log(n_final=4))

    actual = audit_entry(inputs, PDB_ID)

    assert actual.kept == [0, 1, 2]
    assert actual.unaccounted == [3]


def test_unresolved_final_atoms_do_not_keep_input(tmp_path):
    final = input_as_final()
    final[3] = final_atom("ALA", "O", "O", np.nan)
    inputs = write_run(tmp_path, final, step_log(removed=[3], n_final=3))

    actual = audit_entry(inputs, PDB_ID)

    assert actual.kept == [0, 1, 2]
    assert actual.double_counted == []
    assert actual.unaccounted == []


def test_single_moved_atom_is_reported_not_whole_chain(tmp_path):
    final = input_as_final()
    final[2] = final_atom("ALA", "C", "C", 3.5)
    inputs = write_run(tmp_path, final, step_log())

    actual = audit_entry(inputs, PDB_ID)

    assert actual.moved == ["A:ALA1:C"]
    assert actual.kept == [0, 1, 3]
    assert actual.unaccounted == [2]
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    expected_detail = (
        "unaccounted 1; moved 1; unexplained_final 1; "
        "final_count_mismatch (final 4 resolved; kept 3 + copies 0 = 3; log 4)"
    )
    assert finding.detail == expected_detail


def test_identity_copy_is_found_when_transformed_copy_comes_first(tmp_path):
    copy_two = [final_atom("ALA", "O", "O", 4.5, "A_2"), final_atom("ALA", "N", "N", 100.0, "A_2")]
    final = copy_two + input_as_final("A_1")[:3]
    inputs = write_run(tmp_path, final, step_log(n_final=4))

    actual = audit_entry(inputs, PDB_ID)

    assert actual.kept == [0, 1, 2]
    assert actual.unaccounted == [3]
    assert actual.moved == []


def assembly_log(n_copies=1, extra_final=0):
    """Input step, then build_assembly_1 creating n_copies copies of all four input atoms (operators "2", "3", ...)."""
    created = [CreatedGroup(transformation_id=str(2 + k), source_input=[(0, 3)], n_atoms=4) for k in range(n_copies)]
    resolved = 4 + 4 * n_copies + extra_final
    steps = [
        StepRecord(name="input", phase="parse", description="", resolved_atoms=4),
        StepRecord(name="build_assembly_1", phase="assembly", description="", resolved_atoms=resolved,
                   created=created),
    ]
    return StepLog(pdb_id=PDB_ID, input_atoms=4, steps=steps)


def shifted_copy(label_asym_id, shift):
    """All input atoms shifted by shift along x, as one final chain copy."""
    return [final_atom(res, name, element, x + shift, label_asym_id) for res, name, element, x in INPUT_ATOMS]


def test_operator_copy_is_explained_without_its_atoms_being_kept(tmp_path):
    final = input_as_final("A_1") + shifted_copy("A_2", 10.0)
    inputs = write_run(tmp_path, final, assembly_log(), translations={"1": 0.0, "2": 10.0})

    actual = audit_entry(inputs, PDB_ID)

    expected_kept = [0, 1, 2, 3]
    assert actual.kept == expected_kept
    assert actual.unexplained_final == []
    assert actual.final_count_mismatch is None
    assert actual.input_count_mismatch is False
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.status == Status.MATCH, finding.detail


def test_spurious_chain_copy_is_unexplained_final(tmp_path):
    final = input_as_final("A_1") + shifted_copy("A_2", 10.0) + shifted_copy("A_3", 50.0)
    inputs = write_run(tmp_path, final, assembly_log(), translations={"1": 0.0, "2": 10.0})

    actual = audit_entry(inputs, PDB_ID)

    expected_unexplained = ["A_3:ALA1:N", "A_3:ALA1:CA", "A_3:ALA1:C", "A_3:ALA1:O"]
    assert actual.unexplained_final == expected_unexplained
    expected_mismatch = "final 12 resolved; kept 4 + copies 4 = 8; log 8"
    assert actual.final_count_mismatch == expected_mismatch
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.status == Status.UNEXPLAINED
    expected_detail = f"unexplained_final 4; final_count_mismatch ({expected_mismatch})"
    assert finding.detail == expected_detail


def test_log_with_wrong_input_atom_count_is_reported(tmp_path):
    log = step_log()
    log.input_atoms = 5
    inputs = write_run(tmp_path, input_as_final(), log)

    actual = audit_entry(inputs, PDB_ID)

    assert actual.input_count_mismatch is True
    [finding] = VALIDITY_CHECKS["validity.cleanup_audit"](inputs, PDB_ID)
    assert finding.status == Status.UNEXPLAINED
    assert finding.detail == "input_count_mismatch (log 5, input 4)"


def test_final_count_differing_from_the_log_is_reported(tmp_path):
    log = step_log()
    for step in log.steps:
        step.resolved_atoms = 5
    inputs = write_run(tmp_path, input_as_final(), log)

    actual = audit_entry(inputs, PDB_ID)

    expected_mismatch = "final 4 resolved; kept 4 + copies 0 = 4; log 5"
    assert actual.final_count_mismatch == expected_mismatch
    assert actual.unexplained_final == []
