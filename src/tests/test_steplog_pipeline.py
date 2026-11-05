"""parse_structure and clean_structure record every step into a StepRecorder."""

import logging
from pathlib import Path

import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFFile

from assertions import write_emptiable_minimal_3aa
from aw_of3_finetune.cleanup import CleanupSettings, EmptyStructureError, clean_structure
from aw_of3_finetune import pipeline
from aw_of3_finetune.parsing import parse_structure
from aw_of3_finetune.pipeline import prepare_augmented_cif
from aw_of3_finetune.steplog import StepLog, StepRecorder, read_input_atoms

FIXTURES = Path(__file__).parent / "fixtures" / "mmcifs"

ALL_STEPS = [
    "input",
    "atomworks_parse",
    "build_assembly_1",
    "convert_MSE_to_MET",
    "remove_waters",
    "remove_crystallization_aids",
    "remove_small_polymers",
    "remove_fully_unknown_polymers",
    "remove_clashing_chains",
    "remove_non_CCD_atoms",
    "remove_chains_with_CA_gaps",
    "prefilter_bonds",
    "remove_unresolved_covalent_ligand_atoms",
    "remove_terminal_atoms",
]


@pytest.fixture(scope="module")
def ccd(ccd_path: Path) -> CIFFile:
    """The parsed CCD, read once per module."""
    return CIFFile.read(ccd_path)


def _run(cif: Path, ccd: CIFFile, recorder: StepRecorder | None):
    """Parse and clean a CIF, returning the cleaned atoms."""
    parsed = parse_structure(cif, recorder=recorder)
    atoms, _ = clean_structure(
        parsed["assemblies"]["1"][0], cif, cif.stem, ccd, recorder=recorder
    )
    return atoms


def _log(cif: Path, ccd: CIFFile) -> StepLog:
    recorder = StepRecorder(read_input_atoms(cif), cif.stem)
    _run(cif, ccd, recorder)
    return recorder.log()


def _step(log: StepLog, name: str):
    return next(s for s in log.steps if s.name == name)


def _count(ranges: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def test_7axa_step_counts(ccd: CIFFile):
    """Step order and the per-step counts measured by the prototype."""
    log = _log(FIXTURES / "7axa.cif", ccd)

    actual_names = [s.name for s in log.steps]
    assert actual_names == ALL_STEPS
    # 76 atoms of unchosen alternate conformers, dropped by atomworks' altloc choice (first letter per chain).
    assert _count(_step(log, "atomworks_parse").removed_input) == 76
    assert _count(_step(log, "remove_waters").removed_input) == 170
    assert len(_step(log, "atomworks_parse").renamed) == 2
    assert log.unattributed_input() == []
    assert log.balance_errors() == []


def test_highest_occupancy_altloc_is_its_own_step(ccd: CIFFile):
    """With altloc="highest_occupancy" the conformer choice is logged before the parse."""
    cif = FIXTURES / "7axa.cif"
    settings = CleanupSettings(altloc="highest_occupancy")
    recorder = StepRecorder(read_input_atoms(cif), cif.stem)
    parsed = parse_structure(cif, settings, recorder)
    clean_structure(parsed["assemblies"]["1"][0], cif, cif.stem, ccd, settings, recorder=recorder)
    log = recorder.log()

    expected_names = ["input", "altloc_highest_occupancy", "atomworks_parse"]
    actual_names = [s.name for s in log.steps[:3]]
    assert actual_names == expected_names
    assert _count(_step(log, "altloc_highest_occupancy").removed_input) == 76
    assert log.balance_errors() == []


def test_4x1f_assembly_copy_and_water_copies(ccd: CIFFile):
    """The assembly copy's atoms are created, and removing waters removes their copies too."""
    log = _log(FIXTURES / "4x1f.cif", ccd)

    actual_created = sum(g.n_atoms for g in _step(log, "build_assembly_1").created)
    waters = _step(log, "remove_waters")
    actual_removed_created = sum(g.n_atoms for g in waters.removed_created)
    assert actual_created == 2342
    assert _count(waters.removed_input) == 170
    assert actual_removed_created == 170
    assert log.balance_errors() == []


def test_recorder_none_changes_nothing(ccd: CIFFile):
    """The cleaned array is the same with and without a recorder."""
    cif = FIXTURES / "7axa.cif"
    expected = _run(cif, ccd, None)

    actual = _run(cif, ccd, StepRecorder(read_input_atoms(cif), "7axa"))

    assert np.array_equal(actual.coord, expected.coord, equal_nan=True)
    assert np.array_equal(actual.atom_name, expected.atom_name)
    assert np.array_equal(actual.res_name, expected.res_name)


def test_steps_that_change_nothing_are_still_logged(ccd: CIFFile):
    """An X-ray entry logs all 14 steps; a step that leaves the atoms alone records no change."""
    log = _log(FIXTURES / "3u8v.cif", ccd)

    assert [s.name for s in log.steps] == ALL_STEPS
    no_op_steps = [
        step
        for previous, step in zip(log.steps, log.steps[1:])
        if step.resolved_atoms == previous.resolved_atoms and not step.renamed
    ]
    assert no_op_steps != []
    for step in no_op_steps:
        assert step.removed_input == []
        assert step.created == []
        assert step.removed_created == []
    assert log.unattributed_input() == []
    assert log.balance_errors() == []


def test_emptied_structure_still_logs_the_step(tmp_path: Path, ccd: CIFFile):
    """The step that empties the structure is the last one recorded."""
    cif = write_emptiable_minimal_3aa(tmp_path)
    recorder = StepRecorder(read_input_atoms(cif), "minimal_3aa")

    with pytest.raises(EmptyStructureError) as excinfo:
        _run(cif, ccd, recorder)

    expected_name = excinfo.value.removals[-1].step
    actual_name = recorder.log().steps[-1].name
    assert actual_name == expected_name


def test_unwritable_log_dir_does_not_fail_prepare(tmp_path: Path, caplog):
    """A step log that cannot be written logs a warning; the CIF is still prepared."""
    not_a_dir = tmp_path / "cleanup_steps"
    not_a_dir.write_text("a file, not a directory\n")
    out_cif = tmp_path / "7riv.cif"

    with caplog.at_level(logging.WARNING):
        prepare_augmented_cif(FIXTURES / "7riv.cif", "7riv", out_cif, step_log_dir=not_a_dir)

    assert out_cif.exists()
    assert "step log" in caplog.text


def test_unwritable_log_dir_does_not_mask_empty_structure(tmp_path: Path, ccd: CIFFile):
    """EmptyStructureError still propagates when writing the step log fails."""
    cif = write_emptiable_minimal_3aa(tmp_path)
    not_a_dir = tmp_path / "cleanup_steps"
    not_a_dir.write_text("a file, not a directory\n")

    with pytest.raises(EmptyStructureError):
        prepare_augmented_cif(cif, "minimal_3aa", tmp_path / "out.cif", step_log_dir=not_a_dir)


def test_prepare_keeps_snapshots_only_for_step_cifs(tmp_path: Path, monkeypatch):
    """prepare_augmented_cif asks the recorder for snapshots only with write_step_cifs."""
    seen: list[bool] = []
    real_recorder = pipeline.StepRecorder

    def spy(*args, **kwargs):
        seen.append(kwargs.get("keep_snapshots", False))
        return real_recorder(*args, **kwargs)

    monkeypatch.setattr(pipeline, "StepRecorder", spy)
    cif = FIXTURES / "1UBQ.cif"
    prepare_augmented_cif(cif, "1ubq", tmp_path / "a.cif", step_log_dir=tmp_path / "logs")
    prepare_augmented_cif(
        cif, "1ubq", tmp_path / "b.cif", step_log_dir=tmp_path / "logs", write_step_cifs=True
    )

    expected_seen = [False, True]
    assert seen == expected_seen
