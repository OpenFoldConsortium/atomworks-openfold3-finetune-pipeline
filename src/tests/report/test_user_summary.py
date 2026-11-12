import json
import os

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.constants import TEMPLATE_CUTOFF_NOTE
from aw_of3_finetune.report.summary import LARGE_DATASET, TABLE_ROW_CAP, user_summary
from aw_of3_finetune.report.validity import ValidityInputs, entry_missing_findings
from aw_of3_finetune.steplog import StepLog, StepRecord

PREP_REASON = "dropped by finetune-prep; see its log"


def make_inputs(tmp_path, entries, validation=None, cifs=(), step_logs=None):
    """Inputs over a manifest of entries, optional validation.json, input CIF stems and step logs."""
    (tmp_path / "manifest.json").write_text(json.dumps({"structures": list(entries)}))
    if validation is not None:
        (tmp_path / "validation.json").write_text(json.dumps(validation))
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    for stem in cifs:
        (cif_dir / f"{stem}.cif").write_text("")
    (tmp_path / "cleanup_steps").mkdir()
    for pdb_id, log in (step_logs or {}).items():
        (tmp_path / "cleanup_steps" / f"{pdb_id}.json").write_text(log.to_json())
    return ValidityInputs(tmp_path, cif_dir, tmp_path / "ccd.cif")


def origin_finding(entry, origin, detail="detail"):
    status = Status.UNEXPLAINED if origin == "processing" else Status.EXPECTED
    return Finding("validity.bond_lengths", entry, status, detail, {"origin": origin})


def step_log(pdb_id, changed_step=None):
    """A log with the input step and remove_waters / remove_small_polymers; changed_step removes input atoms."""
    steps = [StepRecord(name="input", phase="parse", description="", resolved_atoms=3)]
    for name in ("remove_waters", "remove_small_polymers"):
        removed = [(0, 0)] if name == changed_step else []
        steps.append(StepRecord(name=name, phase="cleanup", description="", resolved_atoms=3, removed_input=removed))
    return StepLog(pdb_id=pdb_id, input_atoms=3, steps=steps)


def test_clean_run_has_no_processing_rows(tmp_path):
    inputs = make_inputs(tmp_path, ["4hhb"])

    actual = user_summary(inputs, [origin_finding("4hhb", "input")], [])

    assert actual["processing"] == []
    assert actual["counts"]["input"] == 1


def test_rows_group_findings_by_structure(tmp_path):
    inputs = make_inputs(tmp_path, ["4hhb", "1abc"])
    findings = [origin_finding("4hhb", "processing", "one"), origin_finding("4hhb", "processing", "two")]

    actual = user_summary(inputs, findings, ["4hhb"])

    expected_rows = [{"id": "4hhb", "items": ["one", "two"], "has_3d": True}]
    assert actual["processing"] == expected_rows


def test_dropped_by_prep_and_validate_are_listed(tmp_path):
    validation = {"entries_in": ["a", "b"], "entries_out": ["a"], "dropped": ["b"],
                  "findings": [{"check": "V3", "severity": "E", "entry": "b", "message": "bad", "detail": {}}]}
    inputs = make_inputs(tmp_path, ["a"], validation, cifs=["a", "b", "c"])
    findings = entry_missing_findings(inputs)

    actual = user_summary(inputs, findings, [])

    expected_dropped = [("b", "validate"), ("c", "finetune-prep")]
    assert [(d["id"], d["stage"]) for d in actual["dropped"]] == expected_dropped
    assert actual["dropped"][1]["reason"] == PREP_REASON
    assert "V3 bad" in actual["dropped"][0]["reason"]
    assert actual["processing"] == []


def test_dropped_by_prep_uses_manifest_without_validation_json(tmp_path):
    inputs = make_inputs(tmp_path, ["a"], cifs=["a", "c"])

    actual = user_summary(inputs, [], [])

    expected_dropped = [{"id": "c", "stage": "finetune-prep", "reason": PREP_REASON}]
    assert actual["dropped"] == expected_dropped


def test_cleanup_counts_changed_structures(tmp_path):
    logs = {"x": step_log("x", "remove_waters"), "y": step_log("y")}
    inputs = make_inputs(tmp_path, ["x", "y"], step_logs=logs)

    actual = user_summary(inputs, [], [])

    expected_cleanup = [{"step": "remove_waters", "count": 1, "structures": ["x"], "has_3d": []}]
    assert actual["cleanup"] == expected_cleanup


def test_cleanup_marks_structures_with_3d(tmp_path):
    logs = {"x": step_log("x", "remove_waters"), "y": step_log("y", "remove_waters")}
    inputs = make_inputs(tmp_path, ["x", "y"], step_logs=logs)

    actual = user_summary(inputs, [], ["y"])

    assert actual["cleanup"][0]["has_3d"] == ["y"]


def test_large_dataset_is_truncated(tmp_path):
    ids = [f"s{i:04d}" for i in range(LARGE_DATASET + 1)]
    inputs = make_inputs(tmp_path, ids)
    findings = [origin_finding(pdb_id, "input") for pdb_id in ids]

    actual = user_summary(inputs, findings, [])

    assert len(actual["input"]) == TABLE_ROW_CAP == 100
    assert actual["truncated"] is True
    assert actual["counts"]["input"] == 501


def test_large_dataset_caps_dropped_and_cleanup_keeping_counts(tmp_path):
    ids = [f"s{i:04d}" for i in range(LARGE_DATASET + 1)]
    logs = {pdb_id: step_log(pdb_id, "remove_waters") for pdb_id in ids}
    gone = [f"gone{i:03d}" for i in range(150)]
    inputs = make_inputs(tmp_path, ids, cifs=ids + gone, step_logs=logs)

    actual = user_summary(inputs, [], ids)

    assert len(actual["dropped"]) == TABLE_ROW_CAP
    assert actual["counts"]["dropped"] == 150
    assert len(actual["cleanup"][0]["structures"]) == TABLE_ROW_CAP
    assert actual["cleanup"][0]["count"] == 501


def v_finding(entry, severity, check="V5", message="forced error"):
    """A validation.json finding."""
    return {"check": check, "severity": severity, "entry": entry, "message": message, "detail": {}}


def test_pipeline_v_check_errors_are_processing_rows(tmp_path):
    validation = {"entries_in": ["a"], "findings": [v_finding("a", "E"), v_finding("a", "I", "V7", "fine")]}
    inputs = make_inputs(tmp_path, ["a"], validation)

    actual = user_summary(inputs, [], [])

    expected_rows = [{"id": "a", "items": ["V5 forced error"], "has_3d": False}]
    assert actual["processing"] == expected_rows
    assert actual["counts"]["processing"] == 1


def test_pipeline_v_check_warnings_are_processing_rows(tmp_path):
    validation = {"entries_in": ["a"], "findings": [v_finding("a", "W", "V9", "odd")]}
    inputs = make_inputs(tmp_path, ["a"], validation)

    actual = user_summary(inputs, [], [])

    expected_items = ["V9 odd"]
    assert actual["processing"][0]["items"] == expected_items


def test_dataset_level_v_check_error_is_one_run_row(tmp_path):
    validation = {"entries_in": ["a"], "findings": [v_finding(None, "E", "V21", "load failed")]}
    inputs = make_inputs(tmp_path, ["a"], validation)

    actual = user_summary(inputs, [], [])

    expected_rows = [{"id": "(dataset)", "items": ["V21 load failed"], "has_3d": False}]
    assert actual["processing"] == expected_rows


def test_processing_issue_count_counts_findings_not_structures(tmp_path):
    validation = {"entries_in": ["a", "b"], "findings": [v_finding("b", "E")]}
    inputs = make_inputs(tmp_path, ["a", "b"], validation)
    findings = [origin_finding("a", "processing", "one"), origin_finding("a", "processing", "two")]

    actual = user_summary(inputs, findings, [])

    expected_issues = 3
    expected_structures = 2
    assert actual["counts"]["processing_issues"] == expected_issues
    assert actual["counts"]["processing"] == expected_structures


def test_exactly_large_dataset_is_not_truncated(tmp_path):
    ids = [f"s{i:04d}" for i in range(LARGE_DATASET)]
    inputs = make_inputs(tmp_path, ids)

    actual = user_summary(inputs, [origin_finding(pdb_id, "input") for pdb_id in ids], [])

    assert actual["truncated"] is False
    assert len(actual["input"]) == LARGE_DATASET


def test_row_cap_is_passed_to_the_page(tmp_path):
    inputs = make_inputs(tmp_path, ["a"])

    actual = user_summary(inputs, [], [])

    assert actual["row_cap"] == TABLE_ROW_CAP


def test_validation_json_older_than_manifest_is_ignored(tmp_path):
    validation = {"entries_in": ["a"], "findings": [v_finding("a", "E")]}
    inputs = make_inputs(tmp_path, ["a"], validation)
    manifest_mtime = (tmp_path / "manifest.json").stat().st_mtime
    os.utime(tmp_path / "validation.json", (manifest_mtime - 60, manifest_mtime - 60))

    actual = user_summary(inputs, [], [])

    assert actual["processing"] == []
    assert actual["validation_skipped"] is True


def test_fresh_validation_json_is_not_skipped(tmp_path):
    inputs = make_inputs(tmp_path, ["a"], {"entries_in": ["a"], "findings": []})

    actual = user_summary(inputs, [], [])

    assert actual["validation_skipped"] is False


def test_missing_validation_json_is_skipped(tmp_path):
    inputs = make_inputs(tmp_path, ["a"])

    actual = user_summary(inputs, [], [])

    assert actual["validation_skipped"] is True


def test_all_structures_lists_every_entry_with_its_status(tmp_path):
    logs = {"a": step_log("a", "remove_waters"), "b": step_log("b"), "c": step_log("c"), "d": step_log("d")}
    inputs = make_inputs(tmp_path, ["a", "b", "c", "d"], step_logs=logs)
    findings = [
        origin_finding("a", "input"), origin_finding("a", "input"),
        origin_finding("b", "processing"), origin_finding("b", "input"),
        origin_finding("c", "not_checked"),
    ]

    actual = user_summary(inputs, findings, ["a", "b"])

    expected_all = [
        {"id": "a", "status": "input", "processing": 0, "input": 2, "not_checked": 0, "cleanup_steps": 1, "has_3d": True},
        {"id": "b", "status": "processing", "processing": 1, "input": 1, "not_checked": 0, "cleanup_steps": 0, "has_3d": True},
        {"id": "c", "status": "not_checked", "processing": 0, "input": 0, "not_checked": 1, "cleanup_steps": 0, "has_3d": False},
        {"id": "d", "status": "clean", "processing": 0, "input": 0, "not_checked": 0, "cleanup_steps": 0, "has_3d": False},
    ]
    assert actual["all"] == expected_all


def test_all_structures_counts_pipeline_v_check_errors_as_processing(tmp_path):
    validation = {"entries_in": ["a"], "findings": [v_finding("a", "E")]}
    inputs = make_inputs(tmp_path, ["a"], validation=validation)

    [actual] = user_summary(inputs, [], [])["all"]

    assert actual["status"] == "processing"
    assert actual["processing"] == 1


def test_all_structures_is_capped_for_large_datasets(tmp_path):
    ids = [f"s{i:04d}" for i in range(LARGE_DATASET + 1)]
    inputs = make_inputs(tmp_path, ids)

    actual = user_summary(inputs, [], [])

    assert len(actual["all"]) == TABLE_ROW_CAP


def test_skipped_pipeline_checks_are_not_run(tmp_path):
    findings = [
        v_finding(None, "I", "V16", "skipped: no --template-cache-dir"),
        v_finding(None, "I", "V7", "fine"),
        v_finding("a", "I", "V13", "skipped: no --organized-msa-dir"),
    ]
    inputs = make_inputs(tmp_path, ["a"], validation={"entries_in": ["a"], "findings": findings})

    actual = user_summary(inputs, [], [])

    expected_not_run = [{"check": "V16", "title": "Templates predate their entry", "message": "skipped: no --template-cache-dir"}]
    assert actual["not_run"] == expected_not_run


def test_template_cutoff_is_a_dataset_note(tmp_path):
    note = f"{TEMPLATE_CUTOFF_NOTE} 2030-01-01, not each structure's release date"
    findings = [v_finding(None, "I", "V16", note), v_finding(None, "I", "V7", "fine")]
    inputs = make_inputs(tmp_path, ["a"], validation={"entries_in": ["a"], "findings": findings})

    actual = user_summary(inputs, [], [])

    expected_notes = [note]
    assert actual["notes"] == expected_notes


def test_each_bad_bond_is_its_own_item(tmp_path):
    inputs = make_inputs(tmp_path, ["4hhb"])
    detail = "2:VAL1:C-2:HIS2:N 0.787 Å, too short (min 0.9 Å); 1:LYS60:CE-1:LYS60:NZ 2.690 Å, too long (max 2.4 Å)"

    actual = user_summary(inputs, [origin_finding("4hhb", "input", detail)], [])

    expected_items = ["2:VAL1:C-2:HIS2:N 0.787 Å, too short (min 0.9 Å)", "1:LYS60:CE-1:LYS60:NZ 2.690 Å, too long (max 2.4 Å)"]
    assert actual["input"][0]["items"] == expected_items
    assert actual["all"][0]["input"] == 2


def test_prep_drop_reason_comes_from_prep_dropped_json(tmp_path):
    inputs = make_inputs(tmp_path, ["a"], cifs=["a", "bad"])
    (tmp_path / "prep_dropped.json").write_text(json.dumps([{"id": "bad", "reason": "failed: KeyError: 'type'"}]))

    actual = user_summary(inputs, [], [])

    expected_dropped = [{"id": "bad", "stage": "finetune-prep", "reason": "failed: KeyError: 'type'"}]
    assert actual["dropped"] == expected_dropped


def _write_dated_cif(tmp_path, pdb_id, revision_date):
    """cifs/<pdb_id>.cif with one pdbx_audit_revision_history row."""
    (tmp_path / "cifs" / f"{pdb_id}.cif").write_text(
        f"data_{pdb_id}\n_pdbx_audit_revision_history.ordinal 1\n"
        f"_pdbx_audit_revision_history.revision_date {revision_date}\n"
    )


def test_placeholder_release_date_from_the_input_is_an_input_issue(tmp_path):
    """OpenBind x1346a: the input CIF itself has the 1970-01-01 placeholder date."""
    validation = {"entries_in": ["a"], "findings": [v_finding("a", "W", "V17", "release_date is 1970-01-01")]}
    inputs = make_inputs(tmp_path, ["a"], validation=validation, cifs=["a"])
    _write_dated_cif(tmp_path, "a", "1970-01-01")

    actual = user_summary(inputs, [], [])

    expected_input_rows = [{"id": "a", "items": ["V17 release_date is 1970-01-01"], "has_3d": False}]
    assert actual["processing"] == []
    assert actual["input"] == expected_input_rows
    assert actual["all"][0]["status"] == "input"


def test_placeholder_release_date_not_in_the_input_is_processing(tmp_path):
    validation = {"entries_in": ["a"], "findings": [v_finding("a", "W", "V17", "release_date is 1970-01-01")]}
    inputs = make_inputs(tmp_path, ["a"], validation=validation, cifs=["a"])
    _write_dated_cif(tmp_path, "a", "2024-11-29")

    actual = user_summary(inputs, [], [])

    assert actual["input"] == []
    assert actual["processing"][0]["items"] == ["V17 release_date is 1970-01-01"]
