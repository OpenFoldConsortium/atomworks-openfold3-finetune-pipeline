import json
from pathlib import Path

import pytest

import aw_of3_finetune.report.validity.data
from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.report.validity.data import build_validate_data

SAMPLE = Path(__file__).parent / "data" / "validate_data_sample.json"
SAMPLE_3D_DIR = Path(__file__).parent / "data" / "validate_3d"
SAMPLE_IDS = ("7axf", "4x1f", "6tfi")


def _no_constants(name: str):
    """parse_constant for json.loads: NaN and Infinity are not part of the contract."""
    raise ValueError(f"non-finite JSON constant {name}")


def strict_loads(text: str):
    """json.loads that refuses NaN and Infinity."""
    return json.loads(text, parse_constant=_no_constants)


# The validate_data.json contract: the report reads these keys and computes no verdict itself.
TOP_KEYS = {"entries", "reasons", "run_findings", "user_summary"}
CRASHED_TOP_KEYS = {"entries", "error", "reasons", "run_findings", "user_summary"}
ENTRY_KEYS = {"grid", "tracks", "ligands", "bonds", "audit", "covalent_links", "findings", "v_checks", "step_view"}
ERROR_ENTRY_KEYS = {"error", "grid"}
GRID_COLUMNS = [
    "cleanup_audit",
    "chain_placement",
    "residue_bonds",
    "bond_lengths",
    "ligand_chemistry",
    "ligand_identity",
    "ligand_geometry",
    "ligand_contacts",
    "covalent_links",
    "ligand_smiles",
    "v_checks",
]
GRID_LABELS = {"none", "same", "equivalent", "expected", "open", "unexplained"}
TRACKS_KEYS = {"ref", "ours"}
TRACK_CHAIN_KEYS = {"id", "label", "auth", "entity", "type", "ref_mol", "paired", "res"}
LIGAND_ROW_KEYS = {"check", "chain", "res_name", "test", "passed", "value", "threshold", "label", "reason"}
IDENTITY_ROW_KEYS = LIGAND_ROW_KEYS | {"reference"}  # "ccd" or "input": what the ligand is compared to
COVALENT_LINK_ROW_KEYS = {
    "conn_id", "conn_type", "copy", "atom_1", "atom_2", "input_distance", "state", "label", "reason",
}  # fmt: skip
BONDS_KEYS = {"residue_bonds", "bond_lengths"}
RESIDUE_BOND_ROW_KEYS = {
    "ccd_bonds": {"chain", "res_name", "res_id", "issue", "extra", "missing", "order_differs", "label", "reason"},
    "not_in_ccd": {"res_name", "chains", "issue", "label", "reason"},
}
BOND_LENGTH_ROW_KEYS = {"chain", "label_asym_id", "res_name", "res_id", "atom_1", "atom_2", "distance", "label", "reason"}
AUDIT_KEYS = {
    "has_log",
    "kept",
    "removed_by_step",
    "unaccounted",
    "double_counted",
    "renamed_unlogged",
    "moved",
    "balance_errors",
    "input_count_mismatch",
    "final_count_mismatch",
    "unexplained_final",
}
FINDING_KEYS = {"check", "status", "label", "detail", "reason"}
V_CHECK_KEYS = {"check", "title", "severity", "message", "origin"}
REASON_KEYS = {"check", "field", "label", "reason"}
STEP_VIEW_KEYS = {"phases", "steps", "chains", "input", "created", "skipped_groups", "before_after_only"}
STEP_VIEW_INPUT_KEYS = {
    "removed_at", "chain", "label_asym_id", "res_name", "res_id", "atom_name", "element", "hetero",
    "renamed_res_name", "renamed_atom_name", "xyz",
}  # fmt: skip
STEP_VIEW_CREATED_KEYS = {
    "created_at", "removed_at", "chain", "res_name", "res_id", "atom_name", "element", "hetero", "xyz",
}  # fmt: skip


def _finding(check: str, entry: str | None, status: Status, data: dict | None = None, reason: str | None = None) -> Finding:
    return Finding(check, entry, status, f"{check} detail", data or {}, reason)


def _tracks() -> dict:
    chain = {"id": "1", "label": "A", "auth": "A", "entity": 1, "type": "PROTEIN", "ref_mol": None, "paired": "A", "res": [[1, "ALA", "ok"]]}
    return {"ref": [chain], "ours": [chain]}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """Inputs over a two-entry manifest with a validation.json; input_tracks is faked."""
    (tmp_path / "manifest.json").write_text(json.dumps({"structures": ["7axf", "1abc"]}))
    validation = [
        {"check": "V3", "severity": "W", "entry": "7axf", "message": "odd", "detail": {}},
        {"check": "V9", "severity": "I", "entry": "1abc", "message": "fine", "detail": {}},
        {"check": "V21", "severity": "W", "entry": None, "message": "dataset odd", "detail": {}},
    ]
    (tmp_path / "validation.json").write_text(json.dumps({"entries_in": [], "entries_out": [], "findings": validation}))
    monkeypatch.setattr(aw_of3_finetune.report.validity.data, "input_tracks", lambda inputs, pdb_id: _tracks())
    return ValidityInputs(tmp_path, tmp_path, tmp_path / "ccd.cif")


def _findings() -> list[Finding]:
    ligand_rows = [{"chain": "B", "res_name": "S6T", "test": "sanitization", "passed": False, "value": None,
                    "threshold": None}]
    identity_rows = [{"chain": "B", "res_name": "S6T", "reference": "input", "test": "input stereo", "passed": True,
                      "value": None, "threshold": None}]
    bond_rows = [{"chain": "A", "res_name": "ALA", "res_id": 1, "issue": "ccd_bonds", "extra": ["N-CA"], "missing": [],
                  "order_differs": []},
                 {"res_name": "QQQ", "chains": ["C"], "issue": "not_in_ccd"}]
    length_rows = [{"chain": "A", "label_asym_id": "A", "res_name": "ALA", "res_id": 1, "atom_1": "N", "atom_2": "CA",
                    "distance": 3.1}]
    link_rows = [
        {"conn_id": "covale1", "conn_type": "covale", "copy": "A", "atom_1": "A:CYS89:SG", "atom_2": "C:S6T1:C6",
         "input_distance": 1.76, "state": "missing", "label": "unexplained", "reason": None},
        {"conn_id": "disulf1", "conn_type": "disulf", "copy": "A", "atom_1": "A:CYS1:SG", "atom_2": "A:CYS5:SG",
         "input_distance": 2.03, "state": "dropped_by_design", "label": "expected",
         "reason": "OF3 prefilter: polymer link between non-consecutive residues"},
    ]
    audit = {
        "kept": [0, 1], "removed_by_step": {"drop_waters": [2]}, "unaccounted": [], "double_counted": [],
        "renamed_unlogged": [], "moved": [], "balance_errors": [], "input_atoms": 3, "log_input_atoms": 3,
        "input_count_mismatch": False, "final_count_mismatch": None, "unexplained_final": [], "has_log": True,
    }
    return [
        _finding("validity.cleanup_audit", "7axf", Status.MATCH, audit),
        _finding("validity.cleanup_audit", "1abc", Status.MATCH, audit),
        _finding("validity.chain_placement", "7axf", Status.MATCH, {"counts": {}, "bad": []}),
        _finding("validity.residue_bonds", "7axf", Status.EXPECTED, {"rows": bond_rows, "label": "open"}, "known"),
        _finding("validity.bond_lengths", "7axf", Status.UNEXPLAINED, {"rows": length_rows}),
        _finding("validity.ligand_chemistry", "7axf", Status.UNEXPLAINED, {"rows": ligand_rows}),
        _finding("validity.ligand_identity", "7axf", Status.MATCH, {"rows": identity_rows}),
        _finding("validity.ligand_geometry", "7axf", Status.MATCH, {"rows": []}),
        _finding("validity.ligand_contacts", "7axf", Status.MATCH, {"rows": []}),
        _finding("validity.covalent_links", "7axf", Status.UNEXPLAINED, {"links": 2, "rows": link_rows}),
        _finding("run.note", None, Status.UNEXPLAINED),
    ]


def _entry_shape(entry: dict) -> None:
    """Every nested item of a normal entry has its contract keys."""
    assert set(entry) == ENTRY_KEYS
    assert list(entry["grid"]) == GRID_COLUMNS
    assert set(entry["grid"].values()) <= GRID_LABELS
    assert set(entry["tracks"]) == TRACKS_KEYS
    for side in ("ref", "ours"):
        for chain in entry["tracks"][side]:
            assert set(chain) == TRACK_CHAIN_KEYS
    for row in entry["ligands"]:
        expected_keys = IDENTITY_ROW_KEYS if row["check"] == "validity.ligand_identity" else LIGAND_ROW_KEYS
        assert set(row) == expected_keys
    assert set(entry["bonds"]) == BONDS_KEYS
    for row in entry["bonds"]["residue_bonds"]:
        assert set(row) == RESIDUE_BOND_ROW_KEYS[row["issue"]]
    for row in entry["bonds"]["bond_lengths"]:
        assert set(row) == BOND_LENGTH_ROW_KEYS
    assert set(entry["audit"]) == AUDIT_KEYS
    for row in entry["covalent_links"]:
        assert set(row) == COVALENT_LINK_ROW_KEYS
    for item in entry["findings"]:
        assert set(item) == FINDING_KEYS
    for item in entry["v_checks"]:
        assert set(item) == V_CHECK_KEYS
    assert entry["step_view"] is None or entry["step_view"].startswith("validate_3d/")


def test_built_data_has_every_contract_shape(inputs):
    """The builder's output has the contract keys at every level."""
    data = build_validate_data(inputs, _findings())

    assert set(data) == TOP_KEYS
    for entry in data["entries"].values():
        _entry_shape(entry)
    for item in data["run_findings"]:
        assert set(item) == FINDING_KEYS
    for item in data["reasons"]:
        assert set(item) == REASON_KEYS


def test_grid_uses_the_worst_label_and_none_without_ligands(inputs):
    """Ligand columns are none for an entry without ligands; a ruled finding shows its rule's label."""
    data = build_validate_data(inputs, _findings())

    expected_7axf = {
        "cleanup_audit": "same", "chain_placement": "same", "residue_bonds": "open", "bond_lengths": "unexplained",
        "ligand_chemistry": "unexplained", "ligand_identity": "same", "ligand_geometry": "same",
        "ligand_contacts": "same", "covalent_links": "unexplained", "ligand_smiles": "none",
        "v_checks": "unexplained",
    }
    assert data["entries"]["7axf"]["grid"] == expected_7axf
    expected_7axf_link_rows = ["unexplained", "expected"]
    assert [row["label"] for row in data["entries"]["7axf"]["covalent_links"]] == expected_7axf_link_rows
    assert data["entries"]["1abc"]["covalent_links"] == []
    expected_1abc_ligands = ["none"] * 4
    actual_1abc_ligands = [data["entries"]["1abc"]["grid"][c] for c in GRID_COLUMNS[4:8]]
    assert actual_1abc_ligands == expected_1abc_ligands
    assert data["entries"]["1abc"]["grid"]["v_checks"] == "same"


def test_entry_parts_come_from_the_findings(inputs):
    """Ligand and bond rows, audit counts, v_checks and the reasons are read from the findings."""
    data = build_validate_data(inputs, _findings())

    entry = data["entries"]["7axf"]
    expected_ligand_checks = ["validity.ligand_chemistry", "validity.ligand_identity"]
    actual_ligand_checks = [row["check"] for row in entry["ligands"]]
    assert actual_ligand_checks == expected_ligand_checks
    assert [row["res_name"] for row in entry["bonds"]["residue_bonds"]] == ["ALA", "QQQ"]
    assert entry["audit"]["kept"] == 2
    assert entry["audit"]["removed_by_step"] == {"drop_waters": 1}
    expected_v_checks = [{"check": "V3", "title": "Atoms match the source CIF", "severity": "W", "message": "odd",
                          "origin": "processing"}]
    assert entry["v_checks"] == expected_v_checks
    assert entry["step_view"] == "validate_3d/7axf.json"
    expected_reasons = [{"check": "validity.residue_bonds", "field": None, "label": "open", "reason": "known"}]
    assert data["reasons"] == expected_reasons
    assert [item["check"] for item in data["run_findings"]] == ["run.note", "V21"]


def test_error_entry_shape(inputs, monkeypatch):
    """An entry whose data fails is {"error", "grid"}, all unexplained, and the others are still built."""

    def broken_tracks(inputs, pdb_id):
        if pdb_id == "7axf":
            raise ValueError("boom")
        return _tracks()

    monkeypatch.setattr(aw_of3_finetune.report.validity.data, "input_tracks", broken_tracks)

    data = build_validate_data(inputs, _findings())

    assert set(data["entries"]["7axf"]) == ERROR_ENTRY_KEYS
    assert set(data["entries"]["7axf"]["grid"].values()) == {"unexplained"}
    _entry_shape(data["entries"]["1abc"])
    expected_errors = [("validity_data.error", "7axf")]
    actual_errors = [(f.check, f.entry) for f in aw_of3_finetune.report.validity.data.validity_error_findings(data)]
    assert actual_errors == expected_errors


def test_crashed_build_shape(inputs, monkeypatch):
    """A crash outside the per-entry guard keeps the top-level keys and adds error."""

    def broken_reasons(findings):
        raise RuntimeError("no reasons")

    monkeypatch.setattr(aw_of3_finetune.report.validity.data, "_reasons", broken_reasons)

    data = build_validate_data(inputs, _findings())

    assert set(data) == CRASHED_TOP_KEYS
    expected_errors = [("validity_data.error", None)]
    actual_errors = [(f.check, f.entry) for f in aw_of3_finetune.report.validity.data.validity_error_findings(data)]
    assert actual_errors == expected_errors


def test_sample_follows_the_contract():
    """The pxr-pdb sample (7axf, 4x1f, 6tfi) has the contract keys at every level."""
    sample = strict_loads(SAMPLE.read_text())

    assert set(sample) == TOP_KEYS
    assert set(sample["entries"]) == set(SAMPLE_IDS)
    for pdb_id in SAMPLE_IDS:
        _entry_shape(sample["entries"][pdb_id])
    for item in sample["reasons"]:
        assert set(item) == REASON_KEYS
    for item in sample["run_findings"]:
        assert set(item) == FINDING_KEYS


def test_sample_7axf_shows_the_known_ligand_failure():
    """In the sample, 7axf's ligand chemistry is unexplained (S6T sanitization) and 4x1f has no such failure."""
    sample = strict_loads(SAMPLE.read_text())

    assert sample["entries"]["7axf"]["grid"]["ligand_chemistry"] == "unexplained"
    failed = [row for row in sample["entries"]["7axf"]["ligands"] if row["check"] == "validity.ligand_chemistry" and not row["passed"]]
    expected_failed = [("S6T", "sanitization"), ("S6T", "InChI convertible")]
    assert [(row["res_name"], row["test"]) for row in failed] == expected_failed
    assert sample["entries"]["6tfi"]["grid"]["cleanup_audit"] == "same"


@pytest.mark.parametrize("pdb_id", SAMPLE_IDS)
def test_sample_step_view_files_follow_the_step_view_shape(pdb_id):
    """Each sample entry's step_view path resolves to a sample file with the step-view keys."""
    sample = strict_loads(SAMPLE.read_text())

    path = SAMPLE_3D_DIR.parent / sample["entries"][pdb_id]["step_view"]
    step_view = strict_loads(path.read_text())
    assert set(step_view) == STEP_VIEW_KEYS
    assert set(step_view["input"]) == STEP_VIEW_INPUT_KEYS
    assert set(step_view["created"]) == STEP_VIEW_CREATED_KEYS


def test_reasons_lists_a_reasoned_row_with_its_field(inputs):
    """A cache-style row with field, label and reason is listed under its field."""
    rows = [{"field": "release_date", "label": "expected", "reason": "deposition date"}]
    finding = Finding("cache.fields", "7axf", Status.UNEXPLAINED, "release_date", {"rows": rows})

    data = build_validate_data(inputs, [finding])

    expected_reasons = [{"check": "cache.fields", "field": "release_date", "label": "expected", "reason": "deposition date"}]
    assert data["reasons"] == expected_reasons


def test_finding_of_an_unlisted_entry_goes_to_run_findings(inputs):
    """A finding whose entry is not in the manifest is kept in run_findings."""
    stray = _finding("validity.chain_placement", "9zzz", Status.UNEXPLAINED)

    data = build_validate_data(inputs, [stray])

    expected_checks = ["validity.chain_placement", "V21"]
    assert [item["check"] for item in data["run_findings"]] == expected_checks


def test_bond_rows_carry_their_findings_label_and_reason(inputs):
    """Bond rows take the owning finding's label and reason (after rules)."""
    data = build_validate_data(inputs, _findings())

    bonds = data["entries"]["7axf"]["bonds"]
    expected_residue_labels = [("open", "known"), ("open", "known")]
    actual_residue_labels = [(row["label"], row["reason"]) for row in bonds["residue_bonds"]]
    assert actual_residue_labels == expected_residue_labels
    expected_length_labels = [("unexplained", None)]
    actual_length_labels = [(row["label"], row["reason"]) for row in bonds["bond_lengths"]]
    assert actual_length_labels == expected_length_labels


def test_step_view_is_null_for_an_entry_with_a_step_view_error(inputs):
    """An entry with a validity.step_view_error finding has step_view None; the others keep their path."""
    error = Finding("validity.step_view_error", "7axf", Status.UNEXPLAINED, "ValueError: boom", {"error": True})

    data = build_validate_data(inputs, [*_findings(), error])

    assert data["entries"]["7axf"]["step_view"] is None
    assert data["entries"]["1abc"]["step_view"] == "validate_3d/1abc.json"


def test_crashed_build_keeps_every_finding_in_run_findings(inputs, monkeypatch):
    """On a crash every finding, entry or not, is listed in run_findings."""

    def broken_reasons(findings):
        raise RuntimeError("no reasons")

    monkeypatch.setattr(aw_of3_finetune.report.validity.data, "_reasons", broken_reasons)
    findings = _findings()

    data = build_validate_data(inputs, findings)

    expected_count = len(findings) + 1
    assert len(data["run_findings"]) == expected_count


def test_dataset_level_v_finding_goes_to_run_findings(inputs):
    """A validation.json finding without an entry is a run_findings item labelled by its severity."""
    data = build_validate_data(inputs, [])

    expected_item = {"check": "V21", "status": "unexplained", "label": "unexplained", "detail": "dataset odd",
                     "reason": None}
    assert data["run_findings"] == [expected_item]


def test_sample_has_no_will_be_removed_and_7axf_rows_are_labelled():
    """The regenerated sample's failed 7axf S6T row carries its finding's label."""
    sample = strict_loads(SAMPLE.read_text())

    failed = [row for row in sample["entries"]["7axf"]["ligands"] if not row["passed"]]
    expected_labels = {"unexplained"}
    assert {row["label"] for row in failed} == expected_labels
    step_view = strict_loads((SAMPLE_3D_DIR / "7axf.json").read_text())
    assert "will_be_removed" not in step_view["input"]


def _could_not_run_finding() -> Finding:
    rows = [{"chain": "3", "res_name": "S6T", "test": "aromatic ring flatness", "passed": False, "value": None,
             "threshold": 0.25, "label": "unexplained", "reason": "test could not run"}]
    detail = "S6T chain 3: aromatic ring flatness could not run"
    return Finding("validity.ligand_geometry", "7axf", Status.UNEXPLAINED, detail, {"rows": rows})


def test_could_not_run_row_without_rule_is_unexplained(inputs):
    data = build_validate_data(inputs, [_could_not_run_finding()])

    [actual_row] = data["entries"]["7axf"]["ligands"]
    assert (actual_row["label"], actual_row["reason"]) == ("unexplained", "test could not run")
    assert actual_row["passed"] is False


def test_info_dataset_level_v_finding_is_not_a_pass(inputs):
    """An informational (I) finding without an entry, e.g. a skipped check, is labelled none, not same."""
    validation = [{"check": "V16", "severity": "I", "entry": None, "message": "skipped: no --template-cache-dir",
                   "detail": {}}]
    (inputs.out_dir / "validation.json").write_text(
        json.dumps({"entries_in": [], "entries_out": [], "findings": validation})
    )

    data = build_validate_data(inputs, [])

    expected_label = "none"
    actual_label = data["run_findings"][0]["label"]
    assert actual_label == expected_label


def test_ligand_smiles_column_shows_its_status_for_a_table_ligand_entry(inputs):
    smiles_finding = _finding("validity.ligand_smiles", "7axf", Status.MATCH, {"rows": []})

    data = build_validate_data(inputs, [*_findings(), smiles_finding])

    assert data["entries"]["7axf"]["grid"]["ligand_smiles"] == "same"
    assert data["entries"]["1abc"]["grid"]["ligand_smiles"] == "none"
