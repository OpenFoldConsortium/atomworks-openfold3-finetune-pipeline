"""validate_data.json: every verdict the validate report draws, from the same computation as the findings."""

from collections.abc import Sequence

from aw_of3_finetune.validate import CHECK_TITLES

from aw_of3_finetune.report.findings import Finding, Status, worst_label
from aw_of3_finetune.report.origin import INPUT, v_check_origin
from aw_of3_finetune.report.summary import user_summary
from aw_of3_finetune.report.validity.audit import CHECK as CLEANUP_AUDIT
from aw_of3_finetune.report.validity.bonds import BOND_LENGTHS, RESIDUE_BONDS
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.report.validity.ligands import COULD_NOT_RUN, LIGAND_CHEMISTRY, LIGAND_CONTACTS, LIGAND_GEOMETRY, LIGAND_IDENTITY
from aw_of3_finetune.report.validity.ligand_smiles import LIGAND_SMILES
from aw_of3_finetune.report.validity.links import COVALENT_LINKS
from aw_of3_finetune.report.validity.steps3d import STEP_VIEW_ERROR
from aw_of3_finetune.report.validity.tracks import CHECK as CHAIN_PLACEMENT, input_tracks

ERROR_CHECK = "validity_data.error"
NO_LIGANDS = "none"
NOT_APPLICABLE = "none"
V_CHECKS = "v_checks"
LIGAND_CHECKS = (LIGAND_CHEMISTRY, LIGAND_IDENTITY, LIGAND_GEOMETRY, LIGAND_CONTACTS)
CHECK_COLUMNS = {
    CLEANUP_AUDIT: "cleanup_audit",
    CHAIN_PLACEMENT: "chain_placement",
    RESIDUE_BONDS: "residue_bonds",
    BOND_LENGTHS: "bond_lengths",
    LIGAND_CHEMISTRY: "ligand_chemistry",
    LIGAND_IDENTITY: "ligand_identity",
    LIGAND_GEOMETRY: "ligand_geometry",
    LIGAND_CONTACTS: "ligand_contacts",
    COVALENT_LINKS: "covalent_links",
    LIGAND_SMILES: "ligand_smiles",
}
GRID_COLUMNS = (*CHECK_COLUMNS.values(), V_CHECKS)
_STATUS_LABELS = {Status.MATCH: "same", Status.EXPECTED: "expected", Status.UNEXPLAINED: "unexplained"}
_SEVERITY_LABELS = {"E": "unexplained", "W": "unexplained", "I": "same"}
_ALL_UNEXPLAINED = {column: "unexplained" for column in GRID_COLUMNS}
_AUDIT_COUNTED = ("unaccounted", "double_counted", "renamed_unlogged", "moved", "unexplained_final")


def build_validate_data(inputs: ValidityInputs, findings: list[Finding], has_3d: Sequence[str] = ()) -> dict:
    """Everything the validate report draws, per entry, with the verdicts already decided.

    Args:
        inputs: The pipeline output that was validated.
        findings: Validity findings after apply_rules and classify_origins.
        has_3d: Ids of structures that have a 3D view (for the user summary).

    Returns:
        {"entries": {pdb_id: entry}, "reasons": [{"check", "field", "label", "reason"}], "run_findings": [item], "user_summary": user_summary(...)}.
        An entry is {"grid", "tracks", "ligands", "bonds", "audit", "covalent_links", "findings", "v_checks", "step_view"}
        (step_view None after a step_view_error); one whose data fails is {"error", "grid"}. run_findings
        also lists validation.json findings without an entry. A crash outside the entries gives
        {"entries": {}, "error", "reasons": [], "run_findings"} with every finding and a validity_data.error item.
    """
    try:
        return _build(inputs, findings, list(has_3d))
    except Exception as error:  # noqa: BLE001 — the run must still get a validate_data.json and a finding
        message = f"{type(error).__name__}: {error}"
        crash = Finding(ERROR_CHECK, None, Status.UNEXPLAINED, message, {"error": True})
        run_findings = _run_findings([*findings, crash], known=[])
        return {"entries": {}, "error": message, "reasons": [], "run_findings": run_findings, "user_summary": None}


def validity_error_findings(validate_data: dict) -> list[Finding]:
    """Unexplained validity_data.error findings for each failed entry and for a crashed build.

    Args:
        validate_data: build_validate_data's output.

    Returns:
        One finding per entry with "error" (entry set), plus one with entry None if the whole build failed.
    """
    errors = [
        Finding(ERROR_CHECK, pdb_id, Status.UNEXPLAINED, entry["error"], {"error": True})
        for pdb_id, entry in validate_data["entries"].items()
        if "error" in entry
    ]
    if "error" in validate_data:
        errors.append(Finding(ERROR_CHECK, None, Status.UNEXPLAINED, validate_data["error"], {"error": True}))
    return errors


def _build(inputs: ValidityInputs, findings: list[Finding], has_3d: list[str]) -> dict:
    """build_validate_data without the crash guard."""
    by_entry: dict[str, list[Finding]] = {}
    for finding in findings:
        if finding.entry is not None:
            by_entry.setdefault(finding.entry, []).append(finding)
    entries = {pdb_id: _entry(inputs, pdb_id, by_entry.get(pdb_id, [])) for pdb_id in inputs.entries()}
    run_findings = _run_findings(findings, inputs.entries()) + [
        _v_check_item(item) for item in inputs.validation_findings(None)
    ]
    return {
        "entries": entries,
        "reasons": _reasons(findings),
        "run_findings": run_findings,
        "user_summary": user_summary(inputs, findings, has_3d),
    }


def _entry(inputs: ValidityInputs, pdb_id: str, entry_findings: list[Finding]) -> dict:
    """One entry's report data, or {"error", "grid"} if building it raises."""
    try:
        v_checks = [
            {"check": item["check"], "title": CHECK_TITLES.get(item["check"], ""), "severity": item["severity"],
             "message": item["message"], "origin": v_check_origin(inputs, item)}
            for item in inputs.validation_findings(pdb_id)
        ]
        has_step_view = not any(finding.check == STEP_VIEW_ERROR for finding in entry_findings)
        return {
            "grid": _grid(entry_findings, v_checks),
            "tracks": input_tracks(inputs, pdb_id),
            "ligands": _ligand_rows(entry_findings),
            "bonds": {check.split(".")[1]: _rows(entry_findings, check) for check in (RESIDUE_BONDS, BOND_LENGTHS)},
            "audit": _audit(entry_findings),
            "covalent_links": _link_rows(entry_findings),
            "findings": [_finding_item(finding) for finding in entry_findings],
            "v_checks": v_checks,
            "step_view": f"validate_3d/{pdb_id}.json" if has_step_view else None,
        }
    except Exception as error:  # noqa: BLE001 — one bad entry must not lose the others
        return {"error": f"{type(error).__name__}: {error}", "grid": dict(_ALL_UNEXPLAINED)}


def _finding_label(finding: Finding) -> str:
    """The finding's data["label"], else derived from its status."""
    return finding.data.get("label") or _STATUS_LABELS[finding.status]


def _finding_item(finding: Finding) -> dict:
    """{"check", "status", "label", "detail", "reason"} of one finding (after rules)."""
    return {
        "check": finding.check,
        "status": finding.status.value,
        "label": _finding_label(finding),
        "detail": finding.detail,
        "reason": finding.reason,
    }


def _v_check_item(item: dict) -> dict:
    """A run_findings item of a dataset-level validation.json finding, labelled by its severity."""
    label = _SEVERITY_LABELS[item["severity"]]
    status = Status.UNEXPLAINED if label == "unexplained" else Status.MATCH
    if item["severity"] == "I":
        label = NOT_APPLICABLE  # informational (e.g. a skipped check): neither a pass nor a failure
    return {"check": item["check"], "status": status.value, "label": label, "detail": item["message"], "reason": None}


def _run_findings(findings: list[Finding], known: list[str] | None = None) -> list[dict]:
    """Items of the findings that belong to no entry, or to one that is not in `known`."""
    return [
        _finding_item(finding)
        for finding in findings
        if finding.entry is None or (known is not None and finding.entry not in known)
    ]


def _rows(entry_findings: list[Finding], check: str) -> list[dict]:
    """The data rows of every finding of one check, each with its finding's label and reason (passed rows: same)."""
    rows = []
    for finding in entry_findings:
        if finding.check != check:
            continue
        label = _finding_label(finding)
        for row in finding.data.get("rows", []):
            passed = row.get("passed") is True
            reason = COULD_NOT_RUN if row.get("reason") == COULD_NOT_RUN else finding.reason
            rows.append({**row, "label": "same" if passed else label, "reason": None if passed else reason})
    return rows


def _ligand_rows(entry_findings: list[Finding]) -> list[dict]:
    """The rows of the four ligand checks, each tagged with its check."""
    return [{"check": check, **row} for check in LIGAND_CHECKS for row in _rows(entry_findings, check)]


def _link_rows(entry_findings: list[Finding]) -> list[dict]:
    """The covalent-link rows; a row still unexplained takes its finding's label and reason (a rule may cover it)."""
    rows = []
    for finding in entry_findings:
        if finding.check != COVALENT_LINKS:
            continue
        for row in finding.data.get("rows", []):
            ruled = row["label"] == "unexplained"
            rows.append({**row, "label": _finding_label(finding) if ruled else row["label"],
                         "reason": finding.reason if ruled else row["reason"]})
    return rows


def _audit(entry_findings: list[Finding]) -> dict:
    """The cleanup audit's counts: kept, removed per step, each problem list's length and the count checks."""
    data = next((finding.data for finding in entry_findings if finding.check == CLEANUP_AUDIT), {})
    audit = {"has_log": bool(data.get("has_log", False)), "kept": len(data.get("kept", []))}
    audit["removed_by_step"] = {step: len(indices) for step, indices in data.get("removed_by_step", {}).items()}
    for name in _AUDIT_COUNTED:
        audit[name] = len(data.get(name, []))
    audit["balance_errors"] = list(data.get("balance_errors", []))
    audit["input_count_mismatch"] = bool(data.get("input_count_mismatch", False))
    audit["final_count_mismatch"] = data.get("final_count_mismatch")
    return audit


def _grid(entry_findings: list[Finding], v_checks: list[dict]) -> dict[str, str]:
    """Worst label per column (same if a column has no findings); ligand columns are none without ligands."""
    labels: dict[str, list[str]] = {column: [] for column in GRID_COLUMNS}
    for finding in entry_findings:
        if finding.check in CHECK_COLUMNS:
            labels[CHECK_COLUMNS[finding.check]].append(_finding_label(finding))
    labels[V_CHECKS] = [INPUT if item.get("origin") == INPUT and item["severity"] != "I" else _SEVERITY_LABELS[item["severity"]]
                        for item in v_checks]
    grid = {column: worst_label(column_labels) for column, column_labels in labels.items()}
    has_ligands = any(finding.check in LIGAND_CHECKS for finding in entry_findings)
    if not has_ligands:
        grid.update({CHECK_COLUMNS[check]: NO_LIGANDS for check in LIGAND_CHECKS})
    has_links = any(finding.check == COVALENT_LINKS and finding.data.get("links") for finding in entry_findings)
    if not has_links:
        grid[CHECK_COLUMNS[COVALENT_LINKS]] = NOT_APPLICABLE
    if not any(finding.check == LIGAND_SMILES for finding in entry_findings):
        grid[CHECK_COLUMNS[LIGAND_SMILES]] = NOT_APPLICABLE
    return grid


def _reasons(findings: list[Finding]) -> list[dict]:
    """Distinct reasons: per expected finding without reasoned rows (field None) and per reasoned row."""
    found = set()
    for finding in findings:
        rows = finding.data.get("rows", [])
        reasoned_rows = [row for row in rows if row.get("reason") and "field" in row and "label" in row]
        ruled_rows = [row for row in reasoned_rows if row["label"] != "equivalent"]
        if finding.status == Status.EXPECTED and finding.reason and not ruled_rows:
            found.add((finding.check, None, finding.data.get("label", "expected"), finding.reason))
        for row in reasoned_rows:
            found.add((finding.check, row["field"], row["label"], row["reason"]))
    ordered = sorted(found, key=lambda item: (item[0], item[1] or "", item[2], item[3]))
    return [{"check": c, "field": field, "label": label, "reason": reason} for c, field, label, reason in ordered]
