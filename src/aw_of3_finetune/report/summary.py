"""The data of the user summary page: findings by origin, dropped structures and what cleanup changed."""

from aw_of3_finetune.constants import TEMPLATE_CUTOFF_NOTE
from aw_of3_finetune.steplog import STEP_PHASES, StepRecord
from aw_of3_finetune.validate import CHECK_TITLES

from aw_of3_finetune.report.findings import Finding
from aw_of3_finetune.report.origin import INPUT, NOT_CHECKED, PROCESSING, v_check_origin
from aw_of3_finetune.report.validity import ENTRY_MISSING, ValidityInputs
from aw_of3_finetune.report.validity.bonds import BOND_LENGTHS

LARGE_DATASET = 500
TABLE_ROW_CAP = 100
PREP_DROPPED = "dropped by finetune-prep; see its log"
DATASET_ROW = "(dataset)"
PROCESSING_SEVERITIES = ("E", "W")  # as the grid's v_checks column: unexplained, so processing
_ORIGINS = (PROCESSING, INPUT, NOT_CHECKED)
CLEAN = "clean"
SKIPPED_PREFIX = "skipped"


def user_summary(inputs: ValidityInputs, findings: list[Finding], has_3d: list[str]) -> dict:
    """The summary page's data from classified findings.

    Args:
        inputs: The pipeline output that was validated.
        findings: Findings after classify_origins.
        has_3d: Ids of structures that have a 3D view.

    Returns:
        {"structures", "dropped", "processing", "input", "not_checked", "cleanup", "all", "not_run", "notes", "truncated",
        "row_cap", "validation_skipped", "counts"} (validation_skipped: no current validation.json);
        processing also holds the E and W findings of validation.json (dataset-level ones as DATASET_ROW);
        each origin list holds {"id", "items", "has_3d"} rows by structure; the origin, dropped and cleanup
        lists are cut to TABLE_ROW_CAP for large datasets, with the full sizes kept in counts and cleanup "count".
        counts["processing_issues"] is the number of processing findings (counts["processing"] counts structures).
        "all" has one row per structure (cut like the others): its worst origin as "status" (or CLEAN), the
        number of issues per origin, and how many cleanup steps changed it. "not_run" lists the pipeline
        validate checks skipped for the whole dataset ({"check", "title", "message"}).
    """
    entries = inputs.entries()
    rows = {origin: _rows(findings, origin, set(has_3d)) for origin in _ORIGINS}
    v_check_rows = _v_check_rows(inputs, entries, set(has_3d))
    rows[PROCESSING] = _merge(rows[PROCESSING], v_check_rows[PROCESSING])
    rows[INPUT] = _merge(rows[INPUT], v_check_rows[INPUT])
    truncated = len(entries) > LARGE_DATASET
    dropped = _dropped(inputs, findings)
    cap = TABLE_ROW_CAP if truncated else None
    changed = _changed_steps(inputs, entries)
    return {
        "structures": len(entries),
        "dropped": dropped[:cap],
        **{origin: origin_rows[:TABLE_ROW_CAP] if truncated else origin_rows for origin, origin_rows in rows.items()},
        "cleanup": _cleanup(changed, set(has_3d), cap),
        "all": _all_rows(entries, rows, changed, set(has_3d))[:cap],
        "not_run": _not_run(inputs),
        "notes": _notes(inputs),
        "truncated": truncated,
        "row_cap": TABLE_ROW_CAP,
        "validation_skipped": not inputs.validation_report(),
        "counts": {
            **{origin: len(origin_rows) for origin, origin_rows in rows.items()},
            "processing_issues": sum(len(row["items"]) for row in rows[PROCESSING]),
            "dropped": len(dropped),
        },
    }


def step_changed(step: StepRecord) -> bool:
    """Whether a cleanup step removed, created or renamed any atom (never true for the input record).

    Args:
        step: One step of a cleanup step log.

    Returns:
        True when the step changed the structure.
    """
    if step.name == "input":
        return False
    return bool(step.removed_input or step.removed_created or step.created or step.renamed)


def _rows(findings: list[Finding], origin: str, has_3d: set[str]) -> list[dict]:
    """One row per structure with findings of this origin, ordered by id; a bond-lengths finding gives one item per bond."""
    items: dict[str, list[str]] = {}
    for finding in findings:
        if finding.entry is None or finding.check == ENTRY_MISSING or finding.data.get("origin") != origin:
            continue
        # check_bond_lengths joins one "; "-separated part per bad bond.
        parts = finding.detail.split("; ") if finding.check == BOND_LENGTHS else [finding.detail]
        items.setdefault(finding.entry, []).extend(parts)
    return [{"id": pdb_id, "items": items[pdb_id], "has_3d": pdb_id in has_3d} for pdb_id in sorted(items)]


def _v_check_rows(inputs: ValidityInputs, entries: list[str], has_3d: set[str]) -> dict[str, list[dict]]:
    """Rows of the pipeline's own E and W validate findings by origin: per entry, and one DATASET_ROW for the dataset."""
    rows: dict[str, list[dict]] = {PROCESSING: [], INPUT: []}
    for pdb_id in [None, *sorted(entries)]:
        items: dict[str, list[str]] = {PROCESSING: [], INPUT: []}
        for item in inputs.validation_findings(pdb_id):
            if item["severity"] in PROCESSING_SEVERITIES:
                items[v_check_origin(inputs, item)].append(f"{item['check']} {item['message']}")
        for origin, origin_items in items.items():
            if origin_items:
                rows[origin].append({"id": pdb_id or DATASET_ROW, "items": origin_items, "has_3d": pdb_id in has_3d})
    return rows


def _merge(rows: list[dict], extra: list[dict]) -> list[dict]:
    """Rows with extra's items appended to the row of the same id (has_3d from rows), ordered by id."""
    merged = {row["id"]: {**row, "items": list(row["items"])} for row in rows}
    for row in extra:
        if row["id"] in merged:
            merged[row["id"]]["items"] += row["items"]
        else:
            merged[row["id"]] = row
    return [merged[pdb_id] for pdb_id in sorted(merged)]


def _dropped(inputs: ValidityInputs, findings: list[Finding]) -> list[dict]:
    """Structures dropped by validate (with their E findings) and by finetune-prep."""
    dropped = [
        {"id": finding.entry, "stage": "validate", "reason": finding.detail}
        for finding in findings
        if finding.check == ENTRY_MISSING
    ]
    report = inputs.validation_report()
    kept = set(report["entries_in"] if "entries_in" in report else inputs.manifest_ids())
    prepped = sorted(pdb_id for pdb_id in inputs.input_cifs() if pdb_id not in kept)
    reasons = inputs.prep_drop_reasons()
    return dropped + [{"id": pdb_id, "stage": "finetune-prep", "reason": reasons.get(pdb_id, PREP_DROPPED)}
                      for pdb_id in prepped]


def _changed_steps(inputs: ValidityInputs, entries: list[str]) -> dict[str, list[str]]:
    """Per entry, the cleanup steps whose step log shows they removed, created or renamed atoms."""
    changed = {}
    for pdb_id in entries:
        log = inputs.step_log(pdb_id)
        changed[pdb_id] = [step.name for step in log.steps if step.name in STEP_PHASES and step_changed(step)] if log else []
    return changed


def _cleanup(changed: dict[str, list[str]], has_3d: set[str], cap: int | None) -> list[dict]:
    """Per cleanup step, the structures it changed.

    Each row has the full "count", the shown "structures" (cut to cap) and which of those have a 3D view.
    """
    by_step: dict[str, list[str]] = {step: [] for step in STEP_PHASES if step != "input"}
    for pdb_id in sorted(changed):
        for step in changed[pdb_id]:
            by_step[step].append(pdb_id)
    return [
        {"step": step, "count": len(ids), "structures": ids[:cap], "has_3d": [i for i in ids[:cap] if i in has_3d]}
        for step, ids in by_step.items()
        if ids
    ]


def _all_rows(entries: list[str], rows: dict[str, list[dict]], changed: dict[str, list[str]], has_3d: set[str]) -> list[dict]:
    """One row per entry in manifest order: worst origin, issue counts per origin and changed cleanup steps."""
    counts = {origin: {row["id"]: len(row["items"]) for row in origin_rows} for origin, origin_rows in rows.items()}
    result = []
    for pdb_id in entries:
        issues = {origin: counts[origin].get(pdb_id, 0) for origin in _ORIGINS}
        status = next((origin for origin in (PROCESSING, INPUT, NOT_CHECKED) if issues[origin]), CLEAN)
        result.append({"id": pdb_id, "status": status, **issues, "cleanup_steps": len(changed[pdb_id]), "has_3d": pdb_id in has_3d})
    return result


def _notes(inputs: ValidityInputs) -> list[str]:
    """Dataset-level facts a reader should see, such as a template cutoff."""
    return [
        item["message"]
        for item in inputs.validation_findings(None)
        if item["severity"] == "I" and item["message"].startswith(TEMPLATE_CUTOFF_NOTE)
    ]


def _not_run(inputs: ValidityInputs) -> list[dict]:
    """The pipeline validate checks that were skipped for the whole dataset (info findings saying so)."""
    return [
        {"check": item["check"], "title": CHECK_TITLES.get(item["check"], ""), "message": item["message"]}
        for item in inputs.validation_findings(None)
        if item["severity"] == "I" and item["message"].startswith(SKIPPED_PREFIX)
    ]
