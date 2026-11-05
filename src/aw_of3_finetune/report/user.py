"""The user-facing validation report of a pipeline output: validity checks, origins, 3D views, page."""

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aw_of3_finetune.report.build import REPORT_MARKER, build_report
from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.highlights import issue_highlights
from aw_of3_finetune.report.jsonsafe import dumps
from aw_of3_finetune.report.origin import classify_origins
from aw_of3_finetune.report.summary import step_changed
from aw_of3_finetune.report.validity import ENTRY_MISSING, ValidityInputs, run_validity
from aw_of3_finetune.report.validity.data import build_validate_data, validity_error_findings
from aw_of3_finetune.report.validity.steps3d import STEP_VIEW_ERROR, step_view_data

logger = logging.getLogger(__name__)

STAGING_PREFIX = ".report_build-"


class ReportError(ValueError):
    """The report cannot be written as asked (a bad report dir or unknown ids)."""


class ReportDirError(ReportError):
    """The report dir would delete something that is not an old report."""


@dataclass(frozen=True)
class _SubsetInputs(ValidityInputs):
    """ValidityInputs whose entries are only the given ids, in manifest order."""

    ids: frozenset[str] = frozenset()

    def entries(self) -> list[str]:
        """Manifest ids that are in the subset.

        Returns:
            The ids from out_dir/manifest.json that are in ids.
        """
        return [pdb_id for pdb_id in self.manifest_ids() if pdb_id in self.ids]


def _holds(folder: Path, path: Path) -> bool:
    """Whether folder is path or one of its parents (both resolved)."""
    folder, path = folder.resolve(), path.resolve()
    return folder == path or folder in path.parents


def check_report_dir(inputs: ValidityInputs, report_dir: Path) -> None:
    """Refuse a report dir whose deletion (build_report clears it) could lose anything but an old report.

    Args:
        inputs: The pipeline output the report is about.
        report_dir: Where the report would be written.

    Raises:
        ReportDirError: report_dir is or contains the output, CIF, NPZ or working directory, or is an
            existing file or a non-empty folder without the REPORT_MARKER file build_report writes.
    """
    protected = {"output dir": inputs.out_dir, "CIF dir": inputs.cif_dir, "working directory": Path.cwd()}
    if inputs.npz_dir is not None:
        protected["NPZ dir"] = inputs.npz_dir
    for label, path in protected.items():
        if _holds(report_dir, path):
            raise ReportDirError(f"refusing to write the report to {report_dir}: it is or contains the {label} {path}")
    if report_dir.is_file():
        raise ReportDirError(f"refusing to write the report to {report_dir}: it is a file")
    if report_dir.is_dir() and any(report_dir.iterdir()) and not (report_dir / REPORT_MARKER).exists():
        raise ReportDirError(
            f"refusing to replace {report_dir}: it is not empty and has no {REPORT_MARKER} marker, so it is not "
            "a report this tool wrote; delete it manually or choose another --report-dir"
        )


def _subset(inputs: ValidityInputs, ids: list[str]) -> ValidityInputs:
    """The inputs restricted to ids; raises ReportError on an id the manifest lacks."""
    unknown = sorted(set(ids) - set(inputs.manifest_ids()))
    if unknown:
        raise ReportError(f"ids not in {inputs.out_dir / 'manifest.json'}: {', '.join(unknown)}")
    return _SubsetInputs(
        inputs.out_dir, inputs.cif_dir, inputs.ccd_path, inputs.npz_dir, inputs.cache_path, ids=frozenset(ids)
    )


def _needs_3d(inputs: ValidityInputs, findings: list[Finding]) -> list[str]:
    """Entries with a non-matching finding or a cleanup step that changed atoms, in manifest order."""
    flagged = {finding.entry for finding in findings if finding.status != Status.MATCH}
    needed = []
    for pdb_id in inputs.entries():
        log = inputs.step_log(pdb_id)
        if pdb_id in flagged or (log is not None and any(step_changed(step) for step in log.steps)):
            needed.append(pdb_id)
    return needed


def _write_3d_data(
    inputs: ValidityInputs, pdb_ids: list[str], directory: Path, findings: list[Finding]
) -> tuple[list[str], list[Finding]]:
    """Write validate_3d/<id>.json per entry, with its issues; an entry that fails gets no file and one finding."""
    directory.mkdir()
    written, errors = [], []
    for pdb_id in pdb_ids:
        try:
            view = step_view_data(inputs, pdb_id) | {"issues": issue_highlights(inputs, pdb_id, findings)}
            (directory / f"{pdb_id}.json").write_text(dumps(view, separators=(",", ":")))
            written.append(pdb_id)
        except Exception as exc:  # noqa: BLE001 — one broken entry must not stop the others
            (directory / f"{pdb_id}.json").unlink(missing_ok=True)
            detail = f"no 3D step data: {type(exc).__name__}: {exc}"
            logger.warning("%s: %s", pdb_id, detail)
            errors.append(Finding(STEP_VIEW_ERROR, pdb_id, Status.UNEXPLAINED, detail, {"error": True}))
    return written, errors


def _dataset_name(inputs: ValidityInputs) -> str:
    """The manifest's dataset name, or the output dir's name."""
    return json.loads((inputs.out_dir / "manifest.json").read_text()).get("name") or inputs.out_dir.name


def write_user_report(
    inputs: ValidityInputs, report_dir: Path, ids: list[str] | None = None, title: str | None = None
) -> Path:
    """Check a pipeline output and write its self-contained report (index.html opens from disk).

    Args:
        inputs: The pipeline output to report on.
        report_dir: Where to write the report; an old report there is replaced.
        ids: Only report these structures; all manifest entries when None.
        title: Report title; defaults to the manifest's dataset name.

    Returns:
        report_dir.

    Raises:
        ReportDirError: report_dir is not safe to replace (see check_report_dir).
        ReportError: an id is not in the manifest.
    """
    check_report_dir(inputs, report_dir)
    manifest_ids = set(inputs.manifest_ids())
    title = title or _dataset_name(inputs)
    if ids is not None:
        inputs = _subset(inputs, ids)
    findings = [
        finding for finding in run_validity(inputs)
        if not (finding.check == ENTRY_MISSING and finding.entry in manifest_ids)  # left out of the subset
    ]
    findings = classify_origins(inputs, findings)
    report_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=report_dir.parent))
    try:
        has_3d, view_errors = _write_3d_data(inputs, _needs_3d(inputs, findings), staging / "validate_3d", findings)
        findings += classify_origins(inputs, view_errors)  # before validate_data, which nulls their step_view
        validate_data = build_validate_data(inputs, findings, has_3d)
        for error in validity_error_findings(validate_data):
            logger.warning("report data error%s: %s", f" ({error.entry})" if error.entry else "", error.detail)
        (staging / "validate_data.json").write_text(dumps(validate_data, separators=(",", ":")))
        build_report(staging, title, report_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return report_dir
