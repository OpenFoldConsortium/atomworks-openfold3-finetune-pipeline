"""Tests for the validate stage core: findings, report, context, runner."""

import json
import shutil
from pathlib import Path

import pytest

from aw_of3_finetune.validate import CHECK_TITLES, CHECKS, run_validation
from aw_of3_finetune.validate.context import ValidationContext
from aw_of3_finetune.validate.findings import Finding, Severity, ValidationReport
from validate_helpers import context, mutable_copy


def _report() -> ValidationReport:
    """Two findings: an error on 1abc (V3) and a warning on 2xyz (V5)."""
    return ValidationReport(
        findings=[
            Finding("V3", Severity.ERROR, "1abc", "bad", {"k": 1}),
            Finding("V5", Severity.WARNING, "2xyz", "meh"),
        ],
        entries_in=["1abc", "2xyz"],
    )


def test_report_round_trips_json(tmp_path):
    """to_json writes the entries, the surviving entries and each finding's fields."""
    report = _report()
    path = tmp_path / "report.json"

    report.to_json(path)
    data = json.loads(path.read_text())

    # Both input entries are listed; only 2xyz survives because 1abc has an error.
    assert data["entries_in"] == ["1abc", "2xyz"]
    assert data["entries_out"] == ["2xyz"]

    # The first finding is serialized with its check id, severity letter and entry.
    actual_first_finding = {
        key: data["findings"][0][key] for key in ("check", "severity", "entry")
    }
    assert actual_first_finding == {"check": "V3", "severity": "E", "entry": "1abc"}

    # Only the entry with an error fails; one error and one warning are counted.
    assert report.failing_entries == {"1abc"}
    assert len(report.errors) == 1
    assert len(report.warnings) == 1


def test_summary_counts_entries_and_checks():
    """The summary text reports entries in/out and one line per check and severity."""
    text = _report().summary_text()

    # 1abc is dropped (error), 2xyz is kept (warning only).
    assert "entries: 2 in → 1 out" in text

    # V3 produced exactly one error.
    assert "V3  E 1" in text


def test_crashing_check_becomes_finding(monkeypatch):
    """An exception inside a check is reported as an error finding, not raised."""

    def boom(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(CHECKS, "VX", boom)
    ctx = context_stub()

    report = run_validation(ctx, ["VX"])

    # Exactly one finding, an error attributed to the crashing check.
    actual_findings = {(f.check, f.severity) for f in report.findings}
    assert actual_findings == {("VX", Severity.ERROR)}
    assert len(report.findings) == 1

    # The message carries the exception text.
    assert "boom" in report.findings[0].message


def context_stub():
    """A ValidationContext with no files on disk and no entries."""
    ctx = ValidationContext(output_dir=Path("/nonexistent"), cif_dir=Path("/nonexistent"))
    ctx.entries = lambda: []
    return ctx


def test_context_loads_built_dataset(built_dataset):
    """The context lists the dataset's entries and finds each entry's files."""
    ctx = context(built_dataset)

    # The five fixture structures, sorted.
    assert ctx.entries() == ["1a3n", "2crb", "2q2k", "3u8v", "7riv"]

    # The NPZ has xyz coordinates (second axis of size 3).
    assert ctx.npz("3u8v")["coord"].shape[1] == 3

    # The source CIF and the augmented CIF exist on disk.
    assert ctx.source_cif("3u8v").exists()
    assert ctx.augmented_cif("3u8v").exists()


def test_missing_npz_is_a_finding(built_dataset, tmp_path, monkeypatch):
    """A deleted NPZ raises FileNotFoundError, and a check that reads it flags only that entry."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    shutil.rmtree(built.output_dir / "of3_compressed_structures" / "3u8v")
    ctx = context(built)

    with pytest.raises(FileNotFoundError):
        ctx.npz("3u8v")

    def npz_check(check_ctx):
        findings = []
        for entry in check_ctx.entries():
            try:
                check_ctx.npz(entry)
            except Exception as exc:
                findings.append(Finding("VX", Severity.ERROR, entry, f"npz unreadable: {exc}"))
        return findings

    monkeypatch.setitem(CHECKS, "VX", npz_check)

    report = run_validation(ctx, ["VX"])

    # Only the entry whose NPZ was deleted fails.
    assert report.failing_entries == {"3u8v"}


def test_run_without_msa_dir_skips_msa_checks(built_dataset):
    """Without --organized-msa-dir the MSA checks (V8-V10) are silent and V13 notes they were skipped."""
    ctx = context(built_dataset, msa_dir=None)

    report = run_validation(ctx)

    # No findings from the MSA content checks.
    msa_check_findings = [f for f in report.findings if f.check in {"V8", "V9", "V10"}]
    assert msa_check_findings == []

    # V13 emits a single INFO finding explaining the missing MSA dir.
    v13_findings = [f for f in report.findings if f.check == "V13"]
    assert [f.severity for f in v13_findings] == [Severity.INFO]
    assert "no --organized-msa-dir" in v13_findings[0].message


def test_unknown_check_id_raises(built_dataset):
    """Asking for a check id that does not exist is a ValueError naming the id."""
    with pytest.raises(ValueError, match="unknown check.*V99"):
        run_validation(context(built_dataset), ["V99"])


def test_unreadable_cache_is_an_error(built_dataset, tmp_path):
    """A corrupt dataset_cache.json yields a single V0 error and no entries."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    (built.output_dir / "dataset_cache.json").write_text("not json {{{")

    report = run_validation(context(built), [])

    # No entries could be loaded.
    assert report.entries_in == []

    # The only finding is the V0 "cache unreadable" error.
    assert [f.check for f in report.errors] == ["V0"]
    assert "dataset cache unreadable" in report.errors[0].message


def test_crashed_check_message_includes_exception_type(built_dataset, monkeypatch):
    """A crashing check's message names the exception type and text."""

    def boom(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(CHECKS, "V5", boom)

    report = run_validation(context(built_dataset), ["V5"])

    assert report.errors[0].message == "check crashed: RuntimeError: boom"


def test_every_check_has_a_title():
    """Each registered check ID has a short human-readable title for reports."""
    expected_ids = set(CHECKS)
    actual_ids = set(CHECK_TITLES)
    assert actual_ids == expected_ids
    assert all(CHECK_TITLES[check_id] for check_id in expected_ids)
