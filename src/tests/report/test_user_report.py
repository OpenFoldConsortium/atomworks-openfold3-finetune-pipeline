"""Tests for write_user_report and the finetune-report command."""

import pytest
from click.testing import CliRunner

from aw_of3_finetune.cli import finetune_pipeline, finetune_report
from aw_of3_finetune.report.build import REPORT_MARKER, build_report
from aw_of3_finetune.report.user import ReportDirError, write_user_report
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.report.validity.steps3d import STEP_VIEW_ERROR
from report.helpers import REPO, entry_report_data, inline_report_data
from validate_helpers import mutable_copy


@pytest.fixture
def run_dirs(tmp_path):
    """Empty output and CIF dirs, and ValidityInputs over them (the refusals run before any read)."""
    out_dir, cif_dir = tmp_path / "out", tmp_path / "cifs"
    out_dir.mkdir()
    cif_dir.mkdir()
    return ValidityInputs(out_dir, cif_dir, tmp_path / "ccd.cif")


@pytest.mark.parametrize("protected", ["out_dir", "cif_dir", "parent_of_out_dir"])
def test_refuses_report_dir_holding_inputs(run_dirs, protected):
    """A report dir that is, or contains, the output or CIF dir is refused before anything is deleted."""
    report_dirs = {
        "out_dir": run_dirs.out_dir,
        "cif_dir": run_dirs.cif_dir,
        "parent_of_out_dir": run_dirs.out_dir.parent,
    }
    marker = run_dirs.out_dir / "keep.txt"
    marker.write_text("x")

    with pytest.raises(ReportDirError):
        write_user_report(run_dirs, report_dirs[protected])

    assert marker.exists()


def test_refuses_report_dir_holding_the_working_directory(run_dirs, tmp_path, monkeypatch):
    """The current working directory, or a folder holding it, is never a report dir."""
    working = tmp_path / "work" / "here"
    working.mkdir(parents=True)
    monkeypatch.chdir(working)

    with pytest.raises(ReportDirError):
        write_user_report(run_dirs, tmp_path / "work")

    assert working.exists()


def test_refuses_non_report_folder(run_dirs, tmp_path):
    """An existing non-empty folder without index.html is not an old report, so it is kept."""
    report_dir = tmp_path / "notes"
    report_dir.mkdir()
    keep = report_dir / "thesis.txt"
    keep.write_text("x")

    with pytest.raises(ReportDirError):
        write_user_report(run_dirs, report_dir)

    assert keep.exists()


def test_finetune_report_refusal_is_a_click_error(run_dirs, tmp_path):
    """finetune-report turns a refused --report-dir into a usage error, not a traceback."""
    ccd = tmp_path / "ccd.cif"
    ccd.write_text("")

    result = CliRunner().invoke(finetune_report, [
        "--output-dir", str(run_dirs.out_dir),
        "--cif-dir", str(run_dirs.cif_dir),
        "--ccd-path", str(ccd),
        "--report-dir", str(run_dirs.out_dir),
    ])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "refusing" in result.output


def test_finetune_report_subset(built_dataset, ccd_path, tmp_path):
    """--ids reports only the listed structures."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("3u8v\n")

    result = CliRunner().invoke(finetune_report, [
        "--output-dir", str(built.output_dir),
        "--cif-dir", str(built.cif_dir),
        "--ccd-path", str(ccd_path),
        "--ids", str(ids_file),
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    index_html = built.output_dir / "report" / "index.html"
    assert str(index_html) in result.output
    validate_data = inline_report_data(index_html.read_text())["validate"]
    expected_structures = 1
    actual_structures = validate_data["user_summary"]["structures"]
    assert actual_structures == expected_structures
    expected_entries = ["3u8v"]
    actual_entries = sorted(validate_data["entries"])
    assert actual_entries == expected_entries
    actual_dropped = validate_data["user_summary"]["dropped"]
    assert actual_dropped == []
    actual_leftovers = sorted(path.name for path in built.output_dir.glob(".report_build*"))
    assert actual_leftovers == []


def test_staging_is_a_fresh_temp_dir(built_dataset, ccd_path, tmp_path):
    """The build stages in its own temp dir: a folder named .report_build nearby is untouched."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    bystander = built.output_dir / ".report_build"
    bystander.mkdir()
    keep = bystander / "keep.txt"
    keep.write_text("x")
    inputs = ValidityInputs(built.output_dir, built.cif_dir, ccd_path)

    write_user_report(inputs, built.output_dir / "report", ids=["3u8v"])

    assert keep.exists()
    actual_leftovers = sorted(path.name for path in built.output_dir.glob(".report_build-*"))
    assert actual_leftovers == []


def test_uppercase_input_cif_name(ccd_path, tmp_path):
    """An input named 7RIV.cif (finetune-prep lowercases ids) is found: not dropped, with its 3D view."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "7RIV.cif").symlink_to((REPO / "src/tests/fixtures/mmcifs/7riv.cif").resolve())
    out = tmp_path / "out"

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    validate_data = inline_report_data((out / "report" / "index.html").read_text())["validate"]
    expected_dropped = 0
    actual_dropped = validate_data["user_summary"]["counts"]["dropped"]
    assert actual_dropped == expected_dropped
    entry = entry_report_data(out / "report", "7riv")["validate"]
    actual_checks = {finding["check"] for finding in entry["findings"]}
    assert STEP_VIEW_ERROR not in actual_checks
    assert (out / "report" / "data" / "validate_3d" / "7riv.js").exists()


def test_custom_cache_output_is_read(ccd_path, tmp_path):
    """With --cache-output elsewhere, the ligand and link checks still read the dataset cache."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "7riv.cif").symlink_to((REPO / "src/tests/fixtures/mmcifs/7riv.cif").resolve())
    out = tmp_path / "out"

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--cache-output", str(tmp_path / "elsewhere.json"), "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    validate_data = inline_report_data((out / "report" / "index.html").read_text())["validate"]
    expected_not_checked = 0
    actual_not_checked = validate_data["user_summary"]["counts"]["not_checked"]
    assert actual_not_checked == expected_not_checked


def test_refuses_folder_with_index_html_but_no_marker(run_dirs, tmp_path):
    """A folder with an index.html that build_report did not write (no marker) is not replaced."""
    report_dir = tmp_path / "site"
    report_dir.mkdir()
    keep = report_dir / "index.html"
    keep.write_text("<html>my page</html>")

    with pytest.raises(ReportDirError, match="delete it manually"):
        write_user_report(run_dirs, report_dir)

    assert keep.exists()


def test_build_report_writes_the_marker(tmp_path):
    """Every report folder build_report writes carries the marker that check_report_dir looks for."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report_dir = build_report(run_dir, "t")

    assert (report_dir / REPORT_MARKER).exists()
