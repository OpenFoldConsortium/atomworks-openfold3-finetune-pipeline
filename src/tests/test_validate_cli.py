"""Tests for the validate CLI, --allow-errors and the finetune-pipeline stage."""

import json

import numpy as np
import yaml
from click.testing import CliRunner
from openfold3.core.data.io.dataset_cache import read_datacache

from aw_of3_finetune.validate.check_ids import DATASET_LOADS, FINITE_COORDINATES
from aw_of3_finetune import cli as cli_module
from aw_of3_finetune import validate as validate_pkg
from aw_of3_finetune.cli import finetune_pipeline, split_dataset, validate
from aw_of3_finetune.split import dataset_cache_sha256, split_paths
from aw_of3_finetune.templates import TEMPLATE_DATA_DIRNAME
from aw_of3_finetune.training_config import build_train_config
from report.helpers import inline_report_data
from validate_helpers import mutable_copy


def _args(built, ccd_path, *extra):
    """Command-line arguments for `validate` on a built dataset, plus any extra options."""
    return [
        "--output-dir", str(built.output_dir),
        "--cif-dir", str(built.cif_dir),
        "--organized-msa-dir", str(built.msa_dir),
        "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
        *extra,
    ]


def _poison_3u8v(built):
    """Set one 3u8v coordinate to infinity so V5 reports an error for that entry."""
    path = built.output_dir / "of3_compressed_structures" / "3u8v" / "3u8v.npz"
    with np.load(path, allow_pickle=True) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["coord"][0, 0] = np.inf
    np.savez(path, **arrays)


def _break_every_entry(built):
    """Set one coordinate to infinity in every entry's NPZ."""
    for path in (built.output_dir / "of3_compressed_structures").glob("*/*.npz"):
        with np.load(path, allow_pickle=True) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["coord"][0, 0] = np.inf
        np.savez(path, **arrays)


def test_validate_cli_clean_exit_zero(built_dataset, ccd_path):
    """On a clean dataset validate exits 0, writes validation.json and keeps all 5 entries."""
    result = CliRunner().invoke(validate, _args(built_dataset, ccd_path), catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert (built_dataset.output_dir / "validation.json").exists()
    assert "entries: 5 in → 5 out" in result.output


def test_validate_cli_fails_on_error(built_dataset, ccd_path, tmp_path):
    """A corrupt entry makes validate exit 1 and the summary shows 4 of 5 entries surviving."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _poison_3u8v(built)

    result = CliRunner().invoke(validate, _args(built, ccd_path), catch_exceptions=False)

    assert result.exit_code == 1
    assert (built.output_dir / "validation.json").exists()
    assert "entries: 5 in → 4 out" in result.output


def test_validate_cli_allow_errors_drops_entry(built_dataset, ccd_path, tmp_path):
    """--allow-errors removes the failing entry from the cache and manifest but keeps its NPZ."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _poison_3u8v(built)

    result = CliRunner().invoke(
        validate, _args(built, ccd_path, "--allow-errors"), catch_exceptions=False
    )

    assert result.exit_code == 0, result.output

    # The cache keeps the other 4 entries and its pretty-printed "_type" header line.
    cache_text = (built.output_dir / "dataset_cache.json").read_text()
    cache = json.loads(cache_text)
    assert "3u8v" not in cache["structure_data"]
    assert len(cache["structure_data"]) == 4
    assert '"_type"' in cache_text.splitlines()[1]

    # The manifest no longer lists the dropped entry.
    manifest = json.loads((built.output_dir / "manifest.json").read_text())
    assert "3u8v" not in manifest["structures"]

    # The dropped entry's NPZ stays on disk.
    assert (built.output_dir / "of3_compressed_structures" / "3u8v" / "3u8v.npz").exists()


def test_validate_cli_defaults_template_structure_dir(built_dataset, ccd_path, monkeypatch, tmp_path):
    """--template-structure-dir defaults to <msa-dir>/template_structures unless given."""
    seen = {}

    def fake_run(ctx, checks=None):
        seen["ctx"] = ctx
        raise SystemExit(0)

    monkeypatch.setattr("aw_of3_finetune.cli.run_validation", fake_run)

    # Default: derived from the MSA directory.
    result = CliRunner().invoke(validate, _args(built_dataset, ccd_path))
    assert result.exit_code == 0, result.output
    assert seen["ctx"].template_structure_dir == built_dataset.msa_dir / "template_structures"

    # Explicit option wins.
    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    result = CliRunner().invoke(
        validate, _args(built_dataset, ccd_path, "--template-structure-dir", str(explicit))
    )
    assert result.exit_code == 0, result.output
    assert seen["ctx"].template_structure_dir == explicit


def _template_cache_dir_seen(built, ccd_path, monkeypatch):
    """The template cache directory `validate` hands to the checks."""
    seen = {}

    def fake_run(ctx, checks=None):
        seen["ctx"] = ctx
        raise SystemExit(0)

    monkeypatch.setattr("aw_of3_finetune.cli.run_validation", fake_run)
    result = CliRunner().invoke(validate, _args(built, ccd_path))
    assert result.exit_code == 0, result.output
    return seen["ctx"].template_cache_dir


def test_validate_cli_finds_the_template_cache(built_dataset, ccd_path, monkeypatch, tmp_path):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    expected_cache_dir = built.msa_dir / TEMPLATE_DATA_DIRNAME / "template_cache"
    expected_cache_dir.mkdir(parents=True)

    actual_cache_dir = _template_cache_dir_seen(built, ccd_path, monkeypatch)

    assert actual_cache_dir == expected_cache_dir


def test_validate_cli_without_a_template_cache_passes_none(built_dataset, ccd_path, monkeypatch):
    actual_cache_dir = _template_cache_dir_seen(built_dataset, ccd_path, monkeypatch)

    assert actual_cache_dir is None


def test_finetune_pipeline_writes_report(cif_protein_monomer, cif_protein_only, ccd_path, tmp_path):
    """finetune-pipeline runs validate (9/10), then the report (10/10) into <output-dir>/report."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
    (cif_dir / "2crb.cif").symlink_to(cif_protein_only.resolve())

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir),
        "--output-dir", str(tmp_path / "out"),
        "--ccd-path", str(ccd_path),
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Stage 9/10: validate" in result.output
    assert (tmp_path / "out" / "validation.json").exists()
    assert "Stage 10/10: report" in result.output
    assert (tmp_path / "out" / "report" / "index.html").exists()
    expected_report_line = f"Report: {tmp_path / 'out' / 'report' / 'index.html'}"
    actual_last_lines = [line for line in result.output.splitlines() if line.strip()][-2:]
    assert actual_last_lines == [expected_report_line, "Done."]


def _single_cif_dir(tmp_path, cif):
    """An input dir holding one symlinked CIF."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif.resolve())
    return cif_dir


def test_skip_report_writes_nothing(cif_protein_monomer, ccd_path, tmp_path):
    """--skip-report prints the skipped banner and leaves no report folder."""
    cif_dir = _single_cif_dir(tmp_path, cif_protein_monomer)
    out = tmp_path / "out"

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--skip-dataset-load", "--skip-report",
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "=== Stage 10/10: report -- skipped (--skip-report) ===" in result.output
    assert not (out / "report").exists()


def test_report_written_when_validate_fails(cif_protein_monomer, ccd_path, tmp_path, monkeypatch):
    """A validate error still exits 1, but only after the report is written."""
    cif_dir = _single_cif_dir(tmp_path, cif_protein_monomer)
    out = tmp_path / "out"

    def always_error(ctx):
        return [validate_pkg.Finding(FINITE_COORDINATES, validate_pkg.Severity.ERROR, "3u8v", "forced error")]

    monkeypatch.setitem(validate_pkg.CHECKS, FINITE_COORDINATES, always_error)

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    index_html = (out / "report" / "index.html").read_text()
    actual_processing = inline_report_data(index_html)["validate"]["user_summary"]["counts"]["processing"]
    assert actual_processing >= 1


def _boom_report(*args, **kwargs):
    """A report stage that crashes."""
    raise RuntimeError("report boom")


def test_report_crash_keeps_clean_exit(cif_protein_monomer, ccd_path, tmp_path, monkeypatch):
    """A crashing report stage after a clean validate warns and still exits 0."""
    cif_dir = _single_cif_dir(tmp_path, cif_protein_monomer)
    out = tmp_path / "out"
    monkeypatch.setattr(cli_module, "write_user_report", _boom_report)

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "report boom" in result.output
    assert "finetune-report --output-dir" in result.output
    assert result.output.strip().endswith("Done.")


def test_report_crash_keeps_validate_failure(cif_protein_monomer, ccd_path, tmp_path, monkeypatch):
    """A crashing report stage after a failing validate exits with validate's 1, not the crash."""
    cif_dir = _single_cif_dir(tmp_path, cif_protein_monomer)
    out = tmp_path / "out"

    def always_error(ctx):
        return [validate_pkg.Finding(FINITE_COORDINATES, validate_pkg.Severity.ERROR, "3u8v", "forced error")]

    monkeypatch.setitem(validate_pkg.CHECKS, FINITE_COORDINATES, always_error)
    monkeypatch.setattr(cli_module, "write_user_report", _boom_report)

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 1, result.output
    assert "report boom" in result.output


def test_leftover_non_report_folder_fails_before_stage_1(cif_protein_monomer, ccd_path, tmp_path):
    """An <output-dir>/report/ that is not an old report is refused before any stage runs."""
    cif_dir = _single_cif_dir(tmp_path, cif_protein_monomer)
    out = tmp_path / "out"
    (out / "report").mkdir(parents=True)
    keep = out / "report" / "notes.txt"
    keep.write_text("x")

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(out), "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
    ])

    assert result.exit_code != 0
    assert "refusing" in result.output
    assert "Stage 1/10" not in result.output
    assert keep.exists()


def test_allow_errors_does_not_hide_dataset_level_errors(built_dataset, ccd_path, tmp_path, monkeypatch):
    """A crashing check is a dataset-level error: --allow-errors still exits 1 and drops nothing."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    cache_text_before = (built.output_dir / "dataset_cache.json").read_text()

    def boom(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(validate_pkg.CHECKS, FINITE_COORDINATES, boom)

    result = CliRunner().invoke(
        validate, _args(built, ccd_path, "--allow-errors"), catch_exceptions=False
    )

    assert result.exit_code == 1
    assert "dataset-level" in result.output

    # The cache is untouched.
    assert (built.output_dir / "dataset_cache.json").read_text() == cache_text_before


def test_validate_cli_honors_custom_cache_path(built_dataset, ccd_path, tmp_path):
    """--cache-path reads (and with --allow-errors rewrites) the cache at a custom location."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    custom = tmp_path / "elsewhere" / "cache.json"
    custom.parent.mkdir()
    (built.output_dir / "dataset_cache.json").rename(custom)

    # Validation reads the moved cache.
    result = CliRunner().invoke(
        validate, _args(built, ccd_path, "--cache-path", str(custom)), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "entries: 5 in → 5 out" in result.output

    # Dropping an entry edits the custom cache and does not recreate the default one.
    _poison_3u8v(built)
    result = CliRunner().invoke(
        validate,
        _args(built, ccd_path, "--cache-path", str(custom), "--allow-errors"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "3u8v" not in json.loads(custom.read_text())["structure_data"]
    assert not (built.output_dir / "dataset_cache.json").exists()


def test_finetune_pipeline_validates_custom_cache_output(cif_protein_monomer, ccd_path, tmp_path):
    """finetune-pipeline validates the cache written to a custom --cache-output path."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
    custom = tmp_path / "elsewhere" / "cache.json"

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir),
        "--output-dir", str(tmp_path / "out"),
        "--ccd-path", str(ccd_path),
        "--cache-output", str(custom),
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "entries: 1 in → 1 out" in result.output


def test_allow_errors_refuses_to_drop_every_entry(built_dataset, ccd_path, tmp_path):
    """If every entry fails, --allow-errors exits 1 and leaves cache, manifest and backups alone."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _break_every_entry(built)
    cache = built.output_dir / "dataset_cache.json"
    manifest = built.output_dir / "manifest.json"
    bytes_before = (cache.read_bytes(), manifest.read_bytes())

    result = CliRunner().invoke(
        validate, _args(built, ccd_path, "--allow-errors"), catch_exceptions=False
    )

    assert result.exit_code == 1
    assert "refusing to drop every entry; fix the systemic error (see validation.json)" in result.output

    # Cache and manifest are byte-identical; no backup files were written.
    assert (cache.read_bytes(), manifest.read_bytes()) == bytes_before
    assert list(built.output_dir.glob("*.pre-validate.json")) == []


def test_allow_errors_writes_backups_and_records_dropped(built_dataset, ccd_path, tmp_path):
    """--allow-errors backs up the original cache and manifest and records the drop in the report."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _poison_3u8v(built)
    cache = built.output_dir / "dataset_cache.json"
    manifest = built.output_dir / "manifest.json"
    cache_bytes_before, manifest_bytes_before = cache.read_bytes(), manifest.read_bytes()

    result = CliRunner().invoke(
        validate, _args(built, ccd_path, "--allow-errors"), catch_exceptions=False
    )

    assert result.exit_code == 0, result.output

    # The backups hold the pre-drop contents.
    backup_cache = built.output_dir / "dataset_cache.pre-validate.json"
    backup_manifest = built.output_dir / "manifest.pre-validate.json"
    assert backup_cache.read_bytes() == cache_bytes_before
    assert backup_manifest.read_bytes() == manifest_bytes_before

    # validation.json says an entry was dropped, and which.
    report = json.loads((built.output_dir / "validation.json").read_text())
    assert report["allow_errors_applied"] is True
    assert report["dropped"] == ["3u8v"]


def test_validation_json_records_no_drop_without_allow_errors(built_dataset, ccd_path, tmp_path):
    """Without --allow-errors, validation.json records that nothing was dropped."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    result = CliRunner().invoke(validate, _args(built, ccd_path), catch_exceptions=False)

    assert result.exit_code == 0
    report = json.loads((built.output_dir / "validation.json").read_text())
    assert report["allow_errors_applied"] is False
    assert report["dropped"] == []


def test_validate_rejects_nonexistent_msa_dir(built_dataset, ccd_path, tmp_path):
    """A nonexistent --organized-msa-dir is a usage error (exit 2) that names the option."""
    args = _args(built_dataset, ccd_path)
    args[args.index("--organized-msa-dir") + 1] = str(tmp_path / "nope")

    result = CliRunner().invoke(validate, args)

    assert result.exit_code == 2
    assert "--organized-msa-dir" in result.output


def test_validate_skip_dataset_load_records_an_info_finding(built_dataset, ccd_path, tmp_path):
    """--skip-dataset-load leaves a single dataset-wide V21 INFO finding saying it was skipped."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    result = CliRunner().invoke(validate, _args(built, ccd_path), catch_exceptions=False)

    assert result.exit_code == 0
    report = json.loads((built.output_dir / "validation.json").read_text())
    dataset_load_findings = [f for f in report["findings"] if f["check"] == DATASET_LOADS]
    actual_v21 = [(f["severity"], f["entry"], f["message"]) for f in dataset_load_findings]
    assert actual_v21 == [("I", None, "skipped: --skip-dataset-load")]


def test_finetune_pipeline_skip_dataset_load(cif_protein_monomer, ccd_path, tmp_path):
    """finetune-pipeline passes --skip-dataset-load through to the validate stage."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir),
        "--output-dir", str(tmp_path / "out"),
        "--ccd-path", str(ccd_path),
        "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "out" / "validation.json").read_text())
    dataset_load_messages = [f["message"] for f in report["findings"] if f["check"] == DATASET_LOADS]
    assert dataset_load_messages == ["skipped: --skip-dataset-load"]


def _split_with_train_yml(built, tmp_path, csv_text):
    """Run split-dataset --organized-msa-dir on a built dataset, writing split/ and train.yml."""
    csv = tmp_path / "my_split.csv"
    csv.write_text(csv_text)
    result = CliRunner().invoke(split_dataset, [
        "--output-dir", str(built.output_dir), "--split-csv", str(csv), "--organized-msa-dir", str(built.msa_dir),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_allow_errors_refreshes_the_split(built_dataset, ccd_path, tmp_path):
    """A train entry dropped by --allow-errors also leaves split/train_cache.json; train.yml stays valid."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _split_with_train_yml(built, tmp_path, "id,split\n7riv,val\n")
    _poison_3u8v(built)

    result = CliRunner().invoke(
        validate, _args(built, ccd_path, "--allow-errors"), catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    paths = split_paths(built.output_dir)
    actual_train_ids = set(read_datacache(paths.train_cache).structure_data)
    expected_train_ids = {"1a3n", "2crb", "2q2k"}
    assert actual_train_ids == expected_train_ids
    actual_val_ids = set(read_datacache(paths.val_cache).structure_data)
    assert actual_val_ids == {"7riv"}

    # source.json matches the rewritten cache, so the split is current, not stale.
    actual_hash = json.loads(paths.source.read_text())["dataset_cache_sha256"]
    expected_hash = dataset_cache_sha256(built.output_dir)
    assert actual_hash == expected_hash

    # train.yml only names split/ paths, so it is what generate-train-config would write now.
    actual_config = yaml.safe_load((built.output_dir / "train.yml").read_text())
    expected_config = build_train_config(built.output_dir, built.msa_dir)
    assert actual_config == expected_config
    assert "split/" in result.output


def test_allow_errors_on_a_val_id_names_it_and_fails(built_dataset, ccd_path, tmp_path):
    """Dropping an id listed in split.csv cannot re-split; the error names it and says what to do."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _split_with_train_yml(built, tmp_path, "id,split\n3u8v,val\n7riv,val\n")
    _poison_3u8v(built)

    result = CliRunner().invoke(validate, _args(built, ccd_path, "--allow-errors"))

    assert result.exit_code != 0
    assert "3u8v" in result.output
    assert "split-dataset" in result.output
    assert "split.csv" in result.output
    assert isinstance(result.exception, SystemExit), "a clear error, not a traceback"
