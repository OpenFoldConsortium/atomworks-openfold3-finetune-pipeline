"""The validate-dataset command and how it drops failing entries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json

from aw_of3_finetune.cli.options import (
    organized_msa_dir_option,
    output_dir_option,
    resolve_organized_msa_dir,
)
from aw_of3_finetune.pipeline import (
    write_slim_manifest,
)
from aw_of3_finetune.split import (
    DATASET_CACHE_NAME,
    SplitCsvError,
    split_csv_ids,
    split_dataset_cache,
    split_paths,
)
from aw_of3_finetune.templates import (
    TEMPLATE_DATA_DIRNAME,
    TEMPLATE_STRUCTURES_DIRNAME,
)
from aw_of3_finetune.validate import (
    CHECKS,
    Finding,
    Severity,
    ValidationContext,
    run_validation,
)
from aw_of3_finetune.validate.check_ids import DATASET_LOADS


def _drop_entries(output_dir: Path, cache_path: Path, failing: set[str]) -> None:
    """Back up, then remove entries from the dataset cache and manifest.json; files on disk stay.

    Backups are ``<name>.pre-validate.json``; one from an earlier run is overwritten.

    Args:
        output_dir: Pipeline output directory (holds manifest.json).
        cache_path: Dataset cache JSON to rewrite.
        failing: Entry IDs to drop.
    """
    manifest_path = output_dir / "manifest.json"
    for original in (cache_path, manifest_path):
        shutil.copy2(original, original.with_name(f"{original.stem}.pre-validate.json"))
    cache = read_datacache(cache_path)
    for entry in failing:
        cache.structure_data.pop(entry, None)
    with cache_path.open("w") as f:
        write_datacache_to_json(cache, f)
    manifest = json.loads(manifest_path.read_text())
    kept = [e for e in manifest["structures"] if e not in failing]
    write_slim_manifest(kept, manifest_path, cache_name=manifest["name"])


@click.command()
@output_dir_option(exists=True,
              help="Pipeline output directory (dataset_cache.json, of3_compressed_structures/, ...).")
@click.option("--cif-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Directory of the original input mmCIF files.")
@organized_msa_dir_option(exists=True, help="MSA folder (holds of3_alignments/ and template_structures/), if it exists.")
@click.option("--template-cache-dir", default=None, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="OF3 template cache directory ({rep_id}.npz). Defaults to <organized-msa-dir>/of3_template_data/template_cache.")
@click.option("--template-structure-dir", default=None, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Template structure directory. Defaults to <organized-msa-dir>/template_structures.")
@click.option("--ccd-path", default=None, type=click.Path(path_type=Path, dir_okay=False),
              help="Chemical Component Dictionary mmCIF. Reserved; currently unused by checks.")
@click.option("--cache-path", default=None, type=click.Path(path_type=Path, dir_okay=False),
              help="Dataset cache JSON. Defaults to <output-dir>/dataset_cache.json.")
@click.option("--expect-ligands/--no-expect-ligands", default=False, show_default=True,
              help="Warn about entries without any ligand (check V4).")
@click.option("--allow-errors/--fail-on-errors", default=False,
              help="Drop failing entries from the dataset cache and manifest.json and exit 0, "
              "instead of exiting 1. Structure/SDF/MSA files are untouched; cache and manifest are "
              "rewritten (backups saved as *.pre-validate.json). Refuses when every entry fails "
              "or on dataset-level errors.")
@click.option("--skip-dataset-load", is_flag=True, default=False,
              help="Skip V21, the check that loads every entry through OF3's dataset (slow).")
@click.pass_context
def validate(
    ctx: click.Context,
    output_dir: Path,
    cif_dir: Path,
    msa_dir: Path | None,
    template_cache_dir: Path | None,
    template_structure_dir: Path | None,
    ccd_path: Path | None,
    cache_path: Path | None,
    expect_ligands: bool,
    allow_errors: bool,
    skip_dataset_load: bool,
) -> None:
    """Validate a prepared fine-tuning dataset; writes <output-dir>/validation.json.

    Exits 1 if any entry has an error, unless --allow-errors drops those entries.
    """
    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=True)
    if template_structure_dir is None and msa_dir is not None:
        template_structure_dir = msa_dir / TEMPLATE_STRUCTURES_DIRNAME
    if template_cache_dir is None and msa_dir is not None:
        built_cache = msa_dir / TEMPLATE_DATA_DIRNAME / "template_cache"
        template_cache_dir = built_cache if built_cache.is_dir() else None
    vctx = ValidationContext(
        output_dir=output_dir,
        cif_dir=cif_dir,
        msa_dir=msa_dir,
        template_cache_dir=template_cache_dir,
        template_structure_dir=template_structure_dir,
        ccd_path=ccd_path,
        expect_ligands=expect_ligands,
        cache_path=cache_path,
    )
    checks = [c for c in CHECKS if not (skip_dataset_load and c == DATASET_LOADS)]
    report = run_validation(vctx, checks)
    if skip_dataset_load:
        report.findings.append(Finding(DATASET_LOADS, Severity.INFO, None, "skipped: --skip-dataset-load"))
    report_path = output_dir / "validation.json"
    report.to_json(report_path, {"allow_errors_applied": False, "dropped": []})
    click.echo(report.summary_text())
    if not report.errors:
        return
    if not allow_errors:
        ctx.exit(1)
    if any(f.entry is None for f in report.errors):
        click.echo("--allow-errors: dataset-level errors (no entry) cannot be dropped; exiting 1")
        ctx.exit(1)
    failing = report.failing_entries
    if not report.entries_out:
        click.echo("refusing to drop every entry; fix the systemic error (see validation.json)")
        ctx.exit(1)
    _drop_entries(output_dir, vctx.cache_path, failing)
    report.to_json(report_path, {"allow_errors_applied": True, "dropped": sorted(failing)})
    if _refresh_split_after_drop(output_dir, vctx.cache_path, failing):
        click.echo(
            f"--allow-errors: dropped {len(failing)} entry(ies) from the dataset cache, manifest.json "
            f"and {split_paths(output_dir).dir}/ (re-split from its split.csv; train.yml names only "
            "split/ files, so it needs no change)"
        )
    else:
        click.echo(f"--allow-errors: dropped {len(failing)} entry(ies) from the dataset cache and manifest.json")


def _refresh_split_after_drop(output_dir: Path, cache_path: Path, failing: set[str]) -> bool:
    """Re-split `<output-dir>/split/` from its recorded split.csv after validate dropped entries.

    Args:
        output_dir: Pipeline output directory.
        cache_path: The dataset cache validate rewrote.
        failing: Entry IDs validate dropped.

    Returns:
        True if split/ was re-split; False if there is no split/ made from this cache.

    Raises:
        click.ClickException: A dropped id is listed in split.csv, or split.csv is missing.
    """
    paths = split_paths(output_dir)
    if not paths.dir.is_dir() or cache_path.resolve() != (output_dir / DATASET_CACHE_NAME).resolve():
        return False
    if not paths.csv.exists():
        raise click.ClickException(
            f"{paths.dir}/ has no split.csv, so it cannot be refreshed after the drop: rerun "
            f"split-dataset --output-dir {output_dir} --split-csv <your csv>."
        )
    listed = sorted(split_csv_ids(paths.csv) & {entry.lower() for entry in failing})
    if listed:
        raise click.ClickException(
            f"validate dropped {', '.join(listed)}, which the split CSV lists ({paths.csv}). "
            f"{paths.dir}/ is now stale. Remove those rows from your split CSV and rerun "
            f"split-dataset --output-dir {output_dir} --split-csv <your csv>."
        )
    try:
        result = split_dataset_cache(output_dir, paths.csv)
    except SplitCsvError as exc:
        raise click.ClickException(
            f"could not refresh {paths.dir}/ after the drop; fix the split CSV and rerun "
            f"split-dataset.\n{exc.message}"
        ) from exc
    for warning in result.warnings:
        click.echo(warning, err=True)
    return True
