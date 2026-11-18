"""Template commands: fetch-templates and preprocess-templates."""

from __future__ import annotations

from pathlib import Path

import click

from aw_of3_finetune.cli.options import (
    organized_msa_dir_option,
    output_dir_option,
    require_organized_msa_dir,
    with_pipeline_logs,
)
from aw_of3_finetune.templates import (
    TEMPLATE_ALIGNMENTS_DIRNAME,
    TEMPLATE_DATA_DIRNAME,
    TEMPLATE_STRUCTURES_DIRNAME,
    TemplateDownloadError,
    TemplateSequenceFilterSettings,
    preprocess_templates_for_of3,
    reorganize_template_alignments_for_of3,
)
from aw_of3_finetune.templates import fetch_templates as fetch_templates_for_dataset


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory produced by finetune-prep (manifest.json + "
    "cleaned_structures/); only this dataset's sequences are processed.",
)
@organized_msa_dir_option(exists=True, help="MSA folder written by generate-msas (with templates on).")
@click.option(
    "--max-hits",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help="Unique PDB entries to resolve per sequence, best e-value first. Applied "
    "before the template sequence and release-date filters, so keep it >= OF3's "
    "max_templates (20); 50 keeps more templates when many hits copy the query.",
)
@click.option(
    "--pdb-mirror-path",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Local PDB mirror (<mirror>/<id[1:3]>/<id>.cif.gz) to read entries from "
    "before downloading. Default: download everything from RCSB.",
)
@with_pipeline_logs
def fetch_templates(
    output_dir: Path, msa_dir: Path | None, max_hits: int, pdb_mirror_path: Path | None
) -> None:
    """Resolve template structures and lay out ColabFold template hits for OF3.

    Writes <organized-msa-dir>/template_structures/ and <organized-msa-dir>/of3_template_alignments/;
    safe to rerun.
    """
    msa_dir = require_organized_msa_dir(msa_dir, output_dir)
    try:
        summary = fetch_templates_for_dataset(
            output_dir, msa_dir, max_hits=max_hits, pdb_mirror_path=pdb_mirror_path
        )
    except TemplateDownloadError as error:
        raise click.ClickException(str(error)) from error
    click.echo(
        f"Resolved {summary.n_structures} template structure(s) for "
        f"{summary.n_alignments} sequence(s) under {msa_dir / TEMPLATE_STRUCTURES_DIRNAME}"
    )
    written, missing = reorganize_template_alignments_for_of3(
        output_dir, msa_dir, msa_dir / TEMPLATE_ALIGNMENTS_DIRNAME
    )
    click.echo(
        f"Reorganized {written} template alignment(s) under "
        f"{msa_dir / TEMPLATE_ALIGNMENTS_DIRNAME}"
        + (f" ({missing} sequence(s) without template hits)" if missing else "")
    )


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory (manifest.json, cleaned_structures/, and dataset_cache.json "
    "from materialize-dataset-cache).",
)
@organized_msa_dir_option(
    exists=True,
    help="MSA folder after fetch-templates (template_structures/ and of3_template_alignments/).",
)
@click.option(
    "--cache-file",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Dataset cache to add template_ids to, in place. Defaults to "
    "<output-dir>/dataset_cache.json.",
)
@click.option(
    "--min-release-date-diff",
    default=60,
    show_default=True,
    type=click.IntRange(min=0),
    help="Days a template must predate each structure's release date. Excludes the "
    "structure itself from its own templates.",
)
@click.option(
    "--max-release-date",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Optional global template release-date cutoff (YYYY-MM-DD), on top of "
    "--min-release-date-diff.",
)
@click.option(
    "--template-cutoff-date",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Keep templates released before this date (YYYY-MM-DD), whatever each structure's own "
    "release date. Replaces the rule that a template must predate its structure, so it can "
    "show the model close homologs a real prediction would not have.",
)
@click.option("--max-templates", default=20, show_default=True, type=click.IntRange(min=1))
@click.option("--n-processes", default=1, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--ccd-path",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="CCD for parsing template structures. Default: Biotite's bundled CCD.",
)
@click.option(
    "--sequence-filter/--no-sequence-filter",
    default=True,
    show_default=True,
    help="Drop template hits that fail OF3's sequence filters (AF3 SI 2.4): near-full-"
    "length exact copies of the query, too short, or aligned to too little of it.",
)
@click.option(
    "--max-subseq",
    default=TemplateSequenceFilterSettings().max_subseq,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="Drop hits whose aligned segment is an exact substring of the query and longer "
    "than this fraction of it.",
)
@click.option(
    "--min-align",
    default=TemplateSequenceFilterSettings().min_align,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="Drop hits aligned to at most this fraction of the query residues.",
)
@click.option(
    "--min-len",
    default=TemplateSequenceFilterSettings().min_len,
    show_default=True,
    type=click.IntRange(min=0),
    help="Drop hits whose aligned template segment is shorter than this many residues.",
)
@with_pipeline_logs
def preprocess_templates(
    output_dir: Path,
    msa_dir: Path | None,
    cache_file: Path | None,
    min_release_date_diff: int,
    max_release_date,
    template_cutoff_date,
    max_templates: int,
    n_processes: int,
    ccd_path: Path | None,
    sequence_filter: bool,
    max_subseq: float,
    min_align: float,
    min_len: int,
) -> None:
    """Build OF3's template cache from fetch-templates' output and add template_ids.

    Drops hits failing OF3's template sequence filters, then runs OpenFold3's train-mode
    template preprocessing into <organized-msa-dir>/of3_template_data/.
    """
    msa_dir = require_organized_msa_dir(msa_dir, output_dir)
    if template_cutoff_date is not None and max_release_date is not None:
        raise click.UsageError("--template-cutoff-date replaces --max-release-date; give one")
    if template_cutoff_date is not None and not sequence_filter:
        raise click.UsageError(
            "--template-cutoff-date needs the sequence filters: without them a structure can "
            "be its own template"
        )
    cache_file = cache_file or output_dir / "dataset_cache.json"
    if not cache_file.exists():
        raise click.UsageError(f"{cache_file} not found; run materialize-dataset-cache first")
    summary = preprocess_templates_for_of3(
        output_dir,
        msa_dir,
        cache_file,
        min_release_date_diff=min_release_date_diff,
        max_release_date=max_release_date,
        template_cutoff_date=template_cutoff_date,
        max_templates=max_templates,
        n_processes=n_processes,
        ccd_path=ccd_path,
        sequence_filter=TemplateSequenceFilterSettings(
            enabled=sequence_filter, max_subseq=max_subseq, min_align=min_align, min_len=min_len
        ),
    )
    if sequence_filter:
        click.echo(
            f"Template sequence filters dropped {summary.n_sequence_filtered} hit(s) "
            f"(listed in {summary.sequence_filter_report_path})"
        )
    click.echo(
        f"Template cache under {msa_dir / TEMPLATE_DATA_DIRNAME}; "
        f"{summary.n_chains_with_templates} chain(s) in {cache_file} now have templates"
    )
    if summary.date_filtered_chains:
        click.echo(
            f"The following {len(summary.date_filtered_chains)} chain(s) lost every template "
            f"hit to the release-date filter: {', '.join(summary.date_filtered_chains)} "
            f"(per-chain counts: {summary.report_path})"
        )
