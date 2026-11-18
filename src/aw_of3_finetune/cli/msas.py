"""MSA commands: generate-msas (ColabFold) and import-msas (user MSAs)."""

from __future__ import annotations

from pathlib import Path

import click
from atomworks.ml.preprocessing.msa.colabfold_server import (
    make_msas_colabfold_server_batch,
)
from atomworks.ml.preprocessing.msa.finding import find_msas, find_paired_msas

from aw_of3_finetune.cli.preflight import check_paired_dirs_writable
from aw_of3_finetune.cli.options import (
    organized_msa_dir_option,
    output_dir_option,
    resolve_organized_msa_dir,
    with_pipeline_logs,
)
from aw_of3_finetune.constants import (
    COLABFOLD_UNPAIRED_DIRNAME,
)
from aw_of3_finetune.msa import (
    collect_complex_queries,
    collect_msa_queries,
)
from aw_of3_finetune.msa_sources import (
    MsaSourcesError,
    add_to_msa_index,
    import_user_msas,
    msa_dir_kind,
)
from aw_of3_finetune.pipeline import (
    ReorganizeSummary,
)

# import-msas lists at most this many unmatched files per source.
UNMATCHED_LISTING_CAP = 20


def reorganize_message(summary: ReorganizeSummary, of3_msa_dir: Path) -> str:
    """One line saying what reorganize_msas_for_of3 wrote, removed and could not find.

    Args:
        summary: Its result.
        of3_msa_dir: Where it wrote.

    Returns:
        E.g. "Reorganized 3 MSA(s) for OF3 under X (colabfold_main: 3); 1 sequence(s) missing an MSA".
    """
    total = sum(summary.written.values())
    per_source = ", ".join(f"{name}: {n}" for name, n in summary.written.items())
    message = f"Reorganized {total} MSA(s) for OF3 under {of3_msa_dir}" + (f" ({per_source})" if per_source else "")
    if summary.removed:
        message += f"; removed {summary.removed} stale MSA(s)"
    if summary.missing:
        message += f"; {summary.missing} sequence(s) missing an MSA"
    return message


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory produced by finetune-prep "
    "(must contain manifest.json and cleaned_structures/).",
)
@organized_msa_dir_option(
    exists=False,
    help="Folder to write MSA files to. "
    "Output layout: unpaired <organized-msa-dir>/unpaired/<shard>/<hash>.a3m.gz, "
    "paired <organized-msa-dir>/paired/<shard-of-complex_id>/<complex_id>/<hash>.a3m.gz.",
)
@click.option(
    "--server",
    default="https://api.colabfold.com",
    show_default=True,
    help="ColabFold MSA server URL.",
)
@click.option(
    "--user-agent",
    default="aw-of3-finetune",
    show_default=True,
    help="User-agent string sent with API requests.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip rep_ids that already have a colabfold_main.a3m file (useful for resuming).",
)
@click.option(
    "--templates/--no-templates",
    default=True,
    show_default=True,
    help="Also request template hits (PDB70) in the unpaired query, written to "
    "<organized-msa-dir>/templates/<shard>/<hash>.m8 (sequences with no hits get no file). "
    "Resolve their structures afterwards with fetch-templates. --skip-existing stays "
    "keyed on unpaired MSAs, so sequences whose MSAs predate enabling templates are "
    "not re-queried: rerun those without --skip-existing.",
)
@with_pipeline_logs
def generate_msas(
    output_dir: Path,
    msa_dir: Path | None,
    server: str,
    user_agent: str,
    skip_existing: bool,
    templates: bool,
) -> None:
    """Generate ColabFold MSAs for every unique protein sequence and complex in the dataset.

    Sequences come from SEQRES in the augmented CIFs (no SEQRES: skipped). Paired MSAs
    (complexes with >1 unique sequence) aren't consumed by any training pipeline yet.
    """
    queries = collect_msa_queries(output_dir)
    if not queries:
        raise click.UsageError(
            "No protein sequences found in augmented CIFs — nothing to align.\n"
            "Check that the CIFs have entity_poly SEQRES records."
        )

    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=False)
    try:
        kind = msa_dir_kind(msa_dir)
    except MsaSourcesError as error:
        raise click.ClickException(str(error)) from error
    if kind == "user":
        raise click.ClickException(
            f"{msa_dir} holds user-provided MSAs (import-msas); one MSA folder holds ColabFold or "
            "user MSAs, not both. Use a fresh --organized-msa-dir."
        )
    msa_dir.mkdir(parents=True, exist_ok=True)

    seqs = list(queries.values())
    if skip_existing:
        # "/0:2/" (make_msas_colabfold_server's default sharding_pattern) corresponds
        # to directory_depth=1 in find_msas's convention — one directory level of
        # 2 characters, not two levels.
        lookup_dir = {"dir": str(msa_dir / "unpaired"), "extension": ".a3m.gz", "directory_depth": 1}
        seqs, _ = find_msas(seqs, [lookup_dir])

    # Paired output is one query per multi-chain complex; complex_id is sharded like
    # sequence hashes, findable via atomworks' get_paired_msa_path.
    complexes = collect_complex_queries(output_dir)
    if skip_existing:
        paired_lookup_dir = {"dir": str(msa_dir / "paired"), "extension": ".a3m.gz"}
        complexes = {
            complex_id: complex_seqs
            for complex_id, complex_seqs in complexes.items()
            # Checks every chain in the complex, not just a proxy -- find_paired_msas
            # re-derives complex_id from complex_seqs itself, same as it would from
            # the a3m files this writes, so this is consistent with the writer by
            # construction rather than by convention.
            if find_paired_msas(complex_seqs, [paired_lookup_dir])[0]  # non-empty "missing" list
        }

    check_paired_dirs_writable(complexes, msa_dir, skip_existing)

    if not seqs and not complexes:
        add_to_msa_index(msa_dir, {"colabfold_main": COLABFOLD_UNPAIRED_DIRNAME})
        click.echo("All MSAs already exist (--skip-existing). Nothing to do.")
        return

    click.echo(
        f"Querying ColabFold ({server}) for {len(seqs)} unique sequence(s) "
        f"and {len(complexes)} unique complex(es)..."
    )
    make_msas_colabfold_server_batch(
        seqs,
        output_dir=msa_dir,
        complexes=list(complexes.values()),
        use_templates=templates,
        user_agent=user_agent,
        host_url=server,
    )
    add_to_msa_index(msa_dir, {"colabfold_main": COLABFOLD_UNPAIRED_DIRNAME})
    click.echo(f"Done. Unpaired under {msa_dir / 'unpaired'}, paired under {msa_dir / 'paired'}.")
    if templates:
        click.echo(f"Template hits under {msa_dir / 'templates'}; fetch-templates resolves their structures.")


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory produced by finetune-prep "
    "(must contain manifest.json and cleaned_structures/).",
)
@organized_msa_dir_option(
    exists=False,
    help="Folder to import the MSAs into: <organized-msa-dir>/msa_sources/<source>/, the source "
    "index msa_index.yml, and of3_alignments/ for OpenFold3. Must not hold ColabFold MSAs.",
)
@click.option(
    "--msa-sources",
    "manifest_path",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Either a YAML/JSON sources manifest mapping each MSA source name (the OpenFold3 alignment "
    "file name, e.g. uniref90_hits) to a directory or glob of your own A3M files, or the "
    "--organized-msa-dir of an earlier generate-msas/import-msas run, to reuse (link) its MSAs. "
    "A folder of your own A3M files needs a manifest. Files are matched to chains by their first "
    "(query) sequence.",
)
@with_pipeline_logs
def import_msas(output_dir: Path, msa_dir: Path | None, manifest_path: Path) -> None:
    """Import your own MSAs instead of querying the ColabFold server.

    Every file is checked first (query sequence, duplicates, OpenFold3's own A3M and
    pairing-header parsers); nothing is copied if any check fails. Then of3_alignments/
    is brought up to date, so generate-train-config can run next.
    """
    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=False)
    try:
        summary = import_user_msas(output_dir, msa_dir, manifest_path)
    except MsaSourcesError as error:
        raise click.ClickException(str(error)) from error
    for name, store in summary.linked.items():
        click.echo(f"{name}: linked {store}, covers {summary.covered[name]} dataset sequence(s)")
    if summary.linked:
        click.echo(
            "Linked, not copied: keep that folder where it is. Rebuilding of3_alignments/ "
            "(materialize-dataset-cache, or rerunning finetune-pipeline) reads the MSAs from it."
        )
    for scan in summary.scans:
        click.echo(
            f"{scan.source}: imported {len(scan.files)} file(s), "
            f"covers {summary.covered[scan.source]} dataset sequence(s)"
        )
        if scan.unmatched:
            shown = [str(path) for path in scan.unmatched[:UNMATCHED_LISTING_CAP]]
            more = len(scan.unmatched) - len(shown)
            listing = "\n    ".join(shown + ([f"... and {more} more"] if more else []))
            click.echo(
                f"  WARNING: {scan.source}: {len(scan.unmatched)} file(s) match no sequence in this "
                f"dataset and were skipped. If one should match, its query row is likely not the chain's SEQRES "
                f"sequence:\n    {listing}",
                err=True,
            )
        if scan.ignored:
            listing = "\n    ".join(str(p) for p in scan.ignored)
            click.echo(f"  {scan.source}: ignored {len(scan.ignored)} non-A3M file(s):\n    {listing}", err=True)
    click.echo(reorganize_message(summary.reorganize, msa_dir / "of3_alignments"))
    if summary.other_sources:
        click.echo(
            f"WARNING: {msa_dir / 'msa_index.yml'} also lists {', '.join(summary.other_sources)} from an "
            "earlier import; they stay in training. To remove one, delete its line from msa_index.yml "
            "and its msa_sources/<source>/ folder, then rerun import-msas.",
            err=True,
        )
    if summary.reorganize.missing:
        click.echo(
            f"WARNING: {summary.reorganize.missing} sequence(s) have no MSA from any source",
            err=True,
        )
