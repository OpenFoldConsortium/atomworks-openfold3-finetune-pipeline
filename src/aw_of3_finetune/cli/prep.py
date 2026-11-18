"""Commands that set up the CCD, prepare inputs and annotate structures: setup-finetuning, prepare-inputs, finetune-prep, materialize-dataset-cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from biotite.structure.io.pdbx import CIFFile

from aw_of3_finetune.cleanup import CleanupSettings, EmptyStructureError
from aw_of3_finetune.cli.checks import warn_missing_msas
from aw_of3_finetune.cli.msas import reorganize_message
from aw_of3_finetune.cli.options import (
    CCD_SOURCE_META_KEY,
    ccd_path_option,
    cleanup_config_option,
    organized_msa_dir_option,
    output_dir_option,
    resolve_organized_msa_dir,
    with_pipeline_logs,
)
from aw_of3_finetune.constants import (
    CLEANED_STRUCTURES_DIRNAME,
    PREP_DROPPED_FILE,
)
from aw_of3_finetune.ligand_inputs import CcdCodeCollisionError, read_ligand_table
from aw_of3_finetune.ligand_names import (
    SignaturesByName,
    format_name_conflicts,
    name_conflicts,
)
from aw_of3_finetune.msa_sources import (
    MsaSourcesError,
)
from aw_of3_finetune.openfold_cache import (
    CCD_S3_BUCKET,
    CCD_S3_KEY,
    CHECKPOINT_ROOT_FILENAME,
    openfold_cache_dir,
    read_stored_ccd_path,
    stored_ccd_path_file,
    write_stored_ccd_path,
)
from aw_of3_finetune.pipeline import (
    ChainMismatchError,
    build_manifest,
    check_chain_consistency,
    prepare_augmented_cif,
    reorganize_msas_for_of3,
    write_slim_manifest,
)
from aw_of3_finetune.prepare_inputs import (
    DEFAULT_RELEASE_DATE,
    PREPARE_DROPPED_FILE,
    PrepareInputsUsageError,
    prepare_directory,
    read_constructs,
)


@click.command()
@click.option(
    "--cif-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Directory of input mmCIF files.",
)
@output_dir_option(
    exists=False,
    help="Output directory. Augmented CIFs go to <output-dir>/cleaned_structures/, "
    "manifest to <output-dir>/manifest.json.",
)
@organized_msa_dir_option(
    exists=True,
    help="MSA folder. If it exists, chains with no MSA under "
    "<organized-msa-dir>/of3_alignments/<rep_id>/ are reported as warnings.",
)
@click.option(
    "--template-cache-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False),
    help="Pre-built OF3 template cache directory ({template-cache-dir}/{rep_id}.npz). "
    "Required only if you ran ColabFold/OF3 TemplatePreprocessor beforehand.",
)
@ccd_path_option(
    required=False,
    help="Path to the Chemical Component Dictionary mmCIF. "
    "When found, reference molecule SDFs are generated under "
    "<output-dir>/reference_mols/ and embedded in the manifest.",
)
@click.option(
    "--name",
    default="finetune",
    show_default=True,
    help="Dataset name written into manifest.json.",
)
@click.option(
    "--skip-inconsistent",
    is_flag=True,
    default=False,
    help="If a structure's chain metadata disagrees with its atoms (see "
    "check_chain_consistency), warn and leave it out of the manifest instead of "
    "failing. Default: fail on the first such structure.",
)
@click.option("--write-step-cifs/--no-write-step-cifs", default=False,
              help="Also write the atoms after each cleanup step as a CIF under "
              "<output-dir>/cleanup_steps/<pdb_id>/ (for viewing; default: off).")
@cleanup_config_option
@with_pipeline_logs
def finetune_prep(
    cif_dir: Path,
    output_dir: Path,
    msa_dir: Path | None,
    template_cache_dir: Path | None,
    ccd_path: Path | None,
    name: str,
    skip_inconsistent: bool,
    write_step_cifs: bool,
    cleanup_settings: CleanupSettings,
) -> None:
    """Prepare an augmented-CIF fine-tuning dataset.

    Writes <output-dir>/cleaned_structures/<pdb_id>.cif (with embedded chain metadata) per input
    CIF, plus a slim manifest.json. Run materialize-dataset-cache next.
    """
    cif_paths = sorted(cif_dir.glob("*.cif"))
    if not cif_paths:
        raise click.UsageError(f"No .cif files found in {cif_dir}")
    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=True)

    structures_dir = output_dir / CLEANED_STRUCTURES_DIRNAME
    structures_dir.mkdir(parents=True, exist_ok=True)
    recorded = {
        **cleanup_settings.model_dump(),
        "ccd_path": str(ccd_path.resolve()) if ccd_path is not None else None,
        "ccd_source": click.get_current_context().meta.get(CCD_SOURCE_META_KEY) if ccd_path is not None else None,
    }
    (output_dir / "cleanup_settings.json").write_text(json.dumps(recorded, indent=2))
    if not cleanup_settings.is_default():
        changed = ", ".join(f"{k}={v}" for k, v in cleanup_settings.non_default().items())
        click.echo(f"Cleanup settings differ from the defaults: {changed}")
    if (altloc_warning := cleanup_settings.altloc_warning()) is not None:
        click.echo(f"Warning: {altloc_warning}")

    ccd: CIFFile | None = None
    ref_mol_dir: Path | None = None
    mol_cache: dict = {}
    ligand_signatures: SignaturesByName = {}
    if ccd_path is not None:
        click.echo(f"Loading CCD from {ccd_path}...")
        ccd = CIFFile.read(ccd_path)
        ref_mol_dir = output_dir / "reference_mols"
        ref_mol_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids: list[str] = []
    dropped: list[dict] = []  # {"id", "reason"}, written to prep_dropped.json for the report
    click.echo(f"Processing {len(cif_paths)} structure(s)...")
    for cif_path in cif_paths:
        pdb_id = cif_path.stem.lower()

        if msa_dir is not None:
            warn_missing_msas(cif_path, pdb_id, msa_dir / "of3_alignments")

        # atomworks parses; OF3 only ingests what atomworks wrote (see
        # prepare_augmented_cif) -- the chain metadata is built from the same
        # atom array the NPZ will be, so the two can't disagree by construction.
        out_cif = structures_dir / f"{pdb_id}.cif"
        try:
            structure_data, _ = prepare_augmented_cif(
                cif_path, pdb_id, out_cif,
                ccd=ccd, ref_mol_dir=ref_mol_dir, mol_cache=mol_cache,
                template_cache_dir=template_cache_dir,
                cleanup_settings=cleanup_settings,
                step_log_dir=output_dir / "cleanup_steps",
                write_step_cifs=write_step_cifs,
                ligand_signatures=ligand_signatures,
            )
        except EmptyStructureError as exc:
            reason = f"empty after cleanup ({exc.removals[-1].step})"
            click.echo(f"WARNING: skipping {pdb_id} -- {reason}", err=True)
            out_cif.unlink(missing_ok=True)
            dropped.append({"id": pdb_id, "reason": reason})
            continue
        except Exception as exc:  # noqa: BLE001 — one bad structure must not stop the others
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            reason = f"failed: {type(exc).__name__}: {first_line}".rstrip(": ")
            click.echo(f"WARNING: skipping {pdb_id} -- {reason}", err=True)
            out_cif.unlink(missing_ok=True)
            dropped.append({"id": pdb_id, "reason": reason})
            continue
        pdb_ids.append(pdb_id)

        # Belt-and-braces: re-derive the chains from the written file and confirm
        # they match the metadata. Tautological given the construction above, but
        # cheap, and it's the invariant training depends on.
        try:
            check_chain_consistency(out_cif, structure_data)
        except ChainMismatchError as exc:
            if not skip_inconsistent:
                raise click.ClickException(str(exc)) from exc
            click.echo(f"WARNING: skipping {pdb_id} -- {exc}", err=True)
            out_cif.unlink()
            pdb_ids.remove(pdb_id)
            dropped.append({"id": pdb_id, "reason": f"chain metadata disagrees with the atoms: {exc}"})
            continue

        click.echo(f"  {pdb_id} → {out_cif.name}")

    conflicts = name_conflicts(ligand_signatures)
    if conflicts:
        raise click.ClickException(format_name_conflicts(conflicts))

    manifest_path = output_dir / "manifest.json"
    write_slim_manifest(pdb_ids, manifest_path, cache_name=name)
    (output_dir / PREP_DROPPED_FILE).write_text(json.dumps(dropped, indent=2))
    click.echo(f"Manifest: {manifest_path} ({len(pdb_ids)} structure(s))")
    click.echo(f"Cleaned structures: {structures_dir} ({len(pdb_ids)} augmented CIF(s))")
    if dropped:
        click.echo(f"Dropped {len(dropped)} structure(s), reasons in {output_dir / PREP_DROPPED_FILE}: "
                   f"{', '.join(item['id'] for item in dropped)}")


@click.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Directory of input .pdb and .cif files; the file stem is the structure id.",
)
@click.option(
    "--ligands",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Ligand table CSV: structure_id, res_name, smiles, optional restraint_cif.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory: <stem>.cif per input plus prepare_dropped.json, ligands.json "
    "and prepare_inputs.json. Use it as --cif-dir for finetune-prep.",
)
@ccd_path_option(required=True, help="Path to the Chemical Component Dictionary mmCIF.")
@click.option(
    "--release-date",
    default=DEFAULT_RELEASE_DATE,
    show_default=True,
    help="YYYY-MM-DD for inputs that carry no revision date.",
)
@click.option(
    "--constructs",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="FASTA of the whole sequences crystallised, unmodelled residues included. Needed for "
    "any chain with gaps in its residue numbers; such a chain is dropped without one.",
)
@with_pipeline_logs
def prepare_inputs(
    input_dir: Path,
    ligands: Path,
    output_dir: Path,
    ccd_path: Path,
    release_date: str,
    constructs: Path | None,
) -> None:
    """Turn in-house .pdb/.cif structures and a ligand table into finetune-prep inputs.

    A structure that cannot be prepared is dropped with its reason in prepare_dropped.json.
    """
    try:
        rows = read_ligand_table(ligands)
        sequences = read_constructs(constructs) if constructs is not None else ()
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    click.echo(f"Loading CCD from {ccd_path}...")
    ccd = CIFFile.read(ccd_path)
    try:
        record = prepare_directory(input_dir, rows, output_dir, ccd, release_date, sequences)
    except PrepareInputsUsageError as error:
        raise click.UsageError(str(error)) from error
    except CcdCodeCollisionError as error:
        raise click.ClickException(str(error)) from error
    dropped = json.loads((output_dir / PREPARE_DROPPED_FILE).read_text())
    for stem, reason in dropped.items():
        click.echo(f"WARNING: dropped {stem} -- {reason}", err=True)
    click.echo(
        f"prepared {len(record['inputs'])}, dropped {len(dropped)} "
        f"(see {PREPARE_DROPPED_FILE})"
    )


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory produced by finetune-prep "
    "(must contain manifest.json and cleaned_structures/).",
)
@click.option(
    "--cache-output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to write the full dataset cache JSON. "
    "Defaults to <output-dir>/dataset_cache.json.",
)
@organized_msa_dir_option(
    exists=True,
    help="MSA folder written by generate-msas. If it exists, unpaired MSAs are also "
    "reorganized into <organized-msa-dir>/of3_alignments/<rep_id>/colabfold_main.a3m "
    "-- the plain-text, directory-per-rep_id layout OF3's WeightedPDBDataset "
    "(alignments_directory) expects, since it cannot read atomworks' native "
    "gzipped/sharded layout directly. Pass <organized-msa-dir>/of3_alignments as "
    "alignments_directory to WeightedPDBDataset, not the folder itself.",
)
@with_pipeline_logs
def materialize_dataset_cache(
    output_dir: Path, cache_output: Path | None, msa_dir: Path | None
) -> None:
    """Materialize the full ClusteredDatasetCache JSON from finetune-prep's augmented CIFs.

    The output is ephemeral: regenerate it before training rather than committing it.
    """
    slim_path = output_dir / "manifest.json"
    if not slim_path.exists():
        raise click.UsageError(
            f"No manifest.json found in {output_dir}; run finetune-prep first"
        )
    slim = json.loads(slim_path.read_text())
    cache_name = slim.get("name", "finetune")

    cif_dir = output_dir / CLEANED_STRUCTURES_DIRNAME
    if not cif_dir.is_dir():
        raise click.UsageError(f"No cleaned_structures/ directory found in {output_dir}")

    if cache_output is None:
        cache_output = output_dir / "dataset_cache.json"

    build_manifest(cif_dir, cache_output, cache_name)
    n = len(list(cif_dir.glob("*.cif")))
    click.echo(f"Dataset cache: {cache_output} ({n} structure(s))")

    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=True)
    if msa_dir is not None:
        of3_msa_dir = msa_dir / "of3_alignments"
        try:
            summary = reorganize_msas_for_of3(output_dir, msa_dir, of3_msa_dir)
        except MsaSourcesError as error:
            raise click.ClickException(str(error)) from error
        click.echo(reorganize_message(summary, of3_msa_dir))


@click.command()
@click.option(
    "--ccd-path",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Record this existing CCD mmCIF instead of downloading one.",
)
@click.option(
    "--download-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False),
    help="Where to download the CCD. Default: the OpenFold3 cache directory ($OPENFOLD_CACHE, else ~/.openfold3).",
)
@click.option("--force", is_flag=True, default=False,
              help="Download and record the CCD again even if a stored path already points at a file.")
def setup_finetuning(ccd_path: Path | None, download_dir: Path | None, force: bool) -> None:
    """One-time setup: download OpenFold3's CCD and store its path in the OpenFold3 cache directory.

    Afterwards, commands that need the CCD (finetune-pipeline, prepare-inputs, finetune-report)
    find it without --ccd-path. The path is written to $OPENFOLD_CACHE/ccd_path (default
    ~/.openfold3/ccd_path), next to the ckpt_root file that OpenFold3's setup_openfold writes.
    """
    cache_dir = openfold_cache_dir()
    stored = read_stored_ccd_path()
    if ccd_path is None and not force and stored is not None and stored.is_file():
        click.echo(f"A CCD is already set up: {stored} (stored in {stored_ccd_path_file()}). "
                   "Pass --force to download it again, or --ccd-path to record another file.")
        return
    if ccd_path is None:
        ccd_path = (download_dir or cache_dir).expanduser() / CCD_S3_KEY
        if force or not ccd_path.is_file():
            # Imported here: openfold3's S3 client is only needed for this download.
            from openfold3.core.utils.s3 import download_s3_file

            click.echo(f"Downloading OpenFold3's CCD (about 430 MB) to {ccd_path}...")
            download_s3_file(CCD_S3_BUCKET, CCD_S3_KEY, ccd_path)
        else:
            click.echo(f"Using the CCD already downloaded at {ccd_path}.")
    record = write_stored_ccd_path(ccd_path)
    click.echo(f"Stored the CCD path in {record}.")
    if os.environ.get("OPENFOLD_CACHE"):
        click.echo(f"$OPENFOLD_CACHE is set to {cache_dir}: set it to the same value in each new "
                   "shell, or the pipeline will not find this CCD.")
    if not (cache_dir / CHECKPOINT_ROOT_FILENAME).is_file():
        click.echo("OpenFold3's model weights are not set up in this cache directory yet; "
                   "run `pixi run setup_openfold` to download them.")
