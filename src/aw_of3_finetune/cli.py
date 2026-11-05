"""CLI commands for the atomworks fine-tuning dataset pipeline."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import click
import yaml
from atomworks.ml.preprocessing.msa.colabfold_server import (
    make_msas_colabfold_server_batch,
)
from atomworks.io.utils.io_utils import apply_sharding_pattern
from atomworks.ml.preprocessing.msa.finding import find_msas, find_paired_msas
from atomworks.ml.utils.misc import get_complex_id
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from pydantic import ValidationError

from aw_of3_finetune.cleanup import CleanupSettings, EmptyStructureError
from aw_of3_finetune.constants import PREP_DROPPED_FILE
from aw_of3_finetune.ligand_inputs import CcdCodeCollisionError, read_ligand_table
from aw_of3_finetune.ligand_names import SignaturesByName, format_name_conflicts, name_conflicts
from aw_of3_finetune.msa import (
    alignment_rep_id,
    chain_canonical_sequences,
    collect_complex_queries,
    collect_msa_queries,
)
from aw_of3_finetune.prepare_inputs import (
    DEFAULT_RELEASE_DATE,
    PREPARE_DROPPED_FILE,
    PrepareInputsUsageError,
    prepare_directory,
    read_constructs,
)
from aw_of3_finetune.report.build import REPORT_SUBDIR
from aw_of3_finetune.report.summary import LARGE_DATASET, TABLE_ROW_CAP
from aw_of3_finetune.report.user import ReportError, check_report_dir, write_user_report
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.pipeline import (
    ChainMismatchError,
    build_manifest,
    check_chain_consistency,
    prepare_augmented_cif,
    reorganize_msas_for_of3,
    save_structures_as_npz,
    write_slim_manifest,
)
from aw_of3_finetune.split import (
    DATASET_CACHE_NAME,
    SPLIT_DIRNAME,
    SplitCsvError,
    read_split_csv,
    split_csv_ids,
    split_dataset_cache,
    split_paths,
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
from aw_of3_finetune.training_config import LEGACY_CACHE_WARNING, is_legacy_dataset_cache, write_train_config
from aw_of3_finetune.validate import CHECKS, Finding, Severity, ValidationContext, run_validation


logger = logging.getLogger(__name__)


class _ClickEchoHandler(logging.Handler):
    """Log handler that writes through click.echo, so CliRunner and pipes capture it."""

    def emit(self, record: logging.LogRecord) -> None:
        click.echo(self.format(record), err=True)


def _with_pipeline_logs(command: Callable) -> Callable:
    """Show this package's INFO-level log lines (every processing step) on stderr.

    Args:
        command: A click command callback.

    Returns:
        The callback, wrapped to configure logging first (idempotent).
    """

    @wraps(command)
    def wrapper(*args, **kwargs):
        log = logging.getLogger("aw_of3_finetune")
        if not any(isinstance(h, _ClickEchoHandler) for h in log.handlers):
            handler = _ClickEchoHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            log.addHandler(handler)
        log.setLevel(logging.INFO)
        return command(*args, **kwargs)

    return wrapper


def _load_cleanup_settings(
    ctx: click.Context, param: click.Parameter, value: Path | None
) -> CleanupSettings:
    """Click callback: read --cleanup-config into CleanupSettings (defaults if not given).

    Args:
        ctx: Click context.
        param: The option.
        value: Path to the YAML/JSON config, or None.

    Returns:
        The cleanup settings.
    """
    if value is None:
        return CleanupSettings()
    try:
        return CleanupSettings.from_file(value)
    except (ValidationError, yaml.YAMLError) as exc:
        raise click.BadParameter(f"{value}: {exc}", ctx=ctx, param=param) from exc


cleanup_config_option = click.option(
    "--cleanup-config",
    "cleanup_settings",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    callback=_load_cleanup_settings,
    help="YAML/JSON file switching structure-cleanup steps off (see "
    "examples/cleanup_settings.yml). Default: OpenFold3's cleanup, every step on.",
)


def _check_paired_dirs_writable(
    complexes: dict[str, list[str]], msa_dir: Path, skip_existing: bool
) -> None:
    """Fail before querying ColabFold if a paired-MSA directory to be written exists.

    atomworks never writes into an existing complex directory (it would mix chains
    from two searches), so finding this out after the query wastes the whole query.

    Args:
        complexes: Complexes about to be queried, {complex_id: sequences}.
        msa_dir: generate-msas output directory.
        skip_existing: Whether --skip-existing was given (complete MSAs already removed).

    Raises:
        click.UsageError: If any of those complexes already has a directory.
    """
    existing = [
        complex_dir
        for seqs in complexes.values()
        if (complex_dir := msa_dir / "paired" / apply_sharding_pattern(get_complex_id(seqs), "/0:2/")).exists()
    ]
    if not existing:
        return
    listing = "\n  ".join(str(d) for d in existing)
    if skip_existing:
        raise click.UsageError(
            f"{len(existing)} paired MSA director(ies) are incomplete (left by an "
            f"interrupted run?):\n  {listing}\ndelete them and rerun to regenerate."
        )
    raise click.UsageError(
        f"{len(existing)} paired MSA(s) already exist under {msa_dir / 'paired'}:\n  "
        f"{listing}\nRerun with --skip-existing to reuse them, or use a fresh --msa-dir."
    )


@click.command()
@click.option(
    "--cif-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Directory of input mmCIF files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory. Augmented CIFs go to <output-dir>/structures/, "
    "manifest to <output-dir>/manifest.json.",
)
@click.option(
    "--msa-dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="MSA alignments directory ({msa-dir}/{rep_id}/*.a3m). "
    "If provided, chains missing MSAs are reported as warnings.",
)
@click.option(
    "--template-cache-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False),
    help="Pre-built OF3 template cache directory ({template-cache-dir}/{rep_id}.npz). "
    "Required only if you ran ColabFold/OF3 TemplatePreprocessor beforehand.",
)
@click.option(
    "--ccd-path",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the Chemical Component Dictionary mmCIF. "
    "When provided, reference molecule SDFs are generated under "
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
@_with_pipeline_logs
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

    Writes <output-dir>/structures/<pdb_id>.cif (with embedded chain metadata) per input
    CIF, plus a slim manifest.json. Run materialize-dataset-cache next.
    """
    cif_paths = sorted(cif_dir.glob("*.cif"))
    if not cif_paths:
        raise click.UsageError(f"No .cif files found in {cif_dir}")

    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleanup_settings.json").write_text(cleanup_settings.model_dump_json(indent=2))
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
            _warn_missing_msas(cif_path, pdb_id, msa_dir)

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
@click.option(
    "--ccd-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the Chemical Component Dictionary mmCIF.",
)
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
@_with_pipeline_logs
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


def _warn_missing_msas(cif_path: Path, pdb_id: str, msa_dir: Path) -> None:
    """Warn on stderr for any polymer chain without an MSA directory.

    Args:
        cif_path: Source mmCIF to read SEQRES sequences from.
        pdb_id: Structure ID, for the message.
        msa_dir: Alignments directory laid out as ``{msa_dir}/{rep_id}/*.a3m``.
    """
    for label_asym_id, seq in chain_canonical_sequences(cif_path).items():
        rep_id = alignment_rep_id(seq)
        msa_subdir = msa_dir / rep_id
        if not (msa_subdir.is_dir() and any(msa_subdir.glob("*.a3m"))):
            click.echo(
                f"  WARNING: {pdb_id} chain {label_asym_id} — no MSA found "
                f"(expected {msa_subdir})",
                err=True,
            )


@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Pipeline output directory produced by finetune-prep "
    "(must contain manifest.json and structures/).",
)
@click.option(
    "--cache-output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to write the full dataset cache JSON. "
    "Defaults to <output-dir>/dataset_cache.json.",
)
@click.option(
    "--msa-dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="MSA directory previously written by generate-msas. If provided, unpaired "
    "MSAs are also reorganized into {msa-dir}/of3_alignments/{rep_id}/colabfold_main.a3m "
    "-- the plain-text, directory-per-rep_id layout OF3's WeightedPDBDataset "
    "(alignments_directory) expects, since it cannot read atomworks' native "
    "gzipped/sharded layout directly. Pass {msa-dir}/of3_alignments as "
    "alignments_directory to WeightedPDBDataset, not --msa-dir itself.",
)
@_with_pipeline_logs
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

    cif_dir = output_dir / "structures"
    if not cif_dir.is_dir():
        raise click.UsageError(f"No structures/ directory found in {output_dir}")

    if cache_output is None:
        cache_output = output_dir / "dataset_cache.json"

    build_manifest(cif_dir, cache_output, cache_name)
    n = len(list(cif_dir.glob("*.cif")))
    click.echo(f"Dataset cache: {cache_output} ({n} structure(s))")

    if msa_dir is not None:
        of3_msa_dir = msa_dir / "of3_alignments"
        written, missing = reorganize_msas_for_of3(output_dir, msa_dir, of3_msa_dir)
        click.echo(
            f"Reorganized {written} MSA(s) for OF3 under {of3_msa_dir}"
            + (f" ({missing} sequence(s) missing an MSA)" if missing else "")
        )


@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Pipeline output directory (must contain dataset_cache.json, "
    "structure_files/, reference_mols/ -- i.e. materialize-dataset-cache and "
    "save_structures_as_npz must already have run).",
)
@click.option(
    "--msa-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="MSA directory previously written by generate-msas / reorganized by "
    "materialize-dataset-cache. Required: alignments_directory in the generated "
    "config points at {msa-dir}/of3_alignments, and OF3's own TrainingDatasetPaths "
    "schema requires exactly one alignment path to be set -- there's no valid "
    "config without it.",
)
@click.option(
    "--config-output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to write the runner-yaml. Defaults to <output-dir>/train.yml.",
)
@click.option(
    "--wandb-project",
    default=None,
    help="Log the training run to this Weights & Biases project (OF3's only "
    "config-driven logger). Off if omitted. Needs a wandb login on the training machine.",
)
@click.option("--wandb-entity", default=None, help="wandb user or team. Only used with --wandb-project.")
@click.option(
    "--wandb-name",
    default=None,
    help="wandb run name. Defaults to the output directory's name. Only used with --wandb-project.",
)
@_with_pipeline_logs
def generate_train_config(
    output_dir: Path,
    msa_dir: Path,
    config_output: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_name: str | None,
) -> None:
    """Write an OF3 training runner-yaml pointed at this pipeline's output.

    Smoke-test defaults (see training_config.py); adds a validation section when a
    current split/ exists (split-dataset). Hand-edit for real runs.
    Run from openfold-3 with:
    pixi run -e openfold3-cuda12 run_openfold train --runner-yaml=<config-output>
    """
    config_path = write_train_config(
        output_dir, msa_dir, config_output,
        wandb_project=wandb_project, wandb_entity=wandb_entity, wandb_experiment_name=wandb_name,
    )
    click.echo(f"Training config: {config_path}")
    cache_path = output_dir / "dataset_cache.json"
    if cache_path.exists() and is_legacy_dataset_cache(cache_path):
        click.echo(LEGACY_CACHE_WARNING, err=True)


@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Pipeline output directory containing dataset_cache.json. The split is written to <output-dir>/split/.",
)
@click.option(
    "--split-csv",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="CSV with header id,split. split is val or test; unlisted entries go to train.",
)
@click.option(
    "--msa-dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="MSA directory. If given, (re)writes <output-dir>/train.yml with default settings and the "
    "validation section. For wandb or custom options, run generate-train-config instead.",
)
@click.pass_context
@_with_pipeline_logs
def split_dataset(ctx: click.Context, output_dir: Path, split_csv: Path, msa_dir: Path | None) -> None:
    """Split the dataset cache: val and test IDs come from the CSV, the rest train.

    Test IDs are left out of both train and val. Outputs go to <output-dir>/split/.

    \b
    CSV (see examples/split_template.csv):
      id,split
      7riv,val
      1a3n,test

    \b
    Examples:
      split-dataset --output-dir out/ --split-csv my_split.csv --msa-dir msas/
      split-dataset --output-dir out/ --split-csv my_split.csv
      generate-train-config --output-dir out/ --msa-dir msas/
    """
    result = split_dataset_cache(output_dir, split_csv)
    for warning in result.warnings:
        click.echo(warning, err=True)
    click.echo(result.report)
    if msa_dir is None:
        click.echo(
            f"Next: generate-train-config --output-dir {output_dir} --msa-dir <msa-dir> "
            "(adds the validation section to train.yml)"
        )
        return
    ctx.invoke(
        generate_train_config,
        output_dir=output_dir,
        msa_dir=msa_dir,
        config_output=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_name=None,
    )


@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Pipeline output directory produced by finetune-prep "
    "(must contain manifest.json and structures/).",
)
@click.option(
    "--msa-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory to write MSA files. "
    "Output layout: unpaired <msa-dir>/unpaired/<shard>/<hash>.a3m.gz, "
    "paired <msa-dir>/paired/<shard-of-complex_id>/<complex_id>/<hash>.a3m.gz.",
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
    "<msa-dir>/templates/<shard>/<hash>.m8 (sequences with no hits get no file). "
    "Resolve their structures afterwards with fetch-templates. --skip-existing stays "
    "keyed on unpaired MSAs, so sequences whose MSAs predate enabling templates are "
    "not re-queried: rerun those without --skip-existing.",
)
@_with_pipeline_logs
def generate_msas(
    output_dir: Path,
    msa_dir: Path,
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

    _check_paired_dirs_writable(complexes, msa_dir, skip_existing)

    if not seqs and not complexes:
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
    click.echo(f"Done. Unpaired under {msa_dir / 'unpaired'}, paired under {msa_dir / 'paired'}.")
    if templates:
        click.echo(f"Template hits under {msa_dir / 'templates'}; fetch-templates resolves their structures.")


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
@click.option("--output-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Pipeline output directory (dataset_cache.json, structure_files/, ...).")
@click.option("--cif-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Directory of the original input mmCIF files.")
@click.option("--msa-dir", default=None, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="MSA directory (holds of3_alignments/ and template_structures/).")
@click.option("--template-cache-dir", default=None, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="OF3 template cache directory ({rep_id}.npz). Defaults to <msa-dir>/of3_template_data/template_cache.")
@click.option("--template-structure-dir", default=None, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Template structure directory. Defaults to <msa-dir>/template_structures.")
@click.option("--ccd-path", default=None, type=click.Path(path_type=Path, dir_okay=False),
              help="Chemical Component Dictionary mmCIF. Reserved; currently unused by checks.")
@click.option("--cache-path", default=None, type=click.Path(path_type=Path, dir_okay=False),
              help="Dataset cache JSON. Defaults to <output-dir>/dataset_cache.json.")
@click.option("--structures-npz-dir", default=None, type=click.Path(path_type=Path, file_okay=False),
              help="NPZ structure files root. Defaults to <output-dir>/structure_files.")
@click.option("--expect-ligands/--no-expect-ligands", default=False, show_default=True,
              help="Warn (V4) about entries without any ligand.")
@click.option("--allow-errors/--fail-on-errors", default=False,
              help="Drop failing entries from the dataset cache and manifest.json and exit 0, "
              "instead of exiting 1. Structure/SDF/MSA files are untouched; cache and manifest are "
              "rewritten (backups saved as *.pre-validate.json). Refuses when every entry fails "
              "or on dataset-level errors.")
@click.option("--skip-dataset-load", is_flag=True, default=False,
              help="Skip V21 (loading every entry through OF3's dataset; slow).")
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
    structures_npz_dir: Path | None,
    expect_ligands: bool,
    allow_errors: bool,
    skip_dataset_load: bool,
) -> None:
    """Validate a prepared fine-tuning dataset; writes <output-dir>/validation.json.

    Exits 1 if any entry has an error, unless --allow-errors drops those entries.
    """
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
        npz_dir=structures_npz_dir,
    )
    checks = [c for c in CHECKS if not (skip_dataset_load and c == "V21")]
    report = run_validation(vctx, checks)
    if skip_dataset_load:
        report.findings.append(Finding("V21", Severity.INFO, None, "skipped: --skip-dataset-load"))
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


def _run_split_stage(output_dir: Path, cif_dir: Path, split_csv: Path) -> None:
    """Split the freshly built cache, naming CSV ids that finetune-prep dropped.

    Args:
        output_dir: Pipeline output directory.
        cif_dir: Input CIF directory.
        split_csv: The user's `id,split` CSV.
    """
    cache_ids = list(read_datacache(output_dir / "dataset_cache.json").structure_data)
    dropped = {p.stem.lower() for p in cif_dir.glob("*.cif")} - {i.lower() for i in cache_ids}
    read_split_csv(split_csv, known_ids=cache_ids, dropped_ids=dropped)
    result = split_dataset_cache(output_dir, split_csv)
    for warning in result.warnings:
        click.echo(warning, err=True)
    click.echo(result.report)


@click.command()
@click.option(
    "--cif-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Directory of input mmCIF files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for every pipeline artifact: augmented CIFs, manifest, "
    "reference mols, dataset cache, and NPZ structure files.",
)
@click.option(
    "--msa-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="MSA directory. If given, runs generate-msas (live ColabFold queries) and "
    "reorganizes unpaired MSAs for OF3. If omitted, the MSA stage is skipped "
    "entirely -- useful if MSAs already exist from a previous run.",
)
@click.option(
    "--ccd-path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the Chemical Component Dictionary mmCIF. "
    "Required (unlike finetune-prep's own --ccd-path, which is optional there): "
    "without it, the augmented CIF never gets a chem_comp category embedded, and "
    "Stage 4's NPZ conversion (OF3's parse_mmcif) always fails with "
    "KeyError: 'chem_comp' as a result -- so there's no point letting this "
    "pipeline get partway through without it.",
)
@click.option(
    "--template-cache-dir",
    default=None,
    type=click.Path(path_type=Path, file_okay=False),
    help="Pre-built OF3 template cache directory ({template-cache-dir}/{rep_id}.npz). "
    "Required only if you ran ColabFold/OF3 TemplatePreprocessor beforehand.",
)
@click.option(
    "--name",
    default="finetune",
    show_default=True,
    help="Dataset name written into manifest.json.",
)
@click.option(
    "--server",
    default="https://api.colabfold.com",
    show_default=True,
    help="ColabFold MSA server URL. Only used if --msa-dir is given.",
)
@click.option(
    "--user-agent",
    default="aw-of3-finetune",
    show_default=True,
    help="User-agent string sent with API requests. Only used if --msa-dir is given.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip sequences/complexes that already have an MSA. Only used if --msa-dir "
    "is given.",
)
@click.option(
    "--template-cutoff-date",
    default=None,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Keep templates released before this date (YYYY-MM-DD), whatever each structure's own "
    "release date. Replaces the rule that a template must predate its structure, so it can "
    "show the model close homologs a real prediction would not have.",
)
@click.option(
    "--templates/--no-templates",
    default=True,
    show_default=True,
    help="Request template hits in generate-msas, then run fetch-templates and "
    "preprocess-templates (downloads hit structures from RCSB). Only used if --msa-dir "
    "is given.",
)
@click.option(
    "--cache-output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to write the full dataset cache JSON. "
    "Defaults to <output-dir>/dataset_cache.json.",
)
@click.option(
    "--structures-npz-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Where to write NPZ structure files for the training dataloader. "
    "Defaults to <output-dir>/structure_files.",
)
@click.option(
    "--skip-inconsistent",
    is_flag=True,
    default=False,
    help="Passed through to finetune-prep: warn and drop structures whose chain "
    "metadata disagrees with their atoms, instead of failing the whole run.",
)
@click.option(
    "--allow-errors",
    is_flag=True,
    default=False,
    help="Passed through to validate: drop failing entries instead of exiting 1.",
)
@click.option(
    "--split-csv",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="CSV with header id,split (see examples/split_template.csv). Adds a split-dataset stage and a "
    "validation section to train.yml. The CSV is checked against the input CIFs before any stage runs.",
)
@click.option("--skip-validation", is_flag=True, default=False, help="Skip the final validate stage.")
@click.option("--skip-report", is_flag=True, default=False,
              help="Skip the report stage (<output-dir>/report/index.html; rebuild it later with finetune-report).")
@click.option("--skip-dataset-load", is_flag=True, default=False,
              help="Passed through to validate: skip V21 (loading every entry through OF3's dataset; slow).")
@click.option("--wandb-project", default=None, help="Passed through to generate-train-config.")
@click.option("--wandb-entity", default=None, help="Passed through to generate-train-config.")
@click.option("--wandb-name", default=None, help="Passed through to generate-train-config.")
@click.option("--write-step-cifs/--no-write-step-cifs", default=False,
              help="Also write the atoms after each cleanup step as a CIF under "
              "<output-dir>/cleanup_steps/<pdb_id>/ (for viewing; default: off).")
@cleanup_config_option
@click.pass_context
@_with_pipeline_logs
def finetune_pipeline(
    ctx: click.Context,
    cif_dir: Path,
    output_dir: Path,
    msa_dir: Path | None,
    ccd_path: Path,
    template_cache_dir: Path | None,
    name: str,
    server: str,
    user_agent: str,
    skip_existing: bool,
    template_cutoff_date,
    templates: bool,
    cache_output: Path | None,
    structures_npz_dir: Path | None,
    skip_inconsistent: bool,
    allow_errors: bool,
    skip_validation: bool,
    skip_report: bool,
    skip_dataset_load: bool,
    split_csv: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_name: str | None,
    write_step_cifs: bool,
    cleanup_settings: CleanupSettings,
) -> None:
    """Run every pipeline stage end-to-end over a directory of CIFs.

    finetune-prep -> generate-msas -> fetch-templates -> materialize-dataset-cache ->
    preprocess-templates -> NPZ -> split-dataset (with --split-csv) -> generate-train-config ->
    validate -> report. All but finetune-prep, materialize, NPZ, validate and report need
    --msa-dir; the template stages also need --templates.
    """
    run_templates = msa_dir is not None and templates
    if split_csv is not None:
        default_cache = output_dir / DATASET_CACHE_NAME
        if cache_output is not None and cache_output.resolve() != default_cache.resolve():
            raise click.ClickException(
                f"--split-csv needs the dataset cache at {default_cache} (split-dataset and "
                "generate-train-config read it there); drop --cache-output or set it to that path."
            )
        # Fail on a typo now, not after the MSA and template stages.
        read_split_csv(split_csv, known_ids={p.stem.lower() for p in cif_dir.glob("*.cif")})
    elif (output_dir / SPLIT_DIRNAME).exists():
        raise click.ClickException(
            f"{output_dir / SPLIT_DIRNAME}/ is left over from an earlier run and would no longer match "
            "the rebuilt dataset cache. Pass --split-csv to re-split, or delete "
            f"{output_dir / SPLIT_DIRNAME}/ to train without validation."
        )
    if not skip_report:  # refuse a report dir that is not an old report before doing any work
        try:
            check_report_dir(ValidityInputs(output_dir, cif_dir, ccd_path, structures_npz_dir), output_dir / REPORT_SUBDIR)
        except ReportError as error:
            raise click.ClickException(str(error)) from error
    click.echo("=== Stage 1/10: finetune-prep ===")
    ctx.invoke(
        finetune_prep,
        cif_dir=cif_dir,
        output_dir=output_dir,
        msa_dir=None,  # MSAs don't exist yet at this point -- nothing to warn about.
        template_cache_dir=template_cache_dir,
        ccd_path=ccd_path,
        name=name,
        skip_inconsistent=skip_inconsistent,
        write_step_cifs=write_step_cifs,
        cleanup_settings=cleanup_settings,
    )

    if msa_dir is not None:
        click.echo("\n=== Stage 2/10: generate-msas ===")
        ctx.invoke(
            generate_msas,
            output_dir=output_dir,
            msa_dir=msa_dir,
            server=server,
            user_agent=user_agent,
            skip_existing=skip_existing,
            templates=templates,
        )
    else:
        click.echo("\n=== Stage 2/10: generate-msas -- skipped (no --msa-dir given) ===")

    if run_templates:
        click.echo("\n=== Stage 3/10: fetch-templates ===")
        ctx.invoke(fetch_templates, output_dir=output_dir, msa_dir=msa_dir)
    else:
        click.echo("\n=== Stage 3/10: fetch-templates -- skipped (no --msa-dir or --no-templates) ===")

    click.echo("\n=== Stage 4/10: materialize-dataset-cache ===")
    ctx.invoke(
        materialize_dataset_cache,
        output_dir=output_dir,
        cache_output=cache_output,
        msa_dir=msa_dir,
    )

    if run_templates:
        click.echo("\n=== Stage 5/10: preprocess-templates ===")
        ctx.invoke(
            preprocess_templates,
            output_dir=output_dir,
            msa_dir=msa_dir,
            cache_file=cache_output,
            template_cutoff_date=template_cutoff_date,
        )
    else:
        click.echo(
            "\n=== Stage 5/10: preprocess-templates -- skipped (no --msa-dir or --no-templates) ==="
        )

    click.echo("\n=== Stage 6/10: NPZ structure files ===")
    npz_dir = structures_npz_dir if structures_npz_dir is not None else output_dir / "structure_files"
    save_structures_as_npz(output_dir / "structures", npz_dir)
    n = len(list((output_dir / "structures").glob("*.cif")))
    click.echo(f"NPZ structure files: {npz_dir} ({n} structure(s))")

    if split_csv is not None:
        click.echo("\n=== Stage 7/10: split-dataset ===")
        _run_split_stage(output_dir, cif_dir, split_csv)
    else:
        click.echo("\n=== Stage 7/10: split-dataset -- skipped (no --split-csv) ===")

    if msa_dir is not None:
        click.echo("\n=== Stage 8/10: generate-train-config ===")
        ctx.invoke(
            generate_train_config,
            output_dir=output_dir,
            msa_dir=msa_dir,
            config_output=None,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_name=wandb_name,
        )
    else:
        click.echo(
            "\n=== Stage 8/10: generate-train-config -- skipped (no --msa-dir given; "
            "OF3's training schema requires a real alignments directory, so no "
            "valid train.yml can be produced without one) ==="
        )

    validate_exit = None
    if skip_validation:
        click.echo("\n=== Stage 9/10: validate -- skipped (--skip-validation) ===")
    else:
        click.echo("\n=== Stage 9/10: validate ===")
        try:
            ctx.invoke(
                validate,
                output_dir=output_dir,
                cif_dir=cif_dir,
                msa_dir=msa_dir,
                template_cache_dir=template_cache_dir,
                template_structure_dir=None,
                ccd_path=ccd_path,
                cache_path=cache_output if cache_output is not None else output_dir / "dataset_cache.json",
                structures_npz_dir=npz_dir,
                expect_ligands=False,
                allow_errors=allow_errors,
                skip_dataset_load=skip_dataset_load,
            )
        except click.exceptions.Exit as exit_:  # the report still shows what failed
            validate_exit = exit_

    if skip_report:
        click.echo("\n=== Stage 10/10: report -- skipped (--skip-report) ===")
    else:
        click.echo("\n=== Stage 10/10: report ===")
        inputs = ValidityInputs(output_dir, cif_dir, ccd_path, npz_dir, cache_output)
        try:
            _write_report(inputs, output_dir / REPORT_SUBDIR)
        except Exception as error:  # noqa: BLE001 — the report never changes the pipeline's outcome
            _warn_report_failed(inputs, structures_npz_dir, error)
    if validate_exit is not None:
        raise validate_exit

    click.echo("\nDone.")


def _warn_report_failed(inputs: ValidityInputs, structures_npz_dir: Path | None, error: Exception) -> None:
    """Log the report stage's traceback and say how to rebuild the report.

    Args:
        inputs: The pipeline output the report was about.
        structures_npz_dir: The pipeline's --structures-npz-dir, if given.
        error: What the report stage raised.
    """
    logger.warning("report stage failed", exc_info=error)
    message = error.message if isinstance(error, click.ClickException) else f"{type(error).__name__}: {error}"
    npz_option = f" --structures-npz-dir {structures_npz_dir}" if structures_npz_dir is not None else ""
    if inputs.cache_path is not None:
        npz_option += f" --cache-path {inputs.cache_path}"
    click.echo(
        f"warning: the report was not written ({message}). The dataset itself is unaffected; "
        f"rebuild the report with: finetune-report --output-dir {inputs.out_dir} --cif-dir {inputs.cif_dir} "
        f"--ccd-path {inputs.ccd_path}{npz_option}"
    )


def _write_report(inputs: ValidityInputs, report_dir: Path, ids: list[str] | None = None) -> None:
    """Write the report and print where it is, with a hint for large datasets.

    Args:
        inputs: The pipeline output to report on.
        report_dir: Where to write the report.
        ids: Only report these structures; all when None.
    """
    try:
        check_report_dir(inputs, report_dir)  # before the hint, which reads the manifest
        if len(ids if ids is not None else inputs.entries()) > LARGE_DATASET:
            click.echo(
                f"report shows the first {TABLE_ROW_CAP} per table; for a subset: finetune-report --ids <file> ..."
            )
        write_user_report(inputs, report_dir, ids)
    except ReportError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Report: {report_dir / 'index.html'}")


@click.command()
@click.option("--output-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Pipeline output directory (manifest.json, dataset_cache.json, cleanup_steps/, ...).")
@click.option("--cif-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Directory of the original input mmCIF files.")
@click.option("--ccd-path", required=True, type=click.Path(path_type=Path, exists=True, dir_okay=False),
              help="Chemical Component Dictionary mmCIF.")
@click.option("--structures-npz-dir", default=None, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="NPZ structure files root. Defaults to <output-dir>/structure_files.")
@click.option("--ids", "ids_file", default=None, type=click.Path(path_type=Path, exists=True, dir_okay=False),
              help="File with one structure id per line: report only these.")
@click.option("--report-dir", default=None, type=click.Path(path_type=Path, file_okay=False),
              help="Where to write the report (replaced if it holds an old one). Defaults to <output-dir>/report.")
@click.option("--cache-path", default=None, type=click.Path(path_type=Path, exists=True, dir_okay=False),
              help="The dataset cache JSON (finetune-pipeline's --cache-output). Defaults to <output-dir>/dataset_cache.json.")
@_with_pipeline_logs
def finetune_report(
    output_dir: Path,
    cif_dir: Path,
    ccd_path: Path,
    structures_npz_dir: Path | None,
    ids_file: Path | None,
    report_dir: Path | None,
    cache_path: Path | None,
) -> None:
    """Rebuild the validation report of an existing pipeline output; prints the index.html path."""
    ids = None
    if ids_file is not None:
        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    inputs = ValidityInputs(output_dir, cif_dir, ccd_path, structures_npz_dir, cache_path)
    _write_report(inputs, report_dir if report_dir is not None else output_dir / REPORT_SUBDIR, ids)



@click.command()
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Pipeline output directory produced by finetune-prep (manifest.json + "
    "structures/); only this dataset's sequences are processed.",
)
@click.option(
    "--msa-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="MSA directory written by generate-msas (with templates on).",
)
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
@_with_pipeline_logs
def fetch_templates(
    output_dir: Path, msa_dir: Path, max_hits: int, pdb_mirror_path: Path | None
) -> None:
    """Resolve template structures and lay out ColabFold template hits for OF3.

    Writes <msa-dir>/template_structures/ and <msa-dir>/of3_template_alignments/; safe to rerun.
    """
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
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Pipeline output directory (manifest.json, structures/, and dataset_cache.json "
    "from materialize-dataset-cache).",
)
@click.option(
    "--msa-dir",
    required=True,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="MSA directory after fetch-templates (template_structures/ and "
    "of3_template_alignments/).",
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
@_with_pipeline_logs
def preprocess_templates(
    output_dir: Path,
    msa_dir: Path,
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
    template preprocessing into <msa-dir>/of3_template_data/.
    """
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
