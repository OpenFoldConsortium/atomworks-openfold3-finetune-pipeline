"""Training hand-off commands: generate-train-config and split-dataset."""

from __future__ import annotations

from pathlib import Path

import click
from openfold3.core.data.io.dataset_cache import read_datacache

from aw_of3_finetune.cli.options import (
    organized_msa_dir_option,
    output_dir_option,
    require_organized_msa_dir,
    resolve_organized_msa_dir,
    with_pipeline_logs,
)
from aw_of3_finetune.split import (
    read_split_csv,
    split_dataset_cache,
)
from aw_of3_finetune.training_config import (
    LEGACY_CACHE_WARNING,
    is_legacy_dataset_cache,
    write_train_config,
)


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory (must contain dataset_cache.json, "
    "of3_compressed_structures/, reference_mols/ -- i.e. materialize-dataset-cache and "
    "save_structures_as_npz must already have run).",
)
@organized_msa_dir_option(
    exists=True,
    help="MSA folder written by generate-msas and reorganized by "
    "materialize-dataset-cache. It must exist: alignments_directory in the generated "
    "config points at <organized-msa-dir>/of3_alignments, and OF3's own TrainingDatasetPaths "
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
@click.option(
    "--init-checkpoint",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Start fine-tuning from these weights (a .pt or .ckpt file) instead of OpenFold3's "
    "default pretrained weights. Only the model and EMA weights are loaded.",
)
@click.option(
    "--from-scratch",
    is_flag=True,
    default=False,
    help="Start from random weights instead of pretrained ones (restart_checkpoint_path: last).",
)
@with_pipeline_logs
def generate_train_config(
    output_dir: Path,
    msa_dir: Path | None,
    config_output: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_name: str | None,
    init_checkpoint: Path | None = None,
    from_scratch: bool = False,
) -> None:
    """Write an OF3 training runner-yaml pointed at this pipeline's output.

    Smoke-test defaults (see training_config.py); adds a validation section when a
    current split/ exists (split-dataset). Hand-edit for real runs. Training starts
    from OpenFold3's pretrained weights (pixi run setup_openfold downloads them) unless
    --init-checkpoint or --from-scratch is given.
    Run from this repository with:
    pixi run run_openfold train --runner-yaml=<config-output>
    """
    if init_checkpoint is not None and from_scratch:
        raise click.UsageError("--init-checkpoint and --from-scratch cannot be used together.")
    msa_dir = require_organized_msa_dir(msa_dir, output_dir)
    config_path = write_train_config(
        output_dir, msa_dir, config_output,
        wandb_project=wandb_project, wandb_entity=wandb_entity, wandb_experiment_name=wandb_name,
        init_checkpoint=init_checkpoint, from_scratch=from_scratch,
    )
    click.echo(f"Training config: {config_path}")
    cache_path = output_dir / "dataset_cache.json"
    if cache_path.exists() and is_legacy_dataset_cache(cache_path):
        click.echo(LEGACY_CACHE_WARNING, err=True)


@click.command()
@output_dir_option(
    exists=True,
    help="Pipeline output directory containing dataset_cache.json. The split is written to <output-dir>/split/.",
)
@click.option(
    "--split-csv",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="CSV with header id,split. split is val or test; unlisted entries go to train.",
)
@organized_msa_dir_option(
    exists=True,
    help="MSA folder. If it exists, (re)writes <output-dir>/train.yml with default settings and the "
    "validation section. For wandb or custom options, run generate-train-config instead.",
)
@click.pass_context
@with_pipeline_logs
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
      split-dataset --output-dir out/ --split-csv my_split.csv --organized-msa-dir msas/
      split-dataset --output-dir out/ --split-csv my_split.csv
      generate-train-config --output-dir out/ --organized-msa-dir msas/
    """
    result = split_dataset_cache(output_dir, split_csv)
    for warning in result.warnings:
        click.echo(warning, err=True)
    click.echo(result.report)
    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=True)
    if msa_dir is None:
        click.echo(
            f"Next: generate-train-config --output-dir {output_dir} --organized-msa-dir <dir> "
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


def run_split_stage(output_dir: Path, cif_dir: Path, split_csv: Path) -> None:
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
