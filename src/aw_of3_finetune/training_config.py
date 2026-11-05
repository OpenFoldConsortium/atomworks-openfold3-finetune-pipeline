"""Generates an OF3 training runner-yaml (train.yml) from this pipeline's own output.

Smoke-test defaults adapted from openfold-3's datasets/train_pdb_subset.yaml: a starting
point to hand-edit for a real run. With a current ``split/`` (from split-dataset) the
config also gets a ValidationPDBDataset section. msa_dir is required: TrainingDatasetPaths
needs exactly one alignment path. See docs: Known limitations.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import click
import yaml

from aw_of3_finetune.split import SplitPaths, dataset_cache_sha256, split_paths
from aw_of3_finetune.templates import TEMPLATE_DATA_DIRNAME

# ColabFold searches proteins only, so RNA chains have no MSA; OF3 (default: protein and RNA)
# would fail on them. RNA is featurized without an MSA, like DNA. RNA is not properly supported.
MSA_SETTINGS: dict[str, Any] = {"moltypes": ["PROTEIN"]}

_TEMPLATE_CONFIG: dict[str, Any] = {
    "experiment_settings": {
        "mode": "train",
        "seed": 42,
        "restart_checkpoint_path": "last",
    },
    "data_module_args": {
        "batch_size": 1,
        "num_workers": 0,
        "num_workers_validation": 0,
        "epoch_len": 1,
    },
    "logging_config": {
        "log_lr": False,
        "wandb_config": None,
    },
    "checkpoint_config": {
        "save_top_k": 0,
        "save_last": True,
    },
    "pl_trainer_args": {
        "devices": 1,
        "num_nodes": 1,
        "precision": "bf16-mixed",
        "max_epochs": 1,
        "log_every_n_steps": 1,
        "mpi_plugin": False,
        "deepspeed_config_path": None,
    },
    "model_update": {
        "presets": ["train"],
        "custom": {
            "settings": {
                "model_selection_weight_scheme": "fine_tuning",
                "memory": {
                    "train": {
                        "msa_module": {"swiglu_seq_chunk_size": 1024},
                        "use_cueq_triangle_kernels": False,
                        "use_deepspeed_evo_attention": False,
                    },
                },
            },
            "architecture": {
                "shared": {
                    "use_confidence_emb_prob": 0.8,
                    "diffusion": {"use_conditioning_prob": 0.8},
                },
                "loss_module": {"diffusion": {"chunk_size": 2}},
            },
        },
    },
    "dataset_configs": {
        "train": {
            "weighted-pdb": {
                "dataset_class": "WeightedPDBDataset",
                "weight": 1.0,
                "config": {
                    "debug_mode": False,
                    "msa": MSA_SETTINGS,
                    "template": {"n_templates": 4, "take_top_k": False},
                    # Flat: each structure equally likely, given cluster_size = datapoints.
                    "sample_weights": {
                        "a_prot": 1.0,
                        "a_nuc": 1.0,
                        "a_ligand": 1.0,
                        "w_chain": 1.0,
                        "w_interface": 0.5,
                    },
                    "crop": {
                        "token_crop": {
                            "enabled": True,
                            "token_budget": 384,
                            "crop_weights": {
                                "contiguous": 0.2,
                                "spatial": 0.4,
                                "spatial_interface": 0.4,
                            },
                        },
                        "chain_crop": {"enabled": True},
                    },
                },
            },
        },
    },
}

_VALIDATION_CONFIG: dict[str, Any] = {
    "dataset_class": "ValidationPDBDataset",
    "config": {
        "debug_mode": False,
        "msa": {"subsample_main": False, **MSA_SETTINGS},
        "template": {"n_templates": 4, "take_top_k": True},
        "crop": {"token_crop": {"enabled": False}},
    },
}


class StaleSplitError(click.ClickException):
    """`split/` was made from a different dataset_cache.json (or has no source.json)."""


class EmptyTrainSplitError(click.ClickException):
    """Every entry went to val/test, leaving an empty train cache."""


def _current_split_paths(output_dir: Path) -> SplitPaths | None:
    """Return the split paths if a current split exists, else None.

    Args:
        output_dir: The pipeline output directory.

    Returns:
        SplitPaths when ``split/`` exists and matches dataset_cache.json, else None.

    Raises:
        StaleSplitError: If ``split/`` is stale, or source.json/train_cache.json is
            missing or unreadable.
        EmptyTrainSplitError: If the split's train cache has no entries.
    """
    paths = split_paths(output_dir)
    if not paths.dir.is_dir():
        return None
    stale = StaleSplitError(
        f"{paths.dir} was not made from the current dataset_cache.json: rerun "
        f"split-dataset, or delete {paths.dir} to train without validation."
    )
    try:
        recorded = json.loads(paths.source.read_text()).get("dataset_cache_sha256")
        train_cache = json.loads(paths.train_cache.read_text())
    except (OSError, ValueError, AttributeError) as exc:
        raise stale from exc
    if recorded != dataset_cache_sha256(output_dir):
        raise stale
    if not train_cache["structure_data"]:
        raise EmptyTrainSplitError(
            f"{paths.train_cache} has no entries: every structure is val or test. "
            "Leave some structures out of the split CSV so they train."
        )
    return paths


def build_dataset_paths(output_dir: Path, msa_dir: Path) -> dict[str, Any]:
    """Build the weighted-pdb dataset paths for the full dataset cache.

    Args:
        output_dir: Pipeline output directory (dataset_cache.json, structure_files/, ...).
        msa_dir: The --msa-dir; template paths are added if preprocess-templates has run.

    Returns:
        ``{"weighted-pdb": {...}}``, never looking at ``split/``.
    """
    dataset_paths = {
        "weighted-pdb": {
            "alignments_directory": str(msa_dir / "of3_alignments"),
            "alignment_db_directory": "none",
            "alignment_array_directory": "none",
            "dataset_cache_file": str(output_dir / "dataset_cache.json"),
            "target_structures_directory": str(output_dir / "structure_files"),
            "target_structure_file_format": "npz",
            "reference_molecule_directory": str(output_dir / "reference_mols"),
            "template_cache_directory": "none",
            "template_structure_array_directory": "none",
            "template_structures_directory": "none",
            "template_file_format": "npz",
            "ccd_file": None,
        },
    }
    # Template paths only once preprocess-templates has run; "none" turns templates off.
    template_data_dir = msa_dir / TEMPLATE_DATA_DIRNAME
    if (template_data_dir / "template_cache").is_dir():
        dataset_paths["weighted-pdb"] |= {
            "template_cache_directory": str(template_data_dir / "template_cache"),
            "template_structure_array_directory": str(
                template_data_dir / "template_structure_arrays"
            ),
        }
    return dataset_paths


def build_train_config(
    output_dir: Path,
    msa_dir: Path,
    training_output_dir: Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_experiment_name: str | None = None,
) -> dict[str, Any]:
    """Build a runner-yaml config dict pointed at this pipeline's output.

    Args:
        output_dir: finetune-prep/materialize-dataset-cache output directory
            (dataset_cache.json, structure_files/, reference_mols/).
        msa_dir: The --msa-dir passed to generate-msas/materialize-dataset-cache;
            alignments_directory is set to ``{msa_dir}/of3_alignments``. Required
            (see module docstring).
        training_output_dir: Where Lightning writes checkpoints/logs. Defaults to
            ``{output_dir}/training_output``.
        wandb_project: If given, emit logging_config.wandb_config (OF3's only
            config-driven logger); needs a wandb login on the training machine.
        wandb_entity: wandb user/team; omitted if None (wandb's logged-in default).
        wandb_experiment_name: Run name; defaults to output_dir's name.

    Returns:
        The runner-yaml config as a dict.

    Raises:
        StaleSplitError: If ``split/`` is out of date with dataset_cache.json.
        EmptyTrainSplitError: If the split left no training structures.
    """
    config = copy.deepcopy(_TEMPLATE_CONFIG)
    if training_output_dir is None:
        training_output_dir = output_dir / "training_output"
    config["experiment_settings"]["output_dir"] = str(training_output_dir)

    if wandb_project is not None:
        wandb_config: dict[str, str] = {"project": wandb_project}
        if wandb_entity is not None:
            wandb_config["entity"] = wandb_entity
        wandb_config["experiment_name"] = wandb_experiment_name or output_dir.name
        config["logging_config"]["wandb_config"] = wandb_config

    config["dataset_paths"] = build_dataset_paths(output_dir, msa_dir)
    split = _current_split_paths(output_dir)
    if split is not None:
        val_paths = copy.deepcopy(config["dataset_paths"]["weighted-pdb"])
        config["dataset_paths"]["weighted-pdb"]["dataset_cache_file"] = str(split.train_cache)
        val_paths["dataset_cache_file"] = str(split.val_cache)
        config["dataset_paths"]["val-weighted-pdb"] = val_paths
        config["dataset_configs"]["validation"] = {
            "val-weighted-pdb": copy.deepcopy(_VALIDATION_CONFIG)
        }
    return config


LEGACY_CACHE_WARNING = (
    "Warning: this dataset was built before interface keys; rebuild from finetune-prep "
    "for even per-structure sampling."
)


def is_legacy_dataset_cache(cache_path: Path) -> bool:
    """Whether the cache has a structure with 2+ chains, no interfaces and all cluster_size 1.

    Args:
        cache_path: Path to dataset_cache.json.

    Returns:
        True if any structure entry has the pre-interface-keys shape.
    """
    cache = json.loads(cache_path.read_text())
    for entry in cache["structure_data"].values():
        chains = entry["chains"].values()
        if len(chains) >= 2 and not entry["interfaces"] and all(c["cluster_size"] == 1 for c in chains):
            return True
    return False


def write_train_config(
    output_dir: Path,
    msa_dir: Path,
    config_path: Path | None = None,
    **build_kwargs: Any,
) -> Path:
    """Build and write the training runner-yaml.

    Args:
        output_dir: finetune-prep/materialize-dataset-cache output directory.
        msa_dir: The --msa-dir passed to generate-msas/materialize-dataset-cache.
        config_path: Where to write; defaults to ``{output_dir}/train.yml``.
        **build_kwargs: Passed to build_train_config (wandb_* etc.).

    Returns:
        The path written.
    """
    if config_path is None:
        config_path = output_dir / "train.yml"
    config = build_train_config(output_dir, msa_dir, **build_kwargs)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path
