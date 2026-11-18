"""CLI commands for the atomworks fine-tuning dataset pipeline.

Each command lives in a submodule; this package re-exports them for the console scripts."""

from aw_of3_finetune.cli.msas import generate_msas, import_msas
from aw_of3_finetune.cli.options import ccd_path_option, cleanup_config_option
from aw_of3_finetune.cli.pipeline import finetune_pipeline
from aw_of3_finetune.cli.prep import (
    finetune_prep,
    materialize_dataset_cache,
    prepare_inputs,
    setup_finetuning,
)
from aw_of3_finetune.cli.report import finetune_report
from aw_of3_finetune.cli.templates import fetch_templates, preprocess_templates
from aw_of3_finetune.cli.training import generate_train_config, split_dataset
from aw_of3_finetune.cli.validation import validate

__all__ = [
    "ccd_path_option",
    "cleanup_config_option",
    "prepare_inputs",
    "finetune_prep",
    "materialize_dataset_cache",
    "generate_msas",
    "import_msas",
    "fetch_templates",
    "preprocess_templates",
    "generate_train_config",
    "split_dataset",
    "finetune_pipeline",
    "validate",
    "finetune_report",
    "setup_finetuning",
]
