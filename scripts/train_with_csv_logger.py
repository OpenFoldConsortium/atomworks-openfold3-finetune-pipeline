"""Run OF3 training with a CSVLogger attached -- a drop-in for `run_openfold train`.

OF3's LoggingConfig has no CSVLogger option (only wandb), so this subclasses
TrainingExperimentRunner to add one; otherwise it mirrors run_openfold's `train`.
Metrics land at {experiment_settings.output_dir}/csv_logs/version_*/metrics.csv.
Needs openfold-3's CUDA pixi env:

  pixi run -e openfold3-cuda12 --manifest-path <path-to-openfold-3>/pixi.toml \
      python scripts/train_with_csv_logger.py --runner-yaml=<path-to-train.yml>
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import click


@click.command()
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters -- same argument "
    "run_openfold train takes.",
)
@click.option("--seed", type=int, default=None, help="Initial seed for all processes.")
@click.option(
    "--data-seed",
    "--data_seed",
    type=int,
    default=None,
    help="Initial seed for the data pipeline. Defaults to --seed if not specified.",
)
def main(runner_yaml: Path, seed: int | None, data_seed: int | None) -> None:
    from openfold3.core.config import config_utils
    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
    from openfold3.entry_points.validator import TrainingExperimentConfig
    from pytorch_lightning.loggers import CSVLogger

    class CSVLoggingTrainingExperimentRunner(TrainingExperimentRunner):
        """TrainingExperimentRunner with a CSVLogger appended to its loggers."""

        @cached_property
        def loggers(self):
            return [*super().loggers, CSVLogger(save_dir=str(self.output_dir), name="csv_logs")]

    runner_dict = config_utils.load_yaml(runner_yaml)
    if seed is not None:
        runner_dict["experiment_settings"]["seed"] = seed
    if data_seed is not None:
        runner_dict["data_module_args"]["data_seed"] = data_seed

    expt_config = TrainingExperimentConfig.model_validate(runner_dict)
    expt_runner = CSVLoggingTrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()


if __name__ == "__main__":
    main()
