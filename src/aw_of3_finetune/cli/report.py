"""The finetune-report command and the report stage of finetune-pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from aw_of3_finetune.cli.options import (
    ccd_path_option,
    output_dir_option,
    with_pipeline_logs,
)
from aw_of3_finetune.report.build import REPORT_SUBDIR
from aw_of3_finetune.report.summary import LARGE_DATASET, TABLE_ROW_CAP
from aw_of3_finetune.report.user import ReportError, check_report_dir, write_user_report
from aw_of3_finetune.report.validity import ValidityInputs

logger = logging.getLogger(__name__)


def warn_report_failed(inputs: ValidityInputs, error: Exception) -> None:
    """Log the report stage's traceback and say how to rebuild the report.

    Args:
        inputs: The pipeline output the report was about.
        error: What the report stage raised.
    """
    logger.warning("report stage failed", exc_info=error)
    message = error.message if isinstance(error, click.ClickException) else f"{type(error).__name__}: {error}"
    cache_option = f" --cache-path {inputs.cache_path}" if inputs.cache_path is not None else ""
    click.echo(
        f"warning: the report was not written ({message}). The dataset itself is unaffected; "
        f"rebuild the report with: finetune-report --output-dir {inputs.out_dir} --cif-dir {inputs.cif_dir} "
        f"--ccd-path {inputs.ccd_path}{cache_option}"
    )


def write_report(inputs: ValidityInputs, report_dir: Path, ids: list[str] | None = None) -> None:
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
@output_dir_option(exists=True,
              help="Pipeline output directory (manifest.json, dataset_cache.json, cleanup_steps/, ...).")
@click.option("--cif-dir", required=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Directory of the original input mmCIF files.")
@ccd_path_option(required=True, help="Chemical Component Dictionary mmCIF.")
@click.option("--ids", "ids_file", default=None, type=click.Path(path_type=Path, exists=True, dir_okay=False),
              help="File with one structure id per line: report only these.")
@click.option("--report-dir", default=None, type=click.Path(path_type=Path, file_okay=False),
              help="Where to write the report (replaced if it holds an old one). Defaults to <output-dir>/report.")
@click.option("--cache-path", default=None, type=click.Path(path_type=Path, exists=True, dir_okay=False),
              help="The dataset cache JSON (finetune-pipeline's --cache-output). Defaults to <output-dir>/dataset_cache.json.")
@with_pipeline_logs
def finetune_report(
    output_dir: Path,
    cif_dir: Path,
    ccd_path: Path,
    ids_file: Path | None,
    report_dir: Path | None,
    cache_path: Path | None,
) -> None:
    """Rebuild the validation report of an existing pipeline output; prints the index.html path."""
    ids = None
    if ids_file is not None:
        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    inputs = ValidityInputs(output_dir, cif_dir, ccd_path, cache_path)
    write_report(inputs, report_dir if report_dir is not None else output_dir / REPORT_SUBDIR, ids)
