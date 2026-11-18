"""Click options and decorators shared by the pipeline commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from pathlib import Path

import click
import yaml
from pydantic import ValidationError

from aw_of3_finetune.cleanup import CleanupSettings
from aw_of3_finetune.constants import (
    DEFAULT_OUTPUT_DIR,
    ORGANIZED_MSAS_DIRNAME,
)
from aw_of3_finetune.openfold_cache import (
    SETUP_HINT,
    resolve_ccd_path,
)


class _ClickEchoHandler(logging.Handler):
    """Log handler that writes through click.echo, so CliRunner and pipes capture it."""

    def emit(self, record: logging.LogRecord) -> None:
        click.echo(self.format(record), err=True)


def with_pipeline_logs(command: Callable) -> Callable:
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


# click Context.meta key: where --ccd-path's value came from (see ccd_path_option).
CCD_SOURCE_META_KEY = "aw_of3_finetune.ccd_source"


def ccd_path_option(*, required: bool, help: str) -> Callable[[Callable], Callable]:
    """Add --ccd-path, falling back to $CCD_PATH and then the path stored by setup-finetuning.

    Args:
        required: If True, the command fails when no CCD is found by any route.
        help: Help text describing what the command uses the CCD for.

    Returns:
        A click option decorator.
    """

    def callback(ctx: click.Context, param: click.Parameter, value: Path | None) -> Path | None:
        path, source = resolve_ccd_path(value)
        if path is None:
            if required:
                raise click.BadParameter(f"no CCD found. {SETUP_HINT}", ctx=ctx, param=param)
            return None
        if not path.is_file():
            raise click.BadParameter(f"CCD file {path} (from {source}) does not exist. {SETUP_HINT}",
                                     ctx=ctx, param=param)
        if value is None:
            click.echo(f"Using the CCD at {path} (from {source}).", err=True)
        # Shared with commands run through ctx.invoke, so finetune-prep can record it.
        ctx.meta[CCD_SOURCE_META_KEY] = source
        return path

    return click.option(
        "--ccd-path",
        default=None,
        type=click.Path(path_type=Path, dir_okay=False),
        callback=callback,
        help=f"{help} Default: $CCD_PATH if set, else the path stored by setup-finetuning.",
    )


def output_dir_option(*, exists: bool, help: str) -> Callable[[Callable], Callable]:
    """Add --output-dir, the pipeline output directory, defaulting to ./output.

    Args:
        exists: Whether the directory must already exist (stages after finetune-prep).
        help: Help text.

    Returns:
        A click option decorator.
    """
    return click.option(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        show_default=True,
        type=click.Path(path_type=Path, exists=exists, file_okay=False),
        help=help,
    )


def organized_msa_dir_option(*, exists: bool, help: str) -> Callable[[Callable], Callable]:
    """Add --organized-msa-dir (parameter ``msa_dir``) plus the hidden, deprecated --msa-dir alias.

    The option's default is None; commands resolve it with resolve_organized_msa_dir.

    Args:
        exists: Whether a path passed on the command line must already exist.
        help: Help text for --organized-msa-dir.

    Returns:
        A decorator for a click command callback.
    """
    path_type = click.Path(path_type=Path, exists=exists, file_okay=False)

    def decorator(command: Callable) -> Callable:
        @wraps(command)
        def wrapper(*args, legacy_msa_dir: Path | None = None, **kwargs):
            if legacy_msa_dir is not None:
                if kwargs.get("msa_dir") is not None:
                    raise click.UsageError(
                        "Pass --organized-msa-dir only; --msa-dir is its deprecated old name."
                    )
                click.echo("--msa-dir is deprecated; use --organized-msa-dir", err=True)
                kwargs["msa_dir"] = legacy_msa_dir
            return command(*args, **kwargs)

        wrapper = click.option("--msa-dir", "legacy_msa_dir", default=None, hidden=True, type=path_type)(wrapper)
        return click.option(
            "--organized-msa-dir", "msa_dir", default=None, type=path_type,
            help=f"{help} Default: <output-dir>/{ORGANIZED_MSAS_DIRNAME}.",
        )(wrapper)

    return decorator


def resolve_organized_msa_dir(msa_dir: Path | None, output_dir: Path, *, must_exist: bool) -> Path | None:
    """Return the MSA folder: the one passed, else <output-dir>/organized_msas.

    Args:
        msa_dir: --organized-msa-dir as given, or None.
        output_dir: Pipeline output directory.
        must_exist: If True, the default is returned only when it exists (None otherwise).

    Returns:
        The folder, or None when it was not given and the default does not exist (must_exist only).
    """
    if msa_dir is not None:
        return msa_dir
    default = output_dir / ORGANIZED_MSAS_DIRNAME
    if must_exist and not default.is_dir():
        return None
    return default


def require_organized_msa_dir(msa_dir: Path | None, output_dir: Path) -> Path:
    """Resolve the MSA folder for a command that cannot run without one.

    Args:
        msa_dir: --organized-msa-dir as given, or None.
        output_dir: Pipeline output directory.

    Returns:
        The existing folder.

    Raises:
        click.UsageError: If the folder does not exist.
    """
    resolved = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=False)
    if not resolved.is_dir():
        raise click.UsageError(
            f"No MSA folder at {resolved}. Pass --organized-msa-dir, or create MSAs there first "
            "with generate-msas."
        )
    return resolved
