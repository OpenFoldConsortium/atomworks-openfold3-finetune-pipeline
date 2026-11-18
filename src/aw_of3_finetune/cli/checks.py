"""Checks that refuse a bad combination of inputs before a command does any work."""

from __future__ import annotations

from pathlib import Path

import click
from atomworks.io.utils.io_utils import apply_sharding_pattern
from atomworks.ml.utils.misc import get_complex_id

from aw_of3_finetune.msa import (
    alignment_rep_id,
    chain_canonical_sequences,
)
from aw_of3_finetune.msa_sources import (
    MsaSourcesError,
    msa_dir_kind,
)


def check_paired_dirs_writable(
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
        f"{listing}\nRerun with --skip-existing to reuse them, or use a fresh --organized-msa-dir."
    )


def warn_missing_msas(cif_path: Path, pdb_id: str, msa_dir: Path) -> None:
    """Warn on stderr for any polymer chain without an MSA directory.

    Args:
        cif_path: Source mmCIF to read SEQRES sequences from.
        pdb_id: Structure ID, for the message.
        msa_dir: OF3 alignments directory laid out as ``{msa_dir}/{rep_id}/*.a3m``.
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


# finetune-pipeline options that only affect ColabFold: parameter name -> the flag that turns it on.
COLABFOLD_ONLY_OPTIONS = {
    "msa_search": "--msa-search",
    "templates": "--templates",
    "server": "--server",
    "user_agent": "--user-agent",
    "skip_existing": "--skip-existing",
}


def refuse_colabfold_flags_with_msa_sources(ctx: click.Context) -> None:
    """Refuse ColabFold-only options given next to --msa-sources, where they would do nothing.

    Turning one off (--no-msa-search, --no-templates, --no-skip-existing) agrees with
    --msa-sources and is accepted.

    Args:
        ctx: finetune-pipeline's context.

    Raises:
        click.UsageError: Naming the first such option given on the command line.
    """
    for name, flag in COLABFOLD_ONLY_OPTIONS.items():
        given = ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
        if given and ctx.params[name] is not False:
            reason = ("templates with --msa-sources are not supported yet" if name == "templates"
                      else "it only applies to ColabFold, which --msa-sources replaces")
            raise click.UsageError(f"{flag} cannot be used with --msa-sources: {reason}.")


def refuse_user_msa_folder_for_colabfold(msa_dir: Path) -> None:
    """Fail before stage 1 if generate-msas would refuse the MSA folder anyway.

    Args:
        msa_dir: The resolved MSA folder.

    Raises:
        click.ClickException: If the folder holds MSAs imported with import-msas.
    """
    try:
        kind = msa_dir_kind(msa_dir)
    except MsaSourcesError as error:
        raise click.ClickException(str(error)) from error
    if kind == "user":
        raise click.ClickException(
            f"{msa_dir} holds user-provided MSAs (import-msas), and one MSA folder holds ColabFold or "
            "user MSAs, not both. Pass --no-msa-search to use them as they are, --msa-sources to "
            "import more, or a fresh --organized-msa-dir to query ColabFold."
        )
