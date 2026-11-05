"""Helpers that build a small pipeline output for validate tests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner

from aw_of3_finetune.cli import finetune_prep, materialize_dataset_cache
from aw_of3_finetune.msa import collect_msa_queries
from aw_of3_finetune.pipeline import save_structures_as_npz
from aw_of3_finetune.validate.context import ValidationContext


@dataclass
class BuiltDataset:
    """Locations of a built dataset: pipeline output, input CIFs and MSA stubs."""

    output_dir: Path
    cif_dir: Path
    msa_dir: Path


def build_dataset(
    root: Path, cif_paths: list[Path], ccd_path: Path, cleanup_config: str | None = None
) -> BuiltDataset:
    """Run prep, cache materialization, NPZ conversion and query-only MSA stubs.

    Args:
        root: Empty directory to build under.
        cif_paths: Input CIFs.
        ccd_path: Chemical component dictionary file.
        cleanup_config: YAML text for finetune-prep's --cleanup-config; None = defaults.

    Returns:
        Paths of the built dataset.
    """
    cif_dir = root / "cif_input"
    cif_dir.mkdir(parents=True)
    for path in cif_paths:
        shutil.copy(path, cif_dir / path.name)
    output_dir = root / "output"
    cleanup_args = []
    if cleanup_config is not None:
        (root / "cleanup.yml").write_text(cleanup_config)
        cleanup_args = ["--cleanup-config", str(root / "cleanup.yml")]
    runner = CliRunner()

    # Run finetune-prep, then materialize the dataset cache.
    result = runner.invoke(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir),
         "--name", "validate", "--ccd-path", str(ccd_path), *cleanup_args],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        materialize_dataset_cache,
        ["--output-dir", str(output_dir),
         "--cache-output", str(output_dir / "dataset_cache.json")],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # Convert the cleaned structures to NPZ.
    save_structures_as_npz(output_dir / "structures", output_dir / "structure_files")
    msa_dir = root / "msa"
    # One stub MSA per representative sequence, containing only the query row.
    for rep_id, sequence in collect_msa_queries(output_dir).items():
        rep_dir = msa_dir / "of3_alignments" / rep_id
        rep_dir.mkdir(parents=True)
        (rep_dir / "colabfold_main.a3m").write_text(f">query\n{sequence}\n")
    return BuiltDataset(output_dir=output_dir, cif_dir=cif_dir, msa_dir=msa_dir)


def mutable_copy(built: BuiltDataset, dest: Path) -> BuiltDataset:
    """Copy a built dataset so a test can corrupt it.

    Args:
        built: Dataset to copy.
        dest: Destination root (must not exist).

    Returns:
        Paths of the copy.
    """
    shutil.copytree(built.output_dir.parent, dest)
    return BuiltDataset(
        output_dir=dest / built.output_dir.name,
        cif_dir=dest / built.cif_dir.name,
        msa_dir=dest / built.msa_dir.name,
    )


def context(built: BuiltDataset, **overrides) -> ValidationContext:
    """Build a ValidationContext for a built dataset.

    Args:
        built: Dataset to validate.
        **overrides: Extra ValidationContext fields; msa_dir defaults to the built MSA dir
            (pass msa_dir=None to validate without MSAs).

    Returns:
        The context.
    """
    overrides.setdefault("msa_dir", built.msa_dir)
    return ValidationContext(output_dir=built.output_dir, cif_dir=built.cif_dir, **overrides)
