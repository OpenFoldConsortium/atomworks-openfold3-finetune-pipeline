"""The finetune-pipeline command, which runs every stage in order."""

from __future__ import annotations

from pathlib import Path

import click

from aw_of3_finetune.cleanup import CleanupSettings
from aw_of3_finetune.cli.preflight import (
    refuse_colabfold_flags_with_msa_sources,
    refuse_user_msa_folder_for_colabfold,
)
from aw_of3_finetune.cli.msas import generate_msas, import_msas
from aw_of3_finetune.cli.options import (
    ccd_path_option,
    cleanup_config_option,
    organized_msa_dir_option,
    output_dir_option,
    resolve_organized_msa_dir,
    with_pipeline_logs,
)
from aw_of3_finetune.cli.prep import finetune_prep, materialize_dataset_cache
from aw_of3_finetune.cli.report import warn_report_failed, write_report
from aw_of3_finetune.cli.templates import fetch_templates, preprocess_templates
from aw_of3_finetune.cli.training import generate_train_config, run_split_stage
from aw_of3_finetune.cli.validate_dataset import validate
from aw_of3_finetune.constants import (
    CLEANED_STRUCTURES_DIRNAME,
    COMPRESSED_STRUCTURES_DIRNAME,
)
from aw_of3_finetune.pipeline import (
    save_structures_as_npz,
)
from aw_of3_finetune.report.build import REPORT_SUBDIR
from aw_of3_finetune.report.user import ReportError, check_report_dir
from aw_of3_finetune.report.validity import ValidityInputs
from aw_of3_finetune.split import (
    DATASET_CACHE_NAME,
    SPLIT_DIRNAME,
    read_split_csv,
)
from aw_of3_finetune.training_config import (
    default_init_checkpoint,
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
    help="Output directory for every pipeline artifact: augmented CIFs, manifest, "
    "reference mols, dataset cache, and NPZ structure files.",
)
@organized_msa_dir_option(
    exists=False,
    help="MSA folder. generate-msas (live ColabFold queries) writes into it unless "
    "--no-msa-search or --msa-sources is given.",
)
@click.option(
    "--msa-search/--no-msa-search",
    default=True,
    show_default=True,
    help="Query the ColabFold server for MSAs (generate-msas). With --no-msa-search, ColabFold "
    "is never contacted and the MSA and template stages are skipped; an existing "
    "--organized-msa-dir (default <output-dir>/organized_msas) is still used by the later stages.",
)
@click.option(
    "--msa-sources",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    help="Use existing MSAs instead of querying ColabFold: either a YAML/JSON sources manifest for your "
    "own A3M files, or the --organized-msa-dir of an earlier run, to reuse (link) its MSAs (see "
    "import-msas). They are imported into --organized-msa-dir (default "
    "<output-dir>/organized_msas). ColabFold is never contacted, and the template stages are skipped.",
)
@ccd_path_option(
    required=True,
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
    help="ColabFold MSA server URL. Not used with --no-msa-search or --msa-sources.",
)
@click.option(
    "--user-agent",
    default="aw-of3-finetune",
    show_default=True,
    help="User-agent string sent with API requests. Not used with --no-msa-search or --msa-sources.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Reuse MSAs already in --organized-msa-dir and query ColabFold only for "
    "sequences/complexes without one. Template hits are not re-requested for reused "
    "MSAs: pass --no-skip-existing to query everything again. Not used with "
    "--no-msa-search or --msa-sources.",
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
    "preprocess-templates (downloads hit structures from RCSB). Not used with "
    "--no-msa-search or --msa-sources.",
)
@click.option(
    "--cache-output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to write the full dataset cache JSON. "
    "Defaults to <output-dir>/dataset_cache.json.",
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
              help="Passed through to validate: skip V21, the check that loads every entry through OF3's dataset (slow).")
@click.option("--wandb-project", default=None, help="Passed through to generate-train-config.")
@click.option("--wandb-entity", default=None, help="Passed through to generate-train-config.")
@click.option("--wandb-name", default=None, help="Passed through to generate-train-config.")
@click.option(
    "--init-checkpoint",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Passed through to generate-train-config: start from these weights instead of "
    "OpenFold3's default pretrained weights.",
)
@click.option("--from-scratch", is_flag=True, default=False,
              help="Passed through to generate-train-config: start from random weights.")
@click.option("--write-step-cifs/--no-write-step-cifs", default=False,
              help="Also write the atoms after each cleanup step as a CIF under "
              "<output-dir>/cleanup_steps/<pdb_id>/ (for viewing; default: off).")
@cleanup_config_option
@click.pass_context
@with_pipeline_logs
def finetune_pipeline(
    ctx: click.Context,
    cif_dir: Path,
    output_dir: Path,
    msa_dir: Path | None,
    msa_search: bool,
    msa_sources: Path | None,
    ccd_path: Path,
    template_cache_dir: Path | None,
    name: str,
    server: str,
    user_agent: str,
    skip_existing: bool,
    template_cutoff_date,
    templates: bool,
    cache_output: Path | None,
    skip_inconsistent: bool,
    allow_errors: bool,
    skip_validation: bool,
    skip_report: bool,
    skip_dataset_load: bool,
    split_csv: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_name: str | None,
    init_checkpoint: Path | None,
    from_scratch: bool,
    write_step_cifs: bool,
    cleanup_settings: CleanupSettings,
) -> None:
    """Run every pipeline stage end-to-end over a directory of CIFs.

    finetune-prep -> generate-msas (or import-msas with --msa-sources) -> fetch-templates ->
    materialize-dataset-cache ->
    preprocess-templates -> NPZ -> split-dataset (with --split-csv) -> generate-train-config ->
    validate -> report. MSAs come from ColabFold unless --no-msa-search or --msa-sources is
    given. All but finetune-prep, materialize, NPZ, validate and report need an MSA folder;
    the template stages also need --templates and MSAs generated in this run.
    """
    click.echo(f"Writing pipeline output to {output_dir.resolve()}", err=True)
    if msa_sources is not None:
        refuse_colabfold_flags_with_msa_sources(ctx)
    run_colabfold = msa_search and msa_sources is None
    msa_dir = resolve_organized_msa_dir(msa_dir, output_dir, must_exist=not run_colabfold and msa_sources is None)
    if run_colabfold:
        refuse_user_msa_folder_for_colabfold(msa_dir)
    if init_checkpoint is not None and from_scratch:
        raise click.UsageError("--init-checkpoint and --from-scratch cannot be used together.")
    if msa_dir is not None and init_checkpoint is None and not from_scratch:
        default_init_checkpoint()  # Fail before any stage runs, not at generate-train-config.
    run_templates = run_colabfold and templates
    templates_skipped = (
        "templates with --msa-sources are not supported yet" if msa_sources is not None
        else "--no-msa-search" if not msa_search
        else "--no-templates"
    )
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
            check_report_dir(ValidityInputs(output_dir, cif_dir, ccd_path), output_dir / REPORT_SUBDIR)
        except ReportError as error:
            raise click.ClickException(str(error)) from error
    click.echo("=== Stage 1/10: finetune-prep ===")
    ctx.invoke(
        finetune_prep,
        cif_dir=cif_dir,
        output_dir=output_dir,
        msa_dir=None,  # finetune-prep checks an existing <output-dir>/organized_msas itself.
        template_cache_dir=template_cache_dir,
        ccd_path=ccd_path,
        name=name,
        skip_inconsistent=skip_inconsistent,
        write_step_cifs=write_step_cifs,
        cleanup_settings=cleanup_settings,
    )

    if msa_sources is not None:
        click.echo("\n=== Stage 2/10: import-msas ===")
        ctx.invoke(import_msas, output_dir=output_dir, msa_dir=msa_dir, manifest_path=msa_sources)
    elif run_colabfold:
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
        click.echo("\n=== Stage 2/10: generate-msas -- skipped (--no-msa-search) ===")

    if run_templates:
        click.echo("\n=== Stage 3/10: fetch-templates ===")
        ctx.invoke(fetch_templates, output_dir=output_dir, msa_dir=msa_dir)
    else:
        click.echo(f"\n=== Stage 3/10: fetch-templates -- skipped ({templates_skipped}) ===")

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
            f"\n=== Stage 5/10: preprocess-templates -- skipped ({templates_skipped}) ==="
        )

    click.echo("\n=== Stage 6/10: NPZ structure files ===")
    npz_dir = output_dir / COMPRESSED_STRUCTURES_DIRNAME
    save_structures_as_npz(output_dir / CLEANED_STRUCTURES_DIRNAME, npz_dir)
    n = len(list((output_dir / CLEANED_STRUCTURES_DIRNAME).glob("*.cif")))
    click.echo(f"NPZ structure files: {npz_dir} ({n} structure(s))")

    if split_csv is not None:
        click.echo("\n=== Stage 7/10: split-dataset ===")
        run_split_stage(output_dir, cif_dir, split_csv)
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
            init_checkpoint=init_checkpoint,
            from_scratch=from_scratch,
        )
    else:
        click.echo(
            "\n=== Stage 8/10: generate-train-config -- skipped (no MSA folder; "
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
        inputs = ValidityInputs(output_dir, cif_dir, ccd_path, cache_output)
        try:
            write_report(inputs, output_dir / REPORT_SUBDIR)
        except Exception as error:  # noqa: BLE001 — the report never changes the pipeline's outcome
            warn_report_failed(inputs, error)
    if validate_exit is not None:
        raise validate_exit

    click.echo("\nDone.")
