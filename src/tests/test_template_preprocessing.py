"""Template stages end to end on 1fdl: raw ColabFold hits -> OF3 template cache -> training.

Offline: hits come from src/tests/fixtures/templates_1fdl/raw_m8 (author chain IDs, as
ColabFold writes them) and structures from its local PDB mirror.

Test structure:
- test_preprocess_templates_writes_template_ids: fetch-templates + preprocess-templates
  fill dataset_cache.json, filtered by 1fdl's release date and OF3's sequence filters.
- test_no_sequence_filter_keeps_identical_template: --no-sequence-filter keeps 5lyz.
- test_changed_sequence_filter_rebuilds_template_cache: rerunning with other filter
  settings does not reuse the previous run's template cache.
- test_weighted_pdb_dataset_loads_templates (integration): the generated train config
  feeds non-empty template features to WeightedPDBDataset.
"""

import gzip
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    DATASET_CONFIG_REGISTRY,
    TrainingDatasetPaths,
)

from aw_of3_finetune.cli import (
    fetch_templates,
    finetune_prep,
    generate_train_config,
    materialize_dataset_cache,
    finetune_pipeline,
    preprocess_templates,
)
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences
from aw_of3_finetune.pipeline import save_structures_as_npz
from aw_of3_finetune.constants import TEMPLATE_CUTOFF_FILENAME

WeightedPDBConfig = DATASET_CONFIG_REGISTRY.get("WeightedPDBDataset")

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "templates_1fdl"


def _run(command, args: list[str]) -> None:
    """Invoke a CLI command and assert it exits 0."""
    result = CliRunner().invoke(command, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output


def _prepare_1fdl(tmp_path: Path, ccd_path: Path | None = None) -> tuple[Path, Path]:
    """Run finetune-prep on 1fdl and stage its MSAs and raw ColabFold template hits.

    Args:
        tmp_path: Scratch directory.
        ccd_path: CCD for reference molecules; needed only to train on the output.

    Returns:
        ``(output_dir, msa_dir)``.
    """
    cif_dir = tmp_path / "cif_input"
    cif_dir.mkdir()
    query = FIXTURE_DIR / "pdb_mirror" / "fd" / "1fdl.cif.gz"
    (cif_dir / "1fdl.cif").write_bytes(gzip.decompress(query.read_bytes()))

    output_dir = tmp_path / "out"
    _run(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir)]
        + (["--ccd-path", str(ccd_path)] if ccd_path else []),
    )

    # Query-only unpaired MSAs, and the raw hits where generate-msas puts them.
    msa_dir = tmp_path / "msas"
    for seq in chain_canonical_sequences(cif_dir / "1fdl.cif").values():
        rep_id = alignment_rep_id(seq)
        a3m = msa_dir / "unpaired" / rep_id[:2] / f"{rep_id}.a3m.gz"
        a3m.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(a3m, "wt") as f:
            f.write(f">query\n{seq}\n")
    for m8 in (FIXTURE_DIR / "raw_m8").glob("*.m8"):
        dest = msa_dir / "templates" / m8.stem[:2] / m8.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m8, dest)
    return output_dir, msa_dir


def _run_template_stages(
    output_dir: Path, msa_dir: Path, preprocess_args: list[str] | None = None
) -> None:
    """Run fetch-templates, materialize-dataset-cache and preprocess-templates in order."""
    _run(
        fetch_templates,
        [
            "--output-dir", str(output_dir),
            "--organized-msa-dir", str(msa_dir),
            "--pdb-mirror-path", str(FIXTURE_DIR / "pdb_mirror"),
        ],
    )
    _run(materialize_dataset_cache, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)])
    _run(
        preprocess_templates,
        ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)] + (preprocess_args or []),
    )


def _template_ids_by_chain(output_dir: Path) -> dict[str, list[str]]:
    """Read 1fdl's template_ids from dataset_cache.json, keyed by label_asym_id."""
    cache = json.loads((output_dir / "dataset_cache.json").read_text())
    chains = cache["structure_data"]["1fdl"]["chains"].values()
    return {chain["label_asym_id"]: chain["template_ids"] or [] for chain in chains}


def test_preprocess_templates_writes_template_ids(tmp_path):
    """Each 1fdl chain gets the older templates that are not copies of its own sequence."""
    output_dir, msa_dir = _prepare_1fdl(tmp_path)

    _run_template_stages(output_dir, msa_dir)

    # 1fdl's dataset-cache date is its first release, 1991-10-15. With the default 60-day
    # minimum, 1fdl itself and every later hit are excluded. 5lyz (lysozyme, chain C) is
    # older but identical to chain C, so OF3's subsequence filter drops it.
    expected_template_ids = {"A": [], "B": ["3hfm_B"], "C": []}
    actual_template_ids = _template_ids_by_chain(output_dir)
    assert actual_template_ids == expected_template_ids

    # The dropped hits are listed per alignment representative: 1fdl's own light and
    # heavy chains (the lysozyme chain's table has no 1fdl row), and 5lyz.
    report = msa_dir / "of3_template_data" / "template_sequence_filter.tsv"
    actual_dropped = {row["template_id"] for row in _read_tsv(report)}
    expected_dropped = {"1fdl_A", "1fdl_B", "5lyz_A"}
    assert actual_dropped == expected_dropped


def test_no_sequence_filter_keeps_identical_template(tmp_path):
    """With --no-sequence-filter, the identical lysozyme template 5lyz is kept."""
    output_dir, msa_dir = _prepare_1fdl(tmp_path)

    _run_template_stages(output_dir, msa_dir, ["--no-sequence-filter"])

    expected_template_ids = {"A": [], "B": ["3hfm_B"], "C": ["5lyz_A"]}
    actual_template_ids = _template_ids_by_chain(output_dir)
    assert actual_template_ids == expected_template_ids


def test_changed_sequence_filter_rebuilds_template_cache(tmp_path):
    """A rerun with different filter settings rebuilds the cache instead of reusing it."""
    output_dir, msa_dir = _prepare_1fdl(tmp_path)
    _run_template_stages(output_dir, msa_dir, ["--no-sequence-filter"])

    _run(preprocess_templates, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)])

    expected_chain_c_templates = []
    actual_chain_c_templates = _template_ids_by_chain(output_dir)["C"]
    assert actual_chain_c_templates == expected_chain_c_templates


def test_preprocess_templates_reports_date_filtered_chains(tmp_path):
    """Chains whose template hits are all dropped by the release-date filter are logged."""
    output_dir, msa_dir = _prepare_1fdl(tmp_path)
    _run(
        fetch_templates,
        [
            "--output-dir", str(output_dir),
            "--organized-msa-dir", str(msa_dir),
            "--pdb-mirror-path", str(FIXTURE_DIR / "pdb_mirror"),
        ],
    )
    _run(materialize_dataset_cache, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)])
    result = CliRunner().invoke(
        preprocess_templates,
        ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    report = msa_dir / "of3_template_data" / "template_date_filter.tsv"
    rows = {row["label_asym_id"]: row for row in _read_tsv(report)}

    # The report has one row per 1fdl chain.
    assert set(rows) == {"A", "B", "C"}

    # Chain A had hits, but every one was dropped; its release date is 1fdl's own.
    assert rows["A"]["release_date"] == "1991-10-15"
    assert int(rows["A"]["n_hits"]) > 0 and int(rows["A"]["n_kept"]) == 0

    # Chain B kept exactly one hit.
    assert int(rows["B"]["n_kept"]) == 1

    # The command output names the chains that lost all their hits. Chain C's remaining
    # hits (1ior, 7ynv) are both newer than 1fdl once the sequence filters drop 5lyz.
    expected_message = (
        "The following 2 chain(s) lost every template hit to the release-date filter: "
        "1fdl/A, 1fdl/C"
    )
    assert expected_message in result.output


def _read_tsv(path: Path) -> list[dict[str, str]]:
    """Read a tab-separated file with a header row into a list of row dicts."""
    header, *lines = path.read_text().splitlines()
    return [dict(zip(header.split("\t"), line.split("\t"))) for line in lines]


@pytest.mark.integration
def test_weighted_pdb_dataset_loads_templates(ccd_path, tmp_path):
    """The generated train config gives WeightedPDBDataset non-empty template features."""
    output_dir, msa_dir = _prepare_1fdl(tmp_path, ccd_path)
    _run_template_stages(output_dir, msa_dir)
    save_structures_as_npz(output_dir / "cleaned_structures", output_dir / "of3_compressed_structures")
    _run(generate_train_config, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)])

    paths = yaml.safe_load((output_dir / "train.yml").read_text())["dataset_paths"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = WeightedPDBConfig(
            name="templates-1fdl",
            dataset_paths=TrainingDatasetPaths(**paths["weighted-pdb"]),
            template={"n_templates": 4, "take_top_k": True},
        )
        dataset = WeightedPDBDataset(config)
        masks = [
            np.asarray(dataset[i]["template_pseudo_beta_mask"])
            for i in range(len(dataset.datapoint_cache))
        ]

    # At least one datapoint has a template residue in its pseudo-beta mask.
    assert any(mask.any() for mask in masks), "no datapoint had a template residue"


def _template_ids_with_args(tmp_path: Path, preprocess_args: list[str]) -> dict[str, list[str]]:
    """1fdl's template_ids by chain after the template stages with these extra arguments."""
    output_dir, msa_dir = _prepare_1fdl(tmp_path)
    _run_template_stages(output_dir, msa_dir, preprocess_args)
    return _template_ids_by_chain(output_dir)


def test_cutoff_date_keeps_hits_newer_than_the_structure(tmp_path):
    actual_ids = _template_ids_with_args(tmp_path, ["--template-cutoff-date", "2100-01-01"])

    # Without a cutoff chain A loses every hit, all being newer than 1fdl.
    assert actual_ids["A"] != []


def test_cutoff_date_drops_hits_released_after_it(tmp_path):
    actual_ids = _template_ids_with_args(tmp_path, ["--template-cutoff-date", "1900-01-01"])

    expected_ids = {"A": [], "B": [], "C": []}
    assert actual_ids == expected_ids


def test_cutoff_date_never_gives_a_structure_itself(tmp_path):
    actual_ids = _template_ids_with_args(tmp_path, ["--template-cutoff-date", "2100-01-01"])

    actual_self_hits = [
        template_id
        for template_ids in actual_ids.values()
        for template_id in template_ids
        if template_id.startswith("1fdl")
    ]
    assert actual_self_hits == []


def test_cutoff_date_is_recorded_and_cleared(tmp_path):
    output_dir, msa_dir = _prepare_1fdl(tmp_path)
    _run_template_stages(output_dir, msa_dir, ["--template-cutoff-date", "2100-01-01"])
    record = msa_dir / "of3_template_data" / TEMPLATE_CUTOFF_FILENAME

    actual_record = json.loads(record.read_text())
    expected_record = {"template_cutoff_date": "2100-01-01"}
    assert actual_record == expected_record

    _run(preprocess_templates, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)])
    assert not record.exists()


@pytest.mark.parametrize(
    "extra_args", [["--no-sequence-filter"], ["--max-release-date", "2000-01-01"]]
)
def test_cutoff_date_conflicting_options_are_usage_errors(tmp_path, extra_args):
    output_dir, msa_dir = _prepare_1fdl(tmp_path)
    args = ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)]

    result = CliRunner().invoke(
        preprocess_templates, args + ["--template-cutoff-date", "2100-01-01"] + extra_args
    )

    assert result.exit_code == 2
    assert "--template-cutoff-date" in result.output


def test_finetune_pipeline_takes_the_cutoff_date():
    actual_options = {opt for param in finetune_pipeline.params for opt in param.opts}

    assert "--template-cutoff-date" in actual_options
