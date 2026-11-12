"""split-dataset command: report on stdout, warnings on stderr, optional train.yml refresh."""

import json

import pytest
import yaml
from click.testing import CliRunner

from aw_of3_finetune.cli import split_dataset
from aw_of3_finetune.split import split_paths
from validate_helpers import mutable_copy

VAL_CSV = "id,split\n7riv,val\n3u8v,val\n"


@pytest.fixture
def built_copy(built_dataset, tmp_path):
    return mutable_copy(built_dataset, tmp_path / "copy")


def invoke(built_copy, tmp_path, csv_text=VAL_CSV, extra=()):
    csv = tmp_path / "my_split.csv"
    csv.write_text(csv_text)
    args = ["--output-dir", str(built_copy.output_dir), "--split-csv", str(csv), *extra]
    return CliRunner().invoke(split_dataset, args, catch_exceptions=False)


def test_prints_the_report_and_writes_split(built_copy, tmp_path):
    result = invoke(built_copy, tmp_path)
    expected_line = "Split: 3 train, 2 val, 0 test (5 entries)"
    assert result.exit_code == 0
    assert expected_line in result.output
    assert split_paths(built_copy.output_dir).val_cache.exists()


def test_without_msa_dir_says_to_run_generate_train_config(built_copy, tmp_path):
    result = invoke(built_copy, tmp_path)
    assert result.exit_code == 0
    assert result.output.count("Next: generate-train-config") == 1


def test_with_msa_dir_writes_train_yml_with_validation(built_copy, tmp_path):
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    result = invoke(built_copy, tmp_path, extra=["--organized-msa-dir", str(msa_dir)])
    config = yaml.safe_load((built_copy.output_dir / "train.yml").read_text())
    assert result.exit_code == 0
    assert "val-weighted-pdb" in config["dataset_configs"]["validation"]


def test_bad_csv_exits_nonzero_listing_rows(built_copy, tmp_path):
    result = invoke(built_copy, tmp_path, csv_text="id,split\nnope,val\n")
    assert result.exit_code != 0
    assert "nope" in result.output
    assert not split_paths(built_copy.output_dir).dir.exists()


def test_msa_dir_path_shares_generate_train_config_legacy_warning(built_copy, tmp_path):
    cache_path = built_copy.output_dir / "dataset_cache.json"
    cache = json.loads(cache_path.read_text())
    for entry in cache["structure_data"].values():
        entry["interfaces"] = {}
        for chain in entry["chains"].values():
            chain["cluster_size"] = 1
    cache_path.write_text(json.dumps(cache, indent=4))
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    result = invoke(built_copy, tmp_path, extra=["--organized-msa-dir", str(msa_dir)])
    expected_text = "built before interface keys"
    assert expected_text in result.output


def test_missing_dataset_cache_exits_nonzero_naming_it(tmp_path):
    """split-dataset on a directory without dataset_cache.json exits 1 with a message, no traceback."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    csv = tmp_path / "my_split.csv"
    csv.write_text(VAL_CSV)

    result = CliRunner().invoke(
        split_dataset, ["--output-dir", str(output_dir), "--split-csv", str(csv)], catch_exceptions=False
    )

    assert result.exit_code == 1
    assert "dataset_cache.json" in result.output
