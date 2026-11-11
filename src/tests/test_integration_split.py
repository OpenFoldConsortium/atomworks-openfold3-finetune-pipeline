"""Integration: split val entries load through OF3's ValidationPDBDataset.

Run with: pixi run python -m pytest tests/test_integration_split.py -m integration -v
"""

import warnings

import pytest
from openfold3.core.data.framework.single_datasets.dataset_utils import check_invalid_feature_dict
from openfold3.core.data.framework.single_datasets.validation import ValidationPDBDataset
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    DATASET_CONFIG_REGISTRY,
    TrainingDatasetPaths,
)

from aw_of3_finetune.split import split_dataset_cache, split_paths
from aw_of3_finetune.training_config import build_train_config
from validate_helpers import mutable_copy

# ValidationPDBConfig is decorated with @register_dataset_config, so fetch it from the registry.
ValidationPDBConfig = DATASET_CONFIG_REGISTRY.get("ValidationPDBDataset")


@pytest.fixture
def built_copy(built_dataset, tmp_path):
    return mutable_copy(built_dataset, tmp_path / "copy")


def split_and_build_val_dataset(built_copy, tmp_path, debug_mode=False):
    """Split 7riv and 3u8v into val and build OF3's ValidationPDBDataset from train.yml's section."""
    csv = tmp_path / "my_split.csv"
    csv.write_text("id,split\n7riv,val\n3u8v,val\n")
    split_dataset_cache(built_copy.output_dir, csv)
    train_config = build_train_config(built_copy.output_dir, built_copy.msa_dir)
    section = train_config["dataset_configs"]["validation"]["val-weighted-pdb"]["config"]
    # The generated config must itself keep debug_mode off; the override is only for the guard test.
    assert section["debug_mode"] is False
    section["debug_mode"] = debug_mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = ValidationPDBConfig(
            name="val-weighted-pdb",
            dataset_paths=TrainingDatasetPaths(**train_config["dataset_paths"]["val-weighted-pdb"]),
            **section,
        )
        return ValidationPDBDataset(config, world_size=1)


@pytest.mark.integration
def test_every_val_entry_loads_with_metric_masks(built_copy, tmp_path):
    dataset = split_and_build_val_dataset(built_copy, tmp_path)
    actual_pdb_ids = set(dataset.datapoint_cache["pdb_id"])
    assert actual_pdb_ids == {"7riv", "3u8v"}
    # 3u8v's only ligand is on OF3's metric exclusion list, so it has no scored interface.
    expected_has_inter_mask = {"7riv": True, "3u8v": False}

    for index in range(len(dataset)):
        pdb_id = dataset.datapoint_cache.iloc[index]["pdb_id"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = dataset[index]
        # Training runs with debug_mode false, which skips this check; run it here.
        check_invalid_feature_dict(features)
        ground_truth = features["ground_truth"]
        has_intra_mask = bool(ground_truth["intra_filter_atomized"].any())
        actual_has_inter_mask = bool(ground_truth["inter_filter_atomized"].any())
        assert has_intra_mask, pdb_id
        assert actual_has_inter_mask == expected_has_inter_mask[pdb_id], pdb_id


@pytest.mark.integration
def test_broken_val_entry_raises(built_copy, tmp_path):
    dataset = split_and_build_val_dataset(built_copy, tmp_path)
    npz = built_copy.output_dir / "of3_compressed_structures" / "7riv" / "7riv.npz"
    npz.unlink()
    index = int(dataset.datapoint_cache.index[dataset.datapoint_cache["pdb_id"] == "7riv"][0])
    with pytest.raises(FileNotFoundError):
        dataset[index]
