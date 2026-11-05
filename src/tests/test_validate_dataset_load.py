"""Tests for V21: every entry loads through OF3's WeightedPDBDataset."""

import json

import pytest
from openfold3.core.data.framework.single_datasets.abstract_single import DATASET_REGISTRY

from aw_of3_finetune.split import split_dataset_cache
from aw_of3_finetune.validate import CHECKS, run_validation
from aw_of3_finetune.validate.findings import Severity
from validate_helpers import context, mutable_copy

pytestmark = pytest.mark.integration


def test_v21_is_registered():
    """V21 is present in the CHECKS registry."""
    assert "V21" in CHECKS


def test_v21_loads_clean_build(built_dataset):
    """Every entry of a clean build loads through WeightedPDBDataset without error."""
    report = run_validation(context(built_dataset), ["V21"])

    assert report.errors == [], [f.message for f in report.errors]


def test_v21_reports_failing_entry(built_dataset, tmp_path):
    """Deleting 7riv's ligand SDF makes only 7riv fail to load, with an error naming P06."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    (built.output_dir / "reference_mols" / "P06.sdf").unlink()

    report = run_validation(context(built), ["V21"])

    # Exactly one error: V21 on 7riv.
    actual_errors = [(f.check, f.severity, f.entry) for f in report.errors]
    assert actual_errors == [("V21", Severity.ERROR, "7riv")]

    # The message points at the missing ligand.
    assert "P06" in report.errors[0].message


def test_v21_without_msa_dir_is_info(built_dataset):
    """Without --msa-dir the dataset cannot be loaded, so V21 emits one dataset-wide INFO."""
    report = run_validation(context(built_dataset, msa_dir=None), ["V21"])

    actual_findings = [(f.check, f.severity, f.entry) for f in report.findings]
    assert actual_findings == [("V21", Severity.INFO, None)]


def test_v21_honors_custom_cache_path(built_dataset, tmp_path):
    """V21 loads the dataset from a cache file moved to a custom path."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    custom = tmp_path / "elsewhere" / "cache.json"
    custom.parent.mkdir()
    (built.output_dir / "dataset_cache.json").rename(custom)

    report = run_validation(context(built, cache_path=custom), ["V21"])

    assert report.errors == [], [f.message for f in report.errors]


def test_v21_loads_each_entry_once(built_dataset, monkeypatch):
    """V21 loads each of the 5 entries exactly once (no entry is loaded twice)."""
    dataset_class = DATASET_REGISTRY["WeightedPDBDataset"]
    original_getitem = dataset_class.__getitem__
    loaded_pdb_ids = []

    def counting(self, index):
        loaded_pdb_ids.append(str(self.datapoint_cache.iloc[index]["pdb_id"]))
        return original_getitem(self, index)

    monkeypatch.setattr(dataset_class, "__getitem__", counting)

    run_validation(context(built_dataset), ["V21"])

    # No entry appears twice, and all five were loaded.
    assert sorted(loaded_pdb_ids) == sorted(set(loaded_pdb_ids))
    assert len(loaded_pdb_ids) == 5


def test_v21_ignores_a_stale_split(built_dataset, tmp_path):
    """A split/ made from an older cache does not stop V21 from loading the full cache."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    csv = tmp_path / "split.csv"
    csv.write_text("id,split\n7riv,val\n")
    split_dataset_cache(built.output_dir, csv)
    cache_path = built.output_dir / "dataset_cache.json"
    cache_path.write_text(cache_path.read_text() + " ")

    report = run_validation(context(built), ["V21"])

    assert report.errors == [], [f.message for f in report.errors]


def test_v21_reports_an_invalid_feature_dict(built_dataset, monkeypatch):
    """A 7riv sample that fails OF3's check_invalid_feature_dict is a V21 error, not a crash."""
    dataset_class = DATASET_REGISTRY["WeightedPDBDataset"]
    original_getitem = dataset_class.__getitem__

    def one_reference_atom_short(self, index):
        features = original_getitem(self, index)
        if features["pdb_id"] == "7riv":
            features["ref_pos"] = features["ref_pos"][..., :-1, :]
        return features

    monkeypatch.setattr(dataset_class, "__getitem__", one_reference_atom_short)

    report = run_validation(context(built_dataset), ["V21"])

    actual_errors = [(f.check, f.severity, f.entry) for f in report.errors]
    assert actual_errors == [("V21", Severity.ERROR, "7riv")]
    assert "reference conformer" in report.errors[0].message


def test_v21_loads_rna_chain_without_msa(built_dataset, tmp_path):
    """An RNA chain has no MSA (ColabFold searches proteins only) and still loads, like DNA."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    cache_path = built.output_dir / "dataset_cache.json"
    cache = json.loads(cache_path.read_text())
    chain = cache["structure_data"]["2crb"]["chains"]["1"]
    chain["molecule_type"] = "RNA"
    chain["alignment_representative_id"] = None
    cache_path.write_text(json.dumps(cache, indent=4))  # read_datacache reads the type from line 2

    report = run_validation(context(built), ["V21"])

    assert report.errors == [], [f.message for f in report.errors]
