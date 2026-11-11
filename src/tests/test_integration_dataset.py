"""Integration smoke test: finetune-prep → materialize-dataset-cache → WeightedPDBDataset.

Run with: pixi run pytest -m integration tests/test_integration_dataset.py -v

Test structure:
- test_weighted_pdb_dataset_e2e: each stage run separately with stubbed MSAs,
  over 3u8v (protein + NI), 2crb (protein only), 7riv (PXR LBD + P06).
- test_finetune_pipeline_orchestrator_e2e: one-shot finetune-pipeline CLI with
  live ColabFold MSAs over two structures.
"""

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    DATASET_CONFIG_REGISTRY,
    TrainingDatasetPaths,
)

# WeightedPDBConfig is decorated with @register_dataset_config which replaces the
# class name with None.  Retrieve the actual class through the registry.
WeightedPDBConfig = DATASET_CONFIG_REGISTRY.get("WeightedPDBDataset")

from aw_of3_finetune.cli import finetune_pipeline, finetune_prep, materialize_dataset_cache
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences
from aw_of3_finetune.pipeline import save_structures_as_npz
from aw_of3_finetune.training_config import build_train_config


@pytest.mark.integration
def test_weighted_pdb_dataset_e2e(cif_for_integration, ccd_path, tmp_path):
    """finetune-prep, materialize-dataset-cache and NPZ conversion yield a dataset
    WeightedPDBDataset loads, with finite non-zero coordinates in dataset[0]."""
    cif_path = cif_for_integration
    pdb_id = cif_path.stem.lower()

    # Step 1: finetune-prep — augmented CIFs + slim manifest + reference mol SDFs.
    cif_input_dir = tmp_path / "cif_input"
    cif_input_dir.mkdir()
    (cif_input_dir / cif_path.name).symlink_to(cif_path)

    output_dir = tmp_path / "finetune_output"
    result = CliRunner().invoke(
        finetune_prep,
        [
            "--cif-dir", str(cif_input_dir),
            "--output-dir", str(output_dir),
            "--name", "smoke",
            "--ccd-path", str(ccd_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # Prep wrote the augmented CIF, the slim manifest and the reference-molecule directory.
    augmented_cif_dir = output_dir / "cleaned_structures"
    slim_manifest_path = output_dir / "manifest.json"
    ref_mol_dir = output_dir / "reference_mols"
    assert (augmented_cif_dir / f"{pdb_id}.cif").exists()
    assert slim_manifest_path.exists()
    assert ref_mol_dir.is_dir()

    # ── slim manifest: only name + structure list ──────────────────────────────
    slim_manifest = json.loads(slim_manifest_path.read_text())

    # The manifest carries the run name and lists the structure.
    assert slim_manifest.get("name") == "smoke"
    assert pdb_id in slim_manifest.get("structures", []), f"{pdb_id} missing from slim manifest"

    # The per-structure data lives in the dataset cache, not here.
    assert "structure_data" not in slim_manifest, "slim manifest should not contain structure_data"

    # Step 2: materialize full dataset cache from augmented CIFs.
    dataset_cache_path = tmp_path / "dataset_cache.json"
    result = CliRunner().invoke(
        materialize_dataset_cache,
        [
            "--output-dir", str(output_dir),
            "--cache-output", str(dataset_cache_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert dataset_cache_path.exists()

    # ── full dataset cache: structure_data + reference_molecule_data ──────────
    dataset_cache = json.loads(dataset_cache_path.read_text())

    # The structure is in the cache and has chains.
    assert pdb_id in dataset_cache["structure_data"], f"{pdb_id} missing from dataset cache"
    chains = dataset_cache["structure_data"][pdb_id]["chains"]
    assert len(chains) > 0, "dataset cache has no chains"

    # Chain keys are integers, from renumber_chain_ids=True.
    for chain_key in chains:
        assert chain_key.isdigit(), (
            f"expected integer chain key (from renumber_chain_ids=True), got {chain_key!r}"
        )

    # Reference molecules are present and each has a conformer strategy, a SMILES and no NaN fallback.
    ref_mol_data = dataset_cache.get("reference_molecule_data", {})
    assert len(ref_mol_data) > 0, "dataset cache has no reference_molecule_data"
    for mol_id, entry in ref_mol_data.items():
        assert "conformer_gen_strategy" in entry, f"missing conformer_gen_strategy for {mol_id}"
        assert "canonical_smiles" in entry, f"missing canonical_smiles for {mol_id}"
        assert entry.get("set_fallback_to_nan") is False

    # Every reference molecule has its SDF file on disk.
    for mol_id in ref_mol_data:
        sdf = ref_mol_dir / f"{mol_id}.sdf"
        assert sdf.exists(), f"SDF missing for {mol_id}: {sdf}"

    # Step 3: stub MSA files.
    # MsaSampleParserTrain looks in {msa_dir}/{rep_id}/ for *.a3m files;
    # an empty directory raises RuntimeError, so provide at least the query row.
    msa_dir = tmp_path / "msas"
    for seq in chain_canonical_sequences(cif_path).values():
        rep_id = alignment_rep_id(seq)
        subdir = msa_dir / rep_id
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "uniref90_hits.a3m").write_text(f">query\n{seq}\n")

    # Step 4: convert augmented CIFs → NPZ.
    # The augmented CIFs are the single source of truth; NPZ files are derived artifacts.
    structures_dir = tmp_path / "of3_compressed_structures"
    save_structures_as_npz(augmented_cif_dir, structures_dir)
    assert (structures_dir / pdb_id / f"{pdb_id}.npz").exists()

    # ── NPZ file ──────────────────────────────────────────────────────────────
    npz = np.load(structures_dir / pdb_id / f"{pdb_id}.npz")

    # The NPZ holds an (n_atoms, 3) coordinate array with some finite values.
    assert "coord" in npz.files, "NPZ missing coord array"
    assert npz["coord"].ndim == 2 and npz["coord"].shape[1] == 3
    assert np.isfinite(npz["coord"]).any(), "no finite coordinates in NPZ"

    # Integer chain IDs ("1", "2", …) must match the dataset cache chain keys.
    assert "chain_id" in npz.files
    assert set(np.unique(npz["chain_id"])) == set(chains.keys())

    # Step 5: instantiate WeightedPDBDataset from the materialized dataset cache.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = WeightedPDBConfig(
            name="smoke",
            dataset_paths=TrainingDatasetPaths(
                dataset_cache_file=dataset_cache_path,
                target_structures_directory=structures_dir,
                target_structure_file_format="npz",
                alignments_directory=msa_dir,
                reference_molecule_directory=ref_mol_dir,
            ),
        )
    dataset = WeightedPDBDataset(config)

    # ── datapoint_cache ───────────────────────────────────────────────────────
    datapoint_cache = dataset.datapoint_cache

    # The datapoint cache is non-empty, has the sampling columns, and starts with our structure.
    assert len(datapoint_cache) > 0, "datapoint_cache is empty"
    assert "pdb_id" in datapoint_cache.columns
    assert "preferred_chain_or_interface" in datapoint_cache.columns
    assert datapoint_cache.iloc[0]["pdb_id"] == pdb_id

    # ── dataset[0]: feature dict with non-zero values ─────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = dataset[0]

    # dataset[0] is a non-empty feature dict.
    assert isinstance(features, dict), "dataset[0] should return a dict"
    assert len(features) > 0, "feature dict is empty"

    # At least one coordinate-like array has finite non-zero values.
    coord_keys = [k for k in features if "coord" in k.lower() or "pos" in k.lower()]
    assert len(coord_keys) > 0, f"no coordinate keys found in features: {list(features)[:10]}"
    for key in coord_keys:
        coords = np.asarray(features[key])
        if coords.size > 0 and np.issubdtype(coords.dtype, np.floating):
            assert np.isfinite(coords).any(), f"all coordinates are non-finite in {key}"
            assert np.any(coords != 0), f"all coordinates are zero in {key}"
            break


@pytest.mark.integration
def test_finetune_pipeline_orchestrator_e2e(
    cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
):
    """finetune-pipeline chains all four stages, with live ColabFold MSAs over two
    structures, into a dataset WeightedPDBDataset can load end-to-end."""
    cif_dir = tmp_path / "cif_input"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
    (cif_dir / "2crb.cif").symlink_to(cif_protein_only.resolve())

    output_dir = tmp_path / "out"
    msa_dir = tmp_path / "msas"
    result = CliRunner().invoke(
        finetune_pipeline,
        [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(output_dir),
            "--ccd-path", str(ccd_path),
            "--msa-dir", str(msa_dir),
            "--name", "pipeline-smoke",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    # Every stage's output exists: cleaned CIFs, dataset cache, NPZ files, OF3 alignments.
    assert (output_dir / "cleaned_structures" / "3u8v.cif").exists()
    assert (output_dir / "cleaned_structures" / "2crb.cif").exists()
    assert (output_dir / "dataset_cache.json").exists()
    assert (output_dir / "of3_compressed_structures" / "3u8v" / "3u8v.npz").exists()
    assert (output_dir / "of3_compressed_structures" / "2crb" / "2crb.npz").exists()
    assert (msa_dir / "of3_alignments").is_dir()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = WeightedPDBConfig(
            name="pipeline-smoke",
            dataset_paths=TrainingDatasetPaths(
                dataset_cache_file=output_dir / "dataset_cache.json",
                target_structures_directory=output_dir / "of3_compressed_structures",
                target_structure_file_format="npz",
                alignments_directory=msa_dir / "of3_alignments",
                reference_molecule_directory=output_dir / "reference_mols",
            ),
        )
        dataset = WeightedPDBDataset(config)

        # The dataset has datapoints, and each one loads as a non-empty feature dict.
        assert len(dataset.datapoint_cache) > 0
        for i in range(len(dataset.datapoint_cache)):
            features = dataset[i]
            assert isinstance(features, dict) and len(features) > 0


@pytest.mark.integration
def test_every_structure_is_sampled_equally(built_dataset):
    """With train.yml's sample_weights, each structure's datapoint probabilities sum to 1,
    interface rows exist, and an interface row loads as a feature dict."""
    train_config = build_train_config(built_dataset.output_dir, built_dataset.msa_dir)
    sample_weights = train_config["dataset_configs"]["train"]["weighted-pdb"]["config"]["sample_weights"]
    dataset_paths = train_config["dataset_paths"]["weighted-pdb"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = WeightedPDBConfig(
            name="equal-sampling",
            dataset_paths=TrainingDatasetPaths(**dataset_paths),
            sample_weights=sample_weights,
        )
        dataset = WeightedPDBDataset(config)
    datapoint_cache = dataset.datapoint_cache.reset_index(drop=True)

    # Interfaces are present for every structure that has more than one chain.
    # OF3's datapoint_cache drops its "type" column: a chain row holds a str, an interface row a list.
    is_interface_row = datapoint_cache["preferred_chain_or_interface"].apply(
        lambda preferred: not isinstance(preferred, str)
    )
    interface_rows = datapoint_cache[is_interface_row]
    actual_pdb_ids_with_interfaces = set(interface_rows["pdb_id"])
    expected_pdb_ids_with_interfaces = {"1a3n", "2q2k", "3u8v", "7riv"}
    assert actual_pdb_ids_with_interfaces == expected_pdb_ids_with_interfaces

    # OF3 does not normalize here; each structure's probabilities already sum to 1.
    actual_share = datapoint_cache.groupby("pdb_id")["datapoint_probabilities"].sum()
    expected_share = 1.0
    assert np.allclose(actual_share.to_numpy(), expected_share)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = dataset[int(interface_rows.index[0])]
    assert len(features) > 0


@pytest.mark.integration
def test_interface_datapoints_load_with_spatial_interface_crop(built_dataset):
    """With the crop forced to spatial_interface and debug_mode off, one interface
    datapoint per structure with interfaces (incl. 7riv's protein-ligand 1_2) loads."""
    train_config = build_train_config(built_dataset.output_dir, built_dataset.msa_dir)
    dataset_config = train_config["dataset_configs"]["train"]["weighted-pdb"]["config"]
    crop = copy.deepcopy(dataset_config["crop"])
    crop["token_crop"]["crop_weights"] = {
        "contiguous": 0.0,
        "spatial": 0.0,
        "spatial_interface": 1.0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = WeightedPDBConfig(
            name="interface-crop",
            dataset_paths=TrainingDatasetPaths(**train_config["dataset_paths"]["weighted-pdb"]),
            sample_weights=dataset_config["sample_weights"],
            crop=crop,
            debug_mode=False,
        )
        dataset = WeightedPDBDataset(config)
    datapoint_cache = dataset.datapoint_cache.reset_index(drop=True)
    is_interface_row = datapoint_cache["preferred_chain_or_interface"].apply(
        lambda preferred: not isinstance(preferred, str)
    )
    first_interface_row_per_pdb_id = datapoint_cache[is_interface_row].groupby("pdb_id").head(1)
    assert "7riv" in set(first_interface_row_per_pdb_id["pdb_id"])

    # Loading must not raise; debug_mode False means a failed load is an error, not a retry.
    for row_index in first_interface_row_per_pdb_id.index:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = dataset[int(row_index)]
        assert len(features) > 0
