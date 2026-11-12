"""Structure prep, manifest building, and the pipeline CLIs.

Two-step flow:
  finetune-prep:              raw CIFs -> augmented CIFs + slim manifest.json
  materialize-dataset-cache:  slim manifest + augmented CIFs -> full dataset_cache.json

Test structure
--------------
TestBuildStructureData     per-chain metadata from prepare_augmented_cif.
TestBuildManifest          augmented CIF dir -> ClusteredDatasetCache JSON.
TestReorganizeMsasForOf3   unpaired .a3m.gz -> OF3's <rep_id>/colabfold_main.a3m.
TestFinetunePrep           finetune-prep CLI end to end.
TestMsaWiring              rep IDs in CIFs match MSA dir names.
TestFinetunePipeline       finetune-pipeline orchestrator CLI.
TestGenerateTrainConfig    generate-train-config output validates against OF3's schema.
"""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import click
import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFFile
from click.testing import CliRunner
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
)
from openfold3.core.data.io.sequence.msa import parse_msas_direct, standardize_filepaths
from openfold3.entry_points.validator import TrainingExperimentConfig

from assertions import write_emptiable_minimal_3aa
from validate_helpers import mutable_copy
import aw_of3_finetune.cli as cli_mod

from aw_of3_finetune.split import split_paths
from aw_of3_finetune.augmented_cif import read_augmented_cif
from aw_of3_finetune.constants import DUMMY_CLUSTER_ID
import json

import yaml

from aw_of3_finetune.cli import (
    finetune_pipeline,
    finetune_prep,
    generate_train_config,
    materialize_dataset_cache,
)
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences
from aw_of3_finetune.msa_sources import MixedMsaDirError
from aw_of3_finetune.split import split_dataset_cache, split_paths
from aw_of3_finetune.training_config import StaleSplitError, build_train_config
from aw_of3_finetune.pipeline import (
    ChainMismatchError,
    NotAugmentedCifError,
    build_manifest,
    check_chain_consistency,
    parse_augmented_cif_for_training,
    prepare_augmented_cif,
    reorganize_msas_for_of3,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_template_cache_npz(rep_id: str, template_ids: list[str], cache_dir: Path) -> Path:
    """Write a fake OF3 template cache NPZ for one alignment rep ID.

    Args:
        rep_id: Alignment rep ID; names the file.
        template_ids: Template IDs to store, in index order.
        cache_dir: Output directory.

    Returns:
        Path of the written ``<cache_dir>/<rep_id>.npz``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{rep_id}.npz"
    data = {
        tid: {
            "index": idx,
            "release_date": "2020-01-01",
            "idx_map": np.zeros((5, 2), dtype=np.int32),
        }
        for idx, tid in enumerate(template_ids)
    }
    np.savez_compressed(cache_file, **data)
    return cache_file


def _write_minimal_cif(cif_path: Path, pdb_id: str, output_dir: Path, **kwargs) -> Path:
    """Write an augmented CIF (no reference mols) for a fixture CIF.

    Args:
        cif_path: Source CIF.
        pdb_id: Structure ID; names the output file.
        output_dir: Directory to write into.
        **kwargs: Passed through to prepare_augmented_cif.

    Returns:
        Path of the written ``<output_dir>/<pdb_id>.cif``.
    """
    out = output_dir / f"{pdb_id}.cif"
    prepare_augmented_cif(cif_path, pdb_id, out, **kwargs)
    return out


def _structure_data(cif_path: Path, pdb_id: str, tmp_path: Path, **kwargs):
    """Run prepare_augmented_cif on a fixture CIF and return its structure data.

    Args:
        cif_path: Source CIF.
        pdb_id: Structure ID.
        tmp_path: Directory for the augmented CIF.
        **kwargs: Passed through to prepare_augmented_cif.

    Returns:
        The ClusteredDatasetStructureData for the structure.
    """
    return prepare_augmented_cif(cif_path, pdb_id, tmp_path / f"{pdb_id}.cif", **kwargs)[0]


# ── TestBuildStructureData ────────────────────────────────────────────────────


class TestBuildStructureData:
    def test_singleton_cluster_id_is_dummy_constant(self, cif_protein_monomer, tmp_path):
        """Every chain gets the same synthetic DUMMY_CLUSTER_ID (no real clustering)."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)

        # Every chain carries the dummy cluster ID.
        actual_cluster_ids = {chain.cluster_id for chain in structure_data.chains.values()}
        assert actual_cluster_ids == {DUMMY_CLUSTER_ID}

    def test_cluster_size_is_datapoint_count(self, cif_protein_monomer, tmp_path):
        """Every chain gets cluster_size = the structure's chains + interfaces."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)

        expected_size = len(structure_data.chains) + len(structure_data.interfaces)
        actual_cluster_sizes = {chain.cluster_size for chain in structure_data.chains.values()}
        assert actual_cluster_sizes == {expected_size}

    def test_protein_chain_has_alignment_rep_id(self, cif_protein_monomer, tmp_path):
        """Polymer chains get a non-None 11-char alignment_representative_id."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)

        # Every protein chain has an 11-character alignment representative ID.
        actual_rep_id_lengths = {
            len(chain.alignment_representative_id or "")
            for chain in structure_data.chains.values()
            if chain.molecule_type == "PROTEIN"
        }
        assert actual_rep_id_lengths == {11}

    def test_ligand_chain_no_alignment_rep_id(self, cif_protein_monomer, tmp_path):
        """LIGAND chains get alignment_representative_id = None (3u8v has NI ions)."""
        # Not 2q2k: atomworks strips its only ligand (HEPES) as a crystallization aid.
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)
        ligands = [chain for chain in structure_data.chains.values() if chain.molecule_type == "LIGAND"]
        assert ligands, "precondition: 3u8v has a ligand chain"

        # No ligand chain has an alignment representative.
        actual_rep_ids = {chain.alignment_representative_id for chain in ligands}
        assert actual_rep_ids == {None}

    def test_dna_chain_no_alignment_rep_id(self, cif_dna_complex, tmp_path):
        """DNA chains get alignment_representative_id = None (MVP: protein-only MSA)."""
        structure_data = _structure_data(cif_dna_complex, "2q2k", tmp_path)
        dna_chains = [chain for chain in structure_data.chains.values() if chain.molecule_type == "DNA"]
        assert dna_chains, "precondition: 2q2k has a DNA chain"

        # No DNA chain has an alignment representative.
        actual_rep_ids = {chain.alignment_representative_id for chain in dna_chains}
        assert actual_rep_ids == {None}

        # No DNA chain has template IDs.
        actual_template_ids = [chain.template_ids for chain in dna_chains]
        assert actual_template_ids == [None] * len(dna_chains)

    def test_template_ids_empty_by_default(self, cif_protein_monomer, tmp_path):
        """Without template_cache_dir, polymer chains get template_ids = []."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)

        # Every protein chain has an empty template list.
        actual_template_ids = [
            chain.template_ids
            for chain in structure_data.chains.values()
            if chain.molecule_type == "PROTEIN"
        ]
        assert actual_template_ids == [[]] * len(actual_template_ids)
        assert actual_template_ids, "precondition: 3u8v has a protein chain"

    def test_template_ids_none_for_ligand(self, cif_protein_monomer, tmp_path):
        """LIGAND chains always get template_ids = None (3u8v has NI ions)."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)
        ligands = [chain for chain in structure_data.chains.values() if chain.molecule_type == "LIGAND"]
        assert ligands, "precondition: 3u8v has a ligand chain"

        # No ligand chain has template IDs.
        actual_template_ids = [chain.template_ids for chain in ligands]
        assert actual_template_ids == [None] * len(ligands)

    def test_template_ids_populated_from_cache(self, cif_protein_monomer, tmp_path):
        """With a template cache, the protein chain gets its template_ids populated."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        expected = ["1abc_A", "2def_B"]
        _make_template_cache_npz(rep_id, expected, tmp_path)

        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path, template_cache_dir=tmp_path)
        # Chain "1" is the first integer chain ID assigned by parse_mmcif for 3u8v
        # bioassembly 1; it corresponds to the protein chain (label_asym_id="A").
        protein_chains = [chain for chain in structure_data.chains.values() if chain.molecule_type == "PROTEIN"]
        assert len(protein_chains) == 1

        # The protein chain's template IDs are the ones stored in the cache, in index order.
        assert protein_chains[0].template_ids == expected

    def test_release_date_extracted(self, cif_protein_monomer, tmp_path):
        """release_date is a date: the earliest pdbx_audit_revision_history date."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)

        # 3u8v was deposited 2011-10-17 and first released 2011-11-09.
        assert isinstance(structure_data.release_date, date)
        assert structure_data.release_date == date(2011, 11, 9)

    def test_resolution_extracted_xray(self, cif_protein_monomer, tmp_path):
        """X-ray structures have a float resolution."""
        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)

        # 3u8v's resolution is 1.90 A.
        assert isinstance(structure_data.resolution, float)
        assert abs(structure_data.resolution - 1.90) < 0.01

    def test_resolution_none_for_nmr(self, cif_protein_only, tmp_path):
        """NMR structures (2crb) return resolution = None."""
        structure_data = _structure_data(cif_protein_only, "2crb", tmp_path)

        # No resolution is recorded.
        assert structure_data.resolution is None

    def test_multimer_all_chains_clustered(self, cif_dna_complex, tmp_path):
        """All chains in a multimer (2q2k) get the same synthetic dummy cluster."""
        structure_data = _structure_data(cif_dna_complex, "2q2k", tmp_path)
        assert len(structure_data.chains) > 1

        # Every chain shares the dummy cluster ID and the structure's datapoint count.
        expected_size = len(structure_data.chains) + len(structure_data.interfaces)
        actual_clusters = {
            (chain.cluster_id, chain.cluster_size) for chain in structure_data.chains.values()
        }
        assert actual_clusters == {(DUMMY_CLUSTER_ID, expected_size)}


# ── TestBuildManifest ─────────────────────────────────────────────────────────


class TestBuildManifest:
    def test_empty_dir_empty_manifest(self, tmp_path):
        """Empty cif_dir produces an empty but valid ClusteredDatasetCache."""
        manifest_path = tmp_path / "manifest.json"
        cache = build_manifest(tmp_path, manifest_path, cache_name="test")

        # No structures, but the manifest file was still written.
        assert cache.structure_data == {}
        assert manifest_path.exists()

    def test_manifest_contains_all_structures(
        self, cif_protein_monomer, cif_protein_only, tmp_path
    ):
        """Two CIFs → two structure_data entries in the manifest."""
        cif_dir = tmp_path / "cleaned_structures"
        cif_dir.mkdir()
        _write_minimal_cif(cif_protein_monomer, "3u8v", cif_dir)
        _write_minimal_cif(cif_protein_only, "2crb", cif_dir)

        cache = build_manifest(cif_dir, tmp_path / "manifest.json", cache_name="test")

        # Both input structures appear in the manifest.
        assert set(cache.structure_data.keys()) == {"3u8v", "2crb"}

    def test_manifest_json_loadable(self, cif_protein_monomer, tmp_path):
        """Manifest JSON round-trips through ClusteredDatasetCache.from_json()."""
        cif_dir = tmp_path / "cleaned_structures"
        cif_dir.mkdir()
        _write_minimal_cif(cif_protein_monomer, "3u8v", cif_dir)

        manifest_path = tmp_path / "manifest.json"
        build_manifest(cif_dir, manifest_path, cache_name="test")

        loaded = ClusteredDatasetCache.from_json(manifest_path)

        # The reloaded cache contains the structure.
        assert "3u8v" in loaded.structure_data

    def test_manifest_cluster_ids_preserved(self, cif_protein_monomer, tmp_path):
        """Cluster IDs written into the CIF survive the manifest round-trip."""
        cif_dir = tmp_path / "cleaned_structures"
        cif_dir.mkdir()
        _write_minimal_cif(cif_protein_monomer, "3u8v", cif_dir)

        manifest_path = tmp_path / "manifest.json"
        cache = build_manifest(cif_dir, manifest_path, cache_name="test")

        # Every chain still has the dummy cluster ID and the structure's datapoint count.
        structure_data = cache.structure_data["3u8v"]
        expected_size = len(structure_data.chains) + len(structure_data.interfaces)
        actual_clusters = {
            (chain.cluster_id, chain.cluster_size) for chain in structure_data.chains.values()
        }
        assert actual_clusters == {(DUMMY_CLUSTER_ID, expected_size)}


# ── TestReorganizeMsasForOf3 ──────────────────────────────────────────────────


def _make_output_dir_from_cif(cif_path: Path, pdb_id: str, tmp_path: Path) -> Path:
    """Build a finetune-prep output_dir for one CIF via the real finetune_prep CLI.

    Args:
        cif_path: Source CIF.
        pdb_id: Structure ID.
        tmp_path: Scratch directory; inputs go in ``input/``, outputs in ``out/``.

    Returns:
        The output directory (manifest.json + cleaned_structures/).
    """
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / f"{pdb_id}.cif").symlink_to(cif_path.resolve())
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(finetune_prep, [
        "--cif-dir", str(cif_dir),
        "--output-dir", str(output_dir),
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return output_dir


def _make_unpaired_msa_gz(seq: str, msa_dir: Path) -> Path:
    """Write a query-only .a3m.gz at ``<msa_dir>/unpaired/<shard>/<hash>.a3m.gz``.

    Args:
        seq: Query sequence.
        msa_dir: MSA root (as generate-msas writes it).

    Returns:
        Path of the written file.
    """
    rep = alignment_rep_id(seq)
    shard_dir = msa_dir / "unpaired" / rep[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"{rep}.a3m.gz"
    with gzip.open(out, "wt") as f:
        f.write(f">query\n{seq}\n")
    return out


class TestReorganizeMsasForOf3:
    def test_reorganizes_matching_msa(self, cif_protein_monomer, tmp_path):
        """A matching unpaired .a3m.gz is decompressed to
        {of3_msa_dir}/{rep_id}/colabfold_main.a3m, the plain-text layout OF3 expects."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        _make_unpaired_msa_gz(seq, tmp_path / "msas")

        of3_msa_dir = tmp_path / "msas" / "of3_alignments"
        summary = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)

        # One MSA was written and none were missing.
        assert summary.written == {"colabfold_main": 1}
        assert summary.missing == 0

        # The file holds the query-only MSA as plain text.
        out_file = of3_msa_dir / rep_id / "colabfold_main.a3m"
        assert out_file.exists()
        assert out_file.read_text() == f">query\n{seq}\n"

    def test_reorganized_msa_is_parseable_by_of3(self, cif_protein_monomer, tmp_path):
        """reorganize_msas_for_of3 output loads with OF3's own parse_msas_direct."""
        # The contract WeightedPDBDataset.alignments_directory depends on; tested on
        # the real writer's output, not a synthetic stand-in.
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _make_unpaired_msa_gz(seq, tmp_path / "msas")

        of3_msa_dir = tmp_path / "msas" / "of3_alignments"
        summary = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)
        assert summary.written == {"colabfold_main": 1}

        rep_id = alignment_rep_id(seq)
        msas = parse_msas_direct(standardize_filepaths(of3_msa_dir / rep_id))

        # OF3 reads one colabfold_main MSA with a single (query) row.
        assert "colabfold_main" in msas
        assert msas["colabfold_main"].msa.shape == (1, len(seq))

    def test_missing_msa_counted_not_errored(self, cif_protein_monomer, tmp_path):
        """A sequence with no matching MSA is counted as missing, not an error."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        of3_msa_dir = tmp_path / "msas" / "of3_alignments"

        summary = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)

        # Nothing was written; the one chain's MSA is counted as missing.
        assert sum(summary.written.values()) == 0
        assert summary.missing == 1

        # No MSA directory content was created.
        assert not of3_msa_dir.exists() or not list(of3_msa_dir.iterdir())

    def test_rerun_is_idempotent(self, cif_protein_monomer, tmp_path):
        """Calling twice doesn't fail or duplicate work -- existing files are skipped."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _make_unpaired_msa_gz(seq, tmp_path / "msas")
        of3_msa_dir = tmp_path / "msas" / "of3_alignments"

        first = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)
        second = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)

        # First call writes the one MSA; second call finds it already reorganized and skips it.
        assert (first.written, first.removed, first.missing) == ({"colabfold_main": 1}, 0, 0)
        assert (second.written, second.removed, second.missing) == ({"colabfold_main": 0}, 0, 0)

    def test_overwrites_changed_store_file(self, cif_protein_monomer, tmp_path):
        """of3_alignments follows the store: a changed store file is rewritten."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        store_file = _make_unpaired_msa_gz(seq, tmp_path / "msas")
        of3_msa_dir = tmp_path / "msas" / "of3_alignments"
        reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)
        with gzip.open(store_file, "wt") as f:
            f.write(f">query\n{seq}\n>hit\n{seq}\n")

        summary = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)

        assert summary.written == {"colabfold_main": 1}
        assert (of3_msa_dir / alignment_rep_id(seq) / "colabfold_main.a3m").read_text() == (
            f">query\n{seq}\n>hit\n{seq}\n"
        )

    def test_removes_file_without_store(self, cif_protein_monomer, tmp_path):
        """An .a3m whose source no longer provides that sequence is removed."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _make_unpaired_msa_gz(seq, tmp_path / "msas")
        of3_msa_dir = tmp_path / "msas" / "of3_alignments"
        stray = of3_msa_dir / alignment_rep_id(seq) / "old_source.a3m"
        stray.parent.mkdir(parents=True)
        stray.write_text(f">query\n{seq}\n")

        summary = reorganize_msas_for_of3(output_dir, tmp_path / "msas", of3_msa_dir)

        assert not stray.exists()
        assert summary.removed == 1
        assert (of3_msa_dir / alignment_rep_id(seq) / "colabfold_main.a3m").exists()

    def test_indexed_store_missing_on_disk_warns(self, cif_protein_monomer, tmp_path, caplog):
        """A source listed in the index whose store was deleted: warning, and its files removed."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        msa_dir = tmp_path / "msas"
        msa_dir.mkdir()
        (msa_dir / "msa_index.yml").write_text("gone: msa_sources/gone\n")
        old = msa_dir / "of3_alignments" / alignment_rep_id(seq) / "gone.a3m"
        old.parent.mkdir(parents=True)
        old.write_text(f">query\n{seq}\n")

        with caplog.at_level("WARNING"):
            summary = reorganize_msas_for_of3(output_dir, msa_dir, msa_dir / "of3_alignments")

        assert "msa_sources/gone" in caplog.text
        assert not old.exists()
        assert summary.removed == 1
        assert summary.missing == 1

    def test_leaves_non_dataset_rep_ids_alone(self, cif_protein_monomer, tmp_path):
        """Directories of sequences not in this dataset are not touched."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _make_unpaired_msa_gz(seq, tmp_path / "msas")
        other = tmp_path / "msas" / "of3_alignments" / "zzzzzzzzzzz" / "x.a3m"
        other.parent.mkdir(parents=True)
        other.write_text(">q\nAAA\n")

        reorganize_msas_for_of3(output_dir, tmp_path / "msas", tmp_path / "msas" / "of3_alignments")

        assert other.exists()

    def test_mixed_index_refused(self, cif_protein_monomer, tmp_path):
        """A hand-written index mixing ColabFold and user MSAs is refused."""
        output_dir = _make_output_dir_from_cif(cif_protein_monomer, "3u8v", tmp_path)
        msa_dir = tmp_path / "msas"
        msa_dir.mkdir()
        (msa_dir / "msa_index.yml").write_text("colabfold_main: unpaired\nx: msa_sources/x\n")

        with pytest.raises(MixedMsaDirError):
            reorganize_msas_for_of3(output_dir, msa_dir, msa_dir / "of3_alignments")

    def test_cli_msa_dir_flag_reorganizes(self, cif_protein_monomer, tmp_path):
        """materialize-dataset-cache --organized-msa-dir reorganizes MSAs as a side effect."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        output_dir = tmp_path / "out"
        CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir), "--output-dir", str(output_dir),
        ], catch_exceptions=False)

        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        msa_dir = tmp_path / "msas"
        _make_unpaired_msa_gz(seq, msa_dir)

        result = CliRunner().invoke(materialize_dataset_cache, [
            "--output-dir", str(output_dir),
            "--organized-msa-dir", str(msa_dir),
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output

        # The command reports the reorganized MSA and the file is in place.
        assert "Reorganized 1 MSA" in result.output
        assert (msa_dir / "of3_alignments" / rep_id / "colabfold_main.a3m").exists()


# ── TestFinetunePrep (CLI) ────────────────────────────────────────────────────


class TestFinetunePrep:
    def _run(self, args: list[str]) -> object:
        """Invoke finetune-prep with ``args``.

        Args:
            args: CLI arguments.

        Returns:
            The CliRunner result.
        """
        runner = CliRunner()
        return runner.invoke(finetune_prep, args, catch_exceptions=False)

    def test_written_cifs_have_no_waters_or_crystallization_aids(
        self, cif_pxr, cif_dna_complex, tmp_path
    ):
        """The augmented CIFs finetune-prep writes are cleaned: no waters, no HEPES (2q2k)."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "7riv.cif").symlink_to(cif_pxr)
        (cif_dir / "2q2k.cif").symlink_to(cif_dna_complex)
        output_dir = tmp_path / "out"

        result = self._run(["--cif-dir", str(cif_dir), "--output-dir", str(output_dir)])
        assert result.exit_code == 0, result.output

        def residue_names(cif_path: Path) -> set[str]:
            return set(CIFFile.read(cif_path).block["atom_site"]["label_comp_id"].as_array(str))

        # The inputs do contain what cleanup should remove.
        assert "HOH" in residue_names(cif_pxr)
        assert "EPE" in residue_names(cif_dna_complex)

        # The written CIFs don't.
        actual_7riv = residue_names(output_dir / "cleaned_structures" / "7riv.cif")
        actual_2q2k = residue_names(output_dir / "cleaned_structures" / "2q2k.cif")
        assert actual_7riv & {"HOH", "DOD"} == set()
        assert actual_2q2k & {"HOH", "DOD", "EPE"} == set()

    def test_creates_cif_per_cif(
        self, cif_protein_monomer, cif_protein_only, tmp_path
    ):
        """One CIF is written per input CIF into <output>/cleaned_structures/."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)
        (cif_dir / "2crb.cif").symlink_to(cif_protein_only)

        result = self._run([
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output

        # One augmented CIF per input structure.
        assert (tmp_path / "out" / "cleaned_structures" / "3u8v.cif").exists()
        assert (tmp_path / "out" / "cleaned_structures" / "2crb.cif").exists()

    def test_creates_manifest_json(self, cif_protein_monomer, tmp_path):
        """finetune-prep always writes manifest.json in the output dir."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)

        result = self._run([
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out" / "manifest.json").exists()

    def test_manifest_slim_format(self, cif_protein_monomer, tmp_path):
        """manifest.json written by finetune-prep is a slim index (name + structures only)."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)

        self._run([
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
            "--name", "my_finetune",
        ])

        slim_manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())

        # The manifest holds the run name and the structure list...
        assert slim_manifest["name"] == "my_finetune"
        assert "3u8v" in slim_manifest["structures"]

        # ...but not the per-structure data.
        assert "structure_data" not in slim_manifest

    def test_materialize_dataset_cache(self, cif_protein_monomer, tmp_path):
        """materialize-dataset-cache reads slim manifest + augmented CIFs → full cache."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)

        self._run([
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
            "--name", "my_finetune",
        ])

        runner = CliRunner()
        result = runner.invoke(
            materialize_dataset_cache,
            ["--output-dir", str(tmp_path / "out")],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        cache = ClusteredDatasetCache.from_json(tmp_path / "out" / "dataset_cache.json")

        # The full cache keeps the run name and has the structure's data.
        assert cache.name == "my_finetune"
        assert "3u8v" in cache.structure_data

    def test_no_cifs_gives_usage_error(self, tmp_path):
        """An empty --cif-dir exits with a non-zero code and a helpful message."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(finetune_prep, [
            "--cif-dir", str(empty_dir),
            "--output-dir", str(tmp_path / "out"),
        ])
        # The command fails, with the "No .cif files" message or an exception.
        assert result.exit_code != 0
        assert "No .cif files" in result.output or (
            result.exception is not None
        )

    def test_msa_dir_warns_missing(self, cif_protein_monomer, tmp_path):
        """--organized-msa-dir provided but no MSA present → warning on stderr, not a failure."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)
        empty_msa_dir = tmp_path / "msas"
        empty_msa_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(finetune_prep, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
            "--organized-msa-dir", str(empty_msa_dir),
        ], catch_exceptions=False)
        # The run succeeds but warns about the missing MSA.
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_writes_symmetry_expanded_structure_without_raising(self, cif_dna_complex, tmp_path):
        """finetune-prep writes 2q2k (symmetry copies of one entity) without raising."""
        # Regression for chain_disambiguation="chain_iid" in parsing.write_cif_buffer;
        # without it atomworks' CIF writer raised "Ambiguous chain annotations detected".
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "2q2k.cif").symlink_to(cif_dna_complex.resolve())

        result = CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        aug_cif = tmp_path / "out" / "cleaned_structures" / "2q2k.cif"
        assert aug_cif.exists()

        block = CIFFile.read(aug_cif).block
        labels = set(block["atom_site"]["label_asym_id"].as_array().tolist())

        # chain_iid disambiguation writes "<label>_<transformation_id>" for symmetry
        # copies (e.g. "A_1", "A_2") rather than colliding "A"/"A".
        assert any("_" in label for label in labels), (
            f"expected chain_iid-disambiguated labels (e.g. 'A_1'), got {labels}"
        )

    def test_chain_mismatch_fails_loudly(self, cif_protein_monomer, tmp_path, monkeypatch):
        """A check_chain_consistency failure is surfaced and finetune-prep exits non-zero."""
        # No real fixture triggers a mismatch any more, so inject one.
        def boom(out_cif, structure_data):
            raise ChainMismatchError(f"{out_cif.name}: injected mismatch")

        monkeypatch.setattr(cli_mod, "check_chain_consistency", boom)
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())

        result = CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
        ])

        # The command fails and shows the mismatch message.
        assert result.exit_code != 0
        assert "3u8v.cif: injected mismatch" in result.output

    def test_chain_mismatch_skipped_with_flag(
        self, cif_protein_monomer, cif_protein_only, tmp_path, monkeypatch
    ):
        """--skip-inconsistent: the flagged structure is warned about and dropped
        (no augmented CIF, not in the manifest); the other still goes through."""
        real = cli_mod.check_chain_consistency

        def fail_2crb(out_cif, structure_data):
            if out_cif.stem == "2crb":
                raise ChainMismatchError(f"{out_cif.name}: injected mismatch")
            real(out_cif, structure_data)

        monkeypatch.setattr(cli_mod, "check_chain_consistency", fail_2crb)
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        (cif_dir / "2crb.cif").symlink_to(cif_protein_only.resolve())
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(output_dir),
            "--skip-inconsistent",
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output

        # The flagged structure is reported as skipped and gets no augmented CIF.
        assert "skipping 2crb" in result.output
        assert not (output_dir / "cleaned_structures" / "2crb.cif").exists()

        # The other structure is written and is the only one in the manifest.
        assert (output_dir / "cleaned_structures" / "3u8v.cif").exists()
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["structures"] == ["3u8v"]

    def test_consistent_structure_passes_check(self, cif_protein_monomer, ccd_path, tmp_path):
        """check_chain_consistency passes on a consistent structure and raises once
        the metadata is tampered with."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        output_dir = tmp_path / "out"
        CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir), "--output-dir", str(output_dir),
            "--ccd-path", str(ccd_path),
        ], catch_exceptions=False)

        aug = output_dir / "cleaned_structures" / "3u8v.cif"
        structure_data = read_augmented_cif(aug, "3u8v", "t").structure_data["3u8v"]
        # The untouched metadata passes the check (must not raise).
        check_chain_consistency(aug, structure_data)

        # Changing chain 1's molecule type to DNA makes the check raise.
        structure_data.chains["1"].molecule_type = "DNA"
        with pytest.raises(ChainMismatchError, match="chain 1: metadata says DNA"):
            check_chain_consistency(aug, structure_data)

    def test_2q2k_metadata_is_built_from_atomworks_atoms(self, cif_dna_complex, ccd_path, tmp_path):
        """2q2k chain metadata describes the chains atomworks produced, not a separate
        OF3 parse of the source CIF (regression for the two-parser chain mismatch)."""
        # Expect 6 chains (HEPES stripped), chain 4 = second DNA copy "A_2", no EPE,
        # and protein rep IDs still set despite "A_1"/"A_2" labels (entity-keyed lookup).
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "2q2k.cif").symlink_to(cif_dna_complex.resolve())
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(output_dir),
            "--ccd-path", str(ccd_path),
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        aug = output_dir / "cleaned_structures" / "2q2k.cif"
        structure_data = read_augmented_cif(aug, "2q2k", "t").structure_data["2q2k"]

        # Two copies of the DNA duplex (1, 4) and four protein chains (2, 3, 5, 6);
        # HEPES (EPE) is removed by cleanup, so it has no chain.
        expected_molecule_types = {
            "1": "DNA", "2": "PROTEIN", "3": "PROTEIN",
            "4": "DNA", "5": "PROTEIN", "6": "PROTEIN",
        }
        actual_molecule_types = {
            chain_id: chain.molecule_type for chain_id, chain in structure_data.chains.items()
        }
        assert actual_molecule_types == expected_molecule_types

        # Chain 4 is the second bioassembly copy of label chain A.
        assert structure_data.chains["4"].label_asym_id == "A_2"

        # Only protein chains get an MSA representative.
        chains_with_msa = {
            chain_id
            for chain_id, chain in structure_data.chains.items()
            if chain.alignment_representative_id is not None
        }
        assert chains_with_msa == {"2", "3", "5", "6"}

        # No chain points at HEPES as its reference molecule.
        reference_mols = {chain.reference_mol_id for chain in structure_data.chains.values()}
        assert "EPE" not in reference_mols

        # No reference-molecule file was written for HEPES.
        assert not (output_dir / "reference_mols" / "EPE.sdf").exists()

        # The metadata is consistent with the written CIF (must not raise).
        check_chain_consistency(aug, structure_data)

    def test_augmented_cif_parses_for_training_without_ccd(self, cif_protein_monomer, tmp_path):
        """Without --ccd-path the augmented CIF still parses for training: chem_comp
        is carried over from the source CIF."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(finetune_prep, [
            "--cif-dir", str(cif_dir), "--output-dir", str(output_dir),
        ], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        aug = output_dir / "cleaned_structures" / "3u8v.cif"

        # chem_comp was carried over from the source CIF.
        assert "chem_comp" in CIFFile.read(aug).block

        atom_array = parse_augmented_cif_for_training(aug).atom_array
        structure_data = read_augmented_cif(aug, "3u8v", "t").structure_data["3u8v"]

        # The parsed atoms use exactly the chain IDs the metadata describes.
        assert set(atom_array.chain_id.tolist()) == set(structure_data.chains)

    def test_training_parse_refuses_raw_cif(self, cif_protein_monomer):
        """parse_augmented_cif_for_training rejects a CIF this pipeline didn't write."""
        # A raw CIF would otherwise parse into an asym-unit NPZ whose chain IDs match no
        # metadata; the of3_structure_meta marker is the only guard.
        with pytest.raises(NotAugmentedCifError, match="not an augmented CIF"):
            parse_augmented_cif_for_training(cif_protein_monomer)


# ── TestMsaWiring ─────────────────────────────────────────────────────────────


def _make_dummy_msa(seq: str, msa_dir: Path) -> Path:
    """Create a minimal A3M file at {msa_dir}/{rep_id}/uniref90_hits.a3m.

    Args:
        seq: Query sequence.
        msa_dir: MSA root.

    Returns:
        The ``{msa_dir}/{rep_id}`` subdirectory.
    """
    rep_id = alignment_rep_id(seq)
    subdir = msa_dir / rep_id
    subdir.mkdir(parents=True, exist_ok=True)
    a3m = subdir / "uniref90_hits.a3m"
    a3m.write_text(f">query\n{seq}\n")
    return subdir


class TestMsaWiring:
    def test_rep_id_in_cif_matches_msa_dir_name(self, cif_protein_monomer, tmp_path):
        """alignment_representative_id stored in the CIF equals the MSA subdir name."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        expected_rep_id = alignment_rep_id(seq)
        msa_subdir = _make_dummy_msa(seq, tmp_path / "msas")
        assert msa_subdir.name == expected_rep_id

        structure_data = _structure_data(cif_protein_monomer, "3u8v", tmp_path)
        # Find the protein chain by molecule_type (chain key is now an integer string)
        protein_chains = [chain for chain in structure_data.chains.values() if chain.molecule_type == "PROTEIN"]
        assert len(protein_chains) >= 1

        # The protein chain's representative ID is the hash the MSA directory is named by.
        assert protein_chains[0].alignment_representative_id == expected_rep_id

    def test_rep_id_survives_cif_roundtrip(self, cif_protein_monomer, tmp_path):
        """alignment_representative_id written into a CIF is recovered unchanged."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        expected_rep_id = alignment_rep_id(seq)

        cif_path = _write_minimal_cif(cif_protein_monomer, "3u8v", tmp_path)
        reconstructed = read_augmented_cif(cif_path, "3u8v", "test")
        protein_chains = [
            chain for chain in reconstructed.structure_data["3u8v"].chains.values()
            if chain.molecule_type == "PROTEIN"
        ]
        assert len(protein_chains) >= 1

        # The representative ID read back equals the one computed from the sequence.
        assert protein_chains[0].alignment_representative_id == expected_rep_id

    def test_cli_no_warning_when_msa_present(self, cif_protein_monomer, tmp_path):
        """--organized-msa-dir with dummy A3Ms under of3_alignments/ → no WARNING in output."""
        msa_dir = tmp_path / "msas"
        for seq in chain_canonical_sequences(cif_protein_monomer).values():
            _make_dummy_msa(seq, msa_dir / "of3_alignments")

        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)

        runner = CliRunner()
        result = runner.invoke(finetune_prep, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
            "--organized-msa-dir", str(msa_dir),
        ], catch_exceptions=False)
        # The run succeeds and does not warn about missing MSAs.
        assert result.exit_code == 0
        assert "WARNING" not in result.output


# ── TestFinetunePipeline (orchestrator CLI) ──────────────────────────────────


class TestFinetunePipeline:
    def test_runs_all_stages_without_msa_dir(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        """Without --organized-msa-dir, generate-msas and generate-train-config are skipped but
        every other stage writes its artifact (CIFs, manifest, cache, NPZs)."""
        # No train.yml: OF3's training schema requires a real alignments directory.
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        (cif_dir / "2crb.cif").symlink_to(cif_protein_only.resolve())
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(output_dir),
            "--ccd-path", str(ccd_path),
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output

        # The two MSA-dependent stages report being skipped.
        assert "skipped (no --organized-msa-dir given)" in result.output
        assert "generate-train-config -- skipped" in result.output

        # Every other stage wrote its artifact.
        assert (output_dir / "cleaned_structures" / "3u8v.cif").exists()
        assert (output_dir / "cleaned_structures" / "2crb.cif").exists()
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "dataset_cache.json").exists()
        assert (output_dir / "of3_compressed_structures" / "3u8v" / "3u8v.npz").exists()
        assert (output_dir / "of3_compressed_structures" / "2crb" / "2crb.npz").exists()

        # No train.yml, since there is no MSA directory.
        assert not (output_dir / "train.yml").exists()

    def test_requires_ccd_path(self, cif_protein_monomer, tmp_path):
        """--ccd-path is required (unlike finetune-prep): NPZ conversion needs chem_comp."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())

        result = CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(tmp_path / "out"),
        ])

        # The command fails and names the missing --ccd-path option.
        assert result.exit_code != 0
        assert "ccd-path" in result.output.lower()


    def _inputs(self, tmp_path, cif_protein_monomer, cif_protein_only):
        """Make an input dir with 3u8v and 2crb.

        Args:
            tmp_path: Scratch directory.
            cif_protein_monomer: 3u8v fixture CIF.
            cif_protein_only: 2crb fixture CIF.

        Returns:
            The input CIF directory.
        """
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        (cif_dir / "2crb.cif").symlink_to(cif_protein_only.resolve())
        return cif_dir

    def _run(self, cif_dir, output_dir, ccd_path, *extra):
        return CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir),
            "--output-dir", str(output_dir),
            "--ccd-path", str(ccd_path),
            *extra,
        ], catch_exceptions=False)

    def test_split_csv_typo_fails_before_stage_1(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        """A CSV id that matches no input CIF stops the run before any stage."""
        cif_dir = self._inputs(tmp_path, cif_protein_monomer, cif_protein_only)
        split_csv = tmp_path / "split.csv"
        split_csv.write_text("id,split\n7rxx,val\n")
        output_dir = tmp_path / "out"

        result = self._run(cif_dir, output_dir, ccd_path, "--split-csv", str(split_csv))

        assert result.exit_code != 0
        assert "7rxx" in result.output
        assert "Stage 1/10" not in result.output
        assert not (output_dir / "dataset_cache.json").exists()

    def test_split_stage_runs_before_train_config(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        """The split stage runs after NPZ conversion and writes val_cache.json."""
        cif_dir = self._inputs(tmp_path, cif_protein_monomer, cif_protein_only)
        split_csv = tmp_path / "split.csv"
        split_csv.write_text("id,split\n3u8v,val\n")
        output_dir = tmp_path / "out"

        result = self._run(cif_dir, output_dir, ccd_path, "--split-csv", str(split_csv))

        assert result.exit_code == 0, result.output
        assert result.output.index("split-dataset") < result.output.index("generate-train-config")
        assert "Next: generate-train-config" not in result.output
        assert split_paths(output_dir).val_cache.exists()

    def test_without_split_csv_banners_say_stage_skipped(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        cif_dir = self._inputs(tmp_path, cif_protein_monomer, cif_protein_only)

        result = self._run(cif_dir, tmp_path / "out", ccd_path)

        assert result.exit_code == 0, result.output
        assert "=== Stage 7/10: split-dataset -- skipped (no --split-csv) ===" in result.output

    def test_id_dropped_mid_pipeline_fails_the_split_stage_only(
        self, cif_protein_monomer, ccd_path, tmp_path
    ):
        """An input CIF that finetune-prep drops cannot be a split id; earlier outputs stay."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        emptied = write_emptiable_minimal_3aa(cif_dir)
        split_csv = tmp_path / "split.csv"
        split_csv.write_text(f"id,split\n{emptied.stem},val\n")
        output_dir = tmp_path / "out"

        result = self._run(cif_dir, output_dir, ccd_path, "--split-csv", str(split_csv))

        assert result.exit_code != 0
        assert "dropped by the pipeline" in result.output
        assert (output_dir / "dataset_cache.json").exists()

    def test_split_csv_with_custom_cache_output_fails_before_stage_1(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        """split-dataset reads <output-dir>/dataset_cache.json, so another --cache-output is refused."""
        cif_dir = self._inputs(tmp_path, cif_protein_monomer, cif_protein_only)
        split_csv = tmp_path / "split.csv"
        split_csv.write_text("id,split\n3u8v,val\n")
        output_dir = tmp_path / "out"
        custom_cache = tmp_path / "elsewhere" / "cache.json"

        result = self._run(
            cif_dir, output_dir, ccd_path, "--split-csv", str(split_csv), "--cache-output", str(custom_cache)
        )

        assert result.exit_code != 0
        assert "--cache-output" in result.output
        assert "Stage 1/10" not in result.output

    def test_existing_split_without_split_csv_fails_before_stage_1(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        """A split/ from an earlier run would go stale; refuse before doing any work."""
        cif_dir = self._inputs(tmp_path, cif_protein_monomer, cif_protein_only)
        output_dir = tmp_path / "out"
        (output_dir / "split").mkdir(parents=True)

        result = self._run(cif_dir, output_dir, ccd_path)

        assert result.exit_code != 0
        assert "--split-csv" in result.output
        assert "split/" in result.output
        assert "Stage 1/10" not in result.output

    def test_existing_split_with_split_csv_is_regenerated(
        self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path
    ):
        cif_dir = self._inputs(tmp_path, cif_protein_monomer, cif_protein_only)
        split_csv = tmp_path / "split.csv"
        split_csv.write_text("id,split\n3u8v,val\n")
        output_dir = tmp_path / "out"
        first = self._run(cif_dir, output_dir, ccd_path, "--split-csv", str(split_csv))
        assert first.exit_code == 0, first.output

        second = self._run(cif_dir, output_dir, ccd_path, "--split-csv", str(split_csv))

        assert second.exit_code == 0, second.output
        assert split_paths(output_dir).val_cache.exists()


class TestGenerateTrainConfig:
    def _make_pipeline_output(self, cif_protein_monomer, ccd_path, tmp_path) -> Path:
        """Build a full pipeline output_dir for 3u8v without MSAs (no ColabFold calls).

        Args:
            cif_protein_monomer: 3u8v fixture CIF.
            ccd_path: Local CCD mirror.
            tmp_path: Scratch directory.

        Returns:
            The output directory.
        """
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(
            finetune_pipeline,
            [
                "--cif-dir", str(cif_dir),
                "--output-dir", str(output_dir),
                "--ccd-path", str(ccd_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        return output_dir

    def _make_msa_dir(self, cif_protein_monomer, output_dir, tmp_path) -> Path:
        """Reorganize a dummy MSA into {msa_dir}/of3_alignments via materialize-dataset-cache.

        Args:
            cif_protein_monomer: 3u8v fixture CIF.
            output_dir: Pipeline output dir from ``_make_pipeline_output``.
            tmp_path: Scratch directory; the MSA root is ``tmp_path/msas``.

        Returns:
            The MSA root directory.
        """
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        msa_dir = tmp_path / "msas"
        _make_unpaired_msa_gz(seq, msa_dir)

        result = CliRunner().invoke(
            materialize_dataset_cache,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        return msa_dir

    def test_writes_schema_valid_config(self, cif_protein_monomer, ccd_path, tmp_path):
        """The generated config validates against OF3's TrainingExperimentConfig schema."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((output_dir / "train.yml").read_text())

        # No validation dataset is configured.
        assert "validation" not in config["dataset_configs"]

        # Alignments come from the reorganized OF3 MSA directory.
        actual_alignments_dir = config["dataset_paths"]["weighted-pdb"]["alignments_directory"]
        assert actual_alignments_dir == str(msa_dir / "of3_alignments")

        # Training output goes under the pipeline output directory.
        assert config["experiment_settings"]["output_dir"] == str(output_dir / "training_output")

        # The whole config validates against OF3's schema (must not raise).
        TrainingExperimentConfig(**config)

    def test_sample_weights_make_every_datapoint_equal(self, cif_protein_monomer, ccd_path, tmp_path):
        """Flat sample_weights: with cluster_size = datapoints per structure, OF3
        samples each structure equally often, and the schema accepts the weights."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((output_dir / "train.yml").read_text())
        expected_weights = {
            "a_prot": 1.0,
            "a_nuc": 1.0,
            "a_ligand": 1.0,
            "w_chain": 1.0,
            "w_interface": 0.5,
        }
        actual_weights = config["dataset_configs"]["train"]["weighted-pdb"]["config"][
            "sample_weights"
        ]
        assert actual_weights == expected_weights

        # OF3's schema accepts the weights.
        TrainingExperimentConfig(**config)

    def test_legacy_dataset_cache_warns(self, cif_protein_monomer, ccd_path, tmp_path):
        """A cache built before interface keys (2+ chains, no interfaces, all
        cluster_size 1) gets a rebuild warning, but the config is still written."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)
        cache_path = output_dir / "dataset_cache.json"
        cache = json.loads(cache_path.read_text())
        for entry in cache["structure_data"].values():
            entry["interfaces"] = {}
            for chain in entry["chains"].values():
                chain["cluster_size"] = 1
        cache_path.write_text(json.dumps(cache))

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        expected_text = "built before interface keys"
        assert result.exit_code == 0, result.output
        assert expected_text in result.output
        assert "rebuild from finetune-prep" in result.output
        assert (output_dir / "train.yml").exists()

    def test_current_dataset_cache_does_not_warn(self, cif_protein_monomer, ccd_path, tmp_path):
        """A freshly built cache prints no legacy-cache warning."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "built before interface keys" not in result.output

    def test_wandb_off_by_default(self, cif_protein_monomer, ccd_path, tmp_path):
        """Without --wandb-project, wandb_config is None."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)
        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        config = yaml.safe_load((output_dir / "train.yml").read_text())

        # wandb logging is not configured.
        assert config["logging_config"]["wandb_config"] is None

    def test_wandb_config_emitted_when_requested(self, cif_protein_monomer, ccd_path, tmp_path):
        """--wandb-project/--wandb-entity set logging_config.wandb_config, with
        experiment_name defaulting to the output dir name; still schema-valid."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)

        result = CliRunner().invoke(
            generate_train_config,
            [
                "--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir),
                "--wandb-project", "atomworks-ft", "--wandb-entity", "jennifer-wei",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((output_dir / "train.yml").read_text())

        # wandb config carries project, entity, and the output dir name as experiment name.
        expected_wandb_config = {
            "project": "atomworks-ft",
            "entity": "jennifer-wei",
            "experiment_name": output_dir.name,
        }
        assert config["logging_config"]["wandb_config"] == expected_wandb_config

        # The config is still schema-valid (must not raise).
        TrainingExperimentConfig(**config)

    def test_wandb_experiment_name_override(self, cif_protein_monomer, ccd_path, tmp_path):
        """--wandb-name overrides experiment_name; entity is omitted when not given."""
        output_dir = self._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)
        msa_dir = self._make_msa_dir(cif_protein_monomer, output_dir, tmp_path)
        result = CliRunner().invoke(
            generate_train_config,
            [
                "--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir),
                "--wandb-project", "atomworks-ft", "--wandb-name", "demo-run",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        config = yaml.safe_load((output_dir / "train.yml").read_text())
        actual_wandb_config = config["logging_config"]["wandb_config"]

        # --wandb-name sets the experiment name.
        assert actual_wandb_config["experiment_name"] == "demo-run"

        # No entity was given, so none is written.
        assert "entity" not in actual_wandb_config

    # -- validation section (split/) -------------------------------------------------

    @pytest.fixture
    def split_copy(self, built_dataset, tmp_path):
        return mutable_copy(built_dataset, tmp_path / "split_copy")

    def _split(self, built, tmp_path, ids_to_val=("7riv", "3u8v")):
        csv = tmp_path / "split.csv"
        csv.write_text("id,split\n" + "".join(f"{i},val\n" for i in ids_to_val))
        split_dataset_cache(built.output_dir, csv)
        return split_paths(built.output_dir)

    def test_train_debug_mode_is_false(self, built_dataset):
        config = build_train_config(built_dataset.output_dir, built_dataset.msa_dir)

        actual_debug_mode = config["dataset_configs"]["train"]["weighted-pdb"]["config"]["debug_mode"]
        assert actual_debug_mode is False

    def test_train_msa_features_are_protein_only(self, built_dataset):
        """ColabFold only gives protein MSAs, so RNA is featurized without one, like DNA."""
        config = build_train_config(built_dataset.output_dir, built_dataset.msa_dir)

        actual_moltypes = config["dataset_configs"]["train"]["weighted-pdb"]["config"]["msa"]["moltypes"]
        assert actual_moltypes == ["PROTEIN"]

    def test_no_split_means_train_only(self, built_dataset):
        config = build_train_config(built_dataset.output_dir, built_dataset.msa_dir)

        expected_cache = str(built_dataset.output_dir / "dataset_cache.json")
        actual_cache = config["dataset_paths"]["weighted-pdb"]["dataset_cache_file"]
        assert "validation" not in config["dataset_configs"]
        assert actual_cache == expected_cache

    def test_current_split_adds_validation(self, split_copy, tmp_path):
        paths = self._split(split_copy, tmp_path)

        config = build_train_config(split_copy.output_dir, split_copy.msa_dir)

        expected_val = {
            "dataset_class": "ValidationPDBDataset",
            "config": {
                "debug_mode": False,
                "msa": {"subsample_main": False, "moltypes": ["PROTEIN"]},
                "template": {"n_templates": 4, "take_top_k": True},
                "crop": {"token_crop": {"enabled": False}},
            },
        }
        actual_val = config["dataset_configs"]["validation"]["val-weighted-pdb"]
        assert actual_val == expected_val
        actual_train_cache = config["dataset_paths"]["weighted-pdb"]["dataset_cache_file"]
        assert actual_train_cache == str(paths.train_cache)
        expected_val_paths = {
            **config["dataset_paths"]["weighted-pdb"],
            "dataset_cache_file": str(paths.val_cache),
        }
        assert config["dataset_paths"]["val-weighted-pdb"] == expected_val_paths
        TrainingExperimentConfig(**config)

    def test_stale_split_is_an_error(self, split_copy, tmp_path):
        self._split(split_copy, tmp_path)
        cache_path = split_copy.output_dir / "dataset_cache.json"
        cache_path.write_text(cache_path.read_text() + " ")

        with pytest.raises(StaleSplitError) as error:
            build_train_config(split_copy.output_dir, split_copy.msa_dir)

        assert "split-dataset" in error.value.message
        assert "delete" in error.value.message

        # Rerunning split-dataset makes the split current again.
        split_dataset_cache(split_copy.output_dir, split_paths(split_copy.output_dir).csv)
        config = build_train_config(split_copy.output_dir, split_copy.msa_dir)
        assert "validation" in config["dataset_configs"]

    def test_split_without_source_is_stale(self, split_copy, tmp_path):
        paths = self._split(split_copy, tmp_path)
        paths.source.unlink()

        with pytest.raises(StaleSplitError):
            build_train_config(split_copy.output_dir, split_copy.msa_dir)

    def test_empty_train_cache_is_an_error(self, split_copy, tmp_path):
        all_ids = ("1a3n", "2crb", "2q2k", "3u8v", "7riv")
        paths = self._split(split_copy, tmp_path, ids_to_val=all_ids)

        with pytest.raises(click.ClickException) as error:
            build_train_config(split_copy.output_dir, split_copy.msa_dir)

        assert str(paths.train_cache) in error.value.message
        assert "every structure is val or test" in error.value.message

    def test_corrupt_source_is_stale(self, split_copy, tmp_path):
        paths = self._split(split_copy, tmp_path)
        paths.source.write_text("{not json")

        with pytest.raises(StaleSplitError):
            build_train_config(split_copy.output_dir, split_copy.msa_dir)


class TestFinetunePrepSkipsEmptied:
    """Structures with no atoms left after cleanup are skipped and reported, exit 0."""

    def _run(self, cif_dir: Path, out: Path) -> object:
        """Invoke finetune-prep on ``cif_dir``.

        Args:
            cif_dir: Input CIF directory.
            out: Output directory.

        Returns:
            The CliRunner result.
        """
        return CliRunner().invoke(
            finetune_prep,
            ["--cif-dir", str(cif_dir), "--output-dir", str(out)],
            catch_exceptions=False,
        )

    def test_emptied_structure_skipped(self, cif_protein_monomer, tmp_path):
        """A 3-residue structure emptied by cleanup is skipped and reported; the other is kept."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        write_emptiable_minimal_3aa(cif_dir)
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)

        result = self._run(cif_dir, tmp_path / "out")

        assert result.exit_code == 0, result.output

        # Only the surviving structure is in the manifest and has an augmented CIF.
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["structures"] == ["3u8v"]
        assert not (tmp_path / "out" / "cleaned_structures" / "minimal_3aa.cif").exists()

        # The skip is reported per structure and in the final summary.
        assert "WARNING: skipping minimal_3aa -- empty after cleanup (remove_small_polymers)" in result.output
        assert "Dropped 1 structure(s), reasons in " in result.output
        assert result.output.rstrip().endswith("prep_dropped.json: minimal_3aa")

    def test_emptied_structure_removes_stale_cif(self, tmp_path):
        """A leftover augmented CIF from an earlier run is deleted when its structure is now emptied."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        write_emptiable_minimal_3aa(cif_dir)
        stale = tmp_path / "out" / "cleaned_structures" / "minimal_3aa.cif"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale from an earlier run")

        result = self._run(cif_dir, tmp_path / "out")

        assert result.exit_code == 0, result.output

        # The stale CIF is gone and the structure is not in the manifest.
        assert not stale.exists()
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert "minimal_3aa" not in manifest["structures"]

    def test_all_emptied_still_exits_zero(self, tmp_path):
        """If every structure is emptied by cleanup, the command still succeeds with an empty manifest."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        write_emptiable_minimal_3aa(cif_dir)

        result = self._run(cif_dir, tmp_path / "out")

        assert result.exit_code == 0, result.output

        # The manifest lists no structures.
        manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
        assert manifest["structures"] == []


# ── TestOrganizedMsaDirOption (--organized-msa-dir and its default) ──────────


def _train_alignments_dir(output_dir: Path) -> str:
    """Return the train dataset's alignments_directory from <output_dir>/train.yml."""
    config = yaml.safe_load((output_dir / "train.yml").read_text())
    return config["dataset_paths"]["weighted-pdb"]["alignments_directory"]


class TestOrganizedMsaDirOption:
    @pytest.fixture
    def pipeline_output(self, cif_protein_monomer, ccd_path, tmp_path) -> Path:
        """A full 3u8v pipeline output directory, built without MSAs."""
        return TestGenerateTrainConfig()._make_pipeline_output(cif_protein_monomer, ccd_path, tmp_path)

    @staticmethod
    def _organize(cif_protein_monomer, output_dir: Path, msa_dir: Path) -> None:
        """Put a ColabFold-style MSA under msa_dir and reorganize it into of3_alignments/."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _make_unpaired_msa_gz(seq, msa_dir)
        result = CliRunner().invoke(
            materialize_dataset_cache,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    def test_new_flag_accepted(self, cif_protein_monomer, pipeline_output, tmp_path):
        """generate-train-config --organized-msa-dir X points train.yml at X/of3_alignments."""
        msa_dir = tmp_path / "msas"
        self._organize(cif_protein_monomer, pipeline_output, msa_dir)

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(pipeline_output), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert _train_alignments_dir(pipeline_output) == str(msa_dir / "of3_alignments")

    def test_old_flag_warns_and_works(self, cif_protein_monomer, pipeline_output, tmp_path):
        """The old --msa-dir still works, with a deprecation warning on stderr."""
        msa_dir = tmp_path / "msas"
        self._organize(cif_protein_monomer, pipeline_output, msa_dir)

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(pipeline_output), "--msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert "deprecated" in result.stderr
        assert _train_alignments_dir(pipeline_output) == str(msa_dir / "of3_alignments")

    def test_both_flags_error(self, cif_protein_monomer, pipeline_output, tmp_path):
        """Passing both the new flag and the deprecated alias is a usage error."""
        msa_dir = tmp_path / "msas"
        self._organize(cif_protein_monomer, pipeline_output, msa_dir)

        result = CliRunner().invoke(
            generate_train_config,
            ["--output-dir", str(pipeline_output), "--organized-msa-dir", str(msa_dir),
             "--msa-dir", str(msa_dir)],
        )

        assert result.exit_code == 2
        assert "--organized-msa-dir" in result.output

    def test_generate_train_config_defaults_to_output_organized_msas(
        self, cif_protein_monomer, pipeline_output
    ):
        """No flag: the MSA folder is <output-dir>/organized_msas."""
        self._organize(cif_protein_monomer, pipeline_output, pipeline_output / "organized_msas")

        result = CliRunner().invoke(
            generate_train_config, ["--output-dir", str(pipeline_output)], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert _train_alignments_dir(pipeline_output) == str(
            pipeline_output / "organized_msas" / "of3_alignments"
        )

    def test_generate_train_config_without_any_msa_folder_errors(self, pipeline_output):
        """No flag and no <output-dir>/organized_msas: fail, naming the flag and the default."""
        result = CliRunner().invoke(generate_train_config, ["--output-dir", str(pipeline_output)])

        assert result.exit_code != 0
        assert "--organized-msa-dir" in result.output
        assert "organized_msas" in result.output
        assert not (pipeline_output / "train.yml").exists()

    def test_optional_commands_ignore_missing_default(self, pipeline_output):
        """materialize-dataset-cache with no flag and no organized_msas/ skips the MSA step."""
        result = CliRunner().invoke(
            materialize_dataset_cache, ["--output-dir", str(pipeline_output)], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "Reorganized" not in result.output

    def test_run_all_without_flags_never_calls_colabfold(
        self, cif_protein_monomer, ccd_path, tmp_path, monkeypatch
    ):
        """run-all with no MSA flags skips the MSA stage and never contacts ColabFold."""
        monkeypatch.setattr(
            cli_mod, "make_msas_colabfold_server_batch",
            lambda *a, **k: pytest.fail("ColabFold must not be called"),
        )
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())

        result = CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir), "--output-dir", str(tmp_path / "out"),
            "--ccd-path", str(ccd_path), "--skip-report",
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert "Stage 2/10: generate-msas -- skipped" in result.output

    def test_run_all_reuses_existing_default_folder(
        self, cif_protein_monomer, ccd_path, tmp_path, monkeypatch
    ):
        """A rerun over an output dir that already has organized_msas/ reuses it without ColabFold."""
        monkeypatch.setattr(
            cli_mod, "make_msas_colabfold_server_batch",
            lambda *a, **k: pytest.fail("ColabFold must not be called"),
        )
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        output_dir = tmp_path / "out"
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _make_unpaired_msa_gz(seq, output_dir / "organized_msas")

        result = CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir), "--output-dir", str(output_dir),
            "--ccd-path", str(ccd_path), "--skip-report",
        ], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert "Stage 2/10: generate-msas -- skipped" in result.output
        assert (output_dir / "organized_msas" / "of3_alignments" / alignment_rep_id(seq)
                / "colabfold_main.a3m").exists()
        assert _train_alignments_dir(output_dir) == str(output_dir / "organized_msas" / "of3_alignments")



# ── TestFinetunePipelineMsaSources (run-all with user MSAs) ──────────────────


class TestFinetunePipelineMsaSources:
    @pytest.fixture
    def no_colabfold(self, monkeypatch):
        monkeypatch.setattr(
            cli_mod, "make_msas_colabfold_server_batch",
            lambda *a, **k: pytest.fail("ColabFold must not be called"),
        )

    @staticmethod
    def _inputs(cif_protein_monomer, tmp_path) -> tuple[Path, Path, str]:
        """(cif_dir, manifest, sequence): 3u8v plus a user MSA for its chain under source my_lab_db."""
        cif_dir = tmp_path / "input"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer.resolve())
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        user = tmp_path / "user_msas"
        user.mkdir()
        (user / "chain_a.a3m").write_text(f">query\n{seq}\n>hit\nW{seq[1:]}\n")
        manifest = tmp_path / "msa_sources.yml"
        manifest.write_text("my_lab_db: user_msas\n")
        return cif_dir, manifest, seq

    @staticmethod
    def _run(cif_dir, output_dir, ccd_path, *extra):
        return CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path),
            "--skip-report", "--skip-dataset-load", *extra,
        ], catch_exceptions=False)

    def test_msa_sources_runs_import_not_colabfold(self, cif_protein_monomer, ccd_path, tmp_path, no_colabfold):
        cif_dir, manifest, seq = self._inputs(cif_protein_monomer, tmp_path)
        output_dir, msa_dir = tmp_path / "out", tmp_path / "msas"

        result = self._run(cif_dir, output_dir, ccd_path, "--organized-msa-dir", str(msa_dir), "--msa-sources", str(manifest))

        assert result.exit_code == 0, result.output
        assert "Stage 2/10: import-msas" in result.output
        assert "templates with --msa-sources are not supported yet" in result.output
        assert (msa_dir / "of3_alignments" / alignment_rep_id(seq) / "my_lab_db.a3m").exists()
        config = yaml.safe_load((output_dir / "train.yml").read_text())
        assert "my_lab_db" in config["dataset_configs"]["train"]["weighted-pdb"]["config"]["msa"]["max_seq_counts"]

    def test_msa_sources_uses_default_folder(self, cif_protein_monomer, ccd_path, tmp_path, no_colabfold):
        cif_dir, manifest, seq = self._inputs(cif_protein_monomer, tmp_path)
        output_dir = tmp_path / "out"

        result = self._run(cif_dir, output_dir, ccd_path, "--msa-sources", str(manifest))

        assert result.exit_code == 0, result.output
        assert (output_dir / "organized_msas" / "of3_alignments" / alignment_rep_id(seq) / "my_lab_db.a3m").exists()
        assert (output_dir / "train.yml").exists()

    @pytest.mark.parametrize("flag", [
        ["--templates"], ["--server", "https://example.org"], ["--user-agent", "me"], ["--skip-existing"],
    ])
    def test_msa_sources_rejects_colabfold_only_flags(self, cif_protein_monomer, ccd_path, tmp_path, flag):
        """Flags that only affect ColabFold or templates would silently do nothing: refused before stage 1."""
        cif_dir, manifest, _ = self._inputs(cif_protein_monomer, tmp_path)
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(finetune_pipeline, [
            "--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path),
            "--msa-sources", str(manifest), *flag,
        ])

        assert result.exit_code == 2
        assert flag[0] in result.output
        assert not (output_dir / "manifest.json").exists()

    def test_msa_sources_accepts_explicit_no_templates(self, cif_protein_monomer, ccd_path, tmp_path, no_colabfold):
        cif_dir, manifest, _ = self._inputs(cif_protein_monomer, tmp_path)

        result = self._run(cif_dir, tmp_path / "out", ccd_path, "--msa-sources", str(manifest), "--no-templates")

        assert result.exit_code == 0, result.output
