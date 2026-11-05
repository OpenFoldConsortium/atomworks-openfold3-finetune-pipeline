"""Template discovery via pre-built OF3 template cache files.

Test structure:
- TestAlignmentRepsFasta: write_alignment_reps_fasta writes valid, deduplicated FASTA.
- TestFindTemplateIds: find_template_ids reads cache .npz files and returns sorted IDs.
- TestTemplateIdsRoundtripViaCIF: cache-derived template_ids survive the augmented CIF.

ColabFold .m8 fetching is covered in test_template_fetch.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from assertions import assert_caches_equal
from aw_of3_finetune.augmented_cif import read_augmented_cif, write_augmented_cif
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences
from aw_of3_finetune.parsing import parse_structure
from aw_of3_finetune.templates import find_template_ids, write_alignment_reps_fasta
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetStructureData,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_template_cache_npz(
    rep_id: str,
    template_ids: list[str],
    cache_dir: Path,
) -> Path:
    """Write a minimal template cache .npz in OF3 TemplatePreprocessor format.

    Keys are template IDs; values hold 'index', 'release_date' and 'idx_map'.

    Args:
        rep_id: Alignment representative ID (file stem).
        template_ids: Template IDs, in index order.
        cache_dir: Cache directory; created if missing.

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


def _parse_aw(cif_path: Path):
    """Parse a CIF with atomworks.

    Args:
        cif_path: Input CIF.

    Returns:
        The first model of the asymmetric unit as an AtomArray.
    """
    return parse_structure(cif_path)["asym_unit"][0]


# ── TestAlignmentRepsFasta ────────────────────────────────────────────────────


class TestAlignmentRepsFasta:
    def test_basic_write(self, cif_protein_monomer, tmp_path):
        """write_alignment_reps_fasta creates a FASTA file with 11-char rep_id headers."""
        out = tmp_path / "reps.fasta"
        rep_map = write_alignment_reps_fasta([cif_protein_monomer], out)
        # The file was written and holds at least one entry.
        assert out.exists()
        assert len(rep_map) > 0

        # Every header is an 11-character rep ID and every sequence is non-empty.
        actual_rep_id_lengths = {len(rep_id) for rep_id in rep_map}
        assert actual_rep_id_lengths == {11}
        actual_empty_sequences = [rep_id for rep_id, seq in rep_map.items() if len(seq) == 0]
        assert actual_empty_sequences == []

    def test_fasta_format_parseable(self, cif_protein_monomer, tmp_path):
        """The written FASTA is parseable and round-trips rep_id → sequence."""
        out = tmp_path / "reps.fasta"
        rep_map = write_alignment_reps_fasta([cif_protein_monomer], out)
        sequences_in_file: dict[str, str] = {}
        current_id = None
        for line in out.read_text().splitlines():
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences_in_file[current_id] = ""
            elif current_id is not None:
                sequences_in_file[current_id] += line

        # Parsing the file gives back exactly the rep_id -> sequence map that was returned.
        assert sequences_in_file == rep_map

    def test_deduplication_by_hash(self, cif_ubiquitin_1, cif_ubiquitin_2, tmp_path):
        """1UBQ and 1UBI share the same canonical sequence — only one FASTA entry."""
        out = tmp_path / "reps.fasta"
        rep_map = write_alignment_reps_fasta([cif_ubiquitin_1, cif_ubiquitin_2], out)
        seq1 = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        seq2 = chain_canonical_sequences(cif_ubiquitin_2)["A"]

        # The two structures hash to the same rep ID.
        assert alignment_rep_id(seq1) == alignment_rep_id(seq2)

        # So the map and the file each have a single entry.
        assert len(rep_map) == 1
        assert out.read_text().count(">") == 1

    def test_multiple_cifs_accumulated(
        self, cif_protein_monomer, cif_dna_complex, tmp_path
    ):
        """Sequences from different CIFs accumulate without collision."""
        rep_map = write_alignment_reps_fasta(
            [cif_protein_monomer, cif_dna_complex], tmp_path / "combined.fasta"
        )
        map_mono = write_alignment_reps_fasta(
            [cif_protein_monomer], tmp_path / "mono.fasta"
        )
        map_dna = write_alignment_reps_fasta([cif_dna_complex], tmp_path / "dna.fasta")
        # The combined map holds the entries of both single-CIF maps.
        assert set(rep_map.keys()) == set(map_mono) | set(map_dna)

    def test_homodimer_deduplicated(self, cif_protein_monomer, tmp_path):
        """3u8v chains A and B share entity 1 — they produce only one FASTA entry."""
        seqs = chain_canonical_sequences(cif_protein_monomer)
        assert seqs["A"] == seqs["B"], "precondition: 3u8v is a homodimer"
        out = tmp_path / "reps.fasta"
        rep_map = write_alignment_reps_fasta([cif_protein_monomer], out)
        # Both chains collapse to one entry.
        assert len(rep_map) == 1, "homodimer chains must not create duplicate entries"


# ── TestFindTemplateIds ───────────────────────────────────────────────────────


class TestFindTemplateIds:
    def test_returns_sorted_template_ids(self, cif_protein_monomer, tmp_path):
        """find_template_ids returns keys in ascending index (e-value rank) order."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        expected = ["1abc_A", "2xyz_B", "3def_C"]
        _make_template_cache_npz(rep_id, expected, tmp_path)

        # IDs come back in index order.
        assert find_template_ids(rep_id, tmp_path) == expected

    def test_missing_cache_returns_empty(self, cif_protein_monomer, tmp_path):
        """No cache file → empty list (not an error)."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)

        # No cache file gives an empty list.
        assert find_template_ids(rep_id, tmp_path) == []

    def test_empty_npz_returns_empty(self, cif_protein_monomer, tmp_path):
        """Cache npz with no keys → empty list."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        np.savez_compressed(tmp_path / f"{rep_id}.npz")

        # A cache file with no entries gives an empty list.
        assert find_template_ids(rep_id, tmp_path) == []

    def test_template_id_format(self, cif_protein_monomer, tmp_path):
        """Template IDs from the cache have the form '{entry_id}_{chain_id}'."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        _make_template_cache_npz(rep_id, ["1abc_A", "2xyz_B"], tmp_path)
        template_ids = find_template_ids(rep_id, tmp_path)

        # Every ID splits into exactly entry ID and chain ID.
        actual_part_counts = {tid: len(tid.split("_")) for tid in template_ids}
        assert actual_part_counts == {"1abc_A": 2, "2xyz_B": 2}

    def test_shared_cache_for_identical_sequences(
        self, cif_ubiquitin_1, cif_ubiquitin_2, tmp_path
    ):
        """Cache written for 1UBQ's rep_id is found when looking up 1UBI's rep_id."""
        seq1 = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        seq2 = chain_canonical_sequences(cif_ubiquitin_2)["A"]
        rep1 = alignment_rep_id(seq1)
        rep2 = alignment_rep_id(seq2)
        assert rep1 == rep2, "precondition: ubiquitin sequences hash identically"

        template_ids = ["1ubi_A", "1ubq_A"]
        _make_template_cache_npz(rep1, template_ids, tmp_path)

        # Both structures' rep IDs find the same template list.
        assert find_template_ids(rep1, tmp_path) == template_ids
        assert find_template_ids(rep2, tmp_path) == template_ids


# ── TestTemplateIdsRoundtripViaCIF ──────────────────────────────────────────


class TestTemplateIdsRoundtripViaCIF:
    def test_template_ids_survive_cif_roundtrip(self, cif_protein_monomer, tmp_path):
        """template_ids populated from a cache survive write_augmented_cif/read."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        cache_templates = ["1abc_A", "2def_B", "3ghi_C"]
        _make_template_cache_npz(rep_id, cache_templates, tmp_path)

        template_ids = find_template_ids(rep_id, tmp_path)
        chain_a = ClusteredDatasetChainData(
            label_asym_id="A",
            auth_asym_id="A",
            entity_id=1,
            molecule_type="PROTEIN",
            reference_mol_id=None,
            alignment_representative_id=rep_id,
            template_ids=template_ids,
            cluster_id="3u8v_A",
            cluster_size=1,
        )
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={"A": chain_a},
            interfaces={},
        )
        original = ClusteredDatasetCache(
            name="test", structure_data={"3u8v": structure_data}, reference_molecule_data={}
        )
        out = tmp_path / "3u8v_augmented.cif"
        write_augmented_cif(_parse_aw(cif_protein_monomer), structure_data, {}, out, pdb_id="3u8v")
        reconstructed = read_augmented_cif(out, pdb_id="3u8v", cache_name="test")

        # The cache read back from the CIF equals the one written, including template_ids.
        assert_caches_equal(original, reconstructed)

    def test_empty_template_ids_roundtrip(self, cif_protein_monomer, tmp_path):
        """An empty template_ids list (no cache) round-trips correctly."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        rep_id = alignment_rep_id(seq)
        template_ids = find_template_ids(rep_id, tmp_path)  # no cache → []

        chain_a = ClusteredDatasetChainData(
            label_asym_id="A",
            auth_asym_id="A",
            entity_id=1,
            molecule_type="PROTEIN",
            reference_mol_id=None,
            alignment_representative_id=rep_id,
            template_ids=template_ids,
            cluster_id="3u8v_A",
            cluster_size=1,
        )
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={"A": chain_a},
            interfaces={},
        )
        original = ClusteredDatasetCache(
            name="test", structure_data={"3u8v": structure_data}, reference_molecule_data={}
        )
        out = tmp_path / "3u8v_augmented.cif"
        write_augmented_cif(_parse_aw(cif_protein_monomer), structure_data, {}, out, pdb_id="3u8v")
        reconstructed = read_augmented_cif(out, pdb_id="3u8v", cache_name="test")

        # The cache read back from the CIF equals the one written, including template_ids.
        assert_caches_equal(original, reconstructed)
