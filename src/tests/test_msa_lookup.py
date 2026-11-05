"""MSA sequence extraction and hash-based alignment rep ID.

Checks CIF -> chain sequence -> alignment_rep_id -> augmented CIF, and that
atomworks' find_msas locates <msa-dir>/unpaired/<shard>/<hash>.a3m.gz by sequence.
OF3 parsing of the reorganized layout is tested in test_pipeline_cli.py.

Test structure
--------------
TestSequenceExtraction       chain_canonical_sequences per molecule type.
TestAlignmentRepId           hash identity/difference across chains.
TestRepIdRoundtripViaCIF     rep ID survives augmented-CIF write/read.
TestMsaPathDiscovery         find_msas finds a hash-sharded .a3m.gz by sequence.
TestCrossStructureMsaLookup  1UBQ/1UBI share one MSA via identical sequences.
TestCollectMsaQueries        collect_msa_queries dedups and skips non-protein.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

from assertions import assert_caches_equal
from atomworks.ml.preprocessing.msa.finding import find_msas
from aw_of3_finetune.augmented_cif import write_augmented_cif, read_augmented_cif
from aw_of3_finetune.msa import (
    alignment_rep_id,
    chain_canonical_sequences,
    collect_complex_queries,
    collect_msa_queries,
)
from aw_of3_finetune.parsing import parse_structure
from aw_of3_finetune.pipeline import prepare_augmented_cif

from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetStructureData,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_aw(cif_path: Path):
    """Parse a CIF with atomworks and return the first asym-unit model.

    Args:
        cif_path: Input mmCIF.

    Returns:
        The atomworks AtomArray for model 0.
    """
    return parse_structure(cif_path)["asym_unit"][0]


MSA_LOOKUP_EXT = ".a3m.gz"


def _msa_lookup_dir(alignments_directory: Path) -> dict:
    """Build a find_msas lookup dict for the default unpaired layout (2-char shard, .a3m.gz).

    Args:
        alignments_directory: MSA root to search.

    Returns:
        A find_msas directory spec.
    """
    return {"dir": str(alignments_directory), "extension": MSA_LOOKUP_EXT, "directory_depth": 1}


def _make_minimal_msa_a3m_gz(seq: str, alignments_directory: Path) -> Path:
    """Write a query-only .a3m.gz in the ``<shard>/<hash>.a3m.gz`` layout find_msas reads.

    Args:
        seq: Query sequence.
        alignments_directory: MSA root.

    Returns:
        Path of the written file.
    """
    rep = alignment_rep_id(seq)
    shard_dir = alignments_directory / rep[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"{rep}{MSA_LOOKUP_EXT}"
    with gzip.open(out, "wt") as f:
        f.write(f">query\n{seq}\n")
    return out


# ── Sequence extraction ───────────────────────────────────────────────────────


class TestSequenceExtraction:
    def test_protein_homodimer(self, cif_protein_monomer):
        """3u8v chains A and B share entity 1; both should return the same sequence."""
        seqs = chain_canonical_sequences(cif_protein_monomer)

        # Both homodimer chains are present.
        assert {"A", "B"} <= set(seqs)

        # They share one non-empty sequence.
        assert seqs["A"] == seqs["B"], "homodimer chains must share the same sequence"
        assert len(seqs["A"]) > 0

    def test_non_polymer_chains_excluded(self, cif_protein_monomer):
        """3u8v chains C (NI ion), D, E (water) must not appear in the output."""
        seqs = chain_canonical_sequences(cif_protein_monomer)

        # Only polymer chains get a sequence.
        non_polymer_chains_present = {"C", "D"} & set(seqs)
        assert non_polymer_chains_present == set()

    def test_dna_protein_complex(self, cif_dna_complex):
        """2q2k: protein chains B+C present; DNA chain A present (it has entity_poly entry)."""
        seqs = chain_canonical_sequences(cif_dna_complex)
        # The protein homodimer chains are present and share one sequence.
        assert "B" in seqs and "C" in seqs
        assert seqs["B"] == seqs["C"], "homodimer protein chains should share the sequence"

        # DNA chain A is a polymer so it has an entity_poly entry too.
        assert "A" in seqs

        # Ligand chain D (EPE) is non-polymer, so it is absent.
        assert "D" not in seqs

    def test_protein_only_excludes_nucleic_acids(self, cif_dna_complex):
        """protein_only=True keeps 2q2k's protein chains B+C and drops DNA chain A."""
        # ColabFold treats nucleic acids as protein (U -> X); a paired query with one hangs.
        seqs = chain_canonical_sequences(cif_dna_complex, protein_only=True)

        # Only the two protein chains remain.
        assert set(seqs) == {"B", "C"}


# ── Hash properties ───────────────────────────────────────────────────────────


class TestAlignmentRepId:
    def test_different_sequences_give_different_hashes(self, cif_dna_complex):
        """Protein chain B and DNA chain A have different sequences → different hashes."""
        seqs = chain_canonical_sequences(cif_dna_complex)

        # The protein and DNA chains hash differently.
        assert alignment_rep_id(seqs["B"]) != alignment_rep_id(seqs["A"])

    def test_homodimer_chains_same_hash(self, cif_dna_complex):
        """Chains B and C of 2q2k are the same entity — they must hash identically."""
        seqs = chain_canonical_sequences(cif_dna_complex)

        # The two homodimer chains hash identically.
        assert alignment_rep_id(seqs["B"]) == alignment_rep_id(seqs["C"])


# ── Rep ID round-trip through augmented CIF ─────────────────────────────────


class TestRepIdRoundtripViaCIF:
    def test_protein_rep_id_survives(self, cif_protein_monomer, tmp_path):
        """alignment_representative_id set to sequence hash survives CIF write/read."""
        seqs = chain_canonical_sequences(cif_protein_monomer)
        rep_id = alignment_rep_id(seqs["A"])

        chain_a = ClusteredDatasetChainData(
            label_asym_id="A",
            auth_asym_id="A",
            entity_id=1,
            molecule_type="PROTEIN",
            reference_mol_id=None,
            alignment_representative_id=rep_id,
            template_ids=[],
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

        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

        # The representative ID is the sequence hash, unchanged.
        actual_rep_id = (
            reconstructed.structure_data["3u8v"].chains["A"].alignment_representative_id
        )
        assert actual_rep_id == rep_id, f"rep_id changed: {rep_id!r} → {actual_rep_id!r}"


# ── MSA path discovery ────────────────────────────────────────────────────────


class TestMsaPathDiscovery:
    def test_find_msa_by_hash(self, cif_protein_monomer, tmp_path):
        """A hash-sharded .a3m.gz file is found via atomworks' find_msas."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        expected_path = _make_minimal_msa_a3m_gz(seq, alignments_directory=tmp_path)

        missing, found = find_msas([seq], [_msa_lookup_dir(tmp_path)])
        # The sequence is found, at the hash-sharded path we wrote.
        assert not missing, "MSA file not found by hash"
        assert found[seq] == expected_path

    def test_missing_sequence_returns_none(self, cif_protein_monomer, tmp_path):
        """A sequence with no matching file is reported missing, not found."""
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        different_seq = seq[:-1] + ("A" if seq[-1] != "A" else "G")

        _make_minimal_msa_a3m_gz(seq, alignments_directory=tmp_path)

        missing, found = find_msas([different_seq], [_msa_lookup_dir(tmp_path)])
        # The altered sequence is reported missing and not found.
        assert missing == [different_seq]
        assert different_seq not in found


# ── Cross-structure MSA sharing ───────────────────────────────────────────────


class TestCrossStructureMsaLookup:
    """1UBQ and 1UBI: distinct ubiquitin structures with identical chain A sequences.

    One MSA file, keyed by the shared hash, must serve both.
    """

    def test_canonical_sequences_are_equal(self, cif_ubiquitin_1, cif_ubiquitin_2):
        """chain A sequences from 1UBQ and 1UBI must be byte-for-byte identical."""
        seq1 = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        seq2 = chain_canonical_sequences(cif_ubiquitin_2)["A"]
        # Same sequence in both structures, and it is the 76-residue ubiquitin.
        assert seq1 == seq2, f"Sequences differ:\n  1UBQ: {seq1}\n  1UBI: {seq2}"
        assert len(seq1) == 76

    def test_rep_ids_are_equal(self, cif_ubiquitin_1, cif_ubiquitin_2):
        """alignment_rep_id must return the same hash for identical sequences."""
        seq1 = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        seq2 = chain_canonical_sequences(cif_ubiquitin_2)["A"]

        # Identical sequences hash to the same rep ID.
        assert alignment_rep_id(seq1) == alignment_rep_id(seq2)

    def test_single_msa_covers_both_structures(
        self, cif_ubiquitin_1, cif_ubiquitin_2, tmp_path
    ):
        """An MSA written for 1UBQ chain A is found when looking up 1UBI chain A."""
        seq1 = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        seq2 = chain_canonical_sequences(cif_ubiquitin_2)["A"]

        # Write MSA only for the 1UBQ sequence
        expected_path = _make_minimal_msa_a3m_gz(seq1, alignments_directory=tmp_path)

        # Lookup using the 1UBI sequence — must find the same file
        missing, found = find_msas([seq2], [_msa_lookup_dir(tmp_path)])
        # The 1UBI lookup resolves to the file written for 1UBQ.
        assert not missing, "MSA not found for 1UBI using 1UBQ-keyed file"
        assert found[seq2] == expected_path

    def test_msa_file_is_named_by_shared_hash(
        self, cif_ubiquitin_1, cif_ubiquitin_2, tmp_path
    ):
        """The MSA file name is the shared alignment_rep_id, not the PDB ID."""
        seq1 = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        seq2 = chain_canonical_sequences(cif_ubiquitin_2)["A"]
        shared_hash = alignment_rep_id(seq1)

        _make_minimal_msa_a3m_gz(seq1, alignments_directory=tmp_path)

        # The file is named by the hash, not "1UBQ_A" or "1UBI_A".
        assert (tmp_path / shared_hash[:2] / f"{shared_hash}{MSA_LOOKUP_EXT}").is_file()
        assert not (tmp_path / "1UBQ_A").exists()
        assert not (tmp_path / "1UBI_A").exists()

        # Both structures' lookups resolve to the same file.
        _, found1 = find_msas([seq1], [_msa_lookup_dir(tmp_path)])
        _, found2 = find_msas([seq2], [_msa_lookup_dir(tmp_path)])
        assert found1[seq1] == found2[seq2]


# ── collect_msa_queries ───────────────────────────────────────────────────────


def _make_output_dir(tmp_path: Path, pdb_ids: list[str], cif_sources: list[Path]) -> Path:
    """Build a minimal finetune-prep output_dir (manifest.json + structures/ symlinks).

    Args:
        tmp_path: Directory to populate.
        pdb_ids: Structure IDs to list in the manifest.
        cif_sources: Source CIFs, parallel to ``pdb_ids``.

    Returns:
        ``tmp_path``.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps({"name": "test", "structures": pdb_ids})
    )
    structures = tmp_path / "structures"
    structures.mkdir()
    for pdb_id, src in zip(pdb_ids, cif_sources):
        (structures / f"{pdb_id}.cif").symlink_to(src.resolve())
    return tmp_path


class TestCollectMsaQueries:
    def test_single_structure_returns_protein_chains(self, cif_protein_monomer, tmp_path):
        """collect_msa_queries returns one rep_id per unique sequence."""
        output_dir = _make_output_dir(tmp_path, ["3u8v"], [cif_protein_monomer])
        queries = collect_msa_queries(output_dir)

        # 3u8v is a homodimer: chains A and B share one sequence, so there is one query,
        # keyed by that sequence's rep ID. 3u8v has no other polymer.
        expected_seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        expected_queries = {alignment_rep_id(expected_seq): expected_seq}
        assert queries == expected_queries

    def test_nucleic_acid_chains_are_not_queried(self, cif_dna_complex, tmp_path):
        """2q2k: only the protein sequence is an MSA query, not the DNA chain."""
        # A DNA query is run as protein (U -> X); paired with the protein it hangs ColabFold.
        output_dir = _make_output_dir(tmp_path, ["2q2k"], [cif_dna_complex])
        queries = collect_msa_queries(output_dir)

        # The single query is the protein sequence.
        protein_seq = chain_canonical_sequences(cif_dna_complex)["B"]
        assert queries == {alignment_rep_id(protein_seq): protein_seq}

    def test_nucleic_acid_chains_not_queried_from_augmented_cif(self, cif_dna_complex, tmp_path):
        """Same, from the augmented CIF finetune-prep writes (its entity_poly must
        distinguish protein from DNA)."""
        structures = tmp_path / "structures"
        structures.mkdir()
        prepare_augmented_cif(cif_dna_complex, "2q2k", structures / "2q2k.cif")
        (tmp_path / "manifest.json").write_text(json.dumps({"name": "t", "structures": ["2q2k"]}))

        queries = collect_msa_queries(tmp_path)

        # The single query is the protein sequence.
        protein_seq = chain_canonical_sequences(cif_dna_complex)["B"]
        assert queries == {alignment_rep_id(protein_seq): protein_seq}

        # With one unique protein there is nothing to pair.
        assert collect_complex_queries(tmp_path) == {}, "one unique protein -> nothing to pair"

    def test_deduplication_across_structures(self, cif_ubiquitin_1, cif_ubiquitin_2, tmp_path):
        """Identical sequences from different structures produce a single rep_id entry."""
        output_dir = _make_output_dir(
            tmp_path, ["1ubq", "1ubi"], [cif_ubiquitin_1, cif_ubiquitin_2]
        )
        queries = collect_msa_queries(output_dir)

        # 1UBQ and 1UBI both have the 76-aa ubiquitin sequence, so there is one entry.
        seq = chain_canonical_sequences(cif_ubiquitin_1)["A"]
        assert queries == {alignment_rep_id(seq): seq}

    def test_missing_cif_is_skipped(self, tmp_path):
        """A structure listed in the manifest but with no CIF is skipped silently."""
        (tmp_path / "manifest.json").write_text(
            json.dumps({"name": "test", "structures": ["ghost"]})
        )
        (tmp_path / "structures").mkdir()
        # No CIF written for "ghost"

        queries = collect_msa_queries(tmp_path)

        # No queries, and no error.
        assert queries == {}

    def test_empty_manifest(self, tmp_path):
        """An empty structures list returns an empty dict."""
        (tmp_path / "manifest.json").write_text(
            json.dumps({"name": "test", "structures": []})
        )
        (tmp_path / "structures").mkdir()

        # No structures means no queries.
        assert collect_msa_queries(tmp_path) == {}
