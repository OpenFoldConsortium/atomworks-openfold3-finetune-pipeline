"""Augmented CIF round-trip.

Each test builds a ClusteredDatasetCache, writes it with write_augmented_cif, reads
it back with read_augmented_cif, and checks the two are bit-identical. Covers
NaN/None resolution, interfaces, reference molecules, and None vs empty for
optional chain fields.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


from assertions import assert_caches_equal
from aw_of3_finetune.augmented_cif import write_augmented_cif, read_augmented_cif
from aw_of3_finetune.parsing import parse_structure

from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    DatasetReferenceMoleculeData,
)


def _parse_aw(cif_path: Path):
    """Parse a CIF with atomworks.

    Args:
        cif_path: Input CIF.

    Returns:
        The first model of the asymmetric unit as an AtomArray.
    """
    return parse_structure(cif_path)["asym_unit"][0]


def _protein_chain(
    label: str = "A",
    cluster_id: str = "c1",
    cluster_size: int = 10,
    template_ids: list[str] | None = None,
    alignment_rep: str | None = "HASH_XYZ",
) -> ClusteredDatasetChainData:
    """Build a protein ClusteredDatasetChainData with overridable cluster/template fields.

    Args:
        label: label_asym_id (also used as auth_asym_id).
        cluster_id: Chain cluster ID.
        cluster_size: Chain cluster size.
        template_ids: Template IDs; None is passed through as None.
        alignment_rep: alignment_representative_id.

    Returns:
        The chain data record.
    """
    return ClusteredDatasetChainData(
        label_asym_id=label,
        auth_asym_id=label,
        entity_id=1,
        molecule_type="PROTEIN",
        reference_mol_id=None,
        alignment_representative_id=alignment_rep,
        template_ids=template_ids if template_ids is not None else [],
        cluster_id=cluster_id,
        cluster_size=cluster_size,
    )


def _roundtrip(
    atom_array,
    structure_data: ClusteredDatasetStructureData,
    reference_molecule_data: dict,
    tmp_path: Path,
    pdb_id: str,
    cache_name: str = "test",
) -> tuple[ClusteredDatasetCache, ClusteredDatasetCache]:
    """Write one structure to an augmented CIF and read it back.

    Args:
        atom_array: Structure to write.
        structure_data: Per-structure cache metadata.
        reference_molecule_data: Reference molecule records keyed by mol ID.
        tmp_path: Output directory.
        pdb_id: PDB ID for the entry.
        cache_name: Cache name for both caches.

    Returns:
        ``(original, reconstructed)`` caches.
    """
    original = ClusteredDatasetCache(
        name=cache_name,
        structure_data={pdb_id: structure_data},
        reference_molecule_data=reference_molecule_data,
    )
    out = tmp_path / f"{pdb_id}_augmented.cif"
    write_augmented_cif(
        atom_array, structure_data, reference_molecule_data, out, pdb_id=pdb_id
    )
    assert out.exists(), f"Expected output file {out} was not created"
    assert out.stat().st_size > 0, "Output CIF is empty"
    reconstructed = read_augmented_cif(out, pdb_id=pdb_id, cache_name=cache_name)
    return original, reconstructed


class TestRoundTrip:
    def test_protein_monomer_basic(self, cif_protein_monomer, tmp_path):
        """Single-chain protein with template_ids and cluster fields."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={"A": _protein_chain(template_ids=["1abc", "2def"])},
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_monomer), structure_data, {}, tmp_path, pdb_id="3u8v"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_nan_resolution(self, cif_protein_monomer, tmp_path):
        """NaN resolution must survive float encoding/decoding."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=float("nan"),
            chains={"A": _protein_chain(template_ids=[])},
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_monomer), structure_data, {}, tmp_path, pdb_id="3u8v"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_none_resolution(self, cif_protein_monomer, tmp_path):
        """None resolution (absent in deposit) must round-trip as None, not NaN."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=None,
            chains={"A": _protein_chain()},
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_monomer), structure_data, {}, tmp_path, pdb_id="3u8v"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_none_template_ids(self, cif_protein_monomer, tmp_path):
        """template_ids=None (ligand/DNA chain) must round-trip as None, not []."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={"A": _protein_chain(template_ids=None, alignment_rep=None)},
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_monomer), structure_data, {}, tmp_path, pdb_id="3u8v"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_none_alignment_rep(self, cif_protein_monomer, tmp_path):
        """alignment_representative_id=None must round-trip as None, not ''."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={"A": _protein_chain(alignment_rep=None, template_ids=[])},
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_monomer), structure_data, {}, tmp_path, pdb_id="3u8v"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_interfaces(self, cif_protein_monomer, tmp_path):
        """Interface cluster data (iface_id, cluster_id, cluster_size) survives."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={
                "A": _protein_chain("A", cluster_id="c1", cluster_size=10),
                "B": ClusteredDatasetChainData(
                    label_asym_id="B",
                    auth_asym_id="B",
                    entity_id=2,
                    molecule_type="PROTEIN",
                    reference_mol_id=None,
                    alignment_representative_id="HASH_B",
                    template_ids=[],
                    cluster_id="c2",
                    cluster_size=5,
                ),
            },
            interfaces={
                "A_B": ClusteredDatasetInterfaceData(cluster_id="i1", cluster_size=3)
            },
        )
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_monomer), structure_data, {}, tmp_path, pdb_id="3u8v"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_reference_molecule(self, cif_protein_only, tmp_path):
        """Reference molecule (SMILES, conformer strategy, set_fallback_to_nan) survives."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(1994, 3, 14),
            resolution=2.0,
            chains={
                "A": _protein_chain("A", alignment_rep="HASH_2CRB"),
                "B": ClusteredDatasetChainData(
                    label_asym_id="B",
                    auth_asym_id="B",
                    entity_id=2,
                    molecule_type="NON-POLYMER",
                    reference_mol_id="LIG",
                    alignment_representative_id=None,
                    template_ids=None,
                    cluster_id="c99",
                    cluster_size=1,
                ),
            },
            interfaces={},
        )
        ref_mol_data = {
            "LIG": DatasetReferenceMoleculeData(
                conformer_gen_strategy="rdkit",
                fallback_conformer_pdb_id=None,
                canonical_smiles="CC(=O)O",
                set_fallback_to_nan=False,
            )
        }
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_only), structure_data, ref_mol_data, tmp_path, pdb_id="2crb"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_ref_mol_set_fallback_true(self, cif_protein_only, tmp_path):
        """set_fallback_to_nan=True must round-trip as True."""
        structure_data = ClusteredDatasetStructureData(
            release_date=date(1994, 3, 14),
            resolution=2.0,
            chains={"A": _protein_chain("A")},
            interfaces={},
        )
        ref_mol_data = {
            "LIG": DatasetReferenceMoleculeData(
                conformer_gen_strategy="rdkit",
                fallback_conformer_pdb_id="1xyz",
                canonical_smiles="c1ccccc1",
                set_fallback_to_nan=True,
            )
        }
        original, reconstructed = _roundtrip(
            _parse_aw(cif_protein_only), structure_data, ref_mol_data, tmp_path, pdb_id="2crb"
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)
