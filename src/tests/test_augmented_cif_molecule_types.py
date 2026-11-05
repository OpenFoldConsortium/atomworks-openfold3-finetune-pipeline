"""Augmented CIF round-trips across all four molecule types.

Chain fields split into tier 1 (CIF-derived: label/auth_asym_id, entity_id,
molecule_type, reference_mol_id) and tier 2 (pipeline annotations: alignment rep,
template_ids, cluster fields), which the augmented CIF must round-trip exactly.

Test structure:
- TestMoleculeTypes: per-type tier-2 null patterns on a synthetic carrier CIF.
- TestDNAComplex: 2q2k with tier-1 data parsed from the CIF at test time.
- TestFromJSON: cache JSON -> augmented CIF -> compare.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from assertions import assert_caches_equal
from aw_of3_finetune.augmented_cif import write_augmented_cif, read_augmented_cif
from aw_of3_finetune.cif_reader import parse_cif_chain_data
from aw_of3_finetune.constants import DUMMY_CLUSTER_ID, DUMMY_CLUSTER_SIZE
from aw_of3_finetune.parsing import parse_structure

from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    DatasetReferenceMoleculeData,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_aw(cif_path: Path):
    """Parse a CIF with atomworks.

    Args:
        cif_path: Input CIF.

    Returns:
        The first model of the asymmetric unit as an AtomArray.
    """
    return parse_structure(cif_path)["asym_unit"][0]


def _annotated_chain(
    label_asym_id: str,
    cif_fields: dict,  # tier-1: auth_asym_id, entity_id, molecule_type, reference_mol_id
    cluster_id: str,  # tier-2: dataset annotation fields
    cluster_size: int,
    alignment_rep: str | None = None,
    template_ids: list[str] | None = None,
) -> ClusteredDatasetChainData:
    """Construct a chain by merging CIF-derived data with annotation fields.

    Args:
        label_asym_id: Chain label.
        cif_fields: Tier-1 fields (auth_asym_id, entity_id, molecule_type, reference_mol_id).
        cluster_id: Chain cluster ID.
        cluster_size: Chain cluster size.
        alignment_rep: alignment_representative_id.
        template_ids: Template IDs.

    Returns:
        The chain data record.
    """
    return ClusteredDatasetChainData(
        label_asym_id=label_asym_id,
        alignment_representative_id=alignment_rep,
        template_ids=template_ids,
        cluster_id=cluster_id,
        cluster_size=cluster_size,
        **cif_fields,
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
    assert out.exists() and out.stat().st_size > 0
    reconstructed = read_augmented_cif(out, pdb_id=pdb_id, cache_name=cache_name)
    return original, reconstructed


def _from_json_roundtrip(
    json_str: str,
    atom_array,
    tmp_path: Path,
    pdb_id: str,
    cache_name: str = "test",
) -> tuple[ClusteredDatasetCache, ClusteredDatasetCache]:
    """Load a ClusteredDatasetCache from a JSON string, then round-trip via CIF.

    Args:
        json_str: Serialized cache.
        atom_array: Structure to write.
        tmp_path: Working directory for the JSON and CIF files.
        pdb_id: PDB ID of the entry in the cache.
        cache_name: Cache name for the reconstructed cache.

    Returns:
        ``(original, reconstructed)`` caches.
    """
    cache_file = tmp_path / f"{pdb_id}_cache.json"
    cache_file.write_text(json_str)
    original = ClusteredDatasetCache.from_json(cache_file)

    structure_data = original.structure_data[pdb_id]
    ref_mol = original.reference_molecule_data
    out = tmp_path / f"{pdb_id}_augmented.cif"
    write_augmented_cif(atom_array, structure_data, ref_mol, out, pdb_id=pdb_id)
    assert out.exists() and out.stat().st_size > 0
    reconstructed = read_augmented_cif(out, pdb_id=pdb_id, cache_name=cache_name)
    return original, reconstructed


# ── Molecule-type null patterns ───────────────────────────────────────────────
# Each test exercises the tier-2 null/non-null pattern characteristic of that
# molecule type.  minimal_3aa.cif (synthetic ALA-GLY-ALA) is the atom_array
# carrier for all four; only the of3_chain_meta encoding is under test.
# Tier-1 data is synthetic dicts — there is no CIF parsing here.
#
# Expected null patterns by molecule type:
#   PROTEIN : alignment_rep set, template_ids is a list (possibly empty)
#   RNA     : alignment_rep=None, template_ids=None
#   DNA     : alignment_rep=None, template_ids=None
#   LIGAND  : alignment_rep=None, template_ids=None, reference_mol_id set


class TestMoleculeTypes:
    def test_protein_type(self, cif_minimal, tmp_path):
        """PROTEIN: alignment_rep and template_ids are both populated."""
        sample_protein_data = {
            "auth_asym_id": "A",
            "entity_id": 1,
            "molecule_type": "PROTEIN",
            "reference_mol_id": None,
        }
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={
                "A": _annotated_chain(
                    "A",
                    sample_protein_data,
                    cluster_id=DUMMY_CLUSTER_ID,
                    cluster_size=DUMMY_CLUSTER_SIZE,
                    alignment_rep="hash_A",
                    template_ids=["1abc_A"],
                )
            },
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_minimal),
            structure_data=structure_data,
            reference_molecule_data={},
            tmp_path=tmp_path,
            pdb_id="protein_test",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_rna_type(self, cif_minimal, tmp_path):
        """RNA: alignment_rep=None and template_ids=None survive as None (not '')."""
        sample_rna_data = {
            "auth_asym_id": "A",
            "entity_id": 1,
            "molecule_type": "RNA",
            "reference_mol_id": None,
        }
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=None,
            chains={
                "A": _annotated_chain(
                    "A",
                    sample_rna_data,
                    cluster_id=DUMMY_CLUSTER_ID,
                    cluster_size=DUMMY_CLUSTER_SIZE,
                    alignment_rep=None,
                    template_ids=None,
                )
            },
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_minimal),
            structure_data=structure_data,
            reference_molecule_data={},
            tmp_path=tmp_path,
            pdb_id="rna_test",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_dna_type(self, cif_minimal, tmp_path):
        """DNA: alignment_rep=None and template_ids=None survive as None (not '')."""
        sample_dna_data = {
            "auth_asym_id": "A",
            "entity_id": 1,
            "molecule_type": "DNA",
            "reference_mol_id": None,
        }
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2015, 3, 1),
            resolution=1.8,
            chains={
                "A": _annotated_chain(
                    "A",
                    sample_dna_data,
                    cluster_id=DUMMY_CLUSTER_ID,
                    cluster_size=DUMMY_CLUSTER_SIZE,
                    alignment_rep=None,
                    template_ids=None,
                )
            },
            interfaces={},
        )
        original, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_minimal),
            structure_data=structure_data,
            reference_molecule_data={},
            tmp_path=tmp_path,
            pdb_id="dna_test",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_ligand_type(self, cif_minimal, tmp_path):
        """LIGAND: reference_mol_id set; alignment_rep=None, template_ids=None."""
        sample_ligand_data = {
            "auth_asym_id": "A",
            "entity_id": 2,
            "molecule_type": "LIGAND",
            "reference_mol_id": "LIG",
        }
        structure_data = ClusteredDatasetStructureData(
            release_date=date(2011, 8, 3),
            resolution=2.2,
            chains={
                "A": _annotated_chain(
                    "A",
                    sample_ligand_data,
                    cluster_id=DUMMY_CLUSTER_ID,
                    cluster_size=DUMMY_CLUSTER_SIZE,
                    alignment_rep=None,
                    template_ids=None,
                )
            },
            interfaces={},
        )
        ref_mol = {
            "LIG": DatasetReferenceMoleculeData(
                conformer_gen_strategy="rdkit",
                fallback_conformer_pdb_id=None,
                canonical_smiles="[Ni+2]",
                set_fallback_to_nan=False,
            )
        }
        original, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_minimal),
            structure_data=structure_data,
            reference_molecule_data=ref_mol,
            tmp_path=tmp_path,
            pdb_id="ligand_test",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)


# ── 2q2k: protein + DNA + ligand complex ─────────────────────────────────────
# Tier-1 data is read at test time from 2q2k.cif via parse_cif_chain_data, so
# the tests stay correct if the fixture file is replaced.
#
# 2q2k entity layout:
#   entity 1: polydeoxyribonucleotide → chain A
#   entity 2: polypeptide(L)          → chains B, C  (homodimer)
#   entity 3: non-polymer, CCD EPE   → chain D


class TestDNAComplex:
    def _structure_data(self, cif_path: Path) -> ClusteredDatasetStructureData:
        """Build 2q2k structure data from CIF-derived tier-1 fields plus fixed annotations.

        Args:
            cif_path: The 2q2k CIF.

        Returns:
            Structure data with chains A-D and interfaces A_B, B_C.
        """
        cif_data = parse_cif_chain_data(cif_path)
        chains = {
            # DNA chain: no template search, no alignment rep
            "A": _annotated_chain(
                "A",
                cif_data["A"],
                cluster_id="dna_c1",
                cluster_size=5,
                alignment_rep=None,
                template_ids=None,
            ),
            # PROTEIN chains: alignment rep + template list
            "B": _annotated_chain(
                "B",
                cif_data["B"],
                cluster_id="prot_c1",
                cluster_size=20,
                alignment_rep="hash_2q2k_B",
                template_ids=["1abc_A", "2def_B"],
            ),
            "C": _annotated_chain(
                "C",
                cif_data["C"],
                cluster_id="prot_c1",
                cluster_size=20,
                alignment_rep="hash_2q2k_C",
                template_ids=["1abc_A"],
            ),
            # LIGAND chain: reference_mol_id from CIF (EPE), no templates
            "D": _annotated_chain(
                "D",
                cif_data["D"],
                cluster_id="lig_c2",
                cluster_size=1,
                alignment_rep=None,
                template_ids=None,
            ),
        }
        interfaces = {
            "A_B": ClusteredDatasetInterfaceData(cluster_id="if1", cluster_size=8),
            "B_C": ClusteredDatasetInterfaceData(cluster_id="if2", cluster_size=12),
        }
        return ClusteredDatasetStructureData(
            release_date=date(2007, 11, 20),
            resolution=2.0,
            chains=chains,
            interfaces=interfaces,
        )

    def _ref_mol_data(self, cif_path: Path) -> dict[str, DatasetReferenceMoleculeData]:
        """Build the reference molecule record for 2q2k's ligand chain D.

        Args:
            cif_path: The 2q2k CIF.

        Returns:
            One-entry dict keyed by chain D's reference_mol_id.
        """
        cif_data = parse_cif_chain_data(cif_path)
        mol_id = cif_data["D"]["reference_mol_id"]  # "EPE" for 2q2k
        return {
            mol_id: DatasetReferenceMoleculeData(
                conformer_gen_strategy="rdkit",
                fallback_conformer_pdb_id=None,
                canonical_smiles="OCCP1CCN(CCCS(=O)(=O)O)CC1",
                set_fallback_to_nan=False,
            )
        }

    def test_all_chain_types_roundtrip(self, cif_dna_complex, tmp_path):
        """All four chains (DNA/PROTEIN/PROTEIN/LIGAND) round-trip correctly."""
        structure_data = self._structure_data(cif_dna_complex)
        ref_mol = self._ref_mol_data(cif_dna_complex)
        original, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_dna_complex),
            structure_data=structure_data,
            reference_molecule_data=ref_mol,
            tmp_path=tmp_path,
            pdb_id="2q2k",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_dna_chain_fields(self, cif_dna_complex, tmp_path):
        """DNA chain: CIF-derived molecule_type + expected None annotation fields."""
        cif_data = parse_cif_chain_data(cif_dna_complex)
        structure_data = self._structure_data(cif_dna_complex)
        ref_mol = self._ref_mol_data(cif_dna_complex)
        _, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_dna_complex),
            structure_data=structure_data,
            reference_molecule_data=ref_mol,
            tmp_path=tmp_path,
            pdb_id="2q2k",
        )
        chain_a = reconstructed.structure_data["2q2k"].chains["A"]

        # Tier 1 (from the CIF): chain A is the DNA strand of entity 1.
        assert chain_a.molecule_type == cif_data["A"]["molecule_type"]  # "DNA"
        assert chain_a.entity_id == cif_data["A"]["entity_id"]  # 1

        # Tier 2: DNA has neither an alignment representative nor templates.
        assert chain_a.alignment_representative_id is None
        assert chain_a.template_ids is None

    def test_ligand_chain_fields(self, cif_dna_complex, tmp_path):
        """LIGAND chain: CIF-derived reference_mol_id + expected None annotation fields."""
        cif_data = parse_cif_chain_data(cif_dna_complex)
        structure_data = self._structure_data(cif_dna_complex)
        ref_mol = self._ref_mol_data(cif_dna_complex)
        _, reconstructed = _roundtrip(
            atom_array=_parse_aw(cif_dna_complex),
            structure_data=structure_data,
            reference_molecule_data=ref_mol,
            tmp_path=tmp_path,
            pdb_id="2q2k",
        )
        chain_d = reconstructed.structure_data["2q2k"].chains["D"]

        # Tier 1 (from the CIF): chain D is the HEPES (EPE) ligand.
        assert chain_d.molecule_type == cif_data["D"]["molecule_type"]  # "LIGAND"
        assert chain_d.reference_mol_id == cif_data["D"]["reference_mol_id"]  # "EPE"

        # Tier 2: a ligand has neither an alignment representative nor templates.
        assert chain_d.alignment_representative_id is None
        assert chain_d.template_ids is None


# ── JSON round-trip (load from cache JSON → CIF → compare) ───────────────────
# Tier-1 values below match parse_cif_chain_data output for the real CIFs.


class TestFromJSON:
    # 3u8v chain A: entity 1, polypeptide(L), auth_asym_id A
    _3U8V_CACHE_JSON = json.dumps(
        {
            "_type": "ClusteredDatasetCache",
            "name": "test-3u8v",
            "structure_data": {
                "3u8v": {
                    "release_date": "2011-08-03",
                    "resolution": 2.2,
                    "chains": {
                        "A": {
                            # tier-1: CIF-derived (struct_asym + entity_poly)
                            "label_asym_id": "A",
                            "auth_asym_id": "A",
                            "entity_id": 1,
                            "molecule_type": "PROTEIN",
                            "reference_mol_id": None,
                            # tier-2: pipeline annotation
                            "alignment_representative_id": "HASH_3U8V_A",
                            "template_ids": ["1abc_A", "2def_B"],
                            "cluster_id": "cluster_42",
                            "cluster_size": 17,
                        },
                    },
                    "interfaces": {},
                },
            },
            "reference_molecule_data": {},
        }
    )

    # 2q2k entity layout:
    #   entity 1: polydeoxyribonucleotide → chain A (auth F)
    #   entity 2: polypeptide(L)          → chains B (auth A), C (auth B)
    #   entity 3: non-polymer, CCD EPE   → chain D (auth A)
    _2Q2K_CACHE_JSON = json.dumps(
        {
            "_type": "ClusteredDatasetCache",
            "name": "test-2q2k",
            "structure_data": {
                "2q2k": {
                    "release_date": "2007-11-20",
                    "resolution": 2.0,
                    "chains": {
                        "A": {
                            # tier-1: CIF-derived (entity 1, polydeoxyribonucleotide)
                            "label_asym_id": "A",
                            "auth_asym_id": "F",
                            "entity_id": 1,
                            "molecule_type": "DNA",
                            "reference_mol_id": None,
                            # tier-2: annotation (DNA has no templates or alignment rep)
                            "alignment_representative_id": None,
                            "template_ids": None,
                            "cluster_id": "dna_c1",
                            "cluster_size": 5,
                        },
                        "B": {
                            # tier-1: CIF-derived (entity 2, polypeptide(L))
                            "label_asym_id": "B",
                            "auth_asym_id": "A",
                            "entity_id": 2,
                            "molecule_type": "PROTEIN",
                            "reference_mol_id": None,
                            # tier-2: annotation
                            "alignment_representative_id": "HASH_2Q2K_B",
                            "template_ids": ["1abc_A"],
                            "cluster_id": "prot_c1",
                            "cluster_size": 20,
                        },
                        "C": {
                            # tier-1: CIF-derived (entity 2, same polymer type as B — homodimer)
                            "label_asym_id": "C",
                            "auth_asym_id": "B",
                            "entity_id": 2,
                            "molecule_type": "PROTEIN",
                            "reference_mol_id": None,
                            # tier-2: annotation (different alignment rep per chain copy)
                            "alignment_representative_id": "HASH_2Q2K_C",
                            "template_ids": ["1abc_A"],
                            "cluster_id": "prot_c1",
                            "cluster_size": 20,
                        },
                        "D": {
                            # tier-1: CIF-derived (entity 3, non-polymer, CCD EPE)
                            "label_asym_id": "D",
                            "auth_asym_id": "A",
                            "entity_id": 3,
                            "molecule_type": "LIGAND",
                            "reference_mol_id": "EPE",
                            # tier-2: annotation (LIGAND has no templates or alignment rep)
                            "alignment_representative_id": None,
                            "template_ids": None,
                            "cluster_id": "lig_c2",
                            "cluster_size": 1,
                        },
                    },
                    "interfaces": {
                        "A_B": {"cluster_id": "if1", "cluster_size": 8},
                        "B_C": {"cluster_id": "if2", "cluster_size": 12},
                    },
                },
            },
            "reference_molecule_data": {
                "EPE": {
                    "conformer_gen_strategy": "rdkit",
                    "fallback_conformer_pdb_id": None,
                    "canonical_smiles": "OCCP1CCN(CCCS(=O)(=O)O)CC1",
                    "set_fallback_to_nan": False,
                },
            },
        }
    )

    def test_protein_monomer_from_json(self, cif_protein_monomer, tmp_path):
        """Protein monomer: load from JSON, round-trip via CIF."""
        original, reconstructed = _from_json_roundtrip(
            json_str=self._3U8V_CACHE_JSON,
            atom_array=_parse_aw(cif_protein_monomer),
            tmp_path=tmp_path,
            pdb_id="3u8v",
            cache_name="test-3u8v",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)

    def test_dna_complex_from_json(self, cif_dna_complex, tmp_path):
        """DNA+protein+ligand complex: load from JSON, round-trip via CIF."""
        original, reconstructed = _from_json_roundtrip(
            json_str=self._2Q2K_CACHE_JSON,
            atom_array=_parse_aw(cif_dna_complex),
            tmp_path=tmp_path,
            pdb_id="2q2k",
            cache_name="test-2q2k",
        )
        # The cache read back from the CIF equals the one written, field for field.
        assert_caches_equal(original, reconstructed)
