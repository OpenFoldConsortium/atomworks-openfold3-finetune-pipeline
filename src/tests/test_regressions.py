"""Regression tests seeded from design-time findings; harness findings graduate here."""


from biotite.structure.io.pdbx import CIFFile

from aw_of3_finetune.cif_reader import parse_cif_chain_data
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences, collect_msa_queries
from aw_of3_finetune.pipeline import prepare_augmented_cif
from aw_of3_finetune.validate import run_validation
from aw_of3_finetune.validate.check_ids import CHAINS_ACCOUNTED, MSA_QUERY_ROWS, SDF_ATOM_NAMES_MATCH
from aw_of3_finetune.validate.findings import Severity
from conftest import FIXTURES_MMCIFS
from validate_helpers import context



def _residue_names(built):
    """Map entry to the set of residue names in its augmented CIF."""
    names = {}
    for cif in (built.output_dir / "cleaned_structures").glob("*.cif"):
        block = CIFFile.read(cif).block
        names[cif.stem] = set(block["atom_site"]["label_comp_id"].as_array().tolist())
    return names


def test_metal_kept_aid_stripped(built_dataset):
    """3u8v keeps its NI ions; 2q2k loses EPE, which V1 reports as a crystallization aid."""
    names = _residue_names(built_dataset)

    # Metal ions are kept; the crystallization aid (HEPES) is removed.
    assert "NI" in names["3u8v"]
    assert "EPE" not in names["2q2k"]

    report = run_validation(context(built_dataset), [CHAINS_ACCOUNTED])

    # V1 records exactly one INFO crystallization-aid removal for 2q2k.
    aid_findings = [
        f for f in report.findings
        if f.entry == "2q2k" and f.severity == Severity.INFO
        and f.detail.get("reason") == "crystallization_aid"
    ]
    assert len(aid_findings) == 1

    # In the source CIF, the removed chain consists solely of EPE residues.
    source_atoms = CIFFile.read(FIXTURES_MMCIFS / "2q2k.cif").block["atom_site"]
    removed_chain_mask = source_atoms["label_asym_id"].as_array() == aid_findings[0].detail["chain"]
    removed_chain_residues = source_atoms["label_comp_id"].as_array()[removed_chain_mask]
    assert set(removed_chain_residues.tolist()) == {"EPE"}


def test_homomer_copies_distinct_chains_shared_rep(built_dataset):
    """2q2k's four protein copies keep chains 2, 3, 5, 6 and share one alignment representative."""
    chains = context(built_dataset).dataset_cache.structure_data["2q2k"].chains
    protein_chains = {
        chain_id: chain for chain_id, chain in chains.items() if chain.molecule_type == "PROTEIN"
    }

    # Four protein copies, each its own chain.
    assert sorted(protein_chains) == ["2", "3", "5", "6"]

    # All four share a single (non-null) MSA representative.
    representative_ids = {
        chain.alignment_representative_id for chain in protein_chains.values()
    }
    assert len(representative_ids) == 1
    assert None not in representative_ids


def test_msa_query_row_matches_sequence(built_dataset):
    """V13 finds no error on the built dataset."""
    report = run_validation(context(built_dataset), [MSA_QUERY_ROWS])

    assert report.errors == []


def test_crystal_atom_names_subset_of_sdf(built_dataset):
    """V10 (crystal heavy-atom names are a subset of the SDF's) finds no error."""
    report = run_validation(context(built_dataset), [SDF_ATOM_NAMES_MATCH])

    assert report.errors == []


def _written_auth_by_label(cif_path):
    """Map each written label_asym_id to the set of auth_asym_ids written for it."""
    atom_site = CIFFile.read(cif_path).block["atom_site"]
    labels = atom_site["label_asym_id"].as_array().tolist()
    auths = atom_site["auth_asym_id"].as_array().tolist()
    written = {}
    for label, auth in zip(labels, auths):
        written.setdefault(label, set()).add(auth)
    return written


def test_ligand_chains_keep_their_author_chain_id(ccd_path, tmp_path):
    """7axf's ligand chains B-H are author chain A in the source CIF; the cache keeps A, as OF3 does."""
    out = tmp_path / "7axf.cif"
    structure_data, _ = prepare_augmented_cif(
        FIXTURES_MMCIFS / "7axf.cif", "7axf", out, ccd=CIFFile.read(ccd_path)
    )

    expected_cache_auth = {label: "A" for label in "ABCDEFGH"}
    actual_cache_auth = {chain.label_asym_id: chain.auth_asym_id for chain in structure_data.chains.values()}
    assert actual_cache_auth == expected_cache_auth



def test_written_auth_chain_ids_stay_unique_per_chain(ccd_path, tmp_path):
    """OF3 re-parses the augmented CIF with author fields, so each chain keeps its own auth id there.

    Writing the source auth id (A for every 4x1f chain) merged chains and lost residue bonds (2qnv).
    """
    out = tmp_path / "4x1f.cif"
    prepare_augmented_cif(FIXTURES_MMCIFS / "4x1f.cif", "4x1f", out, ccd=CIFFile.read(ccd_path))

    expected_written_auth = {"A_1": {"A_1"}, "A_2": {"A_2"}, "B_1": {"B_1"}, "B_2": {"B_2"}}
    actual_written_auth = _written_auth_by_label(out)
    assert actual_written_auth == expected_written_auth


def test_assembly_copies_keep_the_source_author_chain_id(ccd_path, tmp_path):
    """4x1f's assembly copies A_1/A_2 and B_1/B_2 all carry the source author chain A, as OF3 repeats it."""
    out = tmp_path / "4x1f.cif"
    structure_data, _ = prepare_augmented_cif(
        FIXTURES_MMCIFS / "4x1f.cif", "4x1f", out, ccd=CIFFile.read(ccd_path)
    )

    expected_cache_auth = {"A_1": "A", "A_2": "A", "B_1": "A", "B_2": "A"}
    actual_cache_auth = {chain.label_asym_id: chain.auth_asym_id for chain in structure_data.chains.values()}
    assert actual_cache_auth == expected_cache_auth


def _written_atom_names(cif_path, label_asym_id):
    """Atom names written for one label_asym_id of an augmented CIF."""
    atom_site = CIFFile.read(cif_path).block["atom_site"]
    labels = atom_site["label_asym_id"].as_array()
    return set(atom_site["label_atom_id"].as_array()[labels == label_asym_id].tolist())


def test_covalent_ligand_gets_no_unresolved_atoms_added(ccd_path, tmp_path):
    """7axf's S6T in chain C is bonded to CYS SG and lost its CL; as in OF3, CL is not re-added there.

    The non-covalent S6T copy in chain B keeps its (resolved) CL.
    """
    out = tmp_path / "7axf.cif"
    prepare_augmented_cif(FIXTURES_MMCIFS / "7axf.cif", "7axf", out, ccd=CIFFile.read(ccd_path))

    actual_covalent_names = _written_atom_names(out, "C")
    actual_free_names = _written_atom_names(out, "B")

    expected_covalent_atom_count = 20
    assert "CL" not in actual_covalent_names
    assert len(actual_covalent_names) == expected_covalent_atom_count
    assert "CL" in actual_free_names


def _without_entity_poly_type(src, dst):
    """Copy src to dst with the entity_poly.type column removed, as in the OpenBind monomer CIFs."""
    cif = CIFFile.read(src)
    del cif.block["entity_poly"]["type"]
    cif.write(dst)
    return dst


def _entity_poly_types(cif_path):
    """entity_poly entity_id -> type of a CIF."""
    entity_poly = CIFFile.read(cif_path).block["entity_poly"]
    return dict(zip(entity_poly["entity_id"].as_array().tolist(), entity_poly["type"].as_array().tolist()))


def test_missing_entity_poly_type_is_filled_from_atomworks(cif_dna_complex, tmp_path):
    """OpenBind CIFs have no entity_poly.type: the augmented CIF gets the types atomworks infers."""
    source = _without_entity_poly_type(cif_dna_complex, tmp_path / "2q2k_source.cif")
    expected_types = _entity_poly_types(cif_dna_complex)

    prepare_augmented_cif(source, "2q2k", tmp_path / "2q2k.cif")

    actual_types = _entity_poly_types(tmp_path / "2q2k.cif")
    assert actual_types == expected_types


def test_missing_entity_poly_type_still_queries_only_protein(cif_dna_complex, tmp_path):
    source = _without_entity_poly_type(cif_dna_complex, tmp_path / "2q2k_source.cif")
    (tmp_path / "cleaned_structures").mkdir()
    prepare_augmented_cif(source, "2q2k", tmp_path / "cleaned_structures" / "2q2k.cif")
    (tmp_path / "manifest.json").write_text('{"name": "t", "structures": ["2q2k"]}')

    actual_queries = collect_msa_queries(tmp_path)

    protein_seq = chain_canonical_sequences(cif_dna_complex)["B"]
    expected_queries = {alignment_rep_id(protein_seq): protein_seq}
    assert actual_queries == expected_queries


def test_chain_data_reads_a_cif_without_entity_poly_type(cif_dna_complex, tmp_path):
    source = _without_entity_poly_type(cif_dna_complex, tmp_path / "2q2k_source.cif")
    expected_chains = sorted(parse_cif_chain_data(cif_dna_complex))

    actual_chains = sorted(parse_cif_chain_data(source))

    assert actual_chains == expected_chains


def test_ligand_reference_mol_id_without_pdbx_entity_nonpoly(tmp_path):
    """OpenBind CIFs have no pdbx_entity_nonpoly: the ligand's component comes from its atom_site residue name."""
    source = tmp_path / "7riv_source.cif"
    cif = CIFFile.read(FIXTURES_MMCIFS / "7riv.cif")
    del cif.block["pdbx_entity_nonpoly"]
    cif.write(source)
    expected_ids = {label: chain["reference_mol_id"] for label, chain in parse_cif_chain_data(FIXTURES_MMCIFS / "7riv.cif").items()}

    actual_ids = {label: chain["reference_mol_id"] for label, chain in parse_cif_chain_data(source).items()}

    assert actual_ids == expected_ids
