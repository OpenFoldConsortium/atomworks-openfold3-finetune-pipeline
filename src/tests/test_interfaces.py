"""Interface keys and per-structure cluster sizes in the dataset cache."""

import numpy as np
import pytest
from biotite.structure import Atom, array
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.primitives.structure.interface import get_interface_chain_id_pairs

from aw_of3_finetune.constants import DUMMY_CLUSTER_ID
from aw_of3_finetune.pipeline import (
    build_interface_keys,
    parse_augmented_cif_for_training,
    prepare_augmented_cif,
)
from conftest import FIXTURES_MMCIFS


def _atom(chain_id: str, coord: tuple[float, float, float]) -> Atom:
    """One carbon atom of a one-residue chain."""
    return Atom(coord, chain_id=chain_id, res_name="ALA", atom_name="CA", element="C")


def test_two_chains_in_contact_give_one_key():
    atoms = array([_atom("1", (0, 0, 0)), _atom("2", (4, 0, 0))])

    expected_keys = ["1_2"]
    actual_keys = build_interface_keys(atoms)
    assert actual_keys == expected_keys


def test_chains_farther_than_5_angstrom_give_no_key():
    atoms = array([_atom("1", (0, 0, 0)), _atom("2", (6, 0, 0))])

    expected_keys = []
    actual_keys = build_interface_keys(atoms)
    assert actual_keys == expected_keys


def test_unresolved_atoms_make_no_contact():
    atoms = array(
        [
            _atom("1", (0, 0, 0)),
            _atom("2", (np.nan, np.nan, np.nan)),
            _atom("2", (20, 0, 0)),
        ]
    )

    expected_keys = []
    actual_keys = build_interface_keys(atoms)
    assert actual_keys == expected_keys


def test_ten_plus_chains_keep_ofs_string_order():
    atoms = array([_atom(str(i), (10.0 * i, 0, 0)) for i in range(1, 12)])
    atoms.coord[atoms.chain_id == "10", 0] = atoms.coord[atoms.chain_id == "2", 0] + 3.0

    actual_keys = build_interface_keys(atoms)

    expected_keys = ["10_2"]
    assert actual_keys == expected_keys
    first, second = actual_keys[0].split("_")
    assert first.isdigit() and second.isdigit()


@pytest.fixture(scope="module")
def prepared(ccd_path, tmp_path_factory):
    """pdb_id -> (structure_data, out_cif) from prepare_augmented_cif, built once per module."""
    results = {}
    for pdb_id in ("7riv", "1a3n", "2crb"):
        workdir = tmp_path_factory.mktemp(pdb_id)
        out_cif = workdir / f"{pdb_id}.cif"
        structure_data, _ = prepare_augmented_cif(
            FIXTURES_MMCIFS / f"{pdb_id}.cif",
            pdb_id,
            out_cif,
            ccd=CIFFile.read(ccd_path),
            ref_mol_dir=workdir,
        )
        results[pdb_id] = (structure_data, out_cif)
    return results


def test_7riv_protein_ligand_interface(prepared):
    structure_data, _ = prepared["7riv"]

    # chain 1 = PXR (A), chain 2 = P06 (B)
    expected_keys = {"1_2"}
    actual_keys = set(structure_data.interfaces)
    assert actual_keys == expected_keys


def test_1a3n_keys_equal_ofs_pairs_on_the_same_atoms(prepared):
    structure_data, out_cif = prepared["1a3n"]
    atom_array = parse_augmented_cif_for_training(out_cif).atom_array

    pairs = get_interface_chain_id_pairs(atom_array, distance_threshold=5.0).tolist()
    expected_keys = {f"{a}_{b}" for a, b in pairs}

    actual_keys = set(structure_data.interfaces)
    assert actual_keys == expected_keys
    expected_known_keys = {"1_2", "1_3", "1_4", "1_5", "2_3", "2_6", "3_4", "3_7", "4_8"}
    assert expected_keys == expected_known_keys


def test_every_key_names_two_chains_of_the_structure(prepared):
    structure_data, _ = prepared["1a3n"]

    for key in structure_data.interfaces:
        assert set(key.split("_")) <= set(structure_data.chains)


def test_cluster_size_is_the_structures_datapoint_count(prepared):
    structure_data, _ = prepared["1a3n"]
    datapoints = [*structure_data.chains.values(), *structure_data.interfaces.values()]

    expected_size = 17  # 8 chains + 9 interfaces
    actual_sizes = {d.cluster_size for d in datapoints}
    assert actual_sizes == {expected_size}
    actual_ids = {d.cluster_id for d in datapoints}
    assert actual_ids == {DUMMY_CLUSTER_ID}


def test_single_chain_structure_has_no_interfaces_and_size_1(prepared):
    structure_data, _ = prepared["2crb"]

    assert structure_data.interfaces == {}
    assert [c.cluster_size for c in structure_data.chains.values()] == [1]


def test_materialized_cache_has_interface_keys_and_sizes(built_dataset):
    """dataset_cache.json carries every structure's interface keys and its datapoint count."""
    cache = read_datacache(built_dataset.output_dir / "dataset_cache.json")

    actual_keys = {pdb_id: set(entry.interfaces) for pdb_id, entry in cache.structure_data.items()}
    expected_keys = {
        "1a3n": {"1_2", "1_3", "1_4", "1_5", "2_3", "2_6", "3_4", "3_7", "4_8"},
        "2crb": set(),
        # Assembly copies A_1..C_2 are chains 1-6.
        "2q2k": {"1_2", "1_3", "1_4", "1_5", "1_6", "2_3", "2_4", "2_5", "2_6",
                 "3_4", "3_5", "4_5", "4_6", "5_6"},
        "3u8v": {"1_2"},  # protein + Ni
        "7riv": {"1_2"},
    }
    assert actual_keys == expected_keys

    actual_sizes = {
        pdb_id: {d.cluster_size for d in [*entry.chains.values(), *entry.interfaces.values()]}
        for pdb_id, entry in cache.structure_data.items()
    }
    expected_sizes = {"1a3n": {17}, "2crb": {1}, "2q2k": {20}, "3u8v": {3}, "7riv": {3}}
    assert actual_sizes == expected_sizes
