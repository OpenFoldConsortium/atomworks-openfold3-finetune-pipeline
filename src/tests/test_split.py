"""Split CSV parsing: id,split rows become a val/test assignment, with every bad row reported at once."""

import dataclasses
import hashlib
import json
from datetime import date
from pathlib import Path

import biotite.structure as struc
import click
import numpy as np
import pytest
from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz, write_atomarray_to_npz
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    DatasetReferenceMoleculeData,
    ValidationDatasetCache,
)
from openfold3.core.data.primitives.structure.tokenization import get_token_count
from openfold3.core.data.resources.residues import MoleculeType

import aw_of3_finetune.split as split_module
from aw_of3_finetune.split import (
    SplitAssignment,
    SplitCsvError,
    dataset_cache_sha256,
    read_split_csv,
    split_dataset_cache,
    split_paths,
    to_validation_cache,
)
from validate_helpers import mutable_copy

TEMPLATE = Path(__file__).parents[2] / "examples" / "split_template.csv"


def write(tmp_path, text):
    path = tmp_path / "split.csv"
    path.write_bytes(text.encode("utf-8"))
    return path


def error_message(csv, **kwargs):
    with pytest.raises(SplitCsvError) as err:
        read_split_csv(csv, **kwargs)
    return str(err.value.message)


def test_val_and_test_rows_are_assigned(tmp_path):
    csv = write(tmp_path, "id,split\n7riv,val\n3u8v,val\n1a3n,test\n")
    actual = read_split_csv(csv, known_ids={"7riv", "3u8v", "1a3n", "2crb"})
    assert actual == SplitAssignment(val=frozenset({"7riv", "3u8v"}), test=frozenset({"1a3n"}))


def test_ids_match_case_insensitively(tmp_path):
    csv = write(tmp_path, "id,split\n7RIV,VAL\n")
    actual = read_split_csv(csv, known_ids={"7riv"})
    assert actual.val == {"7riv"}


def test_excel_style_csv_parses(tmp_path):
    csv = write(tmp_path, "﻿id,split\r\n 7riv , val \r\n\r\n")
    actual = read_split_csv(csv, known_ids={"7riv"})
    assert actual.val == {"7riv"}


def test_every_bad_row_is_reported_at_once(tmp_path):
    csv = write(tmp_path, "id,split\n7rxx,val\n7riv,train\n3u8v,val\n3u8v,test\n")
    message = error_message(csv, known_ids={"7riv", "3u8v"})
    assert "line 2: '7rxx' is not in the dataset" in message
    assert "line 3" in message and "'train'" in message
    assert "line 5: '3u8v' is listed more than once" in message
    assert "3 problem(s)" in message


def test_no_val_rows_is_an_error(tmp_path):
    csv = write(tmp_path, "id,split\n1a3n,test\n")
    message = error_message(csv, known_ids={"1a3n"})
    assert "no val rows" in message


def test_missing_header_is_an_error(tmp_path):
    csv = write(tmp_path, "pdb,split\n7riv,val\n")
    message = error_message(csv, known_ids={"7riv"})
    assert "'id'" in message


def test_dropped_id_gets_its_own_message(tmp_path):
    csv = write(tmp_path, "id,split\n2Q2K,val\n7riv,val\n")
    message = error_message(csv, known_ids={"7riv"}, dropped_ids={"2q2k"})
    assert "dropped by the pipeline; see the finetune-prep log" in message
    assert "not in the dataset" not in message


def test_example_template_parses():
    actual = read_split_csv(TEMPLATE, known_ids={"7riv", "3u8v", "1a3n"})
    assert actual.val == {"7riv", "3u8v"}
    assert actual.test == {"1a3n"}


def test_error_line_number_matches_the_file_line_with_blank_lines(tmp_path):
    csv = write(tmp_path, "id,split\n7riv,val\n\n\n7rxx,val\n")
    expected_line = 5
    message = error_message(csv, known_ids={"7riv"})
    assert f"line {expected_line}: '7rxx' is not in the dataset" in message


def test_row_with_extra_cells_is_reported_not_a_crash(tmp_path):
    csv = write(tmp_path, "id,split\n7riv,val\n3u8v,val,my note\n")
    message = error_message(csv, known_ids={"7riv", "3u8v"})
    assert "line 3: has extra cells; expected id,split" in message


# --- splitting the dataset cache ---


@pytest.fixture
def built_copy(built_dataset, tmp_path):
    return mutable_copy(built_dataset, tmp_path / "copy")


def split_with(built_copy, tmp_path, text="id,split\n7riv,val\n3u8v,val\n1a3n,test\n"):
    csv = write(tmp_path, text)
    result = split_dataset_cache(built_copy.output_dir, csv)
    return result, split_paths(built_copy.output_dir)


def test_entries_go_to_train_val_or_neither(built_copy, tmp_path):
    result, paths = split_with(built_copy, tmp_path)
    actual_train = set(read_datacache(paths.train_cache).structure_data)
    actual_val = set(read_datacache(paths.val_cache).structure_data)
    assert actual_train == {"2crb", "2q2k"}
    assert actual_val == {"7riv", "3u8v"}
    assert (result.n_train, result.n_val, result.n_test) == (2, 2, 1)


def test_val_cache_is_an_of3_validation_cache(built_copy, tmp_path):
    _, paths = split_with(built_copy, tmp_path)
    val = read_datacache(paths.val_cache)
    assert isinstance(val, ValidationDatasetCache)
    chain = val.structure_data["7riv"].chains["1"]
    assert (chain.low_homology, chain.metric_eligible, chain.use_metrics) == (True, True, True)
    assert chain.source_subset == "base"
    assert chain.ranking_model_fit is None
    assert chain.sabdab_annotation is None
    interface = val.structure_data["7riv"].interfaces["1_2"]
    actual_flags = (interface.low_homology, interface.metric_eligible, interface.use_metrics, interface.source_subset)
    assert actual_flags == (True, True, True, "base")


def test_train_cache_is_a_clustered_cache_with_the_full_names_and_molecules(built_copy, tmp_path):
    _, paths = split_with(built_copy, tmp_path)
    full = read_datacache(built_copy.output_dir / "dataset_cache.json")
    train = read_datacache(paths.train_cache)
    val = read_datacache(paths.val_cache)
    assert isinstance(train, ClusteredDatasetCache)
    assert train.name == full.name
    assert val.name == f"{full.name}-val"
    assert train.reference_molecule_data == full.reference_molecule_data


def test_token_count_is_ofs_on_the_entry_npz(built_copy, tmp_path):
    _, paths = split_with(built_copy, tmp_path)
    val = read_datacache(paths.val_cache)
    npz = built_copy.output_dir / "structure_files" / "7riv" / "7riv.npz"
    expected = get_token_count(read_atomarray_from_npz(npz))
    assert val.structure_data["7riv"].token_count == expected


def test_template_ids_and_chain_fields_are_preserved(built_copy, tmp_path):
    _, paths = split_with(built_copy, tmp_path)
    full = read_datacache(built_copy.output_dir / "dataset_cache.json")
    for cache_path in (paths.train_cache, paths.val_cache):
        for pdb_id, structure in read_datacache(cache_path).structure_data.items():
            expected_structure = full.structure_data[pdb_id]
            assert structure.interfaces.keys() == expected_structure.interfaces.keys()
            for key, interface in structure.interfaces.items():
                assert interface.cluster_id == expected_structure.interfaces[key].cluster_id
                assert interface.cluster_size == expected_structure.interfaces[key].cluster_size
            for chain_id, chain in structure.chains.items():
                expected_chain = dataclasses.asdict(expected_structure.chains[chain_id])
                actual_chain = {name: getattr(chain, name) for name in expected_chain}
                assert actual_chain == expected_chain


def test_reference_molecules_get_residue_count(built_copy, tmp_path):
    _, paths = split_with(built_copy, tmp_path)
    full = read_datacache(built_copy.output_dir / "dataset_cache.json")
    val = read_datacache(paths.val_cache)
    referenced = {
        chain.reference_mol_id
        for structure in val.structure_data.values()
        for chain in structure.chains.values()
        if chain.reference_mol_id
    }
    assert referenced <= set(val.reference_molecule_data)
    assert set(val.reference_molecule_data) < set(full.reference_molecule_data)
    actual_counts = {mol.residue_count for mol in val.reference_molecule_data.values()}
    assert actual_counts == {1}


def test_multi_residue_ligand_gets_its_residue_count(tmp_path, monkeypatch):
    atoms = struc.AtomArray(6)
    atoms.coord = np.arange(18, dtype=float).reshape(6, 3)
    atoms.set_annotation("chain_id", np.array(["B"] * 6))
    atoms.set_annotation("res_id", np.array([1, 1, 2, 2, 3, 3]))
    atoms.set_annotation("res_name", np.array(["NAG", "NAG", "NAG", "NAG", "BMA", "BMA"]))
    atoms.set_annotation("atom_name", np.array(["C1", "C2"] * 3))
    atoms.set_annotation("entity_id", np.full(6, 2))
    atoms.set_annotation("molecule_type_id", np.full(6, int(MoleculeType.LIGAND)))
    out = tmp_path / "structure_files" / "xxxx"
    out.mkdir(parents=True)
    write_atomarray_to_npz(atoms, out / "xxxx.npz")
    monkeypatch.setattr(split_module, "get_token_count", lambda atom_array: 6)

    chain = ClusteredDatasetChainData(
        label_asym_id="B", auth_asym_id="B", entity_id=2, molecule_type="LIGAND",
        reference_mol_id="2_xxxx", alignment_representative_id=None, template_ids=None,
        cluster_id="c", cluster_size=1,
    )
    mol = DatasetReferenceMoleculeData("ccd", None, "C", False)
    cache = ClusteredDatasetCache(
        name="synthetic",
        structure_data={"xxxx": ClusteredDatasetStructureData(date(2020, 1, 1), 2.0, {"1": chain}, {})},
        reference_molecule_data={"2_xxxx": mol, "ALA": mol},
    )

    val = to_validation_cache(cache, {"xxxx"}, tmp_path)
    assert val.reference_molecule_data["2_xxxx"].residue_count == 3
    assert "ALA" not in val.reference_molecule_data


def test_resplit_replaces_the_split(built_copy, tmp_path):
    split_with(built_copy, tmp_path)
    csv = tmp_path / "second.csv"
    csv.write_text("id,split\n2crb,val\n")
    split_dataset_cache(built_copy.output_dir, csv)
    paths = split_paths(built_copy.output_dir)
    actual_val = set(read_datacache(paths.val_cache).structure_data)
    actual_train = set(read_datacache(paths.train_cache).structure_data)
    assert actual_val == {"2crb"}
    assert "7riv" in actual_train
    assert paths.csv.read_text() == "id,split\n2crb,val\n"


def test_source_hash_matches_dataset_cache(built_copy, tmp_path):
    _, paths = split_with(built_copy, tmp_path)
    expected = hashlib.sha256((built_copy.output_dir / "dataset_cache.json").read_bytes()).hexdigest()
    actual = json.loads(paths.source.read_text())["dataset_cache_sha256"]
    assert actual == expected
    assert dataset_cache_sha256(built_copy.output_dir) == expected


def test_csv_error_writes_nothing(built_copy, tmp_path):
    csv = write(tmp_path, "id,split\nnope,val\n")
    with pytest.raises(SplitCsvError):
        split_dataset_cache(built_copy.output_dir, csv)
    assert not split_paths(built_copy.output_dir).dir.exists()


def test_all_val_leaves_empty_train(built_copy, tmp_path):
    ids = ["1a3n", "2crb", "2q2k", "3u8v", "7riv"]
    result, paths = split_with(built_copy, tmp_path, "id,split\n" + "".join(f"{i},val\n" for i in ids))
    assert result.n_train == 0
    assert read_datacache(paths.train_cache).structure_data == {}


def test_cache_without_interfaces_warns(built_copy, tmp_path):
    cache_file = built_copy.output_dir / "dataset_cache.json"
    data = json.loads(cache_file.read_text())
    for structure in data["structure_data"].values():
        structure["interfaces"] = {}
    cache_file.write_text(json.dumps(data, indent=4))
    result, _ = split_with(built_copy, tmp_path)
    assert len(result.warnings) == 1
    assert "no interface metrics" in result.warnings[0]
    assert "finetune-prep" in result.warnings[0]


def test_failed_swap_keeps_the_previous_split(built_copy, tmp_path, monkeypatch):
    _, paths = split_with(built_copy, tmp_path)
    expected_val = set(read_datacache(paths.val_cache).structure_data)
    original_rename = Path.rename

    def failing_rename(self, target):
        if self.name == "split.tmp":
            raise OSError("disk full")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    csv = tmp_path / "second.csv"
    csv.write_text("id,split\n2crb,val\n")
    with pytest.raises(OSError):
        split_dataset_cache(built_copy.output_dir, csv)
    monkeypatch.undo()

    actual_val = set(read_datacache(paths.val_cache).structure_data)
    assert actual_val == expected_val
    assert not paths.dir.with_name("split.tmp").exists()


def test_split_old_left_by_a_hard_kill_is_restored(built_copy, tmp_path):
    """If a kill between the two renames left only split.old, the next run restores it first."""
    _, paths = split_with(built_copy, tmp_path)
    expected_val = set(read_datacache(paths.val_cache).structure_data)
    paths.dir.rename(paths.dir.with_name("split.old"))

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("id,split\nnope,val\n")
    with pytest.raises(SplitCsvError):
        split_dataset_cache(built_copy.output_dir, bad_csv)

    actual_val = set(read_datacache(paths.val_cache).structure_data)
    assert actual_val == expected_val
    assert not paths.dir.with_name("split.old").exists()


def test_missing_dataset_cache_is_a_clear_error(tmp_path):
    """An output dir without dataset_cache.json gives an error naming the file, not a traceback."""
    csv = write(tmp_path, "id,split\n7riv,val\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(click.ClickException) as error:
        split_dataset_cache(output_dir, csv)

    expected_path = str(output_dir / "dataset_cache.json")
    assert expected_path in error.value.message


def excluded_ligand_cache(tmp_path, monkeypatch):
    """A val entry with protein chain 1, ligand P06 (chain 2) and isopropanol IPA (chain 3)."""
    atoms = struc.AtomArray(3)
    atoms.coord = np.zeros((3, 3))
    atoms.set_annotation("chain_id", np.array(["1", "2", "3"]))
    atoms.set_annotation("res_id", np.array([1, 1, 1]))
    atoms.set_annotation("res_name", np.array(["ALA", "P06", "IPA"]))
    atoms.set_annotation("atom_name", np.array(["CA", "C1", "C1"]))
    atoms.set_annotation("entity_id", np.array([1, 2, 3]))
    atoms.set_annotation("molecule_type_id", np.array([int(MoleculeType.PROTEIN)] + [int(MoleculeType.LIGAND)] * 2))
    out = tmp_path / "structure_files" / "xxxx"
    out.mkdir(parents=True)
    write_atomarray_to_npz(atoms, out / "xxxx.npz")
    monkeypatch.setattr(split_module, "get_token_count", lambda atom_array: 3)

    def chain(label, molecule_type, mol_id):
        return ClusteredDatasetChainData(
            label_asym_id=label, auth_asym_id=label, entity_id=1, molecule_type=molecule_type,
            reference_mol_id=mol_id, alignment_representative_id=None, template_ids=None,
            cluster_id="c", cluster_size=1,
        )

    chains = {"1": chain("A", "PROTEIN", None), "2": chain("B", "LIGAND", "P06"), "3": chain("C", "LIGAND", "IPA")}
    interfaces = {key: ClusteredDatasetInterfaceData(cluster_id="c", cluster_size=1) for key in ("1_2", "1_3")}
    mol = DatasetReferenceMoleculeData("ccd", None, "C", False)
    return ClusteredDatasetCache(
        name="synthetic",
        structure_data={"xxxx": ClusteredDatasetStructureData(date(2020, 1, 1), 2.0, chains, interfaces)},
        reference_molecule_data={"P06": mol, "IPA": mol, "ALA": mol},
    )


def test_exclusion_list_ligand_chain_is_not_scored(tmp_path, monkeypatch):
    """A ligand on OF3's metric exclusion list (AF3 SI Tables 9, 10, 12) gets no chain metrics."""
    cache = excluded_ligand_cache(tmp_path, monkeypatch)
    chains = to_validation_cache(cache, {"xxxx"}, tmp_path).structure_data["xxxx"].chains
    actual_flags = {cid: (c.metric_eligible, c.use_metrics) for cid, c in chains.items()}
    expected_flags = {"1": (True, True), "2": (True, True), "3": (False, False)}
    assert actual_flags == expected_flags


def test_interface_with_an_exclusion_list_ligand_is_not_scored(tmp_path, monkeypatch):
    """An interface involving an excluded ligand gets no interface metrics; the others still do."""
    cache = excluded_ligand_cache(tmp_path, monkeypatch)
    interfaces = to_validation_cache(cache, {"xxxx"}, tmp_path).structure_data["xxxx"].interfaces
    actual_flags = {key: (i.metric_eligible, i.use_metrics) for key, i in interfaces.items()}
    expected_flags = {"1_2": (True, True), "1_3": (False, False)}
    assert actual_flags == expected_flags

