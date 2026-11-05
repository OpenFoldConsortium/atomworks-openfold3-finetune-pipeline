"""Ligand definitions from SMILES onto input atom names, and the ligand table.

The SMILES decides bond orders; the restraint CIF (or geometry) only gives connectivity. OF3's
own preprocessing of PXR got carboxylates (restraint `deloc` bonds) and N-H aromatics wrong.
"""

import pytest
from biotite.structure import Atom, AtomArray, BondList, BondType, array
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile
from rdkit import Chem

from aw_of3_finetune import ligand_inputs
from aw_of3_finetune.ligand_inputs import (
    LigandMatchError,
    LigandNamer,
    LigandRow,
    ccd_codes,
    ligand_definition,
    read_ligand_table,
    read_restraint_bonds,
    unlisted_placeholders,
)
from ligand_helpers import (
    ACID_SMILES,
    INDOLE_SMILES,
    X00242_PDB,
    X00242_RESTRAINT,
    X00358_PDB,
    X00358_WITH_H_PDB,
    definition_smiles,
    ligand_atoms,
)


def _inputs(atoms: AtomArray) -> tuple[list[str], list[str], object]:
    """(atom names, elements, coords) as ligand_definition takes them."""
    return list(atoms.atom_name), list(atoms.element), atoms.coord


@pytest.fixture
def acid_atoms() -> AtomArray:
    return ligand_atoms(X00242_PDB, res_id=701)


@pytest.fixture
def indole_atoms() -> AtomArray:
    return ligand_atoms(X00358_PDB, res_id=801)


def test_carboxylic_acid_keeps_its_double_bond(acid_atoms):
    """Restraint CIF says "deloc" for both C-O; the SMILES decides (no gem-diol)."""
    names, elements, coords = _inputs(acid_atoms)
    restraint_bonds = read_restraint_bonds(X00242_RESTRAINT, "LIG")

    actual = ligand_definition(ACID_SMILES, names, elements, coords, restraint_bonds)

    expected_smiles = Chem.CanonSmiles("O=C(O)Cc1cc(Cl)cc(Cl)c1")
    actual_smiles = definition_smiles(actual)
    assert actual_smiles == expected_smiles


def test_indole_stays_aromatic_without_restraint_bonds(indole_atoms):
    """Connectivity from geometry; the N-H ring stays aromatic (OF3 wrote it saturated)."""
    names, elements, coords = _inputs(indole_atoms)

    actual = ligand_definition(INDOLE_SMILES, names, elements, coords, restraint_bonds=None)

    expected_smiles = Chem.CanonSmiles("CNC(=O)c1cccc2cc[nH]c12")
    actual_smiles = definition_smiles(actual)
    assert actual_smiles == expected_smiles


def test_indole_ring_bonds_carry_aromatic_kekule_types(indole_atoms):
    """Ring bonds are AROMATIC_SINGLE/AROMATIC_DOUBLE; the amide C=O is a plain DOUBLE."""
    names, elements, coords = _inputs(indole_atoms)

    actual = ligand_definition(INDOLE_SMILES, names, elements, coords, restraint_bonds=None)

    bond_types = [BondType(t) for t in actual.bonds.as_array()[:, 2]]
    aromatic_types = (BondType.AROMATIC_SINGLE, BondType.AROMATIC_DOUBLE)
    actual_aromatic_bonds = sum(t in aromatic_types for t in bond_types)
    actual_double_bonds = bond_types.count(BondType.DOUBLE)
    expected_aromatic_bonds = 10  # indole: 6-ring + 5-ring sharing one bond
    expected_double_bonds = 1  # amide C=O
    assert actual_aromatic_bonds == expected_aromatic_bonds
    assert actual_double_bonds == expected_double_bonds


def test_atom_names_and_order_follow_the_input(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)

    actual = ligand_definition(ACID_SMILES, names, elements, coords, restraint_bonds=None)

    assert list(actual.atom_name) == names


def test_annotations_of_the_definition(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)

    actual = ligand_definition(ACID_SMILES, names, elements, coords, restraint_bonds=None)

    expected_elements = [element.upper() for element in elements]
    assert list(actual.element) == expected_elements
    assert actual.hetero.all()
    assert not actual.leaving_atom_flag.any()
    assert (actual.coord == coords).all()


def test_charge_comes_from_the_smiles(acid_atoms):
    """A carboxylate SMILES puts -1 on one oxygen."""
    names, elements, coords = _inputs(acid_atoms)

    actual = ligand_definition("[O-]C(=O)Cc1cc(Cl)cc(Cl)c1", names, elements, coords, None)

    expected_total_charge = -1
    actual_total_charge = int(actual.charge.sum())
    assert actual_total_charge == expected_total_charge


def test_hydrogens_in_the_input_are_ignored():
    """x00358-1_with_h.pdb's second copy carries explicit H; matching uses heavy atoms only."""
    atoms_with_h = ligand_atoms(X00358_WITH_H_PDB, res_id=801)
    names_with_h, elements_with_h, coords_with_h = _inputs(atoms_with_h)
    assert "H" in set(elements_with_h)  # the fixture really has hydrogens

    actual = ligand_definition(INDOLE_SMILES, names_with_h, elements_with_h, coords_with_h, None)

    expected_names = [n for n, element in zip(names_with_h, elements_with_h) if element != "H"]
    assert "H" not in set(actual.element)
    assert list(actual.atom_name) == expected_names
    assert definition_smiles(actual) == Chem.CanonSmiles("CNC(=O)c1cccc2cc[nH]c12")


def test_wrong_smiles_raises_match_error(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)

    with pytest.raises(LigandMatchError, match="heavy atoms"):
        ligand_definition("c1ccccc1", names, elements, coords, None)  # 6 atoms vs 12


def test_same_size_wrong_molecule_raises_match_error(acid_atoms):
    """12 heavy atoms but a different molecule: the template does not match."""
    names, elements, coords = _inputs(acid_atoms)

    with pytest.raises(LigandMatchError, match="does not match"):
        ligand_definition("O=C(O)Cc1cc(Br)cc(Cl)c1", names, elements, coords, None)


def test_multi_component_smiles_raises(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)

    with pytest.raises(LigandMatchError, match="more than one molecule"):
        ligand_definition("O=C(O)Cc1cc(Cl)cc(Cl)c1.[Na+]", names, elements, coords, None)


def test_unparsable_smiles_raises(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)

    with pytest.raises(LigandMatchError, match="could not parse"):
        ligand_definition("not a smiles", names, elements, coords, None)


def test_restraint_bond_to_unknown_atom_raises(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)
    restraint_bonds = [*read_restraint_bonds(X00242_RESTRAINT, "LIG"), ("C1", "XX9")]

    with pytest.raises(LigandMatchError, match="XX9"):
        ligand_definition(ACID_SMILES, names, elements, coords, restraint_bonds)


def test_restraint_bonds_are_heavy_atom_pairs_by_name():
    actual = read_restraint_bonds(X00242_RESTRAINT, "LIG")

    expected_bond_count = 12  # 17 bonds minus 5 to hydrogens
    assert len(actual) == expected_bond_count
    assert ("O1", "C1") in actual
    assert not any(name.startswith("H") for pair in actual for name in pair)


def test_restraint_bonds_for_unknown_code_raise():
    with pytest.raises(ValueError, match="XYZ"):
        read_restraint_bonds(X00242_RESTRAINT, "XYZ")


def test_ligand_table_resolves_restraint_paths(tmp_path):
    table = tmp_path / "ligands.csv"
    table.write_text(
        "structure_id,res_name,smiles,restraint_cif\n"
        f"x00242-1,LIG,{ACID_SMILES},restraints/EE_1_L17.cif\n"
        f"x00358-1,LIG,{INDOLE_SMILES},\n"
    )

    actual = read_ligand_table(table)

    expected = [
        LigandRow("x00242-1", "LIG", ACID_SMILES, tmp_path / "restraints" / "EE_1_L17.cif"),
        LigandRow("x00358-1", "LIG", INDOLE_SMILES, None),
    ]
    assert actual == expected


def test_ligand_table_restraint_column_is_optional(tmp_path):
    table = tmp_path / "ligands.csv"
    table.write_text(f"structure_id,res_name,smiles\nx00242-1,LIG,{ACID_SMILES}\n")

    actual = read_ligand_table(table)

    expected = [LigandRow("x00242-1", "LIG", ACID_SMILES, None)]
    assert actual == expected


def test_ligand_table_missing_column_names_it(tmp_path):
    table = tmp_path / "ligands.csv"
    table.write_text("structure_id,res_name\nx00242-1,LIG\n")

    with pytest.raises(ValueError, match="smiles"):
        read_ligand_table(table)


def test_restraint_self_pair_raises_match_error(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)
    restraint_bonds = [*read_restraint_bonds(X00242_RESTRAINT, "LIG"), ("C1", "C1")]

    with pytest.raises(LigandMatchError, match="C1"):
        ligand_definition(ACID_SMILES, names, elements, coords, restraint_bonds)


def test_unrecognised_element_raises_match_error(acid_atoms):
    names, elements, coords = _inputs(acid_atoms)
    elements[0] = "Qq"

    with pytest.raises(LigandMatchError, match="Qq"):
        ligand_definition(ACID_SMILES, names, elements, coords, None)


def test_ligand_table_short_row_names_line_and_column(tmp_path):
    table = tmp_path / "ligands.csv"
    table.write_text(f"structure_id,res_name,smiles\nx00242-1,LIG,{ACID_SMILES}\nx00358-1\n")

    with pytest.raises(ValueError, match=r"line 3.*res_name"):
        read_ligand_table(table)


def test_ligand_table_empty_smiles_names_line_and_column(tmp_path):
    table = tmp_path / "ligands.csv"
    table.write_text("structure_id,res_name,smiles\nx00242-1,LIG,\n")

    with pytest.raises(ValueError, match=r"line 2.*smiles"):
        read_ligand_table(table)


# --- residue names and the placeholder guard ---------------------------------------------


def _ccd(entries: dict[str, list[str]]) -> CIFFile:
    """A tiny CCD: component id -> atom names."""
    ccd = CIFFile()
    for code, atom_names in entries.items():
        block = CIFBlock()
        block["chem_comp"] = CIFCategory({"id": [code]})
        block["chem_comp_atom"] = CIFCategory({"atom_id": atom_names})
        ccd[code] = block
    return ccd


def _residues(*residues: tuple[str, list[str]]) -> AtomArray:
    """A structure of non-polymer residues, one per (res_name, atom names)."""
    atoms = []
    for res_id, (res_name, names) in enumerate(residues, start=1):
        for name in names:
            atoms.append(
                Atom(
                    [0.0, 0.0, 0.0],
                    chain_id="A",
                    res_id=res_id,
                    res_name=res_name,
                    atom_name=name,
                    element=name[0],
                    hetero=True,
                )
            )
    return array(atoms)


@pytest.fixture
def acid_definition(acid_atoms) -> AtomArray:
    names, elements, coords = _inputs(acid_atoms)
    return ligand_definition(ACID_SMILES, names, elements, coords, None)


def test_same_molecule_same_atom_names_share_a_code(acid_definition):
    namer = LigandNamer(ccd_codes=frozenset({"DMS"}))

    first = namer.name_for(acid_definition, ACID_SMILES)
    second = namer.name_for(acid_definition, ACID_SMILES)

    assert second == first


def test_same_molecule_different_atom_names_get_different_codes(acid_definition):
    namer = LigandNamer(ccd_codes=frozenset())
    renamed = acid_definition.copy()
    renamed.atom_name = [f"Z{i}" for i in range(renamed.array_length())]

    first = namer.name_for(acid_definition, ACID_SMILES)
    second = namer.name_for(renamed, ACID_SMILES)

    assert second != first


def test_different_molecules_get_different_codes(acid_definition, indole_atoms):
    namer = LigandNamer(ccd_codes=frozenset())
    names, elements, coords = _inputs(indole_atoms)
    indole = ligand_definition(INDOLE_SMILES, names, elements, coords, None)

    acid_code = namer.name_for(acid_definition, ACID_SMILES)
    indole_code = namer.name_for(indole, INDOLE_SMILES)

    assert indole_code != acid_code


def test_codes_are_short_uppercase_alphanumeric_and_deterministic(acid_definition):
    code = LigandNamer(ccd_codes=frozenset()).name_for(acid_definition, ACID_SMILES)
    again = LigandNamer(ccd_codes=frozenset()).name_for(acid_definition, ACID_SMILES)

    assert len(code) <= 5
    assert code.isalnum() and code == code.upper()
    assert again == code


def test_code_is_independent_of_naming_order(acid_definition, indole_atoms):
    names, elements, coords = _inputs(indole_atoms)
    indole = ligand_definition(INDOLE_SMILES, names, elements, coords, None)
    forward = LigandNamer(ccd_codes=frozenset())
    backward = LigandNamer(ccd_codes=frozenset())

    forward_codes = [
        forward.name_for(acid_definition, ACID_SMILES),
        forward.name_for(indole, INDOLE_SMILES),
    ]
    backward_indole = backward.name_for(indole, INDOLE_SMILES)
    backward_acid = backward.name_for(acid_definition, ACID_SMILES)

    assert forward_codes == [backward_acid, backward_indole]


def test_other_carboxylate_oxygen_double_bonded_gets_a_different_code(acid_definition):
    namer = LigandNamer(ccd_codes=frozenset())
    swapped = acid_definition.copy()
    bonds = swapped.bonds.as_array().copy()
    names = [str(n) for n in swapped.atom_name]
    carbon = next(i for i, n in enumerate(names) if swapped.element[i] == "C"
                  and sum(swapped.element[j] == "O" for j in swapped.bonds.get_bonds(i)[0]) == 2)
    for row in bonds:
        if carbon in row[:2] and swapped.element[row[0] if row[1] == carbon else row[1]] == "O":
            row[2] = BondType.SINGLE if row[2] == BondType.DOUBLE else BondType.DOUBLE
    swapped.bonds = BondList(swapped.array_length(), bonds)

    first = namer.name_for(acid_definition, ACID_SMILES)
    second = namer.name_for(swapped, ACID_SMILES)

    assert second != first


def test_in_run_hash_collision_probes_to_the_next_code(acid_definition, indole_atoms, monkeypatch):
    names, elements, coords = _inputs(indole_atoms)
    indole = ligand_definition(INDOLE_SMILES, names, elements, coords, None)
    monkeypatch.setattr(ligand_inputs, "_stable_hash", lambda key: 0)
    namer = LigandNamer(ccd_codes=frozenset())

    first = namer.name_for(acid_definition, ACID_SMILES)
    second = namer.name_for(indole, INDOLE_SMILES)

    assert (first, second) == ("L0000", "L0001")


def test_code_that_is_a_ccd_code_is_an_error(acid_definition):
    first_code = LigandNamer(ccd_codes=frozenset()).name_for(acid_definition, ACID_SMILES)
    namer = LigandNamer(ccd_codes=frozenset({first_code}))

    with pytest.raises(ValueError, match=first_code):
        namer.name_for(acid_definition, ACID_SMILES)


def test_records_list_smiles_atom_names_and_structures(acid_definition):
    namer = LigandNamer(ccd_codes=frozenset())
    code = namer.name_for(acid_definition, ACID_SMILES)
    namer.add_structure(code, "x00242-1")
    namer.add_structure(code, "x00242-2")

    actual = namer.records()

    expected = {
        code: {
            "smiles": Chem.CanonSmiles(ACID_SMILES),
            "atom_names": [str(n) for n in acid_definition.atom_name],
            "structures": ["x00242-1", "x00242-2"],
        }
    }
    assert actual == expected


def test_ccd_codes_reads_chem_comp_ids():
    ccd = _ccd({"DMS": ["S", "O", "C1", "C2"], "LIG": ["C1"]})

    actual = ccd_codes(ccd)

    assert actual == frozenset({"DMS", "LIG"})


DMS_ATOMS = ["S", "O", "C1", "C2"]


def test_listed_lig_is_not_flagged():
    ccd = _ccd({"LIG": ["C1"]})
    atoms = _residues(("LIG", ["Q1", "Q2"]))

    assert unlisted_placeholders(atoms, listed={"LIG"}, ccd=ccd) == []


def test_unlisted_lig_is_flagged():
    ccd = _ccd({"LIG": ["C1"]})
    atoms = _residues(("LIG", ["C1"]))

    assert unlisted_placeholders(atoms, listed=set(), ccd=ccd) == ["LIG"]


def test_ccd_ligand_with_matching_atoms_is_not_flagged():
    ccd = _ccd({"DMS": DMS_ATOMS})
    atoms = _residues(("DMS", ["S", "C1", "C2"]))

    assert unlisted_placeholders(atoms, listed=set(), ccd=ccd) == []


def test_ccd_name_with_foreign_atom_names_is_flagged():
    ccd = _ccd({"DMS": DMS_ATOMS})
    atoms = _residues(("DMS", ["S", "C1", "N9"]))

    assert unlisted_placeholders(atoms, listed=set(), ccd=ccd) == ["DMS"]


def test_name_missing_from_the_ccd_is_flagged():
    atoms = _residues(("ZZZ", ["C1"]))

    assert unlisted_placeholders(atoms, listed=set(), ccd=_ccd({})) == ["ZZZ"]


def test_water_and_polymer_residues_are_ignored():
    ccd = _ccd({})
    atoms = _residues(("HOH", ["O"]), ("ALA", ["N", "CA", "C", "O"]), ("LIG", ["C1"]))

    actual = unlisted_placeholders(atoms, listed={"LIG"}, ccd=ccd)

    assert actual == []
