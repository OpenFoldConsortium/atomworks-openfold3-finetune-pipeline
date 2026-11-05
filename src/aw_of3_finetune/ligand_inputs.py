"""Ligand definitions for in-house structures: bond orders from a SMILES, atom names from the input.

The SMILES is the only source of bond orders; a restraint CIF (or the geometry) gives
connectivity. A SMILES that cannot be matched onto the atoms is an error, never a fallback.
"""

import csv
import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
from biotite.structure import AtomArray, BondList, BondType, filter_amino_acids, filter_nucleotides
from biotite.structure.io.pdbx import CIFFile
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

REQUIRED_TABLE_COLUMNS = ("structure_id", "res_name", "smiles")
PLACEHOLDER_RES_NAME = "LIG"
PLACEHOLDER_NAMES = frozenset({"LIG", "UNL", "UNX", "DRG", "INH", "UNK"})
_HYDROGENS = {"H", "D"}
_WATERS = {"HOH", "DOD", "WAT"}
_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_CODE_SPACE = 36**4

# (RDKit Kekulé order, aromatic flag) -> biotite bond type.
_BOND_TYPES = {
    (Chem.BondType.SINGLE, False): BondType.SINGLE,
    (Chem.BondType.DOUBLE, False): BondType.DOUBLE,
    (Chem.BondType.TRIPLE, False): BondType.TRIPLE,
    (Chem.BondType.SINGLE, True): BondType.AROMATIC_SINGLE,
    (Chem.BondType.DOUBLE, True): BondType.AROMATIC_DOUBLE,
}


class LigandRow(NamedTuple):
    """One ligand-table row: a residue name in one structure and its chemistry."""

    structure_id: str
    res_name: str
    smiles: str
    restraint_cif: Path | None


class LigandMatchError(ValueError):
    """The SMILES cannot be matched onto the ligand's atoms."""


class CcdCodeCollisionError(ValueError):
    """A generated ligand code equals a CCD component id; the whole run must stop."""


def read_ligand_table(path: Path) -> list[LigandRow]:
    """Read the ligand table CSV; relative restraint paths resolve against its directory.

    Args:
        path: CSV with columns structure_id, res_name, smiles and optionally restraint_cif.

    Returns:
        One row per table line.

    Raises:
        ValueError: A required column is missing, or a required cell is empty (the message
            names the line and column).
    """
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in REQUIRED_TABLE_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path}: ligand table is missing required column(s) {missing}")
        rows = []
        for record in reader:
            line = reader.line_num
            values = {}
            for column in REQUIRED_TABLE_COLUMNS:
                value = (record.get(column) or "").strip()
                if not value:
                    raise ValueError(f"{path}: line {line}: column {column!r} is empty or missing")
                values[column] = value
            restraint = (record.get("restraint_cif") or "").strip()
            rows.append(
                LigandRow(
                    **values,
                    restraint_cif=path.parent / restraint if restraint else None,
                )
            )
    return rows


def read_restraint_bonds(path: Path, res_name: str) -> list[tuple[str, str]]:
    """Heavy-atom bonds (by atom name, any bond type) of one component in a restraint CIF.

    Args:
        path: Restraint CIF with chem_comp_atom and chem_comp_bond.
        res_name: The component id (comp_id) to read.

    Returns:
        (atom name, atom name) pairs in file order.

    Raises:
        ValueError: The file has no chem_comp_bond rows for res_name.
    """
    for block in CIFFile.read(path).values():
        if "chem_comp_bond" not in block or "chem_comp_atom" not in block:
            continue
        atom_cat, bond_cat = block["chem_comp_atom"], block["chem_comp_bond"]
        is_atom = atom_cat["comp_id"].as_array() == res_name
        is_bond = bond_cat["comp_id"].as_array() == res_name
        if not is_bond.any():
            continue
        elements = np.char.upper(atom_cat["type_symbol"].as_array()[is_atom])
        names = atom_cat["atom_id"].as_array()[is_atom]
        heavy = set(names[~np.isin(elements, list(_HYDROGENS))])
        pairs = zip(
            bond_cat["atom_id_1"].as_array()[is_bond],
            bond_cat["atom_id_2"].as_array()[is_bond],
            strict=True,
        )
        return [(str(a), str(b)) for a, b in pairs if a in heavy and b in heavy]
    raise ValueError(f"{path}: no chem_comp_bond rows for {res_name!r}")


def _heavy_atom_mol(
    elements: list[str], coords: np.ndarray, bonds: list[tuple[int, int]] | None
) -> Chem.RWMol:
    """Heavy atoms as a molecule with single bonds; connectivity from geometry if bonds is None."""
    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(elements))
    for i, (element, xyz) in enumerate(zip(elements, coords, strict=True)):
        mol.AddAtom(Chem.Atom(element.capitalize()))
        conformer.SetAtomPosition(i, [float(v) for v in xyz])
    mol.AddConformer(conformer, assignId=True)
    if bonds is None:
        rdDetermineBonds.DetermineConnectivity(mol)
    else:
        for i, j in bonds:
            if mol.GetBondBetweenAtoms(i, j) is None:  # restraint files may list a bond twice
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    return mol


def ligand_definition(
    smiles: str,
    atom_names: list[str],
    elements: list[str],
    coords: np.ndarray,
    restraint_bonds: list[tuple[str, str]] | None,
) -> AtomArray:
    """Assign the SMILES' bond orders and charges onto the input's heavy atoms by template match.

    Hydrogens in the input are ignored. Connectivity comes from restraint_bonds when given,
    else from the geometry.

    Args:
        smiles: The ligand's chemistry; one molecule.
        atom_names: Input atom names, hydrogens allowed.
        elements: Input elements, same order.
        coords: (n, 3) input coordinates, same order.
        restraint_bonds: Heavy-atom bonds by atom name (any order), or None.

    Returns:
        Heavy atoms in input order with Kekulé bonds (aromatic flag kept), charge, atom_name,
        element, res_name placeholder, hetero=True and leaving_atom_flag=False.

    Raises:
        LigandMatchError: The SMILES is unparsable, has several molecules, or does not match.
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise LigandMatchError(f"could not parse SMILES {smiles!r}")
    if len(Chem.GetMolFrags(template)) > 1:
        raise LigandMatchError(f"SMILES {smiles!r} has more than one molecule")

    upper = [element.upper() for element in elements]
    heavy = [i for i, element in enumerate(upper) if element not in _HYDROGENS]
    heavy_names = [atom_names[i] for i in heavy]
    heavy_elements = [upper[i] for i in heavy]
    heavy_coords = np.asarray(coords, dtype=float)[heavy]
    if template.GetNumAtoms() != len(heavy):
        raise LigandMatchError(
            f"SMILES {smiles!r} has {template.GetNumAtoms()} heavy atoms, "
            f"the ligand has {len(heavy)}"
        )

    bond_indices = None
    if restraint_bonds is not None:
        index = {name: i for i, name in enumerate(heavy_names)}
        unknown = sorted({name for pair in restraint_bonds for name in pair} - set(index))
        if unknown:
            raise LigandMatchError(f"restraint bonds name atoms not in the ligand: {unknown}")
        self_pairs = sorted({a for a, b in restraint_bonds if a == b})
        if self_pairs:
            raise LigandMatchError(f"restraint bonds join atoms to themselves: {self_pairs}")
        bond_indices = [(index[a], index[b]) for a, b in restraint_bonds]

    try:
        mol = _heavy_atom_mol(heavy_elements, heavy_coords, bond_indices)
    except Exception as error:  # RDKit raises RuntimeError for bad elements or bonds
        raise LigandMatchError(f"cannot build the ligand's molecule: {error}") from error
    if mol.GetNumBonds() != template.GetNumBonds():
        raise LigandMatchError(
            f"SMILES {smiles!r} does not match: it has {template.GetNumBonds()} bonds, "
            f"the ligand's connectivity has {mol.GetNumBonds()}"
        )
    try:
        assigned = AllChem.AssignBondOrdersFromTemplate(template, mol)
        Chem.Kekulize(assigned, clearAromaticFlags=False)
    except Exception as error:  # RDKit raises ValueError or sanitization errors
        raise LigandMatchError(f"SMILES {smiles!r} does not match the ligand: {error}") from error

    bonds = []
    for bond in assigned.GetBonds():
        key = (bond.GetBondType(), bond.GetIsAromatic())
        if key not in _BOND_TYPES:
            raise LigandMatchError(f"SMILES {smiles!r}: unsupported bond type {key}")
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _BOND_TYPES[key]))

    atoms = AtomArray(len(heavy))
    atoms.coord = heavy_coords
    atoms.atom_name = heavy_names
    atoms.element = heavy_elements
    atoms.res_name[:] = PLACEHOLDER_RES_NAME
    atoms.hetero[:] = True
    atoms.set_annotation("charge", [atom.GetFormalCharge() for atom in assigned.GetAtoms()])
    atoms.set_annotation("leaving_atom_flag", np.zeros(len(heavy), dtype=bool))
    atoms.bonds = BondList(len(heavy), np.array(bonds, dtype=np.int64).reshape(-1, 3))
    return atoms


def ccd_codes(ccd: CIFFile) -> frozenset[str]:
    """All component ids in a CCD.

    Args:
        ccd: Parsed Chemical Component Dictionary.

    Returns:
        The ids (chem_comp.id) of its entries.
    """
    return frozenset(str(code) for code in ccd.keys())


def _stable_hash(key: tuple) -> int:
    """A hash of the key that does not change between runs (unlike the builtin)."""
    return int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "big")


def _code_from_number(number: int) -> str:
    return "L" + "".join(_BASE36[(number // 36**i) % 36] for i in (3, 2, 1, 0))


def _definition_key(definition: AtomArray, smiles: str) -> tuple:
    """Canonical SMILES, atom names, charges and named bonds: what makes two definitions equal."""
    names = [str(name) for name in definition.atom_name]
    charges = tuple(int(c) for c in definition.charge)
    bonds = sorted(
        (*sorted((names[i], names[j])), int(order))
        for i, j, order in definition.bonds.as_array()
    )
    return (Chem.CanonSmiles(smiles), tuple(names), charges, tuple(bonds))


class LigandNamer:
    """Gives each distinct custom ligand a residue name that is not in the CCD.

    A code is "L" plus four base-36 digits from a SHA-256 of the ligand's SMILES, atom names,
    charges and bonds, so it is stable across reruns and input order, and two datasets get
    different codes for different molecules except on a hash collision. A code already used
    by a different ligand in this run moves to the next value.
    """

    def __init__(self, ccd_codes: frozenset[str]):
        """Start with no ligands.

        Args:
            ccd_codes: Component ids that a code must never equal.
        """
        self._ccd_codes = ccd_codes
        self._codes: dict[tuple, str] = {}
        self._records: dict[str, dict] = {}

    def name_for(self, definition: AtomArray, smiles: str) -> str:
        """The code for this ligand; identical SMILES, names, charges and bonds share one.

        Args:
            definition: Output of ligand_definition.
            smiles: The ligand's SMILES.

        Returns:
            The residue name.

        Raises:
            CcdCodeCollisionError: The code is a CCD component id.
        ValueError: All codes are in use.
        """
        key = _definition_key(definition, smiles)
        if key in self._codes:
            return self._codes[key]
        start = _stable_hash(key) % _CODE_SPACE
        for step in range(_CODE_SPACE):
            code = _code_from_number((start + step) % _CODE_SPACE)
            if code in self._ccd_codes:
                raise CcdCodeCollisionError(
                    f"generated ligand code {code!r} is a CCD component id"
                )
            if code not in self._records:
                break
        else:
            raise ValueError("all ligand codes are in use")
        self._codes[key] = code
        self._records[code] = {
            "smiles": key[0],
            "atom_names": list(key[1]),
            "structures": [],
        }
        return code

    def add_structure(self, code: str, structure_id: str) -> None:
        """Record that a structure uses a code.

        Args:
            code: A code from name_for.
            structure_id: The structure that has this ligand.
        """
        structures = self._records[code]["structures"]
        if structure_id not in structures:
            structures.append(structure_id)

    def records(self) -> dict[str, dict]:
        """Every code with its SMILES, atom names and structures.

        Returns:
            {code: {"smiles", "atom_names", "structures"}}.
        """
        return self._records


def unlisted_placeholders(
    atom_array: AtomArray, listed: set[str], ccd: CIFFile
) -> list[str]:
    """Non-polymer residue names that look like placeholders and are not in the ligand table.

    A name is flagged if it is in PLACEHOLDER_NAMES, missing from the CCD, or its heavy atoms
    are not all atom names of that CCD entry. Waters are ignored.

    Args:
        atom_array: The structure's atoms.
        listed: Residue names the ligand table covers.
        ccd: Parsed CCD.

    Returns:
        Flagged residue names, sorted.
    """
    polymer = filter_amino_acids(atom_array) | filter_nucleotides(atom_array)
    elements = np.char.upper(atom_array.element.astype(str))
    heavy = ~np.isin(elements, list(_HYDROGENS))
    flagged = set()
    for res_name in np.unique(atom_array.res_name[~polymer]):
        name = str(res_name)
        if name in listed or name in _WATERS:
            continue
        if name in PLACEHOLDER_NAMES or name not in ccd:
            flagged.add(name)
            continue
        names = set(
            atom_array.atom_name[heavy & ~polymer & (atom_array.res_name == name)]
        )
        ccd_names = set(ccd[name]["chem_comp_atom"]["atom_id"].as_array())
        if not names <= ccd_names:
            flagged.add(name)
    return sorted(flagged)
