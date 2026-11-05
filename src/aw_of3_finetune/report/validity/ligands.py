"""PoseBusters checks of every ligand chain: chemistry, identity, geometry and protein contacts.

PoseBusters' check_radicals is deliberately not run: the fixed-hydrogen scheme of ligand_mol gives each
atom the CCD's non-leaving hydrogens, so a free hydroxyl whose H is a CCD leaving atom (every glycan's
O1) is under-filled and would read as a radical.

Metals (decided 2026-10-28): chemistry runs RDKit's organometallic cleanup first (metal bonds to dative,
as pdbeccdutils does for training); identity compares InChI with metals disconnected, since RDKit cannot
write InChI for dative bonds (rdkit#6853); bond angles centred on a metal are skipped (RDKit's bounds do
not model e.g. square-planar Fe, so even the CCD's HEM fails); metal atoms are judged on covalent radii
for protein contacts, as PoseBusters does for metal cofactors, and left out of the volume overlap.
"""

import math

import numpy as np
from biotite.structure import AtomArray
from posebusters.modules.distance_geometry import check_geometry
from posebusters.modules.flatness import check_flatness
from posebusters.modules.identity import check_identity
from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from posebusters.modules.sanity import check_chemistry
from posebusters.modules.volume_overlap import check_volume_overlap
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Geometry import Point3D

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.bonds import _ccd_bonds
from aw_of3_finetune.report.validity.copies import find_copies, operators
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.steplog import read_input_atoms

LIGAND_CHEMISTRY = "validity.ligand_chemistry"
LIGAND_IDENTITY = "validity.ligand_identity"
LIGAND_GEOMETRY = "validity.ligand_geometry"
LIGAND_CONTACTS = "validity.ligand_contacts"

# Parameters pinned from PoseBusters 0.6.5 config/dock.yml, named after their keys.
BOUND_MATRIX_PARAMS = {"set15bounds": True, "scaleVDW": True, "doTriangleSmoothing": True,
                       "useMacrocycle14config": False}
THRESHOLD_BAD_BOND_LENGTH = 0.25
THRESHOLD_BAD_ANGLE = 0.25
THRESHOLD_CLASH = 0.3
IGNORE_HYDROGENS = True
SANITIZE = True
FLAT_AROMATIC_RINGS = {"aromatic_5_membered_rings_sp2": "[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1",
                       "aromatic_6_membered_rings_sp2": "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1"}
THRESHOLD_FLATNESS_AROMATIC = 0.25
_NONAROMATIC = "[C,O,S,N;R1]~1[C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1][C,O,S,N;R1]1"
NONFLAT_RINGS = {
    "non-aromatic_6_membered_rings": _NONAROMATIC,
    "non-aromatic_6_membered_rings_db03_0": "[C;R1]~1[C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1][C;R1]1",
    "non-aromatic_6_membered_rings_db03_1": "[C;R1]~1[C;R1][C;R1]~[C;R1][C,O,S,N;R1][C;R1]1",
    "non-aromatic_6_membered_rings_db02_0": "[C;R1]~1[C;R1][C;R1][C,O,S,N;R1]~[C,O,S,N;R1][C;R1]1",
    "non-aromatic_6_membered_rings_db02_1": "[C;R1]~1[C;R1][C,O,S,N;R1][C;R1]~[C;R1][C;R1]1",
}
THRESHOLD_FLATNESS_NONFLAT = 0.05
FLAT_DOUBLE_BONDS = {"trigonal_planar_double_bonds": "[C;X3;^2](*)(*)=[C;X3;^2](*)(*)"}
THRESHOLD_FLATNESS_DOUBLE_BOND = 0.25
RADIUS_TYPE = "vdw"
RADIUS_SCALE = 1.0
CLASH_CUTOFF_DISTANCE = 0.75
MAX_DISTANCE = 5.0
CLASH_CUTOFF_OVERLAP = 0.075
VDW_SCALE = 0.8
IGNORE_TYPES = {"hydrogens", "organic_cofactors", "inorganic_cofactors", "waters"}
METAL_RADIUS_TYPE = "covalent"  # dock.yml's "Distance to inorganic cofactors"
NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "As", "Se", "Br", "Kr",
             "Te", "I", "Xe", "At", "Rn"}
COULD_NOT_RUN = "test could not run"  # row reason when PoseBusters returned NaN or None for a pass flag
PROTEIN_RADIUS = 10.0  # Å around the ligand within which protein atoms are compared

# Biotite bond type -> RDKit; aromatic as Kekulé, aromatic triple as triple, coordination and generic aromatic unspecified.
_BOND_TYPES = {0: Chem.BondType.SINGLE, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
               3: Chem.BondType.TRIPLE, 4: Chem.BondType.QUADRUPLE, 5: Chem.BondType.SINGLE,
               6: Chem.BondType.DOUBLE, 7: Chem.BondType.TRIPLE, 8: Chem.BondType.UNSPECIFIED,
               9: Chem.BondType.UNSPECIFIED}
_CHEMISTRY_TESTS = {"passes_rdkit_sanity_checks": "sanitization", "all_atoms_connected": "all atoms connected"}
# InChI layers; the stereo ones are compared only when the CCD gives coordinates to take stereo from.
_IDENTITY_TESTS = {"inchi_crystal_valid": "CCD InChI valid", "inchi_docked_valid": "structure InChI valid",
                   "formula": "formula", "connections": "connections", "hydrogens": "hydrogens",
                   "net_charge": "net charge", "protons": "protons"}
_STEREO_TESTS = {"stereo_dbond": "double bond stereo", "stereo_tetrahedral": "tetrahedral stereo", "stereo": "stereo"}
_INPUT_STEREO_TESTS = {key: f"input {name}" for key, name in _STEREO_TESTS.items()}
CCD_REFERENCE, INPUT_REFERENCE = "ccd", "input"  # identity row "reference": what the ligand is compared to
_ATOM_KEY = ("label_asym_id", "res_id", "ins_code", "res_name", "atom_name", "label_alt_id")  # one input atom
_GEOMETRY_TESTS = {"bond_lengths_within_bounds": "bond lengths", "bond_angles_within_bounds": "bond angles",
                   "no_internal_clash": "internal clash", "aromatic ring flatness": "aromatic ring flatness",
                   "non-aromatic ring non-flatness": "non-aromatic ring non-flatness",
                   "double bond flatness": "double bond flatness"}
_CONTACT_TESTS = {"not_too_far_away": "protein-ligand maximum distance", "no_clashes": "minimum distance to protein",
                  "no_volume_clash": "volume overlap with protein"}


def _ccd_coordinates(table, names: np.ndarray) -> dict[str, tuple | None]:
    """Ideal (else model) coordinates of CCD atoms by name; None where the CCD has none."""
    for prefix, suffix in (("pdbx_model_Cartn_", "_ideal"), ("model_Cartn_", "")):
        keys = [f"{prefix}{axis}{suffix}" for axis in "xyz"]
        if all(key in table for key in keys):
            columns = [table[key].as_array(str) for key in keys]
            try:
                values = {name: tuple(float(c) for c in xyz) for name, *xyz in zip(names, *columns)}
            except ValueError:
                continue
            return values
    return dict.fromkeys(names)


def _ccd_charge(value: str, element: str, hydrogens: int) -> int:
    """A CCD atom's formal charge, tolerating two things wwPDB's 2026 entries do (HEM among them).

    Args:
        value: The charge field; ``?`` and ``.`` mean unknown, which counts as neutral.
        element: The atom's element.
        hydrogens: Non-leaving hydrogens bonded to the atom.

    Returns:
        The charge; a negative oxygen that keeps a hydrogen counts as neutral, as O- with two
        bonds is not valid.
    """
    if value in ("?", "."):
        return 0
    charge = int(value)
    return 0 if element == "O" and charge < 0 and hydrogens > 0 else charge


def _ccd_atoms(inputs: ValidityInputs, res_name: str) -> dict[str, tuple[str, int, bool, int, tuple | None]] | None:
    """CCD heavy atoms of a residue as name -> (element, charge, leaving, hydrogens, coordinates); None if not in CCD.

    Hydrogens are those bonded and not leaving; coordinates are the ideal ones, else the model ones, else None.
    """
    cache = inputs._cache.setdefault("ccd_atoms", {})
    if res_name not in cache:
        ccd = inputs.ccd()
        if res_name not in ccd:
            cache[res_name] = None
        else:
            table = ccd[res_name]["chem_comp_atom"]
            columns = [table[c].as_array(str) for c in
                       ("atom_id", "type_symbol", "charge", "pdbx_leaving_atom_flag")]
            leaving_flags = dict(zip(columns[0], columns[3]))
            hydrogens = {name for name, element in zip(columns[0], columns[1])
                         if element == "H" and leaving_flags[name] != "Y"}
            coordinates = _ccd_coordinates(table, columns[0])
            counts = dict.fromkeys(columns[0], 0)
            for pair in _ccd_bonds(inputs, res_name):
                for atom, other in (pair, pair[::-1]):
                    counts[atom] += other in hydrogens
            cache[res_name] = {name: (element.capitalize(), _ccd_charge(charge, element, counts[name]),
                                      leaving == "Y", counts[name],
                                      coordinates[name])
                               for name, element, charge, leaving in zip(*columns) if element != "H"}
    return cache[res_name]


def _to_mol(atoms: AtomArray, indices: np.ndarray, inputs: ValidityInputs, fixed: np.ndarray) -> Chem.Mol:
    """Unsanitized molecule of the given atoms with the NPZ bonds among them and a conformer (NaN as 0).

    Formal charges come from the CCD. Atoms whose index is in fixed get the CCD's hydrogen count, so a
    bond the CCD does not have (a retained leaving atom) over-fills the valence; the others get implicit hydrogens.
    """
    position = {int(index): i for i, index in enumerate(indices)}
    editable = Chem.RWMol()
    conformer = Chem.Conformer(len(indices))
    for i, index in enumerate(indices):
        atom = Chem.Atom(str(atoms.element[index]).capitalize())
        ccd_atoms = _ccd_atoms(inputs, str(atoms.res_name[index])) or {}
        _, charge, _, hydrogens, _ = ccd_atoms.get(str(atoms.atom_name[index]), ("", 0, False, None, None))
        atom.SetFormalCharge(charge)
        if index in fixed and hydrogens is not None:
            atom.SetNumExplicitHs(hydrogens)
            atom.SetNoImplicit(True)
        editable.AddAtom(atom)
        x, y, z = np.nan_to_num(atoms.coord[index])
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    for i, j, order in atoms.bonds.as_array():
        if int(i) in position and int(j) in position:
            editable.AddBond(position[int(i)], position[int(j)], _BOND_TYPES[int(order)])
    mol = editable.GetMol()
    mol.AddConformer(conformer, assignId=True)
    return mol


def _is_metal(atom: Chem.Atom) -> bool:
    """Whether an atom is a metal (any element outside NONMETALS)."""
    return atom.GetSymbol() not in NONMETALS


def _has_metal(mol: Chem.Mol) -> bool:
    """Whether a molecule has a metal atom."""
    return any(_is_metal(atom) for atom in mol.GetAtoms())


def _cleaned_up(mol: Chem.Mol) -> Chem.Mol:
    """A copy with RDKit's organometallic cleanup applied: metal bonds that over-fill a donor become dative."""
    mol = Chem.Mol(mol)
    Chem.CleanupOrganometallics(mol)
    return mol


def _metal_disconnected(mol: Chem.Mol) -> Chem.Mol:
    """A sanitized copy with bonds to metals removed (RDKit's MetalDisconnector), for InChI comparison."""
    mol = _cleaned_up(mol)
    Chem.SanitizeMol(mol)
    return rdMolStandardize.MetalDisconnector().Disconnect(mol)


def _atom_subset(mol: Chem.Mol, keep: list[int]) -> Chem.Mol:
    """A copy with only the atoms at the keep indices (bonds among them and the conformer kept)."""
    editable = Chem.RWMol(mol)
    for index in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
        editable.RemoveAtom(index)
    return editable.GetMol()


def _partner_indices(atoms: AtomArray, chain_id: str) -> np.ndarray:
    """Indices of atoms of other chains that are bonded to atoms of the chain."""
    in_chain = atoms.chain_id.astype(str) == chain_id
    partners = set()
    for i, j, _ in atoms.bonds.as_array():
        if in_chain[i] != in_chain[j]:
            partners.add(int(j) if in_chain[i] else int(i))
    return np.array(sorted(partners), dtype=int)


def ligand_mol(final_atoms: AtomArray, chain_id: str, with_partners: bool, inputs: ValidityInputs) -> Chem.Mol:
    """Molecule of one ligand chain from the NPZ atoms and bonds, with CCD formal charges.

    Args:
        final_atoms: The pipeline's final atoms.
        chain_id: Chain id of the ligand.
        with_partners: True: all chain atoms (resolved or not, as coordinates are not needed to judge valence)
            plus bonded atoms of other chains; chain atoms get the CCD's hydrogen count. False: only the resolved chain atoms.
        inputs: Gives the CCD for the charges.

    Returns:
        The unsanitized molecule, one conformer.
    """
    in_chain = final_atoms.chain_id.astype(str) == chain_id
    if with_partners:
        indices = np.concatenate([np.flatnonzero(in_chain), _partner_indices(final_atoms, chain_id)])
    else:
        indices = np.flatnonzero(in_chain & ~np.isnan(final_atoms.coord).any(axis=1))
    return _to_mol(final_atoms, indices, inputs, fixed=np.flatnonzero(in_chain) if with_partners else indices[:0])


def _ligand_chains(inputs: ValidityInputs, pdb_id: str) -> list[str]:
    """Chain ids of the ligand chains of an entry, in cache order."""
    chains = inputs.cache_entry(pdb_id)["chains"]
    return [chain for chain, record in chains.items() if record["molecule_type"] == "LIGAND"]


def _label(atoms: AtomArray, chain_id: str) -> tuple[str, str]:
    """Residue names joined by '+' and the finding label '<RES> chain <id>'."""
    names = list(dict.fromkeys(atoms.res_name[atoms.chain_id.astype(str) == chain_id].astype(str)))
    res_name = "+".join(names)
    return res_name, f"{res_name} chain {chain_id}"


def _number(value) -> float | None:
    """A finite float, else None (NaN, pd.NA, None and non-numbers)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rows(
    chain: str, res_name: str, results: dict, tests: dict[str, str], values: dict[str, tuple] | None = None
) -> list[dict]:
    """One row per wanted PoseBusters result key, with its measured value and threshold (each a number or None).

    values maps a result key to (measured value, threshold).
    """
    values = values or {}
    rows = []
    for key, name in tests.items():
        if key in results:
            value, threshold = values.get(key, (None, None))
            flag = results[key]
            could_not_run = _number(flag) is None and not isinstance(flag, (bool, np.bool_))
            row = {"chain": chain, "res_name": res_name, "test": name, "passed": False if could_not_run else bool(flag),
                   "value": _number(value), "threshold": _number(threshold)}
            if could_not_run:
                row |= {"value": None, "label": "unexplained", "reason": COULD_NOT_RUN}
            rows.append(row)
    return rows


def _ratio(value: float) -> str:
    """A ratio as "1.55×"."""
    return f"{value:.2f}×"


# Per failed test: what is wrong, then (value, limit) wording when the row has both.
_DESCRIPTIONS = {
    "sanitization": ("RDKit cannot sanitize it (unusual valence or charge)", None),
    "all atoms connected": ("not all atoms are bonded together", None),
    "bond angles": ("an angle outside its range", lambda v, t: f"{_ratio(v)} the bound, limit {_ratio(t)}"),
    "internal clash": ("atoms inside the ligand too close", lambda v, t: f"{_ratio(v)} their expected distance, limit {_ratio(t)}"),
    "aromatic ring flatness": ("aromatic ring not flat", lambda v, t: f"{v:.2f} Å out of plane, limit {t:.2f} Å"),
    "non-aromatic ring non-flatness": ("non-aromatic ring flat", lambda v, t: f"{v:.2f} Å out of plane, at least {t:.2f} Å expected"),
    "double bond flatness": ("double bond not planar", lambda v, t: f"{v:.2f} Å out of plane, limit {t:.2f} Å"),
    "protein-ligand maximum distance": ("too far from the protein", lambda v, t: f"{v:.1f} Å, limit {t:.1f} Å"),
    "minimum distance to protein": ("too close to the protein", lambda v, t: f"{_ratio(v)} the van der Waals distance, limit {_ratio(t)}"),
    "volume overlap with protein": ("overlaps the protein", lambda v, t: f"{v:.0%} of its volume, limit {t:.0%}"),
    "CCD InChI valid": ("no valid InChI from the CCD", None),
    "structure InChI valid": ("no valid InChI from the structure", None),
}
_REFERENCE_NAMES = {CCD_REFERENCE: "the CCD", INPUT_REFERENCE: "the input ligand"}


def describe_row(row: dict) -> str:
    """What a failed ligand check row found wrong, with its value and limit when it has them.

    Args:
        row: A failed row of a ligand finding.

    Returns:
        A short description, e.g. "a bond too long (1.55× its upper bound, limit 1.25×)".
    """
    test, value, threshold = row["test"], row.get("value"), row.get("threshold")
    if row.get("reason") == COULD_NOT_RUN:
        return f"{test} could not run"
    if test == "bond lengths":
        if value is None or threshold is None:
            return "a bond length out of bounds"
        side = "long" if value >= 1 else "short"
        bound = "upper" if value >= 1 else "lower"
        return f"a bond too {side} ({_ratio(value)} its {bound} bound, limit {_ratio(threshold)})"
    if test in _DESCRIPTIONS:
        text, numbers = _DESCRIPTIONS[test]
        return f"{text} ({numbers(value, threshold)})" if numbers and value is not None and threshold is not None else text
    reference = _REFERENCE_NAMES.get(row.get("reference"))
    return f"{test} differs from {reference}" if reference else f"{test} failed"


def _finding(check: str, pdb_id: str, label: str, rows: list[dict], skipped: str | None = None) -> Finding:
    """One finding for a ligand chain: unexplained if any row failed."""
    failed = [describe_row(row) for row in rows if not row["passed"]]
    data = {"rows": rows}
    if skipped:
        data["skipped"] = skipped
        return Finding(check, pdb_id, Status.MATCH, f"{label}: skipped ({skipped})", data)
    status = Status.UNEXPLAINED if failed else Status.MATCH
    detail = f"{label}: {'; '.join(failed)}" if failed else f"{label}: passed"
    return Finding(check, pdb_id, status, detail, data)


def check_ligand_chemistry(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """RDKit sanitization (after organometallic cleanup) and connectedness of each ligand chain, covalent partners included.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding per ligand chain.
    """
    atoms = inputs.final_atoms(pdb_id)
    findings = []
    for chain in _ligand_chains(inputs, pdb_id):
        res_name, label = _label(atoms, chain)
        mol = _cleaned_up(ligand_mol(atoms, chain, True, inputs))
        results = check_chemistry(mol)["results"]
        findings.append(_finding(LIGAND_CHEMISTRY, pdb_id, label, _rows(chain, res_name, results, _CHEMISTRY_TESTS)))
    return findings


def _ccd_mol(inputs: ValidityInputs, res_name: str, keep: set[str]) -> Chem.Mol:
    """The CCD molecule of a residue restricted to the heavy atoms named in keep with a conformer from the CCD coordinates if it has them for all of them."""
    ccd_atoms = _ccd_atoms(inputs, res_name)
    names = [name for name in ccd_atoms if name in keep]
    editable = Chem.RWMol()
    for name in names:
        atom = Chem.Atom(ccd_atoms[name][0])
        atom.SetFormalCharge(ccd_atoms[name][1])
        editable.AddAtom(atom)
    for (name_1, name_2), order in _ccd_bonds(inputs, res_name).items():
        if name_1 in names and name_2 in names:
            editable.AddBond(names.index(name_1), names.index(name_2), _BOND_TYPES[order])
    mol = editable.GetMol()
    coordinates = [ccd_atoms[name][4] for name in names]
    if names and all(xyz is not None for xyz in coordinates):
        conformer = Chem.Conformer(len(names))
        for i, xyz in enumerate(coordinates):
            conformer.SetAtomPosition(i, Point3D(*xyz))
        mol.AddConformer(conformer, assignId=True)
    return mol


def _input_match(inputs: ValidityInputs, pdb_id: str) -> tuple[AtomArray, dict[int, int], dict[tuple, int]]:
    """The input atom table, find_copies' final position -> input index map, and input indices by atom key.

    The atom key is (label_asym_id, res_id, ins_code, res_name, atom_name, label_alt_id); the last entry is cached.
    """
    cached = inputs._cache.get("input_match")
    if cached is None or cached[0] != pdb_id:
        cif_path = inputs.input_cif(pdb_id)
        input_atoms = read_input_atoms(cif_path)
        copies = find_copies(input_atoms, inputs.final_atoms(pdb_id), operators(cif_path))
        mapped = {position: index for index, position in copies.pairs} | copies.source_of
        keys = zip(*(input_atoms.get_annotation(name).astype(str) for name in _ATOM_KEY))
        by_key = {key: index for index, key in enumerate(keys)}
        cached = (pdb_id, input_atoms, mapped, by_key)
        inputs._cache["input_match"] = cached
    return cached[1], cached[2], cached[3]


def _input_atom(input_atoms: AtomArray, by_key: dict[tuple, int], mapped: int, atom_name: str) -> int | None:
    """The input atom named atom_name in the residue (and altloc) of input atom mapped; None if there is none."""
    residue = [str(input_atoms.get_annotation(name)[mapped]) for name in _ATOM_KEY]
    residue[_ATOM_KEY.index("atom_name")] = atom_name
    same_altloc = by_key.get(tuple(residue))
    if same_altloc is not None:
        return same_altloc
    residue[_ATOM_KEY.index("label_alt_id")] = "."  # an atom without altlocs in a residue that has some
    return by_key.get(tuple(residue))


def input_ligand_mol(inputs: ValidityInputs, pdb_id: str, chain_id: str) -> Chem.Mol | None:
    """ligand_mol of a ligand chain with each atom placed at the same-named atom of its input residue.

    find_copies maps each final atom to an input atom (identity copies by coordinates, assembly copies by
    operator); that atom's residue and altloc give the input residue. Stereo is invariant under the
    proper rotations of assembly operators.

    Args:
        inputs: The pipeline output and its input CIFs.
        pdb_id: Structure id.
        chain_id: Chain id of the ligand.

    Returns:
        The unsanitized molecule at input coordinates; None if the input CIF is missing, or an atom has no
        mapped input atom or no same-named atom in that input residue.
    """
    if not inputs.input_cif(pdb_id).exists():
        return None
    final = inputs.final_atoms(pdb_id)
    input_atoms, mapped, by_key = _input_match(inputs, pdb_id)
    mol = ligand_mol(final, chain_id, False, inputs)
    in_chain = final.chain_id.astype(str) == chain_id
    indices = np.flatnonzero(in_chain & ~np.isnan(final.coord).any(axis=1))
    conformer = mol.GetConformer()
    for i, index in enumerate(indices):
        if int(index) not in mapped:
            return None
        source = _input_atom(input_atoms, by_key, mapped[int(index)], str(final.atom_name[index]))
        if source is None:
            return None
        x, y, z = input_atoms.coord[source]
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol


def _input_rows(inputs: ValidityInputs, pdb_id: str, chain: str, res_name: str, actual: Chem.Mol) -> list[dict]:
    """Stereo rows of a ligand chain against its input ligand; one could-not-run row if there is no input match."""
    could_not_run = {"chain": chain, "res_name": res_name, "reference": INPUT_REFERENCE, "passed": None, "value": None,
                     "threshold": None, "label": "unexplained", "reason": COULD_NOT_RUN}
    reference = input_ligand_mol(inputs, pdb_id, chain)
    if reference is None:
        return [could_not_run | {"test": "input match"}]
    if _has_metal(reference):
        actual, reference = _metal_disconnected(actual), _metal_disconnected(reference)
    try:
        results = check_identity(actual, reference)["results"]
    except RuntimeError:  # RDKit stereo perception on degenerate geometry (e.g. a zero-length vector)
        return [could_not_run | {"test": "input stereo"}]
    return [row | {"reference": INPUT_REFERENCE} for row in _rows(chain, res_name, results, _INPUT_STEREO_TESTS)]


def check_ligand_identity(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """InChI identity of each ligand chain against its CCD molecule (absent leaving atoms removed); stereo also vs the input.

    Rows carry "reference": "ccd" for the CCD comparison, "input" for the stereo comparison with the input ligand.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding per ligand chain; skipped for unresolved non-leaving atoms or multi-residue chains.
    """
    atoms = inputs.final_atoms(pdb_id)
    resolved = ~np.isnan(atoms.coord).any(axis=1)
    findings = []
    for chain in _ligand_chains(inputs, pdb_id):
        res_name, label = _label(atoms, chain)
        in_chain = atoms.chain_id.astype(str) == chain
        if "+" in res_name:
            findings.append(_finding(LIGAND_IDENTITY, pdb_id, label, [], skipped="multi-residue ligand"))
            continue
        ccd_atoms = _ccd_atoms(inputs, res_name)
        if ccd_atoms is None:
            row = {"chain": chain, "res_name": res_name, "reference": CCD_REFERENCE, "test": "in CCD", "passed": False,
                   "value": None, "threshold": None}
            findings.append(_finding(LIGAND_IDENTITY, pdb_id, label, [row]))
            findings[-1].detail = f"{label}: not in CCD"
            continue
        present = {str(name) for name in atoms.atom_name[in_chain & resolved]}
        if any(name not in present and not leaving for name, (_, _, leaving, _, _) in ccd_atoms.items()):
            findings.append(_finding(LIGAND_IDENTITY, pdb_id, label, [], skipped="unresolved atoms"))
            continue
        expected = _ccd_mol(inputs, res_name, present)
        actual = ligand_mol(atoms, chain, False, inputs)
        input_rows = _input_rows(inputs, pdb_id, chain, res_name, Chem.Mol(actual))
        has_stereo = expected.GetNumConformers() > 0
        if not has_stereo:
            actual.RemoveAllConformers()
        if _has_metal(expected):
            actual, expected = _metal_disconnected(actual), _metal_disconnected(expected)
        results = check_identity(actual, expected)["results"]
        tests = _IDENTITY_TESTS | (_STEREO_TESTS if has_stereo else {})
        rows = [row | {"reference": CCD_REFERENCE} for row in _rows(chain, res_name, results, tests)]
        finding = _finding(LIGAND_IDENTITY, pdb_id, label, rows + input_rows)
        if not has_stereo:
            finding.data["stereo_skipped"] = True
        findings.append(finding)
    return findings


def _worst_bond_ratio(results: dict) -> tuple[float | None, float]:
    """The bond-length ratio (distance / bound) furthest out of bounds, with the bound it is compared to."""
    shortest = _number(results.get("shortest_bond_relative_length"))
    longest = _number(results.get("longest_bond_relative_length"))
    if shortest is not None and (longest is None or 1 - shortest > longest - 1):
        return shortest, 1 - THRESHOLD_BAD_BOND_LENGTH
    return longest, 1 + THRESHOLD_BAD_BOND_LENGTH


def _angle_results_without_metal_centres(mol: Chem.Mol, details: dict) -> dict:
    """PoseBusters' angle pass flag and most extreme relative angle, recomputed without angles centred on a metal.

    Args:
        mol: The molecule check_geometry ran on.
        details: check_geometry's details (its "angles" table has one row per angle).

    Returns:
        bond_angles_within_bounds and most_extreme_relative_angle, or {} if there is no angle table.
    """
    angles = details.get("angles")
    if angles is None or len(angles) == 0:
        return {}
    angles = angles[[not _is_metal(mol.GetAtomWithIdx(int(triple[1]))) for triple in angles["angle"]]]
    if len(angles) == 0:
        return {"bond_angles_within_bounds": True, "most_extreme_relative_angle": None}
    lower_extreme = 2 - (angles["distance"] / angles["lower_bound"]).min()
    upper_extreme = (angles["distance"] / angles["upper_bound"]).max()
    n_bad = int((angles["bound_absolute_percent_error"] > THRESHOLD_BAD_ANGLE).sum())
    return {"bond_angles_within_bounds": n_bad == 0, "most_extreme_relative_angle": max(lower_extreme, upper_extreme)}


def _geometry_results(mol: Chem.Mol) -> tuple[dict, dict[str, tuple]]:
    """PoseBusters geometry and flatness results of a molecule with dock.yml's parameters, and (value, threshold) per test."""
    geometry = check_geometry(
        mol, threshold_bad_bond_length=THRESHOLD_BAD_BOND_LENGTH, threshold_clash=THRESHOLD_CLASH,
        threshold_bad_angle=THRESHOLD_BAD_ANGLE, bound_matrix_params=BOUND_MATRIX_PARAMS,
        ignore_hydrogens=IGNORE_HYDROGENS, sanitize=SANITIZE)
    results = geometry["results"] | _angle_results_without_metal_centres(mol, geometry.get("details", {}))
    flatness = {
        "aromatic ring flatness": (check_flatness(mol, THRESHOLD_FLATNESS_AROMATIC, FLAT_AROMATIC_RINGS),
                                   THRESHOLD_FLATNESS_AROMATIC),
        "non-aromatic ring non-flatness": (check_flatness(mol, THRESHOLD_FLATNESS_NONFLAT, NONFLAT_RINGS, True),
                                           THRESHOLD_FLATNESS_NONFLAT),
        "double bond flatness": (check_flatness(mol, THRESHOLD_FLATNESS_DOUBLE_BOND, FLAT_DOUBLE_BONDS),
                                 THRESHOLD_FLATNESS_DOUBLE_BOND),
    }
    values = {
        "bond_lengths_within_bounds": _worst_bond_ratio(results),
        "bond_angles_within_bounds": (results.get("most_extreme_relative_angle"), 1 + THRESHOLD_BAD_ANGLE),
        "no_internal_clash": (results.get("shortest_noncovalent_relative_distance"), 1 - THRESHOLD_CLASH),
    }
    for name, (result, threshold) in flatness.items():
        results = results | {name: result["results"]["flatness_passes"]}
        values[name] = (result["results"]["max_distance"], threshold)
    return results, values


def check_ligand_geometry(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """Bond lengths, angles, internal clash and flatness of each ligand chain's resolved atoms.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding per ligand chain; skipped for single-atom ligands.
    """
    atoms = inputs.final_atoms(pdb_id)
    findings = []
    for chain in _ligand_chains(inputs, pdb_id):
        res_name, label = _label(atoms, chain)
        mol = ligand_mol(atoms, chain, False, inputs)
        if mol.GetNumAtoms() < 2:
            findings.append(_finding(LIGAND_GEOMETRY, pdb_id, label, [], skipped="single atom"))
            continue
        results, values = _geometry_results(mol)
        findings.append(_finding(LIGAND_GEOMETRY, pdb_id, label, _rows(chain, res_name, results, _GEOMETRY_TESTS, values)))
    return findings


def _protein_mol(atoms: AtomArray, indices: np.ndarray) -> Chem.Mol:
    """Bond-less molecule of protein atoms, flagged as non-hetero atoms so dock.yml's ignore types keep them."""
    editable = Chem.RWMol()
    conformer = Chem.Conformer(len(indices))
    for i, index in enumerate(indices):
        atom = Chem.Atom(str(atoms.element[index]).capitalize())
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(str(atoms.res_name[index]))
        info.SetIsHeteroAtom(False)
        atom.SetMonomerInfo(info)
        editable.AddAtom(atom)
        x, y, z = atoms.coord[index]
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol = editable.GetMol()
    mol.AddConformer(conformer, assignId=True)
    return mol


def _nearby_protein(inputs: ValidityInputs, pdb_id: str, atoms: AtomArray, chain: str) -> np.ndarray:
    """Indices of resolved protein heavy atoms within 10 Å of the ligand, covalent partner residues left out."""
    chains = inputs.cache_entry(pdb_id)["chains"]
    protein_chains = [c for c, record in chains.items() if record["molecule_type"] == "PROTEIN"]
    keep = np.isin(atoms.chain_id.astype(str), protein_chains) & ~np.isnan(atoms.coord).any(axis=1)
    keep &= atoms.element.astype(str) != "H"
    for index in _partner_indices(atoms, chain):
        keep &= ~((atoms.chain_id == atoms.chain_id[index]) & (atoms.res_id == atoms.res_id[index]))
    ligand = atoms.coord[(atoms.chain_id.astype(str) == chain) & ~np.isnan(atoms.coord).any(axis=1)]
    candidates = np.flatnonzero(keep)
    if len(ligand) == 0 or len(candidates) == 0:
        return candidates[:0]
    distances = np.linalg.norm(atoms.coord[candidates][:, None, :] - ligand[None, :, :], axis=-1)
    return candidates[distances.min(axis=1) <= PROTEIN_RADIUS]


def _distance_results(ligand: Chem.Mol, protein: Chem.Mol, radius_type: str) -> dict:
    """PoseBusters' intermolecular distance results with dock.yml's parameters and the given radius type."""
    return check_intermolecular_distance(
        ligand, protein, radius_type=radius_type, radius_scale=RADIUS_SCALE,
        clash_cutoff=CLASH_CUTOFF_DISTANCE, ignore_types=IGNORE_TYPES, max_distance=MAX_DISTANCE)["results"]


def _contact_results(ligand: Chem.Mol, protein: Chem.Mol) -> tuple[dict, dict]:
    """Distance and volume-overlap results; metal atoms on covalent radii and out of the overlap.

    Args:
        ligand: The ligand's resolved atoms.
        protein: Nearby protein atoms.

    Returns:
        (distance results, overlap results), keyed as PoseBusters keys them.
    """
    metals = [atom.GetIdx() for atom in ligand.GetAtoms() if _is_metal(atom)]
    others = [i for i in range(ligand.GetNumAtoms()) if i not in metals]
    parts = [_distance_results(_atom_subset(ligand, others), protein, RADIUS_TYPE)] if others else []
    parts += [_distance_results(_atom_subset(ligand, metals), protein, METAL_RADIUS_TYPE)] if metals else []
    distance = {
        "smallest_distance": min(part["smallest_distance"] for part in parts),
        "not_too_far_away": any(part["not_too_far_away"] for part in parts),
        "most_extreme_relative_distance": min(part["most_extreme_relative_distance"] for part in parts),
        "no_clashes": all(part["no_clashes"] for part in parts),
    }
    if not others:
        return distance, {"volume_overlap": 0.0, "no_volume_clash": True}
    overlap = check_volume_overlap(
        _atom_subset(ligand, others), protein, clash_cutoff=CLASH_CUTOFF_OVERLAP, vdw_scale=VDW_SCALE,
        ignore_types=IGNORE_TYPES)["results"]
    return distance, overlap


def check_ligand_contacts(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """Minimum distance and volume overlap of each ligand chain with protein atoms within 10 Å.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding per ligand chain; covalent partner residues are left out. Skipped if no protein is near.
    """
    atoms = inputs.final_atoms(pdb_id)
    findings = []
    for chain in _ligand_chains(inputs, pdb_id):
        res_name, label = _label(atoms, chain)
        ligand = ligand_mol(atoms, chain, False, inputs)
        nearby = _nearby_protein(inputs, pdb_id, atoms, chain)
        if ligand.GetNumAtoms() == 0 or len(nearby) == 0:
            findings.append(_finding(LIGAND_CONTACTS, pdb_id, label, [], skipped="no protein within 10 Å"))
            continue
        protein = _protein_mol(atoms, nearby)
        distance, overlap = _contact_results(ligand, protein)
        values = {
            "not_too_far_away": (distance.get("smallest_distance"), MAX_DISTANCE),
            "no_clashes": (distance.get("most_extreme_relative_distance"), CLASH_CUTOFF_DISTANCE),
            "no_volume_clash": (overlap.get("volume_overlap"), CLASH_CUTOFF_OVERLAP),
        }
        rows = _rows(chain, res_name, distance | overlap, _CONTACT_TESTS, values)
        findings.append(_finding(LIGAND_CONTACTS, pdb_id, label, rows))
    return findings
