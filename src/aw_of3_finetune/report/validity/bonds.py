"""Residue bond orders against the CCD, and bond lengths of the final structure."""

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.inputs import ValidityInputs

RESIDUE_BONDS = "validity.residue_bonds"
BOND_LENGTHS = "validity.bond_lengths"
MIN_BOND, MAX_BOND = 0.9, 2.4  # Å; 2.4 is OF3's prefilter cut-off
_ORDERS = {"SING": 1, "DOUB": 2, "TRIP": 3}
_AROMATIC_ORDERS = {"SING": 5, "DOUB": 6}
_ANY = 0


def _residue_ids(atoms: AtomArray) -> np.ndarray:
    """Residue number of every atom; a new residue starts where chain copy, chain, id or name changes."""
    keys = [atoms.label_asym_id.astype(str), atoms.chain_id.astype(str), atoms.res_id, atoms.res_name]
    changed = np.zeros(len(atoms), dtype=bool)
    changed[0] = True
    for key in keys:
        changed[1:] |= key[1:] != key[:-1]
    return np.cumsum(changed) - 1


def _bond_orders(table, comp_id: str | None = None) -> dict[tuple[str, str], int]:
    """Bond orders by sorted atom-name pair of a chem_comp_bond table (only comp_id's rows when given)."""
    columns = [table[c].as_array(str) for c in ("atom_id_1", "atom_id_2", "value_order", "pdbx_aromatic_flag")]
    comp_ids = table["comp_id"].as_array(str) if comp_id is not None else [None] * len(columns[0])
    bonds = {}
    for owner, name_1, name_2, order, aromatic in zip(comp_ids, *columns):
        if owner == comp_id:
            orders = _AROMATIC_ORDERS if aromatic == "Y" and order in _AROMATIC_ORDERS else _ORDERS
            bonds[tuple(sorted((name_1, name_2)))] = orders.get(order, _ANY)
    return bonds


def _ccd_bonds(inputs: ValidityInputs, res_name: str) -> dict[tuple[str, str], int] | None:
    """CCD bond orders by sorted atom-name pair; None if the residue is not in the CCD."""
    cache = inputs._cache.setdefault("ccd_bonds", {})
    if res_name not in cache:
        ccd = inputs.ccd()
        if res_name not in ccd:
            cache[res_name] = None
        else:
            block = ccd[res_name]
            cache[res_name] = _bond_orders(block["chem_comp_bond"]) if "chem_comp_bond" in block else {}
    return cache[res_name]


def _input_bonds(inputs: ValidityInputs, pdb_id: str, res_name: str) -> dict[tuple[str, str], int] | None:
    """Bond orders the input CIF's own chem_comp_bond gives a residue; None if it defines none."""
    cache = inputs._cache.setdefault("input_bonds", {})
    if pdb_id not in cache:
        block = pdbx.CIFFile.read(inputs.input_cif(pdb_id)).block
        cache[pdb_id] = block["chem_comp_bond"] if "chem_comp_bond" in block else None
    table = cache[pdb_id]
    if table is None or res_name not in set(table["comp_id"].as_array(str)):
        return None
    return _bond_orders(table, res_name)


def _reference_bonds(inputs: ValidityInputs, pdb_id: str, res_name: str) -> dict[tuple[str, str], int] | None:
    """Bond orders to check a residue against: the CCD's, else the input CIF's own; None if neither has it."""
    ccd_bonds = _ccd_bonds(inputs, res_name)
    return ccd_bonds if ccd_bonds is not None else _input_bonds(inputs, pdb_id, res_name)


def _structure_bonds(atoms: AtomArray) -> tuple[np.ndarray, np.ndarray, dict[int, dict[tuple[str, str], int]]]:
    """Residue number and resolved flag per atom, and each residue's bond orders between resolved atoms."""
    resolved = ~np.isnan(atoms.coord).any(axis=1)
    residue = _residue_ids(atoms)
    names = atoms.atom_name.astype(str)
    bonds: dict[int, dict[tuple[str, str], int]] = {}
    for i, j, order in atoms.bonds.as_array():
        if residue[i] == residue[j] and resolved[i] and resolved[j]:
            bonds.setdefault(int(residue[i]), {})[tuple(sorted((names[i], names[j])))] = int(order)
    return residue, resolved, bonds


def _differences(found: dict, reference: dict, present: set[str]) -> tuple[list, list, list]:
    """Extra, missing and order-differing atom-name pairs of a residue's bonds against a reference."""
    extra = sorted(pair for pair in found if pair not in reference)
    missing = sorted(pair for pair in reference if set(pair) <= present and pair not in found)
    differs = sorted(pair for pair in found if pair in reference and _ANY not in (found[pair], reference[pair])
                     and found[pair] != reference[pair])
    return extra, missing, differs


def _row(atoms: AtomArray, first: int, extra: list, missing: list, order_differs: list) -> dict:
    """Report row of the residue starting at atom index first whose bonds differ from the CCD."""
    return {"chain": str(atoms.chain_id[first]), "res_name": str(atoms.res_name[first]),
            "res_id": int(atoms.res_id[first]), "issue": "ccd_bonds", "extra": extra, "missing": missing,
            "order_differs": order_differs}


def check_residue_bonds(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """Compare every residue's intra-residue bonds with the CCD by atom-name pair.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding; "order A-B x vs y" is the structure's order vs the reference (the CCD, else the input CIF). A name in neither
        is one detail part and one row ({res_name, chains, issue}).
    """
    atoms = inputs.final_atoms(pdb_id)
    residue, resolved, structure_bonds = _structure_bonds(atoms)
    names = atoms.atom_name.astype(str)

    parts, rows = [], []
    not_in_ccd: dict[str, list[str]] = {}
    for number in np.unique(residue):
        positions = np.flatnonzero(residue == number)
        first = int(positions[0])
        res_name = str(atoms.res_name[first])
        reference = _reference_bonds(inputs, pdb_id, res_name)
        if reference is None:
            chains = not_in_ccd.setdefault(res_name, [])
            if str(atoms.chain_id[first]) not in chains:
                chains.append(str(atoms.chain_id[first]))
            continue
        present = {str(names[p]) for p in positions if resolved[p]}
        found = structure_bonds.get(int(number), {})
        extra, missing, differs = _differences(found, reference, present)
        parts += [f"{res_name}: extra bond {a}-{b}" for a, b in extra]
        parts += [f"{res_name}: missing bond {a}-{b}" for a, b in missing]
        parts += [f"{res_name}: order {a}-{b} {found[a, b]} vs {reference[a, b]}" for a, b in differs]
        if extra or missing or differs:
            rows.append(_row(atoms, first, [f"{a}-{b}" for a, b in extra], [f"{a}-{b}" for a, b in missing],
                             [f"{a}-{b}" for a, b in differs]))
    parts = [f"{res_name}: not in CCD" for res_name in not_in_ccd] + parts
    rows = [{"res_name": name, "chains": chains, "issue": "not_in_ccd"} for name, chains in not_in_ccd.items()] + rows
    status = Status.UNEXPLAINED if parts else Status.MATCH
    return [Finding(RESIDUE_BONDS, pdb_id, status, "; ".join(parts), {"rows": rows})]


def bonds_follow_input_definitions(inputs: ValidityInputs, pdb_id: str) -> bool:
    """Whether every residue whose bonds differ from the CCD has exactly the bonds the input CIF defines for it.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.

    Returns:
        True when each such residue has a reference and matches the input CIF's own chem_comp_bond; then the
        difference from the CCD is already in the input (e.g. OpenBind's Kekule ligands vs its aromatic CCD).
    """
    atoms = inputs.final_atoms(pdb_id)
    residue, resolved, structure_bonds = _structure_bonds(atoms)
    names = atoms.atom_name.astype(str)
    for number in np.unique(residue):
        positions = np.flatnonzero(residue == number)
        res_name = str(atoms.res_name[positions[0]])
        reference = _reference_bonds(inputs, pdb_id, res_name)
        if reference is None:
            return False
        present = {str(names[p]) for p in positions if resolved[p]}
        found = structure_bonds.get(int(number), {})
        if not any(_differences(found, reference, present)):
            continue
        input_bonds = _input_bonds(inputs, pdb_id, res_name)
        if input_bonds is None or any(_differences(found, input_bonds, present)):
            return False
    return True


def check_bond_lengths(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """Check that every bond between resolved atoms is 0.9-2.4 Å.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding listing each bond outside the range, with one row per bond; atom_2 is
        "<chain>:<RES><res_id>:<ATOM>" when it is in another residue than atom_1.
    """
    atoms = inputs.final_atoms(pdb_id)
    resolved = ~np.isnan(atoms.coord).any(axis=1)
    residue = _residue_ids(atoms)

    def label(index: int) -> str:
        return f"{atoms.chain_id[index]}:{atoms.res_name[index]}{atoms.res_id[index]}:{atoms.atom_name[index]}"

    parts, rows = [], []
    for i, j, _ in atoms.bonds.as_array():
        if not (resolved[i] and resolved[j]):
            continue
        distance = float(np.linalg.norm(atoms.coord[i] - atoms.coord[j]))
        if not MIN_BOND <= distance <= MAX_BOND:
            problem = f"too short (min {MIN_BOND} Å)" if distance < MIN_BOND else f"too long (max {MAX_BOND} Å)"
            parts.append(f"{label(i)}-{label(j)} {distance:.3f} Å, {problem}")
            rows.append({
                "chain": str(atoms.chain_id[i]),
                "label_asym_id": str(atoms.label_asym_id[i]),
                "res_name": str(atoms.res_name[i]),
                "res_id": int(atoms.res_id[i]),
                "atom_1": str(atoms.atom_name[i]),
                "atom_2": str(atoms.atom_name[j]) if residue[i] == residue[j] else label(j),
                "distance": round(distance, 3),
            })
    status = Status.UNEXPLAINED if parts else Status.MATCH
    return [Finding(BOND_LENGTHS, pdb_id, status, "; ".join(parts), {"rows": rows})]
