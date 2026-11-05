"""One name, one molecule: a ligand name outside the CCD must mean one molecule across a dataset.

Reference molecules are stored by residue name, so a CCD name is safe but a made-up one is not:
two molecules under one name would share the first one's reference. Molecules are compared by
atom names, elements and bonds, so different poses of one molecule agree.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LIGAND_MOLECULE_TYPE_ID = 3
MAX_LISTED_ENTRIES = 6

LigandSignature = tuple[frozenset, frozenset]
SignaturesByName = dict[str, dict[LigandSignature, set[str]]]


@dataclass(frozen=True)
class NameConflict:
    """One ligand name used for molecules that differ.

    Attributes:
        name: The residue name.
        entries: Structures holding the name, sorted.
        reason: How two of its molecules differ.
    """

    name: str
    entries: list[str]
    reason: str


def residue_signature(
    atom_names: Sequence, elements: Sequence, rows: Sequence[int], bonds: Sequence
) -> LigandSignature:
    """The name-element pairs and name-pair bonds of one residue.

    Args:
        atom_names: Atom name of every atom in the structure.
        elements: Element of every atom in the structure.
        rows: The residue's atom indices.
        bonds: (atom index, atom index) pairs of the structure.

    Returns:
        (name-element pairs, name-pair bonds).
    """
    names = {int(row): str(atom_names[row]) for row in rows}
    name_elements = frozenset((names[row], str(elements[row]).upper()) for row in names)
    inside = [(a, b) for a, b in bonds if a in names and b in names]
    return name_elements, frozenset(frozenset((names[a], names[b])) for a, b in inside)


def signature_difference(sig_a: LigandSignature, sig_b: LigandSignature) -> str | None:
    """Describe the first way two signatures differ.

    Args:
        sig_a: One residue's signature.
        sig_b: Another residue's signature.

    Returns:
        A short description, or None if they are identical.
    """
    elements_a, elements_b = dict(sig_a[0]), dict(sig_b[0])
    only_one_side = sorted(set(elements_a) ^ set(elements_b))
    if only_one_side:
        return f"atom names differ: {', '.join(only_one_side[:5])}"
    for name in sorted(elements_a):
        if elements_a[name] != elements_b[name]:
            return f"element of atom {name} differs"
    if sig_a[1] != sig_b[1]:
        return "bonds differ"
    return None


def name_conflicts(signatures: SignaturesByName) -> list[NameConflict]:
    """Find names used for more than one molecule.

    Args:
        signatures: name -> signature -> structures holding it.

    Returns:
        One conflict per such name, sorted by name.
    """
    conflicts = []
    for name, by_signature in sorted(signatures.items()):
        if len(by_signature) < 2:
            continue
        first, *others = by_signature
        reason = next(filter(None, (signature_difference(first, other) for other in others)))
        conflicts.append(NameConflict(name, sorted(set().union(*by_signature.values())), reason))
    return conflicts


def format_name_conflicts(conflicts: list[NameConflict]) -> str:
    """The user-facing message for name conflicts.

    Args:
        conflicts: Output of name_conflicts, not empty.

    Returns:
        What is wrong and how to fix it.
    """
    lines = [f"{len(conflicts)} ligand name(s) are used for different molecules:"]
    for conflict in conflicts:
        listed = ", ".join(conflict.entries[:MAX_LISTED_ENTRIES])
        hidden = len(conflict.entries) - MAX_LISTED_ENTRIES
        more = f" and {hidden} more" if hidden > 0 else ""
        lines.append(f"  {conflict.name} in {listed}{more} ({conflict.reason})")
    lines.append(
        "A name outside the CCD must mean one molecule, with the same atom names, elements and "
        "bonds in every structure. Give each molecule its own name, or use prepare-inputs. One "
        "molecule can also differ like this if a crystal is missing atoms and the CIF does not "
        "define the ligand; add its definition or give that copy its own name."
    )
    return "\n".join(lines)


def record_ligand_signatures(
    atom_array, chains: dict, pdb_id: str, ccd_ids, signatures: SignaturesByName
) -> None:
    """Add a structure's single-residue non-CCD ligands to ``signatures``.

    Args:
        atom_array: OF3-parsed atom array of the structure.
        chains: Chain metadata by chain id, for each chain's reference_mol_id.
        pdb_id: Structure ID.
        ccd_ids: Container of the CCD's component ids.
        signatures: Collected so far; mutated in place.
    """
    is_ligand = atom_array.molecule_type_id == LIGAND_MOLECULE_TYPE_ID
    non_ccd_names = [str(n) for n in np.unique(atom_array.res_name[is_ligand]) if str(n) not in ccd_ids]
    if not non_ccd_names:
        return
    rows_by_residue: dict[tuple, list[int]] = defaultdict(list)
    for row in np.flatnonzero(is_ligand & np.isin(atom_array.res_name, non_ccd_names)):
        key = (
            str(atom_array.chain_id[row]), str(atom_array.res_name[row]),
            int(atom_array.res_id[row]), str(atom_array.ins_code[row]),
        )
        rows_by_residue[key].append(int(row))
    bonds = atom_array.bonds.as_array()[:, :2].tolist()
    for (chain, name, _, _), rows in rows_by_residue.items():
        reference = getattr(chains.get(chain), "reference_mol_id", None)
        if reference in (None, name):  # a multi-residue ligand has its own per-structure reference
            signature = residue_signature(atom_array.atom_name, atom_array.element, rows, bonds)
            signatures.setdefault(name, {}).setdefault(signature, set()).add(pdb_id)


def read_ccd_ids(ccd_path: Path) -> frozenset[str]:
    """Read a CCD's component ids without parsing the file.

    Args:
        ccd_path: Text mmCIF CCD.

    Returns:
        The ids of its data blocks.
    """
    prefix = b"data_"
    with ccd_path.open("rb") as handle:
        return frozenset(
            line[len(prefix):].strip().decode() for line in handle if line.startswith(prefix)
        )
