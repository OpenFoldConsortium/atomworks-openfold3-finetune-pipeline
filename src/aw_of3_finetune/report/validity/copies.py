"""Which final atoms are input atoms (identity copies) and which are assembly copies, without the step log.

A chain's identity copy is the copy with most atoms at input coordinates (at least half). Every other
copy is matched to one pdbx_struct_oper_list operator: the operator that, applied to the input atoms
of the copy's chain, lands most copy atoms (at least half) within 0.01 Å. Chained operators (atomworks'
"1X0"-style ids) are out of scope; a copy no single operator explains is left unmatched and reported.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree

_COORD_DECIMALS = 3
COPY_TOLERANCE = 0.01  # Å between a copy atom and operator x source


@dataclass
class Copies:
    """Final atoms explained by the input: identity pairs and operator copies.

    pairs are (input index, final position) of identity copies; operator_of maps every other
    copy label to its operator id (None = unmatched); source_of maps each matched copy atom's
    final position to its input source index.
    """

    pairs: list[tuple[int, int]] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    identity_labels: dict[str, str] = field(default_factory=dict)
    operator_of: dict[str, str | None] = field(default_factory=dict)
    source_of: dict[int, int] = field(default_factory=dict)


def expand(ranges: list[tuple[int, int]]) -> list[int]:
    """Indices in inclusive (start, end) ranges.

    Args:
        ranges: Inclusive (start, end) pairs.

    Returns:
        Every index they cover, in order.
    """
    return [i for start, end in ranges for i in range(start, end + 1)]


def operators(cif_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rotation matrix and translation vector per pdbx_struct_oper_list id of a CIF.

    Args:
        cif_path: Path to the input mmCIF file.

    Returns:
        {operator id: (3x3 matrix, 3-vector)}; empty if the CIF has no operator list.
    """
    block = pdbx.CIFFile.read(cif_path).block
    if "pdbx_struct_oper_list" not in block:
        return {}
    table = block["pdbx_struct_oper_list"]
    result = {}
    for row, oper_id in enumerate(table["id"].as_array(str)):
        matrix = np.array([[table[f"matrix[{i}][{j}]"].as_array(float)[row] for j in (1, 2, 3)] for i in (1, 2, 3)])
        vector = np.array([table[f"vector[{i}]"].as_array(float)[row] for i in (1, 2, 3)])
        result[str(oper_id)] = (matrix, vector)
    return result


def _coord_keys(atoms: AtomArray) -> list[tuple | None]:
    """Rounded coordinates per atom; None for unresolved atoms."""
    rounded = np.round(atoms.coord, _COORD_DECIMALS).tolist()
    resolved = ~np.isnan(atoms.coord).any(axis=1)
    return [tuple(coord) if ok else None for coord, ok in zip(rounded, resolved)]


def _copy_groups(final: AtomArray) -> dict[str, list[int]]:
    """Final atom positions per chain copy, keyed by the full label_asym_id (e.g. "A_2")."""
    groups: dict[str, list[int]] = defaultdict(list)
    for position, label in enumerate(final.label_asym_id.astype(str)):
        groups[label].append(position)
    return groups


def atom_label(atoms: AtomArray, position: int) -> str:
    """"<label_asym_id>:<RES><res_id>:<ATOM>" of one atom.

    Args:
        atoms: The atoms.
        position: Index of the atom.

    Returns:
        The label.
    """
    return f"{atoms.label_asym_id[position]}:{atoms.res_name[position]}{atoms.res_id[position]}:{atoms.atom_name[position]}"


def _identity_atoms(final: AtomArray, input_keys: set) -> tuple[list[int], list[str], dict[str, str]]:
    """Each chain's identity copy: positions at input coordinates, moved-atom labels, chain -> copy label."""
    keys = _coord_keys(final)
    best: dict[str, tuple[int, str, list[int], list[int]]] = {}
    for label, positions in _copy_groups(final).items():
        chain = label.split("_")[0]
        resolved = [p for p in positions if keys[p] is not None]
        hits = [p for p in resolved if keys[p] in input_keys]
        if len(hits) * 2 >= len(resolved) > 0 and (chain not in best or len(hits) > best[chain][0]):
            best[chain] = (len(hits), label, hits, [p for p in resolved if keys[p] not in input_keys])
    kept = [p for _, _, hits, _ in best.values() for p in hits]
    moved = [atom_label(final, p) for _, _, _, others in best.values() for p in others]
    return kept, moved, {chain: label for chain, (_, label, _, _) in best.items()}


def _identity(input_atoms: AtomArray, final: AtomArray) -> tuple[list[tuple[int, int]], list[str], dict[str, str]]:
    """Identity pairs sorted by input index, moved labels and each chain's identity copy label."""
    inputs_by_key: dict[tuple, list[int]] = defaultdict(list)
    for index, key in enumerate(_coord_keys(input_atoms)):
        inputs_by_key[key].append(index)
    final_keys = _coord_keys(final)
    finals_by_key: dict[tuple, list[int]] = defaultdict(list)
    identity_positions, moved, labels = _identity_atoms(final, set(inputs_by_key))
    for position in identity_positions:
        finals_by_key[final_keys[position]].append(position)
    pairs = []
    for key, positions in finals_by_key.items():
        pairs += list(zip(inputs_by_key[key], positions))
    return sorted(pairs), moved, labels


def _is_identity(matrix: np.ndarray, vector: np.ndarray) -> bool:
    """True for the identity operator, which never builds a second copy."""
    return bool(np.allclose(matrix, np.eye(3)) and np.allclose(vector, 0.0))


def _match_operator(
    source_xyz: np.ndarray, copy_xyz: np.ndarray, oper_list: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[str | None, np.ndarray, np.ndarray]:
    """The operator landing most copy atoms on a transformed source atom (at least half).

    Returns:
        (operator id or None, per copy atom whether it hit, per copy atom the nearest source row).
    """
    best: tuple[str | None, np.ndarray, np.ndarray] = (None, np.zeros(len(copy_xyz), bool), np.zeros(0, int))
    if len(source_xyz) == 0 or len(copy_xyz) == 0:
        return best
    for oper_id, (matrix, vector) in oper_list.items():
        if _is_identity(matrix, vector):
            continue
        distance, nearest = cKDTree(source_xyz @ matrix.T + vector).query(copy_xyz)
        hits = distance <= COPY_TOLERANCE
        if hits.sum() * 2 >= len(copy_xyz) and hits.sum() > best[1].sum():
            best = (oper_id, hits, nearest)
    return best


def find_copies(
    input_atoms: AtomArray, final: AtomArray, oper_list: dict[str, tuple[np.ndarray, np.ndarray]]
) -> Copies:
    """Identity pairs plus the operator and input source of every other chain copy's atoms.

    Args:
        input_atoms: The input atom table.
        final: The final structure.
        oper_list: operators() of the input CIF.

    Returns:
        The copies; a non-identity copy no operator explains has operator None and no sources.
    """
    pairs, moved, identity_labels = _identity(input_atoms, final)
    copies = Copies(pairs=pairs, moved=moved, identity_labels=identity_labels)
    input_chain = input_atoms.label_asym_id.astype(str)
    resolved = ~np.isnan(final.coord).any(axis=1)
    for label, positions in _copy_groups(final).items():
        chain = label.split("_")[0]
        if identity_labels.get(chain) == label:
            continue
        copy_positions = np.array([p for p in positions if resolved[p]], dtype=int)
        if len(copy_positions) == 0:
            continue
        sources = np.flatnonzero(input_chain == chain)
        oper_id, hits, nearest = _match_operator(input_atoms.coord[sources], final.coord[copy_positions], oper_list)
        copies.operator_of[label] = oper_id
        if oper_id is not None:
            for position, hit, row in zip(copy_positions, hits, nearest):
                if hit:
                    copies.source_of[int(position)] = int(sources[row])
    return copies
