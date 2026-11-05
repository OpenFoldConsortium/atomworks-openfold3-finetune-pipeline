"""Alternate-location (altloc) choice matching OpenFold3: highest occupancy per residue.

OF3 parses with biotite's ``altloc="occupancy"`` (biotite.structure.filter_highest_occupancy_altloc):
per residue, the lettered conformer with the highest summed occupancy is kept, ties going
to the alphabetically first letter; atoms without an altloc ID are always kept. atomworks
has no such option (its ``"first"`` keeps one letter per chain), so the source CIF's
atom_site is filtered here before atomworks parses it.
"""

from __future__ import annotations

import numpy as np
from biotite.structure.io.pdbx import CIFCategory, CIFColumn, CIFData, CIFFile

# Altloc IDs that mean "no alternate location" (biotite's list).
NO_ALTLOC_IDS = (".", "?", " ", "")

# Residue identity, as OF3 parses it (author fields, per model).
_RESIDUE_KEY_COLUMNS = (
    "pdbx_PDB_model_num",
    "auth_asym_id",
    "auth_seq_id",
    "pdbx_PDB_ins_code",
    "label_comp_id",
)


def _residue_runs(site: CIFCategory) -> list[tuple[int, int]]:
    """``(start, stop)`` row ranges of consecutive rows belonging to one residue.

    Args:
        site: The atom_site category.

    Returns:
        Row ranges, in file order.
    """
    keys = [site[name].as_array(str) for name in _RESIDUE_KEY_COLUMNS if name in site]
    n_rows = len(keys[0])
    changes = np.zeros(n_rows, dtype=bool)
    for key in keys:
        changes[1:] |= key[1:] != key[:-1]
    starts = np.r_[0, np.flatnonzero(changes)]
    stops = np.r_[starts[1:], n_rows]
    return list(zip(starts.tolist(), stops.tolist()))


def select_highest_occupancy_altlocs(cif: CIFFile) -> int:
    """Keep each residue's highest-occupancy conformer in ``cif``'s atom_site, in place.

    Rows of other conformers are dropped and the kept rows' ``label_alt_id`` is set to ".",
    so atomworks' own altloc choice has nothing left to choose.

    Args:
        cif: Parsed mmCIF; its first block's atom_site is replaced.

    Returns:
        Number of residues that had lettered conformers (0: atom_site left untouched,
        also when it has no label_alt_id or occupancy column).
    """
    site = cif.block["atom_site"]
    # Without these columns there is nothing to choose by; OF3 falls back to "first" too.
    if "label_alt_id" not in site or "occupancy" not in site:
        return 0
    alt = site["label_alt_id"].as_array(str)
    occupancy = site["occupancy"].as_array(float)

    keep = np.isin(alt, NO_ALTLOC_IDS)
    n_residues = 0
    for start, stop in _residue_runs(site):
        letters = sorted({a for a in alt[start:stop] if a.isalpha()})
        if not letters:
            continue
        n_residues += 1
        best, best_sum = None, -1.0
        for letter in letters:
            letter_sum = float(occupancy[start:stop][alt[start:stop] == letter].sum())
            if letter_sum > best_sum:
                best, best_sum = letter, letter_sum
        keep[start:stop] |= alt[start:stop] == best
    if n_residues == 0:
        return 0

    columns = {}
    for name in site.keys():
        column = site[name]
        data = column.data.array[keep]
        if name == "label_alt_id":
            data = np.full(len(data), ".")
        mask = None if column.mask is None or name == "label_alt_id" else CIFData(column.mask.array[keep])
        columns[name] = CIFColumn(CIFData(data), mask)
    cif.block["atom_site"] = CIFCategory(columns)
    return n_residues
