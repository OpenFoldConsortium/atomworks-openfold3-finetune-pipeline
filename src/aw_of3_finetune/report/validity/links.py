"""Covalent links of the input CIF's struct_conn against the bonds of the final structure."""

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.copies import Copies, atom_label, expand, find_copies, operators
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.cleanup import _BOND_FILTER_ARGS
from aw_of3_finetune.steplog import StepLog, read_input_atoms

COVALENT_LINKS = "validity.covalent_links"
LINK_TYPES = ("covale", "covale_base", "covale_phosphate", "covale_sugar", "disulf")
SELF_SYMMETRY = "1_555"
_POLYMERS = ("PROTEIN", "DNA", "RNA")
_MOLECULE_TYPES = {0: "PROTEIN", 1: "RNA", 2: "DNA", 3: "LIGAND"}
_EMPTY = (".", "?", "")
_SYMMETRY_REASON = "link to a crystal-symmetry mate; not checked"
_PREFILTER = "OF3 prefilter: "
_UNMAPPED_REASON = "partner atom not in the input atom table"


def _value(table, column: str, row: int) -> str:
    """One struct_conn cell as text; "" when the column is missing or the value is "." or "?"."""
    if column not in table:
        return ""
    text = str(table[column].as_array(str)[row])
    return "" if text in _EMPTY else text


def _links(cif_path) -> list[dict]:
    """The in-scope struct_conn rows: id, type, symmetry flag and the two partner keys."""
    block = pdbx.CIFFile.read(cif_path).block
    if "struct_conn" not in block:
        return []
    table = block["struct_conn"]
    links = []
    for row in range(len(table["id"].as_array(str))):
        conn_type = _value(table, "conn_type_id", row)
        if conn_type not in LINK_TYPES:
            continue
        partners = []
        for n in (1, 2):
            seq = _value(table, f"ptnr{n}_label_seq_id", row)
            partners.append({
                "asym": _value(table, f"ptnr{n}_label_asym_id", row),
                "comp": _value(table, f"ptnr{n}_label_comp_id", row),
                "atom": _value(table, f"ptnr{n}_label_atom_id", row),
                "seq": int(seq) if seq else None,
                "alt": _value(table, f"pdbx_ptnr{n}_label_alt_id", row),
            })
        symmetry = [_value(table, f"ptnr{n}_symmetry", row) or SELF_SYMMETRY for n in (1, 2)]
        links.append({"id": _value(table, "id", row), "type": conn_type, "partners": partners,
                      "symmetry": any(s != SELF_SYMMETRY for s in symmetry)})
    return links


def _partner_atoms(input_atoms: AtomArray, partner: dict) -> list[int]:
    """Input atom indices matching a struct_conn partner (any altloc unless the partner names one)."""
    mask = ((input_atoms.label_asym_id.astype(str) == partner["asym"])
            & (input_atoms.res_name.astype(str) == partner["comp"])
            & (input_atoms.atom_name.astype(str) == partner["atom"]))
    if partner["seq"] is not None:
        mask &= input_atoms.res_id == partner["seq"]
    if partner["alt"]:
        mask &= input_atoms.label_alt_id.astype(str) == partner["alt"]
    return [int(i) for i in np.flatnonzero(mask)]


def _removing_steps(log: StepLog | None) -> dict[int, str]:
    """Name of the step that removed each input atom, per the step log."""
    steps = {}
    if log is not None:
        for step in log.steps:
            for index in expand(step.removed_input):
                steps.setdefault(index, step.name)
    return steps


def _instances(copies: Copies, final: AtomArray, chains: tuple[str, str]) -> list[tuple[str, dict[int, int]]]:
    """(copy label, input index -> final position) for the identity copy and each operator copy of two chains."""
    identity = {index: position for index, position in copies.pairs}
    instances = [(copies.identity_labels.get(chains[0], chains[0]), identity)]
    final_labels = final.label_asym_id.astype(str)
    sources = {}
    for position, index in copies.source_of.items():
        sources.setdefault(final_labels[position], {})[index] = position
    for oper_id in dict.fromkeys(op for op in copies.operator_of.values() if op is not None):
        labels = []
        for chain in dict.fromkeys(chains):
            found = [label for label, op in copies.operator_of.items() if op == oper_id and label.split("_")[0] == chain]
            if not found:
                break
            labels.append(found[0])
        else:
            merged = {}
            for label in labels:
                merged.update(sources.get(label, {}))
            instances.append((labels[0], merged))
    return instances


def _molecule_type(inputs: ValidityInputs, pdb_id: str, final: AtomArray, position: int) -> str:
    """Molecule type of a final atom's chain: the dataset cache, else the molecule_type_id annotation."""
    record = inputs.cache_entry(pdb_id)["chains"].get(str(final.chain_id[position]))
    if record is not None:
        return record["molecule_type"]
    if "molecule_type_id" in final.get_annotation_categories():
        return _MOLECULE_TYPES[int(final.molecule_type_id[position])]
    raise KeyError(f"chain {final.chain_id[position]} is not in the dataset cache and has no molecule_type_id")


def _distance(input_atoms: AtomArray, first: int, second: int) -> float | None:
    """Input distance of two atoms rounded to 0.01 Å; None if either has no coordinates."""
    distance = float(np.linalg.norm(input_atoms.coord[first] - input_atoms.coord[second]))
    return None if np.isnan(distance) else round(distance, 2)


def _input_label(input_atoms: AtomArray, copy: str, index: int) -> str:
    """Atom label from the input table under a copy's label_asym_id."""
    return f"{copy}:{input_atoms.res_name[index]}{input_atoms.res_id[index]}:{input_atoms.atom_name[index]}"


def _dropped_reason(
    inputs: ValidityInputs, pdb_id: str, final: AtomArray, positions: tuple[int, int], distance: float | None
) -> str | None:
    """The OF3 prefilter rule that removes this bond, or None if no rule does.

    The inter-chain dative rule does not apply: the in-scope link types are covalent and metalc is excluded.
    """
    types = [_molecule_type(inputs, pdb_id, final, p) for p in positions]
    both_polymer = all(t in _POLYMERS for t in types)
    same_chain = final.chain_id[positions[0]] == final.chain_id[positions[1]]
    if _BOND_FILTER_ARGS["remove_inter_chain_poly_links"] and both_polymer and not same_chain:
        return f"{_PREFILTER}polymer–polymer link between chains"
    gap = abs(int(final.res_id[positions[0]]) - int(final.res_id[positions[1]]))
    if _BOND_FILTER_ARGS["remove_intra_chain_poly_links"] and both_polymer and same_chain and gap > 1:
        return f"{_PREFILTER}polymer link between non-consecutive residues"
    limit = _BOND_FILTER_ARGS["remove_longer_than"]
    if limit and distance is not None and distance > limit:
        return f"{_PREFILTER}longer than {limit} Å"
    return None


def _row(link: dict, copy: str, atoms: tuple[str, str], distance: float | None, state: str, label: str,
         reason: str | None) -> dict:
    """One report row."""
    return {"conn_id": link["id"], "conn_type": link["type"], "copy": copy, "atom_1": atoms[0], "atom_2": atoms[1],
            "input_distance": distance, "state": state, "label": label, "reason": reason}


def _unmapped_row(link: dict) -> dict:
    """Row of a link with a partner atom that is not in the input atom table."""
    atoms = tuple(f"{p['asym']}:{p['comp']}{p['seq'] if p['seq'] is not None else ''}:{p['atom']}"
                  for p in link["partners"])
    return _row(link, link["partners"][0]["asym"], atoms, None, "unmapped", "unexplained", _UNMAPPED_REASON)


def _link_rows(inputs: ValidityInputs, pdb_id: str, link: dict, candidates: list[list[int]], input_atoms: AtomArray,
               final: AtomArray, copies: Copies, steps: dict[int, str]) -> list[dict]:
    """The rows of one non-symmetry link, one per copy."""
    chains = tuple(str(input_atoms.label_asym_id[c[0]]) for c in candidates)
    rows = []
    for copy, mapping in _instances(copies, final, chains):
        present = [[c for c in options if c in mapping] for options in candidates]
        if not all(present):
            gone = candidates[0] if not present[0] else candidates[1]
            first = gone[0]
            step = steps.get(first, "cleanup")
            atoms = tuple(
                atom_label(final, mapping[c[0]]) if p else _input_label(input_atoms, copy, c[0])
                for c, p in zip(candidates, present)
            )
            distance = _distance(input_atoms, candidates[0][0], candidates[1][0])
            rows.append(_row(link, copy, atoms, distance, "partner_removed", "same", f"partner removed by {step}"))
            continue
        pairs = [(a, b) for a in present[0] for b in present[1]]
        bonded = [(a, b) for a, b in pairs if mapping[b] in final.bonds.get_bonds(mapping[a])[0]]
        a, b = bonded[0] if bonded else min(pairs, key=lambda pair: np.nan_to_num(
            np.linalg.norm(input_atoms.coord[pair[0]] - input_atoms.coord[pair[1]]), nan=np.inf))
        positions = (mapping[a], mapping[b])
        atoms = (atom_label(final, positions[0]), atom_label(final, positions[1]))
        distance = _distance(input_atoms, a, b)
        if bonded:
            rows.append(_row(link, copy, atoms, distance, "present", "same", None))
            continue
        reason = _dropped_reason(inputs, pdb_id, final, positions, distance)
        if reason:
            rows.append(_row(link, copy, atoms, distance, "dropped_by_design", "expected", reason))
        else:
            rows.append(_row(link, copy, atoms, distance, "missing", "unexplained", None))
    return rows


def check_covalent_links(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """Check that every covalent struct_conn link of the input is kept, removed with a partner or dropped by OF3.

    Args:
        inputs: The pipeline output to validate.
        pdb_id: Structure id.

    Returns:
        One finding: unexplained if any row is missing, else match. data is {"links": in-scope link count,
        "unmapped": links with a partner atom not in the input table (also rows), "rows": per link and final copy}.
    """
    cif_path = inputs.input_cif(pdb_id)
    links = _links(cif_path)
    rows, unmapped = [], 0
    if links:
        input_atoms = read_input_atoms(cif_path)
        final = inputs.final_atoms(pdb_id)
        copies = find_copies(input_atoms, final, operators(cif_path))
        steps = _removing_steps(inputs.step_log(pdb_id))
        for link in links:
            candidates = [_partner_atoms(input_atoms, partner) for partner in link["partners"]]
            if not all(candidates):
                unmapped += 1
                rows.append(_unmapped_row(link))
                continue
            if link["symmetry"]:
                copy = copies.identity_labels.get(str(input_atoms.label_asym_id[candidates[0][0]]),
                                                  str(input_atoms.label_asym_id[candidates[0][0]]))
                atoms = tuple(_input_label(input_atoms, copy, c[0]) for c in candidates)
                distance = _distance(input_atoms, candidates[0][0], candidates[1][0])
                rows.append(_row(link, copy, atoms, distance, "symmetry_link", "same", _SYMMETRY_REASON))
                continue
            rows += _link_rows(inputs, pdb_id, link, candidates, input_atoms, final, copies, steps)
    parts = [f"{row['atom_1']}—{row['atom_2']}: {row['state']}" for row in rows if row["state"] in ("missing", "unmapped")]
    status = Status.UNEXPLAINED if parts else Status.MATCH
    return [Finding(COVALENT_LINKS, pdb_id, status, "; ".join(parts),
                    {"links": len(links), "unmapped": unmapped, "rows": rows})]
