"""Where each of our residues sits against the input CIF, and what became of each input residue."""

from collections import Counter

import numpy as np
from biotite.structure import AtomArray, info as residue_info
from biotite.structure.io import pdbx

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.copies import expand, find_copies, operators
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.steplog import StepLog, read_input_atoms

CHECK = "validity.chain_placement"
BAD_STATES = ("wrong_chain", "wrong_residue", "no_counterpart", "split")
_MAX_LISTED = 100
_MOLECULE_TYPES = {0: "PROTEIN", 1: "RNA", 2: "DNA", 3: "LIGAND"}


_DEOXY_NAMES = {"DA", "DC", "DG", "DT", "DI", "DU"}


def _poly_types_from_residues(entity_ids: list[str], site) -> dict[str, str]:
    """entity_poly.type-like strings from each entity's atom_site residue names (for CIFs lacking the column)."""
    amino, nucleic = set(residue_info.amino_acid_names()), set(residue_info.nucleotide_names())
    names = set(zip(site["label_entity_id"].as_array(str), site["label_comp_id"].as_array(str)))
    types = {}
    for entity in entity_ids:
        residues = [name for (owner, name) in names if owner == entity]
        if any(name in _DEOXY_NAMES for name in residues):
            types[entity] = "polydeoxyribonucleotide"
        elif sum(name in nucleic for name in residues) > sum(name in amino for name in residues):
            types[entity] = "polyribonucleotide"
        elif any(name in amino for name in residues):
            types[entity] = "polypeptide(L)"
    return types


def _chain_info(cif_path) -> tuple[dict[str, dict], dict[str, list[tuple[int, str]]] | None]:
    """Per label_asym_id entity, auth id and type, and the polymer residues of the sequence scheme.

    Args:
        cif_path: Path to the input mmCIF file.

    Returns:
        ({label_asym_id: {"entity", "auth", "type"}}, {polymer label_asym_id: [(seq_id, name)]}),
        the second None when the CIF has no pdbx_poly_seq_scheme.
    """
    block = pdbx.CIFFile.read(cif_path).block
    site = block["atom_site"]
    asyms = site["label_asym_id"].as_array(str)
    entities = site["label_entity_id"].as_array(str)
    auths = site["auth_asym_id"].as_array(str)
    poly_types = {}
    if "entity_poly" in block:
        poly = block["entity_poly"]
        if "type" in poly:
            poly_types = dict(zip(poly["entity_id"].as_array(str), poly["type"].as_array(str)))
        else:  # OpenBind's monomer CIFs
            poly_types = _poly_types_from_residues(poly["entity_id"].as_array(str).tolist(), site)
    info = {}
    for asym, entity, auth in zip(asyms, entities, auths):
        if asym not in info:
            kind = poly_types.get(entity, "")
            molecule = "LIGAND"
            if "polypeptide" in kind:
                molecule = "PROTEIN"
            elif "deoxyribonucleotide" in kind:
                molecule = "DNA"
            elif "ribonucleotide" in kind:
                molecule = "RNA"
            info[asym] = {"entity": int(entity) if entity.isdigit() else None, "auth": str(auth), "type": molecule}
    if "pdbx_poly_seq_scheme" not in block:
        return info, None
    scheme = block["pdbx_poly_seq_scheme"]
    polymers: dict[str, list[tuple[int, str]]] = {}
    for asym, seq_id, name in zip(
        scheme["asym_id"].as_array(str), scheme["seq_id"].as_array(int), scheme["mon_id"].as_array(str)
    ):
        polymers.setdefault(str(asym), []).append((int(seq_id), str(name)))
    return info, polymers


def _residue_segments(atoms: AtomArray) -> list[tuple[int, int]]:
    """(start, end) atom ranges of residues: label_asym_id, res_id, ins_code or res_name changes."""
    keys = list(zip(atoms.label_asym_id.astype(str), atoms.res_id, atoms.ins_code.astype(str), atoms.res_name.astype(str)))
    starts = [0] + [i for i in range(1, len(keys)) if keys[i] != keys[i - 1]] if keys else []
    return list(zip(starts, starts[1:] + [len(keys)]))


def _renamed_names(log: StepLog | None) -> dict[int, str]:
    """Residue name each input atom has after its logged renames."""
    names = {}
    if log is not None:
        for step in log.steps:
            for index, _, after in step.renamed:
                names[index] = after.split(":")[0]
    return names


def _our_state(
    final: AtomArray,
    segment: tuple[int, int],
    position_to_input: dict[int, int],
    input_atoms: AtomArray,
    renamed: dict[int, str],
    polymer_asyms: set[str] | None,
    copy_atoms: set[int],
) -> tuple[str, str | None]:
    """The state of one final residue and the input chain its atoms land on (None if none)."""
    start, end = segment
    label, res_id, res_name = str(final.label_asym_id[start]), int(final.res_id[start]), str(final.res_name[start])
    if np.isnan(final.coord[start:end]).any(axis=1).all():
        return "unresolved", None
    base = label.split("_")[0]
    # Non-polymer residues are numbered differently in the input (a counter) and the final (auth ids).
    numbered = polymer_asyms is None or base in polymer_asyms
    if any(position in copy_atoms for position in range(start, end)):
        return "assembly_copy", None
    landed = [position_to_input[p] for p in range(start, end) if p in position_to_input]
    if not landed:
        return "no_counterpart", None
    residues = {(str(input_atoms.label_asym_id[i]), int(input_atoms.res_id[i])) for i in landed}
    asym = Counter(str(input_atoms.label_asym_id[i]) for i in landed).most_common(1)[0][0]
    if len(residues) > 1:
        return "split", asym
    if asym != base:
        return "wrong_chain", asym
    if numbered and next(iter(residues))[1] != res_id:
        return "wrong_residue", asym
    allowed = {str(input_atoms.res_name[i]) for i in landed} | {renamed[i] for i in landed if i in renamed}
    return ("ok" if res_name in allowed else "wrong_residue"), asym


def _our_chains(
    final: AtomArray, states: dict[tuple[int, int], tuple[str, str | None]], segments: list[tuple[int, int]]
) -> list[dict]:
    """The chain records of our side, with residues in file order."""
    names = final.get_annotation_categories()
    chains: dict[str, dict] = {}
    placed: dict[str, Counter] = {}
    for segment in segments:
        start = segment[0]
        label = str(final.label_asym_id[start])
        if label not in chains:
            molecule = int(final.molecule_type_id[start]) if "molecule_type_id" in names else None
            entity = int(final.entity_id[start]) if "entity_id" in names else None
            chains[label] = {
                "id": str(final.chain_id[start]),
                "label": label,
                "auth": str(final.auth_asym_id[start]) if "auth_asym_id" in names else None,
                "entity": entity,
                "type": _MOLECULE_TYPES.get(molecule),
                "ref_mol": str(final.res_name[start]) if molecule == 3 else None,
                "paired": None,
                "res": [],
            }
            placed[label] = Counter()
        state, asym = states[segment]
        chains[label]["res"].append([int(final.res_id[start]), str(final.res_name[start]), state])
        if asym is not None:
            placed[label][asym] += 1
    for label, chain in chains.items():
        if placed[label]:
            chain["paired"] = placed[label].most_common(1)[0][0]
    return list(chains.values())


def _input_chains(
    input_atoms: AtomArray,
    info: dict[str, dict],
    polymers: dict[str, list[tuple[int, str]]] | None,
    kept: set[int],
    removed: set[int],
    paired: dict[str, list[str]],
) -> list[dict]:
    """The chain records of the input side: ok, removed or missing per residue, unresolved from the scheme."""
    by_residue: dict[tuple[str, int], list[int]] = {}
    for index, key in enumerate(zip(input_atoms.label_asym_id.astype(str), input_atoms.res_id.tolist())):
        by_residue.setdefault(key, []).append(index)
    chains = []
    for asym in dict.fromkeys(input_atoms.label_asym_id.astype(str)):
        residues = []
        scheme = polymers.get(asym) if polymers else None
        order = scheme or [(res_id, None) for (a, res_id) in by_residue if a == asym]
        for res_id, scheme_name in order:
            indices = by_residue.get((asym, res_id), [])
            if not indices:
                state, name = "unresolved", scheme_name
            else:
                name = str(input_atoms.res_name[indices[0]])
                if any(i in kept for i in indices):
                    state = "ok"
                elif any(i in removed for i in indices):
                    state = "removed"
                else:
                    state = "missing"
            residues.append([int(res_id), name, state])
        first = int(np.flatnonzero(input_atoms.label_asym_id.astype(str) == asym)[0])
        chain_info = info.get(asym, {})
        chains.append(
            {
                "id": asym,
                "label": asym,
                "auth": chain_info.get("auth"),
                "entity": chain_info.get("entity"),
                "type": chain_info.get("type"),
                "ref_mol": str(input_atoms.res_name[first]) if chain_info.get("type") == "LIGAND" else None,
                "paired": paired.get(asym, []),
                "res": residues,
            }
        )
    return chains


def input_tracks(inputs: ValidityInputs, pdb_id: str) -> dict:
    """Both sides' chains with a state per residue: our final structure against the input CIF.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.

    Returns:
        {"ref": [chain], "ours": [chain]}, parity's track_data shape; "ref" is the input's label_asym_id
        chains (paired = our ids that land on it). Our states: ok, unresolved, wrong_chain, wrong_residue,
        no_counterpart, split, assembly_copy; input states: ok, unresolved, missing, removed.
    """
    input_atoms = read_input_atoms(inputs.input_cif(pdb_id))
    final = inputs.final_atoms(pdb_id)
    log = inputs.step_log(pdb_id)
    info, polymers = _chain_info(inputs.input_cif(pdb_id))
    copies = find_copies(input_atoms, final, operators(inputs.input_cif(pdb_id)))
    pairs = copies.pairs
    position_to_input = {position: index for index, position in pairs}
    polymer_asyms = set(polymers) if polymers is not None else None

    copy_atoms = set(copies.source_of)
    renamed = _renamed_names(log)
    segments = _residue_segments(final)
    states = {
        segment: _our_state(final, segment, position_to_input, input_atoms, renamed, polymer_asyms, copy_atoms)
        for segment in segments
    }
    ours = _our_chains(final, states, segments)

    removed = {i for step in (log.steps if log else []) for i in expand(step.removed_input)}
    paired: dict[str, list[str]] = {}
    for chain in ours:
        if chain["paired"] is not None and chain["paired"] == chain["label"].split("_")[0]:
            paired.setdefault(chain["paired"], []).append(chain["id"])
    ref = _input_chains(input_atoms, info, polymers, {index for index, _ in pairs}, removed, paired)
    return {"ref": ref, "ours": ours}


def check_chain_placement(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """The validity.chain_placement check of one entry.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.

    Returns:
        One finding: match when no residue is wrong_chain, wrong_residue, no_counterpart or split.
    """
    tracks = input_tracks(inputs, pdb_id)
    counts = Counter(state for chain in tracks["ours"] for _, _, state in chain["res"])
    bad = [
        [chain["label"], res_id, res_name, state]
        for chain in tracks["ours"]
        for res_id, res_name, state in chain["res"]
        if state in BAD_STATES
    ]
    status = Status.UNEXPLAINED if bad else Status.MATCH
    detail = ", ".join(f"{state} {counts[state]}" for state in BAD_STATES if counts[state])
    return [Finding(CHECK, pdb_id, status, detail, {"counts": dict(counts), "bad": bad[:_MAX_LISTED]})]
