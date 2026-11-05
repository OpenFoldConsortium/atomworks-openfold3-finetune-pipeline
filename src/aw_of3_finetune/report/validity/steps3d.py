"""Per-entry data for the 3D view of what each cleanup step did to the input atoms."""

import logging
from dataclasses import dataclass, field

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import cKDTree

from aw_of3_finetune.report.validity.copies import expand, find_copies, operators
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.steplog import STEP_PHASES, StepLog, read_input_atoms

logger = logging.getLogger(__name__)

PHASE_NAMES = {"parse": "Parse", "assembly": "Assembly", "cleanup": "Cleanup filters", "final": "Final trim"}
ASSEMBLY_STEP = "build_assembly_1"
STEP_VIEW_ERROR = "validity.step_view_error"
NO_OPERATOR = "no pdbx_struct_oper_list operator"
NO_OWNER = "removed_created group has no created group"
_MATCH_DISTANCE = 0.05
_XYZ_DECIMALS = 2


@dataclass
class _CreatedGroup:
    """Atoms of one assembly copy: the operator, its source input atoms and when each was removed."""

    transformation_id: str
    created_at: int
    sources: list[int]
    removed_at: dict[int, int] = field(default_factory=dict)


def _operator_ids(transformation_id: str, known: set[str]) -> list[str] | None:
    """Split a transformation_id into operator ids; atomworks joins multi-step ids without a separator.

    Args:
        transformation_id: The group's transformation_id.
        known: The ids of the operator list.

    Returns:
        The operator ids in application order, or None if the id cannot be split into known ids.
    """
    if transformation_id in known:
        return [transformation_id]
    for oper_id in sorted(known, key=len, reverse=True):
        if oper_id and transformation_id.startswith(oper_id):
            rest = _operator_ids(transformation_id[len(oper_id):], known)
            if rest is not None:
                return [oper_id, *rest]
    return None


def _created_groups(log: StepLog) -> tuple[list[_CreatedGroup], list[dict]]:
    """The created groups of a log with the step at which each source atom was removed, and orphan removals."""
    groups: list[_CreatedGroup] = []
    skipped: list[dict] = []
    for step_index, step in enumerate(log.steps):
        for removed in step.removed_created:
            sources = set(expand(removed.source_input))
            owner = next(
                (g for g in groups if g.transformation_id == removed.transformation_id and sources <= set(g.sources)),
                None,
            )
            if owner is None:
                skipped.append({"transformation_id": removed.transformation_id, "reason": NO_OWNER})
            else:
                owner.removed_at.update({source: step_index for source in sources})
        for created in step.created:
            groups.append(_CreatedGroup(created.transformation_id, step_index, sorted(expand(created.source_input))))
    return groups, skipped


def _chain_table(final: AtomArray) -> tuple[list[dict], np.ndarray]:
    """The final chain copies in order of appearance and each final atom's chain index."""
    labels = final.label_asym_id.astype(str)
    order = list(dict.fromkeys(labels.tolist()))
    hetero = final.hetero if "hetero" in final.get_annotation_categories() else np.zeros(len(final), dtype=bool)
    chains = []
    for label in order:
        first = int(np.flatnonzero(labels == label)[0])
        chains.append({"id": label, "res_name": str(final.res_name[first]), "hetero": bool(hetero[first])})
    return chains, np.array([order.index(label) for label in labels], dtype=int)


def _columns(atoms: AtomArray, indices: list[int]) -> dict[str, list]:
    """Name, residue and element columns of the given atoms."""
    selected = atoms[indices]
    return {
        "res_name": [str(x) for x in selected.res_name],
        "res_id": [int(x) for x in selected.res_id],
        "atom_name": [str(x) for x in selected.atom_name],
        "element": [str(x) for x in selected.element],
        "hetero": [bool(x) for x in selected.hetero],
    }


def _input_columns(input_atoms: AtomArray, log: StepLog | None, chain_of_input: np.ndarray) -> dict[str, list]:
    """The input-atom columns of the view, with label_asym_id and the names after logged renames."""
    removed_at = np.full(len(input_atoms), -1, dtype=int)
    res_names = [str(x) for x in input_atoms.res_name]
    atom_names = [str(x) for x in input_atoms.atom_name]
    if log is not None:
        for step_index, step in enumerate(log.steps):
            for index in expand(step.removed_input):
                if removed_at[index] == -1:
                    removed_at[index] = step_index
            for index, _, after in step.renamed:
                res_names[index], atom_names[index] = after.split(":", 1)
    return {
        "removed_at": removed_at.tolist(),
        "chain": chain_of_input.tolist(),
        "label_asym_id": [str(x) for x in input_atoms.label_asym_id],
        **_columns(input_atoms, list(range(len(input_atoms)))),
        "renamed_res_name": res_names,
        "renamed_atom_name": atom_names,
        "xyz": np.round(input_atoms.coord, _XYZ_DECIMALS).ravel().tolist(),
    }


def _created_columns(
    log: StepLog | None,
    input_atoms: AtomArray,
    final: AtomArray,
    final_chain: np.ndarray,
    unpaired: list[int],
    oper_list: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, list], list[dict]]:
    """The created-atom columns and the skipped groups.

    Logged groups are drawn at operator x source, with their chain from the nearest unpaired final atom.
    Every other resolved unpaired final atom follows as a row of its own (created at build_assembly_1, or None).
    """
    final_xyz = final.coord[unpaired]
    resolved = ~np.isnan(final_xyz).any(axis=1)
    unpaired_resolved = np.array(unpaired, dtype=int)[resolved]
    tree = cKDTree(final_xyz[resolved]) if resolved.any() else None

    groups, skipped = _created_groups(log) if log is not None else ([], [])
    sources: list[int] = []
    created_at: list[int | None] = []
    removed_at: list[int] = []
    xyz: list[np.ndarray] = []
    for group in groups:
        steps = _operator_ids(group.transformation_id, set(oper_list))
        if steps is None:
            logger.warning("%s: no operator for transformation_id %r; skipping its copy", log.pdb_id, group.transformation_id)
            skipped.append({"transformation_id": group.transformation_id, "reason": NO_OPERATOR})
            continue
        coord = input_atoms.coord[group.sources]
        for oper_id in steps:
            matrix, vector = oper_list[oper_id]
            coord = coord @ matrix.T + vector
        sources += group.sources
        created_at += [group.created_at] * len(group.sources)
        removed_at += [group.removed_at.get(source, -1) for source in group.sources]
        xyz.append(coord)
    coords = np.concatenate(xyz) if xyz else np.zeros((0, 3))

    chain: list[int] = []
    claimed: set[int] = set()
    for source, removed, point in zip(sources, removed_at, coords):
        match = -1
        if removed == -1 and tree is not None:
            distance, nearest = tree.query(point)
            position = int(unpaired_resolved[nearest])
            if distance <= _MATCH_DISTANCE and final.atom_name[position] == input_atoms.atom_name[source]:
                match = int(final_chain[position])
                claimed.add(position)
        chain.append(match)
    columns = {"created_at": created_at, "removed_at": removed_at, "chain": chain, **_columns(input_atoms, sources)}

    step_names = [step.name for step in log.steps] if log is not None else []
    assembly_step = step_names.index(ASSEMBLY_STEP) if ASSEMBLY_STEP in step_names else None
    extra = [int(p) for p in unpaired_resolved if int(p) not in claimed]
    columns["created_at"] += [assembly_step] * len(extra)
    columns["removed_at"] += [-1] * len(extra)
    columns["chain"] += [int(final_chain[p]) for p in extra]
    for name, values in _columns(final, extra).items():
        columns[name] += values
    all_xyz = np.concatenate([coords, final.coord[extra].reshape(-1, 3)])
    columns["xyz"] = np.round(all_xyz, _XYZ_DECIMALS).ravel().tolist()
    return columns, sorted(skipped, key=lambda group: group["transformation_id"])


def _phases(log: StepLog | None) -> tuple[list[dict], list[dict] | None]:
    """The phases with their step names, and the step records for the view (None without a log)."""
    if log is None:
        return [], None
    phases: list[dict] = []
    steps = []
    for step in log.steps:
        name = PHASE_NAMES[STEP_PHASES.get(step.name, step.phase)]
        if not phases or phases[-1]["name"] != name:
            phases.append({"name": name, "steps": []})
        phases[-1]["steps"].append(step.name)
        steps.append(
            {
                "name": step.name,
                "phase": name,
                "description": step.description,
                "removed": sum(end - start + 1 for start, end in step.removed_input)
                + sum(g.n_atoms for g in step.removed_created),
                "created": sum(g.n_atoms for g in step.created),
                "renamed": len(step.renamed),
                "unresolved_added": step.unresolved_added,
            }
        )
    return phases, steps


def step_view_data(inputs: ValidityInputs, pdb_id: str) -> dict:
    """Everything the 3D step view draws for one entry; JSON-serialisable, with no colors.

    Args:
        inputs: The pipeline output.
        pdb_id: Structure id.

    Returns:
        The phases, steps, final chains, input-atom and created-atom columns, skipped created groups
        ({transformation_id, reason}) and before_after_only.
    """
    log = inputs.step_log(pdb_id)
    input_atoms = read_input_atoms(inputs.input_cif(pdb_id))
    final = inputs.final_atoms(pdb_id)
    chains, final_chain = _chain_table(final)
    oper_list = operators(inputs.input_cif(pdb_id))
    pairs = find_copies(input_atoms, final, oper_list).pairs
    chain_of_input = np.full(len(input_atoms), -1, dtype=int)
    for index, position in pairs:
        chain_of_input[index] = final_chain[position]
    paired_positions = {position for _, position in pairs}
    unpaired = [p for p in range(len(final)) if p not in paired_positions]
    phases, steps = _phases(log)
    created, skipped = _created_columns(log, input_atoms, final, final_chain, unpaired, oper_list)
    return {
        "phases": phases,
        "steps": steps,
        "chains": chains,
        "input": _input_columns(input_atoms, log, chain_of_input),
        "created": created,
        "skipped_groups": skipped,
        "before_after_only": log is None,
    }
