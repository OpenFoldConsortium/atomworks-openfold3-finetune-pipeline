"""Per-entry log of which input atoms each parse/cleanup step removed, created or renamed.

Input atoms are indexed by row position in the heavy-atom table of model 1. An atom's
identity across steps is (transformation, coordinates rounded to 0.001 A): atoms without a
transformation_id (parse steps) and atoms of a chain's identity copy (the transformation
whose coordinates match the input) match input atoms by coordinates alone; atoms of every
other assembly copy never match input atoms and are tracked as created atoms. When several
input atoms share coordinates and fewer remain, the removal goes to the highest input index.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from pydantic import BaseModel

STEP_PHASES: dict[str, str] = {
    "input": "parse",
    "altloc_highest_occupancy": "parse",
    "atomworks_parse": "parse",
    "build_assembly_1": "assembly",
    "convert_MSE_to_MET": "cleanup",
    "remove_waters": "cleanup",
    "remove_crystallization_aids": "cleanup",
    "remove_small_polymers": "cleanup",
    "remove_fully_unknown_polymers": "cleanup",
    "remove_clashing_chains": "cleanup",
    "remove_non_CCD_atoms": "cleanup",
    "remove_chains_with_CA_gaps": "cleanup",
    "prefilter_bonds": "final",
    "remove_unresolved_covalent_ligand_atoms": "final",
    "remove_terminal_atoms": "final",
}

logger = logging.getLogger(__name__)

_HYDROGENS = {"H", "D"}
_COORD_DECIMALS = 3


def read_input_atoms(cif_path: Path) -> AtomArray:
    """Read the input atom table: heavy atoms of the lowest model, all altlocs, file order.

    Args:
        cif_path: Path to the input mmCIF file.

    Returns:
        AtomArray with label_asym_id, occupancy and label_alt_id annotations.
    """
    cif = pdbx.CIFFile.read(cif_path)
    block = cif.block
    atoms = pdbx.get_structure(
        cif,
        model=1,
        altloc="all",
        extra_fields=["label_asym_id", "occupancy"],
        use_author_fields=False,
    )
    atom_site = block["atom_site"]
    model_nums = atom_site["pdbx_PDB_model_num"].as_array(int)
    in_model = model_nums == model_nums.min()
    alt_ids = atom_site["label_alt_id"].as_array(str)[in_model]
    elements = np.char.upper(atom_site["type_symbol"].as_array(str)[in_model])
    heavy = ~np.isin(elements, list(_HYDROGENS))
    atoms.set_annotation("label_alt_id", alt_ids)
    return atoms[heavy]


class CreatedGroup(BaseModel):
    """Atoms of one assembly copy that did not exist in the input."""

    transformation_id: str
    source_input: list[tuple[int, int]]
    n_atoms: int


class StepRecord(BaseModel):
    """What one step did to the input atoms."""

    name: str
    phase: str
    description: str
    resolved_atoms: int
    removed_input: list[tuple[int, int]] = []
    removed_created: list[CreatedGroup] = []
    created: list[CreatedGroup] = []
    renamed: list[tuple[int, str, str]] = []
    unresolved_added: int = 0
    unresolved_removed: int = 0
    bonds_removed: int = 0


class StepLog(BaseModel):
    """The full step history of one entry."""

    version: Literal[1] = 1
    pdb_id: str
    input_atoms: int
    kept_input: list[tuple[int, int]] = []
    steps: list[StepRecord]

    def to_json(self) -> str:
        """Serialise the log.

        Returns:
            JSON text.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, text: str) -> "StepLog":
        """Parse a log written by to_json.

        Args:
            text: JSON text.

        Returns:
            The StepLog.
        """
        return cls.model_validate_json(text)

    def unattributed_input(self) -> list[int]:
        """Find input atoms that are neither kept in the final step nor removed by a step.

        Returns:
            Input indices; empty when the log is consistent.
        """
        accounted = set()
        for start, end in self.kept_input:
            accounted.update(range(start, end + 1))
        for step in self.steps:
            for start, end in step.removed_input:
                accounted.update(range(start, end + 1))
        return [i for i in range(self.input_atoms) if i not in accounted]

    def balance_errors(self) -> list[str]:
        """Find steps whose resolved atom count is not the previous count minus removed plus created.

        Returns:
            One message per unbalanced step (the ``input`` step is exempt); empty when balanced.
        """
        errors = []
        for previous, step in zip(self.steps, self.steps[1:]):
            n_removed = _count(step.removed_input) + sum(g.n_atoms for g in step.removed_created)
            n_created = sum(g.n_atoms for g in step.created)
            expected = previous.resolved_atoms - n_removed + n_created
            if expected != step.resolved_atoms:
                errors.append(
                    f"{step.name}: {previous.resolved_atoms} - {n_removed} removed + "
                    f"{n_created} created = {expected}, but {step.resolved_atoms} resolved"
                )
        return errors


def _count(ranges: list[tuple[int, int]]) -> int:
    """Number of indices in inclusive (start, end) ranges."""
    return sum(end - start + 1 for start, end in ranges)


def _ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Compress indices into sorted inclusive (start, end) ranges."""
    ranges: list[tuple[int, int]] = []
    for index in sorted(indices):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], index)
        else:
            ranges.append((index, index))
    return ranges


def _keys(atoms: AtomArray) -> list[tuple[float, float, float] | None]:
    """Rounded-coordinate key per atom; None for unresolved atoms."""
    rounded = np.round(atoms.coord, _COORD_DECIMALS)
    resolved = ~np.isnan(atoms.coord).any(axis=1)
    return [tuple(c) if ok else None for c, ok in zip(rounded.tolist(), resolved)]


def _label(atoms: AtomArray, i: int) -> str:
    return f"{atoms.res_name[i]}:{atoms.atom_name[i]}"


def _chain_bases(atoms: AtomArray) -> np.ndarray:
    """Chain per atom without the copy suffix: the chain_iid prefix before "_", else chain_id."""
    if "chain_iid" in atoms.get_annotation_categories():
        return np.array([c.split("_")[0] for c in atoms.chain_iid.astype(str)], dtype=object)
    return atoms.chain_id.astype(str).astype(object)


def _atom_id(atoms: AtomArray, bases: np.ndarray, i: int) -> tuple[str, int, str, str]:
    """(chain base, res_id, ins_code, atom_name) of atom i."""
    return (str(bases[i]), int(atoms.res_id[i]), str(atoms.ins_code[i]), str(atoms.atom_name[i]))


class StepRecorder:
    """Records each step's effect on the input atoms as the pipeline runs.

    A step that fails to record logs a warning and stops all further recording; ``failed``
    is then True and the log should not be written (it would be incomplete).
    """

    def __init__(self, input_atoms: AtomArray, pdb_id: str, keep_snapshots: bool = False):
        """Start from the input atom table.

        Args:
            input_atoms: Input atoms, indexed by row position.
            pdb_id: Entry identifier for the log.
            keep_snapshots: Keep a copy of the atoms after each step (for step CIFs).
        """
        self._pdb_id = pdb_id
        self._n_input = len(input_atoms)
        self._keep_snapshots = keep_snapshots
        self.failed = False
        self._records: list[StepRecord] = []
        self._snapshots: list[tuple[str, AtomArray]] = []
        input_keys = _keys(input_atoms)
        self._input_keys = {k for k in input_keys if k is not None}
        # input atoms still present: key -> input indices, lowest first
        self._present: dict[tuple, list[int]] = defaultdict(list)
        self._labels: dict[int, str] = {}
        # (chain base, res_id, ins_code, atom_name) of each input atom, as last seen
        self._atom_ids: dict[int, tuple] = {}
        input_bases = _chain_bases(input_atoms)
        for i, key in enumerate(input_keys):
            if key is not None:
                self._present[key].append(i)
            self._labels[i] = _label(input_atoms, i)
            self._atom_ids[i] = _atom_id(input_atoms, input_bases, i)
        # created atoms still present: key -> [(group id, position in group)]
        self._created_present: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        self._groups: list[tuple[str, list[int | None]]] = []
        # chain base -> transformation_id of its identity copy (None: no identity copy)
        self._identity: dict[str, str | None] = {}
        self._unresolved = 0

    def record(
        self,
        name: str,
        description: str,
        atoms: AtomArray | Callable[[], AtomArray],
        *,
        bonds_removed: int = 0,
    ) -> None:
        """Compare atoms to the previous step and store what changed; never raises.

        Args:
            name: Step name, a key of STEP_PHASES.
            description: One-line description of the step.
            atoms: The atoms after the step, or a function returning them.
            bonds_removed: Number of bonds the step removed.
        """
        if self.failed:
            return
        try:
            if not isinstance(atoms, AtomArray):
                atoms = atoms()
            self._record(name, description, atoms, bonds_removed)
        except Exception as exc:  # the log must never fail the pipeline
            self.failed = True
            logger.warning(
                "%s: cleanup step log stopped at step %s (%s: %s); no log will be written",
                self._pdb_id, name, type(exc).__name__, exc,
            )

    def _record(self, name: str, description: str, atoms: AtomArray, bonds_removed: int) -> None:
        """Match atoms to the previous step's by identity key and append the StepRecord."""
        phase = STEP_PHASES[name]
        bases = _chain_bases(atoms)
        keys = self._atom_keys(atoms, bases)
        n_unresolved = sum(k is None for k in keys)
        by_key: dict[tuple, list[int]] = defaultdict(list)
        for position, key in enumerate(keys):
            if key is not None:
                by_key[key].append(position)
        alive_before = sorted(i for indices in self._present.values() for i in indices)

        removed_input: list[int] = []
        removed_created: dict[int, list[int]] = defaultdict(list)
        renamed: list[tuple[int, str, str]] = []
        new_created_positions: list[int] = []

        for key in set(self._present) | set(self._created_present) | set(by_key):
            positions = by_key.get(key, [])
            previous = self._present.get(key, [])
            n_kept = min(len(previous), len(positions))
            removed_input.extend(previous[n_kept:])
            if previous:
                self._present[key] = previous[:n_kept]
            for index, position in zip(previous[:n_kept], positions):
                self._atom_ids[index] = _atom_id(atoms, bases, position)
                after = _label(atoms, position)
                if after != self._labels[index]:
                    renamed.append((index, self._labels[index], after))
                    self._labels[index] = after
            rest = positions[n_kept:]
            previous_created = self._created_present.get(key, [])
            n_kept_created = min(len(previous_created), len(rest))
            for group_id, group_pos in previous_created[n_kept_created:]:
                removed_created[group_id].append(group_pos)
            if previous_created:
                self._created_present[key] = previous_created[:n_kept_created]
            new_created_positions.extend(rest[n_kept_created:])

        created = self._add_created(atoms, bases, new_created_positions, keys, alive_before)
        removed_created_groups = [
            self._group_view(group_id, group_positions)
            for group_id, group_positions in sorted(removed_created.items())
        ]

        self._records.append(
            StepRecord(
                name=name,
                phase=phase,
                description=description,
                resolved_atoms=len(atoms) - n_unresolved,
                removed_input=_ranges(removed_input),
                removed_created=removed_created_groups,
                created=created,
                renamed=sorted(renamed),
                unresolved_added=max(0, n_unresolved - self._unresolved),
                unresolved_removed=max(0, self._unresolved - n_unresolved),
                bonds_removed=bonds_removed,
            )
        )
        self._unresolved = n_unresolved
        if self._keep_snapshots:
            self._snapshots.append((name, atoms.copy()))

    def _atom_keys(self, atoms: AtomArray, bases: np.ndarray) -> list[tuple | None]:
        """Identity key per atom: coordinates for input-matchable atoms, else (transformation, coordinates)."""
        coord_keys = _keys(atoms)
        if "transformation_id" not in atoms.get_annotation_categories():
            return coord_keys
        transformations = atoms.transformation_id.astype(str)
        self._resolve_identity(bases, transformations, coord_keys)
        return [
            None if key is None
            else key if self._identity.get(base) == transformation
            else (transformation, key)
            for key, base, transformation in zip(coord_keys, bases, transformations)
        ]

    def _resolve_identity(self, bases: np.ndarray, transformations: np.ndarray, keys: list) -> None:
        """Pick each new chain base's identity copy: the first whose resolved atoms all match the input."""
        for base in dict.fromkeys(bases.tolist()):
            if base in self._identity:
                continue
            self._identity[base] = None
            in_chain = bases == base
            for transformation in dict.fromkeys(transformations[in_chain].tolist()):
                copy_keys = [
                    keys[i] for i in np.flatnonzero(in_chain & (transformations == transformation))
                    if keys[i] is not None
                ]
                if copy_keys and all(k in self._input_keys for k in copy_keys):
                    self._identity[base] = transformation
                    break

    def _add_created(
        self,
        atoms: AtomArray,
        bases: np.ndarray,
        positions: list[int],
        keys: list,
        alive_before: list[int],
    ) -> list[CreatedGroup]:
        """Group new atoms by (transformation_id, chain) and look up their source input atoms."""
        if not positions:
            return []
        categories = atoms.get_annotation_categories()
        chains = atoms.chain_iid if "chain_iid" in categories else atoms.chain_id
        transformations = (
            atoms.transformation_id if "transformation_id" in categories else np.full(len(atoms), "")
        )
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for position in sorted(positions):
            grouped[(str(transformations[position]), str(chains[position]))].append(position)

        source_of: dict[tuple, int] = {}
        for index in alive_before:
            source_of.setdefault(self._atom_ids[index], index)

        created = []
        for (transformation_id, _), group_positions in grouped.items():
            source = [source_of.get(_atom_id(atoms, bases, p)) for p in group_positions]
            group_id = len(self._groups)
            self._groups.append((transformation_id, source))
            for group_pos, position in enumerate(group_positions):
                self._created_present[keys[position]].append((group_id, group_pos))
            created.append(self._group_view(group_id, list(range(len(group_positions)))))
        return created

    def _group_view(self, group_id: int, group_positions: list[int]) -> CreatedGroup:
        transformation_id, source = self._groups[group_id]
        indices = [source[p] for p in group_positions if source[p] is not None]
        return CreatedGroup(
            transformation_id=transformation_id,
            source_input=_ranges(indices),
            n_atoms=len(group_positions),
        )

    def log(self) -> StepLog:
        """Build the log from the steps recorded so far.

        Returns:
            The StepLog.
        """
        kept = [i for indices in self._present.values() for i in indices]
        return StepLog(
            pdb_id=self._pdb_id,
            input_atoms=self._n_input,
            kept_input=_ranges(kept),
            steps=list(self._records),
        )

    def snapshots(self) -> list[tuple[str, AtomArray]]:
        """Return the atoms after each recorded step; empty unless keep_snapshots was set.

        Returns:
            (step name, atoms) pairs in recording order.
        """
        return list(self._snapshots)
