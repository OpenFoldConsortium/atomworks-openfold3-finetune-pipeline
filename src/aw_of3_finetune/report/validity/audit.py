"""Independent audit of the cleanup step log against the input atoms and the final structure."""

from dataclasses import asdict, dataclass, field

import numpy as np

from aw_of3_finetune.report.findings import Finding, Status
from aw_of3_finetune.report.validity.copies import atom_label, expand, find_copies, operators
from aw_of3_finetune.report.validity.inputs import ValidityInputs
from aw_of3_finetune.steplog import read_input_atoms

CHECK = "validity.cleanup_audit"


@dataclass
class AuditResult:
    """What the audit found for one entry; input atoms are row indices of the input atom table."""

    kept: list[int] = field(default_factory=list)
    removed_by_step: dict[str, list[int]] = field(default_factory=dict)
    unaccounted: list[int] = field(default_factory=list)
    double_counted: list[int] = field(default_factory=list)
    renamed_unlogged: list[tuple[int, str, str]] = field(default_factory=list)
    moved: list[str] = field(default_factory=list)
    balance_errors: list[str] = field(default_factory=list)
    input_atoms: int = 0
    log_input_atoms: int | None = None
    input_count_mismatch: bool = False
    final_count_mismatch: str | None = None
    unexplained_final: list[str] = field(default_factory=list)
    has_log: bool = False

    @property
    def clean(self) -> bool:
        """True when there is a log and every problem field is empty or unset."""
        return self.has_log and not (
            self.unaccounted
            or self.double_counted
            or self.renamed_unlogged
            or self.moved
            or self.balance_errors
            or self.input_count_mismatch
            or self.final_count_mismatch
            or self.unexplained_final
        )


def audit_entry(inputs: ValidityInputs, pdb_id: str) -> AuditResult:
    """Recompute kept and removed input atoms and compare them with the step log.

    Args:
        inputs: The pipeline output to audit.
        pdb_id: Structure id.

    Returns:
        The audit result; kept, copies and the final side come from coordinates and operators, never from the log.
    """
    input_atoms = read_input_atoms(inputs.input_cif(pdb_id))
    final = inputs.final_atoms(pdb_id)
    log = inputs.step_log(pdb_id)

    copies = find_copies(input_atoms, final, operators(inputs.input_cif(pdb_id)))
    kept_pairs = copies.pairs
    result = AuditResult(
        kept=[index for index, _ in kept_pairs], moved=copies.moved, input_atoms=len(input_atoms), has_log=log is not None
    )
    resolved = np.flatnonzero(~np.isnan(final.coord).any(axis=1))
    explained = {position for _, position in kept_pairs} | set(copies.source_of)
    result.unexplained_final = [atom_label(final, int(p)) for p in resolved if p not in explained]
    n_final, n_kept, n_copies = len(resolved), len(kept_pairs), len(copies.source_of)
    logged_final = log.steps[-1].resolved_atoms if log is not None and log.steps else None
    if n_final != n_kept + n_copies or (log is not None and n_final != logged_final):
        result.final_count_mismatch = (
            f"final {n_final} resolved; kept {n_kept} + copies {n_copies} = {n_kept + n_copies}; log {logged_final}"
        )
    if log is None:
        return result
    result.log_input_atoms = log.input_atoms
    result.input_count_mismatch = log.input_atoms != len(input_atoms)

    removed: set[int] = set()
    for step in log.steps:
        indices = expand(step.removed_input)
        if indices:
            result.removed_by_step[step.name] = indices
            removed.update(indices)
    kept = set(result.kept)
    result.double_counted = sorted(kept & removed)
    result.unaccounted = [i for i in range(len(input_atoms)) if i not in kept and i not in removed]
    result.balance_errors = log.balance_errors()

    logged_after = {}
    for step in log.steps:
        for index, _, after in step.renamed:
            logged_after[index] = after
    for index, position in kept_pairs:
        before = f"{input_atoms.res_name[index]}:{input_atoms.atom_name[index]}"
        actual = f"{final.res_name[position]}:{final.atom_name[position]}"
        expected = logged_after.get(index, before)
        if actual != expected:
            result.renamed_unlogged.append((index, before, actual))
    return result


def check_cleanup_audit(inputs: ValidityInputs, pdb_id: str) -> list[Finding]:
    """The validity.cleanup_audit check of one entry.

    Args:
        inputs: The pipeline output to audit.
        pdb_id: Structure id.

    Returns:
        One finding: match when the log accounts for every input atom, every resolved final atom is a kept
        input atom or an operator copy of one, and the counts agree; else unexplained.
    """
    result = audit_entry(inputs, pdb_id)
    parts = []
    if not result.has_log:
        parts.append("no step log")
    if result.input_count_mismatch:
        parts.append(f"input_count_mismatch (log {result.log_input_atoms}, input {result.input_atoms})")
    for name in ("unaccounted", "double_counted", "renamed_unlogged", "moved", "unexplained_final"):
        if getattr(result, name):
            parts.append(f"{name} {len(getattr(result, name))}")
    if result.final_count_mismatch:
        parts.append(f"final_count_mismatch ({result.final_count_mismatch})")
    if result.balance_errors:
        unbalanced = [error.split(":")[0] for error in result.balance_errors]
        parts.append(f"unbalanced steps: {', '.join(unbalanced)}")
    status = Status.MATCH if result.clean else Status.UNEXPLAINED
    return [Finding(CHECK, pdb_id, status, "; ".join(parts), asdict(result))]
