"""Validation stage: checks over a prepared finetuning dataset."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from aw_of3_finetune.validate.alignments import (
    check_v13_msa_query_rows,
    check_v14_rep_ids_by_sequence,
    check_v15_templates_resolve,
    check_v16_template_dates,
    check_v17_release_date,
)
from aw_of3_finetune.validate.context import ValidationContext
from aw_of3_finetune.validate.dataset_load import check_v21_dataset_loads
from aw_of3_finetune.validate.findings import Finding, Severity, ValidationReport
from aw_of3_finetune.validate.ligands import (
    check_v4_ligand_present,
    check_v8_reference_mols_exist,
    check_v9_v10_ref_atoms_match_crystal,
    check_v11_one_chemistry_per_mol_id,
    check_v12_ref_mols_sanitize,
    check_v18_ligands_not_in_polymers,
    check_v19_ligand_copies_separate,
)
from aw_of3_finetune.validate.structure import (
    check_v1_chains_accounted,
    check_v2_polymer_sequences,
    check_v3_atoms_match_source,
    check_v5_finite_coordinates,
    check_v6_cache_matches_structure,
    check_v7_residue_numbering,
)

CHECKS: dict[str, Callable[[ValidationContext], list[Finding]]] = {
    "V1": check_v1_chains_accounted,
    "V2": check_v2_polymer_sequences,
    "V3": check_v3_atoms_match_source,
    "V4": check_v4_ligand_present,
    "V5": check_v5_finite_coordinates,
    "V6": check_v6_cache_matches_structure,
    "V7": check_v7_residue_numbering,
    "V8": check_v8_reference_mols_exist,
    "V9": check_v9_v10_ref_atoms_match_crystal,
    "V10": check_v9_v10_ref_atoms_match_crystal,
    "V11": check_v11_one_chemistry_per_mol_id,
    "V12": check_v12_ref_mols_sanitize,
    "V13": check_v13_msa_query_rows,
    "V14": check_v14_rep_ids_by_sequence,
    "V15": check_v15_templates_resolve,
    "V16": check_v16_template_dates,
    "V17": check_v17_release_date,
    "V18": check_v18_ligands_not_in_polymers,
    "V19": check_v19_ligand_copies_separate,
    "V21": check_v21_dataset_loads,
}


# Short names for reports; the check functions' docstrings say exactly what each one tests.
CHECK_TITLES: dict[str, str] = {
    "V1": "Source chains kept or dropped for a known reason",
    "V2": "Polymer sequences match the source",
    "V3": "Atoms match the source CIF",
    "V4": "Entry has a ligand",
    "V5": "Coordinates are finite",
    "V6": "Dataset cache matches the structure",
    "V7": "Residue numbering matches the sequence",
    "V8": "Every ligand has a reference SDF",
    "V9": "Reference SDF elements match the crystal",
    "V10": "Reference SDF atom names match the crystal",
    "V11": "One chemistry per reference molecule",
    "V12": "Reference SDFs sanitize in RDKit",
    "V13": "MSA query row matches the chain sequence",
    "V14": "Representative IDs follow the sequence",
    "V15": "Template IDs resolve to structure files",
    "V16": "Templates predate their entry",
    "V17": "Entry has a real release date",
    "V18": "Ligands are not inside polymer chains",
    "V19": "Ligand copies stay in separate chains",
    "V21": "Entry loads through OpenFold3's dataset",
}

def run_validation(ctx: ValidationContext, checks: Iterable[str] | None = None) -> ValidationReport:
    """Run the selected checks and collect their findings.

    Args:
        ctx: Validation context.
        checks: Check IDs to run; all registered checks when None.

    Returns:
        Report; checks sharing a function (V9/V10) run once and only requested IDs are kept; unknown IDs raise ValueError; a check that raises becomes an E finding with ``entry=None``.
    """
    ids = list(CHECKS) if checks is None else list(checks)
    unknown = [c for c in ids if c not in CHECKS]
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(unknown)}")
    findings: list[Finding] = []
    ran: set[Callable] = set()
    for check_id in ids:
        if CHECKS[check_id] in ran:
            continue
        ran.add(CHECKS[check_id])
        try:
            found = CHECKS[check_id](ctx)
            findings.extend(f for f in found if f.check in ids or f.check not in CHECKS)
        except Exception as exc:
            findings.append(Finding(check_id, Severity.ERROR, None, f"check crashed: {type(exc).__name__}: {exc}"))
    try:
        entries_in = ctx.entries()
    except Exception as exc:
        entries_in = []
        findings.append(Finding("V0", Severity.ERROR, None, f"dataset cache unreadable: {exc}"))
    return ValidationReport(findings=findings, entries_in=entries_in)


__all__ = ["CHECKS", "Finding", "Severity", "ValidationContext", "ValidationReport", "run_validation"]
