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
from aw_of3_finetune.validate.check_ids import (
    ATOMS_MATCH_SOURCE,
    CACHE_MATCHES_STRUCTURE,
    CHAINS_ACCOUNTED,
    DATASET_CACHE_READABLE,
    DATASET_LOADS,
    FINITE_COORDINATES,
    HAS_LIGAND,
    LIGANDS_NOT_IN_POLYMERS,
    LIGAND_COPIES_SEPARATE,
    MSA_QUERY_ROWS,
    ONE_CHEMISTRY_PER_MOL_ID,
    POLYMER_SEQUENCES,
    REFERENCE_SDFS_EXIST,
    REFERENCE_SDFS_SANITIZE,
    RELEASE_DATE,
    REP_IDS_BY_SEQUENCE,
    RESIDUE_NUMBERING,
    SDF_ATOM_NAMES_MATCH,
    SDF_ELEMENTS_MATCH,
    TEMPLATES_RESOLVE,
    TEMPLATE_DATES,
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
    CHAINS_ACCOUNTED: check_v1_chains_accounted,
    POLYMER_SEQUENCES: check_v2_polymer_sequences,
    ATOMS_MATCH_SOURCE: check_v3_atoms_match_source,
    HAS_LIGAND: check_v4_ligand_present,
    FINITE_COORDINATES: check_v5_finite_coordinates,
    CACHE_MATCHES_STRUCTURE: check_v6_cache_matches_structure,
    RESIDUE_NUMBERING: check_v7_residue_numbering,
    REFERENCE_SDFS_EXIST: check_v8_reference_mols_exist,
    SDF_ELEMENTS_MATCH: check_v9_v10_ref_atoms_match_crystal,
    SDF_ATOM_NAMES_MATCH: check_v9_v10_ref_atoms_match_crystal,
    ONE_CHEMISTRY_PER_MOL_ID: check_v11_one_chemistry_per_mol_id,
    REFERENCE_SDFS_SANITIZE: check_v12_ref_mols_sanitize,
    MSA_QUERY_ROWS: check_v13_msa_query_rows,
    REP_IDS_BY_SEQUENCE: check_v14_rep_ids_by_sequence,
    TEMPLATES_RESOLVE: check_v15_templates_resolve,
    TEMPLATE_DATES: check_v16_template_dates,
    RELEASE_DATE: check_v17_release_date,
    LIGANDS_NOT_IN_POLYMERS: check_v18_ligands_not_in_polymers,
    LIGAND_COPIES_SEPARATE: check_v19_ligand_copies_separate,
    DATASET_LOADS: check_v21_dataset_loads,
}


# Short names for reports; the check functions' docstrings say exactly what each one tests.
CHECK_TITLES: dict[str, str] = {
    CHAINS_ACCOUNTED: "Source chains kept or dropped for a known reason",
    POLYMER_SEQUENCES: "Polymer sequences match the source",
    ATOMS_MATCH_SOURCE: "Atoms match the source CIF",
    HAS_LIGAND: "Entry has a ligand",
    FINITE_COORDINATES: "Coordinates are finite",
    CACHE_MATCHES_STRUCTURE: "Dataset cache matches the structure",
    RESIDUE_NUMBERING: "Residue numbering matches the sequence",
    REFERENCE_SDFS_EXIST: "Every ligand has a reference SDF",
    SDF_ELEMENTS_MATCH: "Reference SDF elements match the crystal",
    SDF_ATOM_NAMES_MATCH: "Reference SDF atom names match the crystal",
    ONE_CHEMISTRY_PER_MOL_ID: "One chemistry per reference molecule",
    REFERENCE_SDFS_SANITIZE: "Reference SDFs sanitize in RDKit",
    MSA_QUERY_ROWS: "MSA query row matches the chain sequence",
    REP_IDS_BY_SEQUENCE: "Representative IDs follow the sequence",
    TEMPLATES_RESOLVE: "Template IDs resolve to structure files",
    TEMPLATE_DATES: "Templates predate their entry",
    RELEASE_DATE: "Entry has a real release date",
    LIGANDS_NOT_IN_POLYMERS: "Ligands are not inside polymer chains",
    LIGAND_COPIES_SEPARATE: "Ligand copies stay in separate chains",
    DATASET_LOADS: "Entry loads through OpenFold3's dataset",
}

def run_validation(ctx: ValidationContext, checks: Iterable[str] | None = None) -> ValidationReport:
    """Run the selected checks and collect their findings.

    Args:
        ctx: Validation context.
        checks: Check IDs to run; all registered checks when None.

    Returns:
        Report; checks sharing a function (V9 and V10, the reference SDF element and atom-name checks) run once and only requested IDs are kept; unknown IDs raise ValueError; a check that raises becomes an E finding with ``entry=None``.
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
        findings.append(Finding(DATASET_CACHE_READABLE, Severity.ERROR, None, f"dataset cache unreadable: {exc}"))
    return ValidationReport(findings=findings, entries_in=entries_in)


__all__ = ["CHECKS", "Finding", "Severity", "ValidationContext", "ValidationReport", "run_validation"]
