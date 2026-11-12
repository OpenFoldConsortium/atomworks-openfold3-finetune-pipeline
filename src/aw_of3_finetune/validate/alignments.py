"""MSA, template and release-date checks V13-V17."""

from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import yaml
from biotite.structure.io.pdbx import CIFFile
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings

from aw_of3_finetune.constants import PREPARE_RECORD_FILE, TEMPLATE_CUTOFF_FILENAME, TEMPLATE_CUTOFF_NOTE
from aw_of3_finetune.msa import alignment_rep_id
from aw_of3_finetune.validate.context import ValidationContext
from aw_of3_finetune.validate.findings import Finding, Severity
from aw_of3_finetune.validate.structure import _entity_sequences, _per_entry

PLACEHOLDER_RELEASE_DATE = "1970-01-01"


def _aligned_chains(ctx: ValidationContext, entry: str) -> list[tuple[str, str, str]]:
    """Return (chain key, rep_id, sequence) for each chain with an alignment_representative_id."""
    entities = _entity_sequences(CIFFile.read(ctx.augmented_cif(entry)).block)
    chains = ctx.dataset_cache.structure_data[entry].chains
    return [
        (key, c.alignment_representative_id, entities.get(str(c.entity_id), ""))
        for key, c in chains.items()
        if c.alignment_representative_id
    ]


def _first_a3m_sequence(path) -> str:
    """Return the first record of an a3m with lowercase insertions removed."""
    seq: list[str] = []
    started = False
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started:
            seq.append(line.strip())
    return re.sub("[a-z]", "", "".join(seq))


def _sources_read_by_train_yml(ctx: ValidationContext) -> set[str] | None:
    """The alignment stems OpenFold3 reads with train.yml's train msa section.

    A stem is read if it is a key of max_seq_counts and is either stacked (aln_order)
    or paired on (msas_to_pair). Values missing from the section are OpenFold3's defaults.

    Args:
        ctx: Validation context.

    Returns:
        The stems, or None when there is no train.yml.
    """
    path = ctx.output_dir / "train.yml"
    if not path.exists():
        return None
    config = yaml.safe_load(path.read_text()) or {}
    section = (
        config.get("dataset_configs", {}).get("train", {}).get("weighted-pdb", {}).get("config", {}).get("msa", {})
    ) or {}
    defaults = MSASettings()
    counts = section.get("max_seq_counts", defaults.max_seq_counts)
    used = set(section.get("aln_order", defaults.aln_order)) | set(section.get("msas_to_pair", defaults.msas_to_pair))
    return set(counts) & used


def _check_v13(ctx: ValidationContext, entry: str, read_by_train_yml: set[str] | None) -> list[Finding]:
    findings = []
    for key, rep_id, sequence in _aligned_chains(ctx, entry):
        rep_dir = ctx.msa_dir / "of3_alignments" / rep_id
        a3ms = sorted(rep_dir.glob("*.a3m")) if rep_dir.is_dir() else []
        if not a3ms:
            findings.append(
                Finding("V13", Severity.ERROR, entry, f"chain {key}: missing {rep_dir}/<source>.a3m",
                        {"chain": key, "rep_id": rep_id})
            )  # fmt: skip
        for path in a3ms:
            detail = {"chain": key, "rep_id": rep_id, "source": path.stem}
            if _first_a3m_sequence(path) != sequence:
                findings.append(
                    Finding("V13", Severity.ERROR, entry,
                            f"chain {key}: first MSA row differs from chain sequence ({path.name})", detail)
                )  # fmt: skip
            if read_by_train_yml is not None and path.stem not in read_by_train_yml:
                findings.append(
                    Finding("V13", Severity.ERROR, entry,
                            f"chain {key}: {path.name} is never read by train.yml (not in its msa "
                            "max_seq_counts and aln_order); regenerate train.yml with generate-train-config",
                            detail)
                )  # fmt: skip
    return findings


def check_v13_msa_query_rows(ctx: ValidationContext) -> list[Finding]:
    """V13: every MSA's query row equals its chain sequence, and train.yml reads every MSA source.

    Checks each ``of3_alignments/<rep_id>/<source>.a3m``. When ``<output-dir>/train.yml``
    exists, a source its msa section never reads is an error: training would ignore it.

    Args:
        ctx: Validation context.

    Returns:
        E findings for missing, mismatched or unread MSAs; one INFO when there is no MSA folder.
    """
    if ctx.msa_dir is None:
        return [Finding("V13", Severity.INFO, None, "skipped: no --organized-msa-dir")]
    read_by_train_yml = _sources_read_by_train_yml(ctx)
    return _per_entry(ctx, "V13", lambda c, e: _check_v13(c, e, read_by_train_yml))


def check_v14_rep_ids_by_sequence(ctx: ValidationContext) -> list[Finding]:
    """V14: rep_id is the hash of the chain sequence, and no rep_id covers two sequences.

    Args:
        ctx: Validation context.

    Returns:
        E findings for wrong rep_ids and for rep_id collisions.
    """
    owners: dict[str, dict[str, str]] = {}

    def one(ctx: ValidationContext, entry: str) -> list[Finding]:
        findings = []
        for key, rep_id, sequence in _aligned_chains(ctx, entry):
            expected = alignment_rep_id(sequence)
            owners.setdefault(rep_id, {}).setdefault(sequence, entry)
            if rep_id != expected:
                findings.append(
                    Finding(
                        "V14", Severity.ERROR, entry,
                        f"chain {key}: rep_id {rep_id} != hash of sequence ({expected})",
                        {"chain": key, "rep_id": rep_id, "expected": expected},
                    )
                )  # fmt: skip
        return findings

    findings = _per_entry(ctx, "V14", one)
    for rep_id, seqs in owners.items():
        if len(seqs) > 1:
            findings.append(
                Finding(
                    "V14", Severity.ERROR, None,
                    f"rep_id {rep_id} shared by {len(seqs)} different sequences",
                    {"rep_id": rep_id, "entries": sorted(seqs.values())},
                )
            )  # fmt: skip
    return findings


def _template_ids(ctx: ValidationContext, entry: str) -> list[tuple[str, str]]:
    chains = ctx.dataset_cache.structure_data[entry].chains
    return [(key, t) for key, c in chains.items() for t in (c.template_ids or [])]


def _check_v15(ctx: ValidationContext, entry: str) -> list[Finding]:
    findings = []
    for key, tid in _template_ids(ctx, entry):
        pdb = tid.split("_")[0]
        root = ctx.template_structure_dir
        if not ((root / pdb / f"{tid}.npz").exists() or (root / f"{pdb}.cif").exists()):
            findings.append(
                Finding(
                    "V15", Severity.ERROR, entry,
                    f"chain {key}: template {tid} has no structure in {root}",
                    {"chain": key, "template_id": tid},
                )
            )  # fmt: skip
    return findings


def check_v15_templates_resolve(ctx: ValidationContext) -> list[Finding]:
    """V15: every template ID resolves to a structure file in template_structure_dir.

    Args:
        ctx: Validation context.

    Returns:
        E findings for unresolved IDs; one INFO if there are no template IDs or no structure dir.
    """
    if not any(_template_ids(ctx, e) for e in ctx.entries()):
        return [Finding("V15", Severity.INFO, None, "no template IDs in dataset")]
    if ctx.template_structure_dir is None:
        return [Finding("V15", Severity.INFO, None, "skipped: no --template-structure-dir")]
    return _per_entry(ctx, "V15", _check_v15)


def _date_part(value) -> date:
    """Parse the date part of ``YYYY-MM-DD`` or a datetime-form string."""
    return date.fromisoformat(re.split(r"[T ]", str(value).strip())[0])


def _template_cutoff(ctx: ValidationContext) -> date | None:
    """The --template-cutoff-date this dataset was built with, if any."""
    record = ctx.template_cache_dir.parent / TEMPLATE_CUTOFF_FILENAME
    if not record.exists():
        return None
    return _date_part(json.loads(record.read_text())["template_cutoff_date"])


def _check_v16(ctx: ValidationContext, entry: str, cutoff: date | None) -> list[Finding]:
    data = ctx.dataset_cache.structure_data[entry]
    listed = [(k, c, t) for k, c in data.chains.items() for t in (c.template_ids or [])]
    if not listed:
        return []
    try:
        entry_date = _date_part(data.release_date)
    except (ValueError, TypeError):
        return [Finding("V16", Severity.ERROR, entry, "cannot check template dates: entry release_date missing")]
    findings = []
    caches: dict[str, dict] = {}
    for key, chain, tid in listed:
        rep_id = chain.alignment_representative_id
        path = ctx.template_cache_dir / f"{rep_id}.npz"
        if rep_id not in caches:
            caches[rep_id] = {}
            if rep_id and path.exists():
                with np.load(path, allow_pickle=True) as npz:
                    caches[rep_id] = {k: npz[k].item() for k in npz.files}
        info = caches[rep_id].get(tid)
        if info is None or not info.get("release_date"):
            findings.append(
                Finding(
                    "V16", Severity.ERROR, entry,
                    f"chain {key}: cannot check template date: {tid} has no release_date in {path}",
                    {"chain": key, "template_id": tid},
                )
            )  # fmt: skip
            continue
        released = str(info["release_date"])
        if cutoff is not None:
            if _date_part(released) >= cutoff:
                findings.append(
                    Finding(
                        "V16", Severity.ERROR, entry,
                        f"chain {key}: template {tid} released {released}, not before the template cutoff {cutoff}",
                        {"chain": key, "template_id": tid, "template_date": released, "cutoff": str(cutoff)},
                    )
                )  # fmt: skip
        elif _date_part(released) >= entry_date:
            findings.append(
                Finding(
                    "V16", Severity.ERROR, entry,
                    f"chain {key}: template {tid} released {released}, not before entry {entry_date}",
                    {"chain": key, "template_id": tid, "template_date": released, "entry_date": str(entry_date)},
                )
            )  # fmt: skip
    return findings


def check_v16_template_dates(ctx: ValidationContext) -> list[Finding]:
    """V16: each template's release_date is strictly before its entry's, or a --template-cutoff-date.

    Args:
        ctx: Validation context.

    Returns:
        E findings for listed templates that are not older than the limit or have no date record; one INFO
        naming a cutoff when one was used; one INFO if template IDs exist but no cache dir.
    """
    if ctx.template_cache_dir is None:
        if any(_template_ids(ctx, e) for e in ctx.entries()):
            return [Finding("V16", Severity.INFO, None, "skipped: no --template-cache-dir")]
        return []
    cutoff = _template_cutoff(ctx)
    findings = _per_entry(ctx, "V16", lambda ctx, entry: _check_v16(ctx, entry, cutoff))
    if cutoff is None:
        return findings
    note = f"{TEMPLATE_CUTOFF_NOTE} {cutoff}, not each structure's release date"
    return [Finding("V16", Severity.INFO, None, note), *findings]


def _defaulted_entries(ctx: ValidationContext) -> tuple[set[str], list[Finding]]:
    """Lowercased stems prepare-inputs defaulted, plus one W finding if the record is malformed.

    Args:
        ctx: Validation context.

    Returns:
        (stems, findings); no stems and no findings without a record file.
    """
    record_path = ctx.cif_dir / PREPARE_RECORD_FILE

    def malformed(problem: str) -> tuple[set[str], list[Finding]]:
        return set(), [Finding("V17", Severity.WARNING, None, f"{PREPARE_RECORD_FILE} is {problem}; ignored")]

    if not record_path.is_file():
        return set(), []
    try:
        record = json.loads(record_path.read_text())
    except ValueError:
        return malformed("not valid JSON")
    if not isinstance(record, dict):
        return malformed("not a JSON object")
    stems = record.get("release_date_defaulted", [])
    if not isinstance(stems, list) or not all(isinstance(stem, str) for stem in stems):
        return malformed("malformed: release_date_defaulted is not a list of strings")
    return {stem.lower() for stem in stems}, []


def _check_v17(ctx: ValidationContext, entry: str, defaulted: set[str]) -> list[Finding]:
    released = str(ctx.dataset_cache.structure_data[entry].release_date or "")
    if released and entry.lower() in defaulted:
        message = f"release_date defaulted by prepare-inputs: {released}"
        return [Finding("V17", Severity.WARNING, entry, message, {"release_date": released})]
    if released and released != PLACEHOLDER_RELEASE_DATE:
        return []
    shown = released or "missing"
    return [Finding("V17", Severity.WARNING, entry, f"release_date is {shown}", {"release_date": released})]


def check_v17_release_date(ctx: ValidationContext) -> list[Finding]:
    """V17: each entry has a real release_date, not missing, 1970-01-01 or defaulted by prepare-inputs.

    Args:
        ctx: Validation context.

    Returns:
        W findings for missing, placeholder or prepare-inputs-defaulted dates.
    """
    defaulted, record_findings = _defaulted_entries(ctx)
    return record_findings + _per_entry(ctx, "V17", lambda c, entry: _check_v17(c, entry, defaulted))
