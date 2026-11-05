"""Tests for the MSA, template and release-date checks V13-V17."""

import json

import numpy as np
import pytest

from aw_of3_finetune.prepare_inputs import PREPARE_RECORD_FILE
from aw_of3_finetune.constants import TEMPLATE_CUTOFF_FILENAME
from aw_of3_finetune.validate import CHECKS, run_validation
from aw_of3_finetune.validate.findings import Severity
from validate_helpers import context, mutable_copy

ALIGNMENT_CHECKS = ["V13", "V14", "V15", "V16", "V17"]


def _edit_cache(built, edit):
    """Apply `edit(structure_data)` to the dataset cache JSON on disk."""
    path = built.output_dir / "dataset_cache.json"
    data = json.loads(path.read_text())
    edit(data["structure_data"])
    path.write_text(json.dumps(data, indent=4))


def _hits(report, check, entry=None):
    """Findings of one check on one entry (entry=None selects dataset-wide findings)."""
    return [f for f in report.findings if f.check == check and f.entry == entry]


def _first_protein_chain(structure_data, entry):
    """The first chain of `entry` that has an MSA representative (i.e. a protein chain)."""
    return next(
        chain
        for chain in structure_data[entry]["chains"].values()
        if chain["alignment_representative_id"]
    )


def test_alignment_checks_are_registered():
    """Every alignment check id is present in the CHECKS registry."""
    registered = {check for check in ALIGNMENT_CHECKS if check in CHECKS}
    assert registered == set(ALIGNMENT_CHECKS)


def test_alignment_checks_pass_on_clean_build(built_dataset):
    """A clean build has no errors or warnings; the only INFO is V15 (no template dir given)."""
    report = run_validation(context(built_dataset), ALIGNMENT_CHECKS)

    assert report.errors == [], [f.message for f in report.errors]
    assert report.warnings == []

    info_checks = [f.check for f in report.findings if f.severity == Severity.INFO]
    assert info_checks == ["V15"]


def test_v13_detects_wrong_query_row(built_dataset, tmp_path):
    """V13 errors when an MSA's first row is not the chain's own sequence."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    rep_id = None

    def find(structure_data):
        nonlocal rep_id
        rep_id = _first_protein_chain(structure_data, "2crb")["alignment_representative_id"]

    _edit_cache(built, find)

    # Change the first residue of the MSA's query row.
    a3m = built.msa_dir / "of3_alignments" / rep_id / "colabfold_main.a3m"
    header, sequence, *_ = a3m.read_text().splitlines()
    swapped = ("A" if sequence[0] != "A" else "G") + sequence[1:]
    a3m.write_text(f"{header}\n{swapped}\n")

    report = run_validation(context(built), ["V13"])

    # Every V13 finding on 2crb is an error, and the first one explains the cause.
    v13_findings = _hits(report, "V13", "2crb")
    assert len(v13_findings) > 0
    assert {f.severity for f in v13_findings} == {Severity.ERROR}
    assert "first MSA row differs" in v13_findings[0].message


def test_v13_ignores_lowercase_insertions(built_dataset, tmp_path):
    """Lowercase letters in an a3m are insertions, so they do not change the query row."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    for a3m in (built.msa_dir / "of3_alignments").glob("*/colabfold_main.a3m"):
        header, sequence = a3m.read_text().splitlines()[:2]
        a3m.write_text(f"{header}\n{sequence[:3]}ccc{sequence[3:]}\n")

    report = run_validation(context(built), ["V13"])

    assert report.errors == []


def test_v14_detects_rep_id_mismatch(built_dataset, tmp_path):
    """V14 errors when a representative id is not the hash of the chain's sequence."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def edit(structure_data):
        chain = _first_protein_chain(structure_data, "2crb")
        chain["alignment_representative_id"] = "00000000000"

    _edit_cache(built, edit)

    report = run_validation(context(built), ["V14"])

    # One error on 2crb, naming the wrong id and what it should have been.
    v14_findings = _hits(report, "V14", "2crb")
    assert len(v14_findings) == 1
    assert "00000000000" in v14_findings[0].message
    assert "hash of sequence" in v14_findings[0].message


def test_v14_detects_rep_id_shared_by_two_sequences(built_dataset, tmp_path):
    """V14 flags one representative id used for two different sequences (2crb and 1a3n)."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def edit(structure_data):
        rep_id_2crb = _first_protein_chain(structure_data, "2crb")["alignment_representative_id"]
        _first_protein_chain(structure_data, "1a3n")["alignment_representative_id"] = rep_id_2crb

    _edit_cache(built, edit)

    report = run_validation(context(built), ["V14"])

    # The dataset-wide finding says the id is shared by 2 sequences.
    messages = [f.message for f in _hits(report, "V14")]
    assert any("shared by 2 different sequences" in message for message in messages)


def test_v15_v16(built_dataset, tmp_path):
    """With template dirs given, a template missing from disk (V15) and one released after the entry (V16) are errors."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    # Template structures: 9aaa_A exists on disk, 9bbb_A does not.
    cache_dir, struct_dir = tmp_path / "tcache", tmp_path / "tstructs"
    cache_dir.mkdir()
    (struct_dir / "9aaa").mkdir(parents=True)
    (struct_dir / "9aaa" / "9aaa_A.npz").write_bytes(b"")
    rep_id = None

    def edit(structure_data):
        nonlocal rep_id
        chain = _first_protein_chain(structure_data, "2crb")
        rep_id = chain["alignment_representative_id"]
        chain["template_ids"] = ["9aaa_A", "9bbb_A"]

    _edit_cache(built, edit)

    # Template release dates: 9aaa_A the year before 2crb, 9bbb_A the year after.
    cache = json.loads((built.output_dir / "dataset_cache.json").read_text())
    entry_year = int(cache["structure_data"]["2crb"]["release_date"][:4])
    np.savez(
        cache_dir / f"{rep_id}.npz",
        **{
            "9aaa_A": {"index": 0, "release_date": f"{entry_year - 1}-01-01"},
            "9bbb_A": {"index": 1, "release_date": f"{entry_year + 1}-01-01"},
        },
    )

    ctx = context(built, template_cache_dir=cache_dir, template_structure_dir=struct_dir)
    report = run_validation(ctx, ["V15", "V16"])

    # V15: only 9bbb_A is missing its structure.
    v15_findings = _hits(report, "V15", "2crb")
    assert len(v15_findings) == 1
    assert "9bbb_A has no structure" in v15_findings[0].message

    # V16: only 9bbb_A is newer than the entry.
    v16_findings = _hits(report, "V16", "2crb")
    assert len(v16_findings) == 1
    assert "9bbb_A released" in v16_findings[0].message

    # Both are errors.
    assert v15_findings[0].severity == Severity.ERROR
    assert v16_findings[0].severity == Severity.ERROR


def _v16_with_cutoff(built_dataset, tmp_path, cutoff: str) -> list:
    """V16 findings for a 2crb chain with a template released the year after the entry."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    cache_dir = tmp_path / "data" / "template_cache"
    cache_dir.mkdir(parents=True)
    (tmp_path / "data" / TEMPLATE_CUTOFF_FILENAME).write_text(json.dumps({"template_cutoff_date": cutoff}))
    rep_id = None

    def edit(structure_data):
        nonlocal rep_id
        chain = _first_protein_chain(structure_data, "2crb")
        rep_id = chain["alignment_representative_id"]
        chain["template_ids"] = ["9bbb_A"]

    _edit_cache(built, edit)
    cache = json.loads((built.output_dir / "dataset_cache.json").read_text())
    entry_year = int(cache["structure_data"]["2crb"]["release_date"][:4])
    np.savez(cache_dir / f"{rep_id}.npz", **{"9bbb_A": {"index": 0, "release_date": f"{entry_year + 1}-01-01"}})
    return run_validation(context(built, template_cache_dir=cache_dir), ["V16"]).findings


def test_v16_allows_a_template_before_the_cutoff(built_dataset, tmp_path):
    actual_findings = _v16_with_cutoff(built_dataset, tmp_path, "2100-01-01")

    actual_severities = [finding.severity for finding in actual_findings]
    assert actual_severities == [Severity.INFO]
    assert "2100-01-01" in actual_findings[0].message


def test_v16_rejects_a_template_after_the_cutoff(built_dataset, tmp_path):
    actual_findings = _v16_with_cutoff(built_dataset, tmp_path, "1900-01-01")

    actual_errors = [finding for finding in actual_findings if finding.severity == Severity.ERROR]
    assert len(actual_errors) == 1
    assert "not before the template cutoff 1900-01-01" in actual_errors[0].message


def test_v15_v16_skip_without_dirs(built_dataset, tmp_path):
    """Without template dirs, V15 and V16 only emit INFO notes and no errors."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def edit(structure_data):
        _first_protein_chain(structure_data, "2crb")["template_ids"] = ["9aaa_A"]

    _edit_cache(built, edit)

    report = run_validation(context(built), ["V15", "V16"])

    assert report.errors == []
    assert [f.severity for f in report.findings] == [Severity.INFO, Severity.INFO]


def test_v17_flags_placeholder_date(built_dataset, tmp_path):
    """V17 warns (not errors) about a 1970-01-01 placeholder release date."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    def edit(structure_data):
        structure_data["2crb"]["release_date"] = "1970-01-01"

    _edit_cache(built, edit)

    report = run_validation(context(built), ["V17"])

    v17_findings = _hits(report, "V17", "2crb")
    assert len(v17_findings) == 1
    assert v17_findings[0].severity == Severity.WARNING
    assert "1970-01-01" in v17_findings[0].message
    assert report.errors == []


def _write_prepare_record(built, defaulted, release_date="2026-01-01"):
    """Write prepare_inputs.json into the built dataset's cif dir."""
    record = {"release_date_defaulted": defaulted, "release_date": release_date, "inputs": defaulted}
    (built.cif_dir / PREPARE_RECORD_FILE).write_text(json.dumps(record))


def _set_release_date(built, entry, release_date):
    _edit_cache(built, lambda structure_data: structure_data[entry].update(release_date=release_date))


def test_v17_flags_entry_defaulted_by_prepare_inputs(built_dataset, tmp_path):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _set_release_date(built, "2crb", "2026-01-01")
    _write_prepare_record(built, ["2crb"])

    report = run_validation(context(built), ["V17"])

    actual_findings = _hits(report, "V17", "2crb")
    assert [f.severity for f in actual_findings] == [Severity.WARNING]
    assert actual_findings[0].message == "release_date defaulted by prepare-inputs: 2026-01-01"
    assert report.errors == []


def test_v17_defaulted_stem_matches_entry_case_insensitively(built_dataset, tmp_path):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _set_release_date(built, "2crb", "2026-01-01")
    _write_prepare_record(built, ["2CRB"])

    report = run_validation(context(built), ["V17"])

    actual_messages = [f.message for f in _hits(report, "V17", "2crb")]
    assert actual_messages == ["release_date defaulted by prepare-inputs: 2026-01-01"]


def test_v17_ignores_entries_not_listed_as_defaulted(built_dataset, tmp_path):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _set_release_date(built, "2crb", "2020-05-05")
    _write_prepare_record(built, ["other"])

    report = run_validation(context(built), ["V17"])

    assert _hits(report, "V17", "2crb") == []


def test_v17_without_a_prepare_record_keeps_the_real_date_clean(built_dataset, tmp_path):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _set_release_date(built, "2crb", "2026-01-01")

    report = run_validation(context(built), ["V17"])

    assert _hits(report, "V17", "2crb") == []


def test_v17_defaulted_and_placeholder_date_gives_one_finding(built_dataset, tmp_path):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _set_release_date(built, "2crb", "1970-01-01")
    _write_prepare_record(built, ["2crb"])

    report = run_validation(context(built), ["V17"])

    assert len(_hits(report, "V17", "2crb")) == 1


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "not a JSON object"),
        ('{"release_date_defaulted": "2crb"}', "release_date_defaulted is not a list of strings"),
    ],
)
def test_v17_malformed_prepare_record_warns_once_and_carries_on(built_dataset, tmp_path, text, problem):
    built = mutable_copy(built_dataset, tmp_path / "copy")
    _set_release_date(built, "2crb", "2020-05-05")
    (built.cif_dir / PREPARE_RECORD_FILE).write_text(text)

    report = run_validation(context(built), ["V17"])

    actual_findings = _hits(report, "V17", None)
    assert [f.severity for f in actual_findings] == [Severity.WARNING]
    assert PREPARE_RECORD_FILE in actual_findings[0].message
    assert problem in actual_findings[0].message
    assert _hits(report, "V17", "2crb") == []
    assert report.errors == []


def _with_templates(built, tmp_path, template_ids, cache=None, entry_date=None):
    """Give 2crb's first protein chain template_ids; write a template cache npz from ``cache``."""
    cache_dir = tmp_path / "tcache"
    cache_dir.mkdir()
    rep_id = None

    def edit(structure_data):
        nonlocal rep_id
        chain = _first_protein_chain(structure_data, "2crb")
        rep_id = chain["alignment_representative_id"]
        chain["template_ids"] = template_ids
        if entry_date is not None:
            structure_data["2crb"]["release_date"] = entry_date

    _edit_cache(built, edit)
    if cache is not None:
        np.savez(cache_dir / f"{rep_id}.npz", **cache)
    return context(built, template_cache_dir=cache_dir)


def test_v16_errors_when_cache_npz_missing(built_dataset, tmp_path):
    """V16 errors when the chain lists templates but the template cache has no file for it."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    ctx = _with_templates(built, tmp_path, ["9aaa_A"])

    v16_findings = _hits(run_validation(ctx, ["V16"]), "V16", "2crb")

    assert len(v16_findings) == 1
    assert v16_findings[0].severity == Severity.ERROR
    assert "cannot check template date" in v16_findings[0].message


def test_v16_errors_when_listed_template_has_no_date(built_dataset, tmp_path):
    """V16 errors when the template cache lacks a date for a listed template (9aaa_A)."""
    built = mutable_copy(built_dataset, tmp_path / "copy")

    # The cache knows 9zzz_A but not the listed 9aaa_A.
    cache = {"9zzz_A": {"index": 0, "release_date": "1990-01-01"}}
    ctx = _with_templates(built, tmp_path, ["9aaa_A"], cache)

    v16_findings = _hits(run_validation(ctx, ["V16"]), "V16", "2crb")

    assert len(v16_findings) == 1
    assert "cannot check template date" in v16_findings[0].message
    assert "9aaa_A" in v16_findings[0].message


def test_v16_ignores_missing_entry_date_without_templates(built_dataset, tmp_path):
    """An entry with no templates needs no release date, so a blank date is not an error."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    ctx = _with_templates(built, tmp_path, [], entry_date="")

    v16_findings = _hits(run_validation(ctx, ["V16"]), "V16", "2crb")

    assert v16_findings == []


def test_v16_errors_on_missing_entry_date_with_templates(built_dataset, tmp_path):
    """With templates listed, a blank entry release date can't be compared and is an error."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    cache = {"9aaa_A": {"index": 0, "release_date": "1990-01-01"}}
    ctx = _with_templates(built, tmp_path, ["9aaa_A"], cache, entry_date="")

    v16_findings = _hits(run_validation(ctx, ["V16"]), "V16", "2crb")

    assert len(v16_findings) == 1
    assert "entry release_date missing" in v16_findings[0].message


def test_v16_accepts_datetime_strings(built_dataset, tmp_path):
    """V16 parses full datetime strings; only the template released at/after the entry is flagged."""
    built = mutable_copy(built_dataset, tmp_path / "copy")
    cache = {
        "9aaa_A": {"index": 0, "release_date": "2019-12-31 00:00:00"},
        "9bbb_A": {"index": 1, "release_date": "2020-01-01T00:00:00"},
    }
    ctx = _with_templates(built, tmp_path, ["9aaa_A", "9bbb_A"], cache, "2020-01-01T08:00:00")

    v16_findings = _hits(run_validation(ctx, ["V16"]), "V16", "2crb")

    # 9aaa_A predates the entry (OK); 9bbb_A is released the same day (flagged).
    assert len(v16_findings) == 1
    assert "9bbb_A released" in v16_findings[0].message
