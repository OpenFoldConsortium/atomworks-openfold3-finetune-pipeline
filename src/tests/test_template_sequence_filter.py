"""OF3's template sequence filters (AF3 SI 2.4) applied to ColabFold template hits.

OF3's legacy hmmsearch path drops template hits that are too short, align to too little
of the query, or are a near-full-length exact copy of the query (check_sequence). Its
.m8 path cannot apply these, so preprocess-templates applies them before OF3 runs.

Test structure:
- fails_template_sequence_filter on synthetic query/template pairs, one rule per test.
- filter_template_alignment on a synthetic .m8 table with matching template CIFs.
"""

import inspect
import random
from pathlib import Path

from openfold3.core.data.primitives.sequence.template import check_sequence

from aw_of3_finetune.templates import (
    TemplateSequenceFilterSettings,
    fails_template_sequence_filter,
    filter_template_alignment,
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# A 200-residue pseudo-random query; long enough that 5% of it is 10 residues.
QUERY = "".join(random.Random(0).choice(AMINO_ACIDS) for _ in range(200))
DEFAULT_SETTINGS = TemplateSequenceFilterSettings()


def _mutate(seq: str, position: int) -> str:
    """Return seq with one residue swapped for a different amino acid."""
    replacement = "W" if seq[position] != "W" else "Y"
    return seq[:position] + replacement + seq[position + 1 :]


def test_default_thresholds_match_of3():
    """Defaults are OF3's check_sequence defaults: 0.95, 0.1 and 10 (AF3 SI 2.4)."""
    of3_defaults = inspect.signature(check_sequence).parameters
    expected = {name: of3_defaults[name].default for name in ("max_subseq", "min_align", "min_len")}
    actual = DEFAULT_SETTINGS.model_dump(exclude={"enabled"})
    assert actual == expected


def test_identical_template_is_dropped():
    """A template identical to the query is a full-length exact subsequence."""
    actual_fails = fails_template_sequence_filter(QUERY, QUERY, DEFAULT_SETTINGS)
    assert actual_fails is True


def test_identical_template_with_tag_is_dropped():
    """An expression tag outside the aligned region does not rescue an identical construct."""
    tagged_template = "MGSSHHHHHHSSG" + QUERY + "LEHHHHHH"
    actual_fails = fails_template_sequence_filter(QUERY, tagged_template, DEFAULT_SETTINGS)
    assert actual_fails is True


def test_aligned_but_different_flank_keeps_the_template():
    """A differing residue inside the aligned region makes the template not an exact copy.

    Mirrors 4j5w/4j5x vs 4s0t_A on pxr-pdb, which OF3 keeps: template ``SNA`` vs query
    ``HHG`` before an identical PXR LBD; N aligns to H (+1 in BLOSUM62).
    """
    query = "MKKHHHHHHG" + QUERY  # 210 residues
    template = "SNA" + QUERY  # identical part is 200 / 210 = 0.952 of the query
    actual_fails = fails_template_sequence_filter(query, template, DEFAULT_SETTINGS)
    assert actual_fails is False


def test_long_exact_fragment_is_dropped():
    """An exact fragment covering more than 95% of the query is dropped."""
    fragment = QUERY[5:]  # 195 / 200 = 0.975
    actual_fails = fails_template_sequence_filter(QUERY, fragment, DEFAULT_SETTINGS)
    assert actual_fails is True


def test_shorter_exact_fragment_is_kept():
    """An exact fragment covering at most 95% of the query is kept."""
    fragment = QUERY[10:]  # 190 / 200 = 0.95, not > 0.95
    actual_fails = fails_template_sequence_filter(QUERY, fragment, DEFAULT_SETTINGS)
    assert actual_fails is False


def test_point_mutant_is_kept():
    """A full-length template with one substitution is not an exact subsequence."""
    mutant = _mutate(QUERY, 100)
    actual_fails = fails_template_sequence_filter(QUERY, mutant, DEFAULT_SETTINGS)
    assert actual_fails is False


def test_short_template_is_dropped():
    """A template shorter than min_len (10) residues is dropped."""
    short_template = QUERY[50:59]  # 9 residues
    actual_fails = fails_template_sequence_filter(QUERY, short_template, DEFAULT_SETTINGS)
    assert actual_fails is True


def test_low_coverage_template_is_dropped():
    """A template aligned to at most 10% of the query is dropped."""
    low_coverage = QUERY[50:70]  # 20 aligned residues / 200 = 0.1, not > 0.1
    actual_fails = fails_template_sequence_filter(QUERY, low_coverage, DEFAULT_SETTINGS)
    assert actual_fails is True


def test_thresholds_are_configurable():
    """Raising max_subseq to 1.0 keeps a long exact fragment."""
    fragment = QUERY[5:]
    settings = TemplateSequenceFilterSettings(max_subseq=1.0)
    actual_fails = fails_template_sequence_filter(QUERY, fragment, settings)
    assert actual_fails is False


def _write_template_cif(path: Path, sequence: str) -> None:
    """Write a minimal mmCIF with one protein entity (label chain A, author chain X)."""
    path.write_text(
        "data_test\n"
        "loop_\n_entity_poly.entity_id\n_entity_poly.type\n"
        "_entity_poly.pdbx_seq_one_letter_code_can\n"
        f"1 'polypeptide(L)' {sequence}\n"
        "loop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\n"
    )


def test_filter_template_alignment_drops_rows(tmp_path):
    """Rows whose template chain fails the filter are dropped; others keep their order."""
    _write_template_cif(tmp_path / "1aaa.cif", QUERY)
    _write_template_cif(tmp_path / "2bbb.cif", _mutate(QUERY, 100))
    m8_text = (
        "101\t1aaa_A\t1.000\t200\t0\t0\t1\t200\t1\t200\t1E-90\t300\t200M\n"
        "101\t2bbb_A\t0.995\t200\t1\t0\t1\t200\t1\t200\t1E-89\t298\t200M\n"
    )

    actual_text, actual_dropped = filter_template_alignment(
        m8_text, QUERY, tmp_path, DEFAULT_SETTINGS
    )

    expected_dropped = ["1aaa_A"]
    expected_text = "101\t2bbb_A\t0.995\t200\t1\t0\t1\t200\t1\t200\t1E-89\t298\t200M\n"
    assert actual_dropped == expected_dropped
    assert actual_text == expected_text


def test_filter_template_alignment_disabled_keeps_everything(tmp_path):
    """With the filter switched off, the table is returned unchanged."""
    _write_template_cif(tmp_path / "1aaa.cif", QUERY)
    m8_text = "101\t1aaa_A\t1.000\t200\t0\t0\t1\t200\t1\t200\t1E-90\t300\t200M\n"
    settings = TemplateSequenceFilterSettings(enabled=False)

    actual_text, actual_dropped = filter_template_alignment(m8_text, QUERY, tmp_path, settings)

    assert actual_dropped == []
    assert actual_text == m8_text
