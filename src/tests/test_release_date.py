"""Dataset-cache release_date follows OF3's rule: the earliest revision date.

OF3's preprocessing (openfold3.core.data.primitives.structure.metadata.get_release_date)
takes the minimum of ``pdbx_audit_revision_history.revision_date``, with no fallback.

Test structure:
- test_release_date_is_earliest_revision_date: real fixture CIFs, release != deposition.
- test_release_date_ignores_revision_row_order: minimum, not first row.
- test_release_date_single_revision: single-row (non-loop) category.
- test_release_date_missing_revision_history_raises: no deposition-date fallback.
"""

from datetime import date
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import pytest

from aw_of3_finetune.pipeline import _read_release_date

FIXTURES_MMCIFS = Path(__file__).parent / "fixtures" / "mmcifs"


def _block_from_text(cif_text: str) -> pdbx.CIFBlock:
    """Parse a CIF string into its single data block.

    Args:
        cif_text: mmCIF text with exactly one data block.

    Returns:
        The parsed data block.
    """
    return pdbx.CIFFile.deserialize(cif_text).block


@pytest.mark.parametrize(
    ("cif_name", "expected_release_date"),
    [
        # 7riv: deposited 2021-07-20, released 2021-08-25.
        ("7riv.cif", date(2021, 8, 25)),
        # 6hj2: deposited 2018-08-31, released 2020-03-25.
        ("6hj2.cif", date(2020, 3, 25)),
        # 3u8v: deposited 2011-10-17, released 2011-11-09 (single-row category).
        ("3u8v.cif", date(2011, 11, 9)),
        # 1fdl: deposited 1990-08-27, released 1991-10-15.
        ("1fdl.cif", date(1991, 10, 15)),
    ],
)
def test_release_date_is_earliest_revision_date(cif_name, expected_release_date):
    """release_date is the first release, not the deposition date."""
    block = pdbx.CIFFile.read(FIXTURES_MMCIFS / cif_name).block

    actual_release_date = _read_release_date(block)

    assert actual_release_date == expected_release_date


def test_release_date_ignores_revision_row_order():
    """The earliest date wins even when it is not the first row."""
    cif_text = """data_TEST
#
_pdbx_database_status.recvd_initial_deposition_date 2000-01-01
#
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2005-06-01
2 2003-02-14
3 2010-12-31
#
"""
    block = _block_from_text(cif_text)
    expected_release_date = date(2003, 2, 14)

    actual_release_date = _read_release_date(block)

    assert actual_release_date == expected_release_date


def test_release_date_single_revision():
    """A single-row (non-loop) revision history is read too."""
    cif_text = """data_TEST
#
_pdbx_audit_revision_history.ordinal 1
_pdbx_audit_revision_history.revision_date 2012-07-04
#
"""
    block = _block_from_text(cif_text)
    expected_release_date = date(2012, 7, 4)

    actual_release_date = _read_release_date(block)

    assert actual_release_date == expected_release_date


def test_release_date_missing_revision_history_raises():
    """Without a revision history there is no release date; deposition is not used."""
    cif_text = """data_TEST
#
_pdbx_database_status.recvd_initial_deposition_date 2000-01-01
#
"""
    block = _block_from_text(cif_text)

    with pytest.raises(ValueError, match="pdbx_audit_revision_history"):
        _read_release_date(block)
