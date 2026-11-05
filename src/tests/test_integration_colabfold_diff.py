"""Integration test: atomworks' ColabFold MSA fetch vs. OF3-native.

Run with: pixi run pytest -m integration tests/test_integration_colabfold_diff.py -v

Live network test: queries https://api.colabfold.com twice per fixture (3u8v, 2crb,
7riv; chain A). atomworks' port (used by generate-msas) changed only retry/timeout
robustness, not request/response handling, so any a3m divergence is a regression.
"""

from __future__ import annotations

import warnings

import pytest
from atomworks.ml.preprocessing.msa.colabfold_server import (
    query_colabfold_msa_server as aw_query_colabfold_msa_server,
)
from openfold3.core.data.tools.colabfold_msa_server import (
    query_colabfold_msa_server as of3_query_colabfold_msa_server,
)

from aw_of3_finetune.msa import chain_canonical_sequences


@pytest.mark.integration
def test_atomworks_and_of3_colabfold_fetch_match(cif_for_integration, tmp_path):
    """atomworks' and OF3's query_colabfold_msa_server return the same a3m for the
    same sequence."""
    seq = chain_canonical_sequences(cif_for_integration)["A"]
    user_agent = "aw-of3-finetune-integration-test"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        of3_a3m = of3_query_colabfold_msa_server(
            x=[seq], prefix=tmp_path / "of3_native", user_agent=user_agent
        )[0]
        aw_a3m = aw_query_colabfold_msa_server(
            x=[seq], prefix=tmp_path / "atomworks", user_agent=user_agent
        )[0]

    assert aw_a3m == of3_a3m, (
        "atomworks' ported query_colabfold_msa_server no longer matches OF3's "
        "original -- if this is an intentional divergence (e.g. atomworks picked up "
        "a new bug fix), update this test's expectation and document why; if not, "
        "something in the port changed request/response handling unexpectedly."
    )
