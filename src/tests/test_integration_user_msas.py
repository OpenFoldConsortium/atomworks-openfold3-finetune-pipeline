"""run-all with user-provided MSAs: OpenFold3's own public training MSAs for four tutorial chains.

The bucket keeps one folder per chain and one file per MSA source
(``<pdb>_<chain>/cfdb_uniref30.a3m``), so the manifest selects them with a glob.
Needs network access; skipped if the downloads fail.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

import pytest
import yaml
from atomworks.ml.preprocessing.msa.organizing import extract_first_sequence_from_msa
from click.testing import CliRunner

from aw_of3_finetune.cli import finetune_pipeline
from aw_of3_finetune.msa import alignment_rep_id
from aw_of3_finetune.msa_sources import read_msa_text
from test_msa_sources import _of3_msa_rows

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
BUCKET = "https://openfold3-data.s3.amazonaws.com/msas/pdb_msas_completed"
CHAINS = ["3u8v_A", "2crb_A", "2q2k_B", "4x1f_A"]


def _download(chain: str, dest_root: Path) -> Path:
    """Fetch <chain>/cfdb_uniref30.a3m into dest_root, keeping the bucket's layout."""
    dest = dest_root / chain / "cfdb_uniref30.a3m"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(f"{BUCKET}/{chain}/cfdb_uniref30.a3m", timeout=60) as response:
            dest.write_bytes(response.read())
    except OSError as error:
        pytest.skip(f"cannot download OpenFold3's MSAs ({chain}): {error}")
    return dest


def test_run_all_with_bucket_msas(ccd_path, tmp_path):
    downloads = [_download(chain, tmp_path / "user_msas") for chain in CHAINS]
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    for pdb_id in sorted({chain.split("_")[0] for chain in CHAINS}):
        shutil.copy(REPO / "examples" / "tutorial" / f"{pdb_id}.cif", cif_dir)
    manifest = tmp_path / "msa_sources.yml"
    manifest.write_text('cfdb_uniref30: "user_msas/*/cfdb_uniref30.a3m"\n')
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path),
        "--msa-sources", str(manifest), "--skip-report", "--skip-dataset-load",
    ], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    msa_dir = output_dir / "organized_msas"
    assert "match no sequence" not in result.output, result.output
    for download in downloads:
        rep_id = alignment_rep_id(extract_first_sequence_from_msa(download))
        imported = msa_dir / "of3_alignments" / rep_id / "cfdb_uniref30.a3m"
        assert imported.read_text() == read_msa_text(download), download
    assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {"cfdb_uniref30": "msa_sources/cfdb_uniref30"}
    config = yaml.safe_load((output_dir / "train.yml").read_text())
    assert config["dataset_configs"]["train"]["weighted-pdb"]["config"]["msa"] == {"moltypes": ["PROTEIN"]}
    assert _of3_msa_rows(output_dir) > 1
