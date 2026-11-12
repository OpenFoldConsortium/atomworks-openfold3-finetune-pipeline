"""ColabFold template hits -> the directory layout OF3 reads.

generate-msas (use_templates=True) leaves raw ColabFold template hits at
  <msa-dir>/templates/<shard>/<rep_id>.m8
and fetch-templates turns them into the layout OF3's template preprocessing reads:
  <msa-dir>/of3_template_alignments/<rep_id>/colabfold_template.m8
  <msa-dir>/template_structures/<pdb_id>.cif      (one flat, uncompressed dir)

Test structure:
- TestGenerateMsasTemplatesFlag: templates on by default; --no-templates turns them off.
- TestReorganizeTemplateAlignments: .m8 -> per-rep_id layout with label chain IDs, kept out
  of of3_alignments/.
- TestMaterializeTemplateStructures: mirror .cif.gz / downloaded .cif -> one flat .cif dir.
- TestFetchTemplates: end to end over a fake local PDB mirror (offline), library and CLI.
- test_fetch_templates_direct_download (integration): no mirror, real RCSB download.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from openfold3.core.data.io.sequence.msa import parse_msas_direct, standardize_filepaths
from requests.exceptions import SSLError

import aw_of3_finetune.cli as cli_mod
from aw_of3_finetune.cli import fetch_templates as fetch_templates_cli
from aw_of3_finetune.cli import finetune_prep, generate_msas
from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences
from aw_of3_finetune.pipeline import reorganize_msas_for_of3
from aw_of3_finetune.templates import (
    TemplateDownloadError,
    fetch_templates,
    materialize_template_structures,
    reorganize_template_alignments_for_of3,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# Synthetic minimal mmCIF body; only its text identity matters for these tests.
_FAKE_CIF = "data_{id}\n_entry.id {ID}\n"


def _m8_rows(hits: list[str]) -> str:
    """Build raw ColabFold pdb70.m8 rows (real column layout), best hit first.

    Args:
        hits: ``<pdb_id>_<chain>`` target IDs, in the order to write them.

    Returns:
        The tab-separated table text.
    """
    return "".join(
        f"101\t{hit}\t0.950\t76\t3\t0\t1\t76\t1\t76\t{1e-30 * (i + 1):.3E}\t150\t76M\n"
        for i, hit in enumerate(hits)
    )


def _write_m8(seq: str, msa_dir: Path, hits: list[str]) -> Path:
    """Write a .m8 where make_msas_colabfold_server_batch(use_templates=True) puts it.

    Args:
        seq: Query sequence; its hash names and shards the file.
        msa_dir: generate-msas output directory.
        hits: Target IDs for the rows (see _m8_rows).

    Returns:
        Path of the written ``templates/<shard>/<hash>.m8``.
    """
    rep = alignment_rep_id(seq)
    out = msa_dir / "templates" / rep[:2] / f"{rep}.m8"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_m8_rows(hits))
    return out


def _write_mirror_entry(mirror: Path, pdb_id: str) -> Path:
    """Write a fake local PDB mirror entry in atomworks' get_pdb_path layout.

    Args:
        mirror: Mirror root.
        pdb_id: Lowercase PDB ID.

    Returns:
        Path of the written ``<mirror>/<id[1:3]>/<id>.cif.gz``.
    """
    out = mirror / pdb_id[1:3] / f"{pdb_id}.cif.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        f.write(_FAKE_CIF.format(id=pdb_id, ID=pdb_id.upper()))
    return out


def _prepped_output_dir(cif_path: Path, pdb_id: str, tmp_path: Path) -> Path:
    """Run finetune-prep on one fixture CIF.

    Args:
        cif_path: Fixture mmCIF.
        pdb_id: ID to file it under.
        tmp_path: Scratch directory for the input and output dirs.

    Returns:
        The output directory (manifest.json + cleaned_structures/).
    """
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / f"{pdb_id}.cif").symlink_to(cif_path.resolve())
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return output_dir


def _seed_structure(msa_dir: Path, cif_path: Path) -> Path:
    """Place a template structure where fetch_templates would have put it.

    Args:
        msa_dir: generate-msas output directory.
        cif_path: Structure to copy; its lowercased stem is the PDB ID.

    Returns:
        Path of the copied ``template_structures/<pdb_id>.cif``.
    """
    dest = msa_dir / "template_structures" / f"{cif_path.stem.lower()}.cif"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cif_path, dest)
    return dest


@pytest.fixture
def ubq(cif_ubiquitin_1, tmp_path):
    """Prep a 1UBQ dataset with finetune-prep.

    Returns:
        ``(output_dir, msa_dir, sequence, rep_id)``; msa_dir is not created.
    """
    output_dir = _prepped_output_dir(cif_ubiquitin_1, "1ubq", tmp_path)
    seq = chain_canonical_sequences(cif_ubiquitin_1)["A"]
    return output_dir, tmp_path / "msas", seq, alignment_rep_id(seq)


# ── TestGenerateMsasTemplatesFlag ─────────────────────────────────────────────


class TestGenerateMsasTemplatesFlag:
    @pytest.fixture
    def server_calls(self, monkeypatch):
        """Capture make_msas_colabfold_server_batch's kwargs instead of querying ColabFold.

        Returns:
            List that receives one kwargs dict per call.
        """
        calls: list[dict] = []
        monkeypatch.setattr(
            cli_mod, "make_msas_colabfold_server_batch", lambda *a, **kw: calls.append(kw)
        )
        return calls

    def test_templates_requested_by_default(self, ubq, server_calls):
        """Without a flag, generate-msas asks ColabFold for templates."""
        output_dir, msa_dir, _, _ = ubq
        result = CliRunner().invoke(
            generate_msas,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        # The server call was made with use_templates=True.
        assert server_calls[0]["use_templates"] is True

    def test_no_templates_flag(self, ubq, server_calls):
        """--no-templates turns the ColabFold template request off."""
        output_dir, msa_dir, _, _ = ubq
        result = CliRunner().invoke(
            generate_msas,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir), "--no-templates"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        # The server call was made with use_templates=False.
        assert server_calls[0]["use_templates"] is False


# ── TestGenerateMsasExistingPaired ────────────────────────────────────────────


class TestGenerateMsasExistingPaired:
    """atomworks refuses to write into an existing paired-MSA directory, so
    generate-msas must stop before querying ColabFold rather than after."""

    @pytest.fixture
    def hemoglobin(self, tmp_path):
        """finetune-prep output for 1a3n (alpha + beta chains: one paired complex)."""
        output_dir = _prepped_output_dir(
            Path(__file__).parent / "fixtures" / "mmcifs" / "1a3n.cif", "1a3n", tmp_path
        )
        return output_dir, tmp_path / "msas"

    @staticmethod
    def _seed_paired_dir(output_dir: Path, msa_dir: Path) -> Path:
        """Create the paired-MSA directory atomworks would write for 1a3n's complex."""
        from atomworks.io.utils.io_utils import apply_sharding_pattern
        from atomworks.ml.utils.misc import get_complex_id

        from aw_of3_finetune.msa import collect_complex_queries

        (seqs,) = collect_complex_queries(output_dir).values()
        complex_dir = msa_dir / "paired" / apply_sharding_pattern(get_complex_id(seqs), "/0:2/")
        complex_dir.mkdir(parents=True)
        return complex_dir

    def test_existing_paired_msas_fail_before_querying(self, hemoglobin, monkeypatch):
        """An existing paired-MSA dir makes generate-msas exit 2 without querying ColabFold."""
        output_dir, msa_dir = hemoglobin
        complex_dir = self._seed_paired_dir(output_dir, msa_dir)
        calls = []
        monkeypatch.setattr(cli_mod, "make_msas_colabfold_server_batch", lambda *a, **kw: calls.append(kw))

        result = CliRunner().invoke(
            generate_msas, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)]
        )

        # Usage error, and ColabFold was never queried.
        assert result.exit_code == 2, result.output
        assert calls == [], "ColabFold must not be queried"

        # The message names the count, the way out (--skip-existing) and the offending directory.
        assert "1 paired MSA(s) already exist" in result.output
        assert "--skip-existing" in result.output
        assert complex_dir.name in result.output

    def test_skip_existing_partial_paired_dir_fails(self, hemoglobin, monkeypatch):
        """An empty/partial complex dir (interrupted run) can't be resumed: say to delete it."""
        output_dir, msa_dir = hemoglobin
        complex_dir = self._seed_paired_dir(output_dir, msa_dir)
        calls = []
        monkeypatch.setattr(cli_mod, "make_msas_colabfold_server_batch", lambda *a, **kw: calls.append(kw))

        result = CliRunner().invoke(
            generate_msas,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir), "--skip-existing"],
        )

        # Usage error, and ColabFold was never queried.
        assert result.exit_code == 2, result.output
        assert calls == []

        # The message says the directory is incomplete, to delete it, and names it.
        assert "incomplete" in result.output and "delete" in result.output
        assert complex_dir.name in result.output

    def test_skip_existing_complete_paired_dir_is_skipped(self, hemoglobin, monkeypatch):
        """With --skip-existing and a complete paired dir, the complex is dropped from the query."""
        from atomworks.ml.utils.misc import hash_sequence

        from aw_of3_finetune.msa import collect_complex_queries

        output_dir, msa_dir = hemoglobin
        complex_dir = self._seed_paired_dir(output_dir, msa_dir)
        (seqs,) = collect_complex_queries(output_dir).values()
        for seq in seqs:
            with gzip.open(complex_dir / f"{hash_sequence(seq)}.a3m.gz", "wt") as f:
                f.write(f">query\n{seq}\n")
        calls = []
        monkeypatch.setattr(cli_mod, "make_msas_colabfold_server_batch", lambda *a, **kw: calls.append(kw))

        result = CliRunner().invoke(
            generate_msas,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir), "--skip-existing"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        # ColabFold was still called (for unpaired MSAs), but with no complexes to pair.
        assert calls and calls[0]["complexes"] == []


# ── TestReorganizeTemplateAlignments ──────────────────────────────────────────


class TestReorganizeTemplateAlignments:
    """2q2k is the hit structure: DNA comes first, so author chain A is label chain B."""

    def test_writes_m8_with_label_chain_ids(self, ubq, cif_dna_complex):
        """The .m8 is rewritten with author chain IDs translated to label chain IDs."""
        output_dir, msa_dir, seq, rep_id = ubq
        src = _write_m8(seq, msa_dir, ["2q2k_A", "2q2k_B"])
        _seed_structure(msa_dir, cif_dna_complex)
        dest_root = msa_dir / "of3_template_alignments"

        written, missing = reorganize_template_alignments_for_of3(output_dir, msa_dir, dest_root)

        # One alignment written, none missing.
        assert (written, missing) == (1, 0)

        # Author chains A, B of 2q2k become label chains B, C.
        dest = dest_root / rep_id / "colabfold_template.m8"
        expected_m8 = src.read_text().replace("2q2k_B", "2q2k_C").replace("2q2k_A", "2q2k_B")
        assert dest.read_text() == expected_m8

    def test_unmappable_hits_dropped(self, ubq, cif_dna_complex):
        """Hits whose structure wasn't fetched, or whose author chain isn't in it, go."""
        output_dir, msa_dir, seq, rep_id = ubq
        _write_m8(seq, msa_dir, ["9xyz_A", "2q2k_Z", "2q2k_A"])
        _seed_structure(msa_dir, cif_dna_complex)
        dest_root = msa_dir / "of3_template_alignments"

        reorganize_template_alignments_for_of3(output_dir, msa_dir, dest_root)

        rows = (dest_root / rep_id / "colabfold_template.m8").read_text().splitlines()

        # Only the mappable hit (2q2k author chain A, now label chain B) remains.
        actual_targets = [row.split("\t")[1] for row in rows]
        assert actual_targets == ["2q2k_B"]

    def test_unfetched_structures_logged_once_not_warned(self, ubq, cif_dna_complex, caplog):
        """Hits beyond --max-hits are expected: one summary line, no per-hit warnings."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["2q2k_A", "9xyz_A", "8xyz_A", "7xyz_B"])
        _seed_structure(msa_dir, cif_dna_complex)

        with caplog.at_level(logging.INFO, logger="aw_of3_finetune.templates"):
            reorganize_template_alignments_for_of3(
                output_dir, msa_dir, msa_dir / "of3_template_alignments"
            )

        # Nothing was logged at warning level or above.
        actual_warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
        assert actual_warnings == []

        # One summary line counts the three unfetched hits.
        summaries = [record.getMessage() for record in caplog.records if "not fetched" in record.getMessage()]
        assert len(summaries) == 1 and "3 hit(s)" in summaries[0]

    def test_unknown_author_chain_warned(self, ubq, cif_dna_complex, caplog):
        """A chain missing from a fetched structure is unexpected, so it is a warning."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["2q2k_A", "2q2k_Z"])
        _seed_structure(msa_dir, cif_dna_complex)

        with caplog.at_level(logging.INFO, logger="aw_of3_finetune.templates"):
            reorganize_template_alignments_for_of3(
                output_dir, msa_dir, msa_dir / "of3_template_alignments"
            )

        # Exactly one warning, naming the unknown chain.
        actual_warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
        assert len(actual_warnings) == 1 and "2q2k_Z" in actual_warnings[0]

    def test_no_mappable_hits_counted_missing(self, ubq):
        """If no hit can be mapped, no .m8 is written; the sequence counts as missing."""
        output_dir, msa_dir, seq, rep_id = ubq
        _write_m8(seq, msa_dir, ["9xyz_A"])
        dest_root = msa_dir / "of3_template_alignments"

        # Nothing written; the one sequence counts as missing.
        assert reorganize_template_alignments_for_of3(output_dir, msa_dir, dest_root) == (0, 1)

        # No directory was created for the sequence.
        assert not (dest_root / rep_id).exists()

    def test_sequence_without_hits_counted_missing(self, ubq):
        """No PDB70 hits -> ColabFold writes no .m8. That's counted, not an error."""
        output_dir, msa_dir, _, _ = ubq
        dest_root = msa_dir / "of3_template_alignments"

        # Nothing written; the one sequence counts as missing.
        assert reorganize_template_alignments_for_of3(output_dir, msa_dir, dest_root) == (0, 1)

    def test_rerun_is_idempotent(self, ubq, cif_dna_complex):
        """A second run finds the alignment already written and does nothing."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["2q2k_A"])
        _seed_structure(msa_dir, cif_dna_complex)
        dest_root = msa_dir / "of3_template_alignments"

        # First run writes the alignment; the second finds it and writes nothing.
        assert reorganize_template_alignments_for_of3(output_dir, msa_dir, dest_root) == (1, 0)
        assert reorganize_template_alignments_for_of3(output_dir, msa_dir, dest_root) == (0, 0)

    def test_msa_dir_stays_parseable_by_of3(self, ubq, cif_dna_complex):
        """The .m8 never lands in of3_alignments/<rep_id>/, which OF3 parses file-by-file."""
        output_dir, msa_dir, seq, rep_id = ubq
        rep = alignment_rep_id(seq)
        unpaired = msa_dir / "unpaired" / rep[:2] / f"{rep}.a3m.gz"
        unpaired.parent.mkdir(parents=True)
        with gzip.open(unpaired, "wt") as f:
            f.write(f">query\n{seq}\n")
        _write_m8(seq, msa_dir, ["2q2k_A"])
        _seed_structure(msa_dir, cif_dna_complex)

        reorganize_msas_for_of3(output_dir, msa_dir, msa_dir / "of3_alignments")
        reorganize_template_alignments_for_of3(
            output_dir, msa_dir, msa_dir / "of3_template_alignments"
        )

        files = standardize_filepaths(msa_dir / "of3_alignments" / rep_id)
        # The only file OF3 sees in the rep_id directory is the main MSA...
        assert [f.name for f in files] == ["colabfold_main.a3m"]

        # ...and OF3 parses it.
        assert "colabfold_main" in parse_msas_direct(files)


# ── TestMaterializeTemplateStructures ─────────────────────────────────────────


class TestMaterializeTemplateStructures:
    def test_mirror_cif_gz_is_decompressed_into_flat_dir(self, tmp_path):
        """A mirror .cif.gz is decompressed to <dest>/<id>.cif."""
        src = _write_mirror_entry(tmp_path / "mirror", "1abc")
        dest_dir = tmp_path / "template_structures"

        result = materialize_template_structures({"1abc": src}, dest_dir)

        # The returned path is the flat destination file...
        assert result == {"1abc": dest_dir / "1abc.cif"}

        # ...holding the decompressed text.
        assert (dest_dir / "1abc.cif").read_text() == _FAKE_CIF.format(id="1abc", ID="1ABC")

    def test_download_already_in_dest_is_left_alone(self, tmp_path):
        """Direct downloads land in dest_dir already -- nothing to do."""
        dest_dir = tmp_path / "template_structures"
        dest_dir.mkdir()
        downloaded = dest_dir / "2xyz.cif"
        downloaded.write_text("downloaded")

        result = materialize_template_structures({"2xyz": downloaded}, dest_dir)

        # The path is returned unchanged and the file keeps its content.
        assert result == {"2xyz": downloaded}
        assert downloaded.read_text() == "downloaded"

        # No extra files appeared in the directory.
        assert sorted(p.name for p in dest_dir.iterdir()) == ["2xyz.cif"]

    def test_plain_cif_elsewhere_is_copied(self, tmp_path):
        """An uncompressed .cif outside the destination is copied in, and the source stays."""
        src = tmp_path / "elsewhere" / "3def.cif"
        src.parent.mkdir()
        src.write_text("plain")
        dest_dir = tmp_path / "template_structures"

        result = materialize_template_structures({"3def": src}, dest_dir)

        # The copy has the same text, and the source file is untouched.
        assert result["3def"].read_text() == "plain"
        assert src.exists()

    def test_existing_dest_file_not_rewritten(self, tmp_path):
        """A destination file that already exists is not overwritten on a second call."""
        src = _write_mirror_entry(tmp_path / "mirror", "1abc")
        dest_dir = tmp_path / "template_structures"
        materialize_template_structures({"1abc": src}, dest_dir)
        (dest_dir / "1abc.cif").write_text("sentinel")

        materialize_template_structures({"1abc": src}, dest_dir)

        # The sentinel text is still there.
        assert (dest_dir / "1abc.cif").read_text() == "sentinel"

    def test_mixed_layouts_end_up_flat(self, tmp_path):
        """fetch_template_structures can return mirror and download paths together."""
        dest_dir = tmp_path / "template_structures"
        dest_dir.mkdir()
        (dest_dir / "2xyz.cif").write_text("downloaded")
        paths = {
            "1abc": _write_mirror_entry(tmp_path / "mirror", "1abc"),
            "2xyz": dest_dir / "2xyz.cif",
        }

        materialize_template_structures(paths, dest_dir)

        # Both structures are in the one flat directory.
        assert sorted(p.name for p in dest_dir.iterdir()) == ["1abc.cif", "2xyz.cif"]


def _flaky_rcsb(monkeypatch, failing_id: str, failures: int) -> list[str]:
    """Replace RCSB downloads with local writes where one entry drops the connection.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        failing_id: Entry whose download raises SSLError.
        failures: How many times it raises before succeeding.

    Returns:
        The list that collects each entry actually downloaded, in order.
    """
    remaining = {"failures": failures}
    downloads: list[str] = []

    def fake_fetch(pdb_ids: str, format: str, target_path: str) -> str:
        path = Path(target_path) / f"{pdb_ids}.cif"
        if path.exists():
            return str(path)
        if pdb_ids == failing_id and remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise SSLError("EOF occurred in violation of protocol")
        path.write_text(_FAKE_CIF.format(id=pdb_ids, ID=pdb_ids.upper()))
        downloads.append(pdb_ids)
        return str(path)

    monkeypatch.setattr("atomworks.ml.preprocessing.msa.template_structures.fetch", fake_fetch)
    monkeypatch.setattr("aw_of3_finetune.templates.time.sleep", lambda seconds: None)
    return downloads


# ── TestFetchTemplates ────────────────────────────────────────────────────────


class TestFetchTemplates:
    def test_resolves_hits_from_mirror_into_flat_dir(self, ubq, tmp_path):
        """Hits are looked up in the local mirror and land in template_structures/."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["1ubi_A", "1ubq_A"])
        mirror = tmp_path / "mirror"
        for pdb_id in ("1ubi", "1ubq"):
            _write_mirror_entry(mirror, pdb_id)

        summary = fetch_templates(output_dir, msa_dir, pdb_mirror_path=mirror)

        structure_dir = msa_dir / "template_structures"
        # Both hit structures were fetched into the flat directory.
        assert sorted(p.name for p in structure_dir.iterdir()) == ["1ubi.cif", "1ubq.cif"]

        # The summary counts two structures from one alignment.
        assert summary.n_structures == 2
        assert summary.n_alignments == 1

    def test_max_hits_limits_entries_per_alignment(self, ubq, tmp_path):
        """max_hits=2 fetches only the two best hits of an alignment."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["1ubi_A", "1ubq_A", "1aar_A"])
        mirror = tmp_path / "mirror"
        for pdb_id in ("1ubi", "1ubq", "1aar"):
            _write_mirror_entry(mirror, pdb_id)

        fetch_templates(output_dir, msa_dir, max_hits=2, pdb_mirror_path=mirror)

        names = sorted(p.name for p in (msa_dir / "template_structures").iterdir())

        # The first two hits are fetched; 1aar, the third, is not.
        assert names == ["1ubi.cif", "1ubq.cif"]

    def test_no_alignments_is_not_an_error(self, ubq):
        """With no .m8 files, fetch_templates returns an all-zero summary."""
        output_dir, msa_dir, _, _ = ubq
        summary = fetch_templates(output_dir, msa_dir)

        # No alignments and no structures.
        assert (summary.n_alignments, summary.n_structures) == (0, 0)

    def test_dropped_connection_is_retried_and_keeps_finished_downloads(self, ubq, monkeypatch):
        """A connection dropped mid-run costs a retry of the unfinished entries only."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["1ubi_A", "2k25_A"])
        downloads = _flaky_rcsb(monkeypatch, failing_id="2k25", failures=2)

        summary = fetch_templates(output_dir, msa_dir)

        expected_downloads = ["1ubi", "2k25"]
        assert downloads == expected_downloads
        assert summary.n_structures == 2

    def test_unreachable_rcsb_says_how_to_resume(self, ubq, monkeypatch):
        """When retries run out the error names the source and the way to resume."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["1ubi_A", "2k25_A"])
        downloads = _flaky_rcsb(monkeypatch, failing_id="2k25", failures=1000)

        with pytest.raises(TemplateDownloadError) as caught:
            fetch_templates(output_dir, msa_dir)

        actual_message = str(caught.value)
        assert "RCSB" in actual_message
        assert "--skip-existing" in actual_message
        assert (msa_dir / "template_structures" / "1ubi.cif").exists()
        assert downloads == ["1ubi"]

    def test_cli_reports_unreachable_rcsb_without_a_traceback(self, ubq, monkeypatch):
        """The CLI turns the download failure into a one-message exit."""
        output_dir, msa_dir, seq, _ = ubq
        _write_m8(seq, msa_dir, ["2k25_A"])
        _flaky_rcsb(monkeypatch, failing_id="2k25", failures=1000)

        result = CliRunner().invoke(
            fetch_templates_cli, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)]
        )

        assert result.exit_code == 1
        assert "--skip-existing" in result.output
        assert isinstance(result.exception, SystemExit)

    def test_cli_fetches_and_reorganizes(self, ubq, cif_dna_complex, tmp_path):
        """The fetch-templates CLI fetches the structure and writes the label-chain .m8."""
        output_dir, msa_dir, seq, rep_id = ubq
        _write_m8(seq, msa_dir, ["2q2k_A"])
        mirror = tmp_path / "mirror"
        (mirror / "q2").mkdir(parents=True)
        (mirror / "q2" / "2q2k.cif.gz").write_bytes(gzip.compress(cif_dna_complex.read_bytes()))

        result = CliRunner().invoke(
            fetch_templates_cli,
            [
                "--output-dir", str(output_dir),
                "--organized-msa-dir", str(msa_dir),
                "--pdb-mirror-path", str(mirror),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output

        # The hit structure was fetched.
        assert (msa_dir / "template_structures" / "2q2k.cif").exists()

        # The alignment targets label chain B (author chain A of 2q2k).
        m8 = msa_dir / "of3_template_alignments" / rep_id / "colabfold_template.m8"
        assert m8.read_text().split("\t")[1] == "2q2k_B"


@pytest.mark.integration
def test_fetch_templates_direct_download(ubq):
    """Default path: no mirror, so the hit is downloaded from RCSB as <id>.cif."""
    output_dir, msa_dir, seq, _ = ubq
    _write_m8(seq, msa_dir, ["1ubi_A"])

    summary = fetch_templates(output_dir, msa_dir)

    cif = msa_dir / "template_structures" / "1ubi.cif"

    # One structure was downloaded, and the file is a CIF for 1UBI.
    assert summary.n_structures == 1
    assert cif.read_text().startswith("data_1UBI")


def test_max_hits_defaults_to_50():
    """fetch-templates (and finetune-pipeline, which uses its defaults) resolves 50 hits per sequence."""
    from aw_of3_finetune.cli import fetch_templates as fetch_templates_command

    expected_default = 50
    actual_default = next(p.default for p in fetch_templates_command.params if p.name == "max_hits")
    assert actual_default == expected_default
