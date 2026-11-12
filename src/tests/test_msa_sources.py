"""User-provided MSAs: the sources manifest, file checks, the source index and import.

Test structure
--------------
TestLoadManifest    sources manifest (YAML/JSON): directories, globs, source-name rules.
TestScanSources     per-file checks before import: query match, duplicates, OF3 format.
TestMsaIndex        msa_index.yml: source name -> store folder, legacy folders, no mixing.
TestImportUserMsas  import_user_msas: stores, index, of3_alignments, reruns.
TestImportMsasCli   the import-msas command.
TestGenerateMsasIndex  generate-msas writes the source index and refuses user folders.
TestMsaSettings     train.yml's msa section: OF3 defaults, extended for new source names.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml
from atomworks.ml.preprocessing.msa.finding import find_msas
from atomworks.ml.preprocessing.msa.organizing import MSAOrganizationConfig, organize_msas
from click.testing import CliRunner

import aw_of3_finetune.cli as cli_mod
from aw_of3_finetune.cli import (
    finetune_pipeline,
    finetune_prep,
    generate_msas,
    generate_train_config,
    import_msas,
)
from aw_of3_finetune.training_config import discover_msa_sources, msa_settings_for_sources
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings

from aw_of3_finetune.msa import alignment_rep_id, chain_canonical_sequences
from aw_of3_finetune.msa_sources import (
    MixedMsaDirError,
    MsaSourcesError,
    SourceSpec,
    add_to_msa_index,
    import_user_msas,
    load_msa_sources_manifest,
    msa_dir_kind,
    read_msa_index,
    scan_sources,
)

SEQ_A = "MKTAYIAKQRQISFVKSHFSRQ"
SEQ_B = "GSHMLEDPVDAFQLWLKEQ"


def _touch(path: Path, text: str = ">q\nAAA\n") -> Path:
    """Write a small file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _write_manifest(path: Path, mapping) -> Path:
    """Write a manifest as YAML (or JSON for a .json path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping) if path.suffix == ".json" else yaml.safe_dump(mapping))
    return path


class TestLoadManifest:
    def test_yaml_and_json_load_same_mapping(self, tmp_path):
        """The same mapping as YAML or JSON gives the same sources and files."""
        _touch(tmp_path / "sub" / "x.a3m")
        from_yaml = load_msa_sources_manifest(_write_manifest(tmp_path / "m.yml", {"uniref90_hits": "sub"}))
        from_json = load_msa_sources_manifest(_write_manifest(tmp_path / "m.json", {"uniref90_hits": "sub"}))

        assert from_yaml == from_json
        assert from_yaml["uniref90_hits"].files == [tmp_path / "sub" / "x.a3m"]

    def test_relative_dir_resolves_against_manifest(self, tmp_path, monkeypatch):
        """A relative directory is read relative to the manifest's folder, not the working directory."""
        _touch(tmp_path / "sub" / "x.a3m")
        manifest = _write_manifest(tmp_path / "m.yml", {"uniref90_hits": "sub"})
        monkeypatch.chdir(tmp_path.parent)

        assert load_msa_sources_manifest(manifest)["uniref90_hits"].files == [tmp_path / "sub" / "x.a3m"]

    def test_dir_value_is_recursive(self, tmp_path):
        """A directory value selects files in nested folders too."""
        nested = _touch(tmp_path / "sub" / "a" / "x.a3m")
        top = _touch(tmp_path / "sub" / "y.a3m")

        spec = load_msa_sources_manifest(_write_manifest(tmp_path / "m.yml", {"s": "sub"}))["s"]

        assert spec.files == sorted([nested, top])
        assert spec.spec == "sub"

    def test_glob_value_selects_one_file_per_chain_dir(self, tmp_path):
        """A glob picks the one source file out of each per-chain folder."""
        a = _touch(tmp_path / "chains" / "A" / "cfdb_uniref30.a3m")
        _touch(tmp_path / "chains" / "A" / "rfam_hits.a3m")
        b = _touch(tmp_path / "chains" / "B" / "cfdb_uniref30.a3m")

        manifest = _write_manifest(tmp_path / "m.yml", {"cfdb_uniref30": "chains/*/cfdb_uniref30.a3m"})

        assert load_msa_sources_manifest(manifest)["cfdb_uniref30"].files == [a, b]

    def test_glob_matching_nothing_errors(self, tmp_path):
        """A glob that matches no file is an error naming the pattern."""
        manifest = _write_manifest(tmp_path / "m.yml", {"s": "chains/*/nothing_here.a3m"})

        with pytest.raises(MsaSourcesError, match="nothing_here"):
            load_msa_sources_manifest(manifest)

    def test_missing_dir_errors(self, tmp_path):
        """A directory that does not exist is an error naming it."""
        manifest = _write_manifest(tmp_path / "m.yml", {"s": "does_not_exist"})

        with pytest.raises(MsaSourcesError, match="does_not_exist"):
            load_msa_sources_manifest(manifest)

    @pytest.mark.parametrize("name", ["colabfold_paired", "dummy", "templates"])
    def test_rejected_names_error(self, tmp_path, name):
        """Names with an OpenFold3-internal or reserved meaning are refused."""
        _touch(tmp_path / "sub" / "x.a3m")
        manifest = _write_manifest(tmp_path / "m.yml", {name: "sub"})

        with pytest.raises(MsaSourcesError, match=name):
            load_msa_sources_manifest(manifest)

    @pytest.mark.parametrize("name", ["has.dot", "", "with space"])
    def test_invalid_name_errors(self, tmp_path, name):
        """Source names are letters, digits, _ and - only, since they become file names."""
        _touch(tmp_path / "sub" / "x.a3m")
        manifest = _write_manifest(tmp_path / "m.yml", {name: "sub"})

        with pytest.raises(MsaSourcesError):
            load_msa_sources_manifest(manifest)

    def test_uniprot_names_accepted(self, tmp_path):
        """uniprot_hits and uniprot are allowed (their headers are the user's responsibility)."""
        _touch(tmp_path / "sub" / "x.a3m")
        manifest = _write_manifest(tmp_path / "m.yml", {"uniprot_hits": "sub", "uniprot": "sub"})

        assert set(load_msa_sources_manifest(manifest)) == {"uniprot_hits", "uniprot"}

    @pytest.mark.parametrize("content", [["a", "b"], {"a": 3}, "just a string"])
    def test_not_a_mapping_errors(self, tmp_path, content):
        """The manifest must map source names to path strings."""
        manifest = _write_manifest(tmp_path / "m.yml", content)

        with pytest.raises(MsaSourcesError):
            load_msa_sources_manifest(manifest)

    def test_empty_manifest_errors(self, tmp_path):
        """An empty manifest names no sources: an error, not a silent no-op."""
        manifest = tmp_path / "m.yml"
        manifest.write_text("")

        with pytest.raises(MsaSourcesError, match="no sources"):
            load_msa_sources_manifest(manifest)

    def test_all_problems_reported_together(self, tmp_path):
        """One missing directory and one rejected name are both in a single error."""
        _touch(tmp_path / "sub" / "x.a3m")
        manifest = _write_manifest(tmp_path / "m.yml", {"good": "does_not_exist", "dummy": "sub"})

        with pytest.raises(MsaSourcesError) as error:
            load_msa_sources_manifest(manifest)

        assert "does_not_exist" in str(error.value)
        assert "dummy" in str(error.value)


def _a3m(query: str, rows: list[tuple[str, str]] = ()) -> str:
    """A3M text: the query (header 'query') then (header, aligned sequence) rows."""
    return "".join(f">{h}\n{seq}\n" for h, seq in [("query", query), *rows])


def _spec(name: str, *files: Path) -> SourceSpec:
    """A manifest entry selecting exactly these files."""
    return SourceSpec(name=name, spec="(test)", files=sorted(files))


def _queries(*seqs: str) -> dict[str, str]:
    """rep_id -> sequence, as collect_msa_queries returns."""
    return {alignment_rep_id(s): s for s in seqs}


def _store_file(store_root: Path, source: str, seq: str, text: str) -> Path:
    """Write an existing store file (as a previous import would) for seq."""
    rep_id = alignment_rep_id(seq)
    path = store_root / source / rep_id[:2] / f"{rep_id}.a3m.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        f.write(text)
    return path


class TestScanSources:
    def test_matched_file_has_rep_id(self, tmp_path):
        f = _touch(tmp_path / "a.a3m", _a3m(SEQ_A))

        scan = scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), tmp_path / "store")

        assert scan[0].source == "s"
        assert scan[0].files[0].rep_id == alignment_rep_id(SEQ_A)
        assert scan[0].files[0].path == f
        assert scan[0].unmatched == []

    def test_unmatched_file_listed(self, tmp_path):
        """A file whose query is not in the dataset is listed, not an error."""
        f = _touch(tmp_path / "b.a3m", _a3m(SEQ_B))

        scan = scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), tmp_path / "store")

        assert scan[0].unmatched == [f]

    def test_query_with_insertions_is_unmatched(self, tmp_path):
        """A query row with lowercase or gaps does not hash to the chain: reported unmatched by filename."""
        lower = _touch(tmp_path / "lower.a3m", _a3m(SEQ_A[:5] + "x" + SEQ_A[5:]))
        gapped = _touch(tmp_path / "gapped.a3m", _a3m(SEQ_A[:5] + "-" + SEQ_A[6:]))

        scan = scan_sources({"s": _spec("s", lower, gapped)}, _queries(SEQ_A), tmp_path / "store")

        assert sorted(scan[0].unmatched) == sorted([lower, gapped])
        assert scan[0].files == []

    def test_wrapped_query_is_matched(self, tmp_path):
        """A query wrapped over several lines (valid for OpenFold3) matches its chain."""
        wrapped = "\n".join(SEQ_A[i:i + 8] for i in range(0, len(SEQ_A), 8))
        f = _touch(tmp_path / "wrapped.a3m", f">query\n{wrapped}\n>hit\n{SEQ_A}\n")

        scan = scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), tmp_path / "store")

        assert [x.rep_id for x in scan[0].files] == [alignment_rep_id(SEQ_A)]

    def test_wrapped_queries_differing_after_first_line_are_not_duplicates(self, tmp_path):
        one = _touch(tmp_path / "one.a3m", f">query\n{SEQ_A[:8]}\n{SEQ_A[8:]}\n")
        two = _touch(tmp_path / "two.a3m", f">query\n{SEQ_A[:8]}\n{SEQ_B}\n")

        scan = scan_sources({"s": _spec("s", one, two)}, _queries(SEQ_A), tmp_path / "store")

        assert [x.path for x in scan[0].files] == [one]
        assert scan[0].unmatched == [two]

    def test_unmatched_files_are_not_format_checked(self, tmp_path):
        """A malformed file for a chain outside the dataset does not block the import."""
        good = _touch(tmp_path / "good.a3m", _a3m(SEQ_A))
        broken = _touch(tmp_path / "broken.a3m", _a3m(SEQ_B, [("h", SEQ_B + "K")]))

        scan = scan_sources({"s": _spec("s", good, broken)}, _queries(SEQ_A), tmp_path / "store")

        assert [x.path for x in scan[0].files] == [good]
        assert scan[0].unmatched == [broken]

    def test_unmatched_duplicates_are_not_errors(self, tmp_path):
        """Two files for the same out-of-dataset sequence are skipped, not refused."""
        one = _touch(tmp_path / "one.a3m", _a3m(SEQ_B))
        two = _touch(tmp_path / "two.a3m", _a3m(SEQ_B))
        good = _touch(tmp_path / "good.a3m", _a3m(SEQ_A))

        scan = scan_sources({"s": _spec("s", one, two, good)}, _queries(SEQ_A), tmp_path / "store")

        assert sorted(scan[0].unmatched) == [one, two]

    def test_duplicate_in_source_errors(self, tmp_path):
        one = _touch(tmp_path / "one.a3m", _a3m(SEQ_A))
        two = _touch(tmp_path / "two.a3m", _a3m(SEQ_A, [("h", SEQ_A)]))

        with pytest.raises(MsaSourcesError) as error:
            scan_sources({"s": _spec("s", one, two)}, _queries(SEQ_A), tmp_path / "store")

        assert "one.a3m" in str(error.value) and "two.a3m" in str(error.value)

    def test_duplicate_across_extensions(self, tmp_path):
        """x.a3m and x.a3m.gz with the same query are duplicates too."""
        plain = _touch(tmp_path / "x.a3m", _a3m(SEQ_A))
        zipped = tmp_path / "x.a3m.gz"
        with gzip.open(zipped, "wt") as f:
            f.write(_a3m(SEQ_A))

        with pytest.raises(MsaSourcesError, match="x.a3m.gz"):
            scan_sources({"s": _spec("s", plain, zipped)}, _queries(SEQ_A), tmp_path / "store")

    def test_same_query_in_two_sources_is_fine(self, tmp_path):
        """One MSA per source per sequence: the same query under two sources is expected."""
        a = _touch(tmp_path / "a" / "x.a3m", _a3m(SEQ_A))
        b = _touch(tmp_path / "b" / "x.a3m", _a3m(SEQ_A))

        scan = scan_sources({"a": _spec("a", a), "b": _spec("b", b)}, _queries(SEQ_A), tmp_path / "store")

        assert [len(s.files) for s in scan] == [1, 1]

    def test_complex_a3m_errors(self, tmp_path):
        """A ColabFold complex (multi-chain) A3M is refused, not imported as its first chain's MSA."""
        f = _touch(tmp_path / "pair.a3m", f"#10,12\t1,1\n>101\t102\n{SEQ_A}{SEQ_B}\n")

        with pytest.raises(MsaSourcesError, match="multi-chain"):
            scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), tmp_path / "store")

    def test_ragged_row_errors(self, tmp_path):
        """A row whose aligned length differs from the query fails OF3's parser: error naming the file."""
        f = _touch(tmp_path / "ragged.a3m", _a3m(SEQ_A, [("h", SEQ_A + "K")]))

        with pytest.raises(MsaSourcesError, match="ragged.a3m"):
            scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), tmp_path / "store")

    def test_uniprot_hits_bad_headers_error(self, tmp_path):
        """uniprot_hits feeds species pairing, so its headers must parse; other sources don't need to."""
        f = _touch(tmp_path / "u.a3m", _a3m(SEQ_A, [("seq1", SEQ_A)]))

        with pytest.raises(MsaSourcesError, match="u.a3m"):
            scan_sources({"uniprot_hits": _spec("uniprot_hits", f)}, _queries(SEQ_A), tmp_path / "store")
        scan = scan_sources({"uniref90_hits": _spec("uniref90_hits", f)}, _queries(SEQ_A), tmp_path / "store")
        assert len(scan[0].files) == 1

    def test_uniprot_hits_good_headers_pass(self, tmp_path):
        f = _touch(tmp_path / "u.a3m", _a3m(SEQ_A, [("tr|A0A1W9RZR3|A0A1W9RZR3_9BACT/19-121", SEQ_A)]))

        scan = scan_sources({"uniprot_hits": _spec("uniprot_hits", f)}, _queries(SEQ_A), tmp_path / "store")

        assert len(scan[0].files) == 1

    def test_existing_store_file_with_different_content_errors(self, tmp_path):
        store = tmp_path / "store"
        _store_file(store, "s", SEQ_A, _a3m(SEQ_A, [("old", SEQ_A)]))
        f = _touch(tmp_path / "a.a3m", _a3m(SEQ_A))

        with pytest.raises(MsaSourcesError, match="msa_sources/s"):
            scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), store)

    def test_existing_identical_store_file_ok(self, tmp_path):
        store = tmp_path / "store"
        _store_file(store, "s", SEQ_A, _a3m(SEQ_A))
        f = _touch(tmp_path / "a.a3m", _a3m(SEQ_A))

        scan = scan_sources({"s": _spec("s", f)}, _queries(SEQ_A), store)

        assert len(scan[0].files) == 1

    def test_non_a3m_files_ignored(self, tmp_path):
        f = _touch(tmp_path / "a.a3m", _a3m(SEQ_A))
        notes = _touch(tmp_path / "notes.txt", "hello")

        scan = scan_sources({"s": _spec("s", f, notes)}, _queries(SEQ_A), tmp_path / "store")

        assert scan[0].ignored == [notes]

    def test_source_with_no_a3m_errors(self, tmp_path):
        notes = _touch(tmp_path / "notes.txt", "hello")

        with pytest.raises(MsaSourcesError, match="no A3M"):
            scan_sources({"s": _spec("s", notes)}, _queries(SEQ_A), tmp_path / "store")

    def test_errors_from_two_sources_reported_together(self, tmp_path):
        ragged = _touch(tmp_path / "ragged.a3m", _a3m(SEQ_A, [("h", SEQ_A + "K")]))
        pair = _touch(tmp_path / "pair.a3m", f"#10,12\t1,1\n>101\t102\n{SEQ_A}{SEQ_B}\n")

        with pytest.raises(MsaSourcesError) as error:
            scan_sources({"a": _spec("a", ragged), "b": _spec("b", pair)}, _queries(SEQ_A), tmp_path / "store")

        assert "ragged.a3m" in str(error.value) and "pair.a3m" in str(error.value)


def _write_index(msa_dir: Path, mapping: dict) -> Path:
    """Write <msa_dir>/msa_index.yml."""
    msa_dir.mkdir(parents=True, exist_ok=True)
    path = msa_dir / "msa_index.yml"
    path.write_text(yaml.safe_dump(mapping))
    return path


class TestMsaIndex:
    def test_reads_user_index(self, tmp_path):
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, {"b": "msa_sources/b", "a": "msa_sources/a"})

        index = read_msa_index(msa_dir)

        assert list(index) == ["a", "b"]
        assert index == {"a": msa_dir / "msa_sources" / "a", "b": msa_dir / "msa_sources" / "b"}
        assert msa_dir_kind(msa_dir) == "user"

    def test_reads_colabfold_index(self, tmp_path):
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, {"colabfold_main": "unpaired"})

        assert read_msa_index(msa_dir) == {"colabfold_main": msa_dir / "unpaired"}
        assert msa_dir_kind(msa_dir) == "colabfold"

    def test_legacy_unpaired_without_index(self, tmp_path):
        """A folder from before the index existed reads as colabfold_main; reading writes nothing."""
        msa_dir = tmp_path / "msas"
        (msa_dir / "unpaired").mkdir(parents=True)

        assert read_msa_index(msa_dir) == {"colabfold_main": msa_dir / "unpaired"}
        assert msa_dir_kind(msa_dir) == "colabfold"
        assert not (msa_dir / "msa_index.yml").exists()

    def test_empty_dir(self, tmp_path):
        assert read_msa_index(tmp_path) == {}
        assert msa_dir_kind(tmp_path) == "empty"
        assert read_msa_index(tmp_path / "missing") == {}

    def test_mixed_index_errors(self, tmp_path):
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, {"colabfold_main": "unpaired", "x": "msa_sources/x"})

        with pytest.raises(MixedMsaDirError, match="--organized-msa-dir"):
            read_msa_index(msa_dir)

    @pytest.mark.parametrize("value", ["/abs/path", "../elsewhere", "msa_sources/../../out"])
    def test_absolute_or_escaping_value_errors(self, tmp_path, value):
        """Store folders stay inside the MSA folder, so the folder can be moved."""
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, {"x": value})

        with pytest.raises(MsaSourcesError, match="msa_index.yml"):
            read_msa_index(msa_dir)

    def test_rejected_name_in_index_errors(self, tmp_path):
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, {"dummy": "msa_sources/dummy"})

        with pytest.raises(MsaSourcesError, match="dummy"):
            read_msa_index(msa_dir)

    def test_malformed_index_errors(self, tmp_path):
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, ["not", "a", "mapping"])

        with pytest.raises(MsaSourcesError, match="msa_index.yml"):
            read_msa_index(msa_dir)

    def test_add_merges_and_newer_wins(self, tmp_path):
        msa_dir = tmp_path / "msas"
        add_to_msa_index(msa_dir, {"a": "msa_sources/a", "b": "msa_sources/old_b"})
        add_to_msa_index(msa_dir, {"b": "msa_sources/b", "c": "msa_sources/c"})

        assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {
            "a": "msa_sources/a", "b": "msa_sources/b", "c": "msa_sources/c",
        }

    def test_add_refuses_to_write_a_mix(self, tmp_path):
        msa_dir = tmp_path / "msas"
        index = _write_index(msa_dir, {"colabfold_main": "unpaired"})
        before = index.read_text()

        with pytest.raises(MixedMsaDirError):
            add_to_msa_index(msa_dir, {"x": "msa_sources/x"})

        assert index.read_text() == before

    def test_add_to_legacy_folder_keeps_colabfold_main(self, tmp_path):
        """Adding to a legacy folder starts from its implicit colabfold_main entry."""
        msa_dir = tmp_path / "msas"
        (msa_dir / "unpaired").mkdir(parents=True)

        add_to_msa_index(msa_dir, {"colabfold_main": "unpaired"})

        assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {"colabfold_main": "unpaired"}

    def test_add_refuses_malformed_existing_index(self, tmp_path):
        """A hand-broken index is reported, not silently overwritten."""
        msa_dir = tmp_path / "msas"
        index = _write_index(msa_dir, ["not", "a", "mapping"])
        before = index.read_text()

        with pytest.raises(MsaSourcesError, match="msa_index.yml"):
            add_to_msa_index(msa_dir, {"x": "msa_sources/x"})

        assert index.read_text() == before


def _dataset(cif_path: Path, tmp_path: Path) -> tuple[Path, str]:
    """A finetune-prep output directory for one CIF, and its protein chain's sequence."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif_path.resolve())
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        finetune_prep, ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return output_dir, chain_canonical_sequences(cif_path)["A"]


class TestImportUserMsas:
    def test_store_layout_matches_organize_msas(self, cif_protein_monomer, tmp_path):
        """An imported file lands where atomworks' organize_msas would put it."""
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        src = _touch(tmp_path / "user" / "chain_a.a3m", _a3m(seq))
        manifest = _write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"})
        msa_dir = tmp_path / "msas"

        import_user_msas(output_dir, msa_dir, manifest)
        organize_msas(tmp_path / "user", tmp_path / "aw", MSAOrganizationConfig(
            input_extension=".a3m", output_extension=".a3m.gz", sharding_pattern="/0:2/",
        ))

        ours = {p.relative_to(msa_dir / "msa_sources" / "my_lab_db") for p in (msa_dir / "msa_sources" / "my_lab_db").rglob("*.a3m.gz")}
        theirs = {p.relative_to(tmp_path / "aw") for p in (tmp_path / "aw").rglob("*.a3m.gz")}
        assert ours == theirs != set()
        assert src.exists()  # copied, not moved

    def test_import_writes_store_and_of3_alignments(self, cif_protein_monomer, tmp_path):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        text = _a3m(seq, [("hit", seq)])
        _touch(tmp_path / "user" / "a.a3m", text)
        msa_dir = tmp_path / "msas"

        import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"}))

        missing, _ = find_msas(
            [seq], [{"dir": str(msa_dir / "msa_sources" / "my_lab_db"), "extension": ".a3m.gz", "directory_depth": 1}]
        )
        assert missing == []
        assert (msa_dir / "of3_alignments" / alignment_rep_id(seq) / "my_lab_db.a3m").read_text() == text

    def test_two_sources_two_files(self, cif_protein_monomer, tmp_path):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "a" / "x.a3m", _a3m(seq))
        _touch(tmp_path / "b" / "x.a3m", _a3m(seq, [("hit", seq)]))
        msa_dir = tmp_path / "msas"

        summary = import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m.yml", {"src_a": "a", "src_b": "b"}))

        rep_dir = msa_dir / "of3_alignments" / alignment_rep_id(seq)
        assert sorted(p.name for p in rep_dir.iterdir()) == ["src_a.a3m", "src_b.a3m"]
        assert summary.covered == {"src_a": 1, "src_b": 1}
        assert summary.reorganize.written == {"src_a": 1, "src_b": 1}

    def test_index_written_and_merged(self, cif_protein_monomer, tmp_path):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "a" / "x.a3m", _a3m(seq))
        _touch(tmp_path / "b" / "x.a3m", _a3m(seq))
        msa_dir = tmp_path / "msas"

        import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m1.yml", {"a": "a"}))
        assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {"a": "msa_sources/a"}
        import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m2.yml", {"b": "b"}))

        assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {"a": "msa_sources/a", "b": "msa_sources/b"}
        assert sorted(p.name for p in msa_dir.iterdir()) == ["msa_index.yml", "msa_sources", "of3_alignments"]

    @pytest.mark.parametrize("with_index", [True, False])
    def test_import_refuses_colabfold_dir(self, cif_protein_monomer, tmp_path, with_index):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "a" / "x.a3m", _a3m(seq))
        msa_dir = tmp_path / "msas"
        (msa_dir / "unpaired").mkdir(parents=True)
        if with_index:
            _write_index(msa_dir, {"colabfold_main": "unpaired"})

        with pytest.raises(MixedMsaDirError, match="unpaired"):
            import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m.yml", {"a": "a"}))

        assert not (msa_dir / "msa_sources").exists()

    def test_failed_scan_copies_nothing(self, cif_protein_monomer, tmp_path):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "a" / "good.a3m", _a3m(seq))
        _touch(tmp_path / "b" / "bad.a3m", _a3m(seq, [("h", seq + "K")]))
        msa_dir = tmp_path / "msas"

        with pytest.raises(MsaSourcesError, match="bad.a3m"):
            import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m.yml", {"a": "a", "b": "b"}))

        assert not (msa_dir / "msa_sources").exists()
        assert not (msa_dir / "msa_index.yml").exists()

    def test_reimport_same_manifest_is_noop(self, cif_protein_monomer, tmp_path):
        """Importing the same manifest again rewrites nothing."""
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "a" / "x.a3m", _a3m(seq))
        manifest = _write_manifest(tmp_path / "m.yml", {"a": "a"})
        msa_dir = tmp_path / "msas"
        import_user_msas(output_dir, msa_dir, manifest)
        files = sorted(p for p in msa_dir.rglob("*") if p.is_file() and p.name != "msa_index.yml")
        mtimes = {p: p.stat().st_mtime_ns for p in files}

        summary = import_user_msas(output_dir, msa_dir, manifest)

        assert {p: p.stat().st_mtime_ns for p in files} == mtimes
        assert summary.reorganize.written == {"a": 0}

    def test_all_unmatched_warns_not_errors(self, cif_protein_monomer, tmp_path):
        """A manifest whose files match no dataset chain imports, and says nothing is covered."""
        output_dir, _ = _dataset(cif_protein_monomer, tmp_path)
        other = _touch(tmp_path / "src" / "other.a3m", _a3m(SEQ_B))
        msa_dir = tmp_path / "msas"

        summary = import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m.yml", {"src": "src"}))

        assert summary.covered == {"src": 0}
        assert summary.scans[0].unmatched == [other]
        assert summary.reorganize.missing == 1
        # Skipped: files for chains outside this dataset are not copied into the store.
        assert not list((msa_dir / "msa_sources").rglob("*.a3m.gz"))


class TestImportMsasCli:
    def test_cli_imports_and_reports(self, cif_protein_monomer, tmp_path):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "user" / "a.a3m", _a3m(seq))
        _touch(tmp_path / "user" / "other.a3m", _a3m(SEQ_B))
        manifest = _write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"})

        result = CliRunner().invoke(
            import_msas, ["--output-dir", str(output_dir), "--msa-sources", str(manifest)], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "my_lab_db: imported 1 file(s)" in result.stdout and "covers 1" in result.stdout
        assert "other.a3m" in result.stderr  # unmatched files are a warning
        assert (output_dir / "organized_msas" / "of3_alignments" / alignment_rep_id(seq) / "my_lab_db.a3m").exists()

    def test_cli_reports_sequences_without_msa(self, cif_protein_monomer, tmp_path):
        output_dir, _ = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "user" / "other.a3m", _a3m(SEQ_B))
        manifest = _write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"})

        result = CliRunner().invoke(
            import_msas, ["--output-dir", str(output_dir), "--msa-sources", str(manifest)], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "1 sequence(s) have no MSA from any source" in result.stderr

    def test_cli_bad_file_exits_1_with_message(self, cif_protein_monomer, tmp_path):
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "user" / "ragged.a3m", _a3m(seq, [("h", seq + "K")]))
        manifest = _write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"})

        result = CliRunner().invoke(import_msas, ["--output-dir", str(output_dir), "--msa-sources", str(manifest)])

        assert result.exit_code == 1
        assert "ragged.a3m" in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)


    def test_cli_warns_about_indexed_sources_missing_from_manifest(self, cif_protein_monomer, tmp_path):
        """A source imported earlier but left out of this manifest stays in training: say so, and how to remove it."""
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "a" / "x.a3m", _a3m(seq))
        _touch(tmp_path / "b" / "x.a3m", _a3m(seq))
        args = ["--output-dir", str(output_dir), "--msa-sources"]
        CliRunner().invoke(import_msas, [*args, str(_write_manifest(tmp_path / "m1.yml", {"old_db": "a"}))], catch_exceptions=False)

        result = CliRunner().invoke(
            import_msas, [*args, str(_write_manifest(tmp_path / "m2.yml", {"new_db": "b"}))], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "old_db" in result.stderr
        assert "msa_index.yml" in result.stderr

    def test_cli_caps_unmatched_listing(self, cif_protein_monomer, tmp_path):
        """A large library of unrelated MSAs gives a count and a short sample, not one line per file."""
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        _touch(tmp_path / "lib" / "match.a3m", _a3m(seq))
        for i in range(30):
            _touch(tmp_path / "lib" / f"other_{i:02d}.a3m", _a3m(SEQ_B + "A" * (i + 1)))
        manifest = _write_manifest(tmp_path / "m.yml", {"lib": "lib"})

        result = CliRunner().invoke(
            import_msas, ["--output-dir", str(output_dir), "--msa-sources", str(manifest)], catch_exceptions=False
        )

        assert result.exit_code == 0, result.output
        assert "30 file(s) match no sequence" in result.stderr
        assert result.stderr.count("other_") == 20
        assert "and 10 more" in result.stderr


class TestGenerateMsasIndex:
    @pytest.fixture
    def no_colabfold(self, monkeypatch):
        """ColabFold replaced by a no-op that records its calls."""
        calls: list[dict] = []
        monkeypatch.setattr(cli_mod, "make_msas_colabfold_server_batch", lambda *a, **kw: calls.append(kw))
        return calls

    def test_generate_msas_writes_index(self, cif_protein_monomer, tmp_path, no_colabfold):
        output_dir, _ = _dataset(cif_protein_monomer, tmp_path)
        msa_dir = tmp_path / "msas"

        result = CliRunner().invoke(
            generate_msas, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert len(no_colabfold) == 1
        assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {"colabfold_main": "unpaired"}

    def test_generate_msas_skip_existing_writes_index_for_legacy_dir(self, cif_protein_monomer, tmp_path, no_colabfold):
        """A legacy folder whose MSAs all exist: nothing to query, but the index is written."""
        output_dir, seq = _dataset(cif_protein_monomer, tmp_path)
        msa_dir = tmp_path / "msas"
        rep_id = alignment_rep_id(seq)
        store = msa_dir / "unpaired" / rep_id[:2] / f"{rep_id}.a3m.gz"
        store.parent.mkdir(parents=True)
        with gzip.open(store, "wt") as f:
            f.write(_a3m(seq))

        result = CliRunner().invoke(
            generate_msas,
            ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir), "--skip-existing"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert no_colabfold == []
        assert yaml.safe_load((msa_dir / "msa_index.yml").read_text()) == {"colabfold_main": "unpaired"}

    def test_generate_msas_failure_writes_no_index(self, cif_protein_monomer, tmp_path, monkeypatch):
        output_dir, _ = _dataset(cif_protein_monomer, tmp_path)
        msa_dir = tmp_path / "msas"

        def fail(*a, **kw):
            raise RuntimeError("server down")

        monkeypatch.setattr(cli_mod, "make_msas_colabfold_server_batch", fail)
        result = CliRunner().invoke(
            generate_msas, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)]
        )

        assert result.exit_code != 0
        assert not (msa_dir / "msa_index.yml").exists()

    def test_generate_msas_refuses_user_dir(self, cif_protein_monomer, tmp_path, monkeypatch):
        output_dir, _ = _dataset(cif_protein_monomer, tmp_path)
        msa_dir = tmp_path / "msas"
        _write_index(msa_dir, {"x": "msa_sources/x"})
        monkeypatch.setattr(
            cli_mod, "make_msas_colabfold_server_batch", lambda *a, **k: pytest.fail("ColabFold must not be called")
        )

        result = CliRunner().invoke(
            generate_msas, ["--output-dir", str(output_dir), "--organized-msa-dir", str(msa_dir)]
        )

        assert result.exit_code == 1
        assert "fresh --organized-msa-dir" in result.output


def _pipeline_output(tmp_path: Path, ccd_path: Path, cifs: dict[str, Path], *extra: str) -> Path:
    """A full finetune-pipeline output (no MSAs) for these {pdb_id: cif} inputs."""
    cif_dir = tmp_path / "input"
    cif_dir.mkdir()
    for pdb_id, cif in cifs.items():
        (cif_dir / f"{pdb_id}.cif").symlink_to(cif.resolve())
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(finetune_pipeline, [
        "--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path),
        "--skip-report", "--skip-validation", *extra,
    ], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return output_dir


class TestMsaSettings:
    @pytest.mark.parametrize("sources", [{"colabfold_main"}, {"uniref90_hits", "uniprot_hits"}])
    def test_known_names_keep_today_section(self, sources):
        """Names OF3 already reads: the msa section stays as it is today, so OF3's defaults apply."""
        assert msa_settings_for_sources(sources) == {"moltypes": ["PROTEIN"]}

    def test_new_name_extends_defaults(self):
        defaults = MSASettings()

        settings = msa_settings_for_sources({"my_lab_db"})

        assert settings["moltypes"] == ["PROTEIN"]
        assert settings["max_seq_counts"] == {**defaults.max_seq_counts, "my_lab_db": 16384}
        assert settings["aln_order"] == [*defaults.aln_order, "my_lab_db"]

    def test_new_names_appended_alphabetically(self):
        assert msa_settings_for_sources({"zeta", "alpha"})["aln_order"][-2:] == ["alpha", "zeta"]

    def test_mixed_known_and_new_names(self):
        """One OF3 name and one new name: every OF3 default kept, only the new name added."""
        defaults = MSASettings()

        settings = msa_settings_for_sources({"uniref90_hits", "my_lab_db"})

        assert settings["max_seq_counts"] == {**defaults.max_seq_counts, "my_lab_db": 16384}
        assert settings["aln_order"] == [*defaults.aln_order, "my_lab_db"]

    def test_uniprot_gets_count_not_aln_order(self):
        """uniprot is paired on, not stacked: OF3 must read it (a count) but not list it in aln_order."""
        settings = msa_settings_for_sources({"uniprot"})

        assert "uniprot" in settings["max_seq_counts"]
        assert "uniprot" not in settings["aln_order"]

    def test_settings_validate_with_of3(self):
        MSASettings(**msa_settings_for_sources({"my_lab_db", "uniprot"}))

    def test_discover_reads_stems(self, tmp_path):
        _touch(tmp_path / "of3" / "aaa" / "my_lab_db.a3m")
        _touch(tmp_path / "of3" / "bbb" / "uniref90_hits.a3m")
        _touch(tmp_path / "of3" / "bbb" / "notes.txt")

        assert discover_msa_sources(tmp_path / "of3") == {"my_lab_db", "uniref90_hits"}
        assert discover_msa_sources(tmp_path / "missing") == set()

    def test_train_yml_has_settings_in_train_and_validation(self, cif_protein_monomer, cif_protein_only, ccd_path, tmp_path):
        """generate-train-config writes the extended settings into both dataset sections."""
        split_csv = _touch(tmp_path / "split.csv", "id,split\n3u8v,val\n")
        output_dir = _pipeline_output(
            tmp_path, ccd_path, {"3u8v": cif_protein_monomer, "2crb": cif_protein_only}, "--split-csv", str(split_csv)
        )
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        _touch(tmp_path / "user" / "a.a3m", _a3m(seq))
        result = CliRunner().invoke(import_msas, [
            "--output-dir", str(output_dir), "--msa-sources", str(_write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"})),
        ], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        result = CliRunner().invoke(generate_train_config, ["--output-dir", str(output_dir)], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        config = yaml.safe_load((output_dir / "train.yml").read_text())
        train_msa = config["dataset_configs"]["train"]["weighted-pdb"]["config"]["msa"]
        val_msa = config["dataset_configs"]["validation"]["val-weighted-pdb"]["config"]["msa"]
        assert train_msa["max_seq_counts"]["my_lab_db"] == 16384
        assert val_msa["max_seq_counts"]["my_lab_db"] == 16384
        assert val_msa["subsample_main"] is False

    def test_empty_alignments_warns_and_keeps_today_section(self, cif_protein_monomer, ccd_path, tmp_path):
        output_dir = _pipeline_output(tmp_path, ccd_path, {"3u8v": cif_protein_monomer})
        (output_dir / "organized_msas").mkdir()

        result = CliRunner().invoke(generate_train_config, ["--output-dir", str(output_dir)], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert "no MSAs" in result.stderr
        config = yaml.safe_load((output_dir / "train.yml").read_text())
        assert config["dataset_configs"]["train"]["weighted-pdb"]["config"]["msa"] == {"moltypes": ["PROTEIN"]}

    def test_of3_dataset_reads_new_source(self, cif_protein_monomer, ccd_path, tmp_path):
        """OpenFold3's own dataset, built from train.yml, loads every row of a new source."""
        output_dir = _pipeline_output(tmp_path, ccd_path, {"3u8v": cif_protein_monomer})
        seq = chain_canonical_sequences(cif_protein_monomer)["A"]
        hits = [("h1", "A" + seq[1:]), ("h2", "W" + seq[1:])]  # distinct rows: OF3 drops duplicates
        _touch(tmp_path / "user" / "a.a3m", _a3m(seq, hits))
        result = CliRunner().invoke(import_msas, [
            "--output-dir", str(output_dir), "--msa-sources", str(_write_manifest(tmp_path / "m.yml", {"my_lab_db": "user"})),
        ], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        result = CliRunner().invoke(generate_train_config, ["--output-dir", str(output_dir)], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        n_rows = _of3_msa_rows(output_dir)

        # Training subsamples the main MSA (at least one row), so a source OF3 reads gives the
        # query plus >= 1 row; a source it ignores leaves the query alone.
        assert n_rows > 1


def _of3_msa_rows(output_dir: Path) -> int:
    """MSA rows OpenFold3's WeightedPDBDataset loads for the first datapoint, configured from train.yml."""
    import warnings

    from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
    from openfold3.projects.of3_all_atom.config.dataset_configs import DATASET_CONFIG_REGISTRY, TrainingDatasetPaths

    WeightedPDBConfig = DATASET_CONFIG_REGISTRY.get("WeightedPDBDataset")

    config = yaml.safe_load((output_dir / "train.yml").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = WeightedPDBDataset(WeightedPDBConfig(
            name="user-msas",
            dataset_paths=TrainingDatasetPaths(**config["dataset_paths"]["weighted-pdb"]),
            msa=config["dataset_configs"]["train"]["weighted-pdb"]["config"]["msa"],
            crop={"token_crop": {"enabled": False}},
        ))
        item = dataset[0]
    return int(item["msa"].shape[0])
