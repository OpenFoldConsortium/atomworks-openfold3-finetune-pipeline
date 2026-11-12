"""User-provided MSAs: the sources manifest, file checks, the source index and import.

Test structure
--------------
TestLoadManifest    sources manifest (YAML/JSON): directories, globs, source-name rules.
TestScanSources     per-file checks before import: query match, duplicates, OF3 format.
TestMsaIndex        msa_index.yml: source name -> store folder, legacy folders, no mixing.
TestImportUserMsas  import_user_msas: stores, index, of3_alignments, reruns.
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

from aw_of3_finetune.cli import finetune_prep

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

        assert [u.path for u in scan[0].unmatched] == [f]

    def test_query_with_insertions_is_unmatched(self, tmp_path):
        """A query row with lowercase or gaps does not hash to the chain: reported unmatched by filename."""
        lower = _touch(tmp_path / "lower.a3m", _a3m(SEQ_A[:5] + "x" + SEQ_A[5:]))
        gapped = _touch(tmp_path / "gapped.a3m", _a3m(SEQ_A[:5] + "-" + SEQ_A[6:]))

        scan = scan_sources({"s": _spec("s", lower, gapped)}, _queries(SEQ_A), tmp_path / "store")

        assert sorted(u.path for u in scan[0].unmatched) == sorted([lower, gapped])
        assert scan[0].files == []

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
        _touch(tmp_path / "a" / "bad.a3m", _a3m(seq[:-1] + "W", [("h", seq + "K")]))
        msa_dir = tmp_path / "msas"

        with pytest.raises(MsaSourcesError, match="bad.a3m"):
            import_user_msas(output_dir, msa_dir, _write_manifest(tmp_path / "m.yml", {"a": "a"}))

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
        assert [u.path for u in summary.scans[0].unmatched] == [other]
        assert summary.reorganize.missing == 1
        # Still imported: the store is keyed by sequence, so a later dataset with this chain finds it.
        assert (msa_dir / "msa_sources" / "src" / alignment_rep_id(SEQ_B)[:2] / f"{alignment_rep_id(SEQ_B)}.a3m.gz").exists()
