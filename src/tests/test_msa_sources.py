"""User-provided MSAs: the sources manifest, file checks, the source index and import.

Test structure
--------------
TestLoadManifest    sources manifest (YAML/JSON): directories, globs, source-name rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aw_of3_finetune.msa_sources import MsaSourcesError, load_msa_sources_manifest


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
