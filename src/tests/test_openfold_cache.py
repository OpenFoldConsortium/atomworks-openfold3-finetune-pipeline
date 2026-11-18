"""Finding the CCD without --ccd-path, setup-finetuning, and the ./output default.

conftest's isolate_ccd_settings points $OPENFOLD_CACHE at an empty scratch directory and
unsets $CCD_PATH for every test, so each test sets up exactly the CCD settings it needs.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import aw_of3_finetune.cli
from aw_of3_finetune.cli.options import ccd_path_option
from aw_of3_finetune.cli.prep import setup_finetuning
from aw_of3_finetune.constants import DEFAULT_OUTPUT_DIR
from aw_of3_finetune.openfold_cache import (
    CHECKPOINT_ROOT_FILENAME,
    openfold_cache_dir,
    read_stored_ccd_path,
    resolve_ccd_path,
    stored_ccd_path_file,
    write_stored_ccd_path,
)


@pytest.fixture
def fake_ccd(tmp_path) -> Path:
    path = tmp_path / "components.cif"
    path.write_text("data_FAKE\n")
    return path


@pytest.fixture
def other_ccd(tmp_path) -> Path:
    path = tmp_path / "other.cif"
    path.write_text("data_OTHER\n")
    return path


@click.command()
@ccd_path_option(required=True, help="CCD.")
def _show_ccd(ccd_path: Path) -> None:
    click.echo(f"CCD={ccd_path}")


class TestResolveCcdPath:
    def test_nothing_set_finds_nothing(self):
        assert resolve_ccd_path(None) == (None, "")

    def test_stored_path_is_used_when_nothing_else_is_set(self, fake_ccd):
        write_stored_ccd_path(fake_ccd)
        path, source = resolve_ccd_path(None)
        assert path == fake_ccd.resolve()
        assert source == str(openfold_cache_dir() / "ccd_path")

    def test_env_var_wins_over_stored_path(self, fake_ccd, other_ccd, monkeypatch):
        write_stored_ccd_path(fake_ccd)
        monkeypatch.setenv("CCD_PATH", str(other_ccd))
        assert resolve_ccd_path(None) == (other_ccd, "$CCD_PATH")

    def test_flag_wins_over_env_var(self, fake_ccd, other_ccd, monkeypatch):
        monkeypatch.setenv("CCD_PATH", str(other_ccd))
        assert resolve_ccd_path(fake_ccd) == (fake_ccd, "--ccd-path")

    def test_stored_path_lives_in_openfold_cache(self, fake_ccd, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENFOLD_CACHE", str(tmp_path / "cache"))
        record = write_stored_ccd_path(fake_ccd)
        assert record == tmp_path / "cache" / "ccd_path"
        assert read_stored_ccd_path() == fake_ccd.resolve()


class TestCcdPathOption:
    def test_uses_stored_path_and_says_where_it_came_from(self, fake_ccd):
        write_stored_ccd_path(fake_ccd)
        result = CliRunner().invoke(_show_ccd, [])
        assert result.exit_code == 0, result.output
        assert f"CCD={fake_ccd.resolve()}" in result.output
        assert "from " in result.output

    def test_required_and_unset_points_at_setup(self):
        result = CliRunner().invoke(_show_ccd, [])
        assert result.exit_code != 0
        assert "setup-finetuning" in result.output
        assert "--ccd-path" in result.output

    def test_stored_path_to_a_missing_file_is_an_error(self, tmp_path):
        write_stored_ccd_path(tmp_path / "gone.cif")
        result = CliRunner().invoke(_show_ccd, [])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_optional_and_unset_gives_none(self):
        assert _optional_ccd() is None


def _optional_ccd() -> Path | None:
    seen = {}

    @click.command()
    @ccd_path_option(required=False, help="CCD.")
    def command(ccd_path: Path | None) -> None:
        seen["ccd"] = ccd_path

    result = CliRunner().invoke(command, [])
    assert result.exit_code == 0, result.output
    return seen["ccd"]


class TestSetupFinetuning:
    def test_records_an_existing_file(self, fake_ccd):
        result = CliRunner().invoke(setup_finetuning, ["--ccd-path", str(fake_ccd)])
        assert result.exit_code == 0, result.output
        assert read_stored_ccd_path() == fake_ccd.resolve()
        assert str(stored_ccd_path_file()) in result.output

    def test_already_set_up_does_nothing(self, fake_ccd, monkeypatch):
        write_stored_ccd_path(fake_ccd)
        monkeypatch.setattr("openfold3.core.utils.s3.download_s3_file",
                            lambda *a: pytest.fail("should not download"))
        result = CliRunner().invoke(setup_finetuning, [])
        assert result.exit_code == 0, result.output
        assert "already set up" in result.output

    def test_downloads_into_the_cache_dir(self, monkeypatch):
        calls = []

        def fake_download(bucket, key, local_path):
            calls.append((bucket, key))
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("data_FAKE\n")

        monkeypatch.setattr("openfold3.core.utils.s3.download_s3_file", fake_download)
        result = CliRunner().invoke(setup_finetuning, [])
        assert result.exit_code == 0, result.output
        assert calls == [("openfold3-data", "chemical_component_dictionary.cif")]
        assert read_stored_ccd_path() == (openfold_cache_dir() / "chemical_component_dictionary.cif").resolve()

    def test_points_at_setup_openfold_when_weights_are_not_set_up(self, fake_ccd):
        result = CliRunner().invoke(setup_finetuning, ["--ccd-path", str(fake_ccd)])
        assert "setup_openfold" in result.output

    def test_quiet_about_weights_when_setup_openfold_has_run(self, fake_ccd):
        cache = openfold_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        (cache / CHECKPOINT_ROOT_FILENAME).write_text(str(cache))
        result = CliRunner().invoke(setup_finetuning, ["--ccd-path", str(fake_ccd)])
        assert "setup_openfold" not in result.output

    def test_reminds_to_keep_openfold_cache_set(self, fake_ccd):
        # conftest sets $OPENFOLD_CACHE for every test.
        result = CliRunner().invoke(setup_finetuning, ["--ccd-path", str(fake_ccd)])
        assert "$OPENFOLD_CACHE" in result.output


class TestOutputDirDefault:
    def test_every_command_on_the_pipeline_folder_defaults_to_output(self):
        # prepare-inputs is left out: its --output-dir is a separate folder of prepared CIFs.
        modules = [importlib.import_module(f"aw_of3_finetune.cli.{m.name}")
                   for m in pkgutil.iter_modules(aw_of3_finetune.cli.__path__)]
        commands = {c for module in modules for c in vars(module).values()
                    if isinstance(c, click.Command) and c.name != "prepare-inputs"}
        with_output_dir = {c.name: next(p for p in c.params if p.name == "output_dir")
                           for c in commands if any(p.name == "output_dir" for p in c.params)}
        assert "finetune-pipeline" in with_output_dir
        assert {name: (o.default, o.required) for name, o in with_output_dir.items()} == {
            name: (DEFAULT_OUTPUT_DIR, False) for name in with_output_dir
        }
        assert DEFAULT_OUTPUT_DIR == Path("output")
