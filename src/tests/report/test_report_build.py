import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

import aw_of3_finetune.report.build
from aw_of3_finetune.report.build import build_report, report_is_stale


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


@pytest.fixture
def run_dir(tmp_path):
    """A run dir with parity and validate data, one 3D file and a run.json."""
    directory = tmp_path / "run"
    _write_json(directory / "parity_data.json", {"entries": {"7axf": {}, "1abc": {}}})
    _write_json(directory / "validate_data.json", {"entries": {"7axf": {}, "2xyz": {}}})
    _write_json(directory / "validate_3d" / "7axf.json", {"steps": []})
    run_json = {"git_sha": "abc123", "atomworks_version": "3.0", "openfold3_commit": "def456"}
    _write_json(directory / "run.json", run_json)
    return directory


def _report_data(index_html: str) -> dict:
    """The REPORT_DATA object inlined in a built index.html."""
    start = index_html.index("window.REPORT_DATA = ") + len("window.REPORT_DATA = ")
    end = index_html.index(";</script>", start)
    return json.loads(index_html[start:end].replace("<\\/", "</").replace("<\\!--", "<!--"))


def test_build_writes_the_page_bundle_and_3d_scripts(run_dir):
    """The report folder holds index.html, the bundle, the 3D script and a manifest."""
    report_dir = build_report(run_dir, "my title")

    expected_dir = run_dir / "report"
    assert report_dir == expected_dir
    expected_files = ["index.html", "report.css", "vendor/3Dmol-min.js", "js/report.js", "data/validate_3d/7axf.js", "data/manifest.json"]
    actual_missing = [name for name in expected_files if not (report_dir / name).exists()]
    assert actual_missing == []
    actual_manifest = json.loads((report_dir / "data" / "manifest.json").read_text())
    expected_manifest = {
        "title": "my title",
        "sections": ["parity", "validate"],
        "entries": ["1abc", "2xyz", "7axf"],
        "has_3d": ["7axf"],
        "run": {"git_sha": "abc123", "atomworks_version": "3.0", "openfold3_commit": "def456"},
    }
    assert actual_manifest == expected_manifest


def test_built_page_inlines_manifest_and_sections(run_dir):
    """REPORT_DATA in index.html carries the manifest and each present section's data."""
    report_dir = build_report(run_dir, "my title")

    actual_data = _report_data((report_dir / "index.html").read_text())
    expected_keys = ["manifest", "parity", "validate"]
    assert sorted(actual_data) == expected_keys
    assert actual_data["manifest"]["title"] == "my title"
    assert actual_data["validate"] == {"entries": {"7axf": {}, "2xyz": {}}}


def test_built_page_has_no_module_script(run_dir):
    """The built page loads classic scripts only and carries its data inline."""
    report_dir = build_report(run_dir, "t")

    index_html = (report_dir / "index.html").read_text()
    assert 'type="module"' not in index_html
    assert "window.REPORT_DATA" in index_html


def test_inline_data_cannot_close_script(tmp_path):
    """A finding detail holding </script> cannot end the inline data script early."""
    detail = "</script><b>x"
    _write_json(tmp_path / "validate_data.json", {"entries": {"1abc": {"detail": detail}}})

    report_dir = build_report(tmp_path, "t")

    index_html = (report_dir / "index.html").read_text()
    expected_count = index_html.count("<script")
    actual_count = index_html.count("</script>")
    assert actual_count == expected_count
    actual_detail = _report_data(index_html)["validate"]["entries"]["1abc"]["detail"]
    assert actual_detail == detail


def test_inline_data_cannot_open_a_comment(tmp_path):
    """A finding detail holding <!-- is escaped in index.html and reads back unchanged."""
    detail = "odd <!-- name"
    run_findings = [{"check": "x", "status": "unexplained", "label": "unexplained", "detail": detail, "reason": None}]
    _write_json(tmp_path / "validate_data.json", {"entries": {"1abc": {}}, "run_findings": run_findings})

    report_dir = build_report(tmp_path, "t")

    index_html = (report_dir / "index.html").read_text()
    assert "<!--" not in index_html.split("window.REPORT_DATA", 1)[1]
    actual_detail = _report_data(index_html)["validate"]["run_findings"][0]["detail"]
    assert actual_detail == detail


def test_entry_details_are_not_inlined(tmp_path):
    """Per-entry data goes to data/entries/<id>.js; index.html keeps only each entry's grid."""
    entry = {"grid": {"bond_lengths": "same"}, "findings": [{"detail": "unique-detail-7f3a"}]}
    _write_json(tmp_path / "validate_data.json", {"entries": {"1abc": entry}})

    report_dir = build_report(tmp_path, "t")

    index_html = (report_dir / "index.html").read_text()
    assert "unique-detail-7f3a" not in index_html
    expected_stub = {"grid": {"bond_lengths": "same"}}
    actual_stub = _report_data(index_html)["validate"]["entries"]["1abc"]
    assert actual_stub == expected_stub


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_odd_ids_get_entry_scripts(tmp_path):
    """An entry my-struct_1 gets a script that hands its id and every section's entry data to the page."""
    _write_json(tmp_path / "parity_data.json", {"entries": {"my-struct_1": {"grid": {}, "tracks": "p"}}})
    _write_json(tmp_path / "validate_data.json", {"entries": {"my-struct_1": {"grid": {}, "tracks": "</script>"}}})

    report_dir = build_report(tmp_path, "t")

    script = report_dir / "data" / "entries" / "my-struct_1.js"
    actual_text = script.read_text()
    expected_start = 'window.__reportEntry("my-struct_1", '
    assert actual_text.startswith(expected_start)
    actual_payload = json.loads(actual_text.removeprefix(expected_start).removesuffix(");\n"))
    expected_payload = {"parity": {"grid": {}, "tracks": "p"}, "validate": {"grid": {}, "tracks": "</script>"}}
    assert actual_payload == expected_payload
    result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_load_entries_keep_groups_inline(tmp_path):
    """The load section's summary grid needs each entry's groups; its differences go to the entry script."""
    _write_json(tmp_path / "parity_data.json", {"entries": {"a": {"grid": {}}}, "load": {"available": True}})
    load = {"entries": {"a": {"groups": {"load.keys": "same"}, "differences": [{"detail": "diff-9c1e"}]}}}
    _write_json(tmp_path / "parity_load.json", load)

    report_dir = build_report(tmp_path, "t")

    index_html = (report_dir / "index.html").read_text()
    assert "diff-9c1e" not in index_html
    expected_stub = {"groups": {"load.keys": "same"}}
    actual_stub = _report_data(index_html)["load"]["entries"]["a"]
    assert actual_stub == expected_stub
    assert "diff-9c1e" in (report_dir / "data" / "entries" / "a.js").read_text()


def test_odd_ids_get_3d_scripts(tmp_path):
    """A 3D file for id my-struct_1 becomes a script that hands its id and data to the page."""
    _write_json(tmp_path / "validate_data.json", {"entries": {"my-struct_1": {}}})
    _write_json(tmp_path / "validate_3d" / "my-struct_1.json", {"steps": []})

    report_dir = build_report(tmp_path, "t")

    script = report_dir / "data" / "validate_3d" / "my-struct_1.js"
    assert script.exists()
    actual_text = script.read_text()
    expected_start = 'window.__reportStep3d("my-struct_1",'
    assert actual_text.startswith(expected_start)


def test_rebuild_removes_stale_3d(tmp_path):
    """Rebuilding after a 3D file is removed drops that entry's script."""
    _write_json(tmp_path / "validate_data.json", {"entries": {"a": {}, "b": {}}})
    for name in ["a", "b"]:
        _write_json(tmp_path / "validate_3d" / f"{name}.json", {"steps": []})
    build_report(tmp_path, "t")

    (tmp_path / "validate_3d" / "b.json").unlink()
    report_dir = build_report(tmp_path, "t")

    assert (report_dir / "data" / "validate_3d" / "a.js").exists()
    assert not (report_dir / "data" / "validate_3d" / "b.js").exists()


def test_build_into_another_report_dir(run_dir, tmp_path):
    """report_dir puts the report somewhere other than run_dir/report."""
    other = tmp_path / "elsewhere"

    actual_dir = build_report(run_dir, "t", report_dir=other)

    assert actual_dir == other
    assert (other / "index.html").exists()
    assert not (run_dir / "report").exists()


def test_manifest_lists_present_sections_only(tmp_path):
    """A run dir with only validate_data.json gives sections == ['validate'] and no run info."""
    _write_json(tmp_path / "validate_data.json", {"entries": {"1abc": {}}})

    build_report(tmp_path, "t")

    actual_manifest = json.loads((tmp_path / "report" / "data" / "manifest.json").read_text())
    assert actual_manifest["sections"] == ["validate"]
    assert actual_manifest["has_3d"] == []
    assert actual_manifest["run"] is None


def test_manifest_includes_load_when_parity_load_exists(tmp_path):
    """parity_load.json is copied into report/data and adds 'load' to the sections."""
    _write_json(tmp_path / "parity_data.json", {"entries": {"1abc": {}}, "load": {"available": True}})
    _write_json(tmp_path / "parity_load.json", {"entries": {"1abc": {}}})

    report_dir = build_report(tmp_path, "t")

    actual_manifest = json.loads((report_dir / "data" / "manifest.json").read_text())
    assert actual_manifest["sections"] == ["parity", "load"]
    actual_data = _report_data((report_dir / "index.html").read_text())
    assert actual_data["load"] == {"entries": {"1abc": {}}}


def test_manifest_has_no_load_when_parity_data_says_unavailable(tmp_path):
    """A leftover parity_load.json does not add 'load' when parity_data.json says load is unavailable."""
    _write_json(tmp_path / "parity_data.json", {"entries": {"1abc": {}}, "load": {"available": False}})
    _write_json(tmp_path / "parity_load.json", {"entries": {"1abc": {}}})

    report_dir = build_report(tmp_path, "t")

    actual_manifest = json.loads((report_dir / "data" / "manifest.json").read_text())
    expected_sections = ["parity"]
    assert actual_manifest["sections"] == expected_sections


def test_report_is_stale_when_parity_load_is_newer(tmp_path):
    """Rewriting parity_load.json after the build makes the report stale."""
    _write_json(tmp_path / "parity_load.json", {"entries": {}})
    build_report(tmp_path, "t")
    assert not report_is_stale(tmp_path)

    future = time.time() + 100
    os.utime(tmp_path / "parity_load.json", (future, future))

    assert report_is_stale(tmp_path)


def test_rebuild_removes_stale_files(run_dir):
    """A file left in the old report folder is gone after the next build."""
    report_dir = build_report(run_dir, "t")
    leftover = report_dir / "data" / "leftover.json"
    leftover.write_text("{}")

    build_report(run_dir, "t")

    assert not leftover.exists()


def test_report_is_stale_after_data_changes(run_dir):
    """Stale when the report is missing, fresh after a build, stale again when a source file is newer."""
    assert report_is_stale(run_dir)
    report_dir = build_report(run_dir, "t")
    assert not report_is_stale(run_dir)

    newer = time.time() + 10
    os.utime(run_dir / "parity_data.json", (newer, newer))

    assert report_is_stale(run_dir)
    assert report_dir.exists()


def test_report_is_stale_when_an_asset_is_newer(run_dir, tmp_path, monkeypatch):
    """A packaged page asset newer than the built report makes it stale, so serve never shows an old UI."""
    assets = tmp_path / "assets"
    shutil.copytree(aw_of3_finetune.report.build.ASSETS_DIR, assets)
    monkeypatch.setattr(aw_of3_finetune.report.build, "ASSETS_DIR", assets)
    build_report(run_dir, "t")
    assert not report_is_stale(run_dir)

    newer = time.time() + 10
    os.utime(assets / "js" / "tracks.js", (newer, newer))

    assert report_is_stale(run_dir)


def test_report_is_stale_when_a_3d_file_is_newer(run_dir):
    """A rewritten validate_3d file makes the report stale."""
    build_report(run_dir, "t")

    newer = time.time() + 10
    os.utime(run_dir / "validate_3d" / "7axf.json", (newer, newer))

    assert report_is_stale(run_dir)


def test_built_index_explains_how_to_build_it(run_dir):
    """index.html carries a static no-data hint naming finetune-report, shown if the page has no data."""
    report_dir = build_report(run_dir, "t")

    actual_index = (report_dir / "index.html").read_text()
    expected_hint = "No data loaded. Build this report with <code>finetune-report"
    assert expected_hint in actual_index


def test_validate_only_manifest_selects_user_summary(tmp_path):
    """Only a validate section gets the user summary; adding parity keeps the battle summary."""
    _write_json(tmp_path / "validate_data.json", {"entries": {"1abc": {}}})
    build_report(tmp_path, "t")
    actual_manifest = json.loads((tmp_path / "report" / "data" / "manifest.json").read_text())
    assert actual_manifest["summary"] == "user"

    _write_json(tmp_path / "parity_data.json", {"entries": {"1abc": {}}})
    build_report(tmp_path, "t")
    actual_manifest = json.loads((tmp_path / "report" / "data" / "manifest.json").read_text())
    assert "summary" not in actual_manifest
