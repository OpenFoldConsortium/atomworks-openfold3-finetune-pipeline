"""Assemble a run's report folder: a page with its summary data inline and per-entry scripts, so it opens from disk."""

import json
import shutil
from pathlib import Path

from aw_of3_finetune.report.bundle import bundle_modules

ASSETS_DIR = Path(__file__).parent / "assets"
REPORT_SUBDIR = "report"
REPORT_MARKER = ".aw_of3_report"  # marks a folder build_report wrote, so it may be replaced
DATA_FILES = ["parity_data.json", "validate_data.json", "run.json", "parity.json", "validate.json", "parity_load.json"]
TEMPLATE_SCRIPTS = '<script src="vendor/3Dmol-min.js"></script>\n<script type="module" src="js/main.js"></script>'
PAGE_ASSETS = ["report.css", "vendor"]
BUILD_CODE = [Path(__file__), Path(__file__).with_name("bundle.py")]
RUN_KEYS = ["git_sha", "atomworks_version", "openfold3_commit"]
SECTIONS = {"parity": "parity_data.json", "validate": "validate_data.json", "load": "parity_load.json"}
# What the summary grids read of each entry; the rest is in data/entries/<id>.js, loaded when the entry is opened.
SUMMARY_KEYS = {"parity": ("grid", "error", "status", "detail"), "validate": ("grid", "error", "status", "detail"),
                "load": ("groups",)}


def _source_files(run_dir: Path) -> list[Path]:
    """The run-dir data files that go into the report."""
    files = [run_dir / name for name in DATA_FILES if (run_dir / name).exists()]
    files += sorted((run_dir / "validate_3d").glob("*.json"))
    return files


def _inline_json(content: dict) -> str:
    """JSON safe to place inside an inline <script>: it cannot close the tag or open a comment."""
    return json.dumps(content).replace("</", "<\\/").replace("<!--", "<\\!--")


def _index_html(report_data: dict) -> str:
    """The dev template with its module script replaced by inline data, 3Dmol and the bundle."""
    template = (ASSETS_DIR / "index.html").read_text()
    if TEMPLATE_SCRIPTS not in template:
        raise ValueError(f"{ASSETS_DIR / 'index.html'} lacks the script tags the build replaces: {TEMPLATE_SCRIPTS!r}")
    scripts = (
        f"<script>window.REPORT_DATA = {_inline_json(report_data)};</script>\n"
        '<script src="vendor/3Dmol-min.js"></script>\n'
        '<script src="js/report.js"></script>'
    )
    return template.replace(TEMPLATE_SCRIPTS, scripts)


def _write_3d_scripts(run_dir: Path, data_dir: Path) -> None:
    """Write each validate_3d/<id>.json as a script that hands its data to the page."""
    out_dir = data_dir / "validate_3d"
    out_dir.mkdir(parents=True)
    for source in sorted((run_dir / "validate_3d").glob("*.json")):
        call = f"window.__reportStep3d({json.dumps(source.stem)}, {source.read_text().strip()});\n"
        (out_dir / f"{source.stem}.js").write_text(call)


def _split_entries(report_data: dict) -> dict[str, dict]:
    """Cut each section's entries down to SUMMARY_KEYS in place; returns {id: {section: full entry}}."""
    payloads: dict[str, dict] = {}
    for name, keys in SUMMARY_KEYS.items():
        entries = report_data.get(name, {}).get("entries")
        if not isinstance(entries, dict):
            continue
        for pdb_id, entry in entries.items():
            payloads.setdefault(pdb_id, {})[name] = entry
            entries[pdb_id] = {key: entry[key] for key in keys if key in entry}
    return payloads


def _write_entry_scripts(payloads: dict[str, dict], data_dir: Path) -> None:
    """Write each entry's data as data/entries/<id>.js, a script that hands it to the page."""
    out_dir = data_dir / "entries"
    out_dir.mkdir(parents=True)
    for pdb_id, payload in sorted(payloads.items()):
        call = f"window.__reportEntry({json.dumps(pdb_id)}, {json.dumps(payload, separators=(',', ':'))});\n"
        (out_dir / f"{pdb_id}.js").write_text(call)


def _entry_ids(run_dir: Path) -> list[str]:
    ids = set()
    for name, file in SECTIONS.items():
        if _has_section(run_dir, name):
            ids.update(json.loads((run_dir / file).read_text()).get("entries", {}))
    return sorted(ids)


def _has_section(run_dir: Path, name: str) -> bool:
    """Whether the section's data file exists; load also needs parity_data.json to say it is available."""
    if not (run_dir / SECTIONS[name]).exists():
        return False
    if name != "load":
        return True
    parity_data = run_dir / SECTIONS["parity"]
    return parity_data.exists() and json.loads(parity_data.read_text()).get("load", {}).get("available") is True


def _manifest(run_dir: Path, title: str) -> dict:
    run_json = run_dir / "run.json"
    run = None
    if run_json.exists():
        content = json.loads(run_json.read_text())
        run = {key: content.get(key) for key in RUN_KEYS}
    sections = [name for name in SECTIONS if _has_section(run_dir, name)]
    manifest = {
        "title": title,
        "sections": sections,
        "entries": _entry_ids(run_dir),
        "has_3d": sorted(path.stem for path in (run_dir / "validate_3d").glob("*.json")),
        "run": run,
    }
    if sections == ["validate"]:
        manifest["summary"] = "user"
    return manifest


def build_report(run_dir: Path, title: str, report_dir: Path | None = None) -> Path:
    """Rebuild the report folder: summary data inline in index.html, each entry's data in data/entries/<id>.js.

    Args:
        run_dir: The run directory holding the data files.
        title: Title shown in the report header.
        report_dir: Where to write the report; defaults to <run_dir>/report.

    Returns:
        The report folder.
    """
    report_dir = report_dir or run_dir / REPORT_SUBDIR
    shutil.rmtree(report_dir, ignore_errors=True)
    report_dir.mkdir(parents=True)
    (report_dir / REPORT_MARKER).write_text("written by aw_of3_finetune's report build; replaced on rebuild\n")
    for name in PAGE_ASSETS:
        source = ASSETS_DIR / name
        if source.is_dir():
            shutil.copytree(source, report_dir / name)
        else:
            shutil.copy2(source, report_dir / name)
    (report_dir / "js").mkdir()
    (report_dir / "js" / "report.js").write_text(bundle_modules(ASSETS_DIR / "js"))
    data_dir = report_dir / "data"
    _write_3d_scripts(run_dir, data_dir)
    manifest = _manifest(run_dir, title)
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report_data = {"manifest": manifest}
    for name in manifest["sections"]:
        report_data[name] = json.loads((run_dir / SECTIONS[name]).read_text())
    _write_entry_scripts(_split_entries(report_data), data_dir)
    (report_dir / "index.html").write_text(_index_html(report_data))
    return report_dir


def report_is_stale(run_dir: Path) -> bool:
    """Whether report/ is missing or older than any source data file, page asset or build code.

    Args:
        run_dir: The run directory.

    Returns:
        True when the report needs rebuilding.
    """
    index = run_dir / REPORT_SUBDIR / "index.html"
    if not index.exists():
        return True
    built_at = index.stat().st_mtime
    assets = [path for path in ASSETS_DIR.rglob("*") if path.is_file()]
    return any(source.stat().st_mtime > built_at for source in [*_source_files(run_dir), *assets, *BUILD_CODE])
