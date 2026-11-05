"""finetune-prep writes the cleanup step log, and step CIFs with --write-step-cifs."""

import json
import logging
from pathlib import Path

import pytest
from biotite.structure.io.pdbx import CIFCategory, CIFFile
from click.testing import CliRunner

from assertions import write_emptiable_minimal_3aa
from aw_of3_finetune.cli import finetune_prep
from aw_of3_finetune.pipeline import prepare_augmented_cif
from aw_of3_finetune.steplog import StepLog

FIXTURES = Path(__file__).parent / "fixtures" / "mmcifs"


def _run_prep(tmp_path: Path, ccd_path: Path, *extra: str) -> Path:
    """Run finetune-prep on 7riv; return the output dir."""
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir(exist_ok=True)
    link = cif_dir / "7riv.cif"
    if not link.exists():
        link.symlink_to((FIXTURES / "7riv.cif").resolve())
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path), *extra],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return output_dir


def _without_write_time(cif_path: Path) -> list[str]:
    """CIF lines minus the write timestamp, which differs between two writes."""
    return [line for line in cif_path.read_text().splitlines() if not line.startswith("_entry.time")]


def _load_log(output_dir: Path) -> StepLog:
    return StepLog.from_json((output_dir / "cleanup_steps" / "7riv.json").read_text())


def test_step_log_written_by_default(tmp_path: Path, ccd_path: Path):
    output_dir = _run_prep(tmp_path, ccd_path)

    actual_log = _load_log(output_dir)

    assert actual_log.unattributed_input() == []
    assert not (output_dir / "cleanup_steps" / "7riv").exists()


def test_write_step_cifs_writes_one_cif_per_step(tmp_path: Path, ccd_path: Path):
    output_dir = _run_prep(tmp_path, ccd_path, "--write-step-cifs")

    log = _load_log(output_dir)
    expected_names = [f"{i:02d}_{s.name}.cif" for i, s in enumerate(log.steps)]
    actual_names = sorted(p.name for p in (output_dir / "cleanup_steps" / "7riv").glob("*.cif"))

    assert actual_names == expected_names


def test_rerun_removes_stale_step_cifs(tmp_path: Path, ccd_path: Path):
    stale = tmp_path / "out" / "cleanup_steps" / "7riv" / "99_stale.cif"
    stale.parent.mkdir(parents=True)
    stale.write_text("data_stale\n")

    _run_prep(tmp_path, ccd_path, "--write-step-cifs")

    assert not stale.exists()


@pytest.mark.parametrize("extra", [[], ["--write-step-cifs"]], ids=["no_cifs", "step_cifs"])
def test_emptied_structure_is_skipped_and_log_written(tmp_path: Path, extra: list[str]):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    write_emptiable_minimal_3aa(cif_dir)
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir), *extra],
        catch_exceptions=False,
    )

    actual_log = StepLog.from_json((output_dir / "cleanup_steps" / "minimal_3aa.json").read_text())
    actual_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert result.exit_code == 0, result.output
    assert actual_manifest["structures"] == []
    assert actual_log.steps[-1].name.startswith("remove_")
    assert actual_log.steps[-1].resolved_atoms == 0


def test_unreadable_input_table_still_prepares(tmp_path: Path, ccd_path: Path, caplog):
    """A CIF the step-log reader rejects (no label_alt_id) prepares as without the log."""
    cif = CIFFile.read(FIXTURES / "1UBQ.cif")
    site = cif.block["atom_site"]
    cif.block["atom_site"] = CIFCategory({k: site[k] for k in site.keys() if k != "label_alt_id"})
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    cif_path = cif_dir / "1ubq.cif"
    cif.write(cif_path)
    output_dir = tmp_path / "out"
    expected_cif = tmp_path / "expected.cif"
    prepare_augmented_cif(cif_path, "1ubq", expected_cif, ccd=CIFFile.read(ccd_path),
                          ref_mol_dir=tmp_path / "expected_mols")

    with caplog.at_level(logging.WARNING):
        result = CliRunner().invoke(
            finetune_prep,
            ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir), "--ccd-path", str(ccd_path)],
            catch_exceptions=False,
        )

    actual_cif = output_dir / "structures" / "1ubq.cif"
    assert result.exit_code == 0, result.output
    assert _without_write_time(actual_cif) == _without_write_time(expected_cif)
    assert "step log" in caplog.text
    assert not (output_dir / "cleanup_steps" / "1ubq.json").exists()


def test_failing_structure_is_dropped_and_the_rest_prepared(tmp_path: Path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "7riv.cif").symlink_to((FIXTURES / "7riv.cif").resolve())
    (cif_dir / "bad.cif").write_text("data_bad\n_entry.id bad\n")
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(finetune_prep, ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir)])

    assert result.exit_code == 0, result.output
    actual_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert actual_manifest["structures"] == ["7riv"]
    [actual_dropped] = json.loads((output_dir / "prep_dropped.json").read_text())
    assert actual_dropped["id"] == "bad"
    assert actual_dropped["reason"].startswith("failed: ")
    assert not (output_dir / "structures" / "bad.cif").exists()


def test_emptied_structure_is_recorded_as_dropped(tmp_path: Path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    write_emptiable_minimal_3aa(cif_dir)
    output_dir = tmp_path / "out"

    CliRunner().invoke(finetune_prep, ["--cif-dir", str(cif_dir), "--output-dir", str(output_dir)],
                       catch_exceptions=False)

    [actual_dropped] = json.loads((output_dir / "prep_dropped.json").read_text())
    assert actual_dropped["id"] == "minimal_3aa"
    assert actual_dropped["reason"].startswith("empty after cleanup (remove_")
