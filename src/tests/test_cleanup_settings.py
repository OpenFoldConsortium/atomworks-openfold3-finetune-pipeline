"""CleanupSettings: per-step switches for structure cleanup, loaded from a YAML/JSON file.

Defaults reproduce OF3's cleanup_structure_of3; a user turns individual steps off.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from assertions import write_emptiable_minimal_3aa
from aw_of3_finetune.cleanup import CleanupSettings, clean_structure
from aw_of3_finetune.cli import finetune_pipeline, finetune_prep
from aw_of3_finetune.parsing import parse_structure

STEP_FIELDS = [
    "altloc",
    "remove_waters",
    "convert_mse_to_met",
    "remove_crystallization_aids",
    "remove_small_polymers",
    "remove_fully_unknown_polymers",
    "remove_clashing_chains",
    "remove_non_ccd_atoms",
    "remove_chains_with_ca_gaps",
    "prefilter_bonds",
]


def test_defaults_match_of3():
    """Every step is on by default (OF3's cleanup), with atomworks' first-letter altloc choice."""
    settings = CleanupSettings()
    actual_settings = settings.model_dump()

    # The settings expose exactly the nine cleanup steps.
    assert set(actual_settings) == set(STEP_FIELDS)

    # Every step is switched on.
    assert all(actual_settings.values())

    # The class itself reports these as its defaults.
    assert settings.is_default()


def test_from_file_overrides_listed_steps_only(tmp_path):
    """A YAML file turns off only the steps it lists; every other step stays on."""
    path = tmp_path / "cleanup.yml"
    path.write_text("remove_clashing_chains: false\n")

    settings = CleanupSettings.from_file(path)

    # The listed step is off.
    assert settings.remove_clashing_chains is False

    # All other steps keep their default (on).
    actual_other_steps = {k: v for k, v in settings.model_dump().items() if k != "remove_clashing_chains"}
    assert all(actual_other_steps.values())

    # The settings no longer count as the defaults.
    assert not settings.is_default()


def test_from_file_reads_json(tmp_path):
    """A JSON file is accepted as well as YAML."""
    path = tmp_path / "cleanup.json"
    path.write_text(json.dumps({"remove_waters": False}))

    settings = CleanupSettings.from_file(path)

    # The listed step is off.
    assert settings.remove_waters is False


def test_unknown_key_is_an_error(tmp_path):
    """A misspelled step must not be silently ignored."""
    path = tmp_path / "cleanup.yml"
    path.write_text("remove_clashing_chain: false\n")

    # The validation error names the misspelled key.
    with pytest.raises(ValidationError, match="remove_clashing_chain"):
        CleanupSettings.from_file(path)


def test_example_file_lists_every_step_at_its_default():
    """examples/cleanup_settings.yml documents all steps and loads to the defaults."""
    example = Path(__file__).parents[2] / "examples" / "cleanup_settings.yml"
    example_text = example.read_text()

    settings = CleanupSettings.from_file(example)

    # Loading the example gives the default settings.
    assert settings.is_default()

    # The file mentions every step by name.
    steps_missing_from_file = [field for field in STEP_FIELDS if f"{field}:" not in example_text]
    assert steps_missing_from_file == []


def test_waters_kept_when_switched_off(cif_pxr):
    """remove_waters=False keeps the waters the parse leaves for cleanup."""
    arr = parse_structure(cif_pxr)["assemblies"]["1"][0]

    kept, _ = clean_structure(arr, cif_pxr, "7riv", None, CleanupSettings(remove_waters=False))
    default, _ = clean_structure(arr, cif_pxr, "7riv", None)

    # With the step off, water (HOH) remains.
    assert (kept.res_name == "HOH").any()

    # With default settings, no water remains.
    assert not (default.res_name == "HOH").any()


def test_mse_kept_when_switched_off():
    """convert_mse_to_met=False leaves selenomethionine (MSE) as MSE."""
    src = Path(__file__).parent / "fixtures" / "mmcifs" / "4zey.cif"
    arr = parse_structure(src)["assemblies"]["1"][0]

    kept, _ = clean_structure(arr, src, "4zey", None, CleanupSettings(convert_mse_to_met=False))

    # MSE residues are still present.
    assert (kept.res_name == "MSE").any()


def test_switched_off_filter_is_not_applied(tmp_path):
    """With remove_small_polymers off, a 3-residue peptide survives cleanup."""
    src = write_emptiable_minimal_3aa(tmp_path)
    arr = parse_structure(src)["assemblies"]["1"][0]

    cleaned, removals = clean_structure(
        arr, src, "minimal_3aa", None, CleanupSettings(remove_small_polymers=False)
    )

    # The peptide is still in the structure.
    assert len(cleaned) > 0

    # No small-polymer removal was recorded.
    actual_steps = [removal.step for removal in removals]
    assert "remove_small_polymers" not in actual_steps


def test_crystallization_aids_kept_when_switched_off(cif_dna_complex):
    """2q2k is X-ray with HEPES (EPE); switching aid removal off keeps it."""
    arr = parse_structure(cif_dna_complex)["assemblies"]["1"][0]

    cleaned, _ = clean_structure(
        arr, cif_dna_complex, "2q2k", None, CleanupSettings(remove_crystallization_aids=False)
    )

    # HEPES is still in the cleaned structure.
    assert np.isin("EPE", cleaned.res_name)


class TestCli:
    def test_finetune_prep_records_default_settings(self, cif_protein_monomer, tmp_path):
        """With no --cleanup-config, prep records the default settings and prints no warning."""
        cif_dir = tmp_path / "in"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)

        result = CliRunner().invoke(
            finetune_prep, ["--cif-dir", str(cif_dir), "--output-dir", str(tmp_path / "out")]
        )

        assert result.exit_code == 0, result.output

        # The settings written next to the output are the defaults.
        actual_recorded = json.loads((tmp_path / "out" / "cleanup_settings.json").read_text())
        assert actual_recorded == CleanupSettings().model_dump()

        # Default settings trigger no "differs from OpenFold3" warning.
        assert "differ from the defaults" not in result.output

    def test_finetune_prep_applies_config_and_warns(self, cif_pxr, tmp_path):
        """--cleanup-config turns off a step, records it, warns, and the output keeps waters."""
        cif_dir = tmp_path / "in"
        cif_dir.mkdir()
        (cif_dir / "7riv.cif").symlink_to(cif_pxr)
        config = tmp_path / "cleanup.yml"
        config.write_text("remove_waters: false\n")

        result = CliRunner().invoke(
            finetune_prep,
            [
                "--cif-dir", str(cif_dir),
                "--output-dir", str(tmp_path / "out"),
                "--cleanup-config", str(config),
            ],
        )

        assert result.exit_code == 0, result.output

        # The recorded settings show the step switched off.
        actual_recorded = json.loads((tmp_path / "out" / "cleanup_settings.json").read_text())
        assert actual_recorded["remove_waters"] is False

        # The CLI warns about the deviation from OpenFold3.
        assert "Cleanup settings differ from the defaults: remove_waters=False" in result.output

        # The cleaned CIF still contains water (HOH).
        actual_cleaned_cif = (tmp_path / "out" / "cleaned_structures" / "7riv.cif").read_text()
        assert "HOH" in actual_cleaned_cif

    def test_bad_config_is_a_usage_error(self, cif_protein_monomer, tmp_path):
        """A misspelled step in --cleanup-config makes the CLI exit with a usage error naming the key."""
        cif_dir = tmp_path / "in"
        cif_dir.mkdir()
        (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)
        config = tmp_path / "cleanup.yml"
        config.write_text("remove_wters: false\n")

        result = CliRunner().invoke(
            finetune_prep,
            [
                "--cif-dir", str(cif_dir),
                "--output-dir", str(tmp_path / "out"),
                "--cleanup-config", str(config),
            ],
        )

        # Click's usage-error exit code.
        assert result.exit_code == 2

        # The output names the misspelled key.
        assert "remove_wters" in result.output

    def test_finetune_prep_prints_cleanup_steps(self, cif_pxr, tmp_path):
        """Every cleanup step that changes the structure shows up in the CLI output."""
        cif_dir = tmp_path / "in"
        cif_dir.mkdir()
        (cif_dir / "7riv.cif").symlink_to(cif_pxr)

        result = CliRunner().invoke(
            finetune_prep, ["--cif-dir", str(cif_dir), "--output-dir", str(tmp_path / "out")]
        )

        assert result.exit_code == 0, result.output

        # The water removal and the summary line are printed.
        assert "7riv: remove_waters removed" in result.output
        assert "7riv: cleanup done" in result.output

        # The cleaned-structures directory is printed on the line right after the manifest.
        expected_structures_line = f"Cleaned structures: {tmp_path / 'out' / 'cleaned_structures'} (1 augmented CIF(s))"
        lines = result.output.splitlines()
        manifest_index = next(i for i, line in enumerate(lines) if line.startswith("Manifest:"))
        assert lines[manifest_index + 1] == expected_structures_line

    def test_finetune_pipeline_has_cleanup_config(self):
        """The full pipeline command also exposes --cleanup-config."""
        result = CliRunner().invoke(finetune_pipeline, ["--help"])

        # The option appears in the help text.
        assert "--cleanup-config" in result.output


def test_altloc_choice_is_validated(tmp_path):
    """altloc takes clash_aware, highest_occupancy or first; anything else is an error."""
    path = tmp_path / "cleanup.yml"
    path.write_text("altloc: occupancy\n")

    with pytest.raises(ValidationError, match="altloc"):
        CleanupSettings.from_file(path)


def test_highest_occupancy_is_warned_about():
    """highest_occupancy (OF3's choice) is allowed but not recommended: the settings say why."""
    actual_warning = CleanupSettings(altloc="highest_occupancy").altloc_warning()

    assert actual_warning is not None
    assert actual_warning.startswith("altloc: highest_occupancy is not the recommended choice")


@pytest.mark.parametrize("choice", ["first", "clash_aware"])
def test_first_and_clash_aware_have_no_warning(choice):
    """first (the default) and clash_aware are both recommended choices."""
    assert CleanupSettings(altloc=choice).altloc_warning() is None


def test_finetune_prep_warns_about_highest_occupancy(cif_protein_monomer, tmp_path):
    """finetune-prep prints the altloc warning once when the config picks highest_occupancy."""
    cif_dir = tmp_path / "in"
    cif_dir.mkdir()
    (cif_dir / "3u8v.cif").symlink_to(cif_protein_monomer)
    config = tmp_path / "cleanup.yml"
    config.write_text("altloc: highest_occupancy\n")

    result = CliRunner().invoke(
        finetune_prep,
        ["--cif-dir", str(cif_dir), "--output-dir", str(tmp_path / "out"), "--cleanup-config", str(config)],
    )

    assert result.exit_code == 0, result.output
    expected_count = 1
    actual_count = result.output.count("altloc: highest_occupancy is not the recommended choice")
    assert actual_count == expected_count
