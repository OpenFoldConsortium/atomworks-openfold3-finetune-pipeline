"""Shared test fixtures for the atomworks-OF3 finetune pipeline tests."""

import os
from pathlib import Path

import pytest
from openfold3.entry_points.parameters import DEFAULT_CHECKPOINT_NAME, OPENFOLD_MODEL_CHECKPOINT_REGISTRY

from aw_of3_finetune.openfold_cache import resolve_ccd_path
from environ import restored_environ
from validate_helpers import build_dataset

# CIFs live in the pipeline repo so tests don't depend on the openfold-3 clone.
FIXTURES_MMCIFS = Path(__file__).parent / "fixtures" / "mmcifs"

CIF_MINIMAL = (
    FIXTURES_MMCIFS / "minimal_3aa.cif"
)  # synthetic 3-residue peptide (ALA-GLY-ALA)
CIF_PROTEIN_MONOMER = FIXTURES_MMCIFS / "3u8v.cif"  # protein homodimer (chains A, B share entity 1)
CIF_PROTEIN_ONLY = FIXTURES_MMCIFS / "2crb.cif"  # protein-only, no bound ligand in the CIF itself
CIF_DNA_COMPLEX = FIXTURES_MMCIFS / "2q2k.cif"  # protein + DNA + ligand
CIF_PXR = FIXTURES_MMCIFS / "7riv.cif"  # PXR LBD + drug-like ligand P06 + HOH
# Two distinct ubiquitin crystal structures — identical chain A canonical sequences.
# Used to verify cross-structure MSA sharing (same rep_id → same alignments directory).
CIF_UBIQUITIN_1 = FIXTURES_MMCIFS / "1UBQ.cif"
CIF_UBIQUITIN_2 = FIXTURES_MMCIFS / "1UBI.cif"


@pytest.fixture(scope="session")
def cif_minimal() -> Path:
    """Path to the synthetic 3-residue peptide (ALA-GLY-ALA) fixture CIF."""
    assert CIF_MINIMAL.exists(), f"Test CIF missing: {CIF_MINIMAL}"
    return CIF_MINIMAL


@pytest.fixture(scope="session")
def cif_protein_monomer() -> Path:
    """Path to the 3u8v (protein + NI ions + water) fixture CIF."""
    assert CIF_PROTEIN_MONOMER.exists(), f"Test CIF missing: {CIF_PROTEIN_MONOMER}"
    return CIF_PROTEIN_MONOMER


@pytest.fixture(scope="session")
def cif_protein_only() -> Path:
    """Path to the 2crb fixture CIF: NMR protein, no bound ligand.

    Some round-trip tests overlay synthetic ligand metadata on it; that data is
    built by the test, not read from this CIF.
    """
    assert CIF_PROTEIN_ONLY.exists(), f"Test CIF missing: {CIF_PROTEIN_ONLY}"
    return CIF_PROTEIN_ONLY


@pytest.fixture(scope="session")
def cif_dna_complex() -> Path:
    """Path to the 2q2k (protein + DNA + HEPES ligand) fixture CIF."""
    assert CIF_DNA_COMPLEX.exists(), f"Test CIF missing: {CIF_DNA_COMPLEX}"
    return CIF_DNA_COMPLEX


@pytest.fixture(scope="session")
def cif_ubiquitin_1() -> Path:
    """Path to the 1UBQ (ubiquitin, chain A) fixture CIF."""
    assert CIF_UBIQUITIN_1.exists(), f"Test CIF missing: {CIF_UBIQUITIN_1}"
    return CIF_UBIQUITIN_1


@pytest.fixture(scope="session")
def cif_ubiquitin_2() -> Path:
    """Path to the 1UBI (ubiquitin, same chain A sequence as 1UBQ) fixture CIF."""
    assert CIF_UBIQUITIN_2.exists(), f"Test CIF missing: {CIF_UBIQUITIN_2}"
    return CIF_UBIQUITIN_2


@pytest.fixture(scope="session")
def cif_pxr() -> Path:
    """Path to the 7riv (PXR LBD + ligand P06 + HOH) fixture CIF."""
    assert CIF_PXR.exists(), f"Test CIF missing: {CIF_PXR}"
    return CIF_PXR


DEFAULT_WEIGHTS_FILE_NAME = OPENFOLD_MODEL_CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT_NAME].file_name

# Resolved once at import, before isolate_ccd_settings hides $CCD_PATH and the stored path.
_CCD_PATH, _CCD_SOURCE = resolve_ccd_path(None)


@pytest.fixture(autouse=True)
def _restore_environ():
    """Undo any os.environ change a test makes, including ones by code under test.

    monkeypatch already undoes its own setenv/delenv; this also covers direct writes
    to os.environ (as openfold3's setup_openfold makes to OPENFOLD_CACHE), which would
    otherwise reach later tests -- under pytest -n only on whichever worker ran the test.
    """
    with restored_environ():
        yield


@pytest.fixture(autouse=True)
def isolate_ccd_settings(_restore_environ, tmp_path_factory):
    """Hide the user's $CCD_PATH and stored CCD path, so commands see only what a test passes.

    The fresh $OPENFOLD_CACHE holds an empty file named like OpenFold3's default weights,
    which generate-train-config needs, so tests never read ~/.openfold3 or the real file.
    Changes only this test's copy of os.environ: _restore_environ puts both variables
    back afterwards.
    """
    os.environ.pop("CCD_PATH", None)
    cache = tmp_path_factory.mktemp("openfold_cache")
    (cache / DEFAULT_WEIGHTS_FILE_NAME).touch()
    os.environ["OPENFOLD_CACHE"] = str(cache)


@pytest.fixture
def fake_pretrained_weights(isolate_ccd_settings) -> Path:
    """The empty stand-in for OpenFold3's default weights in this test's $OPENFOLD_CACHE."""
    return Path(os.environ["OPENFOLD_CACHE"]) / DEFAULT_WEIGHTS_FILE_NAME


@pytest.fixture(autouse=True)
def no_colabfold_outside_integration(request, monkeypatch):
    """Fail any non-integration test that would query the public ColabFold server.

    finetune-pipeline searches for MSAs by default, so a test that leaves out
    --no-msa-search would otherwise go to the network. Tests can still monkeypatch
    make_msas_colabfold_server_batch themselves.
    """
    if request.node.get_closest_marker("integration") is not None:
        return

    def _fail(*args, **kwargs):
        pytest.fail("ColabFold must not be called outside integration tests (pass --no-msa-search)")

    monkeypatch.setattr("aw_of3_finetune.cli.msas.make_msas_colabfold_server_batch", _fail)


@pytest.fixture(scope="session")
def ccd_path() -> Path:
    """The CCD from $CCD_PATH or setup-finetuning's stored path; skips the test if neither is set."""
    if _CCD_PATH is None:
        pytest.skip("No CCD set up (run `pixi run setup-finetuning`, or set CCD_PATH)")
    if not _CCD_PATH.is_file():
        pytest.skip(f"CCD file not found: {_CCD_PATH} (from {_CCD_SOURCE})")
    return _CCD_PATH


@pytest.fixture(
    params=[
        pytest.param("cif_protein_monomer", id="3u8v"),
        pytest.param("cif_protein_only", id="2crb"),
        pytest.param("cif_pxr", id="7riv"),
        # Symmetry-expanded protein/DNA complex whose HEPES ligand atomworks strips;
        # regression case for deriving chain metadata from atomworks' atoms.
        pytest.param("cif_dna_complex", id="2q2k"),
    ]
)
def cif_for_integration(
    request,
    cif_protein_monomer: Path,
    cif_protein_only: Path,
    cif_pxr: Path,
    cif_dna_complex: Path,
) -> Path:
    """Parametrized fixture that cycles through structures for end-to-end tests."""
    fixtures = {
        "cif_protein_monomer": cif_protein_monomer,
        "cif_protein_only": cif_protein_only,
        "cif_pxr": cif_pxr,
        "cif_dna_complex": cif_dna_complex,
    }
    return fixtures[request.param]


@pytest.fixture(scope="session")
def built_dataset(ccd_path: Path, tmp_path_factory):
    """Pipeline output over 1a3n, 2crb, 2q2k, 3u8v, 7riv with query-only MSA stubs."""
    cifs = [FIXTURES_MMCIFS / f"{name}.cif" for name in ("1a3n", "2crb", "2q2k", "3u8v", "7riv")]
    return build_dataset(tmp_path_factory.mktemp("built_dataset"), cifs, ccd_path)


@pytest.fixture(scope="session")
def built_dataset_with_waters(ccd_path: Path, tmp_path_factory):
    """Like built_dataset, over 1a3n and 3u8v, with cleanup's water removal switched off."""
    cifs = [FIXTURES_MMCIFS / f"{name}.cif" for name in ("1a3n", "3u8v")]
    return build_dataset(
        tmp_path_factory.mktemp("built_dataset_with_waters"), cifs, ccd_path,
        cleanup_config="remove_waters: false\n",
    )
