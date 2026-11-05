"""Shared test fixtures for the atomworks-OF3 finetune pipeline tests."""

import os
from pathlib import Path

import pytest

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


@pytest.fixture(scope="session")
def ccd_path() -> Path:
    """The CCD file named by the CCD_PATH env var; skips the test if it is unset or missing."""
    ccd = os.environ.get("CCD_PATH")
    if not ccd:
        pytest.skip("CCD_PATH is not set (see 'Get the CCD' in the README)")
    path = Path(ccd).expanduser()
    if not path.is_file():
        pytest.skip(f"CCD_PATH file not found: {path}")
    return path


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
