"""Bit-exact comparison helpers shared across test phases.

Two public functions:
    assert_feature_dicts_equal  — compare two OF3 feature dicts (str → numpy arrays)
    assert_caches_equal         — compare two ClusteredDatasetCache instances
"""

from __future__ import annotations

from dataclasses import fields
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
)


# ── Feature-dict comparison ──────────────────────────────────────────────────


def assert_feature_dicts_equal(
    a: dict,
    b: dict,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    path: str = "",
) -> None:
    """Assert two OF3 feature dicts are bit-exact (integer/bool) or close (float).

    Args:
        a: First feature dict. Values may be numpy arrays, scalars, or nested dicts.
        b: Second feature dict, same value types as ``a``.
        rtol: Relative tolerance for floating-point arrays.
        atol: Absolute tolerance for floating-point arrays.
        path: Dot-separated key path shown in failure messages (populated by recursion).
    """
    _check_keys(a, b, path)
    for key in a:
        key_path = f"{path}.{key}" if path else key
        va, vb = a[key], b[key]
        _assert_values_equal(va, vb, rtol=rtol, atol=atol, path=key_path)


def _check_keys(a: dict, b: dict, path: str) -> None:
    """Raise if two feature dicts have different key sets.

    Args:
        a: First dict.
        b: Second dict.
        path: Key path of the dicts, for the error message.
    """
    missing = set(a) - set(b)
    extra = set(b) - set(a)
    messages = []
    if missing:
        messages.append(f"keys in a but not b: {sorted(missing)}")
    if extra:
        messages.append(f"keys in b but not a: {sorted(extra)}")
    if messages:
        loc = f" at '{path}'" if path else ""
        raise AssertionError(f"Feature dict key mismatch{loc}: {'; '.join(messages)}")


def _assert_values_equal(
    va: Any, vb: Any, *, rtol: float, atol: float, path: str
) -> None:
    """Compare one feature value: recurse into dicts, shape/dtype-check arrays.

    Args:
        va: Value from the first dict.
        vb: Value from the second dict.
        rtol: Relative tolerance for floating-point arrays.
        atol: Absolute tolerance for floating-point arrays.
        path: Key path of the value, for failure messages.
    """
    if isinstance(va, dict) and isinstance(vb, dict):
        assert_feature_dicts_equal(va, vb, rtol=rtol, atol=atol, path=path)
        return

    if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
        va = np.asarray(va)
        vb = np.asarray(vb)
        if va.shape != vb.shape:
            raise AssertionError(
                f"Shape mismatch at '{path}': {va.shape} vs {vb.shape}"
            )
        if va.dtype != vb.dtype:
            raise AssertionError(
                f"Dtype mismatch at '{path}': {va.dtype} vs {vb.dtype}"
            )
        if np.issubdtype(va.dtype, np.floating):
            np.testing.assert_allclose(
                va,
                vb,
                rtol=rtol,
                atol=atol,
                err_msg=f"Float array mismatch at '{path}'",
            )
        else:
            np.testing.assert_array_equal(va, vb, err_msg=f"Array mismatch at '{path}'")
        return

    assert va == vb, f"Value mismatch at '{path}': {va!r} != {vb!r}"


# ── Cache comparison ─────────────────────────────────────────────────────────


def assert_caches_equal(a: ClusteredDatasetCache, b: ClusteredDatasetCache) -> None:
    """Assert two ClusteredDatasetCache instances are structurally identical.

    Compares every field of every dataclass in the hierarchy:
      ClusteredDatasetCache
        ├── name
        ├── structure_data
        │   └── {pdb_id: ClusteredDatasetStructureData}
        │       ├── release_date, resolution
        │       ├── chains: {chain_id: ClusteredDatasetChainData}
        │       └── interfaces: {iface_id: ClusteredDatasetInterfaceData}
        └── reference_molecule_data
            └── {mol_id: DatasetReferenceMoleculeData}

    Args:
        a: First cache.
        b: Second cache.
    """
    assert type(a) is type(b), f"Cache type mismatch: {type(a)} vs {type(b)}"
    assert a.name == b.name, f"name mismatch: {a.name!r} vs {b.name!r}"
    _assert_structure_data_equal(a.structure_data, b.structure_data)
    _assert_ref_mol_data_equal(a.reference_molecule_data, b.reference_molecule_data)


def _assert_structure_data_equal(a: dict, b: dict) -> None:
    """Compare two ``structure_data`` dicts entry by entry.

    Args:
        a: First ``{pdb_id: ClusteredDatasetStructureData}`` dict.
        b: Second dict of the same shape.
    """
    _assert_same_keys(a, b, label="structure_data")
    for pdb_id in a:
        _assert_structure_entry_equal(pdb_id, a[pdb_id], b[pdb_id])


def _assert_structure_entry_equal(
    pdb_id: str,
    a: ClusteredDatasetStructureData,
    b: ClusteredDatasetStructureData,
) -> None:
    """Compare one structure entry: dates, resolution, chains, and interfaces.

    Args:
        pdb_id: Entry key, for failure messages.
        a: First structure entry.
        b: Second structure entry.
    """
    prefix = f"structure_data[{pdb_id!r}]"
    assert type(a) is type(b), f"{prefix}: type mismatch {type(a)} vs {type(b)}"
    _assert_field_equal(a.release_date, b.release_date, f"{prefix}.release_date")
    _assert_resolution_equal(a.resolution, b.resolution, f"{prefix}.resolution")
    _assert_same_keys(a.chains, b.chains, label=f"{prefix}.chains")
    for chain_id in a.chains:
        _assert_chain_data_equal(
            f"{prefix}.chains[{chain_id!r}]", a.chains[chain_id], b.chains[chain_id]
        )
    _assert_same_keys(a.interfaces, b.interfaces, label=f"{prefix}.interfaces")
    for iface_id in a.interfaces:
        _assert_interface_data_equal(
            f"{prefix}.interfaces[{iface_id!r}]",
            a.interfaces[iface_id],
            b.interfaces[iface_id],
        )


def _assert_chain_data_equal(
    path: str, a: ClusteredDatasetChainData, b: ClusteredDatasetChainData
) -> None:
    """Compare every field of two chain entries.

    Args:
        path: Location of the chain, for failure messages.
        a: First chain entry.
        b: Second chain entry.
    """
    assert type(a) is type(b), f"{path}: type mismatch"
    for field in fields(a):
        va, vb = getattr(a, field.name), getattr(b, field.name)
        _assert_field_equal(va, vb, f"{path}.{field.name}")


def _assert_interface_data_equal(
    path: str, a: ClusteredDatasetInterfaceData, b: ClusteredDatasetInterfaceData
) -> None:
    """Compare every field of two interface entries.

    Args:
        path: Location of the interface, for failure messages.
        a: First interface entry.
        b: Second interface entry.
    """
    assert type(a) is type(b), f"{path}: type mismatch"
    for field in fields(a):
        va, vb = getattr(a, field.name), getattr(b, field.name)
        _assert_field_equal(va, vb, f"{path}.{field.name}")


def _assert_ref_mol_data_equal(a: dict, b: dict) -> None:
    """Compare two ``reference_molecule_data`` dicts field by field.

    Args:
        a: First ``{mol_id: DatasetReferenceMoleculeData}`` dict.
        b: Second dict of the same shape.
    """
    _assert_same_keys(a, b, label="reference_molecule_data")
    for mol_id in a:
        ma, mb = a[mol_id], b[mol_id]
        path = f"reference_molecule_data[{mol_id!r}]"
        assert type(ma) is type(mb), f"{path}: type mismatch"
        for field in fields(ma):
            _assert_field_equal(
                getattr(ma, field.name), getattr(mb, field.name), f"{path}.{field.name}"
            )


def _assert_resolution_equal(a: float | None, b: float | None, path: str) -> None:
    """Compare two resolutions, treating None==None and NaN==NaN as equal.

    Args:
        a: First resolution.
        b: Second resolution.
        path: Location of the field, for failure messages.
    """
    if a is None and b is None:
        return
    if a is None or b is None:
        raise AssertionError(f"{path}: one is None, the other is {a or b!r}")
    if np.isnan(a) and np.isnan(b):
        return
    if np.isnan(a) or np.isnan(b):
        raise AssertionError(f"{path}: NaN mismatch: {a} vs {b}")
    assert a == b, f"{path}: {a} != {b}"


def _assert_field_equal(va: Any, vb: Any, path: str) -> None:
    """Shallow equality for scalar cache fields (str, int, float, date, list, None).

    A date and its ISO string compare equal: from_json leaves release_date a str.

    Args:
        va: First value.
        vb: Second value.
        path: Location of the field, for failure messages.
    """
    if isinstance(va, float) or isinstance(vb, float):
        _assert_resolution_equal(va, vb, path)
        return
    # Normalize date vs ISO-string so from_json and the CIF reader compare equal.
    if isinstance(va, date) and isinstance(vb, str):
        vb = date.fromisoformat(vb)
    elif isinstance(vb, date) and isinstance(va, str):
        va = date.fromisoformat(va)
    assert va == vb, f"{path}: {va!r} != {vb!r}"


def _assert_same_keys(a: dict, b: dict, label: str) -> None:
    """Raise if two dicts have different key sets.

    Args:
        a: First dict.
        b: Second dict.
        label: Name of the dicts, for the error message.
    """
    missing = set(a) - set(b)
    extra = set(b) - set(a)
    messages = []
    if missing:
        messages.append(f"keys in a but not b: {sorted(missing)}")
    if extra:
        messages.append(f"keys in b but not a: {sorted(extra)}")
    if messages:
        raise AssertionError(f"{label} key mismatch: {'; '.join(messages)}")


# ── Input builders ───────────────────────────────────────────────────────────

_MINIMAL_3AA = Path(__file__).parent / "fixtures" / "mmcifs" / "minimal_3aa.cif"
_MINIMAL_3AA_HEADER = (
    "data_structure\n#\nloop_\n_chem_comp.id\n_chem_comp.type\n"
    "ALA 'L-peptide linking'\nGLY 'peptide linking'\n#\n_exptl.method 'X-RAY DIFFRACTION'\n#\n"
)


def write_emptiable_minimal_3aa(dest_dir: Path) -> Path:
    """Write minimal_3aa.cif plus chem_comp/_exptl so cleanup runs and empties it.

    Args:
        dest_dir: Directory to write ``minimal_3aa.cif`` into.

    Returns:
        Path of the written CIF.
    """
    text = _MINIMAL_3AA.read_text()
    assert "data_structure\n" in text
    out = dest_dir / _MINIMAL_3AA.name
    out.write_text(text.replace("data_structure\n", _MINIMAL_3AA_HEADER))
    return out
