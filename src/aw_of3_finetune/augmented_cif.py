"""Writer and reader-shim for augmented CIF files.

An augmented CIF is a text CIF (so OF3's DatasetCachePreprocessor finds it as ``*.cif``)
with atomworks' standard atom_site plus the cache metadata as extra categories:
``of3_structure_meta``, ``of3_chain_meta``, ``of3_interfaces`` and ``of3_ref_mol`` (the
last two absent when empty). Optional fields encode None as the mmCIF ``"."`` sentinel.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import biotite.structure.io.pdbx as pdbx
from atomworks.enums import ChainType

from aw_of3_finetune.parsing import build_chem_comp_category, write_cif_buffer
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetInterfaceData,
    ClusteredDatasetStructureData,
    DatasetReferenceMoleculeData,
)

_NULL = "."  # mmCIF convention for "not applicable"


# ── Encoding helpers ─────────────────────────────────────────────────────────


def _enc_optional_str(s: str | None) -> str:
    return _NULL if s is None else s


def _dec_optional_str(s: str) -> str | None:
    return None if s == _NULL else s


def _enc_template_ids(ids: list[str] | None) -> str:
    if ids is None:
        return _NULL  # "."  → None on decode
    return "|".join(ids)  # ""   → []  on decode, "a|b" → ["a","b"]


def _dec_template_ids(s: str) -> list[str] | None:
    if s == _NULL:
        return None
    if not s:
        return []
    return s.split("|")


def _enc_resolution(r: float | None) -> str:
    # Keep None and NaN distinct so round-trip is bit-exact.
    if r is None:
        return ""
    if isinstance(r, float) and np.isnan(r):
        return "nan"
    return str(float(r))


def _dec_resolution(s: str) -> float | None:
    if s == "":
        return None
    if s == "nan":
        return float("nan")
    return float(s)


def _col_str(col, i: int) -> str:
    """Read one string value from a CIF column, mapping masked entries to ``"."``.

    Args:
        col: Biotite CIF column.
        i: Row index.

    Returns:
        The value as a string, or ``"."`` if masked.
    """
    if col.mask is not None and col.mask._array[i] != 0:  # 0 = MaskValue.PRESENT
        return _NULL
    return str(col.as_array()[i])


def _rows_to_cols(rows: list[dict]) -> dict[str, list]:
    if not rows:
        return {}
    return {key: [r[key] for r in rows] for key in rows[0]}


# ── Writer ───────────────────────────────────────────────────────────────────


def build_extra_categories(
    structure_data: ClusteredDatasetStructureData,
    reference_molecule_data: dict[str, DatasetReferenceMoleculeData],
) -> dict[str, dict[str, list]]:
    """Build the ``of3_*`` metadata categories to pass to write_cif_buffer.

    Args:
        structure_data: Cache metadata for one structure.
        reference_molecule_data: {mol_id: reference molecule data}.

    Returns:
        {category: {column: values}}; interface/ref-mol categories only if non-empty.
    """
    cats: dict[str, dict[str, list]] = {}

    # Structure-level metadata (1 row)
    cats["of3_structure_meta"] = {
        "release_date": [str(structure_data.release_date)],
        "resolution": [_enc_resolution(structure_data.resolution)],
    }

    # Per-chain metadata.
    # chain_id is the dict key used in ClusteredDatasetStructureData.chains; it may
    # differ from label_asym_id (e.g. integer IDs assigned by preprocess_cif_dir_of3).
    if structure_data.chains:
        chain_rows = [
            {
                "chain_id": chain_key,
                "label_asym_id": cd.label_asym_id,
                "auth_asym_id": cd.auth_asym_id,
                "entity_id": int(cd.entity_id),
                "molecule_type": cd.molecule_type,
                "reference_mol_id": _enc_optional_str(cd.reference_mol_id),
                "alignment_representative_id": _enc_optional_str(
                    cd.alignment_representative_id
                ),
                "template_ids": _enc_template_ids(cd.template_ids),
                "cluster_id": cd.cluster_id,
                "cluster_size": int(cd.cluster_size),
            }
            for chain_key, cd in structure_data.chains.items()
        ]
        cats["of3_chain_meta"] = _rows_to_cols(chain_rows)

    # Per-interface metadata
    if structure_data.interfaces:
        iface_rows = [
            {
                "iface_id": iface_id,
                "cluster_id": iface.cluster_id,
                "cluster_size": int(iface.cluster_size),
            }
            for iface_id, iface in structure_data.interfaces.items()
        ]
        cats["of3_interfaces"] = _rows_to_cols(iface_rows)

    # Reference molecule data
    if reference_molecule_data:
        ref_rows = [
            {
                "mol_id": mol_id,
                "conformer_gen_strategy": rm.conformer_gen_strategy,
                "fallback_conformer_pdb_id": _enc_optional_str(
                    rm.fallback_conformer_pdb_id
                ),
                "canonical_smiles": rm.canonical_smiles,
                "set_fallback_to_nan": "true" if rm.set_fallback_to_nan else "false",
                "fallback_conformer_source": _enc_optional_str(
                    rm.fallback_conformer_source
                ),
            }
            for mol_id, rm in reference_molecule_data.items()
        ]
        cats["of3_ref_mol"] = _rows_to_cols(ref_rows)

    return cats


UNKNOWN_POLY_TYPE = "?"  # mmCIF unknown: an entity_poly entity with no atoms left after cleanup


def _poly_types(entity_ids: list, atom_array, atom_site) -> list[str]:
    """entity_poly.type per entity from atomworks' chain_type (it infers it when the CIF lacks the column).

    Args:
        entity_ids: The entity_poly entities.
        atom_array: The structure being written; chain_id is the label_asym_id.
        atom_site: The source's atom_site, which maps label_asym_id to entity.

    Returns:
        One mmCIF polymer type per entity.
    """
    label_asym_ids = atom_site["label_asym_id"].as_array(str)
    label_entity_ids = atom_site["label_entity_id"].as_array(str)
    types = []
    for entity_id in entity_ids:
        asym_ids = np.unique(label_asym_ids[label_entity_ids == str(entity_id)])
        chain_types = atom_array.chain_type[np.isin(atom_array.chain_id, asym_ids)]
        types.append(ChainType(int(chain_types[0])).to_mmcif_string() if len(chain_types) else UNKNOWN_POLY_TYPE)
    return types


def source_categories(source_cif: Path, atom_array=None) -> dict[str, dict[str, list]]:
    """Return the source mmCIF categories that must be carried over verbatim.

    The atomworks writer emits only atom_site, so these are copied explicitly.

    Args:
        source_cif: Original mmCIF file.
        atom_array: The structure being written; gives entity_poly.type when the source lacks it
            (OpenBind's monomer CIFs). Without it such a source raises KeyError.

    Returns:
        {category: {column: values}} for whichever of ``chem_comp`` (id, type),
        ``entity_poly`` and ``struct_asym`` exist in the source.
    """
    block = pdbx.CIFFile.read(source_cif).block
    categories: dict[str, dict[str, list]] = {}
    # chem_comp: OF3's parse_mmcif needs it to assign molecule types; carrying the
    # source's table makes the CIF OF3-parseable without a CCD.
    if "chem_comp" in block:
        cc = block["chem_comp"]
        categories["chem_comp"] = {
            "id": list(cc["id"].as_array()),
            "type": list(cc["type"].as_array()),
        }
    # entity_poly + struct_asym: chain_canonical_sequences reads SEQRES from these.
    if "entity_poly" in block:
        ep = block["entity_poly"]
        categories["entity_poly"] = {
            "entity_id": list(ep["entity_id"].as_array()),
            # protein vs DNA/RNA, for MSA query selection
            "type": list(ep["type"].as_array()) if "type" in ep else _poly_types(list(ep["entity_id"].as_array()), atom_array, block["atom_site"]),
            "pdbx_seq_one_letter_code_can": list(
                ep["pdbx_seq_one_letter_code_can"].as_array()
            ),
        }
    if "struct_asym" in block:
        sa = block["struct_asym"]
        categories["struct_asym"] = {
            "id": list(sa["id"].as_array()),
            "entity_id": list(sa["entity_id"].as_array()),
        }
    return categories


def write_augmented_cif(
    atom_array,
    structure_data: ClusteredDatasetStructureData,
    reference_molecule_data: dict[str, DatasetReferenceMoleculeData],
    path: Path,
    pdb_id: str = "structure",
    source_cif: Path | None = None,
    ccd: pdbx.CIFFile | None = None,
) -> None:
    """Write atom_array plus cache metadata to a text augmented CIF.

    Args:
        atom_array: Structure to write.
        structure_data: Cache metadata for the structure.
        reference_molecule_data: {mol_id: reference molecule data}.
        path: Output CIF path.
        pdb_id: Entry ID written as the data block name.
        source_cif: Original mmCIF; its chem_comp, entity_poly and struct_asym are
            copied over (needed by generate-msas and OF3's parse_mmcif).
        ccd: Chemical Component Dictionary; used to build a minimal chem_comp
            (parsing.build_chem_comp_category) only if source_cif provides none.
    """
    extra_categories = build_extra_categories(structure_data, reference_molecule_data)
    if source_cif is not None:
        extra_categories.update(source_categories(source_cif, atom_array))
    if ccd is not None and "chem_comp" not in extra_categories:
        extra_categories["chem_comp"] = build_chem_comp_category(atom_array, ccd)
    buf = write_cif_buffer(atom_array, id=pdb_id, extra_categories=extra_categories, as_bcif=False)
    path.write_text(buf.read())


# ── Reader-shim ──────────────────────────────────────────────────────────────


def read_augmented_cif(
    path: Path,
    pdb_id: str,
    cache_name: str,
) -> ClusteredDatasetCache:
    """Read an augmented CIF and reconstruct its ClusteredDatasetCache.

    Args:
        path: Augmented CIF to read.
        pdb_id: Key for the structure in the cache's structure_data.
        cache_name: Name given to the returned cache.

    Returns:
        A single-structure ClusteredDatasetCache.
    """
    cif_file = pdbx.CIFFile.read(path)
    block = cif_file.block

    structure_data = _parse_structure_data(block)
    reference_molecule_data = _parse_ref_mol_data(block)

    return ClusteredDatasetCache(
        name=cache_name,
        structure_data={pdb_id: structure_data},
        reference_molecule_data=reference_molecule_data,
    )


def _parse_structure_data(block) -> ClusteredDatasetStructureData:
    meta = block["of3_structure_meta"]
    release_date = date.fromisoformat(str(meta["release_date"].as_array()[0]))
    resolution = _dec_resolution(str(meta["resolution"].as_array()[0]))

    chains = _parse_chain_meta(block)
    interfaces = _parse_interfaces(block)

    return ClusteredDatasetStructureData(
        release_date=release_date,
        resolution=resolution,
        chains=chains,
        interfaces=interfaces,
    )


def _parse_chain_meta(block) -> dict[str, ClusteredDatasetChainData]:
    if "of3_chain_meta" not in block:
        return {}
    cat = block["of3_chain_meta"]

    # chain_id is the dict key (may be an integer string like "1" from
    # preprocess_cif_dir_of3, or any other string for manually-built caches).
    # Fall back to label_asym_id for CIFs written before chain_id was added.
    if "chain_id" in cat:
        key_col = "chain_id"
    else:
        key_col = "label_asym_id"

    n = len(cat["label_asym_id"].as_array())
    chains: dict[str, ClusteredDatasetChainData] = {}
    for i in range(n):
        chain_key = _col_str(cat[key_col], i)
        chains[chain_key] = ClusteredDatasetChainData(
            label_asym_id=_col_str(cat["label_asym_id"], i),
            auth_asym_id=_col_str(cat["auth_asym_id"], i),
            entity_id=int(cat["entity_id"].as_array()[i]),
            molecule_type=_col_str(cat["molecule_type"], i),
            reference_mol_id=_dec_optional_str(_col_str(cat["reference_mol_id"], i)),
            alignment_representative_id=_dec_optional_str(
                _col_str(cat["alignment_representative_id"], i)
            ),
            template_ids=_dec_template_ids(_col_str(cat["template_ids"], i)),
            cluster_id=_col_str(cat["cluster_id"], i),
            cluster_size=int(cat["cluster_size"].as_array()[i]),
        )
    return chains


def _parse_interfaces(block) -> dict[str, ClusteredDatasetInterfaceData]:
    if "of3_interfaces" not in block:
        return {}
    cat = block["of3_interfaces"]

    n = len(cat["iface_id"].as_array())
    return {
        _col_str(cat["iface_id"], i): ClusteredDatasetInterfaceData(
            cluster_id=_col_str(cat["cluster_id"], i),
            cluster_size=int(cat["cluster_size"].as_array()[i]),
        )
        for i in range(n)
    }


def _parse_ref_mol_data(block) -> dict[str, DatasetReferenceMoleculeData]:
    if "of3_ref_mol" not in block:
        return {}
    cat = block["of3_ref_mol"]

    n = len(cat["mol_id"].as_array())
    ref_mols: dict[str, DatasetReferenceMoleculeData] = {}
    for i in range(n):
        ref_mols[_col_str(cat["mol_id"], i)] = DatasetReferenceMoleculeData(
            conformer_gen_strategy=_col_str(cat["conformer_gen_strategy"], i),
            fallback_conformer_pdb_id=_dec_optional_str(
                _col_str(cat["fallback_conformer_pdb_id"], i)
            ),
            canonical_smiles=_col_str(cat["canonical_smiles"], i),
            set_fallback_to_nan=_col_str(cat["set_fallback_to_nan"], i).lower()
            == "true",
            fallback_conformer_source=_dec_optional_str(
                _col_str(cat["fallback_conformer_source"], i)
            ),
        )
    return ref_mols
