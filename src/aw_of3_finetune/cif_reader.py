"""Helpers for extracting per-chain structural data from mmCIF files."""

from __future__ import annotations

from pathlib import Path

import biotite.structure.io.pdbx as pdbx


def parse_cif_chain_data(cif_path: Path) -> dict[str, dict]:
    """Return per-chain structural data from a mmCIF file.

    Reads struct_asym, entity, entity_poly, pdbx_entity_nonpoly and atom_site with
    biotite's text CIF parser. Water chains are excluded.

    Args:
        cif_path: mmCIF file to read.

    Returns:
        A dict mapping label_asym_id → {
            "auth_asym_id":     str,
            "entity_id":        int,
            "molecule_type":    "PROTEIN" | "RNA" | "DNA" | "LIGAND",
            "reference_mol_id": str | None,   # CCD code for non-polymer LIGAND, else None
        }
    """
    block = pdbx.CIFFile.read(cif_path).block

    # struct_asym: label_asym_id → entity_id
    sa = block["struct_asym"]
    label_to_entity: dict[str, str] = {
        str(a): str(e) for a, e in zip(sa["id"].as_array(), sa["entity_id"].as_array())
    }

    # entity: entity_id → type string (polymer/non-polymer/water/branched)
    ent = block["entity"]
    entity_type: dict[str, str] = {
        str(eid): str(etype)
        for eid, etype in zip(ent["id"].as_array(), ent["type"].as_array())
    }

    # entity_poly: entity_id → lowercased sequence type
    #   examples: "polypeptide(l)", "polyribonucleotide", "polydeoxyribonucleotide"
    #   Some sources (OpenBind) lack the type column; their polymers fall through to LIGAND below.
    entity_poly_type: dict[str, str] = {}
    if "entity_poly" in block and "type" in block["entity_poly"]:
        ep = block["entity_poly"]
        for eid, ptype in zip(ep["entity_id"].as_array(), ep["type"].as_array()):
            entity_poly_type[str(eid)] = str(ptype).lower()

    # pdbx_entity_nonpoly: entity_id → CCD code (absent in polymer-only structures)
    entity_ccd: dict[str, str] = {}
    if "pdbx_entity_nonpoly" in block:
        np_cat = block["pdbx_entity_nonpoly"]
        for eid, ccd in zip(
            np_cat["entity_id"].as_array(), np_cat["comp_id"].as_array()
        ):
            entity_ccd[str(eid)] = str(ccd)

    # atom_site: take the first auth_asym_id seen for each label_asym_id
    at = block["atom_site"]
    label_to_auth: dict[str, str] = {}
    for label, auth in zip(
        at["label_asym_id"].as_array(), at["auth_asym_id"].as_array()
    ):
        label = str(label)
        if label not in label_to_auth:
            label_to_auth[label] = str(auth)

    # A non-polymer entity pdbx_entity_nonpoly does not name (OpenBind's CIFs have no such table):
    # its component is the residue name of its first atom_site row.
    for eid, comp in zip(at["label_entity_id"].as_array(), at["label_comp_id"].as_array()):
        eid = str(eid)
        if entity_type.get(eid) == "non-polymer" and eid not in entity_ccd:
            entity_ccd[eid] = str(comp)

    def _molecule_type(entity_id: str) -> str:
        if entity_type.get(entity_id) == "polymer":
            ptype = entity_poly_type.get(entity_id, "")
            if "polypeptide" in ptype:
                return "PROTEIN"
            if "deoxy" in ptype:  # polydeoxyribonucleotide — check before 'ribo'
                return "DNA"
            if "ribonucleotide" in ptype:
                return "RNA"
        return "LIGAND"  # non-polymer, branched, or unrecognised polymer type

    chains: dict[str, dict] = {}
    for label, entity_id in label_to_entity.items():
        if entity_type.get(entity_id) == "water":
            continue
        reference_mol_id = None
        if entity_type.get(entity_id) == "non-polymer":
            reference_mol_id = entity_ccd.get(entity_id)
        chains[label] = {
            "auth_asym_id": label_to_auth.get(label, label),
            "entity_id": int(entity_id),
            "molecule_type": _molecule_type(entity_id),
            "reference_mol_id": reference_mol_id,
        }
    return chains
