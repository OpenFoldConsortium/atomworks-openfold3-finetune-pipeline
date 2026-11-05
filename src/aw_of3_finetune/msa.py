"""MSA discovery and sequence utilities for fine-tuning dataset preparation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
from atomworks.ml.utils.misc import get_complex_id
from atomworks.ml.utils.misc import hash_sequence as _hash_sequence

logger = logging.getLogger(__name__)


def alignment_rep_id(sequence: str) -> str:
    """Return the MSA alignment representative ID for a polymer sequence.

    atomworks' SHA-256[:11] hash, the same key its find_msas/get_msa_path and
    make_msas_colabfold_server_batch use for MSA files.

    Args:
        sequence: Polymer sequence.

    Returns:
        The 11-character rep_id.
    """
    return _hash_sequence(sequence)


def chain_canonical_sequences(cif_path: Path, protein_only: bool = False) -> dict[str, str]:
    """Extract canonical 1-letter sequences per polymer chain from a mmCIF file.

    Reads ``entity_poly.pdbx_seq_one_letter_code_can``, the SEQRES-derived sequence
    OF3 uses as the canonical source for MSA hashing.

    Args:
        cif_path: mmCIF file with entity_poly and struct_asym.
        protein_only: Keep only polypeptide entities, dropping DNA/RNA. Use this for
            anything that feeds ColabFold: it treats nucleic acids as protein
            (U -> X), and a paired query containing one hangs.

    Returns:
        {label_asym_id: sequence} for polymer chains only.
    """
    block = pdbx.CIFFile.read(cif_path).block

    ep = block["entity_poly"]
    entity_ids = ep["entity_id"].as_array()
    seqs = ep["pdbx_seq_one_letter_code_can"].as_array()
    if protein_only:
        types = ep["type"].as_array()
        keep = [str(t).lower().startswith("polypeptide") for t in types]
    else:
        keep = [True] * len(entity_ids)
    entity_seq: dict[str, str] = {
        str(eid): str(seq).replace("\n", "")
        for eid, seq, k in zip(entity_ids, seqs, keep)
        if k
    }

    sa = block["struct_asym"]
    label_to_entity: dict[str, str] = {
        str(label): str(eid)
        for label, eid in zip(sa["id"].as_array(), sa["entity_id"].as_array())
    }

    return {
        label: entity_seq[eid]
        for label, eid in label_to_entity.items()
        if eid in entity_seq
    }


def collect_msa_queries(output_dir: Path) -> dict[str, str]:
    """Collect unique protein MSA query sequences from a finetune-prep output directory.

    Scans the augmented CIFs listed in manifest.json. Structures without entity_poly
    (e.g. no SEQRES) are skipped with a warning, a known limitation for benchmark CIFs.

    Args:
        output_dir: finetune-prep output directory (manifest.json + structures/).

    Returns:
        {alignment_representative_id: sequence}, deduplicated across structures.
    """
    slim = json.loads((output_dir / "manifest.json").read_text())
    pdb_ids: list[str] = slim.get("structures", [])

    rep_id_to_seq: dict[str, str] = {}
    cif_dir = output_dir / "structures"

    for pdb_id in pdb_ids:
        cif_path = cif_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            logger.warning("%s: augmented CIF not found at %s, skipping", pdb_id, cif_path)
            continue
        try:
            seqs = chain_canonical_sequences(cif_path, protein_only=True)
        except Exception as exc:
            logger.warning("%s: could not read sequences (%s), skipping", pdb_id, exc)
            continue
        for seq in seqs.values():
            rep_id = alignment_rep_id(seq)
            rep_id_to_seq.setdefault(rep_id, seq)

    return rep_id_to_seq


def collect_complex_queries(output_dir: Path) -> dict[str, list[str]]:
    """Collect unique multi-chain complexes for paired MSA generation.

    Like collect_msa_queries, but per structure; only structures with >1 unique
    protein sequence are kept (homomers and monomers have nothing to pair).

    Args:
        output_dir: finetune-prep output directory (manifest.json + structures/).

    Returns:
        {complex_id: unique sequences}, deduplicated across structures regardless of
        chain order. Sequences are sorted by rep_id, so seqs[i] matches the i-th
        rep_id in ``complex_id.split("-")``.
    """
    slim = json.loads((output_dir / "manifest.json").read_text())
    pdb_ids: list[str] = slim.get("structures", [])

    complex_id_to_seqs: dict[str, list[str]] = {}
    cif_dir = output_dir / "structures"

    for pdb_id in pdb_ids:
        cif_path = cif_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            logger.warning("%s: augmented CIF not found at %s, skipping", pdb_id, cif_path)
            continue
        try:
            seqs = chain_canonical_sequences(cif_path, protein_only=True)
        except Exception as exc:
            logger.warning("%s: could not read sequences (%s), skipping", pdb_id, exc)
            continue

        unique_seqs = list(dict.fromkeys(seqs.values()))
        if len(unique_seqs) <= 1:
            continue  # homomer or monomer -- nothing to pair

        ordered_seqs = sorted(unique_seqs, key=alignment_rep_id)
        # atomworks' get_complex_id hashes raw sequences itself; it is the canonical
        # definition shared with make_msas_colabfold_server/get_paired_msa_path.
        complex_id = get_complex_id(ordered_seqs)
        complex_id_to_seqs.setdefault(complex_id, ordered_seqs)

    return complex_id_to_seqs
