"""Structure checks V1-V3 and V5-V7: chains, sequences, coordinates, numbering."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from atomworks.constants import CRYSTALLIZATION_AIDS
from biotite.structure.io.pdbx import CIFFile
from rdkit import Chem
from scipy.spatial import cKDTree

from aw_of3_finetune.pipeline import ChainMismatchError, check_chain_consistency
from aw_of3_finetune.validate.context import ValidationContext
from aw_of3_finetune.validate.findings import Finding, Severity

COORD_TOL = 1e-3
POLYMER_MOLECULE_TYPE_IDS = (0, 1, 2)
PROTEIN_MOLECULE_TYPE_ID = 0
# Symmetry-equivalent atom pairs that atomworks may rename; a name swaps only with its partner.
_EQUIVALENT_PAIRS = {
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "LEU": (("CD1", "CD2"),),
    "VAL": (("CG1", "CG2"),),
}
EQUIVALENT_ATOM_NAMES = {
    res: {**dict(pairs), **{b: a for a, b in pairs}}
    for res, pairs in _EQUIVALENT_PAIRS.items()
}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


def _base_label(label: str) -> str:
    """Strip a chain_iid symmetry suffix: ``"A_2"`` becomes ``"A"``."""
    return label.split("_")[0]


def _per_entry(
    ctx: ValidationContext,
    check: str,
    fn: Callable[[ValidationContext, str], list[Finding]],
) -> list[Finding]:
    """Run ``fn`` on every entry, turning an exception into an E finding on that entry."""
    findings: list[Finding] = []
    for entry in ctx.entries():
        try:
            findings.extend(fn(ctx, entry))
        except Exception as exc:
            findings.append(
                Finding(check, Severity.ERROR, entry, f"{type(exc).__name__}: {exc}")
            )
    return findings


def _column(block, category: str, name: str) -> list[str]:
    """Return a CIF column as strings, or an empty list if the category or column is absent."""
    if category not in block or name not in block[category]:
        return []
    return [str(v) for v in block[category][name].as_array(str)]


def _output_labels(block) -> list[str]:
    """Return the ``label_asym_id`` values in an augmented CIF's ``of3_chain_meta``."""
    return _column(block, "of3_chain_meta", "label_asym_id")


def _entity_sequences(block) -> dict[str, str]:
    """Map entity id to canonical one-letter sequence (newlines stripped)."""
    ids = _column(block, "entity_poly", "entity_id")
    seqs = _column(block, "entity_poly", "pdbx_seq_one_letter_code_can")
    return {i: s.replace("\n", "") for i, s in zip(ids, seqs)}


class _SourceAtoms:
    """Model-1 ``atom_site`` rows of a source CIF, grouped by ``label_asym_id``."""

    def __init__(self, block) -> None:
        site = block["atom_site"]
        model = _column(block, "atom_site", "pdbx_PDB_model_num")
        keep = np.array([m == model[0] for m in model]) if model else slice(None)
        self.asym = np.array(_column(block, "atom_site", "label_asym_id"))[keep]
        self.comp = np.array(_column(block, "atom_site", "label_comp_id"))[keep]
        self.name = np.array(_column(block, "atom_site", "label_atom_id"))[keep]
        self.element = np.char.upper(
            np.array(_column(block, "atom_site", "type_symbol"))
        )[keep]
        self.xyz = np.stack(
            [site[c].as_array(float) for c in ("Cartn_x", "Cartn_y", "Cartn_z")], axis=1
        )[keep]
        self._trees: dict[str, tuple[np.ndarray, cKDTree]] = {}

    def tree(self, asym: str) -> tuple[np.ndarray, cKDTree | None]:
        """Return (row indices, KD-tree) of the atoms of one source chain; tree is None if empty."""
        if asym not in self._trees:
            idx = np.flatnonzero(self.asym == asym)
            self._trees[asym] = (idx, cKDTree(self.xyz[idx]) if len(idx) else None)
        return self._trees[asym]


def _check_v1(ctx: ValidationContext, entry: str) -> list[Finding]:
    source = CIFFile.read(ctx.source_cif(entry)).block
    out = CIFFile.read(ctx.augmented_cif(entry)).block
    kept = {_base_label(lab) for lab in _output_labels(out)}
    atoms = _SourceAtoms(source)
    in_assembly: set[str] | None = None
    ids = _column(source, "pdbx_struct_assembly_gen", "assembly_id")
    if ids:
        lists = _column(source, "pdbx_struct_assembly_gen", "asym_id_list")
        in_assembly = {
            a for i, lst in zip(ids, lists) if i == "1" for a in lst.split(",")
        }
    findings = []
    for chain in _column(source, "struct_asym", "id"):
        if chain in kept:
            continue
        comps = set(atoms.comp[atoms.asym == chain])
        if in_assembly is not None and chain not in in_assembly:
            reason = "not_in_assembly_1"
        elif comps and comps <= {"HOH", "DOD"}:
            reason = "water"
        elif comps and comps <= set(CRYSTALLIZATION_AIDS):
            reason = "crystallization_aid"
        elif not comps:
            reason = "no_resolved_atoms"
        else:
            findings.append(
                Finding(
                    "V1",
                    Severity.ERROR,
                    entry,
                    f"chain {chain} dropped without a known reason",
                    {"chain": chain, "residues": sorted(str(c) for c in comps)},
                )
            )
            continue
        findings.append(
            Finding(
                "V1", Severity.INFO, entry, reason, {"chain": chain, "reason": reason}
            )
        )
    return findings


def check_v1_chains_accounted(ctx: ValidationContext) -> list[Finding]:
    """V1: every source chain is in the output or dropped for a known reason (INFO), else E.

    Args:
        ctx: Validation context.

    Returns:
        INFO findings for explained drops, E findings for unexplained ones.
    """
    return _per_entry(ctx, "V1", _check_v1)


def _check_v2(ctx: ValidationContext, entry: str) -> list[Finding]:
    out = _entity_sequences(CIFFile.read(ctx.augmented_cif(entry)).block)
    src = _entity_sequences(CIFFile.read(ctx.source_cif(entry)).block)
    findings = []
    for entity, seq in out.items():
        if entity not in src:
            findings.append(
                Finding(
                    "V2", Severity.ERROR, entry, f"entity {entity} missing from source"
                )
            )
        elif src[entity] != seq:
            findings.append(
                Finding(
                    "V2",
                    Severity.ERROR,
                    entry,
                    f"entity {entity} sequence differs from source",
                    {
                        "entity": entity,
                        "output_length": len(seq),
                        "source_length": len(src[entity]),
                    },
                )
            )
    return findings


def check_v2_polymer_sequences(ctx: ValidationContext) -> list[Finding]:
    """V2: each output entity's canonical polymer sequence equals the source's.

    Args:
        ctx: Validation context.

    Returns:
        E findings for differing or missing entities.
    """
    return _per_entry(ctx, "V2", _check_v2)


def _kabsch_max_deviation(moving: np.ndarray, fixed: np.ndarray) -> float:
    """Largest per-atom distance after optimal rigid (proper-rotation) superposition."""
    a = moving - moving.mean(axis=0)
    b = fixed - fixed.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    sign = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, sign]) @ vt
    return float(np.sqrt(np.sum((a @ rot - b) ** 2, axis=1)).max())


def _name_match(src_name: str, name: str, res_name: str) -> str | None:
    """Return "exact", "swap" (partner name of a symmetric pair) or None."""
    if src_name == name:
        return "exact"
    if EQUIVALENT_ATOM_NAMES.get(str(res_name), {}).get(name) == src_name:
        return "swap"
    return None


def _chain_matches_source(
    npz: dict, mask: np.ndarray, src: _SourceAtoms, asym: str
) -> tuple[bool, dict, set[tuple[int, str]]]:
    """Check heavy atoms of one chain; return (ok, worst-atom detail, residues matched via a swap)."""
    idx, tree = src.tree(asym)
    names, elements = npz["atom_name"][mask], np.char.upper(npz["element"][mask])
    res_names, res_ids = npz["res_name"][mask], npz["res_id"][mask]
    coords = npz["coord"][mask].astype(float)
    worst, bad, swapped = None, 0, set()
    for i in range(len(coords)):
        hits = tree.query_ball_point(coords[i], COORD_TOL) if tree is not None else []
        kinds = {
            _name_match(src.name[idx[h]], names[i], res_names[i])
            for h in hits
            if src.element[idx[h]] == elements[i]
        } - {None}
        if kinds:
            if "exact" not in kinds:
                swapped.add((int(res_ids[i]), str(res_names[i])))
            continue
        bad += 1
        dist = float(tree.query(coords[i])[0]) if tree is not None else None
        if worst is None or (
            dist is not None and (worst["distance"] is None or dist > worst["distance"])
        ):
            worst = {"atom": f"{names[i]} (res_id {int(res_ids[i])})", "distance": dist}
    return bad == 0, {"chain": asym, "n_unmatched": bad, **(worst or {})}, swapped


def _atom_keys(npz: dict, mask: np.ndarray) -> list[tuple]:
    return list(
        zip(
            npz["res_id"][mask].tolist(),
            npz["ins_code"][mask].tolist(),
            npz["atom_name"][mask].tolist(),
        )
    )


def _check_v3(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    src = _SourceAtoms(CIFFile.read(ctx.source_cif(entry)).block)
    heavy = ~np.isin(np.char.upper(npz["element"]), ["H", "D"])
    # Unresolved atoms (NaN coordinates, added by cleanup) have nothing to match.
    resolved = ~np.isnan(npz["coord"].astype(float)).any(axis=1)
    labels = list(dict.fromkeys(npz["label_asym_id"].tolist()))
    masks = {lab: heavy & resolved & (npz["label_asym_id"] == lab) for lab in labels}
    passed: dict[str, str] = {}
    failed: dict[str, dict] = {}
    findings: list[Finding] = []
    for lab in labels:
        ok, detail, swapped = _chain_matches_source(
            npz, masks[lab], src, _base_label(lab)
        )
        if ok:
            passed.setdefault(_base_label(lab), lab)
            findings.extend(
                Finding(
                    "V3",
                    Severity.INFO,
                    entry,
                    "expected (atom-name swap)",
                    {"chain": lab, "res_id": res_id, "res_name": res_name},
                )
                for res_id, res_name in sorted(swapped)
            )
        else:
            failed[lab] = detail
    for lab, detail in failed.items():
        suffix = lab.split("_", 1)[1] if "_" in lab else ""
        ref = passed.get(_base_label(lab))
        if suffix.isdigit() and int(suffix) > 1 and ref is not None:
            keys, ref_keys = _atom_keys(npz, masks[lab]), _atom_keys(npz, masks[ref])
            if keys == ref_keys:
                max_dev = _kabsch_max_deviation(
                    npz["coord"][masks[lab]].astype(float),
                    npz["coord"][masks[ref]].astype(float),
                )
                if max_dev <= COORD_TOL:
                    findings.append(
                        Finding(
                            "V3",
                            Severity.INFO,
                            entry,
                            f"chain {lab}: symmetry copy of {ref}",
                            {"chain": lab, "reference": ref, "max_deviation": max_dev},
                        )
                    )
                    continue
        findings.append(
            Finding(
                "V3",
                Severity.ERROR,
                entry,
                f"chain {lab}: {detail['n_unmatched']} atom(s) not in source",
                detail,
            )
        )
    return findings


def check_v3_atoms_match_source(ctx: ValidationContext) -> list[Finding]:
    """V3: every resolved output heavy atom has a same-name source atom within 1e-3 A (any altloc).

    Args:
        ctx: Validation context.

    Returns:
        E findings with the worst atom; INFO findings for accepted symmetry copies.
    """
    return _per_entry(ctx, "V3", _check_v3)


def _sdf_problem(path) -> str | None:
    """Return why an SDF's conformer coordinates are unusable, or None if finite."""
    mol = next(
        iter(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)), None
    )
    if mol is None or mol.GetNumConformers() == 0:
        return "unreadable or no conformer"
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        return "non-finite conformer coordinates"
    return None


def _check_v5(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    coord = npz["coord"].astype(float)
    findings = []
    if np.isinf(coord).any():
        n = int(np.isinf(coord).any(axis=1).sum())
        findings.append(
            Finding("V5", Severity.ERROR, entry, "infinite coordinates", {"n_atoms": n})
        )
    resolved = ~np.isnan(coord).any(axis=1)
    if not np.nan_to_num(coord[resolved], posinf=0.0, neginf=0.0).any():
        findings.append(
            Finding("V5", Severity.ERROR, entry, "no resolved atom has a non-zero coordinate")
        )
    for chain in np.unique(npz["chain_id"]):
        if not resolved[npz["chain_id"] == chain].any():
            findings.append(
                Finding(
                    "V5",
                    Severity.ERROR,
                    entry,
                    f"chain {chain} has no resolved atoms (all coordinates NaN)",
                    {"chain": str(chain)},
                )
            )
    chains = ctx.dataset_cache.structure_data[entry].chains.values()
    mol_ids = set(ctx.npz(entry)["res_name"].tolist())
    mol_ids |= {c.reference_mol_id for c in chains if c.reference_mol_id}
    for mol_id in sorted(mol_ids):
        path = ctx.sdf(mol_id)
        problem = _sdf_problem(path) if path.exists() else None
        if problem:
            findings.append(
                Finding(
                    "V5",
                    Severity.ERROR,
                    entry,
                    f"{mol_id}.sdf: {problem}",
                    {"mol_id": mol_id},
                )
            )
    return findings


def check_v5_finite_coordinates(ctx: ValidationContext) -> list[Finding]:
    """V5: NPZ coordinates usable: NaN only for unresolved atoms, no inf, no all-NaN chain.

    Unresolved atoms carry NaN coordinates by design (OF3 cleanup). Reference SDFs used
    by the entry must be finite.

    Args:
        ctx: Validation context.

    Returns:
        E findings for bad coordinates.
    """
    return _per_entry(ctx, "V5", _check_v5)


def _check_v6(ctx: ValidationContext, entry: str) -> list[Finding]:
    try:
        check_chain_consistency(
            ctx.augmented_cif(entry), ctx.dataset_cache.structure_data[entry]
        )
    except ChainMismatchError as exc:
        return [Finding("V6", Severity.ERROR, entry, str(exc))]
    return []


def check_v6_cache_matches_structure(ctx: ValidationContext) -> list[Finding]:
    """V6: dataset-cache chain metadata agrees with the augmented CIF's atoms.

    Args:
        ctx: Validation context.

    Returns:
        E findings carrying the mismatch message.
    """
    return _per_entry(ctx, "V6", _check_v6)


def _check_v7(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    seqs = _entity_sequences(CIFFile.read(ctx.augmented_cif(entry)).block)
    findings = []
    for chain in dict.fromkeys(npz["chain_id"].tolist()):
        mask = npz["chain_id"] == chain
        if int(npz["molecule_type_id"][mask][0]) not in POLYMER_MOLECULE_TYPE_IDS:
            continue
        label = str(npz["label_asym_id"][mask][0])
        seq = seqs.get(str(int(npz["entity_id"][mask][0])), "")
        keys = list(
            dict.fromkeys(
                zip(
                    npz["res_id"][mask].tolist(),
                    npz["ins_code"][mask].tolist(),
                    npz["res_name"][mask].tolist(),
                )
            )
        )
        problem = None
        prev = 0
        for res_id, _, res_name in keys:
            if res_id <= prev:
                problem = f"residue ids not strictly increasing at {res_id}"
            elif not 1 <= res_id <= len(seq):
                problem = f"residue id {res_id} outside 1..{len(seq)}"
            elif (
                int(npz["molecule_type_id"][mask][0]) == PROTEIN_MOLECULE_TYPE_ID
                and res_name in THREE_TO_ONE
                and THREE_TO_ONE[res_name] != seq[res_id - 1]
            ):
                problem = (
                    f"{res_name} at res_id {res_id} but sequence has {seq[res_id - 1]}"
                )
            if problem:
                break
            prev = res_id
        if problem:
            findings.append(
                Finding(
                    "V7",
                    Severity.ERROR,
                    entry,
                    f"chain {label}: {problem}",
                    {"chain": label},
                )
            )
    return findings


def check_v7_residue_numbering(ctx: ValidationContext) -> list[Finding]:
    """V7: polymer residue ids increase within 1..len(sequence) and match the sequence letters.

    Args:
        ctx: Validation context.

    Returns:
        E findings, one per offending chain.
    """
    return _per_entry(ctx, "V7", _check_v7)
