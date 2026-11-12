"""Ligand checks: reference molecules and ligand chain integrity.

V4 has a ligand; V8-V12 reference SDFs exist, match the crystal (elements, atom names), agree
per mol ID and sanitize; V18 ligands are not inside polymers; V19 ligand copies stay separate.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from biotite.structure.io.pdbx import CIFFile
from openfold3.core.data.io.structure.mol import read_single_annotated_sdf
from rdkit import Chem
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from aw_of3_finetune.ligand_names import (
    LIGAND_MOLECULE_TYPE_ID,
    read_ccd_ids,
    residue_signature,
    signature_difference,
)
from aw_of3_finetune.validate.check_ids import (
    HAS_LIGAND,
    LIGANDS_NOT_IN_POLYMERS,
    LIGAND_COPIES_SEPARATE,
    ONE_CHEMISTRY_PER_MOL_ID,
    REFERENCE_SDFS_EXIST,
    REFERENCE_SDFS_SANITIZE,
    SDF_ATOM_NAMES_MATCH,
    SDF_ELEMENTS_MATCH,
)
from aw_of3_finetune.validate.context import ValidationContext
from aw_of3_finetune.validate.findings import Finding, Severity
from aw_of3_finetune.validate.structure import (
    POLYMER_MOLECULE_TYPE_IDS,
    _column,
    _per_entry,
)

WATER_NAMES = frozenset({"HOH", "DOD", "WAT"})
HYDROGENS = ("H", "D")


def _residues(ctx: ValidationContext, entry: str) -> list[dict]:
    """Return the ligand residues of an entry with their chain, atom rows and SDF id."""
    npz = ctx.npz(entry)
    chains = ctx.dataset_cache.structure_data[entry].chains
    keys = list(zip(npz["chain_id"].tolist(), npz["res_id"].tolist(), npz["ins_code"].tolist()))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for row, key in enumerate(keys):
        if int(npz["molecule_type_id"][row]) == LIGAND_MOLECULE_TYPE_ID:
            groups[key].append(row)
    residues = []
    for (chain, _, _), rows in groups.items():
        meta = chains.get(chain)
        res_name = str(npz["res_name"][rows[0]])
        ref = getattr(meta, "reference_mol_id", None)
        residues.append(
            {
                "chain": chain,
                "label": str(npz["label_asym_id"][rows[0]]),
                "res_name": res_name,
                "mol_id": ref or res_name,
                "rows": np.array(rows),
            }
        )
    return residues


def _is_trivial(res: dict) -> bool:
    """Whether a ligand residue is water or a single-atom ion."""
    return res["res_name"] in WATER_NAMES or len(res["rows"]) == 1


def _check_v4(ctx: ValidationContext, entry: str) -> list[Finding]:
    if any(not _is_trivial(r) for r in _residues(ctx, entry)):
        return []
    return [
        Finding(HAS_LIGAND, Severity.WARNING, entry, "no ligand other than water and ions")
    ]


def check_v4_ligand_present(ctx: ValidationContext) -> list[Finding]:
    """V4: warn when an entry has no ligand beyond water and single-atom ions.

    Args:
        ctx: Validation context.

    Returns:
        W findings; empty unless ``ctx.expect_ligands``.
    """
    if not ctx.expect_ligands:
        return []
    return _per_entry(ctx, HAS_LIGAND, _check_v4)


def _check_v8(ctx: ValidationContext, entry: str) -> list[Finding]:
    findings = []
    seen = set()
    for res in _residues(ctx, entry):
        mol_id = res["mol_id"]
        if mol_id in seen or ctx.sdf(mol_id).exists():
            continue
        seen.add(mol_id)
        findings.append(
            Finding(
                REFERENCE_SDFS_EXIST,
                Severity.ERROR,
                entry,
                f"{mol_id}.sdf missing from reference_mols (chain {res['label']})",
                {"mol_id": mol_id, "chain": res["label"]},
            )
        )
    return findings


def check_v8_reference_mols_exist(ctx: ValidationContext) -> list[Finding]:
    """V8: every ligand residue has a reference SDF (chain mol id, else residue name).

    Args:
        ctx: Validation context.

    Returns:
        E findings, one per missing SDF per entry.
    """
    return _per_entry(ctx, REFERENCE_SDFS_EXIST, _check_v8)


def _sdf_atoms(ctx: ValidationContext, mol_id: str) -> dict[str, str] | None:
    """Map ``annot_atom_name`` to element symbol for an SDF; None if the file is absent.

    Raises:
        Exception: If the SDF exists but its atom names cannot be read.
    """
    path = ctx.sdf(mol_id)
    if not path.exists():
        return None
    mol = read_single_annotated_sdf(path)
    return {a.GetProp("annot_atom_name"): a.GetSymbol().upper() for a in mol.GetAtoms()}


def _check_v9_v10(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    findings = []
    cache: dict[str, dict[str, str] | None] = {}
    unreadable: set[str] = set()
    for res in _residues(ctx, entry):
        mol_id = res["mol_id"]
        if mol_id in unreadable:
            continue
        if mol_id not in cache:
            try:
                cache[mol_id] = _sdf_atoms(ctx, mol_id)
            except Exception as exc:
                unreadable.add(mol_id)
                findings.append(
                    Finding(
                        SDF_ATOM_NAMES_MATCH,
                        Severity.ERROR,
                        entry,
                        f"{mol_id}.sdf: atom names could not be read ({type(exc).__name__}: {exc})",
                        {"mol_id": mol_id},
                    )
                )
                continue
        ref = cache[mol_id]
        if ref is None:
            continue
        names = npz["atom_name"][res["rows"]].tolist()
        elements = np.char.upper(npz["element"][res["rows"]]).tolist()
        heavy = [(n, e) for n, e in zip(names, elements) if e not in HYDROGENS]
        extra = sorted({n for n, _ in heavy if n not in ref})
        if extra:
            findings.append(
                Finding(
                    SDF_ATOM_NAMES_MATCH,
                    Severity.ERROR,
                    entry,
                    f"{mol_id}: crystal atom name(s) {extra} not in reference SDF "
                    f"(chain {res['label']})",
                    {"mol_id": mol_id, "atoms": extra, "chain": res["label"]},
                )
            )
        wrong = sorted({(n, e, ref[n]) for n, e in heavy if n in ref and ref[n] != e})
        for name, crystal, sdf in wrong:
            findings.append(
                Finding(
                    SDF_ELEMENTS_MATCH,
                    Severity.ERROR,
                    entry,
                    f"{mol_id}: atom {name} has element {crystal} in crystal "
                    f"but {sdf} in reference SDF (chain {res['label']})",
                    {"mol_id": mol_id, "atom": name, "crystal": crystal, "sdf": sdf,
                     "chain": res["label"]},
                )
            )
    return findings


def check_v9_v10_ref_atoms_match_crystal(ctx: ValidationContext) -> list[Finding]:
    """V9/V10: crystal heavy atoms have matching elements (V9) and known names (V10) in the SDF.

    Args:
        ctx: Validation context.

    Returns:
        E findings tagged ``SDF_ELEMENTS_MATCH`` (V9) or ``SDF_ATOM_NAMES_MATCH`` (V10).
    """
    findings: list[Finding] = []
    for entry in ctx.entries():
        try:
            findings.extend(_check_v9_v10(ctx, entry))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            findings.extend(
                Finding(check, Severity.ERROR, entry, message) for check in (SDF_ELEMENTS_MATCH, SDF_ATOM_NAMES_MATCH)
            )
    return findings


def _conflict(sig_a: tuple, sig_b: tuple) -> str | None:
    """Describe how two residue signatures of one molecule disagree, or None."""
    elem_a, elem_b = dict(sig_a[0]), dict(sig_b[0])
    shared = sorted(set(elem_a) & set(elem_b))
    if not shared:
        return "no atom names in common (different molecules under one id)"
    for name in shared:
        if elem_a[name] != elem_b[name]:
            return f"element of atom {name} is {elem_a[name]} vs {elem_b[name]}"
    for a, b in combinations(shared, 2):
        pair = frozenset((a, b))
        if (pair in sig_a[1]) != (pair in sig_b[1]):
            return f"bond {a}-{b} present in one residue but not the other"
    return None


def check_v11_one_chemistry_per_mol_id(ctx: ValidationContext) -> list[Finding]:
    """V11: residues mapped to one SDF agree dataset-wide.

    CCD ligands agree on elements and bonds of the atoms they share, since crystals often miss
    atoms. Given ctx.ccd_path, single-residue ligands with non-CCD names must match exactly.

    Args:
        ctx: Validation context.

    Returns:
        E findings on every entry holding a disagreeing residue.
    """
    findings: list[Finding] = []
    sigs: dict[str, dict[tuple, set[str]]] = defaultdict(lambda: defaultdict(set))
    ccd_ids = read_ccd_ids(ctx.ccd_path) if ctx.ccd_path is not None else None
    exact_ids: set[str] = set()
    for entry in ctx.entries():
        try:
            npz = ctx.npz(entry)
            bonds = npz["bonds"][:, :2].tolist()
            for res in _residues(ctx, entry):
                sig = residue_signature(npz["atom_name"], npz["element"], res["rows"], bonds)
                sigs[res["mol_id"]][sig].add(entry)
                is_single_residue = res["mol_id"] == res["res_name"]
                if ccd_ids is not None and is_single_residue and res["res_name"] not in ccd_ids:
                    exact_ids.add(res["mol_id"])
        except Exception as exc:
            findings.append(
                Finding(ONE_CHEMISTRY_PER_MOL_ID, Severity.ERROR, entry, f"{type(exc).__name__}: {exc}")
            )
    for mol_id, by_sig in sorted(sigs.items()):
        reasons: dict[str, str] = {}
        for (sig_a, ents_a), (sig_b, ents_b) in combinations(by_sig.items(), 2):
            reason = _conflict(sig_a, sig_b)
            if reason is None and mol_id in exact_ids:
                reason = signature_difference(sig_a, sig_b)
            if reason:
                for entry in ents_a | ents_b:
                    reasons.setdefault(entry, reason)
        involved = sorted(reasons)
        for entry in involved:
            findings.append(
                Finding(
                    ONE_CHEMISTRY_PER_MOL_ID,
                    Severity.ERROR,
                    entry,
                    f"{mol_id}: inconsistent chemistry across residues ({reasons[entry]})",
                    {"mol_id": mol_id, "entries": involved},
                )
            )
    return findings


def _sanitizes(path) -> bool:
    """Whether RDKit reads, sanitizes and writes SMILES for the first molecule of an SDF."""
    try:
        mol = next(iter(Chem.SDMolSupplier(str(path), sanitize=True)), None)
        return mol is not None and bool(Chem.MolToSmiles(mol) or mol.GetNumAtoms() == 0)
    except Exception:
        return False


def _check_v12(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    chains = ctx.dataset_cache.structure_data[entry].chains.values()
    mol_ids = set(npz["res_name"].tolist())
    mol_ids |= {c.reference_mol_id for c in chains if c.reference_mol_id}
    findings = []
    for mol_id in sorted(mol_ids):
        path = ctx.sdf(mol_id)
        if path.exists() and not _sanitizes(path):
            findings.append(
                Finding(
                    REFERENCE_SDFS_SANITIZE,
                    Severity.ERROR,
                    entry,
                    f"{mol_id}.sdf does not sanitize in RDKit",
                    {"mol_id": mol_id},
                )
            )
    return findings


def check_v12_ref_mols_sanitize(ctx: ValidationContext) -> list[Finding]:
    """V12: every reference SDF an entry uses sanitizes and yields SMILES in RDKit.

    Args:
        ctx: Validation context.

    Returns:
        E findings, one per unsanitizable SDF per entry.
    """
    return _per_entry(ctx, REFERENCE_SDFS_SANITIZE, _check_v12)


def _source_polymer_residues(ctx: ValidationContext, entry: str) -> dict[str, set[str]]:
    """Map entity id to the ``mon_id`` set of the source CIF's ``entity_poly_seq``."""
    block = CIFFile.read(ctx.source_cif(entry)).block
    out: dict[str, set[str]] = defaultdict(set)
    for ent, mon in zip(
        _column(block, "entity_poly_seq", "entity_id"),
        _column(block, "entity_poly_seq", "mon_id"),
    ):
        out[ent].add(mon)
    return out


def _check_v18(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    allowed = _source_polymer_residues(ctx, entry)
    findings = []
    for chain in dict.fromkeys(npz["chain_id"].tolist()):
        mask = npz["chain_id"] == chain
        if int(npz["molecule_type_id"][mask][0]) not in POLYMER_MOLECULE_TYPE_IDS:
            continue
        entity = str(int(npz["entity_id"][mask][0]))
        extra = sorted(set(npz["res_name"][mask].tolist()) - allowed.get(entity, set()))
        if extra:
            label = str(npz["label_asym_id"][mask][0])
            findings.append(
                Finding(
                    LIGANDS_NOT_IN_POLYMERS,
                    Severity.ERROR,
                    entry,
                    f"polymer chain {label} contains residue(s) {extra} "
                    f"not in entity {entity} sequence",
                    {"chain": label, "residues": extra},
                )
            )
    return findings


def check_v18_ligands_not_in_polymers(ctx: ValidationContext) -> list[Finding]:
    """V18: polymer chain residues are all listed in the source CIF's entity sequence.

    Args:
        ctx: Validation context.

    Returns:
        E findings, one per offending polymer chain.
    """
    return _per_entry(ctx, LIGANDS_NOT_IN_POLYMERS, _check_v18)


def _check_v19(ctx: ValidationContext, entry: str) -> list[Finding]:
    npz = ctx.npz(entry)
    chain_id = npz["chain_id"]
    labels = {c: str(npz["label_asym_id"][chain_id == c][0]) for c in dict.fromkeys(chain_id.tolist())}
    ligand = {
        c for c in labels if int(npz["molecule_type_id"][chain_id == c][0]) == LIGAND_MOLECULE_TYPE_ID
    }
    bonds = npz["bonds"][:, :2].astype(int)
    findings = []
    cross = {
        tuple(sorted((chain_id[a], chain_id[b])))
        for a, b in bonds
        if chain_id[a] != chain_id[b] and chain_id[a] in ligand and chain_id[b] in ligand
    }
    for a, b in sorted(cross):
        findings.append(
            Finding(
                LIGAND_COPIES_SEPARATE,
                Severity.ERROR,
                entry,
                f"bond between ligand chains {labels[a]} and {labels[b]}",
                {"chains": [labels[a], labels[b]]},
            )
        )
    n = len(chain_id)
    same = bonds[chain_id[bonds[:, 0]] == chain_id[bonds[:, 1]]]
    graph = coo_matrix((np.ones(len(same)), (same[:, 0], same[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    multi: dict[str, int] = defaultdict(int)
    for c in ligand:
        comps = set(comp[chain_id == c].tolist())
        multi[c] = sum(1 for k in comps if sizes[k] > 1)
    for c, count in sorted(multi.items()):
        if count > 1:
            findings.append(
                Finding(
                    LIGAND_COPIES_SEPARATE,
                    Severity.ERROR,
                    entry,
                    f"ligand chain {labels[c]} has {count} bonded multi-atom components "
                    "(merged copies)",
                    {"chain": labels[c], "components": int(count)},
                )
            )
    return findings


def check_v19_ligand_copies_separate(ctx: ValidationContext) -> list[Finding]:
    """V19: ligand copies stay in separate chains and are not bonded to each other.

    Args:
        ctx: Validation context.

    Returns:
        E findings for cross-chain bonds and chains holding several multi-atom copies.
    """
    return _per_entry(ctx, LIGAND_COPIES_SEPARATE, _check_v19)
