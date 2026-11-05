"""Regenerate the ligand-input fixtures from the raw PXR part1 entries.

Writes trimmed `<entry>/<entry>.pdb` files (chain A: LIG copies + protein residues within
8 A, heavy atoms only), copies the restraint CIF and smiles.txt, and derives the x00358-1
hydrogen and altloc variants and the x00242-1 variant that keeps the deposited ligand
hydrogens. See PROVENANCE.md.

Usage: pixi run python src/tests/fixtures/ligand_inputs/make_fixtures.py --raw-dir DIR
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

OUT_DIR = Path(__file__).parent
ENTRIES = {"x00242-1": "EE_1_L17.cif", "x00358-1": "DSI_1_F19.cif"}
CHAIN = "A"
LIGAND = "LIG"
CUTOFF = 8.0
HEADER_RECORDS = ("REMARK", "CRYST1", "SCALE")

# Derived variants of x00358-1 (Review Focus 1-2).
H_RES_ID = 801  # the second LIG copy gets RDKit hydrogens
ALTLOC_RES_ID = 701
ALTLOC_ATOMS = ("C1", "N1", "C2")
ALTLOC_SHIFT = np.array([0.3, 0.0, 0.0])

# Derived variant of x00242-1: the raw file's refinement hydrogens on LIG, kept.
DEPOSITED_H_ENTRY = "x00242-1"


def _is_atom_line(line: str) -> bool:
    return line.startswith(("ATOM", "HETATM"))


def _line_key(line: str) -> tuple[str, int]:
    return line[21], int(line[22:26])


def _line_element(line: str) -> str:
    return line[76:78].strip().upper()


def _residues_to_keep(pdb_path: Path) -> set[tuple[str, int]]:
    """Chain-A LIG residues plus protein residues with any atom within CUTOFF of them.

    Args:
        pdb_path: Raw entry .pdb.

    Returns:
        (chain_id, res_id) pairs to keep.
    """
    atoms = PDBFile.read(pdb_path).get_structure(model=1)
    atoms = atoms[atoms.chain_id == CHAIN]
    is_ligand = atoms.res_name == LIGAND
    ligand_coords = atoms.coord[is_ligand]
    protein = atoms[~atoms.hetero]
    distances = np.linalg.norm(protein.coord[:, None, :] - ligand_coords[None, :, :], axis=-1)
    near = distances.min(axis=1) <= CUTOFF
    keep = {(CHAIN, int(r)) for r in protein.res_id[near]}
    keep |= {(CHAIN, int(r)) for r in atoms.res_id[is_ligand]}
    return keep


def trim_entry(raw_pdb: Path, out_pdb: Path) -> None:
    """Write the header and the kept residues' heavy atoms, original lines verbatim.

    Args:
        raw_pdb: Raw entry .pdb.
        out_pdb: Trimmed .pdb to write.
    """
    keep = _residues_to_keep(raw_pdb)
    lines = raw_pdb.read_text().splitlines()
    header = [line for line in lines if line.startswith(HEADER_RECORDS)]
    protein, ligands = [], []
    for line in lines:
        if not _is_atom_line(line) or _line_key(line) not in keep or _line_element(line) == "H":
            continue
        (ligands if line.startswith("HETATM") else protein).append(line)
    out_pdb.write_text("\n".join([*header, *protein, "TER", *ligands, "END"]) + "\n")


def _residue_mol(lines: list[str], smiles: str) -> Chem.Mol:
    """RDKit molecule with explicit hydrogens for one residue's heavy-atom lines.

    Args:
        lines: HETATM lines of the residue (heavy atoms).
        smiles: The ligand's SMILES.

    Returns:
        Molecule whose first atoms are the input atoms in order, then added H with coords.
    """
    mol = Chem.MolFromPDBBlock("\n".join(lines), removeHs=False, proximityBonding=True)
    mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), mol)
    return Chem.AddHs(mol, addCoords=True)


def write_with_h(trimmed_pdb: Path, smiles: str, out_pdb: Path) -> None:
    """Add RDKit hydrogens (named H1, H2, ...) after the second LIG copy's heavy atoms.

    Args:
        trimmed_pdb: Trimmed x00358-1.pdb.
        smiles: The ligand's SMILES.
        out_pdb: Variant to write.
    """
    lines = trimmed_pdb.read_text().splitlines()
    residue = [
        line for line in lines if _is_atom_line(line) and _line_key(line) == (CHAIN, H_RES_ID)
    ]
    mol = _residue_mol(residue, smiles)
    conformer = mol.GetConformer()
    template = residue[-1]
    last_serial = int(template[6:11])
    h_lines = []
    for number, atom in enumerate(list(mol.GetAtoms())[len(residue) :], start=1):
        position: Point3D = conformer.GetAtomPosition(atom.GetIdx())
        name = f" H{number}"
        h_lines.append(
            f"HETATM{last_serial + number:5d} {name:<4}{template[16:30]}"
            f"{position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"{template[54:76]} H  "
        )
    last = lines.index(residue[-1])
    out_pdb.write_text("\n".join([*lines[: last + 1], *h_lines, *lines[last + 1 :]]) + "\n")


def write_altloc(trimmed_pdb: Path, out_pdb: Path) -> None:
    """Split three atoms of the first LIG copy into altlocs A/B (B shifted, 0.5 occupancy each).

    Args:
        trimmed_pdb: Trimmed x00358-1.pdb.
        out_pdb: Variant to write.
    """
    out = []
    for line in trimmed_pdb.read_text().splitlines():
        split = (
            _is_atom_line(line)
            and _line_key(line) == (CHAIN, ALTLOC_RES_ID)
            and line[12:16].strip() in ALTLOC_ATOMS
        )
        if not split:
            out.append(line)
            continue
        coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        shifted = coord + ALTLOC_SHIFT
        out.append(f"{line[:16]}A{line[17:54]}{0.5:6.2f}{line[60:]}")
        out.append(
            f"{line[:16]}B{line[17:30]}{shifted[0]:8.3f}{shifted[1]:8.3f}{shifted[2]:8.3f}"
            f"{0.5:6.2f}{line[60:]}"
        )
    out_pdb.write_text("\n".join(out) + "\n")


def write_deposited_h(raw_pdb: Path, trimmed_pdb: Path, out_pdb: Path) -> None:
    """Insert the raw file's chain-A LIG hydrogen lines after each copy's heavy atoms.

    Args:
        raw_pdb: Raw entry .pdb, with refinement hydrogens.
        trimmed_pdb: Trimmed .pdb of the same entry.
        out_pdb: Variant to write.
    """
    raw_h = [
        line
        for line in raw_pdb.read_text().splitlines()
        if _is_atom_line(line)
        and line[17:20] == LIGAND
        and line[21] == CHAIN
        and _line_element(line) == "H"
    ]
    lines = trimmed_pdb.read_text().splitlines()
    last_line_of = {
        _line_key(line): i
        for i, line in enumerate(lines)
        if _is_atom_line(line) and line[17:20] == LIGAND
    }
    out = []
    for i, line in enumerate(lines):
        out.append(line)
        if _is_atom_line(line) and last_line_of.get(_line_key(line)) == i:
            out.extend(h for h in raw_h if _line_key(h) == _line_key(line))
    out_pdb.write_text("\n".join(out) + "\n")


def main() -> None:
    """Regenerate every fixture file from the raw entries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--synthetic", action="store_true", help="Synthetic look-alikes (not implemented yet)."
    )
    args = parser.parse_args()
    if args.synthetic:
        raise NotImplementedError(
            "--synthetic is a placeholder for when the PXR data cannot be redistributed: "
            "build 1ubq-fixture residues + RDKit-embedded ligands with the same atom names."
        )
    for entry, restraint in ENTRIES.items():
        raw, out = args.raw_dir / entry, args.out_dir / entry
        out.mkdir(parents=True, exist_ok=True)
        trim_entry(raw / f"{entry}.pdb", out / f"{entry}.pdb")
        shutil.copy(raw / restraint, out / restraint)
        shutil.copy(raw / "smiles.txt", out / "smiles.txt")
    x00358 = args.out_dir / "x00358-1"
    smiles = (x00358 / "smiles.txt").read_text().strip()
    write_with_h(x00358 / "x00358-1.pdb", smiles, x00358 / "x00358-1_with_h.pdb")
    write_altloc(x00358 / "x00358-1.pdb", x00358 / "x00358-1_altloc.pdb")
    entry = DEPOSITED_H_ENTRY
    write_deposited_h(
        args.raw_dir / entry / f"{entry}.pdb",
        args.out_dir / entry / f"{entry}.pdb",
        args.out_dir / entry / f"{entry}_deposited_h.pdb",
    )


if __name__ == "__main__":
    main()
