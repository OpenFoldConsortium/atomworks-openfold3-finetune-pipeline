"""Fixture paths and SMILES helpers shared by the ligand-input tests."""

from pathlib import Path

import numpy as np
from biotite.structure import AtomArray, BondType
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFBlock, CIFFile
from rdkit import Chem

from aw_of3_finetune.pipeline import prepare_augmented_cif

LIGAND_INPUTS = Path(__file__).parent / "fixtures" / "ligand_inputs"

X00242_PDB = LIGAND_INPUTS / "x00242-1" / "x00242-1.pdb"
X00242_RESTRAINT = LIGAND_INPUTS / "x00242-1" / "EE_1_L17.cif"
X00242_DEPOSITED_H_PDB = LIGAND_INPUTS / "x00242-1" / "x00242-1_deposited_h.pdb"
X00358_PDB = LIGAND_INPUTS / "x00358-1" / "x00358-1.pdb"
X00358_RESTRAINT = LIGAND_INPUTS / "x00358-1" / "DSI_1_F19.cif"
X00358_WITH_H_PDB = LIGAND_INPUTS / "x00358-1" / "x00358-1_with_h.pdb"
X00358_ALTLOC_PDB = LIGAND_INPUTS / "x00358-1" / "x00358-1_altloc.pdb"

# 3CTB/3HVL (tethered PXR LBD and SRC-1) from G142 to the end: the PXR challenge crystals' residues.
PXR_CONSTRUCT = (
    "GLTEEQRMMIRELMDAQMKTFDTTFSHFKNFRLPGVLSSGCELPESLQAPSREEAAKWSQVRKDLCSLKVSLQLRGEDGS"
    "VWNYKPPADSGGKEIFSLLPHMADMSTYMFKGIISFAKVISYFRDLPIEDQISLLKGAAFELCQLRFNTVFNAETGTWEC"
    "GRLSYCLEDTAGGFQQLLLEPMLKFHYMLKKLQLHEEEYVLMQAISLFSPDRPGVLQHRVVDQLQEQFAITLKSYIECNR"
    "PQPAHRFLFLKIMAMLTELRSINAQHTQRLLRIQDIHPFATPLMQELFGITGSGGSGGSSHSSLTERHKILHRLLQEGSP"
    "S"
)
PXR_CONSTRUCTS = (PXR_CONSTRUCT,)


def write_pxr_constructs(path: Path) -> Path:
    """Write the PXR construct as a FASTA file."""
    path.write_text(f">pxr_lbd_src1\n{PXR_CONSTRUCT}\n")
    return path


ACID_SMILES = "O=C(O)Cc1cc(Cl)cc(Cl)c1"  # x00242-1
INDOLE_SMILES = "CNC(=O)c1cccc2cc[nH]c12"  # x00358-1

# x00242-1's LIG heavy atoms in file order.
ACID_ATOM_NAMES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "O1", "O2", "CL1", "CL2"]

# Kekulé orders; the aromatic flag is dropped and RDKit re-perceives aromaticity.
_RDKIT_BOND = {
    BondType.SINGLE: Chem.BondType.SINGLE,
    BondType.DOUBLE: Chem.BondType.DOUBLE,
    BondType.TRIPLE: Chem.BondType.TRIPLE,
    BondType.AROMATIC_SINGLE: Chem.BondType.SINGLE,
    BondType.AROMATIC_DOUBLE: Chem.BondType.DOUBLE,
}
_CIF_ORDER = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
}


def ligand_atoms(pdb_path: Path, res_id: int, chain_id: str = "A") -> AtomArray:
    """Atoms of one residue from a .pdb (first altloc), hydrogens included.

    Args:
        pdb_path: Input .pdb.
        res_id: Residue number.
        chain_id: Chain id.

    Returns:
        The residue's atoms in file order.
    """
    atoms = PDBFile.read(pdb_path).get_structure(model=1)
    return atoms[(atoms.chain_id == chain_id) & (atoms.res_id == res_id)]


def _canonical_smiles(
    elements: list[str], charges: list[int], bonds: list[tuple[int, int, Chem.BondType]]
) -> str:
    mol = Chem.RWMol()
    for element, charge in zip(elements, charges, strict=True):
        atom = Chem.Atom(element.capitalize())
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)
    for i, j, order in bonds:
        mol.AddBond(int(i), int(j), order)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def definition_smiles(atom_array: AtomArray) -> str:
    """Stereo-free canonical SMILES of a ligand definition (AtomArray with Kekulé bonds).

    Args:
        atom_array: Heavy atoms with `element`, `charge` and `bonds`.

    Returns:
        Canonical SMILES without stereo.
    """
    bonds = [(i, j, _RDKIT_BOND[BondType(t)]) for i, j, t in atom_array.bonds.as_array()]
    return _canonical_smiles(list(atom_array.element), list(atom_array.charge), bonds)


def definition_smiles_from_block(block: CIFBlock, code: str) -> str:
    """Stereo-free canonical SMILES from a CIF block's chem_comp_atom/chem_comp_bond for one code.

    Args:
        block: CIF block holding the component's chem_comp_atom and chem_comp_bond rows.
        code: The component id.

    Returns:
        Canonical SMILES without stereo, heavy atoms only.
    """
    atom_cat, bond_cat = block["chem_comp_atom"], block["chem_comp_bond"]
    is_code = atom_cat["comp_id"].as_array() == code
    names = atom_cat["atom_id"].as_array()[is_code]
    elements = atom_cat["type_symbol"].as_array()[is_code]
    charges = atom_cat["charge"].as_array(int, masked_value=0)[is_code]
    heavy = np.char.upper(elements) != "H"
    index = {name: i for i, name in enumerate(names[heavy])}
    bonds = []
    bond_rows = zip(
        bond_cat["comp_id"].as_array(),
        bond_cat["atom_id_1"].as_array(),
        bond_cat["atom_id_2"].as_array(),
        bond_cat["value_order"].as_array(),
        strict=True,
    )
    for comp_id, name_1, name_2, order in bond_rows:
        if comp_id == code and name_1 in index and name_2 in index:
            bonds.append((index[name_1], index[name_2], _CIF_ORDER[order.upper()]))
    return _canonical_smiles(list(elements[heavy]), list(charges[heavy]), bonds)


def reference_smiles_by_chain(
    cif_path: Path, ligand_codes: set[str], work_dir: Path, ccd: CIFFile
) -> dict[str, tuple[str, str]]:
    """Run finetune-prep's prepare_augmented_cif; {ligand chain: (mol_id, stereo-free SMILES)}.

    Args:
        cif_path: A prepared mmCIF.
        ligand_codes: Residue names of the custom ligands to report.
        work_dir: Scratch directory for the augmented CIF and reference SDFs.
        ccd: Parsed CCD.

    Returns:
        Chain id -> (reference mol id, SMILES of the reference SDF without stereo).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    ref_mol_dir = work_dir / "reference_mols"
    structure_data, _ = prepare_augmented_cif(
        cif_path, cif_path.stem, work_dir / "augmented.cif", ccd=ccd, ref_mol_dir=ref_mol_dir
    )
    smiles_by_chain = {}
    for chain_id, chain in structure_data.chains.items():
        if chain.reference_mol_id not in ligand_codes:
            continue
        sdf = ref_mol_dir / f"{chain.reference_mol_id}.sdf"
        if not sdf.exists():
            smiles_by_chain[chain_id] = (chain.reference_mol_id, "no reference SDF written")
            continue
        mol = Chem.RemoveHs(next(Chem.SDMolSupplier(str(sdf), removeHs=False)))
        smiles_by_chain[chain_id] = (
            chain.reference_mol_id,
            Chem.MolToSmiles(mol, isomericSmiles=False),
        )
    return smiles_by_chain
