"""Named IDs for the validate stage's checks.

The values are the stable check IDs written to reports and to the ``check`` field of
``validate_data.json``; the names say what each check tests. There is no V20.
"""

from __future__ import annotations

DATASET_CACHE_READABLE = "V0"  # the dataset cache can be read at all
CHAINS_ACCOUNTED = "V1"  # every source chain is kept or dropped for a known reason
POLYMER_SEQUENCES = "V2"  # polymer sequences match the source
ATOMS_MATCH_SOURCE = "V3"  # output atoms match the source CIF
HAS_LIGAND = "V4"  # the entry has a ligand beyond water and ions
FINITE_COORDINATES = "V5"  # coordinates are finite and present
CACHE_MATCHES_STRUCTURE = "V6"  # dataset-cache chain metadata matches the structure
RESIDUE_NUMBERING = "V7"  # residue numbering matches the sequence
REFERENCE_SDFS_EXIST = "V8"  # every ligand has a reference SDF
SDF_ELEMENTS_MATCH = "V9"  # reference SDF elements match the crystal
SDF_ATOM_NAMES_MATCH = "V10"  # reference SDF atom names match the crystal
ONE_CHEMISTRY_PER_MOL_ID = "V11"  # residues sharing a reference SDF agree dataset-wide
REFERENCE_SDFS_SANITIZE = "V12"  # reference SDFs sanitize in RDKit
MSA_QUERY_ROWS = "V13"  # MSA query rows equal their chain sequences
REP_IDS_BY_SEQUENCE = "V14"  # representative IDs follow the chain sequence
TEMPLATES_RESOLVE = "V15"  # template IDs resolve to structure files
TEMPLATE_DATES = "V16"  # templates predate their entry
RELEASE_DATE = "V17"  # the entry has a real release date
LIGANDS_NOT_IN_POLYMERS = "V18"  # ligands are not inside polymer chains
LIGAND_COPIES_SEPARATE = "V19"  # ligand copies stay in separate chains
DATASET_LOADS = "V21"  # every entry loads through OpenFold3's dataset
