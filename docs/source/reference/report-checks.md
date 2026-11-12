# Report checks

## How findings are labelled

| Label | Meaning |
|---|---|
| **Processing** (red) | Not in your input. Likely a pipeline bug, so please report it |
| **Input structures** (orange) | Already in your input. The structure is kept |
| **Not checked** (grey) | A check could not run on this structure |
| **Not run** (grey) | A check was skipped for the whole dataset |

## Validate checks

These run in the `validate` stage. An error stops the pipeline with exit code 1, unless you pass `--allow-errors`, which drops the entry.

| Check | Area | What it confirms |
|---|---|---|
| V0 | Dataset | The dataset cache can be read. If it cannot, no other check has entries to look at |
| V1 | Structure | Every source chain is in the output, or dropped for a known reason |
| V2 | Structure | Each output polymer sequence equals the source's |
| V3 | Structure | Every resolved heavy atom has a same-named source atom within 0.001 Å |
| V4 | Ligands | Warns if an entry has no ligand beyond water and ions, with `--expect-ligands` |
| V5 | Structure | NPZ coordinates are usable: NaN only for unresolved atoms |
| V6 | Structure | Dataset cache chain data agrees with the structure's atoms |
| V7 | Structure | Residue numbers fit the sequence and match its letters |
| V8 | Ligands | Every ligand residue has a reference SDF |
| V9 | Ligands | Crystal heavy atoms match the SDF's elements |
| V10 | Ligands | Crystal heavy atoms have names the SDF knows |
| V11 | Ligands | Residues sharing an SDF agree on elements and bonds. Ligands with names outside the CCD must match exactly |
| V12 | Ligands | Every reference SDF sanitizes in RDKit |
| V13 | MSAs | Each chain's MSA query row equals its sequence, and `train.yml` reads every MSA source |
| V14 | MSAs | `alignment_representative_id` is the sequence's hash |
| V15 | Templates | Every template ID resolves to a structure file |
| V16 | Templates | Each template predates its structure, or the fixed cutoff |
| V17 | Dates | Each entry has a real release date |
| V18 | Ligands | Polymer residues are all in the source's sequence |
| V19 | Ligands | Ligand copies stay in separate chains and are not bonded |
| V21 | Load | Each entry loads through OpenFold3's dataset class. Skip with `--skip-dataset-load` |

## Report checks

These run in the `report` stage. They never stop the pipeline.

| Check | What it looks at |
|---|---|
| `cleanup_audit` | Every input atom is accounted for by a cleanup step |
| `chain_placement` | Each output residue sits on the input residue it came from |
| `covalent_links` | Every covalent link in the input is kept, or removed for a known reason |
| `bond_lengths` | Every bond between resolved atoms is 0.9-2.4 Å |
| `residue_bonds` | Each residue's bonds match the CCD, or the input's own definition if it is not in the CCD |
| `ligand_identity` | Each ligand's InChI matches its CCD molecule, with stereo also checked against the input |
| `ligand_chemistry` | Each ligand sanitizes in RDKit and is connected |
| `ligand_geometry` | Bond lengths, angles, internal clashes and flatness of each ligand |
| `ligand_contacts` | Minimum distance and overlap of each ligand with nearby protein atoms |
| `ligand_smiles` | Each `prepare-inputs` ligand's reference SDF is its given SMILES |
