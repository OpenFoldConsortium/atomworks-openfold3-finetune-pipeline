# Why these design decisions

This page records the design decisions that are most likely to be unexpected, and the reasons for them.

## CA gaps and construct sequences

**In short:** the pipeline needs the full protein sequence, not only the residues in the crystal structure. Otherwise, real gaps look like broken chains.

### The problem

A crystal structure rarely contains every residue. Flexible loops and linkers have no modelled atoms, so they are missing from the file.

For example, PXR structure x00004-1 has 291 modelled residues:

| Author numbers | What it is |
|---|---|
| 142-176 | PXR ligand-binding domain |
| 177-191 | Missing loop |
| 192-432 | PXR ligand-binding domain |
| 433-681 | Numbers never used: a linker joins PXR to SRC-1 here |
| 682-696 | SRC-1 peptide, in SRC-1's own numbering |

### Why numbering matters

Residue numbers serve two purposes, which conflict when the sequence is incomplete:

- **Indexing:** OpenFold3 uses the number to find a residue's position in the sequence and MSA. With a 291-residue sequence, residue 696 falls past the end.
- **Marking gaps:** OpenFold3 deletes a chain if consecutive residues are more than 10 Å apart. It skips the pair only if the residue numbers are not consecutive.

Renumbering the residues from 1 to 291 solves the first problem but causes the second. Residues 35 and 36 appear to be neighbours but are 27.6 Å apart, so the chain is deleted. In our first PXR run, this removed the protein from all 110 entries.

### What we do

`prepare-inputs --constructs` takes the full sequence, including unmodelled residues. Residue numbers then index the real sequence and still skip at gaps, as in RCSB entries.

A chain with gaps and no construct sequence is dropped with a clear reason. We prefer this to deleting the protein without notice.

## Even sampling per structure

**In short:** every structure is equally likely to be sampled, and structures are not clustered by similarity.

OpenFold3 clusters its training data so that near-duplicates do not dominate. Fine-tuning sets are small and often deliberately related, such as one protein with many ligands, so clustering would merge them into a few clusters.

Instead, `cluster_size` is set to the number of chains and interfaces in each structure, which gives every structure the same total weight. The trade-off is that a dataset with many near-duplicates trains on them as often as anything else.

## Ligand definitions stay in the structure

**In short:** a custom ligand is defined inside its own CIF, and the CCD is never modified.

`LIG` is a real CCD code for an unrelated molecule, so relying on the name would give the wrong chemistry. `prepare-inputs` gives each ligand a new code, checks that the code is not in the CCD, and writes the ligand's definition into the CIF.

Bond orders come from the SMILES you provide, and are never inferred. If a SMILES does not fit the ligand's atoms, the structure is dropped.

## Drop and continue

**In short:** a single invalid structure does not stop the run, but it is always reported.

A failure in one structure of a large dataset should not stop the rest of the run. A structure that cannot be prepared is therefore left out, and the report lists it with the reason.

Errors that affect the whole dataset are handled differently. For example, if the ligand codes collide with the CCD, the whole command stops.

