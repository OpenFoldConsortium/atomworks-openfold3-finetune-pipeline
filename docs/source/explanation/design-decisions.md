# Why these design decisions

This page records the decisions most likely to surprise you, and why we made them.

## CA gaps and construct sequences

**In short:** give the pipeline the whole protein sequence, not just the residues in the crystal. Otherwise real gaps look like broken chains.

### The problem

A crystal structure rarely contains every residue. Flexible loops and linkers have no modelled atoms, so they are missing from the file.

Take PXR structure x00004-1, with 291 modelled residues:

| Author numbers | What it is |
|---|---|
| 142-176 | PXR ligand-binding domain |
| 177-191 | Missing loop |
| 192-432 | PXR ligand-binding domain |
| 433-681 | Numbers never used: a linker joins PXR to SRC-1 here |
| 682-696 | SRC-1 peptide, in SRC-1's own numbering |

### Why numbering matters

Residue numbers do two jobs, and they conflict when the sequence is incomplete:

- **Indexing:** OpenFold3 uses the number to find a residue's place in the sequence and MSA. With 291 letters, residue 696 runs off the end.
- **Marking gaps:** OpenFold3 deletes a chain if consecutive residues are over 10 Å apart. It skips the pair only if the numbers jump.

Renumbering 1 to 291 fixes the first job but breaks the second. Residues 35 and 36 look like neighbours but sit 27.6 Å apart, so the chain is deleted. This removed the protein from all 110 entries in our first PXR run.

### What we do

`prepare-inputs --constructs` takes the whole sequence, including unmodelled residues. Numbers then index the real sequence and still jump at gaps, which is how RCSB entries work.

A chain with gaps and no construct is dropped with a clear reason. We prefer that to silently deleting the protein.

## Even sampling per structure

**In short:** every structure is equally likely to be drawn, and we don't cluster by similarity.

OpenFold3 clusters its training data so that near-duplicates don't dominate. Fine-tuning sets are small and often deliberately related, such as one protein with many ligands, so clustering would collapse them.

Instead, `cluster_size` is set to the number of chains and interfaces in each structure, which gives every structure the same total weight. The trade-off is that a dataset with many near-duplicates trains on them as often as anything else.

## Ligand definitions stay in the structure

**In short:** a custom ligand is defined inside its own CIF, and we never change the CCD.

`LIG` is a real CCD code for an unrelated molecule, so trusting the name would give the wrong chemistry. `prepare-inputs` gives each ligand a new code, checks that the code is not in the CCD, and writes its definition into the CIF.

Bond orders come from the SMILES you provide, never from a guess. A SMILES that does not fit the ligand's atoms drops the structure.

## Drop and carry on

**In short:** one bad structure does not stop the run, but it is always reported.

A failure in a large dataset should not cost you the rest of the run. So a structure that cannot be prepared is left out with its reason, and the report lists it.

Errors that affect everything are different. If the ligand codes collide with the CCD, the whole command stops.

