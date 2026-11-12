# Known limitations

Limits specific to `prepare-inputs` are listed in [Prepare a dataset from PDB files](../how-to/pdb-datasets.md#limitations).

## Ligand names

A ligand name outside the CCD must mean one molecule across your dataset, with the same atom names, elements and bonds in every structure. `finetune-prep` stops before the MSA step if it doesn't, and names the ligand and structures. Different poses of one molecule are fine. `prepare-inputs` gives each molecule its own name, so it never hits this.

| Limitation | Effect |
|---|---|
| **A missing atom can look like a different molecule** | If a crystal is missing atoms and the CIF does not define the ligand, the copies count as different molecules. Add the ligand's definition to the CIF, or give that copy its own name. |
| **Bond orders and charges are not compared** | Two molecules that differ only in these, such as some tautomers, can share a name and one reference molecule without an error. Give them different names. |

## CCD version

The docs download OpenFold3's copy of the CCD, whose newest entry is from May 2024 (SHA-256 `cc4867d44ec0259b52fb49e253f922536eedec12658110567f850f7f6bbb6131`). wwPDB's current CCD changes the formal charges of some ligands, and those charges are model inputs: heme's reference molecule has net charge 0 in this copy and -4 in the current one, with its iron's charge unknown. A newer CCD runs, but builds a different dataset.

## MSAs

| Limitation | Effect |
|---|---|
| **Protein MSAs only** | ColabFold searches proteins only, and `train.yml` reads protein MSAs only, so RNA and DNA chains get no MSA. Structures with RNA load and train, but expect worse RNA predictions than OpenFold3's. |
| **No cross-chain MSA pairing from ColabFold** | Complexes get no pairing signal from ColabFold MSAs. OpenFold3 finds pairing data by filename, and ColabFold's output does not match. Your own `uniprot_hits` MSAs, with species in their headers, do give pairing: see [Use your own MSAs](../how-to/user-msas.md). |
| **No templates with your own MSAs** | `--msa-sources` skips the template stages. |
| **Your own MSAs must be on local disk** | Copy them from cloud storage first. |

## Templates

| Limitation | Effect |
|---|---|
| **Old structures get few templates by default** | A template must be at least 60 days older than its structure, and this filter runs after `--max-hits`. `preprocess-templates` lists the chains that lose every hit. To change this, see [Use a fixed template cutoff](../how-to/template-cutoff.md). |

## Validation scoring

| Limitation | Effect |
|---|---|
| **Multi-residue ligands are not filtered** | Glycans and similar ligands are scored like any other ligand. OpenFold3's own validation set drops them, and ligands with a low ranking model fit. |
| **Metrics are for reading in aggregate** | Interface types are not sampled as in OpenFold3's validation set, so read validation results across the dataset, not per interface. |
