# Known limitations

Limitations specific to `prepare-inputs` are listed in [Prepare a dataset from PDB files](../how-to/pdb-datasets.md#limitations).

## Ligand names

A ligand name that is not in the CCD must refer to a single molecule across your dataset, with the same atom names, elements and bonds in every structure. If it does not, `finetune-prep` stops before the MSA step and names the ligand and the structures involved. Different poses of the same molecule are allowed. `prepare-inputs` gives each molecule its own name, so its output is not affected.

| Limitation | Effect |
|---|---|
| **A missing atom can look like a different molecule** | If a crystal structure is missing atoms and the CIF does not define the ligand, the copies are treated as different molecules. You can add the ligand's definition to the CIF, or give that copy its own name. |
| **Bond orders and charges are not compared** | Two molecules that differ only in bond orders or charges, such as some tautomers, can share a name and a single reference molecule without an error. To keep them distinct, give them different names. |

## CCD version

The pipeline uses OpenFold3's copy of the CCD, whose newest entry is from May 2024 (SHA-256 `cc4867d44ec0259b52fb49e253f922536eedec12658110567f850f7f6bbb6131`). The current wwPDB CCD changes the formal charges of some ligands, and these charges are model inputs. For example, heme's reference molecule has a net charge of 0 in this copy and -4 in the current one, with the charge of its iron unknown. A newer CCD can be used, but it produces a different dataset.

## MSAs

| Limitation | Effect |
|---|---|
| **Protein MSAs only** | ColabFold searches proteins only, and `train.yml` reads protein MSAs only, so RNA and DNA chains have no MSA. Structures with RNA load and train, but RNA predictions are likely to be worse than OpenFold3's. |
| **No cross-chain MSA pairing from ColabFold** | Complexes receive no pairing signal from ColabFold MSAs, because OpenFold3 finds pairing data by file name and ColabFold's output does not follow that naming. Your own `uniprot_hits` MSAs, with species in their headers, do provide pairing; see [Use your own MSAs](../how-to/user-msas.md). |
| **No templates with your own MSAs** | `--msa-sources` skips the template stages. |
| **Your own MSAs must be on local disk** | The pipeline does not read MSAs from cloud storage. |

## Templates

| Limitation | Effect |
|---|---|
| **Old structures get few templates by default** | A template must have been released at least 60 days before its structure, and this filter is applied after `--max-hits`. `preprocess-templates` lists the chains that lose every hit. To change this behaviour, see [Use a fixed template cutoff](../how-to/template-cutoff.md). |

## Validation scoring

| Limitation | Effect |
|---|---|
| **Multi-residue ligands are not filtered** | Glycans and similar ligands are scored like any other ligand. OpenFold3's own validation set excludes them, as well as ligands with a low ranking model fit. |
| **Metrics are for reading in aggregate** | Interface types are not sampled as they are in OpenFold3's validation set, so validation results should be read across the whole dataset rather than per interface. |
