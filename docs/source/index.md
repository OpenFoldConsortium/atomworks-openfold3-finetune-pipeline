# atomworks-openfold3-finetune-pipeline

Turn a directory of structures into an OpenFold3 training dataset, using AtomWorks. You get cleaned structures, MSAs, templates, a `train.yml` and a validation report.

It is for structural biology groups who want to fine-tune OpenFold3 on their own structures.

## Quick start

### 1. Install

Clone this repository, then install with [pixi](https://pixi.sh).

```bash
git clone https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline.git
cd atomworks-openfold3-finetune-pipeline
pixi install
```

### 2. Get the CCD

The pipeline uses the Chemical Component Dictionary (CCD) to identify residues and ligands, and many of the tests need it too. It is not bundled, so download OpenFold3's copy (430 MB). Use this copy rather than wwPDB's latest, whose ligand charges have changed. `CCD_PATH` lasts for the current shell, so set it again in a new one.

```bash
curl -O https://openfold3-data.s3.amazonaws.com/chemical_component_dictionary.cif
export CCD_PATH=$PWD/chemical_component_dictionary.cif
```

Then run the tests, which are in `src/tests`. If `CCD_PATH` is not set, tests that need the CCD are skipped.

```bash
pixi run test
```

`pixi run test-integration` runs the slower end-to-end tests.

### 3. Build a dataset

Run the pipeline on the two bundled structures in [examples/quickstart](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/tree/main/examples/quickstart): `1stp.cif` (streptavidin with biotin) and `1ubq.cif` (ubiquitin).

```bash
pixi run finetune-pipeline \
  --cif-dir examples/quickstart \
  --output-dir output/quickstart \
  --organized-msa-dir output/quickstart_msas \
  --ccd-path $CCD_PATH \
  --name quickstart
```

Most of the run is waiting on the public ColabFold server for MSAs.

### 4. Check the result

Open `output/quickstart/report/index.html`. It says what is unusual about your structures and whether processing introduced any issues.

## Do you need `prepare-inputs`?

Most users do not. `finetune-pipeline` works directly on mmCIF files from the PDB or similar sources.

| Your input | What to run |
|---|---|
| mmCIF files whose ligands are CCD codes, or are defined in the file | `finetune-pipeline` only |
| `.pdb` files | `prepare-inputs`, then `finetune-pipeline` |
| Ligands under a placeholder name such as `LIG` | `prepare-inputs`, then `finetune-pipeline` |

`prepare-inputs` is a separate command, and `finetune-pipeline` never runs it. See [Prepare a dataset from PDB files](how-to/pdb-datasets.md).

## Where to go next

- **Learn the pipeline:** [Your first dataset](tutorial/first-dataset.md).
- **Do a specific job:** [Switch cleanup steps off](how-to/cleanup-steps.md), [prepare a dataset from PDB files](how-to/pdb-datasets.md), or [use your own MSAs](how-to/user-msas.md).
- **Look something up:** [Known limitations](reference/known-limitations.md).
- **Understand a decision:** [Why these design decisions](explanation/design-decisions.md).

```{toctree}
:hidden:
:caption: Tutorial

tutorial/first-dataset
```

```{toctree}
:hidden:
:caption: How-to guides

how-to/cleanup-steps
how-to/pdb-datasets
how-to/validation-split
how-to/template-cutoff
how-to/user-msas
```

```{toctree}
:hidden:
:caption: Reference

reference/commands
reference/config-files
reference/output-files
reference/report-checks
reference/known-limitations
```

```{toctree}
:hidden:
:caption: Explanation

explanation/how-it-works
explanation/validate
explanation/design-decisions
```
