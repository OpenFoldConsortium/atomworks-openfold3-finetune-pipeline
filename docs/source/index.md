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

### 2. Set up the CCD

The pipeline uses the Chemical Component Dictionary (CCD) to identify residues and ligands, and many of the tests need it too. It is not bundled. Run this once to download OpenFold3's copy (430 MB) and store its path:

```bash
pixi run setup-finetuning
```

It downloads the CCD to OpenFold3's cache directory, `~/.openfold3`, and stores its path there in `ccd_path`. After that, every command finds the CCD on its own. OpenFold3's copy is used rather than wwPDB's latest, whose ligand charges have changed.

- **Already have the CCD?** Run `pixi run setup-finetuning --ccd-path <your-ccd.cif>` to store its path instead of downloading it again.
- **Using another cache directory?** OpenFold3 reads `$OPENFOLD_CACHE` in place of `~/.openfold3`. If you set it when you run setup, set it to the same directory in every shell where you build datasets or train. Otherwise the pipeline and OpenFold3 look in `~/.openfold3` and do not find the CCD or the weights.
- **Training from OpenFold3's weights?** If you have not already, run `pixi run setup_openfold`. It is OpenFold3's own setup, and downloads the model weights into the same cache directory.
- **Another CCD for one run?** Pass `--ccd-path`, or set `$CCD_PATH`. Either one overrides the stored path.

Then run the tests, which are in `src/tests`. Tests that need the CCD are skipped if it is not set up.

```bash
pixi run test
```

`pixi run test-integration` runs the slower end-to-end tests.

### 3. Build a dataset

Run the pipeline on the two bundled structures in [examples/quickstart](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/tree/main/examples/quickstart): `1stp.cif` (streptavidin with biotin) and `1ubq.cif` (ubiquitin).

```bash
pixi run finetune-pipeline \
  --cif-dir examples/quickstart \
  --organized-msa-dir output/organized_msas \
  --name quickstart
```

It writes everything to `./output`. Pass `--output-dir` to write somewhere else.

Most of the run is waiting on the public ColabFold server for MSAs.

### 4. Check the result

Open `output/report/index.html`. It says what is unusual about your structures and whether processing introduced any issues.

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
