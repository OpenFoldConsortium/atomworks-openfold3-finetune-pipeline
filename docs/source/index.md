# atomworks-openfold3-finetune-pipeline

This pipeline turns a directory of structures into an OpenFold3 training dataset, using AtomWorks. It produces cleaned structures, MSAs, templates, a training config (`train.yml`) and a validation report.

It is intended for structural biology groups who want to fine-tune OpenFold3 on their own structures.

## Quick start

### 1. Install

Clone this repository, then install it with [pixi](https://pixi.sh).

```bash
git clone https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline.git
cd atomworks-openfold3-finetune-pipeline
pixi install
```

### 2. Set up the CCD

The pipeline uses the Chemical Component Dictionary (CCD) to identify residues and ligands, and many of the tests also require it. The CCD is not bundled with the pipeline. The following command downloads OpenFold3's copy (430 MB) and stores its path, and needs to be run only once:

```bash
pixi run setup-finetuning
```

The CCD is downloaded to OpenFold3's cache directory, `~/.openfold3`, and its path is stored there in `ccd_path`. Every command then finds the CCD without further setup. The pipeline uses OpenFold3's copy rather than the latest wwPDB release, in which some ligand charges have changed.

- **If you already have the CCD,** you can store its path instead of downloading it again, with `pixi run setup-finetuning --ccd-path <your-ccd.cif>`.
- **If you use another cache directory,** OpenFold3 reads `$OPENFOLD_CACHE` in place of `~/.openfold3`. If it is set during setup, it must be set to the same directory in every shell where you build datasets or train. Otherwise, the pipeline and OpenFold3 look in `~/.openfold3` and do not find the CCD or the model weights.
- **To use a different CCD for a single run,** you can specify it with `--ccd-path` or `$CCD_PATH`. Either one overrides the stored path.

You can then run the tests, which are in `src/tests`. Tests that require the CCD are skipped if it is not set up.

```bash
pixi run test
```

The slower end-to-end tests can be run with `pixi run test-integration`.

### 3. Download OpenFold3's weights

The `train.yml` that the pipeline writes fine-tunes OpenFold3's pretrained weights, so the weights must be downloaded before you build a dataset. The following command runs OpenFold3's own setup, which downloads the default weights (about 2.3 GB) to `~/.openfold3`. It needs to be run only once:

```bash
pixi run setup_openfold --non-interactive --skip-biotite-ccd-update
```

`--skip-biotite-ccd-update` leaves the copy of the CCD bundled with Biotite in this environment unchanged. To keep the weights in a different directory, or to train without them, see [Train on your dataset](how-to/training.md).

### 4. Build a dataset

The following command runs the pipeline on the two structures in [examples/quickstart](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/tree/main/examples/quickstart): `1stp.cif` (streptavidin with biotin) and `1ubq.cif` (ubiquitin).

```bash
pixi run finetune-pipeline \
  --cif-dir examples/quickstart \
  --organized-msa-dir output/organized_msas \
  --name quickstart
```

The default output directory is `./output`. You can specify a different output directory with `--output-dir`.

Most of the run time is spent waiting for MSAs from the public ColabFold server.

### 5. Check the result

The report, `output/report/index.html`, describes anything unusual about your structures and whether processing introduced any issues.

## Do you need `prepare-inputs`?

Most users do not need it. `finetune-pipeline` works directly on mmCIF files from the PDB or similar sources.

| Your input | What to run |
|---|---|
| mmCIF files whose ligands are CCD codes, or are defined in the file | `finetune-pipeline` only |
| `.pdb` files | `prepare-inputs`, then `finetune-pipeline` |
| Ligands under a placeholder name such as `LIG` | `prepare-inputs`, then `finetune-pipeline` |

`prepare-inputs` is a separate command, and `finetune-pipeline` does not run it. For details, see [Prepare a dataset from PDB files](how-to/pdb-datasets.md).

## Where to go next

- **Learn the pipeline:** [Your first dataset](tutorial/first-dataset.md).
- **Do a specific job:** [Switch cleanup steps off](how-to/cleanup-steps.md), [prepare a dataset from PDB files](how-to/pdb-datasets.md), [use your own MSAs](how-to/user-msas.md), or [train on your dataset](how-to/training.md).
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
how-to/training
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
