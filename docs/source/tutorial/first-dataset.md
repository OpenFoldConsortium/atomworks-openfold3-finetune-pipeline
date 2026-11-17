# Your first dataset

In this tutorial, you build an OpenFold3 training dataset from 8 structures and read its validation report. You can also run a training step, which is optional.

By the end, you will know what the pipeline produces and how to tell whether a dataset is ready for training. You can then apply the same steps to your own structures.

**What you need**

- This repository, installed, with the CCD and OpenFold3's weights set up (see [Quick start](../index.md#quick-start), steps 1 to 3).
- About 10 minutes. The MSAs are precomputed, so there is no wait for the ColabFold server.
- About 1 GB of disk, plus 6.3 GB for the optional training checkpoint.
- A GPU, for the optional training section only.

## 1. Look at the input

The sample structures are in [examples/tutorial](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/tree/main/examples/tutorial). Each one is an mmCIF file from the PDB.

```bash
ls examples/tutorial/*.cif
```

The set includes monomers, complexes, ligands and DNA, so it produces several kinds of output. The deposited ligand in 4x1f also has a known issue, which you will inspect in the report.

## 2. Get the precomputed MSAs

The following commands download the archive (about 120 MB), verify its checksum and unpack it to `output/msas/`.

```bash
curl -LO https://openfold3-data.s3.amazonaws.com/finetuning-tutorial/tutorial-msas.tar.gz
sha256sum -c examples/tutorial/msas.sha256
mkdir -p output
tar -xzf tutorial-msas.tar.gz -C output
```

If the download fails, or the checksum is not reported as `OK`, see [Use live MSAs instead](#use-live-msas-instead).

## 3. Build the dataset

Run the pipeline as follows. `--skip-existing` reuses the MSAs you unpacked in the previous step.

```bash
pixi run finetune-pipeline \
  --cif-dir examples/tutorial \
  --output-dir output/tutorial \
  --organized-msa-dir output/msas \
  --name tutorial \
  --skip-existing
```

This takes under a minute. The pipeline runs ten stages, and the validate stage should print:

```text
entries: 8 in → 8 out
```

## 4. Look at the output

All output is in `output/tutorial/`.

| Output | What it is |
|---|---|
| `cleaned_structures/` | The cleaned structures as mmCIF, with waters and crystallization aids removed |
| `of3_compressed_structures/` | The same structures as NPZ, which OpenFold3 loads in training |
| `reference_mols/` | One SDF per ligand and residue type, such as `3WF.sdf` |
| `dataset_cache.json` | The index of every structure and chain |
| `train.yml` | A ready-to-run OpenFold3 training config |
| `cleanup_steps/` | What each cleanup step did to each structure |
| `report/index.html` | The validation report |

In `dataset_cache.json`, find the entry for `7riv`. Its protein chain lists its MSA under `alignment_representative_id`, and its templates under `template_ids`.

Next, find `1ubq`. It has no templates, because it was released in 1987, and by default a template must have been released before the structure. You can change this with a fixed cutoff date; see [Use a fixed template cutoff](../how-to/template-cutoff.md).

## 5. Read the report

Open `output/tutorial/report/index.html` in a browser. No web server is needed.

- **Processing** should be green. Red here would indicate that the pipeline changed something it should not have.
- **Input structures** lists one structure, 4x1f. Orange indicates that the issue is already present in the input file.

Select 4x1f. Its ligand `3WF` has a carbon-carbon bond of 0.893 Å, just under the 0.9 Å limit:

```text
2:3WF501:CAE-2:3WF501:CAA 0.893 Å, too short (min 0.9 Å)
3WF chain 2: a bond too short (0.75× its lower bound, limit 0.75×)
```

This bond is in the deposited structure, in both copies of the ligand. The pipeline did not cause it and keeps the structure, so you can decide whether it matters for your training.

This set has no **Dropped** section. For your own data, that section lists any structure that was left out, with the reason.

## 6. Train for one step (optional)

This section requires a GPU. `train.yml` fine-tunes the OpenFold3 weights downloaded during setup, and the following command runs a single training step, which confirms that the dataset trains.

```bash
pixi run run_openfold train --runner-yaml=output/tutorial/train.yml
```

Training writes a 6.3 GB checkpoint to `output/tutorial/training_output/checkpoints/last.ckpt`. [Train on your dataset](../how-to/training.md) describes how to choose other starting weights, resume a run and log metrics.

## Next steps

To build a dataset from your own structures, you can point `--cif-dir` at your mmCIF files and run the same command without `--skip-existing`.

- For `.pdb` files, or ligands named `LIG`, see [Prepare a dataset from PDB files](../how-to/pdb-datasets.md).
- To hold out structures for validation, see [Add a validation split](../how-to/validation-split.md).
- To keep waters or short peptides, see [Switch cleanup steps off](../how-to/cleanup-steps.md).
- To train on your dataset, see [Train on your dataset](../how-to/training.md).

## Use live MSAs instead

If you skip step 2, the pipeline can fetch MSAs from the public ColabFold server instead. To do this, run step 3 with an empty `--organized-msa-dir` and without `--skip-existing`.

This took about 7 minutes in our tests. The time depends on the server's queue, and can be much longer.
