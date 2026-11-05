# Your first dataset

In this tutorial you build an OpenFold3 training dataset from 8 structures. You then read its validation report, and can optionally run a few training steps.

By the end you will know what the pipeline produces and how to tell whether a dataset is ready. You can then run it on your own structures.

**What you need**

- This repository, installed (see [Quick start](../index.md#quick-start)).
- The Chemical Component Dictionary (CCD), about 430 MB to download. Step 2 fetches it.
- About 10 minutes. The MSAs are precomputed, so you do not wait on the ColabFold server.
- About 1 GB of disk, plus 6.3 GB for the optional training checkpoint.
- A GPU only for the optional training section.

## 1. Look at the input

The sample structures are in [examples/tutorial](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/tree/main/examples/tutorial). Each one is an mmCIF file from the PDB.

```bash
ls examples/tutorial/*.cif
```

The set mixes monomers, complexes, ligands and DNA, so you see several kinds of output. 4x1f also has a known issue in its deposited ligand, which you will inspect in the report.

## 2. Get the CCD

The pipeline uses the CCD to identify residues and ligands. Download OpenFold3's copy, then set its path for this shell.

```bash
curl -O https://openfold3-data.s3.amazonaws.com/chemical_component_dictionary.cif
export CCD_PATH=$PWD/chemical_component_dictionary.cif
```

Every command below uses this variable.

## 3. Get the precomputed MSAs

Download the archive, about 120 MB, and check it. Unpacking gives `output/msas/`.

```bash
curl -LO https://openfold3-data.s3.amazonaws.com/finetuning-tutorial/tutorial-msas.tar.gz
sha256sum -c examples/tutorial/msas.sha256
mkdir -p output
tar -xzf tutorial-msas.tar.gz -C output
```

If the download fails or the check does not say `OK`, see [Use live MSAs instead](#use-live-msas-instead).

## 4. Build the dataset

Run the pipeline. `--skip-existing` reuses the MSAs you just unpacked.

```bash
pixi run finetune-pipeline \
  --cif-dir examples/tutorial \
  --output-dir output/tutorial \
  --msa-dir output/msas \
  --ccd-path $CCD_PATH \
  --name tutorial \
  --skip-existing
```

This takes under a minute. It runs ten stages, and the validate stage should print:

```text
entries: 8 in → 8 out
```

## 5. Look at what it made

Everything is in `output/tutorial/`.

| Output | What it is |
|---|---|
| `structures/` | The cleaned structures as mmCIF. Waters and crystallization aids are gone |
| `structure_files/` | The same structures as NPZ, which OpenFold3 loads in training |
| `reference_mols/` | One SDF per ligand and residue type, such as `3WF.sdf` |
| `dataset_cache.json` | The index of every structure and chain |
| `train.yml` | A ready-to-run OpenFold3 training config |
| `cleanup_steps/` | What each cleanup step did to each structure |
| `report/index.html` | The validation report |

Open `dataset_cache.json` and find `7riv`. Its protein chain lists its MSA as `alignment_representative_id`, and its templates as `template_ids`.

Now find `1ubq`. It has no templates, because it was released in 1987 and by default a template must be older than the structure. To change this, see [Use a fixed template cutoff](../how-to/template-cutoff.md).

## 6. Read the report

Open `output/tutorial/report/index.html` in a browser. No server is needed.

- **Processing** should be green. Red here would mean the pipeline changed something it should not have.
- **Input structures** lists one structure, 4x1f. Orange means the issue is already in your input file.

Click 4x1f. Its ligand `3WF` has a carbon-carbon bond of 0.893 Å, just under the 0.9 Å limit:

```text
2:3WF501:CAE-2:3WF501:CAA 0.893 Å, too short (min 0.9 Å)
3WF chain 2: a bond too short (0.75× its lower bound, limit 0.75×)
```

This is in the deposited structure, in both copies of the ligand. The pipeline did not cause it and keeps the structure, so you decide whether it matters for your training.

This set has no **Dropped** section. On your own data, that section lists any structure left out, with the reason.

## 7. Train for one step (optional)

This needs a GPU and a working OpenFold3 install from an [`openfold-3`](https://github.com/aqlaboratory/openfold-3) checkout next to this repository. `train.yml` runs one short training step, to prove the dataset trains.

```bash
pixi run -e openfold3-cuda12 --manifest-path ../openfold-3/pixi.toml \
  run_openfold train --runner-yaml=output/tutorial/train.yml
```

It writes a 6.3 GB checkpoint to `output/tutorial/training_output/checkpoints/last.ckpt`. For a real run, raise `epoch_len` and `max_epochs` in `train.yml`.

## Next steps

Point `--cif-dir` at your own mmCIFs and run the same command without `--skip-existing`.

- Have `.pdb` files or ligands named `LIG`? See [Prepare a dataset from PDB files](../how-to/pdb-datasets.md).
- Want to hold out structures? See [Add a validation split](../how-to/validation-split.md).
- Want to keep waters or short peptides? See [Switch cleanup steps off](../how-to/cleanup-steps.md).

## Use live MSAs instead

If you skip step 3, the pipeline fetches MSAs from the public ColabFold server. Run step 4 with an empty `--msa-dir`, and without `--skip-existing`.

This took about 7 minutes in our tests. It depends on the server's queue and can take much longer.
