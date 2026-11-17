# Add a validation split

Validation structures are scored during training, which shows whether the model is improving on structures it has not seen. Test structures are held out for your own evaluation after training. All other structures are used for training.

The split is made from the finished dataset, so nothing is recomputed.

## 1. Write the CSV

The CSV has one row for each structure to set aside. `id` is the CIF file name without `.cif`, and `split` is `val` or `test`.

```text
id,split
7riv,val
3u8v,val
1a3n,test
```

At least one row must be `val`. Structures that are not listed are used for training.

## 2. Split the dataset

To split the dataset during a full run, you can add `--split-csv` to `finetune-pipeline`. The CSV is checked before any stage runs, so an error in it is reported within seconds rather than after the MSA step.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --organized-msa-dir <organized-msa-dir> \
  --name <dataset-name> \
  --split-csv my_split.csv
```

To split a dataset that is already built, you can use `split-dataset`. This command also rewrites `train.yml` with default settings.

```bash
pixi run split-dataset \
  --output-dir <output-dir> \
  --split-csv my_split.csv \
  --organized-msa-dir <organized-msa-dir>
```

To change the split later, you can edit the CSV and run `split-dataset` again.

## 3. Read the split report

The command prints a report, which is also saved as `<output-dir>/split/split_report.txt`.

```text
Split: 3 train, 1 val, 0 test (4 entries)
Most similar proteins: val 7riv chain 1 <-> train 2crb chain 1, 32.0% identity
Most similar ligands:  val 7riv P06 <-> train 1a3n HEM, Tanimoto 0.05
```

A high value means that a validation structure closely resembles a training structure, so its score says little about generalisation. The report is for information only and does not stop the split.

## If something goes wrong

If the CSV has errors, the command lists every invalid row and writes nothing.

| Message | Fix |
|---|---|
| `line 3: 'xyz' is not in the dataset` | The ID must match a CIF file name |
| `line 2: '1abc' was dropped by the pipeline` | Cleanup removed that structure, so the row must be removed |
| `line 2: 'validation' is not a valid split` | The split must be `val` or `test` |
| `no val rows` | At least one structure must be marked `val` |
| `split/ was not made from the current dataset_cache.json` | The dataset was rebuilt since the split. Run `split-dataset` again |

## Things to know

- Rerunning `finetune-pipeline` on an output directory that contains `split/` requires `--split-csv` again. Alternatively, you can delete `split/`.
- Ligands on OpenFold3's metric exclusion list, such as ions and common additives, are still used in training but are not scored.
