# Add a validation split

Validation structures are scored during training, which shows whether the model is improving on structures it has not seen. Test structures are held out for your own evaluation afterwards. Everything else trains.

The split is made from the finished dataset, so nothing is recomputed.

## 1. Write the CSV

Add one row per structure to set aside. `id` is the CIF file name without `.cif`, and `split` is `val` or `test`.

```text
id,split
7riv,val
3u8v,val
1a3n,test
```

At least one row must be `val`. Structures you don't list train.

## 2. Split the dataset

To split during a full run, add `--split-csv`. A typo in the CSV then fails in seconds, not after the MSA step.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --organized-msa-dir <organized-msa-dir> \
  --ccd-path $CCD_PATH \
  --name <dataset-name> \
  --split-csv my_split.csv
```

To split a dataset you already built, use `split-dataset`. This also rewrites `train.yml` with default settings.

```bash
pixi run split-dataset \
  --output-dir <output-dir> \
  --split-csv my_split.csv \
  --organized-msa-dir <organized-msa-dir>
```

To change the split later, edit the CSV and run `split-dataset` again.

## 3. Read the split report

The command prints a report, also saved as `<output-dir>/split/split_report.txt`.

```text
Split: 3 train, 1 val, 0 test (4 entries)
Most similar proteins: val 7riv chain 1 <-> train 2crb chain 1, 32.0% identity
Most similar ligands:  val 7riv P06 <-> train 1a3n HEM, Tanimoto 0.05
```

A high number means a validation structure looks like a training one, so its score says little about generalisation. The report is information only and never stops the split.

## If something goes wrong

A bad CSV lists every bad row and writes nothing.

| Message | Fix |
|---|---|
| `line 3: 'xyz' is not in the dataset` | The ID must match a CIF file name |
| `line 2: '1abc' was dropped by the pipeline` | Cleanup removed that structure, so remove the row |
| `line 2: 'validation' is not a valid split` | Write `val` or `test` |
| `no val rows` | Mark at least one structure `val` |
| `split/ was not made from the current dataset_cache.json` | The dataset was rebuilt. Run `split-dataset` again |

## Things to know

- Rerunning `finetune-pipeline` into a folder that has `split/` needs `--split-csv` again, or delete `split/`.
- Ligands on OpenFold3's metric exclusion list, such as ions and common additives, still train but are not scored.
