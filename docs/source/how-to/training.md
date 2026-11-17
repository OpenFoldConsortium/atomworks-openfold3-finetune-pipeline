# Train on your dataset

`generate-train-config` writes `<output-dir>/train.yml`, and `finetune-pipeline` runs it for you. This page covers which weights training starts from, where checkpoints go, how to resume, and how to log metrics.

Run training from this repository, whose pixi environment installs OpenFold3. It needs a GPU.

```bash
pixi run run_openfold train --runner-yaml=output/my_dataset/train.yml
```

`train.yml` runs one short step, to prove the dataset trains. For a real run, raise `epoch_len` and `max_epochs` (see [Config files](../reference/config-files.md#trainyml)).

## Choose the starting weights

By default, training fine-tunes OpenFold3's pretrained weights. Two options change this, on either `generate-train-config` or `finetune-pipeline`.

| Option | Starts from |
|---|---|
| None | OpenFold3's default pretrained weights |
| `--init-checkpoint <file>` | The weights in that file: a released `.pt` file, or a `.ckpt` from an earlier run |
| `--from-scratch` | Random weights |

In the first two cases, `train.yml` loads only the model and EMA weights. The optimizer, learning-rate schedule and step count start fresh. It looks like this:

```yaml
experiment_settings:
  restart_checkpoint_path: /path/to/.openfold3/of3-ob-2025-06-30-174k.pt
  ckpt_load_settings:
    manual_checkpoint_loading: true
```

### Get the default weights

The pipeline reads the file name of OpenFold3's default weights from OpenFold3 itself. It looks for them in OpenFold3's cache, `~/.openfold3` unless `$OPENFOLD_CACHE` is set. Download them, about 2.3 GB, with:

```bash
pixi run setup_openfold
```

If the weights are missing, `generate-train-config` stops and says where it looked. `finetune-pipeline` checks before its first stage, so you do not wait on ColabFold first.

## Find the checkpoints

Training writes `<output-dir>/training_output/checkpoints/last.ckpt`, about 6.3 GB, at the end of each epoch. It holds the full training state: model and EMA weights, optimizer, learning-rate schedule and step count.

To write it somewhere else, change `experiment_settings.output_dir` in `train.yml`.

## Resume a run

Running the same `train.yml` again does **not** resume. With `manual_checkpoint_loading: true`, it loads the starting weights again and trains from epoch 0. It leaves `last.ckpt` alone and writes `last-v1.ckpt`, then `last-v2.ckpt`, and so on.

To carry on from `last.ckpt` with its optimizer, schedule and step count, edit `train.yml`:

1. Set `restart_checkpoint_path` to the checkpoint.
2. Delete `ckpt_load_settings`, or set `manual_checkpoint_loading: false`.
3. Raise `max_epochs`. The run carries on from the epoch after the checkpoint's, so with `max_epochs` unchanged there is nothing left to train.

The resumed run updates `last.ckpt` in place.

```yaml
experiment_settings:
  restart_checkpoint_path: /path/to/output/my_dataset/training_output/checkpoints/last.ckpt
  ckpt_load_settings:
    manual_checkpoint_loading: false
```

`restart_checkpoint_path: last` also works. It means the newest `last*.ckpt` in this run's `checkpoints/` folder, and random weights when there is none, as with `--from-scratch`.

To start a *new* fine-tune from a checkpoint's weights only, with a fresh schedule, pass the checkpoint to `--init-checkpoint` instead.

Regenerating `train.yml`, with `generate-train-config`, `split-dataset` or `finetune-pipeline`, overwrites these edits.

## Log metrics

OpenFold3's config-driven logger is Weights & Biases. Without it, no metrics are logged. Turn it on with these options on `generate-train-config` or `finetune-pipeline`:

| Option | What it does |
|---|---|
| `--wandb-project` | Log to this project. Turns logging on |
| `--wandb-entity` | Your wandb user or team. Defaults to the logged-in user's |
| `--wandb-name` | The run name. Defaults to the output folder's name |

The training machine needs a wandb login (`pixi run wandb login`).

To log to CSV without Weights & Biases, run [scripts/train_with_csv_logger.py](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/blob/main/scripts/train_with_csv_logger.py) in place of `run_openfold train`. It takes the same `--runner-yaml`, and writes `<output-dir>/training_output/csv_logs/version_*/metrics.csv`.

```bash
pixi run python scripts/train_with_csv_logger.py --runner-yaml=output/my_dataset/train.yml
```

## Use OpenFold3's faster kernels

`train.yml` turns off OpenFold3's DeepSpeed and cuEquivariance kernels, which this environment does not install. To turn them on, run the same command in the `openfold3-cuda12` environment of an [`openfold-3`](https://github.com/aqlaboratory/openfold-3) checkout, with `pixi run -e openfold3-cuda12 --manifest-path ../openfold-3/pixi.toml`.
