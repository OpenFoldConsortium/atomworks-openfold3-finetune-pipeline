# Train on your dataset

`generate-train-config` writes the OpenFold3 training config, `<output-dir>/train.yml`, and `finetune-pipeline` writes it as one of its stages. This page describes which weights training starts from, where checkpoints are written, how to resume a run, and how to log metrics.

## 1. Download OpenFold3's weights first

By default, `train.yml` starts from OpenFold3's pretrained weights, so the weights must be downloaded **before** you run `generate-train-config` or `finetune-pipeline`. Both commands write the path of the weights into `train.yml`, and stop with an error if the file is not there yet. If you followed the [Quick start](../index.md#quick-start), the weights are already downloaded.

The weights, about 2.3 GB, are downloaded with OpenFold3's own setup command, which is installed in this repository's pixi environment. We recommend the following command, which downloads the default weights to `~/.openfold3` without asking any questions, and needs to be run only once:

```bash
pixi run setup_openfold --non-interactive --skip-biotite-ccd-update
```

`--skip-biotite-ccd-update` leaves the copy of the Chemical Component Dictionary bundled with Biotite in this environment unchanged.

The pipeline looks for the weights in the OpenFold3 cache directory, which is `$OPENFOLD_CACHE` if set, and `~/.openfold3` otherwise. To keep the weights in a different directory, see [Downloading OpenFold3 model parameters](https://openfold-3.readthedocs.io/en/latest/Installation.html#downloading-openfold3-model-parameters) in the OpenFold3 documentation, and set `$OPENFOLD_CACHE` to the same directory when you build a dataset.

If the weights are missing, `generate-train-config` stops with an error that gives the path it checked. `finetune-pipeline` checks for the weights before its first stage, so a missing file is reported before any time is spent on MSAs.

The download is not needed if you start from your own weights with `--init-checkpoint`, or from random weights with `--from-scratch` (see [Choose the starting weights](#choose-the-starting-weights)).

## 2. Run training

OpenFold3 is installed in this repository's pixi environment, so training can be run from here. Training requires a GPU. The following command trains with the `train.yml` written earlier by `generate-train-config` or `finetune-pipeline`, which is in the output directory, `./output` by default:

```bash
pixi run run_openfold train --runner-yaml=output/train.yml
```

The default `train.yml` runs a single short step, which confirms that the dataset trains. For a full run, increase `epoch_len` and `max_epochs` (see [Config files](../reference/config-files.md#trainyml)).

## Choose the starting weights

By default, training fine-tunes OpenFold3's pretrained weights, downloaded in step 1 above. You can choose different starting weights when you write `train.yml`. For example, the following command writes `output/train.yml` so that training starts from the weights in `my_weights.ckpt`:

```bash
pixi run generate-train-config \
  --output-dir output \
  --organized-msa-dir output/organized_msas \
  --init-checkpoint /path/to/my_weights.ckpt
```

The starting weights are chosen with the following options, which are available on both `generate-train-config` and `finetune-pipeline`:

| Option | Training starts from |
|---|---|
| None (default) | OpenFold3's default pretrained weights |
| `--init-checkpoint <file>` | The weights in that file: a released `.pt` file, or a `.ckpt` file from an earlier run |
| `--from-scratch` | Random weights |

With the default or `--init-checkpoint`, `train.yml` loads only the model and EMA weights. The optimizer, learning-rate schedule and step count start fresh. The relevant section of `train.yml` is:

```yaml
experiment_settings:
  restart_checkpoint_path: /path/to/.openfold3/of3-ob-2025-06-30-174k.pt
  ckpt_load_settings:
    manual_checkpoint_loading: true
```

## Find the checkpoints

Training writes the checkpoint `<output-dir>/training_output/checkpoints/last.ckpt`, about 6.3 GB, at the end of each epoch. It holds the full training state: the model and EMA weights, the optimizer, the learning-rate schedule and the step count.

You can write checkpoints to a different folder by changing `experiment_settings.output_dir` in `train.yml`.

## Resume a run

Running the same `train.yml` again does **not** resume training. With `manual_checkpoint_loading: true`, the run loads the starting weights again and trains from epoch 0. It leaves `last.ckpt` unchanged and writes `last-v1.ckpt`, then `last-v2.ckpt` on the next run, and so on.

To continue from `last.ckpt` with its optimizer, schedule and step count, edit `train.yml` as follows:

1. Set `restart_checkpoint_path` to the checkpoint.
2. Set `manual_checkpoint_loading` to `false`, or remove `ckpt_load_settings`.
3. Increase `max_epochs`. The run continues from the epoch after the checkpoint's, so if `max_epochs` is unchanged there is nothing left to train.

```yaml
experiment_settings:
  restart_checkpoint_path: /path/to/output/training_output/checkpoints/last.ckpt
  ckpt_load_settings:
    manual_checkpoint_loading: false
```

The resumed run updates `last.ckpt` in place.

You can also set `restart_checkpoint_path: last`. This selects the newest `last*.ckpt` in the run's `checkpoints/` folder, or random weights if there is none, as with `--from-scratch`.

To start a new fine-tuning run from a checkpoint's weights alone, with a fresh optimizer and schedule, you can instead pass the checkpoint to `--init-checkpoint`.

Note that `generate-train-config`, `split-dataset` and `finetune-pipeline` each rewrite `train.yml`, which discards these edits.

## Log metrics

OpenFold3's config-driven logger is Weights & Biases. Without it, no metrics are logged. You can turn it on with the following options on `generate-train-config` or `finetune-pipeline`:

| Option | What it does |
|---|---|
| `--wandb-project` | Logs to this project, and turns logging on |
| `--wandb-entity` | Sets the wandb user or team. The default is the logged-in user |
| `--wandb-name` | Sets the run name. The default is the output directory's name |

The training machine must be logged in to wandb, which you can do with `pixi run wandb login`.

To log metrics to CSV without Weights & Biases, you can run [scripts/train_with_csv_logger.py](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/blob/main/scripts/train_with_csv_logger.py) in place of `run_openfold train`. It takes the same `--runner-yaml` option, and writes `<output-dir>/training_output/csv_logs/version_*/metrics.csv`.

```bash
pixi run python scripts/train_with_csv_logger.py --runner-yaml=output/train.yml
```

## Use OpenFold3's faster kernels

`train.yml` turns off OpenFold3's DeepSpeed and cuEquivariance kernels, because this environment does not install them. To install and use the kernels, see [Installation with kernels](https://openfold-3.readthedocs.io/en/latest/Installation.html#installation-with-kernels) in the OpenFold3 documentation.
