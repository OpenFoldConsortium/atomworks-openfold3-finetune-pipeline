# Switch cleanup steps off

By default, the pipeline cleans structures in the same way that OpenFold3 cleans its own training data. This guide describes how to keep something that a cleanup step would otherwise remove, such as crystal waters or a short peptide.

## 1. Write a settings file

The settings file lists only the steps you want to switch off. Steps that are not listed keep their defaults.

```yaml
# cleanup.yml
remove_waters: false
remove_clashing_chains: false
```

[examples/cleanup_settings.yml](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/blob/main/examples/cleanup_settings.yml) lists every setting and what it does. A misspelled step name is reported as an error.

## 2. Run the pipeline with it

You can pass the settings file to `finetune-pipeline` or `finetune-prep` with `--cleanup-config`.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --organized-msa-dir <organized-msa-dir> \
  --cleanup-config cleanup.yml
```

## 3. Check what was used

The run prints the settings that differ from OpenFold3's defaults. Each dataset also records its settings in `<output-dir>/cleanup_settings.json`.

## Steps you cannot switch off

The model depends on the following three steps, which are always applied:

- Adding unresolved atoms.
- Removing terminal `OXT` and `OP3` atoms.
- Removing hydrogens.
