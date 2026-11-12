# Switch cleanup steps off

By default the pipeline cleans structures the way OpenFold3 cleans its own training data. This guide shows how to keep something a cleanup step would remove, such as crystal waters or a short peptide.

## 1. Write a settings file

List only the steps you want off. Unlisted steps keep their defaults.

```yaml
# cleanup.yml
remove_waters: false
remove_clashing_chains: false
```

[examples/cleanup_settings.yml](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/blob/main/examples/cleanup_settings.yml) lists every switch with what it does. A misspelled step name is an error.

## 2. Run the pipeline with it

Pass the file with `--cleanup-config`. It works on `finetune-pipeline` and `finetune-prep`.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --ccd-path $CCD_PATH \
  --organized-msa-dir <organized-msa-dir> \
  --cleanup-config cleanup.yml
```

## 3. Check what was used

The run prints the settings that differ from OpenFold3's. Every dataset also records its settings in `<output-dir>/cleanup_settings.json`.

## Steps you cannot switch off

The model depends on these three:

- Adding unresolved atoms.
- Removing terminal `OXT` and `OP3` atoms.
- Removing hydrogens.
