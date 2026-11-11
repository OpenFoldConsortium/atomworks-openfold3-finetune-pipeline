# Changelog

Big changes to the pipeline, newest first. Small fixes live in the git history.

## Unreleased

- Renamed two output subdirectories. finetune-prep now writes the cleaned mmCIFs to `<output-dir>/cleaned_structures/` (was `structures/`), and finetune-pipeline writes their NPZ files to `<output-dir>/of3_compressed_structures/` (was `structure_files/`). To use an output directory from 0.1.0 with the later commands, rename both, or regenerate `train.yml` since it records the NPZ path.

## 0.1.0

First public release.
