# Changelog

Big changes to the pipeline, newest first. Small fixes live in the git history.

## Unreleased

- Renamed `--msa-dir` to `--organized-msa-dir` on every command. The old name still works for now, with a deprecation warning. The folder now defaults to `<output-dir>/organized_msas`: commands that write or need it use the default when the option is left out, and commands where it was optional use the default only if it exists. `finetune-pipeline` still contacts ColabFold only when `--organized-msa-dir` is passed; a rerun without it reuses an existing `<output-dir>/organized_msas`. `finetune-prep` now looks for MSAs under `<organized-msa-dir>/of3_alignments/` (it previously took the `of3_alignments` folder itself).
- Renamed two output subdirectories. finetune-prep now writes the cleaned mmCIFs to `<output-dir>/cleaned_structures/` (was `structures/`), and finetune-pipeline writes their NPZ files to `<output-dir>/of3_compressed_structures/` (was `structure_files/`). To use an output directory from 0.1.0 with the later commands, rename both, or regenerate `train.yml` since it records the NPZ path.
- Removed `--structures-npz-dir` from finetune-pipeline, validate and finetune-report. The NPZ files are always in `<output-dir>/of3_compressed_structures/`. On finetune-pipeline the option wrote the NPZ files somewhere `train.yml` and the split stage did not look.

## 0.1.0

First public release.
