# Changelog

Big changes to the pipeline, newest first. Small fixes live in the git history.

## Unreleased

- Added `import-msas` and `finetune-pipeline --msa-sources`, to train on your own MSAs instead of querying the public ColabFold server. A sources manifest maps each MSA source name to a folder or glob of A3M files, matched to chains by query sequence. Every file is checked with OpenFold3's own readers before import; `train.yml` adds any source name OpenFold3 does not know; `validate-dataset` checks every source. `--msa-sources` also takes an MSA folder from an earlier run, to reuse its MSAs (ColabFold or imported) in any output directory without contacting ColabFold. That folder is linked, not copied, so keep it in place. Template stages are skipped with `--msa-sources` for now. See the new how-to page.
- MSA folders now hold a source index, `msa_index.yml`, written by `generate-msas` and `import-msas`. Folders from earlier runs without one still work.
- Renamed `--msa-dir` to `--organized-msa-dir` on every command. The old name still works for now, with a deprecation warning. The folder now defaults to `<output-dir>/organized_msas`: commands that write or need it use the default when the option is left out, and commands where it was optional use the default only if it exists. `finetune-pipeline` still contacts ColabFold only when `--organized-msa-dir` is passed; a rerun without it reuses an existing `<output-dir>/organized_msas`. `finetune-prep` now looks for MSAs under `<organized-msa-dir>/of3_alignments/` (it previously took the `of3_alignments` folder itself).
- Validate check IDs (V0-V21) now have named constants in `aw_of3_finetune.validate.check_ids`, such as `MSA_QUERY_ROWS` for V13, and docstrings and CLI help say what each one checks. The IDs in reports and `validate_data.json` are unchanged. The docs table now lists V0 (dataset cache readable).
- Renamed two output subdirectories. finetune-prep now writes the cleaned mmCIFs to `<output-dir>/cleaned_structures/` (was `structures/`), and finetune-pipeline writes their NPZ files to `<output-dir>/of3_compressed_structures/` (was `structure_files/`). To use an output directory from 0.1.0 with the later commands, rename both, or regenerate `train.yml` since it records the NPZ path.
- Removed `--structures-npz-dir` from finetune-pipeline, validate and finetune-report. The NPZ files are always in `<output-dir>/of3_compressed_structures/`. On finetune-pipeline the option wrote the NPZ files somewhere `train.yml` and the split stage did not look.

## 0.1.0

First public release.
