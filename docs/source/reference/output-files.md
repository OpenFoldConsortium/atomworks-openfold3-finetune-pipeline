# Output files

## `<output-dir>`

| Path | What it is |
|---|---|
| `cleaned_structures/` | Cleaned structures as mmCIF |
| `of3_compressed_structures/` | The same structures as NPZ, which OpenFold3 loads in training |
| `reference_mols/` | One SDF per ligand and residue type |
| `dataset_cache.json` | Index of every structure, chain and interface |
| `manifest.json` | The structures in the dataset |
| `prep_dropped.json` | Structures `finetune-prep` left out, with reasons |
| `cleanup_settings.json` | The cleanup settings used |
| `cleanup_steps/<id>.json` | What each cleanup step did to a structure |
| `validation.json` | Findings of the V checks |
| `train.yml` | OpenFold3 training config. Written only with `--msa-dir` |
| `report/index.html` | The validation report |
| `split/` | Train and validation caches, with `--split-csv` |
| `training_output/` | Checkpoints, once you train |

## `<msa-dir>`

| Path | What it is |
|---|---|
| `unpaired/` | MSAs per sequence from ColabFold |
| `paired/` | Paired MSAs per complex from ColabFold |
| `templates/` | Raw template hits from ColabFold |
| `template_structures/` | Downloaded template structures |
| `of3_alignments/` | MSAs laid out for OpenFold3 |
| `of3_template_alignments/` | Template hits laid out for OpenFold3 |
| `of3_template_data/` | The template cache and filter records |

### `of3_template_data/`

| Path | What it is |
|---|---|
| `template_cache/` | OpenFold3's template cache |
| `template_date_filter.tsv` | Hits and kept templates per chain |
| `template_sequence_filter.tsv` | Hits the sequence filters dropped |
| `template_cutoff_date.json` | The cutoff, with `--template-cutoff-date` |

## `prepare-inputs` output

| Path | What it is |
|---|---|
| `<id>.cif` | The prepared structure |
| `prepare_dropped.json` | Structures left out, with reasons |
| `ligands.json` | Each new ligand code, its SMILES and structures |
| `prepare_inputs.json` | The release date used and which structures got it |
