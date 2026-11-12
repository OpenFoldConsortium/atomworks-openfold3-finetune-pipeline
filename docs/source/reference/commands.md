# Commands and options

Run each command with `pixi run <command>`. `finetune-pipeline` runs the six stage commands for you, so you only need those to rerun one stage. The [pipeline diagram](../explanation/how-it-works.md#the-three-steps) shows the order they run in and which flags skip them.

## `finetune-pipeline`

Run every stage over a directory of mmCIFs.

| Option | Default | What it does |
|---|---|---|
| `--cif-dir` | Required | Directory of input mmCIF files. |
| `--output-dir` | Required | Output directory for every pipeline artifact. |
| `--organized-msa-dir` | None | MSA folder. If given, generate-msas queries ColabFold into it. If not, ColabFold is never contacted, and an existing `<output-dir>/organized_msas` from an earlier run is used. |
| `--msa-sources` | None | Use existing MSAs instead of ColabFold: a sources manifest for your own MSA files, or the MSA folder of an earlier run (see [Use your own MSAs](../how-to/user-msas.md)). Skips the template stages. |
| `--ccd-path` | Required | Path to the Chemical Component Dictionary mmCIF. |
| `--template-cache-dir` | None | Pre-built OpenFold3 template cache directory. |
| `--name` | `finetune` | Dataset name written into manifest.json. |
| `--server` | `https://api.colabfold.com` | ColabFold MSA server URL. |
| `--user-agent` | `aw-of3-finetune` | User-agent string sent with API requests. |
| `--skip-existing` | Off | Skip sequences/complexes that already have an MSA. |
| `--template-cutoff-date` | None | Keep templates released before this date (YYYY-MM-DD), whatever each structure's own release date. |
| `--templates` / `--no-templates` | On | Request template hits in generate-msas, then run fetch-templates and preprocess-templates (downloads hit structures from RCSB). |
| `--cache-output` | None | Path to write the full dataset cache JSON. |
| `--skip-inconsistent` | Off | Passed through to finetune-prep. |
| `--allow-errors` | Off | Passed through to validate. |
| `--split-csv` | None | CSV with header id,split. |
| `--skip-validation` | Off | Skip the final validate stage. |
| `--skip-report` | Off | Skip the report stage (<output-dir>/report/index.html. |
| `--skip-dataset-load` | Off | Passed through to validate. |
| `--wandb-project` | None | Passed through to generate-train-config. |
| `--wandb-entity` | None | Passed through to generate-train-config. |
| `--wandb-name` | None | Passed through to generate-train-config. |
| `--write-step-cifs` / `--no-write-step-cifs` | Off | Also write the atoms after each cleanup step as a CIF under <output-dir>/cleanup_steps/<pdb_id>/ (for viewing. |
| `--cleanup-config` | None | YAML/JSON file switching structure-cleanup steps off. |

## `prepare-inputs`

Turn `.pdb` files and a ligand table into mmCIFs. Run it first, only if you need it.

| Option | Default | What it does |
|---|---|---|
| `--input-dir` | Required | Directory of input .pdb and .cif files. |
| `--ligands` | Required | Ligand table CSV. |
| `--output-dir` | Required | Output directory. |
| `--ccd-path` | Required | Path to the Chemical Component Dictionary mmCIF. |
| `--release-date` | `2026-01-01` | YYYY-MM-DD for inputs that carry no revision date. |
| `--constructs` | None | FASTA of the whole sequences crystallised, unmodelled residues included. |

## `split-dataset`

Split a finished dataset into train, val and test.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory containing dataset_cache.json. |
| `--split-csv` | Required | CSV with header id,split. |
| `--organized-msa-dir` | `<output-dir>/organized_msas`, if it exists | MSA folder. |

## `finetune-report`

Rebuild the report of a finished dataset.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory (manifest.json, dataset_cache.json, cleanup_steps/, ...). |
| `--cif-dir` | Required | Directory of the original input mmCIF files. |
| `--ccd-path` | Required | Chemical Component Dictionary mmCIF. |
| `--ids` | None | File with one structure id per line. |
| `--report-dir` | None | Where to write the report (replaced if it holds an old one). |
| `--cache-path` | None | The dataset cache JSON (finetune-pipeline's --cache-output). |

## `validate-dataset`

Run the V checks on a dataset.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory (dataset_cache.json, of3_compressed_structures/, ...). |
| `--cif-dir` | Required | Directory of the original input mmCIF files. |
| `--organized-msa-dir` | `<output-dir>/organized_msas`, if it exists | MSA folder (holds of3_alignments/ and template_structures/). |
| `--template-cache-dir` | None | OpenFold3 template cache directory. Defaults to `<organized-msa-dir>/of3_template_data/template_cache`. |
| `--template-structure-dir` | None | Template structure directory. |
| `--ccd-path` | None | Chemical Component Dictionary mmCIF. |
| `--cache-path` | None | Dataset cache JSON. |
| `--expect-ligands` / `--no-expect-ligands` | Off | Warn about entries without any ligand (check V4). |
| `--allow-errors` / `--fail-on-errors` | Off | Drop failing entries from the dataset cache and manifest.json and exit 0, instead of exiting 1. |
| `--skip-dataset-load` | Off | Skip V21, the check that loads every entry through OF3's dataset (slow). |

## `finetune-prep`

Clean structures and build reference molecules.

| Option | Default | What it does |
|---|---|---|
| `--cif-dir` | Required | Directory of input mmCIF files. |
| `--output-dir` | Required | Output directory. |
| `--organized-msa-dir` | `<output-dir>/organized_msas`, if it exists | MSA folder. Chains with no MSA under `of3_alignments/` are reported as warnings. |
| `--template-cache-dir` | None | Pre-built OpenFold3 template cache directory. |
| `--ccd-path` | None | Path to the Chemical Component Dictionary mmCIF. |
| `--name` | `finetune` | Dataset name written into manifest.json. |
| `--skip-inconsistent` | Off | If a structure's chain metadata disagrees with its atoms, warn and leave it out of the manifest instead of failing. |
| `--write-step-cifs` / `--no-write-step-cifs` | Off | Also write the atoms after each cleanup step as a CIF under <output-dir>/cleanup_steps/<pdb_id>/ (for viewing. |
| `--cleanup-config` | None | YAML/JSON file switching structure-cleanup steps off. |

## `generate-msas`

Fetch MSAs from ColabFold.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory produced by finetune-prep (must contain manifest.json and cleaned_structures/). |
| `--organized-msa-dir` | `<output-dir>/organized_msas` | Folder to write MSA files to. |
| `--server` | `https://api.colabfold.com` | ColabFold MSA server URL. |
| `--user-agent` | `aw-of3-finetune` | User-agent string sent with API requests. |
| `--skip-existing` | Off | Skip rep_ids that already have a colabfold_main.a3m file (useful for resuming). |
| `--templates` / `--no-templates` | On | Also request template hits (PDB70) in the unpaired query, written to <organized-msa-dir>/templates/<shard>/<hash>.m8 (sequences with no hits get no file). |

## `import-msas`

Import your own MSAs instead of querying ColabFold. See [Use your own MSAs](../how-to/user-msas.md).

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory produced by finetune-prep (must contain manifest.json and cleaned_structures/). |
| `--organized-msa-dir` | `<output-dir>/organized_msas` | Folder to import the MSAs into. Must not hold ColabFold MSAs. |
| `--msa-sources` | Required | A sources manifest mapping each MSA source name to a folder or glob of your own A3M files, or the MSA folder of an earlier run. |

## `fetch-templates`

Download template structures.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory produced by finetune-prep (manifest.json + cleaned_structures/). |
| `--organized-msa-dir` | `<output-dir>/organized_msas` | MSA folder written by generate-msas (with templates on). |
| `--max-hits` | `50` | Unique PDB entries to resolve per sequence, best e-value first. |
| `--pdb-mirror-path` | None | Local PDB mirror (<mirror>/<id[1:3]>/<id>.cif.gz) to read entries from before downloading. |

## `materialize-dataset-cache`

Build `dataset_cache.json`.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory produced by finetune-prep (must contain manifest.json and cleaned_structures/). |
| `--cache-output` | None | Path to write the full dataset cache JSON. |
| `--organized-msa-dir` | `<output-dir>/organized_msas`, if it exists | MSA folder written by generate-msas. |

## `preprocess-templates`

Filter templates and add them to the dataset cache.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory (manifest.json, cleaned_structures/, and dataset_cache.json from materialize-dataset-cache). |
| `--organized-msa-dir` | `<output-dir>/organized_msas` | MSA folder after fetch-templates (template_structures/ and of3_template_alignments/). |
| `--cache-file` | None | Dataset cache to add template_ids to, in place. |
| `--min-release-date-diff` | `60` | Days a template must predate each structure's release date. |
| `--max-release-date` | None | Optional global template release-date cutoff (YYYY-MM-DD), on top of --min-release-date-diff. |
| `--template-cutoff-date` | None | Keep templates released before this date (YYYY-MM-DD), whatever each structure's own release date. |
| `--max-templates` | `20` | Templates kept per chain. |
| `--n-processes` | `1` | Worker processes. |
| `--ccd-path` | None | CCD for parsing template structures. |
| `--sequence-filter` / `--no-sequence-filter` | On | Drop template hits that fail OF3's sequence filters (AF3 SI 2.4). |
| `--max-subseq` | `0.95` | Drop hits whose aligned segment is an exact substring of the query and longer than this fraction of it. |
| `--min-align` | `0.1` | Drop hits aligned to at most this fraction of the query residues. |
| `--min-len` | `10` | Drop hits whose aligned template segment is shorter than this many residues. |

## `generate-train-config`

Write `train.yml`.

| Option | Default | What it does |
|---|---|---|
| `--output-dir` | Required | Pipeline output directory (must contain dataset_cache.json, of3_compressed_structures/, reference_mols/ -- i.e. |
| `--organized-msa-dir` | `<output-dir>/organized_msas` | MSA folder written by generate-msas and reorganized by materialize-dataset-cache. Must exist. |
| `--config-output` | None | Path to write the runner-yaml. |
| `--wandb-project` | None | Log the training run to this Weights & Biases project (OF3's only config-driven logger). |
| `--wandb-entity` | None | wandb user or team. |
| `--wandb-name` | None | wandb run name. |
