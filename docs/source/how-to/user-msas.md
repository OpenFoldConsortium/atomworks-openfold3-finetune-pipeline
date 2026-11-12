# Use your own MSAs

By default the pipeline sends every protein sequence to the public ColabFold server. If your sequences cannot leave your network, search them yourself (with a local ColabFold or MMseqs2 server, HHblits, jackhmmer, or anything else that writes A3M) and give the pipeline the results instead. ColabFold is then never contacted.

`--msa-sources` takes one of two things:

| You have | Pass to `--msa-sources` | See |
|---|---|---|
| Your own MSA files, from any search tool | A sources manifest that names each source | Steps 1 and 2 below |
| The MSA folder of an earlier `finetune-pipeline`, `generate-msas` or `import-msas` run | That folder | [Reuse MSAs from an earlier run](#reuse-msas-from-an-earlier-run) |

A folder of your own A3M files is not accepted on its own, because the pipeline cannot tell which source they are. Passing one gives an error that shows the manifest to write.

## What you need

- **Local copies of your MSAs.** The pipeline reads files on local disk only.
- **One unpaired MSA per sequence, as A3M** (`.a3m`, `.a3m.gz` or `.a3m.zst`), with the query as the first sequence. File and folder names do not matter: each file is matched to the chains in your dataset by its query sequence.

## 1. Write a sources manifest

A sources manifest is a YAML or JSON file. Each key is an MSA source name, and each value is the folder or glob holding that source's files. Relative paths are read from the manifest's own folder.

```yaml
# msa_sources.yml
uniref90_hits: searches/uniref90            # a folder, searched recursively
mgnify_hits: "searches/*/mgnify_hits.a3m"   # a glob: one file per chain folder
uniprot_hits: searches/uniprot
```

The source name becomes the file name OpenFold3 reads, so it decides how training uses the MSA:

| Source name | What training does with it |
|---|---|
| A name OpenFold3 knows, such as `uniref90_hits`, `mgnify_hits`, `cfdb_uniref30` or `colabfold_main` | Uses it as OpenFold3 does. `train.yml` keeps OpenFold3's MSA settings. |
| `uniprot_hits` or `uniprot` | Uses it to pair MSAs across the chains of a complex, by species. The headers must carry the species (below). |
| Any other name of letters, digits, `_` and `-` | Adds it to `train.yml`, after OpenFold3's own sources, with OpenFold3's largest row limit. |

For `uniprot_hits` and `uniprot`, every header after the query must look like `tr|A0A1W9RZR3|A0A1W9RZR3_9BACT/19-121`: UniProt ID, then ID and species, then the hit's range. The pipeline checks that they parse, not that the species are right.

`colabfold_paired`, `dummy` and `templates` are not allowed: they mean something else to OpenFold3 or the pipeline. Pre-paired MSAs are not supported, because OpenFold3 training pairs MSAs itself from `uniprot_hits`.

## 2. Run the pipeline

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --ccd-path $CCD_PATH \
  --msa-sources msa_sources.yml \
  --name <dataset-name>
```

Stage 2 runs `import-msas` instead of `generate-msas`. The MSAs go into `<output-dir>/organized_msas` unless you pass `--organized-msa-dir`. The template stages are skipped: templates with your own MSAs are not supported yet, so `--templates`, `--server`, `--user-agent` and `--skip-existing` are refused alongside `--msa-sources`.

To run the stages yourself, run `import-msas` after `finetune-prep`, then `generate-train-config`:

```bash
pixi run import-msas --output-dir <output-dir> --msa-sources msa_sources.yml
```

## What gets checked

Every file is checked before anything is copied, and all problems are reported together:

- **Format.** Each file goes through OpenFold3's own A3M reader, so a file training would fail on fails here. ColabFold's multi-chain (complex) A3Ms are refused.
- **Duplicates.** Two files in one source with the same query sequence are an error.
- **Reruns.** Importing the same files again changes nothing. A different MSA for a sequence already imported is an error; delete `<organized-msa-dir>/msa_sources/<source>/` to replace that source.

Files whose query matches no chain in the dataset are skipped: they are not checked or copied, so pointing the manifest at a whole MSA library is fine. They are listed as a warning in case one should have matched, usually because its query row is not the chain's SEQRES sequence. Chains with no MSA from any source are counted as a warning, not an error.

`validate-dataset` checks the query row of every MSA, and that `train.yml` reads every source.

## Reuse MSAs from an earlier run

To reuse MSAs from an earlier run, pass that run's MSA folder (its `--organized-msa-dir`, by default `<output-dir>/organized_msas`) to `--msa-sources` instead of a manifest. This works for ColabFold MSAs from `generate-msas`, including folders made before `msa_index.yml` existed, and for MSAs you imported earlier. The new run can use any output directory, and ColabFold is never contacted.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <new-output-dir> \
  --ccd-path $CCD_PATH \
  --msa-sources <earlier-output-dir>/organized_msas \
  --name <dataset-name>
```

**The earlier folder is linked, not copied.** The new run's `msa_index.yml` records the earlier folder's absolute path for each source, for example:

```yaml
colabfold_main: /data/runs/first/organized_msas/unpaired
```

The new MSA folder then holds only `msa_index.yml` and `of3_alignments/`, the plain-text MSAs OpenFold3 reads, for the chains in this dataset.

**Keep the earlier folder where it is.** `materialize-dataset-cache`, and rerunning `finetune-pipeline`, rebuild `of3_alignments/` from it. If it has moved or been deleted, they warn that the source's folder is missing and remove that source's MSAs from `of3_alignments/`. Training reads only `of3_alignments/`, so a finished `train.yml` keeps working. If you need a self-contained copy, write a manifest instead: a manifest's files are always copied in.

Templates are not carried over this way yet. To reuse ColabFold MSAs and keep templates, point `--organized-msa-dir` at the earlier folder and add `--skip-existing`, which queries ColabFold only for sequences the folder does not have.

## Remove a source

Importing adds to what is already there: a source left out of a later manifest stays in training, and `import-msas` warns about it. To remove a source, delete its line from `<organized-msa-dir>/msa_index.yml` and its `<organized-msa-dir>/msa_sources/<source>/` folder, then rerun `import-msas`. `of3_alignments/` is rebuilt from the stores each time, so do not put files there by hand.

## One kind of MSA per folder

An MSA folder holds either ColabFold's MSAs or your own, not both. `import-msas` refuses a folder `generate-msas` wrote, and the other way round. Use a separate `--organized-msa-dir` for each.
