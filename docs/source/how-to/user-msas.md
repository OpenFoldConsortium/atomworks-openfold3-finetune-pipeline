# Use your own MSAs

By default, the pipeline sends every protein sequence to the public ColabFold server. If your sequences cannot leave your network, you can search them yourself (with a local ColabFold or MMseqs2 server, HHblits, jackhmmer, or any other tool that writes A3M) and provide the results to the pipeline instead. ColabFold is then not contacted.

`--msa-sources` accepts either of the following:

| You have | Pass to `--msa-sources` | See |
|---|---|---|
| Your own MSA files, from any search tool | A sources manifest that names each source | Steps 1 and 2 below |
| The MSA folder of an earlier `finetune-pipeline`, `generate-msas` or `import-msas` run | That folder | [Reuse MSAs from an earlier run](#reuse-msas-from-an-earlier-run) |

A folder of your own A3M files is not accepted on its own, because the pipeline cannot tell which source the files come from. Passing such a folder gives an error that shows the manifest to write.

## What you need

- **Local copies of your MSAs.** The pipeline reads files on local disk only.
- **One unpaired MSA per sequence, in A3M format** (`.a3m`, `.a3m.gz` or `.a3m.zst`), with the query as the first sequence. File and folder names do not matter, because each file is matched to the chains in your dataset by its query sequence.

## 1. Write a sources manifest

A sources manifest is a YAML or JSON file. Each key is an MSA source name, and each value is the folder or glob holding that source's files. Relative paths are read from the manifest's own folder.

```yaml
# msa_sources.yml
uniref90_hits: searches/uniref90            # a folder, searched recursively
mgnify_hits: "searches/*/mgnify_hits.a3m"   # a glob: one file per chain folder
uniprot_hits: searches/uniprot
```

The source name becomes the file name that OpenFold3 reads, and should describe where the MSAs came from, such as the database searched. The name also determines how training uses the MSA:

| Source name | What training does with it |
|---|---|
| A name OpenFold3 knows, such as `uniref90_hits`, `mgnify_hits`, `cfdb_uniref30` or `colabfold_main` | Uses it as OpenFold3 does. `train.yml` keeps OpenFold3's MSA settings. |
| `uniprot_hits` or `uniprot` | Uses it to pair MSAs across the chains of a complex, by species. The headers must carry the species (below). |
| Any other name of letters, digits, `_` and `-` | Adds it to `train.yml`, after OpenFold3's own sources, with OpenFold3's largest row limit. |

For `uniprot_hits` and `uniprot`, every header after the query must have the form `tr|A0A1W9RZR3|A0A1W9RZR3_9BACT/19-121`: the UniProt ID, then the ID and species, then the hit's range. The pipeline checks that the headers can be parsed, but not that the species are correct.

The names `colabfold_paired`, `dummy` and `templates` are not allowed, because they have other meanings in OpenFold3 or the pipeline. Pre-paired MSAs are not supported, because OpenFold3 training pairs MSAs itself, using `uniprot_hits`.

## 2. Run the pipeline

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --msa-sources msa_sources.yml \
  --name <dataset-name>
```

Stage 2 runs `import-msas` instead of `generate-msas`. The MSAs are written to `<output-dir>/organized_msas`; you can specify a different folder with `--organized-msa-dir`. The template stages are skipped, because templates are not yet supported with your own MSAs. For the same reason, `--msa-search`, `--templates`, `--server`, `--user-agent` and `--skip-existing` cannot be combined with `--msa-sources`.

You can also run the stages individually: `import-msas` after `finetune-prep`, followed by `generate-train-config`:

```bash
pixi run import-msas --output-dir <output-dir> --msa-sources msa_sources.yml
```

## What gets checked

Every file is checked before anything is copied, and all problems are reported together.

- **Format.** Each file is read with OpenFold3's own A3M reader, so a file that would cause training to fail is reported here instead. ColabFold's multi-chain (complex) A3Ms are not accepted.
- **Duplicates.** Two files in one source with the same query sequence are reported as an error.
- **Reruns.** Importing the same files again changes nothing. A different MSA for a sequence that is already imported is an error. To replace a source, you can delete `<organized-msa-dir>/msa_sources/<source>/` and import it again.

Files whose query matches no chain in the dataset are skipped, and are not checked or copied, so the manifest can point at an entire MSA library. They are listed in a warning in case one of them should have matched; the usual cause is a query row that differs from the chain's SEQRES sequence. Chains with no MSA from any source are reported as a warning, not an error.

`validate-dataset` checks the query row of every MSA, and that `train.yml` reads every source.

## Reuse MSAs from an earlier run

To reuse MSAs from an earlier run, you can pass that run's MSA folder (its `--organized-msa-dir`, by default `<output-dir>/organized_msas`) to `--msa-sources` instead of a manifest. This works for ColabFold MSAs from `generate-msas`, including folders created before `msa_index.yml` existed, and for MSAs imported earlier. The new run can use any output directory, and ColabFold is not contacted.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <new-output-dir> \
  --msa-sources <earlier-output-dir>/organized_msas \
  --name <dataset-name>
```

**The earlier folder is linked, not copied.** The new run's `msa_index.yml` records the earlier folder's absolute path for each source, for example:

```yaml
colabfold_main: /data/runs/first/organized_msas/unpaired
```

The new MSA folder then holds only `msa_index.yml` and `of3_alignments/`, which contains the plain-text MSAs that OpenFold3 reads for the chains in this dataset.

**The earlier folder must stay where it is.** `materialize-dataset-cache`, and any rerun of `finetune-pipeline`, rebuild `of3_alignments/` from it. If the folder has been moved or deleted, they warn that the source's folder is missing and remove that source's MSAs from `of3_alignments/`. Training reads only `of3_alignments/`, so a finished `train.yml` continues to work. If you need a self-contained copy, you can write a manifest instead, since files listed in a manifest are always copied.

Templates are not yet carried over in this way. To reuse ColabFold MSAs and keep templates, you can point `--organized-msa-dir` at the earlier folder instead. The pipeline then queries ColabFold only for sequences that are not already in the folder.

## Remove a source

Importing adds to the existing sources: a source left out of a later manifest remains in training, and `import-msas` warns about it. To remove a source, delete its line from `<organized-msa-dir>/msa_index.yml` and its `<organized-msa-dir>/msa_sources/<source>/` folder, then rerun `import-msas`. `of3_alignments/` is rebuilt from the source folders each time, so files should not be added to it by hand.

## One kind of MSA per folder

An MSA folder holds either ColabFold's MSAs or your own, but not both. `import-msas` does not accept a folder written by `generate-msas`, and vice versa, so each kind requires its own `--organized-msa-dir`.
