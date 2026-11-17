# Use a fixed template cutoff

By default, a template must have been released at least 60 days before the structure it is used for. Under this rule, older structures can lose all of their templates. A fixed cutoff applies the same date to every structure instead.

## 1. Run with a cutoff

You can set a fixed cutoff with the `--template-cutoff-date` option of `finetune-pipeline`. Templates released before that date are kept, regardless of each structure's own release date.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --organized-msa-dir <organized-msa-dir> \
  --name <dataset-name> \
  --template-cutoff-date 2030-01-01
```

## 2. Check the report

The report shows the cutoff as a note. The cutoff is also recorded in `<organized-msa-dir>/of3_template_data/template_cutoff_date.json`.

## Before you use it

- **Templates can be close homologs.** The model may see structures that would not be available for a real prediction. This is usually acceptable when fine-tuning on your own targets, which is why the cutoff is an option rather than the default.
- **The sequence filters must remain on.** They prevent a structure from being its own template, so this option cannot be combined with `--no-sequence-filter`.
- **It replaces `--max-release-date`.** Only one of the two can be given.
