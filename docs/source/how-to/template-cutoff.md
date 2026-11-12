# Use a fixed template cutoff

By default, a template must be at least 60 days older than the structure it serves. Old structures can lose every template to this rule. A fixed cutoff gives every structure the same date instead.

## 1. Run with a cutoff

Pass `--template-cutoff-date` to `finetune-pipeline`. Templates released before that date are kept, whatever each structure's own date.

```bash
pixi run finetune-pipeline \
  --cif-dir <dir-of-mmcifs> \
  --output-dir <output-dir> \
  --ccd-path $CCD_PATH \
  --organized-msa-dir <organized-msa-dir> \
  --name <dataset-name> \
  --template-cutoff-date 2030-01-01
```

## 2. Check the report

The report shows the cutoff as a note. It is also recorded in `<organized-msa-dir>/of3_template_data/template_cutoff_date.json`.

## Before you use it

- **Templates can be close homologs.** The model may see structures a real prediction would not have. This is usually fine when fine-tuning on your own targets, which is why it is an option and not the default.
- **The sequence filters must stay on.** They keep a structure from being its own template, so the option cannot be combined with `--no-sequence-filter`.
- **It replaces `--max-release-date`.** Give one or the other.
