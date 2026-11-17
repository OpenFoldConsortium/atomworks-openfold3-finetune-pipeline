# atomworks-openfold3-finetune-pipeline

Turn a directory of structures into an OpenFold3 training dataset, using AtomWorks. You get cleaned structures, MSAs, templates, a `train.yml` and a validation report.

Check out the full documentation at https://atomworks-openfold3-finetune-pipeline.readthedocs.io/en/latest/

## Install

Clone this repository, then install with [pixi](https://pixi.sh).

```bash
git clone https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline.git
cd atomworks-openfold3-finetune-pipeline
pixi install
```

## Set up the CCD

The pipeline uses the Chemical Component Dictionary (CCD) to identify residues and ligands, and many of the tests need it too. It is not bundled. Run this once to download OpenFold3's copy (430 MB) and store its path:

```bash
pixi run setup-finetuning
```

It downloads the CCD to OpenFold3's cache directory, `~/.openfold3`, and stores its path there in `ccd_path`. After that, every command finds the CCD on its own. OpenFold3's copy is used rather than wwPDB's latest, whose ligand charges have changed.

- **Already have the CCD?** Run `pixi run setup-finetuning --ccd-path <your-ccd.cif>` to store its path instead of downloading it again.
- **Using another cache directory?** OpenFold3 reads `$OPENFOLD_CACHE` in place of `~/.openfold3`. If you set it when you run setup, set it to the same directory in every shell where you build datasets or train. Otherwise the pipeline and OpenFold3 look in `~/.openfold3` and do not find the CCD or the weights.
- **Training from OpenFold3's weights?** If you have not already, run `pixi run setup_openfold`. It is OpenFold3's own setup, and downloads the model weights into the same cache directory.
- **Another CCD for one run?** Pass `--ccd-path`, or set `$CCD_PATH`. Either one overrides the stored path.

Then run the tests, which are in `src/tests`. Tests that need the CCD are skipped if it is not set up.

```bash
pixi run test
```

`pixi run test-integration` runs the slower end-to-end tests.

## Run it

[examples/quickstart](examples/quickstart) holds two structures: `1stp.cif` (streptavidin with biotin) and `1ubq.cif` (ubiquitin).

```bash
pixi run finetune-pipeline \
  --cif-dir examples/quickstart \
  --organized-msa-dir output/organized_msas \
  --name quickstart
```

It writes everything to `./output`. Pass `--output-dir` to write somewhere else.

Then open `output/report/index.html`.

## Documentation

- [Home and quick start](docs/source/index.md)
- [Tutorial: your first dataset](docs/source/tutorial/first-dataset.md)
- [Known limitations](docs/source/reference/known-limitations.md)

Build the site with `pixi run -e docs docs`.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
