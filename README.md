# atomworks-openfold3-finetune-pipeline

This pipeline turns a directory of structures into an OpenFold3 training dataset, using AtomWorks. It produces cleaned structures, MSAs, templates, a training config (`train.yml`) and a validation report.

The full documentation is available at https://atomworks-openfold3-finetune-pipeline.readthedocs.io/en/latest/

## Install

Clone this repository, then install it with [pixi](https://pixi.sh).

```bash
git clone https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline.git
cd atomworks-openfold3-finetune-pipeline
pixi install
```

## Set up the CCD

The pipeline uses the Chemical Component Dictionary (CCD) to identify residues and ligands, and many of the tests also require it. The CCD is not bundled with the pipeline. The following command downloads OpenFold3's copy (430 MB) and stores its path, and needs to be run only once:

```bash
pixi run setup-finetuning
```

The CCD is downloaded to OpenFold3's cache directory, `~/.openfold3`, and its path is stored there in `ccd_path`. Every command then finds the CCD without further setup. The pipeline uses OpenFold3's copy rather than the latest wwPDB release, in which some ligand charges have changed.

- **If you already have the CCD,** you can store its path instead of downloading it again, with `pixi run setup-finetuning --ccd-path <your-ccd.cif>`.
- **If you use another cache directory,** OpenFold3 reads `$OPENFOLD_CACHE` in place of `~/.openfold3`. If it is set during setup, it must be set to the same directory in every shell where you build datasets or train. Otherwise, the pipeline and OpenFold3 look in `~/.openfold3` and do not find the CCD or the model weights.
- **To use a different CCD for a single run,** you can specify it with `--ccd-path` or `$CCD_PATH`. Either one overrides the stored path.

You can then run the tests, which are in `src/tests`. Tests that require the CCD are skipped if it is not set up.

```bash
pixi run test
```

The slower end-to-end tests can be run with `pixi run test-integration`.

## Download OpenFold3's weights

The `train.yml` that the pipeline writes fine-tunes OpenFold3's pretrained weights, so the weights must be downloaded before you build a dataset. The following command runs OpenFold3's own setup, which downloads the default weights (about 2.3 GB) to `~/.openfold3`. It needs to be run only once:

```bash
pixi run setup_openfold --non-interactive --skip-biotite-ccd-update
```

`--skip-biotite-ccd-update` leaves the copy of the CCD bundled with Biotite in this environment unchanged. To keep the weights in a different directory, or to train without them, see [Train on your dataset](docs/source/how-to/training.md).

## Run it

[examples/quickstart](examples/quickstart) holds two structures: `1stp.cif` (streptavidin with biotin) and `1ubq.cif` (ubiquitin).

```bash
pixi run finetune-pipeline \
  --cif-dir examples/quickstart \
  --name quickstart
```

The default output directory is `./output`. You can specify a different output directory with `--output-dir`.

You can view a report on the processed examples with:

```bash
open output/report/index.html
```

## Documentation

- [Home and quick start](docs/source/index.md)
- [Tutorial: your first dataset](docs/source/tutorial/first-dataset.md)
- [Known limitations](docs/source/reference/known-limitations.md)

The documentation site can be built with `pixi run -e docs docs`.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
