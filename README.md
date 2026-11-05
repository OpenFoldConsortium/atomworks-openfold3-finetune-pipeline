# atomworks-openfold3-finetune-pipeline

Turn a directory of structures into an OpenFold3 training dataset, using AtomWorks. You get cleaned structures, MSAs, templates, a `train.yml` and a validation report.

## Install

Clone this repository, then install with [pixi](https://pixi.sh).

```bash
git clone https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline.git
cd atomworks-openfold3-finetune-pipeline
pixi install
```

## Get the CCD

The pipeline and many of the tests need the Chemical Component Dictionary (CCD), which is not bundled. Download OpenFold3's copy (430 MB). Use this copy rather than wwPDB's latest, whose ligand charges have changed. `CCD_PATH` lasts for the current shell, so set it again in a new one.

```bash
curl -O https://openfold3-data.s3.amazonaws.com/chemical_component_dictionary.cif
export CCD_PATH=$PWD/chemical_component_dictionary.cif
```

Then run the tests, which are in `src/tests`. If `CCD_PATH` is not set, tests that need the CCD are skipped.

```bash
pixi run test
```

`pixi run test-integration` runs the slower end-to-end tests.

## Run it

[examples/quickstart](examples/quickstart) holds two structures: `1stp.cif` (streptavidin with biotin) and `1ubq.cif` (ubiquitin).

```bash
pixi run finetune-pipeline \
  --cif-dir examples/quickstart \
  --output-dir output/quickstart \
  --msa-dir output/quickstart_msas \
  --ccd-path $CCD_PATH \
  --name quickstart
```

Then open `output/quickstart/report/index.html`.

## Documentation

- [Home and quick start](docs/source/index.md)
- [Tutorial: your first dataset](docs/source/tutorial/first-dataset.md)
- [Known limitations](docs/source/reference/known-limitations.md)

Build the site with `pixi run -e docs docs`.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
