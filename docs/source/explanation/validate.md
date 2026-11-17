# What validate does

Validation runs sanity checks on your dataset before any GPU time is spent on training. Most dataset problems do not cause a pipeline to fail: a wrong bond order or a shifted residue trains without any error, and produces a worse model.

## Two kinds of check

- **Validate checks (V0-V21)** ask whether the dataset is consistent and loads, for example V3 (atoms match the source CIF) and V21 (every entry loads through OpenFold3's dataset). They compare outputs against your inputs and against each other. An error stops the pipeline.
- **Report checks** ask whether the chemistry and geometry look correct. They do not stop the pipeline.

See [Report checks](../reference/report-checks.md) for the full list.

## What the colours mean

The key question for any finding is where it originated.

- **Orange (input):** the issue is already present in your file, and the pipeline kept your structure as it was. You can decide whether it matters for your use.
- **Red (processing):** the pipeline may have introduced the issue. Please report it.

Deposited structures often have small geometry issues, so orange findings are common. In the tutorial, 4x1f has a ligand bond of 0.893 Å, just under the 0.9 Å limit.

## How we tested the pipeline

We compared the pipeline's output with OpenFold3's own preprocessing on PXR structures from the PDB and on the [OpenBind EV-A71 2A protease dataset](https://doi.org/10.5281/zenodo.20026660). The known differences are intentional, such as the use of ColabFold MSAs and even sampling.

## What a clean report means

A clean report means that the dataset passed these sanity checks: it matches your inputs, and every entry loads in OpenFold3. The checks do not assess the quality of the structures themselves.
