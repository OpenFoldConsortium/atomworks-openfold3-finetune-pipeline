# What validate does

Validate runs sanity checks on your dataset before you spend GPU time. Most dataset problems don't crash a pipeline: a wrong bond order or a shifted residue trains silently and just makes a worse model.

## Two kinds of check

- **Validate checks (V1-V21)** ask whether the dataset is consistent and loads. They compare outputs against your inputs, and each other. An error stops the pipeline.
- **Report checks** ask whether the chemistry and geometry look right. They never stop the pipeline.

See [Report checks](../reference/report-checks.md) for the full list.

## What the colours mean

The key question for any finding is where it came from.

- **Orange (input):** the issue is already in your file. The pipeline kept your structure as it was. Decide whether it matters to you.
- **Red (processing):** the pipeline may have introduced it. Please report it.

Deposited structures often have small geometry issues, so orange is normal. In the tutorial, 4x1f has a ligand bond of 0.893 Å, just under the 0.9 Å limit.

## How we tested the pipeline

We compared the pipeline's output with OpenFold3's own preprocessing on PXR structures from the PDB and on the [OpenBind EV-A71 2A protease dataset](https://doi.org/10.5281/zenodo.20026660). Known differences are by design, such as ColabFold MSAs and even sampling.

## What a clean report means

The dataset passed these sanity checks: it matches your inputs, and every entry loads in OpenFold3. The checks do not review the structures themselves.
