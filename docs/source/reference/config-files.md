# Config files

## Cleanup settings

Pass these to `--cleanup-config` as YAML or JSON. List only what you want to change. A misspelled name is an error.

| Setting | Default | What it does |
|---|---|---|
| `altloc` | `first` | Which alternate conformer to keep: `first`, `clash_aware` or `highest_occupancy` |
| `remove_waters` | `true` | Remove HOH and DOD |
| `convert_mse_to_met` | `true` | Convert selenomethionine to methionine |
| `remove_crystallization_aids` | `true` | Remove crystallization aids, in X-ray structures only |
| `remove_small_polymers` | `true` | Remove polymers of 3 residues or fewer |
| `remove_fully_unknown_polymers` | `true` | Remove polymers made only of unknown residues |
| `remove_clashing_chains` | `true` | Remove chains with over 30% of atoms within 1.7 Å of another chain |
| `remove_non_ccd_atoms` | `true` | Remove atoms not in the residue's CCD entry |
| `remove_chains_with_ca_gaps` | `true` | Remove chains whose consecutive C-alpha atoms are over 10 Å apart |
| `prefilter_bonds` | `true` | Drop inter-chain dative and polymer links, and bonds over 2.4 Å |

Three steps cannot be switched off: adding unresolved atoms, removing terminal `OXT` and `OP3` atoms, and removing hydrogens.

## `train.yml`

`generate-train-config` writes this OpenFold3 config. Its defaults prove a dataset trains, and are not training-scale settings.

| Setting | Default | What it does |
|---|---|---|
| `epoch_len` | `1` | Steps per epoch. Raise it for a real run |
| `max_epochs` | `1` | Epochs to run. Raise it for a real run |
| `batch_size` | `1` | Structures per step |
| `sample_weights` | See below | How crops are chosen within a structure |

### Sampling

Every structure is equally likely to be drawn. Within a structure, each chain and interface is equally likely to centre the crop.

| Weight | Ours | OpenFold3 default |
|---|---|---|
| `a_prot` | 1 | 3 |
| `a_nuc` | 1 | 3 |
| `a_ligand` | 1 | 1 |
| `w_chain` | 1 | 0.5 |
| `w_interface` | 0.5 | 1 |

Changing these, or `cluster_size` in `dataset_cache.json`, gives up even per-structure sampling.

### Logging

OpenFold3's config-driven logger is Weights & Biases. Add `--wandb-project` to `generate-train-config` to turn it on. Without it, no metrics are logged.

To log to CSV without Weights & Biases, run [scripts/train_with_csv_logger.py](https://github.com/OpenFoldConsortium/atomworks-openfold3-finetune-pipeline/blob/main/scripts/train_with_csv_logger.py) in place of `run_openfold train`.

## Ligand table

See [Prepare a dataset from PDB files](../how-to/pdb-datasets.md#2-write-a-ligand-table).

## Split CSV

See [Add a validation split](../how-to/validation-split.md#1-write-the-csv).
