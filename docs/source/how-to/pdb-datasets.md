# Prepare a dataset from PDB files

> **Prototype.** This route works, but involves several manual steps and is likely to change.

This guide applies if your structures are `.pdb` files, or have ligands under placeholder names such as `LIG`. mmCIF files with CCD ligands do not need it, and can be passed to `finetune-pipeline` directly.

`prepare-inputs` converts your files into mmCIFs that `finetune-pipeline` accepts. It is a separate command, run before the pipeline.

## 1. Choose your chains

`prepare-inputs` keeps every chain. If your crystal structures contain copies that you do not want to train on, you can remove them from the files beforehand.

## 2. Write a ligand table

The ligand table is a CSV with one row for each combination of structure and ligand residue name. The SMILES is the only source of the ligand's bond orders.

```text
structure_id,res_name,smiles,restraint_cif
x00086-1,LIG,ClC=1C=CC=2OC(=O)NC2C1,restraints/EE_1_F08.cif
x00242-1,LIG,OC(=O)CC=1C=C(Cl)C=C(Cl)C1,
```

| Column | Required | Meaning |
|---|---|---|
| `structure_id` | Yes | The input file's name without its suffix |
| `res_name` | Yes | The ligand's residue name in that file |
| `smiles` | Yes | The ligand's chemistry |
| `restraint_cif` | No | A restraint CIF from eLBOW, GRADE or AceDRG. Only its bonds are used |

## 3. Write a construct file

The construct file gives the full crystallised sequence, including unmodelled residues, in FASTA format. It is required for any chain with gaps in its residue numbering.

```text
>my_protein_construct
MKTAYIAKQRQISFVKSHFSRQ
```

## 4. Run `prepare-inputs`

```bash
pixi run prepare-inputs \
  --input-dir <dir-of-pdb-or-cif> \
  --ligands ligands.csv \
  --constructs constructs.fasta \
  --output-dir <cif-dir> \
  --release-date 2026-01-01
```

`--release-date` should be the date the structure was solved. Templates must have been released at least 60 days before this date.

## 5. Check what was dropped

A structure that cannot be prepared is left out, and the remaining structures are processed as usual. Each dropped structure prints a `WARNING: dropped <id>` line, and is listed with its reason in `<cif-dir>/prepare_dropped.json`.

Common reasons include:

- A SMILES does not fit the ligand's atoms.
- A chain has gaps and no construct fits it.
- A placeholder ligand such as `LIG` is not in the table.

## 6. Run the pipeline

Pass the `prepare-inputs` output directory to the pipeline as `--cif-dir`. The report reads the records that `prepare-inputs` wrote there.

```bash
pixi run finetune-pipeline \
  --cif-dir <cif-dir> \
  --output-dir <output-dir> \
  --organized-msa-dir <organized-msa-dir> \
  --name <dataset-name>
```

The report checks each ligand against its SMILES, and lists any release dates that were set to a default under Input structures.

## Limitations

| Limitation | Effect |
|---|---|
| **Chains with gaps need a construct** | Without one, the structure is dropped with a reason. See [CA gaps and construct sequences](../explanation/design-decisions.md#ca-gaps-and-construct-sequences). |
| **Constructs are for proteins** | Unmodelled residues in a DNA or RNA construct are written as `UNK`. |
| **`.cif` sequences are not read** | `prepare-inputs` ignores any sequence already in a `.cif` file. If the chain has gaps, you can provide its sequence with `--constructs`. |
