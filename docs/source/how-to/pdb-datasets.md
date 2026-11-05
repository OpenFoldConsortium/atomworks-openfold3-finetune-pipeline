# Prepare a dataset from PDB files

> **Prototype.** This route works, but has several manual steps. Expect it to change.

Use this guide if your structures are `.pdb` files, or have ligands under placeholder names such as `LIG`. If you have mmCIF files with CCD ligands, skip this and run `finetune-pipeline` directly.

`prepare-inputs` turns your files into mmCIFs that `finetune-pipeline` accepts. It is a separate step that you run first.

## 1. Choose your chains

`prepare-inputs` keeps every chain. If your crystals have copies you don't want to train on, trim the files first.

## 2. Write a ligand table

Make a CSV with one row per structure and ligand residue name. The SMILES is the only source of the ligand's bond orders.

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

Give the whole crystallised sequence, unmodelled residues included, as FASTA. You need this for any chain with gaps in its residue numbers.

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
  --ccd-path $CCD_PATH \
  --release-date 2026-01-01
```

Set `--release-date` to when the structure was solved. Templates must be 60 days older than this date.

## 5. Check what was dropped

A structure that cannot be prepared is left out, and the rest carry on. Each one prints a `WARNING: dropped <id>` line and is listed in `<cif-dir>/prepare_dropped.json` with its reason.

Common reasons:

- A SMILES does not fit the ligand's atoms.
- A chain has gaps and no construct fits it.
- A placeholder ligand such as `LIG` is not in the table.

## 6. Run the pipeline

Pass the output directory as `--cif-dir`. The report reads the records `prepare-inputs` left there.

```bash
pixi run finetune-pipeline \
  --cif-dir <cif-dir> \
  --output-dir <output-dir> \
  --msa-dir <msa-dir> \
  --ccd-path $CCD_PATH \
  --name <dataset-name>
```

The report checks each ligand against its SMILES, and lists any defaulted release dates under Input structures.

## Limitations

| Limitation | Effect |
|---|---|
| **Chains with gaps need a construct** | Without one, the structure is dropped with a reason. See [CA gaps and construct sequences](../explanation/design-decisions.md#ca-gaps-and-construct-sequences). |
| **Constructs are for proteins** | Unmodelled residues in a DNA or RNA construct are written as `UNK`. |
| **`.cif` sequences are not read** | `prepare-inputs` ignores a sequence already in a `.cif`. Pass it with `--constructs` if the chain has gaps. |
