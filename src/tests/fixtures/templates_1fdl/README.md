# 1fdl template fixture

Offline inputs for the template stages (`fetch-templates`, `preprocess-templates`) on
1fdl (D1.3 Fab light chain, heavy chain, hen egg-white lysozyme; deposited 1990-08-27).

- `raw_m8/<hash>.m8`: rows of the real ColabFold pdb70 hit tables for the three 1fdl
  chains, unmodified, so hits still use author chain IDs (`1fdl_L`, `3hfm_H`). Named
  by atomworks' sequence hash, as `make_msas_colabfold_server_batch` writes them.
- `pdb_mirror/<id[1:3]>/<id>.cif.gz`: the hit structures from RCSB, in the local PDB
  mirror layout `fetch-templates --pdb-mirror-path` reads. `1fdl.cif.gz` doubles as
  the query structure.

| Chain | Hash | Hits (row in full table, release date) |
|---|---|---|
| light | `6c014ff3a02` | 1fdl_L (1, 1991-10-15), 1qbl_L (3, 1998-12-02) |
| heavy | `48fd76faca2` | 1fdl_H (1, 1991-10-15), 1qbl_H (5, 1998-12-02), 3hfm_H (62, 1989-07-12) |
| lysozyme | `eaf1f8d7f5b` | 1ior_A (1, 2001-04-11), 7ynv_A (5, 2022-09-21), 5lyz_A (7, 1977-04-12) |

Same hits and structures as openfold-3's
`tests/test_data/template_preprocessing/1fdl/`, which has label chain IDs instead.
