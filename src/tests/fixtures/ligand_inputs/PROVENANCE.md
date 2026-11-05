# Ligand-input fixtures: provenance

**Source.** OpenADMET PXR challenge structure data package (Fraser Lab, UCSF; structures
unblinded 2026-07-03,
https://openadmet.org/blog/its-the-end-of-the-pxr-challenge-as-we-know-it-and-i-feel-fine/;
Zenodo 10.5281/zenodo.21786098 is the archive of that post, not of the structures). Entries `x00242-1` and `x00358-1` from
`open_challenge_set_part1_fragments`.

**Permission.** OpenADMET confirmed to the project on 2026-11-03 that the PXR examples may be used
in this test set. The post itself states no licence for the structures.

## Files

| file | content |
|---|---|
| `x00242-1/x00242-1.pdb` | carboxylic acid `O=C(O)Cc1cc(Cl)cc(Cl)c1`, one `LIG` (A 701) |
| `x00242-1/EE_1_L17.cif` | restraint CIF, verbatim (C–O bonds typed `deloc`) |
| `x00358-1/x00358-1.pdb` | indole `CNC(=O)c1cccc2cc[nH]c12`, two `LIG` copies (A 701, A 801) |
| `x00358-1/DSI_1_F19.cif` | restraint CIF, verbatim |
| `*/smiles.txt` | verbatim |
| `x00358-1/x00358-1_with_h.pdb` | derived: RDKit hydrogens (H1–H10, `AddHs(addCoords=True)`) after A 801's heavy atoms |
| `x00358-1/x00358-1_altloc.pdb` | derived: A 701 atoms C1, N1, C2 split into altlocs A/B (occupancy 0.5 each, B shifted +0.3 Å in x) |
| `x00242-1/x00242-1_deposited_h.pdb` | derived: the trimmed file plus the raw file's refinement hydrogens on A 701 (H1–H5, lines verbatim) |

## Trimming

From each raw `<entry>.pdb`, `make_fixtures.py` keeps:

- the header records (`REMARK`, incl. `REMARK 3` resolution, `CRYST1`, `SCALE`), verbatim;
- chain A only: its `LIG` residues and the protein residues with any atom within 8 Å of a
  `LIG` atom, original lines verbatim (serial numbers not renumbered);
- heavy atoms only. The raw files carry refinement hydrogens on the ligands (and on DMS); they
  are dropped so the hydrogen cases are the explicit `_with_h` and `_deposited_h` variants.

Dropped: chain B, waters, DMS, protein residues further away and the original `TER` records
(one `TER` is written after the protein).

Regenerate: `pixi run python src/tests/fixtures/ligand_inputs/make_fixtures.py --raw-dir <dir>`.
