import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { bondClass, cellSize, crossBondClass, matrixCells, matrixNames } from "../js/ligands.js";
import { bondRows, formatCell } from "../js/rows.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../../../../tests/report/data/${name}`, import.meta.url)));
const parity = load("parity_data_sample.json");
const validate = load("validate_data_sample.json");

test("7axf S6T chain-3 matrix outlines exactly the CL bond", () => {
  const matrix = parity.entries["7axf"].ligands.find((l) => l.mol === "S6T" && l.ours_chains.includes("3"));
  const actual_outlined = matrixCells(matrix).filter((c) => c.outlined);
  assert.equal(actual_outlined.length, 1);
  assert.equal(actual_outlined[0].side, "ours");
  const names = matrixNames(matrix);
  assert.deepEqual([names[actual_outlined[0].i], names[actual_outlined[0].j]].sort(), ["C6", "CL"]);
});

test("a matrix with equal sides has no outlines, and ours fills the upper triangle", () => {
  const matrix = parity.entries["4x1f"].ligands[0];
  const actual_cells = matrixCells(matrix);
  assert.ok(actual_cells.every((c) => !c.outlined));
  assert.ok(actual_cells.filter((c) => c.side === "ours").every((c) => c.j > c.i));
  assert.ok(actual_cells.filter((c) => c.side === "ref").every((c) => c.j < c.i));
});

test("bond types map to classes and the cell size has a 6 px floor", () => {
  assert.deepEqual([1, 2, 5, 6, 7].map(bondClass), ["b1", "b2", "b5", "b6", "bn"]);
  assert.ok(cellSize(500) >= 6);
});

test("row helpers flatten bonds", () => {
  const bonds = { residue_bonds: [{ chain: "1" }], bond_lengths: [{ chain: "2" }, { chain: "3" }] };
  const actual_rows = bondRows(bonds);
  assert.deepEqual(actual_rows.map((r) => r.kind), ["residue bond", "bond length", "bond length"]);
  assert.equal(bondRows(validate.entries["4x1f"].bonds).length, validate.entries["4x1f"].bonds.bond_lengths.length + validate.entries["4x1f"].bonds.residue_bonds.length);
  assert.equal(formatCell(null), "");
  assert.equal(formatCell(true), "yes");
});

test("cross-bond chips keep the raw state and only the class is mapped", () => {
  assert.equal(crossBondClass("same"), "same");
  for (const state of ["ours_only", "ref_only", "type_differs"]) assert.equal(crossBondClass(state), "bad");
});
