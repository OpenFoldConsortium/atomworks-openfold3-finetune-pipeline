import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { HOVER_LABEL_MAX, atomLayers, hoverText, issuePoints, shortLabel, loadStepData, phaseRows, stepRange } from "../js/steps3d.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../../../../tests/report/data/validate_3d/${name}.json`, import.meta.url)));
const view = load("4x1f");

test("phaseRows on 4x1f lists the four phases with Assembly creating 2342 atoms", () => {
  const actual_rows = phaseRows(view);
  const expected_names = ["Parse", "Assembly", "Cleanup filters", "Final trim"];
  assert.deepEqual(actual_rows.map((r) => r.name), expected_names);
  const actual_assembly = actual_rows[1];
  assert.equal(actual_assembly.counts.created, 2342);
  assert.deepEqual(actual_assembly.range, [3, 3]);
});

test("phaseRows keeps only steps that changed atoms, never the input step", () => {
  const actual_cleanup = phaseRows(view)[2];
  const expected_steps = ["remove_waters", "remove_crystallization_aids"];
  assert.deepEqual(actual_cleanup.steps.map((s) => s.name), expected_steps);
  assert.equal(actual_cleanup.counts.removed, 372);
});

test("atomLayers at remove_waters puts the 170 input waters in removedNow", () => {
  const index = view.steps.findIndex((s) => s.name === "remove_waters");
  const actual = atomLayers(view, index, index);
  const actual_waters = actual.removedNow.filter((i) => view.input.res_name[i] === "HOH");
  assert.equal(actual_waters.length, 170);
});

test("atomLayers at the assembly phase calls the waters laterRemoved, not removedEarlier", () => {
  const [lo, hi] = stepRange(view, "Assembly");
  const actual = atomLayers(view, lo, hi);
  const actual_later_waters = actual.laterRemoved.filter((i) => view.input.res_name[i] === "HOH");
  assert.equal(actual_later_waters.length, 170);
  assert.equal(actual.removedEarlier.filter((i) => view.input.res_name[i] === "HOH").length, 0);
});

test("atomLayers createdNow lists created atoms made inside the range", () => {
  const actual = atomLayers(view, 3, 3);
  assert.equal(actual.createdNow.length, 2342);
  assert.equal(atomLayers(view, 1, 2).createdNow.length, 0);
});

test("a no-log view gives no phase rows", () => {
  const nolog = { ...view, steps: null, phases: [], before_after_only: true };
  assert.deepEqual(phaseRows(nolog), []);
});

test("loadStepData resolves on callback with the data for its id", async () => {
  const expected_data = { steps: ["x"] };
  const appended = [];
  globalThis.window = globalThis;
  globalThis.document = {
    createElement: () => ({ remove() {} }),
    head: {
      append: (script) => {
        appended.push(script.src);
        window.__reportStep3d("my struct", expected_data);
      },
    },
  };
  try {
    const actual_data = await loadStepData("my struct");
    assert.deepEqual(actual_data, expected_data);
    assert.deepEqual(appended, ["data/validate_3d/my%20struct.js"]);
  } finally {
    delete globalThis.document;
    delete globalThis.window;
    delete globalThis.__reportStep3d;
  }
});

test("issuePoints lists each issue atom with its origin and label", () => {
  const issues = [
    { origin: "input", check: "validity.bond_lengths", label: "b1", xyz: [[0, 0, 0], [3, 0, 0]] },
    { origin: "processing", check: "validity.ligand_geometry", label: "l1", xyz: [[1, 1, 1]] },
  ];
  const actual = issuePoints({ issues });
  const expected = [
    { origin: "input", label: "b1", xyz: [0, 0, 0] },
    { origin: "input", label: "b1", xyz: [3, 0, 0] },
    { origin: "processing", label: "l1", xyz: [1, 1, 1] },
  ];
  assert.deepEqual(actual, expected);
});

test("issuePoints of a view without issues is empty", () => {
  assert.deepEqual(issuePoints(view), []);
});

test("hoverText names a structure atom by its chain id, residue and atom", () => {
  const chains = [{ id: "A" }, { id: "B" }];
  const atom = { chain: "B", resn: "VAL", resi: 1, atom: "CA", serial: 7 };
  assert.equal(hoverText(atom, chains, []), "B VAL1 CA");
});

test("hoverText of a removed atom says it was removed", () => {
  const atom = { chain: "-", resn: "HOH", resi: 301, atom: "O", serial: 7 };
  assert.equal(hoverText(atom, [], []), "removed · HOH301 O");
});

test("hoverText of an issue marker is the issue label", () => {
  const points = [{ origin: "input", label: "b1", xyz: [0, 0, 0] }, { origin: "input", label: "b2", xyz: [1, 0, 0] }];
  const atom = { chain: "-", resn: "ISS", resi: 1, atom: "X", serial: 1 };
  assert.equal(hoverText(atom, [], points), "b2");
});

test("shortLabel keeps a short hover label whole", () => {
  assert.equal(shortLabel("B VAL1 CA"), "B VAL1 CA");
});

test("shortLabel cuts a long hover label with an ellipsis", () => {
  const text = "HEM chain 5: a bond too long (1.55× its upper bound, limit 1.25×)";
  const actual = shortLabel(text);
  assert.equal(actual.length, HOVER_LABEL_MAX);
  assert.ok(actual.endsWith("…"));
  assert.ok(text.startsWith(actual.slice(0, -1)));
});
