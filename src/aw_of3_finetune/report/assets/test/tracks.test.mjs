import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { chainNote, letterClass, ligandChips, stateClass, trackLayout } from "../js/tracks.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../../../../tests/report/data/${name}`, import.meta.url)));
const parity = load("parity_data_sample.json");
const validate = load("validate_data_sample.json");

test("parity 4x1f gives one OF3/ours row pair per polymer chain", () => {
  const actual_layout = trackLayout(parity.entries["4x1f"].tracks, 700);
  const expected_who = ["OF3", "ours", "OF3", "ours"];
  assert.deepEqual(actual_layout.rows.map((r) => r.who), expected_who);
  assert.deepEqual(actual_layout.rows.map((r) => r.chainId), ["1", "1", "6", "3"]);
});

test("validate 4x1f marks the assembly copy residues with the copy class", () => {
  const actual_layout = trackLayout(validate.entries["4x1f"].tracks, 700, { who: ["input", "ours"] });
  const actual_copy_row = actual_layout.rows.find((r) => r.chainId === "3");
  const actual_classes = new Set(actual_copy_row.cells.map((c) => c.stateClass));
  assert.ok(actual_classes.has("st-copy"));
  assert.ok(actual_layout.rows.some((r) => r.who === "input" && r.chainId === null));
});

test("cell width is at least 0.8 px, and 11 px with letters", () => {
  const tracks = parity.entries["6tfi"].tracks;
  assert.equal(trackLayout(tracks, 10).px, 0.8);
  assert.equal(trackLayout(tracks, 10, { letters: true }).px, 11);
});

test("stateClass maps known states and any other state to the bad class", () => {
  assert.equal(stateClass("ok"), "st-ok");
  assert.equal(stateClass("unresolved"), "st-unres");
  assert.equal(stateClass("removed"), "st-removed");
  assert.equal(stateClass("wrong_chain"), "st-bad");
  assert.equal(stateClass("missing"), "st-bad");
});

test("letters on outlined, unresolved and removed cells use the foreground, dark and red cells the background", () => {
  assert.equal(letterClass("assembly_copy"), "tr-letter-fg");
  assert.equal(letterClass("unresolved"), "tr-letter-fg");
  assert.equal(letterClass("removed"), "tr-letter-fg");
  assert.equal(letterClass("ok"), "tr-letter-bg");
  assert.equal(letterClass("split"), "tr-letter-bg");
});

test("7axf parity ligand chips include S6T chain 3 placed on our chain 3", () => {
  const actual_chips = ligandChips(parity.entries["7axf"].tracks);
  const actual_s6t = actual_chips.find((c) => c.refMol === "S6T" && c.refId === "3");
  const expected_s6t = { refMol: "S6T", refId: "3", ourId: "3", kind: "ok", glyph: "✓", text: "S6T 3→3 ✓" };
  assert.deepEqual(actual_s6t, expected_s6t);
  assert.equal(actual_chips.length, 7);
});

test("4x1f validate shows the removed crystallization aids and water as removed chips", () => {
  const actual_chips = ligandChips(validate.entries["4x1f"].tracks);
  const actual_removed = actual_chips.filter((c) => c.kind === "removed").map((c) => `${c.refMol} ${c.refId}`);
  const expected_removed = ["MPD C", "MPD D", "HOH E"];
  assert.deepEqual(actual_removed, expected_removed);
  const actual_removed_text = actual_chips.find((c) => c.refId === "C").text;
  assert.equal(actual_removed_text, "MPD C removed");
});

test("an unpaired assembly-copy ligand of ours gets an outlined copy chip", () => {
  const actual_chips = ligandChips(validate.entries["4x1f"].tracks);
  const actual_copy = actual_chips.find((c) => c.ourId === "4");
  const expected_copy = { refMol: "3WF", refId: null, ourId: "4", kind: "copy", glyph: "", text: "3WF —→4 copy" };
  assert.deepEqual(actual_copy, expected_copy);
});

test("a ligand with a bad residue state gets a cross", () => {
  const tracks = {
    ref: [{ id: "B", type: "LIGAND", ref_mol: "ATP", paired: "2", res: [[1, "ATP", "ok"]] }],
    ours: [{ id: "2", type: "LIGAND", ref_mol: null, paired: "B", res: [[1, "ATP", "wrong_chain"]] }],
  };
  const [actual_chip] = ligandChips(tracks);
  assert.equal(actual_chip.kind, "bad");
  assert.equal(actual_chip.text, "ATP B→2 ✗");
});

test("chainNote counts matching residues, or residues off when any is bad", () => {
  const sample_chain = parity.entries["4x1f"].tracks.ours.find((c) => c.id === "1");
  const expected_ok = sample_chain.res.filter((r) => r[2] === "ok").length;
  const actual_note = chainNote(sample_chain);
  assert.deepEqual(actual_note, { text: `✓ ${expected_ok} match`, cls: "ok-t" });

  const bad_chain = { res: [[1, "ALA", "ok"], [2, "GLY", "wrong_chain"], [3, "SER", "missing"], [4, "LYS", "unresolved"]] };
  const actual_bad_note = chainNote(bad_chain);
  assert.deepEqual(actual_bad_note, { text: "✗ 2 residues off", cls: "bad-t" });
  assert.equal(chainNote(null), null);
});
