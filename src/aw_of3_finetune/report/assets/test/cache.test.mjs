import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { cacheTableRows, formatValue } from "../js/cache.js";

const parity = JSON.parse(readFileSync(new URL("../../../../tests/report/data/parity_data_sample.json", import.meta.url)));
const rows7axf = parity.entries["7axf"].cache_rows;

test("the folded 7axf auth_asym_id row shows its chains and per-chain ours", () => {
  const actual = cacheTableRows(rows7axf).rows.find((r) => r.kind === "row" && r.field === "auth_asym_id");
  assert.equal(actual.chains, "2, 3, 4, 5, 6, 7, 8");
  assert.match(actual.ours, /2: B\n3: C/);
  assert.equal(actual.ref, "A");
  assert.equal(actual.label, "unexplained");
});

test("same and equivalent rows are hidden unless toggled, with a count", () => {
  const hiddenView = cacheTableRows(rows7axf);
  const shownView = cacheTableRows(rows7axf, { showSame: true });
  const quiet = rows7axf.filter((r) => ["same", "equivalent"].includes(r.label)).length;
  assert.equal(hiddenView.hidden, quiet);
  assert.equal(shownView.hidden, 0);
  assert.ok(shownView.rows.length > hiddenView.rows.length);
  assert.ok(hiddenView.rows.every((r) => r.kind === "section" || !["same", "equivalent"].includes(r.label)));
});

test("a section row precedes each change of section and nulls print as null", () => {
  const { rows } = cacheTableRows(rows7axf, { showSame: true });
  assert.equal(rows[0].kind, "section");
  const sections = rows.filter((r) => r.kind === "section").map((r) => r.section);
  sections.slice(1).forEach((section, k) => assert.notEqual(section, sections[k]));
  assert.equal(formatValue(null), "null");
});
