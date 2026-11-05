import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { bannerFor, columnGroups, cleanupViewHref, legendLabels, legendText, gridColumns, gridRows, parseRoute, rowView, tiles } from "../js/summary.js";
import { activeTab, entryTabs, groupReasons, neighbours, panelMessage, runLevelRows, vCheckRows, whyHeading, worstLabel } from "../js/entry.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../../../../tests/report/data/${name}`, import.meta.url)));
const parity = load("parity_data_sample.json");
const validate = load("validate_data_sample.json");

function countLabel(data, label) {
  return Object.values(data.entries).flatMap((e) => Object.values(e.grid)).filter((l) => l === label).length;
}

test("gridRows gives one row per entry with columns in grid key order", () => {
  const actual_rows = gridRows(parity, "parity");
  assert.equal(actual_rows.length, 3);
  const expected_columns = Object.keys(parity.entries["7axf"].grid);
  assert.deepEqual(actual_rows[0].cells.map((c) => c.column), expected_columns);
  assert.deepEqual(gridColumns(parity), expected_columns);
});

test("error and missing entries carry error and status", () => {
  const data = {
    entries: {
      a: { error: "boom", grid: { x: "unexplained" } },
      b: { status: "missing", detail: "no file", grid: { x: "none" } },
    },
  };
  const actual_rows = gridRows(data, "parity");
  assert.equal(actual_rows[0].error, "boom");
  assert.equal(actual_rows[1].status, "missing");
});

test("tiles read counts straight from the grids", () => {
  const actual = tiles(parity, validate);
  assert.equal(actual.entries, 3);
  assert.deepEqual(actual.sections.parity, { unexplained: 4, expected: 3, open: 1, exception: 0 });
  assert.deepEqual(actual.sections.validate, { unexplained: 3, expected: 0, open: 0, exception: 0 });
  for (const [name, data] of [["parity", parity], ["validate", validate]]) {
    assert.equal(actual.sections[name].unexplained, countLabel(data, "unexplained"));
    assert.equal(actual.sections[name].expected, countLabel(data, "expected"));
    assert.equal(actual.sections[name].open, countLabel(data, "open"));
  }
});

test("a validate-only run works without parity", () => {
  const actual = tiles(null, validate);
  assert.deepEqual(Object.keys(actual.sections), ["validate"]);
  assert.equal(gridRows(validate, "validate").length, 3);
  assert.equal(tiles(null, null).entries, 0);
});

test("parseRoute reads entry and 3d hashes", () => {
  assert.deepEqual(parseRoute("#4x1f/3d"), { id: "4x1f", view: "3d" });
  assert.deepEqual(parseRoute("#4x1f"), { id: "4x1f", view: "entry" });
  assert.equal(parseRoute(""), null);
});

test("neighbours and groupReasons", () => {
  assert.deepEqual(neighbours(["a", "b", "c"], "b"), { prev: "a", next: "c" });
  assert.deepEqual(neighbours(["a", "b"], "a"), { prev: null, next: "b" });
  const reason = { check: "c", field: "f", label: "expected", reason: "r" };
  const actual_groups = groupReasons([[reason], [reason]]);
  assert.equal(actual_groups.get("c").length, 1);
});

test("error and missing rows get a red id with the message and keep their grid labels", () => {
  const data = {
    entries: {
      m: { status: "missing", detail: "no file", grid: { x: "unexplained", y: "unexplained" } },
      e: { error: "boom", grid: { x: "unexplained", y: "unexplained" } },
    },
  };
  const [actual_missing, actual_error] = gridRows(data, "parity").map(rowView);
  assert.equal(actual_missing.idClass, "id bad-t");
  assert.equal(actual_missing.idTitle, "missing: no file");
  assert.equal(actual_error.idTitle, "error: boom");
  const expected_labels = ["unexplained", "unexplained"];
  assert.deepEqual(actual_missing.cells.map((c) => c.label), expected_labels);
});

test("rowView never rewrites a label: the grid is the only verdict", () => {
  const data = { entries: { m: { status: "missing", detail: "no file", grid: { x: "none" } } } };
  const actual_view = rowView(gridRows(data, "parity")[0]);
  const expected_labels = ["none"];
  assert.deepEqual(actual_view.cells.map((c) => c.label), expected_labels);
});

test("normal rows keep their labels and no bad marker", () => {
  const actual_view = rowView(gridRows(parity, "parity")[0]);
  assert.equal(actual_view.idClass, "id");
  assert.equal(actual_view.cells[0].label, parity.entries["7axf"].grid.chain_placement);
});

test("run-level rows carry status and are not counted in tiles", () => {
  const data = { run_findings: [{ check: "V16", status: "unexplained", label: "unexplained", detail: "d", reason: null }], entries: { a: { grid: { x: "same" } } } };
  const actual_rows = runLevelRows({ validate: data });
  assert.deepEqual(actual_rows[0], { section: "validate", check: "V16", status: "unexplained", label: "unexplained", detail: "d", reason: "" });
  assert.equal(tiles(null, data).sections.validate.unexplained, 0);
});

test("a crashed build counts its run-level unexplained finding separately from the per-entry cells", () => {
  const crashed = { entries: {}, error: "build crashed", run_findings: [{ check: "V0", status: "unexplained", label: "unexplained", detail: "d", reason: null }] };
  const actual = tiles(null, crashed);
  const expected_run_level = { unexplained: 1, exception: 0, expected: 0, open: 0 };
  assert.deepEqual(actual.runLevel.validate, expected_run_level);
  assert.deepEqual(actual.sections.validate, { unexplained: 0, exception: 0, expected: 0, open: 0 });
});

test("bannerFor gives a section's top-level error text, or null", () => {
  const actual_banner = bannerFor({ entries: {}, error: "build crashed" });
  assert.equal(actual_banner, "build crashed");
  assert.equal(bannerFor(validate), null);
  assert.equal(bannerFor(null), null);
});

test("missing entries render red like error entries", () => {
  const actual_missing = panelMessage({ status: "missing", detail: "no file" });
  assert.deepEqual(actual_missing, { cls: "bad-t", text: "missing: no file" });
  const actual_error = panelMessage({ error: "boom" });
  assert.deepEqual(actual_error, { cls: "bad-t", text: "error: boom" });
  assert.equal(panelMessage({ grid: {} }), null);
});

test("cleanupViewHref links entries that have 3D data to their cleanup view", () => {
  const has3d = ["4x1f"];
  const expected_href = "#4x1f/3d";
  const actual_href = cleanupViewHref("4x1f", has3d);
  assert.equal(actual_href, expected_href);
  assert.equal(cleanupViewHref("7axf", has3d), null);
});

test("parseRoute reads the issues focus after the 3d tab", () => {
  assert.deepEqual(parseRoute("#4hhb/3d/issues"), { id: "4hhb", view: "3d", focus: "issues" });
  assert.deepEqual(parseRoute("#4hhb/3d/nonsense"), { id: "4hhb", view: "3d" });
});

test("parseRoute reads a tab name after the entry id", () => {
  assert.deepEqual(parseRoute("#4x1f/validate"), { id: "4x1f", view: "validate" });
  assert.deepEqual(parseRoute("#4x1f/load"), { id: "4x1f", view: "load" });
  assert.deepEqual(parseRoute("#4x1f/nonsense"), { id: "4x1f", view: "entry" });
});

test("entryTabs lists the data sections, then 3D and load when present", () => {
  const expected_keys = ["parity", "validate", "3d", "load"];
  const actual_keys = entryTabs(["parity", "validate"], { has3d: true, hasLoad: true }).map((t) => t.key);
  assert.deepEqual(actual_keys, expected_keys);
});

test("entryTabs on a validate-only run without 3D data has one tab", () => {
  const expected_keys = ["validate"];
  const actual_keys = entryTabs(["validate"], { has3d: false, hasLoad: false }).map((t) => t.key);
  assert.deepEqual(actual_keys, expected_keys);
});

test("activeTab falls back to the first tab when the route names none", () => {
  const tabs = entryTabs(["parity", "validate"], { has3d: true, hasLoad: false });
  assert.equal(activeTab(tabs, "entry"), "parity");
  assert.equal(activeTab(tabs, "3d"), "3d");
  assert.equal(activeTab(tabs, "load"), "parity");
});

test("worstLabel picks the most severe label in a grid", () => {
  const grid = { a: "same", b: "expected", c: "unexplained", d: "exception" };
  const expected_label = "unexplained";
  const actual_label = worstLabel(grid);
  assert.equal(actual_label, expected_label);
  assert.equal(worstLabel({}), null);
});

test("legendLabels lists only the labels that occur, in legend order", () => {
  const sections = {
    validate: {
      entries: { a: { grid: { x: "same", y: "exception" } }, b: { grid: { x: "unexplained" } } },
      run_findings: [{ label: "none" }],
    },
  };
  const expected_labels = ["same", "exception", "unexplained", "none"];
  const actual_labels = legendLabels(sections);
  assert.deepEqual(actual_labels, expected_labels);
});

test("whyHeading speaks of differences only when there is a reference", () => {
  assert.equal(whyHeading(["parity", "validate"]), "Why the expected differences are expected");
  assert.equal(whyHeading(["validate"]), "Accepted exceptions and why");
});

test("legendText calls a validate pass 'passed', since there is nothing to be the same as", () => {
  assert.equal(legendText("same", ["validate"]), "passed");
  assert.equal(legendText("same", ["parity", "validate"]), "same / passed");
  assert.equal(legendText("same", ["parity"]), "same");
  assert.equal(legendText("exception", ["validate"]), "exception");
});

test("columnGroups groups checks by prefix, keeping first-seen group order", () => {
  const columns = ["structures.atoms", "msa.depth", "structures.bonds", "rep_ids.classes"];
  const actual_groups = columnGroups(columns).map((g) => [g.group, g.columns.map((c) => c.short)]);
  const expected_groups = [["structures", ["atoms", "bonds"]], ["msa", ["depth"]], ["rep ids", ["classes"]]];
  assert.deepEqual(actual_groups, expected_groups);
});

test("columnGroups keeps the full key and spaces out underscores in short names", () => {
  const [actual_group] = columnGroups(["validity.ligand_contacts"]);
  const [actual_column] = actual_group.columns;
  assert.equal(actual_group.group, "validity");
  assert.equal(actual_column.key, "validity.ligand_contacts");
  assert.equal(actual_column.short, "ligand contacts");
});

test("columnGroups puts a column without a prefix in its own group", () => {
  const actual_groups = columnGroups(["cleanup_audit"]).map((g) => [g.group, g.columns.map((c) => c.short)]);
  assert.deepEqual(actual_groups, [["", ["cleanup audit"]]]);
});

test("vCheckRows keeps only errors and warnings", () => {
  const vChecks = [
    { check: "V1", title: "Chains accounted for", severity: "I", message: "water" },
    { check: "V5", title: "Finite coordinates", severity: "E", message: "inf" },
    { check: "V4", title: "Ligand present", severity: "W", message: "no ligand" },
  ];
  const actual_checks = vCheckRows(vChecks).map((v) => v.check);
  assert.deepEqual(actual_checks, ["V5", "V4"]);
});
