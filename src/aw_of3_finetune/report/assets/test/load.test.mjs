import assert from "node:assert/strict";
import test from "node:test";
import { loadColumns, loadRows, loadSummaryView } from "../js/load.js";

const loadData = {
  entries: {
    "1abc": { groups: { "load.templates": "same", "load.atoms": "same", "load.msa": "same" }, differences: [] },
    "7axf": { groups: { "load.templates": "same", "load.atoms": "unexplained", "load.msa": "same" }, differences: [] },
    "2bbb": { groups: { "load.templates": "same", "load.atoms": "expected", "load.msa": "same" }, differences: [] },
  },
};

test("loadRows puts entries with an unexplained group first, then sorts by id, cells in group order", () => {
  const actual_rows = loadRows(loadData);
  const expected_cells_7axf = [
    { group: "load.atoms", label: "unexplained" },
    { group: "load.msa", label: "same" },
    { group: "load.templates", label: "same" },
  ];
  const expected_ids = ["7axf", "1abc", "2bbb"];
  assert.deepEqual(actual_rows.map((row) => row.id), expected_ids);
  assert.deepEqual(actual_rows[0].cells, expected_cells_7axf);
});

test("loadRows of missing data is empty", () => {
  assert.deepEqual(loadRows(null), []);
});

test("columns are the union of every row's groups in the fixed order, even when the first row is a crash", () => {
  const withCrash = {
    entries: {
      "0aaa": { groups: { "load.crash": "unexplained" }, differences: [] },
      "1abc": { groups: { "load.keys": "same", "load.ground_truth": "same", "load.tokens": "same" }, differences: [] },
    },
  };
  const actual_columns = loadColumns(loadRows(withCrash));
  const expected_columns = ["load.keys", "load.tokens", "load.ground_truth", "load.crash"];
  assert.deepEqual(actual_columns, expected_columns);
});

test("loadSummaryView carries the setup error and no rows when setup failed", () => {
  const failed = { setup_error: "FileNotFoundError: /nowhere/train_no_crop.yaml", entries: {} };
  const actual_view = loadSummaryView(failed);
  const expected_view = { setupError: "FileNotFoundError: /nowhere/train_no_crop.yaml", columns: [], rows: [] };
  assert.deepEqual(actual_view, expected_view);
});

test("loadSummaryView without a setup error has a null setupError", () => {
  const actual_view = loadSummaryView(loadData);
  const expected_columns = ["load.atoms", "load.msa", "load.templates"];
  assert.equal(actual_view.setupError, null);
  assert.deepEqual(actual_view.columns, expected_columns);
});
