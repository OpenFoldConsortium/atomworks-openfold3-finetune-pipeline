import assert from "node:assert/strict";
import test from "node:test";
import { labelClass, SEVERITY } from "../js/labels.js";
import { DATASET_ROW, ITEMS_SHOWN, browseOptionText, noteText, splitItems, cleanupRowView, hasDetails, heading, issueHref, notRunText, statusLines, truncatedNote, validationNote } from "../js/usersummary.js";

const counts = (processing, input, not_checked, processing_issues = processing) => ({ counts: { processing, processing_issues, input, not_checked } });

test("no processing issues gives a green line", () => {
  const actual_line = statusLines(counts(0, 0, 0))[0];
  const expected_line = { dot: "g", text: "Processing introduced no issues", count: null };
  assert.deepEqual(actual_line, expected_line);
});

test("processing issues give a red line", () => {
  const actual_line = statusLines(counts(1, 0, 0))[0];
  const expected_line = { dot: "r", text: "Processing: 1 issue(s) not in the input CIF (likely a pipeline bug; please report)", count: 1 };
  assert.deepEqual(actual_line, expected_line);
});

test("input issues give an orange line", () => {
  const actual_lines = statusLines(counts(0, 2, 0));
  const expected_line = { dot: "o", text: "Input structures: 2 with issues already in the CIF (kept)", count: 2 };
  assert.deepEqual(actual_lines[1], expected_line);
});

test("not-checked line appears only when there are such structures", () => {
  assert.equal(statusLines(counts(0, 0, 0)).length, 1);
  const actual_lines = statusLines(counts(0, 0, 3));
  const expected_line = { dot: "n", text: "Not checked: 3 structures", count: 3 };
  assert.deepEqual(actual_lines[actual_lines.length - 1], expected_line);
});

test("input and not_checked labels are styled and ranked below open findings", () => {
  assert.equal(labelClass("input"), "warn");
  assert.equal(labelClass("not_checked"), "muted");
  assert.ok(SEVERITY.indexOf("input") > SEVERITY.indexOf("exception"));
  assert.ok(SEVERITY.indexOf("not_checked") < SEVERITY.indexOf("expected"));
});

test("cleanup rows show the count and link only structures with 3D", () => {
  const row = { step: "remove_waters", count: 3, structures: ["a", "b"], has_3d: ["b"] };
  const actual_view = cleanupRowView(row);
  const expected_view = { step: "remove_waters", count: 3, structures: [{ id: "a", href: null }, { id: "b", href: "#b/3d" }] };
  assert.deepEqual(actual_view, expected_view);
});

test("heading counts dropped structures from counts when capped", () => {
  const summary = { structures: 600, dropped: [], counts: { dropped: 150 } };
  assert.equal(heading(summary), "600 structures processed, 150 dropped");
});

test("count text is split from the line text for colouring", () => {
  const actual_line = statusLines(counts(1, 0, 0))[0];
  assert.equal(actual_line.count, 1);
});

test("the dataset row has no details link; structure rows do", () => {
  assert.equal(hasDetails({ id: DATASET_ROW, items: [], has_3d: false }), false);
  assert.equal(hasDetails({ id: "4hhb", items: [], has_3d: false }), true);
});

test("the processing line counts issues, not structures", () => {
  const actual_line = statusLines(counts(1, 0, 0, 3))[0];
  const expected_line = { dot: "r", text: "Processing: 3 issue(s) not in the input CIF (likely a pipeline bug; please report)", count: 3 };
  assert.deepEqual(actual_line, expected_line);
});

test("the truncated note uses the summary's row cap", () => {
  const actual_note = truncatedNote({ row_cap: 25 }, 600);
  const expected_note = "showing first 25 of 600; build a report on a subset with finetune-report --ids <file>";
  assert.equal(actual_note, expected_note);
});

test("a skipped validation stage gives one muted note", () => {
  const expected_note = "Validation stage was skipped; load checks not included";
  assert.equal(validationNote({ validation_skipped: true }), expected_note);
  assert.equal(validationNote({ validation_skipped: false }), null);
});

test("an issue row's structure links to its 3D view focused on the issues", () => {
  assert.equal(issueHref({ id: "4hhb", items: [], has_3d: true }), "#4hhb/3d/issues");
  assert.equal(issueHref({ id: "4hhb", items: [], has_3d: false }), null);
  assert.equal(issueHref({ id: DATASET_ROW, items: [], has_3d: false }), null);
});

test("a browse option names the structure, its issues and cleanup steps", () => {
  const row = { id: "a", status: "input", processing: 1, input: 2, not_checked: 0, cleanup_steps: 3, has_3d: true };
  assert.equal(browseOptionText(row), "a — 1 processing, 2 input · 3 cleanup steps");
});

test("a browse option for a clean structure", () => {
  const row = { id: "b", status: "clean", processing: 0, input: 0, not_checked: 0, cleanup_steps: 1, has_3d: false };
  assert.equal(browseOptionText(row), "b — no issues · 1 cleanup step");
});

test("a browse option for an untouched structure", () => {
  const row = { id: "c", status: "clean", processing: 0, input: 0, not_checked: 0, cleanup_steps: 0, has_3d: false };
  assert.equal(browseOptionText(row), "c — no issues · unchanged by cleanup");
});

test("a skipped pipeline check reads as not run, with why", () => {
  const item = { check: "V16", title: "Templates predate their entry", message: "skipped: no --template-cache-dir" };
  assert.equal(notRunText(item), "Not run: V16 Templates predate their entry (no --template-cache-dir)");
});

test("a dataset note reads as a note", () => {
  assert.equal(noteText("templates use a fixed cutoff of 2030-01-01"), "Note: templates use a fixed cutoff of 2030-01-01");
});

test("long issue lists show the first few and fold the rest", () => {
  const items = Array.from({ length: ITEMS_SHOWN + 2 }, (_, i) => `issue ${i}`);
  const actual = splitItems(items);
  assert.deepEqual(actual.shown, items.slice(0, ITEMS_SHOWN));
  assert.deepEqual(actual.hidden, items.slice(ITEMS_SHOWN));
});

test("short issue lists are shown whole", () => {
  assert.deepEqual(splitItems(["a", "b"]), { shown: ["a", "b"], hidden: [] });
});
