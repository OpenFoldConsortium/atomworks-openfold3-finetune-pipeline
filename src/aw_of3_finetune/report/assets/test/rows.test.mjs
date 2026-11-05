import assert from "node:assert/strict";
import test from "node:test";
import { ligandSummaries } from "../js/rows.js";

const row = (chain, res_name, test_name, passed, label, extra = {}) => ({
  check: "validity.ligand_contacts", chain, res_name, test: test_name, passed, value: null, threshold: null, label, reason: null, ...extra,
});

test("one summary per ligand, in first-seen order, counting passed checks", () => {
  const rows = [row("3", "IPA", "a", true, "same"), row("2", "SRL", "a", true, "same"), row("3", "IPA", "b", true, "same")];
  const actual_summaries = ligandSummaries(rows).map((s) => [s.chain, s.res_name, s.n_passed, s.n_checks]);
  const expected_summaries = [["3", "IPA", 2, 2], ["2", "SRL", 1, 1]];
  assert.deepEqual(actual_summaries, expected_summaries);
});

test("a ligand with every check passing has no failed checks and label same", () => {
  const [actual_summary] = ligandSummaries([row("2", "SRL", "a", true, "same")]);
  assert.equal(actual_summary.label, "same");
  assert.deepEqual(actual_summary.failed, []);
});

test("failed checks are listed and the ligand takes the worst label", () => {
  const rows = [
    row("4", "NQD", "stereo", false, "exception", { reason: "deposited E vs CCD Z" }),
    row("4", "NQD", "max distance", false, "unexplained", { threshold: 5.0 }),
    row("4", "NQD", "formula", true, "same"),
  ];
  const [actual_summary] = ligandSummaries(rows);
  const actual_failed_tests = actual_summary.failed.map((r) => r.test);
  assert.equal(actual_summary.label, "unexplained");
  assert.deepEqual(actual_failed_tests, ["stereo", "max distance"]);
  assert.equal(actual_summary.n_passed, 1);
});

test("the same residue name on two chains gives two summaries", () => {
  const rows = [row("2", "3WF", "a", true, "same"), row("4", "3WF", "a", true, "same")];
  const actual_chains = ligandSummaries(rows).map((s) => s.chain);
  assert.deepEqual(actual_chains, ["2", "4"]);
});
