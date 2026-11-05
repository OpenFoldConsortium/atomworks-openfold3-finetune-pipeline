import assert from "node:assert/strict";
import test from "node:test";
import { chipAttrs, labelClass, labelText } from "../js/labels.js";

test("each label maps to its class", () => {
  const expected_classes = { same: "same", equivalent: "equiv", expected: "expect", open: "expect", exception: "exc", unexplained: "bad", none: "muted" };
  for (const [label, expected_class] of Object.entries(expected_classes)) {
    assert.equal(labelClass(label), expected_class);
  }
});

test("unknown label falls back to muted and its own text", () => {
  assert.equal(labelClass("weird"), "muted");
  assert.equal(labelText("weird"), "weird");
});

test("chip attributes name the label word for screen readers", () => {
  const actual_attrs = chipAttrs("same");
  const expected_attrs = { class: "chip same", title: "same", "aria-label": "same" };
  assert.deepEqual(actual_attrs, expected_attrs);
});

test("open and exception chips read differently", () => {
  const expected_texts = { expected: "exp", open: "open", exception: "exc" };
  for (const [label, expected_text] of Object.entries(expected_texts)) {
    assert.equal(labelText(label), expected_text);
  }
});
