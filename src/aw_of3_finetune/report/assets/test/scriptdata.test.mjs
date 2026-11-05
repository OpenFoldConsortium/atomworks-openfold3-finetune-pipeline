import assert from "node:assert/strict";
import test from "node:test";
import { loadEntryData, withEntryData } from "../js/entry.js";
import { loadScriptData } from "../js/scriptdata.js";

// A fake document whose appended scripts either call the page callback (onAppend) or fail.
function fakeDocument(onAppend) {
  const appended = [];
  globalThis.window = globalThis;
  globalThis.document = {
    createElement: () => ({ remove() {} }),
    head: {
      append: (script) => {
        appended.push(script.src);
        onAppend(script);
      },
    },
  };
  return appended;
}

function cleanUp(...callbacks) {
  delete globalThis.document;
  delete globalThis.window;
  for (const name of callbacks) delete globalThis[name];
}

test("loadScriptData resolves with the data its script hands back for that id", async () => {
  const expected_data = { x: 1 };
  const appended = fakeDocument(() => window.__testData("a b", expected_data));
  try {
    const actual_data = await loadScriptData("__testData", "things", "a b");
    assert.deepEqual(actual_data, expected_data);
    assert.deepEqual(appended, ["data/things/a%20b.js"]);
  } finally {
    cleanUp("__testData");
  }
});

test("loadScriptData rejects when the script fails to load", async () => {
  fakeDocument((script) => script.onerror());
  try {
    await assert.rejects(loadScriptData("__testData", "things", "gone"), /could not load data\/things\/gone\.js/);
  } finally {
    cleanUp("__testData");
  }
});

test("loadScriptData rejects when the script loads but never hands back its data", async () => {
  fakeDocument((script) => script.onload());
  try {
    await assert.rejects(loadScriptData("__testData", "things", "broken"), /data\/things\/broken\.js did not provide data/);
  } finally {
    cleanUp("__testData");
  }
});

test("loadEntryData loads data/entries/<id>.js through window.__reportEntry", async () => {
  const expected_data = { validate: { grid: {} } };
  const appended = fakeDocument(() => window.__reportEntry("my-struct_1", expected_data));
  try {
    const actual_data = await loadEntryData("my-struct_1");
    assert.deepEqual(actual_data, expected_data);
    assert.deepEqual(appended, ["data/entries/my-struct_1.js"]);
  } finally {
    cleanUp("__reportEntry");
  }
});

test("withEntryData puts each section's full entry in place of its summary stub", () => {
  const sections = { validate: { entries: { a: { grid: { x: "same" } }, b: { grid: {} } }, reasons: [] } };
  const loadData = { entries: { a: { groups: {} } } };
  const payload = { validate: { grid: { x: "same" }, findings: ["f"] }, load: { groups: {}, differences: ["d"] } };

  const actual = withEntryData(sections, loadData, "a", payload);

  const expected_validate = { entries: { a: payload.validate, b: { grid: {} } }, reasons: [] };
  const expected_load = { entries: { a: payload.load } };
  assert.deepEqual(actual.sections.validate, expected_validate);
  assert.deepEqual(actual.loadData, expected_load);
  assert.deepEqual(sections.validate.entries.a, { grid: { x: "same" } });
});

test("withEntryData keeps a null loadData", () => {
  const actual = withEntryData({}, null, "a", { load: { groups: {} } });
  assert.equal(actual.loadData, null);
});
