// Entry view: header, per-section panels, findings tables, and the page-level reason lists.
import { drawCacheTable } from "./cache.js";
import { el, chip } from "./dom.js";
import { SEVERITY } from "./labels.js";
import { drawLigands } from "./ligands.js";
import { renderLoadEntry } from "./load.js";
import { BOND_COLUMNS, LINK_COLUMNS, bondRows, drawLigandChecks, drawRowsTable } from "./rows.js";
import { loadScriptData } from "./scriptdata.js";
import { cancelStepView, drawStepEntry } from "./steps3d.js";
import { drawTracksPanel } from "./tracks.js";

export const FIGURE_IDS = {
  parity: ["parity-tracks", "parity-ligands", "parity-cache"],
  validate: ["validate-tracks", "validate-rows"],
};

export function worstLabel(grid) {
  const labels = Object.values(grid ?? {});
  return SEVERITY.find((label) => labels.includes(label)) ?? null;
}

export function entryTabs(sectionNames, { has3d, hasLoad }) {
  const tabs = sectionNames.filter((name) => name in FIGURE_IDS).map((name) => ({ key: name, label: name }));
  if (has3d) tabs.push({ key: "3d", label: "3D cleanup steps" });
  if (hasLoad) tabs.push({ key: "load", label: "load" });
  return tabs;
}

export function activeTab(tabs, view) {
  return tabs.some((tab) => tab.key === view) ? view : (tabs[0]?.key ?? null);
}

export function neighbours(ids, id) {
  const index = ids.indexOf(id);
  return { prev: index > 0 ? ids[index - 1] : null, next: index >= 0 && index < ids.length - 1 ? ids[index + 1] : null };
}

export function groupReasons(reasonLists) {
  const seen = new Set();
  const groups = new Map();
  for (const item of reasonLists.flat()) {
    const key = `${item.check}|${item.field}|${item.label}|${item.reason}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (!groups.has(item.check)) groups.set(item.check, []);
    groups.get(item.check).push(item);
  }
  return groups;
}

function table(headers, rows) {
  const head = el("tr", {}, ...headers.map((h) => el("th", {}, h)));
  return el("div", { class: "scroller" }, el("table", { class: "diff" }, el("thead", {}, head), el("tbody", {}, ...rows)));
}

function findingsTable(findings) {
  const rows = findings.map((f) =>
    el("tr", {}, el("td", { class: "v" }, f.check), el("td", {}, chip(f.label)), el("td", { class: "v" }, f.detail ?? ""), el("td", { class: "why" }, f.reason ?? "")),
  );
  return table(["check", "label", "detail", "reason"], rows);
}

// Pipeline validate results worth reading on their own: errors and warnings (info lines are drop reasons and notes).
export function vCheckRows(vChecks) {
  return (vChecks ?? []).filter((v) => v.severity === "E" || v.severity === "W");
}

const SEVERITY_WORDS = { E: "error", W: "warning", I: "info" };

function vChecksTable(vChecks) {
  const rows = vChecks.map((v) =>
    el("tr", {}, el("td", { class: "v" }, v.check), el("td", {}, v.title ?? ""), el("td", { class: "v" }, SEVERITY_WORDS[v.severity] ?? v.severity), el("td", {}, v.message ?? "")),
  );
  return table(["check", "name", "severity", "message"], rows);
}

function fillFigures(name, entry, panel) {
  const at = (id) => panel.querySelector(`#${id}`);
  if (name === "parity") {
    drawTracksPanel(at("parity-tracks"), entry.tracks, { who: ["OF3", "ours"] });
    drawLigands(at("parity-ligands"), entry.ligands, entry.cross_bonds);
    drawCacheTable(at("parity-cache"), entry.cache_rows);
  } else if (name === "validate") {
    drawTracksPanel(at("validate-tracks"), entry.tracks, { who: ["input", "ours"] });
    const rows = at("validate-rows");
    drawLigandChecks(rows, entry.ligands);
    drawRowsTable(rows, bondRows(entry.bonds), BOND_COLUMNS, "Bonds");
    drawRowsTable(rows, entry.covalent_links, LINK_COLUMNS, "Covalent links");
  }
}

export function panelMessage(entry) {
  if (entry.error !== undefined) return { cls: "bad-t", text: `error: ${entry.error}` };
  if (entry.status !== undefined) return { cls: "bad-t", text: `${entry.status}${entry.detail ? `: ${entry.detail}` : ""}` };
  return null;
}

function sectionPanel(name, entry) {
  const panel = el("div", { class: "panel", id: `panel-${name}` }, el("h2", {}, name));
  if (entry === undefined) {
    panel.append(el("p", { class: "muted" }, "no data for this entry"));
    return panel;
  }
  const message = panelMessage(entry);
  if (message) {
    panel.append(el("p", { class: message.cls }, message.text));
    return panel;
  }
  for (const id of FIGURE_IDS[name] ?? []) panel.append(el("div", { id, class: "figure scroller" }));
  fillFigures(name, entry, panel);
  // validate shows each finding in its own table (ligands, bonds, links), so only parity lists them again.
  if (name !== "validate" && entry.findings?.length) panel.append(el("h3", {}, "Findings"), findingsTable(entry.findings));
  const vChecks = vCheckRows(entry.v_checks);
  if (vChecks.length) panel.append(el("h3", {}, "Pipeline validate: errors and warnings"), vChecksTable(vChecks));
  return panel;
}

function stepsPanel(id, focus) {
  const figure = el("div", { id: "validate-3d", class: "figure scroller" });
  const panel = el("div", { class: "panel", id: "panel-3d" }, el("h2", {}, "3D cleanup steps"), figure);
  drawStepEntry(figure, { id, has3d: true, open: true, focus });
  return panel;
}

function tabPanel(key, { id, sections, loadData, focus }) {
  if (key === "3d") return stepsPanel(id, focus);
  if (key === "load") {
    const loadEntry = loadData?.entries?.[id];
    return loadEntry ? renderLoadEntry(loadEntry) : el("div", { class: "panel" }, el("p", { class: "muted" }, "no load data for this entry"));
  }
  return sectionPanel(key, sections[key].entries?.[id]);
}

export function renderEntry(container, { id, ids, sections, loadData = null, navigate, has3d = [], view = "entry", focus }) {
  const stale = container.querySelector("#validate-3d");
  if (stale) cancelStepView(stale);
  const tabs = entryTabs(Object.keys(sections), { has3d: has3d.includes(id), hasLoad: loadData !== null });
  const active = activeTab(tabs, view);
  const routeFor = (target) => (active === tabs[0]?.key ? target : `${target}/${active}`);
  const { prev, next } = neighbours(ids, id);
  const select = el("select", { id: "e-sel" }, ...ids.map((i) => el("option", i === id ? { value: i, selected: "" } : { value: i }, i)));
  select.addEventListener("change", () => navigate(routeFor(select.value)));
  const button = (text, target) => {
    const node = el("button", { type: "button" }, text);
    if (target === null) node.disabled = true;
    else node.addEventListener("click", () => navigate(routeFor(target)));
    return node;
  };
  const head = el(
    "div",
    { class: "entry-head" },
    el("a", { href: "#", class: "back" }, "← all entries"),
    el("span", { class: "big" }, id),
    el("label", { class: "muted", for: "e-sel" }, "entry ", select),
    button("◀ prev", prev),
    button("next ▶", next),
  );
  const tabBar = el(
    "nav",
    { class: "tabs", "aria-label": "entry sections" },
    ...tabs.map((tab) => {
      const worst = tab.key in sections ? worstLabel(sections[tab.key].entries?.[id]?.grid) : null;
      const attrs = { href: `#${id}/${tab.key}`, class: tab.key === active ? "tab active" : "tab" };
      if (tab.key === active) attrs["aria-current"] = "page";
      return el("a", attrs, tab.label, ...(worst && worst !== "same" ? [" ", chip(worst)] : []));
    }),
  );
  const panel = active === null ? el("p", { class: "muted" }, "no data for this entry") : tabPanel(active, { id, sections, loadData, focus });
  container.replaceChildren(head, tabBar, panel);
}

// Inline REPORT_DATA holds only each entry's summary fields; build.py writes the rest to data/entries/<id>.js.
export function loadEntryData(id) {
  return loadScriptData("__reportEntry", "entries", id);
}

const entryLoads = new Map();

function cachedEntryData(id) {
  if (!entryLoads.has(id)) {
    entryLoads.set(id, loadEntryData(id).catch((error) => {
      entryLoads.delete(id);
      throw error;
    }));
  }
  return entryLoads.get(id);
}

// The sections and load data with one entry's summary stub replaced by its full data; the inputs are not changed.
export function withEntryData(sections, loadData, id, payload) {
  const merged = {};
  for (const [name, data] of Object.entries(sections)) {
    merged[name] = name in payload ? { ...data, entries: { ...data.entries, [id]: payload[name] } } : data;
  }
  const load = loadData && payload.load ? { ...loadData, entries: { ...loadData.entries, [id]: payload.load } } : loadData;
  return { sections: merged, loadData: load };
}

let entryToken = 0;

// Stops a pending showEntry from drawing (the route changed before its data arrived).
export function cancelShowEntry() {
  entryToken += 1;
}

// renderEntry once the entry's data script has loaded: a loading line first, an error line if the script fails.
export async function showEntry(container, options) {
  const token = ++entryToken;
  const stale = container.querySelector("#validate-3d");
  if (stale) cancelStepView(stale);
  container.replaceChildren(el("p", { class: "muted" }, "loading entry data…"));
  let payload;
  try {
    payload = await cachedEntryData(options.id);
  } catch (error) {
    if (token === entryToken) container.replaceChildren(el("p", { class: "bad-t", role: "alert" }, `could not load entry data: ${error.message}`));
    return;
  }
  if (token !== entryToken) return;
  renderEntry(container, { ...options, ...withEntryData(options.sections, options.loadData ?? null, options.id, payload) });
}

export function whyHeading(sectionNames) {
  return sectionNames.includes("parity") ? "Why the expected differences are expected" : "Accepted exceptions and why";
}

export function renderWhy(container, sections) {
  const groups = groupReasons(Object.values(sections).map((data) => data.reasons ?? []));
  if (groups.size === 0) return container.replaceChildren();
  const items = [];
  for (const [check, reasons] of groups) {
    items.push(el("li", {}, el("strong", {}, check)));
    for (const r of reasons) items.push(el("li", {}, chip(r.label), " ", el("code", {}, r.field), " ", r.reason));
  }
  container.replaceChildren(el("div", { class: "panel" }, el("h2", {}, whyHeading(Object.keys(sections))), el("ul", { class: "why" }, ...items)));
}

export function runLevelRows(sections) {
  const rows = [];
  for (const [name, data] of Object.entries(sections)) {
    for (const f of data.run_findings ?? []) {
      rows.push({ section: name, check: f.check, status: f.status, label: f.label, detail: f.detail ?? "", reason: f.reason ?? "" });
    }
  }
  return rows;
}

export function renderRunLevel(container, sections) {
  const rows = runLevelRows(sections).map((r) =>
    el("tr", {}, el("td", {}, r.section), el("td", { class: "v" }, r.check), el("td", { class: "v" }, r.status), el("td", {}, chip(r.label)), el("td", { class: "v" }, r.detail), el("td", { class: "why" }, r.reason)),
  );
  if (rows.length === 0) return container.replaceChildren();
  container.replaceChildren(el("div", { class: "panel" }, el("h2", {}, "Run-level"), table(["section", "check", "status", "label", "detail", "reason"], rows)));
}
