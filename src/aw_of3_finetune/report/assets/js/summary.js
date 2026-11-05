// Summary: pure data helpers (tested in Node) plus DOM rendering that runs only when called.
import { el, chip } from "./dom.js";
import { renderLoadSummary } from "./load.js";
import { LABELS, labelClass } from "./labels.js";

const TAB_VIEWS = ["parity", "validate", "3d", "load"];

export function parseRoute(hash) {
  const text = (hash ?? "").replace(/^#/, "");
  if (!text) return null;
  const [id, view, focus] = text.split("/");
  if (!id) return null;
  const tab = TAB_VIEWS.includes(view) ? view : "entry";
  // #<id>/3d/issues opens the 3D view zoomed to the entry's issues.
  return { id, view: tab, ...(tab === "3d" && focus === "issues" ? { focus } : {}) };
}

export function cleanupViewHref(id, has3d) {
  return has3d.includes(id) ? `#${id}/3d` : null;
}

export function legendLabels(sections) {
  const present = new Set();
  for (const data of Object.values(sections)) {
    for (const entry of Object.values(data?.entries ?? {})) Object.values(entry.grid ?? {}).forEach((label) => present.add(label));
    for (const finding of data?.run_findings ?? []) present.add(finding.label);
  }
  return LABELS.filter((label) => present.has(label));
}

export function legendText(label, sectionNames) {
  if (label !== "same") return label;
  const hasParity = sectionNames.includes("parity");
  if (!sectionNames.includes("validate")) return "same";
  return hasParity ? "same / passed" : "passed";  // validate has no reference: a pass is not "the same"
}

export function gridColumns(data) {
  const columns = [];
  for (const entry of Object.values(data?.entries ?? {})) {
    for (const key of Object.keys(entry.grid ?? {})) {
      if (!columns.includes(key)) columns.push(key);
    }
  }
  return columns;
}

// Columns grouped by the prefix before the first "." (structures.atoms -> structures / atoms), groups in first-seen order.
export function columnGroups(columns) {
  const groups = new Map();
  for (const key of columns) {
    const dot = key.indexOf(".");
    const group = dot < 0 ? "" : key.slice(0, dot).replaceAll("_", " ");
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push({ key, short: key.slice(dot + 1).replaceAll("_", " ") });
  }
  return [...groups].map(([group, cols]) => ({ group, columns: cols }));
}

export function gridRows(data, sectionName) {
  return Object.entries(data?.entries ?? {}).map(([id, entry]) => {
    const row = {
      id,
      section: sectionName,
      cells: Object.entries(entry.grid ?? {}).map(([column, label]) => ({ column, label })),
    };
    if (entry.error !== undefined) row.error = entry.error;
    if (entry.status !== undefined) row.status = entry.status;
    if (entry.detail !== undefined) row.detail = entry.detail;
    return row;
  });
}

export function rowView(row) {
  const failed = row.error !== undefined || row.status !== undefined;
  const message = row.error !== undefined ? `error: ${row.error}` : `${row.status}: ${row.detail ?? ""}`;
  return {
    idClass: failed ? "id bad-t" : "id",
    idTitle: failed ? message : "",
    cells: row.cells,
  };
}

export function bannerFor(data) {
  return data?.error ?? null;
}

const COUNTED = ["unexplained", "exception", "expected", "open"];

function sectionCounts(data) {
  const counts = Object.fromEntries(COUNTED.map((label) => [label, 0]));
  for (const row of gridRows(data, "")) {
    for (const cell of row.cells) {
      if (cell.label in counts) counts[cell.label] += 1;
    }
  }
  return counts;
}

function runLevelCounts(data) {
  const counts = Object.fromEntries(COUNTED.map((label) => [label, 0]));
  for (const finding of data.run_findings ?? []) {
    if (finding.label in counts) counts[finding.label] += 1;
  }
  return counts;
}

export function tiles(parityData, validateData) {
  const sections = {};
  const runLevel = {};
  const ids = new Set();
  for (const [name, data] of [["parity", parityData], ["validate", validateData]]) {
    if (!data) continue;
    Object.keys(data.entries ?? {}).forEach((id) => ids.add(id));
    sections[name] = sectionCounts(data);
    runLevel[name] = runLevelCounts(data);
  }
  return { entries: ids.size, sections, runLevel };
}

export function renderTiles(container, tileData) {
  const items = [{ n: tileData.entries, l: "entries", cls: "" }];
  for (const [name, counts] of Object.entries(tileData.sections)) {
    for (const label of COUNTED) {
      items.push({ n: counts[label], l: `${name}: ${label}`, cls: labelClass(label) });
    }
  }
  for (const [name, counts] of Object.entries(tileData.runLevel ?? {})) {
    for (const label of COUNTED) {
      if (counts[label]) items.push({ n: counts[label], l: `${name}: run-level ${label}`, cls: labelClass(label) });
    }
  }
  const children = items.map((item) =>
    el("div", { class: "tile" }, el("span", { class: `n ${item.cls}-t` }, String(item.n)), el("span", { class: "l" }, item.l)),
  );
  container.replaceChildren(el("div", { class: "tiles" }, ...children));
}

function cleanupViewCell(id, has3d) {
  const href = cleanupViewHref(id, has3d);
  if (href === null) return el("td", {}, "");
  const link = el("a", { href, class: "view3d", title: `${id}: 3D cleanup steps` }, "3D ▸");
  link.addEventListener("click", (event) => event.stopPropagation());
  return el("td", {}, link);
}

function renderGrid(name, data, onSelect, has3d = []) {
  const groups = columnGroups(gridColumns(data));
  const columns = groups.flatMap((g) => g.columns);
  const with3d = name === "validate" && has3d.length > 0;
  const fixed = [el("th", { rowspan: "2" }, "entry"), ...(with3d ? [el("th", { rowspan: "2" }, "3D")] : [])];
  const groupHead = el("tr", { class: "groups" }, ...fixed,
    ...groups.map((g) => el("th", { colspan: String(g.columns.length), class: "group" }, g.group)));
  const checkHead = el("tr", { class: "checks" },
    ...columns.map((c) => el("th", { class: "check", title: c.key }, el("span", {}, c.short))));
  const body = gridRows(data, name).map((row) => {
    const view = rowView(row);
    const byColumn = Object.fromEntries(view.cells.map((cell) => [cell.column, cell.label]));
    const cells = columns.map((c) => el("td", { class: "cell", title: c.key in byColumn ? `${c.key}: ${byColumn[c.key]}` : c.key }, c.key in byColumn ? chip(byColumn[c.key]) : ""));
    const tr = el("tr", { tabindex: "0", role: "link", "data-id": row.id }, el("td", { class: view.idClass, title: view.idTitle }, row.id), ...(with3d ? [cleanupViewCell(row.id, has3d)] : []), ...cells);
    tr.addEventListener("click", () => onSelect(row.id));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter") onSelect(row.id);
    });
    return tr;
  });
  return el(
    "section",
    { class: "panel" },
    el("h2", {}, `${name}: all entries`),
    el("div", { class: "gridwrap" }, el("table", { class: "grid" }, el("thead", {}, groupHead, checkHead), el("tbody", {}, ...body))),
  );
}

export function renderSummary(container, sections, onSelect, loadData = null, has3d = []) {
  const legend = el(
    "div",
    { class: "legend" },
    ...legendLabels(sections).map((label) => el("span", {}, chip(label), legendText(label, Object.keys(sections)))),
  );
  const banners = Object.entries(sections)
    .filter(([, data]) => bannerFor(data) !== null)
    .map(([name, data]) => el("p", { class: "banner", role: "alert" }, `${name} failed: ${bannerFor(data)}`));
  const grids = Object.entries(sections).map(([name, data]) => renderGrid(name, data, onSelect, has3d));
  const tileData = tiles(sections.parity ?? null, sections.validate ?? null);
  const tileBox = el("div", {});
  renderTiles(tileBox, tileData);
  const loadPanel = loadData ? [renderLoadSummary(loadData, onSelect)] : [];
  container.replaceChildren(...banners, tileBox, legend, ...grids, ...loadPanel);
}
