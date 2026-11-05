// Feature load panel: the label grid on the summary and the differences table per entry.
import { el, chip } from "./dom.js";

// Column order of the grid; the same order as GROUPS in load_parity.py.
export const LOAD_GROUPS = ["load.keys", "load.tokens", "load.atoms", "load.msa", "load.templates", "load.ground_truth", "load.crash"];

function groupOrder(group) {
  const index = LOAD_GROUPS.indexOf(group);
  return index === -1 ? LOAD_GROUPS.length : index;
}

function byGroupOrder(a, b) {
  return groupOrder(a) - groupOrder(b) || a.localeCompare(b);
}

// One row per entry, cells in group order; entries with any unexplained group first, then by id.
export function loadRows(loadData) {
  const rows = Object.entries(loadData?.entries ?? {}).map(([id, entry]) => ({
    id,
    cells: Object.keys(entry.groups ?? {})
      .sort(byGroupOrder)
      .map((group) => ({ group, label: entry.groups[group] })),
  }));
  const unexplained = (row) => (row.cells.some((cell) => cell.label === "unexplained") ? 0 : 1);
  return rows.sort((a, b) => unexplained(a) - unexplained(b) || a.id.localeCompare(b.id));
}

// The grid columns: every group any row has, in group order.
export function loadColumns(rows) {
  const groups = new Set(rows.flatMap((row) => row.cells.map((cell) => cell.group)));
  return [...groups].sort(byGroupOrder);
}

// What the summary panel shows: the setup error, or the grid's columns and rows.
export function loadSummaryView(loadData) {
  const setupError = loadData?.setup_error ?? null;
  const rows = setupError ? [] : loadRows(loadData);
  return { setupError, columns: loadColumns(rows), rows };
}

function listLine(title, items) {
  return el("p", { class: "muted" }, `${title}: ${items.join(", ")}`);
}

export function renderLoadSummary(loadData, onSelect) {
  const { setupError, columns: groups, rows } = loadSummaryView(loadData);
  const head = el("tr", {}, el("th", {}, "entry"), ...groups.map((g) => el("th", {}, g)));
  const body = rows.map((row) => {
    const byGroup = Object.fromEntries(row.cells.map((cell) => [cell.group, cell.label]));
    const cells = groups.map((g) => el("td", {}, g in byGroup ? chip(byGroup[g]) : ""));
    const tr = el("tr", { tabindex: "0", role: "link", "data-id": row.id }, el("td", { class: "id" }, row.id), ...cells);
    tr.addEventListener("click", () => onSelect(row.id));
    tr.addEventListener("keydown", (event) => {
      if (event.key === "Enter") onSelect(row.id);
    });
    return tr;
  });
  const notLoaded = loadData.not_loaded ?? {};
  const lists = [];
  if (notLoaded.ours_only?.length) lists.push(listLine("loaded only from ours", notLoaded.ours_only));
  if (notLoaded.ref_only?.length) lists.push(listLine("loaded only from OF3", notLoaded.ref_only));
  const content = setupError
    ? el("p", { class: "banner", role: "alert" }, `feature load setup failed: ${setupError}`)
    : el("div", { class: "gridwrap" }, el("table", { class: "grid" }, el("thead", {}, head), el("tbody", {}, ...body)));
  return el(
    "section",
    { class: "panel", id: "load-summary" },
    el("h2", {}, "Feature load"),
    el("p", { class: "muted" }, `openfold3 commit ${loadData.openfold3_commit ?? "?"}`),
    content,
    ...lists,
  );
}

export function renderLoadEntry(entry) {
  const panel = el("div", { class: "panel", id: "panel-load" }, el("h2", {}, "load"));
  const differences = entry.differences ?? [];
  if (!differences.length) {
    panel.append(el("p", { class: "muted" }, "no differences"));
    return panel;
  }
  const head = el("tr", {}, ...["group", "feature", "detail"].map((h) => el("th", {}, h)));
  const rows = differences.map((d) => el("tr", {}, el("td", { class: "v" }, d.group), el("td", { class: "v" }, d.feature), el("td", {}, d.detail ?? "")));
  panel.append(el("div", { class: "scroller" }, el("table", { class: "diff" }, el("thead", {}, head), el("tbody", {}, ...rows))));
  return panel;
}
