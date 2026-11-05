// Cache-field comparison table: section rows, folded repeated-across-chains rows, same/equivalent hidden by default.
import { el, chip } from "./dom.js";

const QUIET_LABELS = ["same", "equivalent"];

export function formatValue(value) {
  if (value === null || value === undefined) return "null";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

function foldedOurs(row) {
  if (row.chains.length === row.ours.length) return row.chains.map((chain, k) => `${chain}: ${formatValue(row.ours[k])}`).join("\n");
  return row.ours.map(formatValue).join(", ");
}

export function cacheTableRows(cacheRows, { showSame = false } = {}) {
  const rows = [];
  let lastSection = null;
  let hidden = 0;
  for (const row of cacheRows) {
    if (!showSame && QUIET_LABELS.includes(row.label)) {
      hidden += 1;
      continue;
    }
    if (row.section !== lastSection) {
      rows.push({ kind: "section", section: row.section });
      lastSection = row.section;
    }
    const folded = row.chains !== undefined;
    rows.push({
      kind: "row",
      field: row.field,
      chains: folded ? row.chains.join(", ") : row.chain,
      ours: folded ? foldedOurs(row) : formatValue(row.ours),
      ref: folded ? row.ref.map(formatValue).join(", ") : formatValue(row.ref),
      label: row.label,
      reason: row.reason ?? "",
    });
  }
  return { rows, hidden };
}

export function drawCacheTable(container, cacheRows, { showSame = false } = {}) {
  if (!cacheRows?.length) return;
  const render = (show) => {
    const { rows, hidden } = cacheTableRows(cacheRows, { showSame: show });
    const body = rows.map((r) =>
      r.kind === "section"
        ? el("tr", { class: "secrow" }, el("td", { colspan: 6 }, r.section))
        : el("tr", {}, el("td", { class: "v" }, r.field), el("td", { class: "v" }, r.chains ?? ""), el("td", { class: "v pre" }, r.ours), el("td", { class: "v" }, r.ref), el("td", {}, chip(r.label)), el("td", { class: "why" }, r.reason)),
    );
    if (hidden) body.push(el("tr", {}, el("td", { colspan: 6, class: "muted" }, `${hidden} same/equivalent fields hidden`)));
    const head = el("tr", {}, ...["field", "chains", "ours", "OF3", "label", "why"].map((h) => el("th", {}, h)));
    const toggle = el("input", { type: "checkbox", id: "show-same" });
    toggle.checked = show;
    toggle.addEventListener("change", () => {
      render(toggle.checked);
      container.querySelector("#show-same").focus();
    });
    container.replaceChildren(el("label", { class: "muted" }, toggle, " show same/equivalent"), el("div", { class: "scroller" }, el("table", { class: "diff" }, el("thead", {}, head), el("tbody", {}, ...body))));
  };
  render(showSame);
}
