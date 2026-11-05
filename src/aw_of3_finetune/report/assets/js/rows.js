// Generic row table for the validate figures (ligand checks, bonds, covalent links).
import { el, chip } from "./dom.js";
import { SEVERITY } from "./labels.js";

export function formatCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function bondRows(bonds) {
  return [
    ...(bonds?.residue_bonds ?? []).map((r) => ({ kind: "residue bond", ...r })),
    ...(bonds?.bond_lengths ?? []).map((r) => ({ kind: "bond length", ...r })),
  ];
}

export function drawRowsTable(container, rows, columns, title = null) {
  if (!rows?.length) return;
  const cell = (row, col) => {
    if (col.kind === "label") return el("td", {}, chip(row[col.key]));
    return el("td", { class: col.kind === "why" ? "why" : "v" }, formatCell(row[col.key]));
  };
  const head = el("tr", {}, ...columns.map((c) => el("th", {}, c.header)));
  const body = rows.map((row) => el("tr", {}, ...columns.map((c) => cell(row, c))));
  const parts = title ? [el("h3", {}, title)] : [];
  container.append(...parts, el("div", { class: "scroller" }, el("table", { class: "diff" }, el("thead", {}, head), el("tbody", {}, ...body))));
}

// One summary per ligand chain: worst label, passed count, and the rows that did not pass.
export function ligandSummaries(rows) {
  const byLigand = new Map();
  for (const row of rows ?? []) {
    const key = `${row.chain}\u0000${row.res_name}`;
    if (!byLigand.has(key)) byLigand.set(key, { chain: row.chain, res_name: row.res_name, rows: [] });
    byLigand.get(key).rows.push(row);
  }
  return [...byLigand.values()].map(({ chain, res_name, rows: ligandRows }) => {
    const labels = ligandRows.map((r) => r.label);
    return {
      chain,
      res_name,
      label: SEVERITY.find((label) => labels.includes(label)) ?? null,
      n_checks: ligandRows.length,
      n_passed: ligandRows.filter((r) => r.passed === true).length,
      failed: ligandRows.filter((r) => r.passed !== true),
    };
  });
}

const formatNumber = (value) => (typeof value === "number" ? String(Number(value.toFixed(2))) : null);

function failedCheckItem(row) {
  const group = String(row.check ?? "").replace(/^validity\.ligand_/, "");
  const value = formatNumber(row.value);
  const limit = formatNumber(row.threshold);
  const numbers = [value !== null ? `value ${value}` : null, limit !== null ? `limit ${limit}` : null].filter(Boolean).join(", ");
  return el("li", {}, chip(row.label), " ", el("span", { class: "v" }, row.test), el("span", { class: "muted" }, ` (${group}${numbers ? `; ${numbers}` : ""})`),
    row.reason ? el("div", { class: "why" }, row.reason) : null);
}

// Ligand checks: one row per ligand with only its failed checks, then every check row behind a dropdown.
export function drawLigandChecks(container, rows) {
  if (!rows?.length) return;
  const summaries = ligandSummaries(rows);
  const nFailing = summaries.filter((s) => s.failed.length).length;
  const head = el("tr", {}, ...["chain", "residue", "result", "passed", "failed checks"].map((h) => el("th", {}, h)));
  const body = summaries.map((s) =>
    el("tr", {}, el("td", { class: "v" }, s.chain), el("td", { class: "v" }, s.res_name), el("td", {}, chip(s.label)),
      el("td", { class: "v" }, `${s.n_passed}/${s.n_checks}`),
      el("td", {}, s.failed.length ? el("ul", { class: "failed-checks" }, ...s.failed.map(failedCheckItem)) : el("span", { class: "muted" }, "none"))),
  );
  const all = el("details", { class: "all-checks" }, el("summary", {}, `All ligand checks (${rows.length} rows)`));
  drawRowsTable(all, rows, LIGAND_COLUMNS);
  container.append(
    el("h3", {}, "Ligand checks"),
    el("p", { class: "muted" }, `${summaries.length} ligand(s), ${nFailing} with failed checks.`),
    el("div", { class: "scroller" }, el("table", { class: "diff" }, el("thead", {}, head), el("tbody", {}, ...body))),
    all,
  );
}

export const LIGAND_COLUMNS = [
  { key: "chain", header: "chain" }, { key: "res_name", header: "residue" }, { key: "test", header: "test" },
  { key: "passed", header: "passed" }, { key: "value", header: "value" }, { key: "threshold", header: "threshold" },
  { key: "label", header: "label", kind: "label" }, { key: "reason", header: "reason", kind: "why" },
];
export const BOND_COLUMNS = [
  { key: "kind", header: "kind" }, { key: "chain", header: "chain" }, { key: "res_name", header: "residue" }, { key: "res_id", header: "id" },
  { key: "atom_1", header: "atom 1" }, { key: "atom_2", header: "atom 2" }, { key: "distance", header: "distance" },
  { key: "label", header: "label", kind: "label" }, { key: "reason", header: "reason", kind: "why" },
];
export const LINK_COLUMNS = [
  { key: "conn_id", header: "id" }, { key: "conn_type", header: "type" }, { key: "copy", header: "copy" },
  { key: "atom_1", header: "atom 1" }, { key: "atom_2", header: "atom 2" }, { key: "input_distance", header: "input Å" }, { key: "state", header: "state" },
  { key: "label", header: "label", kind: "label" }, { key: "reason", header: "reason", kind: "why" },
];
