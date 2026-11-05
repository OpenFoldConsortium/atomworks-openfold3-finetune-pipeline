// Ligand bond matrices (upper triangle ours, lower OF3) and the cross-chain bond list.
import { el, svgEl } from "./dom.js";

const MIN_CELL = 6;
const BOND_CLASSES = { 1: "b1", 2: "b2", 5: "b5", 6: "b6" };
const BOND_NAMES = { 1: "single", 2: "double", 5: "aromatic single", 6: "aromatic double" };

export function bondClass(type) {
  return BOND_CLASSES[type] ?? "bn";
}

export function bondKey(a, b) {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

export function matrixNames(matrix) {
  return [...matrix.names_ref, ...matrix.names_ours.filter((n) => !matrix.names_ref.includes(n))];
}

export function matrixCells(matrix) {
  const names = matrixNames(matrix);
  const flagged = new Set([...matrix.bonds_one_side.map((b) => b.key), ...matrix.bonds_type_differ]);
  const cells = [];
  for (let i = 0; i < names.length; i++) {
    for (let j = 0; j < names.length; j++) {
      if (i === j) continue;
      const side = j > i ? "ours" : "ref";
      const key = bondKey(names[i], names[j]);
      const type = (side === "ours" ? matrix.bonds_ours : matrix.bonds_ref)[key];
      if (type === undefined) continue;
      cells.push({ i, j, side, type, outlined: flagged.has(key) });
    }
  }
  return cells;
}

export function cellSize(count) {
  const size = count > 40 ? 8 : count > 25 ? 11 : 14;
  return Math.max(size, MIN_CELL);
}

export function drawLigandMatrix(container, matrix) {
  const names = matrixNames(matrix);
  const n = names.length;
  const cell = cellSize(n);
  const left = 34;
  const top = 34;
  const size = left + n * cell + 4;
  const oneSide = new Set(matrix.atoms_one_side.map((a) => a.name));
  const svg = svgEl("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, role: "img", "aria-label": `bond matrix ${matrix.mol}` });
  const font = Math.min(9, cell - 2);
  names.forEach((name, k) => {
    const cls = oneSide.has(name) ? "mx-name mx-bad" : "mx-name";
    svg.append(svgEl("text", { x: left - 2, y: top + k * cell + cell - 2, "text-anchor": "end", "font-size": font, class: cls }, name));
    const x = left + k * cell + cell - 2;
    svg.append(svgEl("text", { x, y: top - 2, transform: `rotate(-70 ${x} ${top - 2})`, "font-size": font, class: cls }, name));
  });
  svg.append(svgEl("line", { x1: left, y1: top, x2: left + n * cell, y2: top + n * cell, class: "mx-diag" }));
  for (const c of matrixCells(matrix)) {
    const rect = svgEl("rect", { x: left + c.j * cell + 0.5, y: top + c.i * cell + 0.5, width: cell - 1, height: cell - 1, class: `${bondClass(c.type)}${c.outlined ? " mx-out" : ""}` });
    rect.append(svgEl("title", {}, `${names[c.i]}–${names[c.j]} · ${c.side === "ours" ? "ours" : "OF3"}: ${BOND_NAMES[c.type] ?? c.type}${c.outlined ? " (differs)" : ""}`));
    svg.append(rect);
  }
  const caption = `${matrix.mol} · ours ${matrix.ours_chains.join(",")} · OF3 ${matrix.ref_chains.join(",")}`;
  container.append(el("div", { class: "mat" }, el("div", { class: "cap" }, caption), el("div", { class: "scroller" }, svg)));
}

export function crossBondClass(state) {
  return state === "same" ? "same" : "bad";
}

export function drawCrossBonds(container, crossBonds) {
  if (!crossBonds?.bonds?.length) return;
  const rows = crossBonds.bonds.map((b) =>
    el("tr", {}, el("td", { class: "v" }, b.ours ?? "–"), el("td", { class: "v" }, b.ref ?? "–"), el("td", { class: "v" }, `${b.type_ours ?? "–"} / ${b.type_ref ?? "–"}`), el("td", {}, el("span", { class: `chip ${crossBondClass(b.state)}` }, b.state))),
  );
  const head = el("tr", {}, ...["ours", "OF3", "type ours / OF3", "state"].map((h) => el("th", {}, h)));
  container.append(el("div", { class: "mat" }, el("div", { class: "cap" }, "cross-chain bonds"), el("div", { class: "scroller" }, el("table", { class: "diff" }, el("thead", {}, head), el("tbody", {}, ...rows)))));
}

export function drawLigands(container, ligands, crossBonds) {
  const wrap = el("div", { class: "mats" });
  for (const matrix of ligands ?? []) drawLigandMatrix(wrap, matrix);
  drawCrossBonds(wrap, crossBonds);
  if (wrap.childElementCount) container.replaceChildren(wrap);
}
