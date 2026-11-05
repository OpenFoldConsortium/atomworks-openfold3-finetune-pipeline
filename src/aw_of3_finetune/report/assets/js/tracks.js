// Chain tracks: one row pair per polymer chain, a cell per residue, then one chip per ligand chain.
// Colors, glyphs and counts come from the data's states; nothing here judges a chain.
import { el, svgEl } from "./dom.js";

const GOOD_CLASSES = { ok: "st-ok", unresolved: "st-unres", removed: "st-removed", assembly_copy: "st-copy" };
const LABEL_W = 230;
const TRACK_H = 12;
const TRACK_H_LETTERS = 16;
const PAIR_GAP = 3;
const CHAIN_GAP = 18;
const MIN_CELL_PX = 0.8;
const LETTER_CELL_PX = 11;
const RULER_Y = 30;
const NOTE_W = 130;

export function stateClass(state) {
  return GOOD_CLASSES[state] ?? "st-bad";
}

export function letterClass(state) {
  return stateClass(state) === "st-ok" || stateClass(state) === "st-bad" ? "tr-letter-bg" : "tr-letter-fg";
}

function asList(value) {
  if (value === null || value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

export function pairChains(tracks) {
  const ours = tracks.ours ?? [];
  const polymer = (chain) => chain.type !== "LIGAND";
  const used = new Set();
  const pairs = [];
  for (const ref of (tracks.ref ?? []).filter(polymer)) {
    const wanted = asList(ref.paired);
    const ourChain = ours.find((c) => polymer(c) && wanted.includes(c.id)) ?? null;
    if (ourChain) used.add(ourChain.id);
    pairs.push({ ref, ours: ourChain });
  }
  for (const c of ours.filter(polymer)) if (!used.has(c.id)) pairs.push({ ref: null, ours: c });
  return pairs;
}

const isLigand = (chain) => chain.type === "LIGAND";
const statesOf = (chain) => (chain ? chain.res.map((r) => r[2]) : []);
const molOf = (chain) => chain.ref_mol ?? chain.res[0]?.[1] ?? "?";
const GLYPHS = { ok: "✓", bad: "✗", removed: "", copy: "" };

function chipKind(states) {
  const classes = states.map(stateClass);
  if (classes.includes("st-bad")) return "bad";
  if (classes.length && classes.every((c) => c === "st-removed")) return "removed";
  if (classes.length && classes.every((c) => c === "st-copy")) return "copy";
  return "ok";
}

function chipText(kind, mol, refId, ourId) {
  if (kind === "removed") return `${mol} ${refId} removed`;
  const pair = `${mol} ${refId ?? "—"}→${ourId ?? "—"}`;
  return kind === "copy" ? `${pair} copy` : `${pair} ${GLYPHS[kind]}`;
}

function makeChip(ref, ours) {
  const kind = chipKind([...statesOf(ref), ...statesOf(ours)]);
  const refId = ref?.id ?? null;
  const ourId = ours?.id ?? null;
  const mol = molOf(ref ?? ours);
  return { refMol: mol, refId, ourId, kind, glyph: GLYPHS[kind], text: chipText(kind, mol, refId, ourId) };
}

export function ligandChips(tracks) {
  const ours = (tracks?.ours ?? []).filter(isLigand);
  const used = new Set();
  const chips = [];
  for (const ref of (tracks?.ref ?? []).filter(isLigand)) {
    const wanted = asList(ref.paired);
    const ourChain = ours.find((c) => wanted.includes(c.id)) ?? null;
    if (ourChain) used.add(ourChain.id);
    chips.push(makeChip(ref, ourChain));
  }
  for (const c of ours) if (!used.has(c.id)) chips.push(makeChip(null, c));
  return chips;
}

export function chainNote(chain) {
  if (!chain) return null;
  const classes = statesOf(chain).map(stateClass);
  const bad = classes.filter((c) => c === "st-bad").length;
  const ok = classes.filter((c) => c === "st-ok").length;
  if (bad) return { text: `✗ ${bad} residues off`, cls: "bad-t" };
  if (ok) return { text: `✓ ${ok} match`, cls: "ok-t" };
  return null;
}

function rowLabel(who, chain) {
  if (!chain) return `${who} · no chain`;
  return `${who} chain ${chain.id} · ${chain.label} (auth ${chain.auth}) · ent ${chain.entity}`;
}

export function trackLayout(tracks, width, { letters = false, who = ["OF3", "ours"] } = {}) {
  const pairs = pairChains(tracks);
  const ids = pairs.flatMap((p) => [p.ref, p.ours].filter(Boolean).flatMap((c) => c.res.map((r) => r[0])));
  let minRes = 1;
  let maxRes = 10;
  for (const id of ids) {
    if (id < minRes) minRes = id;
    if (id > maxRes) maxRes = id;
  }
  const span = maxRes - minRes + 1;
  const px = letters ? LETTER_CELL_PX : Math.max(width / span, MIN_CELL_PX);
  const trackH = letters ? TRACK_H_LETTERS : TRACK_H;
  const step = span > 200 ? 50 : 20;
  const ruler = [];
  for (let r = step; r <= maxRes; r += step) ruler.push({ resId: r, x: (r - minRes + 0.5) * px });
  const rows = [];
  let y = RULER_Y;
  for (const pair of pairs) {
    [[who[0], pair.ref], [who[1], pair.ours]].forEach(([name, chain], k) => {
      const cells = chain ? chain.res.map(([resId, resName, state]) => ({ resId, resName, state, x: (resId - minRes) * px, w: px, stateClass: stateClass(state) })) : [];
      const note = k === 1 ? chainNote(chain) : null;
      rows.push({ who: name, chainId: chain?.id ?? null, label: rowLabel(name, chain), y: y + k * (trackH + PAIR_GAP), cells, note });
    });
    y += 2 * trackH + PAIR_GAP + CHAIN_GAP;
  }
  const noteX = LABEL_W + span * px + 8;
  return { ruler, rows, px, trackH, labelW: LABEL_W, noteX, width: noteX + NOTE_W, height: y };
}

export function drawTracks(container, tracks, { letters = false, who = ["OF3", "ours"], width = 700 } = {}) {
  const layout = trackLayout(tracks, width, { letters, who });
  const svg = svgEl("svg", { width: layout.width, height: layout.height, viewBox: `0 0 ${layout.width} ${layout.height}`, role: "img", "aria-label": `chain tracks ${who[0]} vs ${who[1]}` });
  svg.append(svgEl("text", { x: 0, y: 14, class: "tr-axis" }, "residue number →"));
  for (const tick of layout.ruler) {
    const x = layout.labelW + tick.x;
    svg.append(svgEl("line", { x1: x, x2: x, y1: 18, y2: 24, class: "tr-tick" }), svgEl("text", { x, y: 14, "text-anchor": "middle", class: "tr-axis" }, tick.resId));
  }
  for (const row of layout.rows) {
    svg.append(svgEl("text", { x: 0, y: row.y + layout.trackH - 2, class: row.who === who[0] ? "tr-label tr-ref" : "tr-label" }, row.label));
    for (const cell of row.cells) {
      const rect = svgEl("rect", { x: layout.labelW + cell.x, y: row.y, width: Math.max(cell.w - (cell.w > 3 ? 0.5 : 0), MIN_CELL_PX), height: layout.trackH, class: cell.stateClass });
      rect.append(svgEl("title", {}, `${row.who} chain ${row.chainId}: ${cell.resName} ${cell.resId} · ${cell.state}`));
      svg.append(rect);
      if (letters) svg.append(svgEl("text", { x: layout.labelW + cell.x + cell.w / 2, y: row.y + layout.trackH - 3, "text-anchor": "middle", class: `tr-letter ${letterClass(cell.state)}` }, cell.resName.slice(0, 1)));
    }
    if (row.note) svg.append(svgEl("text", { x: layout.noteX, y: row.y + layout.trackH - 2, class: `tr-note ${row.note.cls}` }, row.note.text));
  }
  container.replaceChildren(svg);
}

const LEGEND = [["st-ok", "ok"], ["st-unres", "unresolved"], ["st-bad", "misplaced / missing"], ["st-copy", "assembly copy"], ["st-removed", "removed"]];

function drawLigandChips(chips, who) {
  if (chips.length === 0) return null;
  const items = chips.map((c) => el("span", { class: `lchip lchip-${c.kind}`, title: `${c.refMol}: ${who[0]} chain ${c.refId ?? "—"} → ${who[1]} chain ${c.ourId ?? "—"} · ${c.kind}` }, c.text));
  return el("div", { class: "lchips" }, el("span", { class: "muted" }, `ligands (${who[0]} chain → ${who[1]} chain)`), ...items);
}

export function drawTracksPanel(container, tracks, { who = ["OF3", "ours"] } = {}) {
  if (!tracks) return;
  const hasPolymers = pairChains(tracks).length > 0;
  const chips = drawLigandChips(ligandChips(tracks), who);
  if (!hasPolymers && !chips) return;
  const holder = el("div", { class: "scroller" });
  const toggle = el("input", { type: "checkbox" });
  const redraw = () => drawTracks(holder, tracks, { letters: toggle.checked, who });
  toggle.addEventListener("change", redraw);
  const swatches = LEGEND.map(([cls, text]) => el("span", {}, el("span", { class: `sw sw-${cls}` }), text));
  const legend = hasPolymers ? el("div", { class: "legend" }, el("label", {}, toggle, " residue letters"), ...swatches) : null;
  container.replaceChildren(...[legend, hasPolymers ? holder : null, chips].filter(Boolean));
  if (hasPolymers) redraw();
}
