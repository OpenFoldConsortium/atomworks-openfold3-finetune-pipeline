// Phase-grouped 3D cleanup view. phaseRows and atomLayers are pure (tested in Node); everything else
// touches the DOM or window.$3Dmol only when called, never at import time.
import { el } from "./dom.js";
import { loadScriptData } from "./scriptdata.js";

const COUNT_KEYS = ["removed", "created", "renamed", "unresolved_added"];
const CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const REPS = ["cartoon", "sticks", "lines", "spheres"];

const loads = new WeakMap();

export function stepChanged(step, index) {
  return index > 0 && (step.removed > 0 || step.created > 0 || step.renamed > 0 || step.unresolved_added > 0);
}

export function stepRange(view, phaseName) {
  const phase = (view.phases ?? []).find((p) => p.name === phaseName);
  if (!phase || !view.steps) return null;
  const indices = view.steps.map((s, i) => (phase.steps.includes(s.name) ? i : -1)).filter((i) => i >= 0);
  return indices.length ? [Math.min(...indices), Math.max(...indices)] : null;
}

export function phaseRows(view) {
  if (!view.steps || !view.phases) return [];
  const rows = [];
  for (const phase of view.phases) {
    const range = stepRange(view, phase.name);
    if (!range) continue;
    const counts = Object.fromEntries(COUNT_KEYS.map((key) => [key, 0]));
    const steps = [];
    for (let i = range[0]; i <= range[1]; i++) {
      const step = view.steps[i];
      for (const key of COUNT_KEYS) counts[key] += Math.max(0, step[key]);
      if (stepChanged(step, i)) steps.push({ ...step, index: i });
    }
    rows.push({ name: phase.name, range, counts, steps });
  }
  return rows;
}

// Step at which an atom is removed. Without a log nothing is logged, so an unmatched input atom (chain < 0) counts as removed by the end.
function removedStep(view, table, i) {
  if (table.removed_at[i] >= 0) return table.removed_at[i];
  return view.steps === null && table.chain[i] < 0 ? 1 : -1;
}

function classify(view, table, lo, hi, createdAt) {
  const layers = { present: [], removedNow: [], removedEarlier: [], laterRemoved: [] };
  for (let i = 0; i < table.removed_at.length; i++) {
    if (createdAt !== null && createdAt(i) > hi) continue;
    const at = removedStep(view, table, i);
    if (at === -1) layers.present.push(i);
    else if (at > hi) layers.laterRemoved.push(i);
    else if (at >= lo) layers.removedNow.push(i);
    else layers.removedEarlier.push(i);
  }
  return layers;
}

// Layers of the structure after steps lo..hi. Lists index the input table, except `created` (same lists, for the created table)
// and createdNow (created atoms made inside lo..hi that are still present after hi, including ones a later step removes).
export function atomLayers(view, lo, hi) {
  const layers = classify(view, view.input, lo, hi, null);
  const made = (i) => view.created.created_at[i] ?? 1;
  const created = classify(view, view.created, lo, hi, made);
  const createdNow = [...created.present, ...created.laterRemoved].filter((i) => made(i) >= lo).sort((a, b) => a - b);
  return { ...layers, createdNow, created };
}

// One point per issue atom (input or processing findings, from the entry's "issues"); a marker's serial indexes this list.
export function issuePoints(view) {
  return (view.issues ?? []).flatMap((issue) => issue.xyz.map((xyz) => ({ origin: issue.origin, label: issue.label, xyz })));
}

const ISSUE_RESN = "ISS";
export const HOVER_LABEL_MAX = 40; // characters; 3Dmol labels are one line, so longer text is cut here and shown whole below the viewer

export function shortLabel(text) {
  return text.length <= HOVER_LABEL_MAX ? text : `${text.slice(0, HOVER_LABEL_MAX - 1)}…`;
}

// Hover label, as atomworks' view() labels atoms: the issue label for a marker, else chain id, residue and atom name.
export function hoverText(atom, chains, points) {
  if (atom.resn === ISSUE_RESN) return points[atom.serial]?.label ?? "";
  const residue = `${atom.resn}${atom.resi} ${atom.atom}`;
  if (atom.chain === "-") return `removed · ${residue}`;
  const index = CHAIN_LETTERS.indexOf(atom.chain);
  return `${chains[index]?.id ?? atom.chain} ${residue}`;
}

const ISSUE_COLORS = { input: "--warn", processing: "--bad" };
const ISSUE_ZOOM_DISTANCE = 12; // Å around the issues that zooming to them keeps in view
const ISSUE_LIST_CAP = 30;

export function pdbLine(serial, hetero, name, resName, chain, resId, x, y, z, element) {
  const atomName = name.length < 4 ? ` ${name.padEnd(3)}` : name.slice(0, 4);
  const coord = (v) => v.toFixed(3).padStart(8);
  return `${hetero ? "HETATM" : "ATOM  "}${String(serial % 100000).padStart(5)} ${atomName} ${resName.slice(0, 3).padStart(3)} ${chain}${String(resId % 10000).padStart(4)}    ${coord(x)}${coord(y)}${coord(z)}  1.00  0.00          ${element.padStart(2)}`;
}

function chainLetter(index) {
  return index < 0 ? "-" : CHAIN_LETTERS[index % CHAIN_LETTERS.length];
}

function tableLines(table, indices, offset, chainOverride) {
  return indices.map((i) =>
    pdbLine(offset + i, table.hetero[i], table.atom_name[i], table.res_name[i], chainOverride ?? chainLetter(table.chain[i]), table.res_id[i], table.xyz[3 * i], table.xyz[3 * i + 1], table.xyz[3 * i + 2], table.element[i]),
  );
}

function style(rep, extra) {
  if (rep === "cartoon") return { cartoon: extra };
  if (rep === "sticks") return { stick: { radius: 0.18, ...extra } };
  if (rep === "lines") return { line: extra };
  return { sphere: { scale: 0.3, ...extra } };
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function chipText(counts) {
  const parts = [];
  if (counts.removed) parts.push(el("span", { class: "rm" }, `−${counts.removed}`));
  if (counts.created) parts.push(el("span", { class: "add" }, `+${counts.created}`));
  if (counts.renamed) parts.push(`${counts.renamed} renamed`);
  if (counts.unresolved_added > 0) parts.push(`+${counts.unresolved_added} unresolved`);
  if (parts.length === 0) return ["no change"];
  return parts.flatMap((p, i) => (i ? [" ", p] : [p]));
}

function logText(view) {
  const lines = view.steps.map((s, i) => `${String(i).padStart(2)} ${s.phase.padEnd(16)} ${s.name.padEnd(30)} -${s.removed} +${s.created} renamed ${s.renamed} unresolved +${s.unresolved_added}`);
  return lines.join("\n");
}

const LAYER_DEFS = [
  { key: "polymer", label: "Polymer chains", on: true, rep: "cartoon", color: null },
  { key: "ligand", label: "Ligands and other HETATM", on: true, rep: "sticks", color: null },
  { key: "now", label: "Removed in this phase/step", on: true, rep: "sticks", color: "--s3d-now" },
  { key: "later", label: "Removed in a later step", on: true, rep: "sticks", color: "--s3d-later" },
  { key: "earlier", label: "Removed earlier", on: false, rep: "lines", color: "--s3d-gone" },
];

function buildView(container, view, signal, focus) {
  const layers = LAYER_DEFS.map((d) => ({ ...d }));
  const rows = phaseRows(view);
  const noLog = rows.length === 0;
  const state = { kind: "phase", index: 0, lo: 0, hi: 0, label: "", open: new Set(), zoomed: false, showIssues: true, issueModels: [] };
  const points = issuePoints(view);
  const hasIssues = points.length > 0;
  let hoverLabel = null;
  const status = el("div", { class: "s3d-status" }, "");
  const hoverLine = el("div", { class: "s3d-hover", "aria-live": "polite" });
  const viewerDiv = el("div", { class: "s3d-canvas", role: "img", "aria-label": "3D view of the selected cleanup step" });
  const stepsBox = el("div", { class: "s3d-side" });
  const layersBox = el("div", { class: "s3d-side" });
  let viewer = null;

  const chainColor = (i) => (i < 0 ? css("--s3d-gone") : css(`--c${(i % 8) + 1}`));

  function draw() {
    if (!viewer) return;
    const found = atomLayers(view, state.lo, state.hi);
    viewer.removeAllModels();
    viewer.setBackgroundColor(css("--viewer"));
    const map = {};
    view.chains.forEach((_, i) => (map[chainLetter(i)] = chainColor(i)));
    map["-"] = css("--s3d-gone");
    const scheme = { prop: "chain", map };
    const input = view.input;
    const created = view.created;
    const offset = 100000;
    const keptLines = (hetero) => [
      ...tableLines(input, found.present.filter((i) => input.hetero[i] === hetero), 0),
      ...tableLines(created, found.created.present.filter((i) => created.hetero[i] === hetero), offset),
    ];
    const add = (lines, styleSpec) => {
      if (!lines.length) return;
      viewer.addModel(lines.join("\n"), "pdb").setStyle({}, styleSpec);
    };
    for (const [key, hetero] of [["polymer", false], ["ligand", true]]) {
      const layer = layers.find((l) => l.key === key);
      if (layer.on) add(keptLines(hetero), style(layer.rep, { colorscheme: scheme }));
    }
    if (found.createdNow.length) add(tableLines(created, found.createdNow, offset), { stick: { radius: 0.34, colorscheme: scheme, opacity: 0.35 } });
    const extras = { now: [found.removedNow, found.created.removedNow], later: [found.laterRemoved, found.created.laterRemoved], earlier: [found.removedEarlier, found.created.removedEarlier] };
    for (const [key, [inputIdx, createdIdx]] of Object.entries(extras)) {
      const layer = layers.find((l) => l.key === key);
      if (!layer.on) continue;
      const lines = [...tableLines(input, inputIdx, 0, "-"), ...tableLines(created, createdIdx, offset, "-")];
      if (!lines.length) continue;
      const model = viewer.addModel(lines.join("\n"), "pdb");
      const color = css(layer.color);
      model.setStyle({}, style(layer.rep === "cartoon" ? "lines" : layer.rep, { color, ...(layer.rep === "sticks" ? { radius: key === "earlier" ? 0.12 : 0.22 } : {}) }));
      model.setStyle({ resn: "HOH" }, { sphere: { scale: 0.3, color } });
    }
    state.issueModels = [];
    if (hasIssues && state.showIssues) {
      for (const origin of Object.keys(ISSUE_COLORS)) {
        const lines = points.flatMap((point, i) => (point.origin === origin ? [pdbLine(i, true, "X", ISSUE_RESN, "-", 1, ...point.xyz, "X")] : []));
        if (!lines.length) continue;
        const model = viewer.addModel(lines.join("\n"), "pdb");
        model.setStyle({}, { sphere: { radius: 0.9, color: css(ISSUE_COLORS[origin]), opacity: 0.6 } });
        state.issueModels.push(model);
      }
    }
    hoverLabel = null;
    viewer.setHoverable({}, true, (atom) => {
      if (hoverLabel) viewer.removeLabel(hoverLabel);
      const text = hoverText(atom, view.chains, points);
      hoverLine.textContent = text;
      hoverLabel = viewer.addLabel(shortLabel(text), { position: atom, backgroundColor: css("--panel"), fontColor: css("--fg"), borderColor: css("--rule"), borderThickness: 1, fontSize: 12 });
      viewer.render();
    }, () => {
      if (hoverLabel) viewer.removeLabel(hoverLabel);
      hoverLabel = null;
      hoverLine.textContent = "";
      viewer.render();
    });
    if (!state.zoomed) {
      if (focus === "issues" && state.issueModels.length) viewer.zoomTo({ model: state.issueModels });
      else viewer.zoomTo();
      state.zoomed = true;
    }
    viewer.render();
    status.textContent = `${state.label} · drag to rotate`;
  }

  function select(kind, index, lo, hi, label) {
    Object.assign(state, { kind, index, lo, hi, label });
    renderSteps();
    draw();
  }
  const selectPhase = (p) => select("phase", p, rows[p].range[0], rows[p].range[1], `after ${rows[p].name.toLowerCase()}`);
  const selectStep = (i) => select("step", i, i, i, `after ${view.steps[i].name.replace(/_/g, " ")}`);
  const selectBeforeAfter = (which) => select("ba", which, which, which, which === 0 ? "before cleanup" : "after cleanup");

  function renderSteps() {
    if (noLog) {
      const pick = (which, text) => {
        const button = el("button", { type: "button", class: "s3d-step", "aria-current": String(state.kind === "ba" && state.index === which) }, el("span", { class: "t" }, text));
        button.addEventListener("click", () => selectBeforeAfter(which));
        return button;
      };
      stepsBox.replaceChildren(el("h2", {}, "Before / After"), pick(0, "Before"), pick(1, "After"), el("p", { class: "muted s3d-note" }, "No step log for this entry; only the input and the final structure are shown."));
      return;
    }
    const nodes = [el("h2", {}, "Phases")];
    rows.forEach((row, p) => {
      const selected = state.kind === "phase" && state.index === p;
      const expandable = row.steps.length > 1 || (row.steps.length === 1 && view.phases.find((ph) => ph.name === row.name).steps.length > 1);
      const head = el(
        "button",
        { type: "button", class: "s3d-step", "aria-current": String(selected) },
        el("span", { class: "n" }, expandable ? (state.open.has(p) ? "▾" : "▸") : ""),
        el("span", { class: "t" }, row.name, " ", el("span", { class: "chg" }, ...chipText(row.counts))),
      );
      head.addEventListener("click", () => {
        if (selected) {
          if (state.open.has(p)) state.open.delete(p);
          else state.open.add(p);
          renderSteps();
        } else selectPhase(p);
      });
      const group = el("div", { class: "s3d-phase" }, head);
      if (state.open.has(p)) {
        for (const step of row.steps) {
          const current = state.kind === "step" && state.index === step.index;
          const sub = el("button", { type: "button", class: "s3d-step sub", "aria-current": String(current), title: step.description }, el("span", { class: "n" }), el("span", { class: "t" }, step.name.replace(/_/g, " "), " ", el("span", { class: "chg" }, ...chipText(step))), current ? el("span", { class: "d" }, step.description) : null);
          sub.addEventListener("click", () => selectStep(step.index));
          group.append(sub);
        }
      }
      nodes.push(group);
    });
    stepsBox.replaceChildren(...nodes);
  }

  function renderLayers() {
    const nodes = [el("h2", {}, "Layers")];
    for (const layer of layers) {
      const box = el("input", { type: "checkbox", id: `s3d-ly-${layer.key}`, ...(layer.on ? { checked: "" } : {}) });
      box.addEventListener("change", () => {
        layer.on = box.checked;
        draw();
      });
      const swatch = el("span", { class: `sw ${layer.color ? `sw-${layer.key}` : "sw-chains"}` });
      const rep = el("select", { "aria-label": `representation for ${layer.label}` }, ...REPS.map((r) => el("option", r === layer.rep ? { selected: "" } : {}, r)));
      rep.addEventListener("change", () => {
        layer.rep = rep.value;
        draw();
      });
      nodes.push(el("div", { class: "s3d-layer" }, box, el("label", { for: `s3d-ly-${layer.key}` }, swatch, layer.label), rep));
    }
    nodes.push(el("h2", {}, "Chains (after cleanup)"));
    view.chains.forEach((chain, i) => {
      const swatch = el("span", { class: "sw" });
      swatch.dataset.chain = String(i);
      nodes.push(el("div", { class: "s3d-layer plain" }, swatch, el("span", {}, `${chain.id}${chain.hetero ? ` · ${chain.res_name}` : ""}`)));
    });
    nodes.push(el("p", { class: "muted s3d-note" }, "Created atoms (symmetry copies) have a translucent halo. Hover an atom or marker for its name."));
    if (hasIssues) nodes.push(...issueNodes());
    layersBox.replaceChildren(...nodes);
    paintSwatches();
  }

  function issueNodes() {
    const box = el("input", { type: "checkbox", id: "s3d-ly-issues", ...(state.showIssues ? { checked: "" } : {}) });
    box.addEventListener("change", () => {
      state.showIssues = box.checked;
      draw();
    });
    const zoomButton = (text, zoom) => {
      const button = el("button", { type: "button" }, text);
      button.addEventListener("click", () => {
        zoom();
        viewer?.render();
      });
      return button;
    };
    const list = view.issues.slice(0, ISSUE_LIST_CAP).map((issue) =>
      el("li", {}, el("span", { class: `dot ${issue.origin === "input" ? "o" : "r"}` }), issue.label));
    const more = view.issues.length - list.length;
    return [
      el("h2", {}, "Issues"),
      el("div", { class: "s3d-layer" }, box, el("label", { for: "s3d-ly-issues" }, "Mark issue atoms (orange: in the input CIF, red: from processing)")),
      el("div", { class: "s3d-zoom" },
        zoomButton("zoom to issues", () => state.issueModels.length && viewer?.zoomTo({ model: state.issueModels })),
        zoomButton("whole structure", () => viewer?.zoomTo())),
      el("ul", { class: "s3d-issues" }, ...list, ...(more > 0 ? [el("li", { class: "muted" }, `+${more} more`)] : [])),
    ];
  }

  function paintSwatches() {
    layersBox.querySelectorAll(".sw[data-chain]").forEach((s) => (s.style.background = chainColor(Number(s.dataset.chain))));
  }

  function step(dir) {
    if (noLog) return selectBeforeAfter(dir > 0 ? 1 : 0);
    if (state.kind === "phase") {
      const next = state.index + dir;
      if (next >= 0 && next < rows.length) selectPhase(next);
      return undefined;
    }
    if (state.kind === "step") {
      let i = state.index + dir;
      while (i > 0 && i < view.steps.length && !stepChanged(view.steps[i], i)) i += dir;
      if (i > 0 && i < view.steps.length) selectStep(i);
    }
    return undefined;
  }

  const navButton = (text, dir) => {
    const button = el("button", { type: "button" }, text);
    button.addEventListener("click", () => step(dir));
    return button;
  };
  const nav = el("div", { class: "s3d-nav" }, navButton("◀ prev", -1), navButton("next ▶", 1), el("span", { class: "muted" }, "keys: ← → step through"));
  const children = [nav, el("div", { class: "s3d-main" }, stepsBox, el("div", { class: "s3d-box" }, viewerDiv, status, hoverLine), layersBox)];
  if (view.skipped_groups?.length) {
    const items = view.skipped_groups.map((g) => el("li", {}, el("code", {}, g.transformation_id), ` ${g.reason}`));
    children.push(el("div", { class: "s3d-skipped" }, el("strong", {}, `${view.skipped_groups.length} created group${view.skipped_groups.length === 1 ? "" : "s"} not drawn`), el("ul", {}, ...items)));
  }
  if (!noLog) children.push(el("details", {}, el("summary", {}, "Step log for this entry"), el("pre", { class: "s3d-log" }, logText(view))));
  container.replaceChildren(...children);

  renderLayers();
  renderSteps();
  if (!window.$3Dmol) {
    status.textContent = "3D viewer library did not load";
    return;
  }
  viewer = window.$3Dmol.createViewer(viewerDiv, { backgroundColor: css("--viewer"), minimumZoomToDistance: ISSUE_ZOOM_DISTANCE });
  viewer.setHoverDuration(150);
  signal.addEventListener(
    "abort",
    () => {
      const canvas = viewerDiv.querySelector("canvas");
      const gl = canvas?.getContext("webgl2") ?? canvas?.getContext("webgl");
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    },
    { once: true },
  );
  if (noLog) selectBeforeAfter(1);
  else {
    const changed = (name) => rows.findIndex((r) => r.name === name && (r.counts.removed || r.counts.created));
    const first = [changed("Assembly"), changed("Cleanup filters")].find((p) => p >= 0);
    selectPhase(first ?? 0);
  }
  document.addEventListener(
    "keydown",
    (e) => {
      if (["SELECT", "INPUT", "TEXTAREA"].includes(e.target.tagName)) return;
      if (e.key === "ArrowRight") step(1);
      if (e.key === "ArrowLeft") step(-1);
    },
    { signal },
  );
  window.addEventListener("resize", () => viewer.resize(), { signal });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener(
    "change",
    () => {
      paintSwatches();
      draw();
    },
    { signal },
  );
}

export function cancelStepView(container) {
  loads.get(container)?.abort();
  loads.delete(container);
}

// Loads data/validate_3d/<id>.js, which calls window.__reportStep3d(id, data).
export function loadStepData(id) {
  return loadScriptData("__reportStep3d", "validate_3d", id);
}

// Loads the 3D data for one entry and draws the view. A newer call on the same container cancels an older load.
export async function openStepView(container, id, focus) {
  cancelStepView(container);
  const controller = new AbortController();
  loads.set(container, controller);
  container.replaceChildren(el("p", { class: "muted" }, "loading 3D data…"));
  let view;
  try {
    view = await loadStepData(id);
  } catch (error) {
    if (controller.signal.aborted) return;
    container.replaceChildren(el("p", { class: "bad-t" }, `could not load 3D data: ${error.message}`));
    return;
  }
  if (controller.signal.aborted) return;
  buildView(container, view, controller.signal, focus);
}

// The container for one entry: a button that opens the view (via the #id/3d route), or the view itself.
export function drawStepEntry(container, { id, has3d, open, focus }) {
  if (!has3d) return;
  if (open) {
    openStepView(container, id, focus);
    return;
  }
  const button = el("button", { type: "button" }, "Open 3D cleanup view");
  button.addEventListener("click", () => {
    location.hash = `${id}/3d`;
  });
  container.replaceChildren(button);
}
