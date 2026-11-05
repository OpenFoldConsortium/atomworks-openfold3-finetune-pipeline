// User summary: one quiet status line per section (colour only on the dot and counts), then tables with details links.
import { el } from "./dom.js";

// summary.py's DATASET_ROW: the row of dataset-level pipeline validate findings (listed under Run-level, no entry page).
export const DATASET_ROW = "(dataset)";

export function hasDetails(row) {
  return row.id !== DATASET_ROW;
}

// The 3D view of an issue row's structure, zoomed to its issues; null without a 3D view.
export function issueHref(row) {
  return hasDetails(row) && row.has_3d ? `#${row.id}/3d/issues` : null;
}

export const ITEMS_SHOWN = 5;

// A structure's issues: the first ITEMS_SHOWN, and the rest for a "+N more" fold.
export function splitItems(items) {
  return { shown: items.slice(0, ITEMS_SHOWN), hidden: items.slice(ITEMS_SHOWN) };
}

function itemsCell(items) {
  const { shown, hidden } = splitItems(items);
  const lines = (list) => list.map((item) => el("div", {}, item));
  const more = hidden.length ? [el("details", {}, el("summary", {}, `+${hidden.length} more`), ...lines(hidden))] : [];
  return el("td", { class: "items" }, ...lines(shown), ...more);
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function browseOptionText(row) {
  const issues = [["processing", "processing"], ["input", "input"], ["not_checked", "not checked"]]
    .filter(([key]) => row[key] > 0)
    .map(([key, word]) => `${row[key]} ${word}`);
  const cleanup = row.cleanup_steps ? plural(row.cleanup_steps, "cleanup step") : "unchanged by cleanup";
  return `${row.id} — ${issues.length ? issues.join(", ") : "no issues"} · ${cleanup}`;
}

export function notRunText(item) {
  const why = item.message.replace(/^skipped:\s*/, "");
  return `Not run: ${item.check}${item.title ? ` ${item.title}` : ""} (${why})`;
}

export function noteText(note) {
  return `Note: ${note}`;
}

export function truncatedNote(summary, count) {
  return `showing first ${summary.row_cap} of ${count}; build a report on a subset with finetune-report --ids <file>`;
}

export function validationNote(summary) {
  return summary.validation_skipped ? "Validation stage was skipped; load checks not included" : null;
}

export function statusLines(summary) {
  const { processing_issues: issues, input, not_checked } = summary.counts;
  const lines = [
    issues === 0
      ? { dot: "g", text: "Processing introduced no issues", count: null }
      : { dot: "r", text: `Processing: ${issues} issue(s) not in the input CIF (likely a pipeline bug; please report)`, count: issues },
  ];
  if (input > 0) lines.push({ dot: "o", text: `Input structures: ${input} with issues already in the CIF (kept)`, count: input });
  if (not_checked > 0) lines.push({ dot: "n", text: `Not checked: ${not_checked} structures`, count: not_checked });
  return lines;
}

export function heading(summary) {
  return `${summary.structures} structures processed, ${summary.counts.dropped ?? summary.dropped.length} dropped`;
}

function routeLink(href, text, navigate) {
  const link = el("a", { href }, text);
  link.addEventListener("click", (event) => {
    event.preventDefault();
    navigate(href.slice(1));
  });
  return link;
}

function detailsLink(id, navigate) {
  return routeLink(`#${id}`, "details", navigate);
}

// The count inside the line text is wrapped so only it takes the line's colour.
function statusLine(line) {
  const parts = line.count === null ? [line.text] : splitCount(line.text, String(line.count));
  const children = parts.map((part) => (typeof part === "string" ? part : el("span", { class: `count ${line.dot}-t` }, part.count)));
  return el("p", { class: "status-line" }, el("span", { class: `dot ${line.dot}` }), ...children);
}

function splitCount(text, count) {
  const at = text.indexOf(count);
  return [text.slice(0, at), { count }, text.slice(at + count.length)];
}

function note(summary, count, shown) {
  return summary.truncated && count > shown ? el("p", { class: "muted" }, truncatedNote(summary, count)) : null;
}

function table(headers, rows) {
  const head = el("tr", {}, ...headers.map((h) => el("th", {}, h)));
  return el("div", { class: "scroller" }, el("table", { class: "grid" }, el("thead", {}, head), el("tbody", {}, ...rows)));
}

function issueTable(summary, rows, count, navigate) {
  const body = rows.map((row) => {
    const href = issueHref(row);
    const name = href ? routeLink(href, row.id, navigate) : row.id;
    return el("tr", {}, el("td", { class: "id" }, name), itemsCell(row.items), el("td", {}, hasDetails(row) ? detailsLink(row.id, navigate) : ""));
  });
  return [table(["structure", "issues", ""], body), note(summary, count, rows.length)];
}

function droppedTable(summary) {
  const body = summary.dropped.map((row) => el("tr", {}, el("td", { class: "id" }, row.id), el("td", {}, row.stage), el("td", {}, row.reason)));
  return [table(["structure", "stage", "reason"], body), note(summary, summary.counts.dropped ?? summary.dropped.length, summary.dropped.length)];
}

export function cleanupRowView(row) {
  const structures = row.structures.map((id) => ({ id, href: row.has_3d.includes(id) ? `#${id}/3d` : null }));
  return { step: row.step, count: row.count, structures };
}

// A dropdown of every structure (opens its page, where prev/next step through the rest) and a link to the first.
function browse(summary, navigate) {
  const select = el("select", { id: "browse", "aria-label": "open a structure" },
    el("option", { value: "", selected: "" }, "choose a structure…"),
    ...summary.all.map((row) => el("option", { value: row.id }, browseOptionText(row))));
  select.addEventListener("change", () => select.value && navigate(select.value));
  const first = summary.all[0].id;
  const capped = summary.truncated && summary.structures > summary.all.length ? ` (first ${summary.all.length} of ${summary.structures})` : "";
  return el("p", { class: "browse" }, el("label", { for: "browse" }, `Browse structures${capped} `), select, " or ", routeLink(`#${first}`, "start with the first structure →", navigate));
}

function cleanupTable(summary, navigate) {
  const body = summary.cleanup.map((row) => {
    const view = cleanupRowView(row);
    const links = view.structures.map(({ id, href }) => {
      if (href === null) return el("span", {}, id);
      const link = el("a", { href }, id);
      link.addEventListener("click", (event) => {
        event.preventDefault();
        navigate(`${id}/3d`);
      });
      return link;
    });
    return el("tr", {}, el("td", {}, view.step), el("td", { class: "id" }, String(view.count)),
      el("td", {}, ...links.flatMap((link, i) => (i ? [" ", link] : [link]))));
  });
  const cut = summary.cleanup.some((row) => row.count > row.structures.length);
  return [table(["step", "structures changed", "structures"], body), cut ? el("p", { class: "muted" }, truncatedNote(summary, Math.max(...summary.cleanup.map((r) => r.count)))) : null];
}

export function renderUserSummary(container, summary, navigate) {
  if (!summary) {
    container.replaceChildren(el("p", { class: "bad-t", role: "alert" }, "the summary could not be built; see the per-structure pages"));
    return;
  }
  const lines = statusLines(summary);
  const sections = [
    [lines[0], summary.processing, summary.counts.processing],
    ...(summary.counts.input > 0 ? [[lines[1], summary.input, summary.counts.input]] : []),
    ...(summary.counts.not_checked > 0 ? [[lines[lines.length - 1], summary.not_checked, summary.counts.not_checked]] : []),
  ];
  const children = [el("h2", {}, heading(summary))];
  if (summary.all?.length > 0) children.push(browse(summary, navigate));
  const skipped = validationNote(summary);
  if (skipped) children.push(el("p", { class: "muted" }, skipped));
  for (const [line, rows, count] of sections) {
    children.push(statusLine(line));
    if (rows.length > 0) children.push(...issueTable(summary, rows, count, navigate));
  }
  for (const item of summary.not_run ?? []) children.push(statusLine({ dot: "n", text: notRunText(item), count: null }));
  for (const note of summary.notes ?? []) children.push(statusLine({ dot: "n", text: noteText(note), count: null }));
  if (summary.dropped.length > 0) children.push(el("h3", {}, "Dropped"), ...droppedTable(summary));
  if (summary.cleanup.length > 0) children.push(el("h3", {}, "What cleanup changed"), ...cleanupTable(summary, navigate));
  // The tables' notes are null when nothing is cut; replaceChildren would print them.
  container.replaceChildren(...children.filter((child) => child !== null));
}
