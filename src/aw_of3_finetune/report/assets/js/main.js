// Reads the inline REPORT_DATA (manifest and summary-level sections), then routes between the summary and one
// entry (#id, #id/3d), whose full data showEntry loads from data/entries/<id>.js.
// Later tasks import their figure modules here and fill the containers entry.js creates.
import { el } from "./dom.js";
import { cancelShowEntry, renderRunLevel, renderWhy, showEntry } from "./entry.js";
import { cancelStepView } from "./steps3d.js";
import { parseRoute, renderSummary } from "./summary.js";
import { renderUserSummary } from "./usersummary.js";

const data = window.REPORT_DATA ?? null;
const manifest = data?.manifest ?? null;
if (manifest) {
  document.title = manifest.title;
  document.getElementById("title").textContent = manifest.title;
  const meta = document.getElementById("meta");
  if (manifest.run) meta.textContent = Object.entries(manifest.run).map(([k, v]) => `${k} ${v ?? "?"}`).join(" · ");

  const sections = {};
  for (const name of manifest.sections) {
    if (name !== "load" && data[name]) sections[name] = data[name];
  }
  const loadData = data.load ?? null;

  const navigate = (id) => {
    location.hash = id;
  };
  if (manifest.summary === "user") {
    // The user summary already says why input findings are kept, and lists dataset-level errors and skipped checks.
    renderUserSummary(document.getElementById("summary"), sections.validate?.user_summary ?? null, navigate);
  } else {
    renderSummary(document.getElementById("summary"), sections, navigate, loadData, manifest.has_3d ?? []);
    renderWhy(document.getElementById("why"), sections);
    renderRunLevel(document.getElementById("run-level"), sections);
  }

  const show = () => {
    const route = parseRoute(location.hash);
    const container = document.getElementById("entry");
    const leave = () => {
      cancelShowEntry();
      const view3d = container.querySelector("#validate-3d");
      if (view3d) cancelStepView(view3d);
    };
    document.body.dataset.view = route ? "entry" : "summary";
    window.scrollTo(0, 0);
    if (!route) {
      leave();
      return container.replaceChildren();
    }
    if (!manifest.entries.includes(route.id)) {
      leave();
      return container.replaceChildren(el("p", { class: "bad-t" }, `entry not found: ${route.id}`));
    }
    showEntry(container, { id: route.id, ids: manifest.entries, sections, loadData, navigate, has3d: manifest.has_3d ?? [], view: route.view, focus: route.focus });
  };
  window.addEventListener("hashchange", show);
  show();
} else {
  document.getElementById("title").textContent = "no report data found";
}
