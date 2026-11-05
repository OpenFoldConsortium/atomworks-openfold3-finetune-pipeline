// Label -> CSS class and chip text. Pure: the data decides the label, this only styles it.
const CLASSES = {
  same: "same",
  equivalent: "equiv",
  expected: "expect",
  open: "expect",
  exception: "exc",
  unexplained: "bad",
  none: "muted",
  input: "warn",
  not_checked: "muted",
};
const TEXTS = {
  same: "✓",
  equivalent: "≈",
  expected: "exp",
  open: "open",
  exception: "exc",
  unexplained: "✗",
  none: "–",
  input: "!",
  not_checked: "–",
};

export function labelClass(label) {
  return CLASSES[label] ?? "muted";
}

export function labelText(label) {
  return TEXTS[label] ?? String(label ?? "");
}

export const LABELS = Object.keys(CLASSES);

// Most to least severe; the worst label present is the one a summary shows.
export const SEVERITY = ["unexplained", "open", "exception", "input", "not_checked", "expected", "equivalent", "same", "none"];

export function chipAttrs(label) {
  return { class: `chip ${labelClass(label)}`, title: String(label), "aria-label": String(label) };
}
