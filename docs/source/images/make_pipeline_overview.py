"""Draw pipeline-overview.svg, the figure in explanation/how-it-works.md.

Edit this script and rerun it, rather than editing the SVG by hand:

    python docs/source/images/make_pipeline_overview.py
"""

import math
from pathlib import Path

OUT = Path(__file__).with_name("pipeline-overview.svg")
W = 720
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

# fill, stroke, text per step
STEPS = {
    "annotate": ("#eff6ff", "#dbeafe", "#3b82f6", "#1e3a5f"),
    "search": ("#faf5ff", "#f3e8ff", "#a855f7", "#3b0764"),
    "handoff": ("#fefce8", "#fef9c3", "#ca8a04", "#713f12"),
    "check": ("#f0fdf4", "#dcfce7", "#22c55e", "#14532d"),
}
GRAY = "#8a8a8a"
GROUP_W, NODE_W, NODE_H, PAD = 300, 260, 56, 20
COL_X = (20, 400)
GAP = 36  # vertical space between nodes in a group

out = []


def group(step, x, y, h, title):
    group_fill, _, stroke, text = STEPS[step]
    out.append(f'<rect x="{x}" y="{y}" width="{GROUP_W}" height="{h}" rx="14" fill="{group_fill}" stroke="{stroke}" stroke-width="1.5"/>')
    out.append(f'<text x="{x + 16}" y="{y + 26}" font-family="{SANS}" font-size="16" font-weight="600" fill="{text}">{title}</text>')


def node(step, x, y, title, sub=None, optional=False):
    _, fill, stroke, text = STEPS[step]
    dash = ' stroke-dasharray="6 4"' if optional else ""
    out.append(f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    cx = x + NODE_W / 2
    if sub is None:
        out.append(f'<text x="{cx}" y="{y + NODE_H / 2}" text-anchor="middle" dominant-baseline="central" font-family="{MONO}" font-size="14" font-weight="600" fill="{text}">{title}</text>')
    else:
        out.append(f'<text x="{cx}" y="{y + 20}" text-anchor="middle" dominant-baseline="central" font-family="{MONO}" font-size="14" font-weight="600" fill="{text}">{title}</text>')
        out.append(f'<text x="{cx}" y="{y + 39}" text-anchor="middle" dominant-baseline="central" font-family="{SANS}" font-size="12.5" fill="{text}" opacity="0.85">{sub}</text>')


def arrow(x1, y1, x2, y2, color=GRAY, width=2, dashed=False, head=10):
    """A line from (x1, y1) whose arrow tip sits exactly at (x2, y2)."""
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="{width}"{dash}/>')
    px, py = -math.sin(ang), math.cos(ang)
    hw = head * 0.55
    pts = [(x2, y2), (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)]
    out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="{color}"/>')


def block_arrow(cx, top, bottom, step):
    """A filled block arrow pointing down, from top to bottom."""
    _, _, stroke, _ = STEPS[step]
    shaft, headw = 7, 15
    neck = bottom - 12
    pts = [(cx - shaft, top), (cx + shaft, top), (cx + shaft, neck), (cx + headw, neck),
           (cx, bottom), (cx - headw, neck), (cx - shaft, neck)]
    out.append('<polygon points="' + " ".join(f"{a},{b}" for a, b in pts) + f'" fill="{stroke}"/>')


def column(x0, y0, rows):
    """Nodes stacked from y0; returns each node's top y."""
    return [y0 + i * (NODE_H + GAP) for i in range(rows)]


# Row 1: Annotate -> Search
r1 = 20
r1_nodes = column(r1 + 40, 0, 2)
r1_nodes = [r1 + 40 + y for y in r1_nodes]
r1_h = r1_nodes[-1] + NODE_H + PAD - r1
group("annotate", COL_X[0], r1, r1_h, "1. Annotate")
group("search", COL_X[1], r1, r1_h, "2. Search")
nx = [c + (GROUP_W - NODE_W) / 2 for c in COL_X]
node("annotate", nx[0], r1_nodes[0], "prepare-inputs", "optional, run by hand", optional=True)
node("annotate", nx[0], r1_nodes[1], "finetune-prep", "cleaned_structures/")
node("search", nx[1], r1_nodes[0], "generate-msas or import-msas", "organized_msas/", optional=True)
node("search", nx[1], r1_nodes[1], "fetch-templates", optional=True)
for col, step in ((0, "annotate"), (1, "search")):
    cx = nx[col] + NODE_W / 2
    arrow(cx, r1_nodes[0] + NODE_H + 4, cx, r1_nodes[1] - 5, color=STEPS[step][2], dashed=True)
mid1 = r1 + r1_h / 2
arrow(COL_X[0] + GROUP_W + 6, mid1, COL_X[1] - 6, mid1, width=2.5, head=12)

# Row 2: Hand off -> Check
r2 = r1 + r1_h + 90
r2_nodes = [r2 + 40 + i * (NODE_H + GAP) for i in range(5)]
r2_h = r2_nodes[-1] + NODE_H + PAD - r2
check_h = r2_nodes[1] + NODE_H + PAD - r2
group("handoff", COL_X[0], r2, r2_h, "3. Hand off")
group("check", COL_X[1], r2, check_h, "Check")
handoff = [
    ("materialize-dataset-cache", "dataset_cache.json", False),
    ("preprocess-templates", None, True),
    ("NPZ structure files", "of3_compressed_structures/", False),
    ("split-dataset", None, True),
    ("generate-train-config", "train.yml", True),
]
for y, (title, sub, opt) in zip(r2_nodes, handoff):
    node("handoff", nx[0], y, title, sub, optional=opt)
cx = nx[0] + NODE_W / 2
for a, b in zip(r2_nodes, r2_nodes[1:]):
    block_arrow(cx, a + NODE_H + 5, b - 5, "handoff")
node("check", nx[1], r2_nodes[0], "validate-dataset", "validation.json", optional=True)
node("check", nx[1], r2_nodes[1], "finetune-report", "report/index.html", optional=True)
cx = nx[1] + NODE_W / 2
arrow(cx, r2_nodes[0] + NODE_H + 4, cx, r2_nodes[1] - 5, color=STEPS["check"][2], dashed=True)
mid2 = r2 + check_h / 2
arrow(COL_X[0] + GROUP_W + 6, mid2, COL_X[1] - 6, mid2, width=2.5, head=12)

# Search -> Hand off: one arrow pointing at 210 degrees (down and to the left, 30 degrees below horizontal).
ang = math.radians(210)
dx, dy = math.cos(ang), -math.sin(ang)  # SVG y points down
end = (COL_X[0] + GROUP_W - 40, r2 - 6)
length = (end[1] - (r1 + r1_h + 6)) / dy
start = (end[0] - dx * length, end[1] - dy * length)
arrow(*start, *end, width=2.5, head=12)

H = r2 + r2_h + 20
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="title desc">',
    '<title id="title">Pipeline commands in run order</title>',
    '<desc id="desc">Four groups in a two by two grid. 1. Annotate: prepare-inputs (optional, run by hand), then finetune-prep. '
    "2. Search: generate-msas or import-msas, then fetch-templates. 3. Hand off: materialize-dataset-cache, preprocess-templates, "
    "NPZ structure files, split-dataset, generate-train-config. Check: validate-dataset, then finetune-report. "
    "Dashed boxes are stages that are optional or that a flag can skip.</desc>",
    *out,
    "</svg>",
]
open(OUT, "w").write("\n".join(svg) + "\n")
print(f"wrote {OUT}")
