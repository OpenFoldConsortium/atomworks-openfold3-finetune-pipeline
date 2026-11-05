"""Constants shared by the report tests."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def inline_report_data(index_html: str) -> dict:
    """The REPORT_DATA object inlined in a built index.html."""
    start = index_html.index("window.REPORT_DATA = ") + len("window.REPORT_DATA = ")
    end = index_html.index(";</script>", start)
    return json.loads(index_html[start:end].replace("<\\/", "</").replace("<\\!--", "<!--"))


def entry_report_data(report_dir: Path, pdb_id: str) -> dict:
    """The {section: entry} data a built report writes to data/entries/<id>.js."""
    text = (report_dir / "data" / "entries" / f"{pdb_id}.js").read_text()
    prefix = f"window.__reportEntry({json.dumps(pdb_id)}, "
    return json.loads(text.removeprefix(prefix).removesuffix(");\n"))
