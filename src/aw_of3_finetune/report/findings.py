"""One comparison result: which check, which entry, and whether the difference is explained."""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Status(str, Enum):
    """match: no difference; expected: a known, explained difference; unexplained: neither."""

    MATCH = "match"
    EXPECTED = "expected"
    UNEXPLAINED = "unexplained"


LABEL_ORDER = ("same", "equivalent", "expected", "not_checked", "input", "open", "exception", "unexplained")  # best to worst


def worst_label(labels: list[str]) -> str:
    """The worst of some labels in LABEL_ORDER.

    Args:
        labels: Labels from LABEL_ORDER.

    Returns:
        The worst one, or "same" if there are none.
    """
    return LABEL_ORDER[max((LABEL_ORDER.index(label) for label in labels), default=0)]


@dataclass
class Finding:
    """The outcome of one check on one entry (entry None for dataset-level checks)."""

    check: str
    entry: str | None
    status: Status
    detail: str
    data: dict = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict:
        """A JSON-serializable dict; status becomes its string value.

        Returns:
            The finding as plain dict.
        """
        result = asdict(self)
        result["status"] = self.status.value
        return result


def load_findings(run_dir: Path, filename: str = "findings.jsonl") -> list[Finding]:
    """Read a findings jsonl file of a run dir.

    Args:
        run_dir: A run dir containing the file.
        filename: The file name (validate writes validate_findings.jsonl).

    Returns:
        The stored findings.
    """
    findings = []
    for line in (run_dir / filename).read_text().splitlines():
        data = json.loads(line)
        data["status"] = Status(data["status"])
        findings.append(Finding(**data))
    return findings


def gate_line(findings: list[Finding]) -> str:
    """One line stating the gate verdict: any unexplained fails; per-entry exceptions pass but are counted.

    Args:
        findings: Findings after apply_rules.

    Returns:
        e.g. "gate: pass (2 exceptions)" or "gate: FAIL, 1 unexplained (1 exception)".
    """
    unexplained = sum(f.status == Status.UNEXPLAINED for f in findings)
    exceptions = sum(f.status == Status.EXPECTED and f.data.get("label") == "exception" for f in findings)
    noun = "exception" if exceptions == 1 else "exceptions"
    verdict = f"FAIL, {unexplained} unexplained" if unexplained else "pass"
    return f"gate: {verdict} ({exceptions} {noun})"
