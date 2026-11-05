"""Finding and report types for the validate stage."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    """Severity of a finding: error, warning or info."""

    ERROR = "E"
    WARNING = "W"
    INFO = "I"


@dataclass(frozen=True)
class Finding:
    """One validation result; ``check`` is the check ID such as ``"V3"``."""

    check: str
    severity: Severity
    entry: str | None
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict with the severity as its letter."""
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ValidationReport:
    """All findings from a validation run plus the entries that were checked."""

    findings: list[Finding]
    entries_in: list[str]

    @property
    def errors(self) -> list[Finding]:
        """Findings with severity E."""
        return [f for f in self.findings if f.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        """Findings with severity W."""
        return [f for f in self.findings if f.severity == Severity.WARNING]

    @property
    def failing_entries(self) -> set[str]:
        """Entries with at least one E finding."""
        return {f.entry for f in self.errors if f.entry is not None}

    @property
    def entries_out(self) -> list[str]:
        """Input entries minus failing entries, in input order."""
        failing = self.failing_entries
        return [e for e in self.entries_in if e not in failing]

    def to_json(self, path: Path, extra: dict | None = None) -> None:
        """Write the report as JSON.

        Args:
            path: Destination file.
            extra: Additional top-level keys, e.g. what ``--allow-errors`` dropped.
        """
        data = {
            "entries_in": self.entries_in,
            "entries_out": self.entries_out,
            "findings": [f.to_dict() for f in self.findings],
            **(extra or {}),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def summary_text(self) -> str:
        """Return entries in/out, findings tallied by check and severity, and V1 drop reasons."""
        lines = [f"entries: {len(self.entries_in)} in → {len(self.entries_out)} out"]
        tally = Counter((f.check, f.severity.value) for f in self.findings)
        for (check, sev), n in sorted(tally.items()):
            lines.append(f"{check}  {sev} {n}")
        drops = Counter(f.message for f in self.findings if f.check == "V1")
        if drops:
            lines.append("drop reasons (V1):")
            lines.extend(f"  {n}  {reason}" for reason, n in drops.most_common())
        return "\n".join(lines)
