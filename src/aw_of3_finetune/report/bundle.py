"""Bundle the report's ES modules into one classic script, so the page works from file://."""

import re
from pathlib import Path

IMPORT_RE = re.compile(r'^import \{ ([\w$, ]+) \} from "\./([\w.-]+\.js)";$')
EXPORT_RE = re.compile(r"^export ((?:async )?function\*? |const |let |class )([\w$]+)")
MODULE_LINE_RE = re.compile(r"^\s*(import|export)\b")


def _parse_module(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Rewrite one module's import/export lines.

    Args:
        path: The module file.

    Returns:
        The rewritten lines, the imported module names, and the exported names.

    Raises:
        ValueError: On an import or export form the bundler does not handle.
    """
    lines, imports, exports = [], [], []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if match := IMPORT_RE.match(line):
            names, module = match.groups()
            imports.append(module)
            lines.append(f'const {{ {names} }} = __m["{module}"];')
        elif match := EXPORT_RE.match(line):
            exports.append(match.group(2))
            lines.append(line.removeprefix("export "))
        elif MODULE_LINE_RE.match(line):
            raise ValueError(f"{path.name}:{number}: unsupported import/export form: {line.strip()}")
        else:
            lines.append(line)
    return lines, imports, exports


def bundle_modules(js_dir: Path, entry: str = "main.js") -> str:
    """One classic script holding each module as a scope, dependencies first.

    Args:
        js_dir: Folder holding the modules.
        entry: The entry module, wrapped in an async scope for its top-level await.

    Returns:
        The bundle's source.

    Raises:
        ValueError: On an import or export form the bundler does not handle, or an import cycle.
    """
    parsed, order = {}, []

    def visit(name: str) -> None:
        if name in parsed:
            if name not in order:
                raise ValueError(f"{name}: import cycle")
            return
        parsed[name] = _parse_module(js_dir / name)
        for dependency in parsed[name][1]:
            visit(dependency)
        order.append(name)

    visit(entry)
    parts = ['"use strict";', "const __m = {};"]
    for name in order:
        lines, _, exports = parsed[name]
        body = "\n".join(lines)
        if name == entry:
            parts.append(f"// {name}\n(async () => {{\n{body}\n}})();")
        else:
            parts.append(f'// {name}\n__m["{name}"] = (() => {{\n{body}\nreturn {{ {", ".join(exports)} }};\n}})();')
    return "\n\n".join(parts) + "\n"
