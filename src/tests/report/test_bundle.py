import shutil
import subprocess

import pytest

from aw_of3_finetune.report.build import ASSETS_DIR
from aw_of3_finetune.report.bundle import bundle_modules


def test_bundle_has_no_module_syntax():
    """The bundle of the report modules has no import or export statements left."""
    actual = bundle_modules(ASSETS_DIR / "js")

    actual_export_lines = [line for line in actual.splitlines() if line.startswith("export ")]
    assert "import {" not in actual
    assert actual_export_lines == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_bundle_parses_in_node(tmp_path):
    """node --check accepts the bundle as a classic script."""
    bundle_file = tmp_path / "report.js"
    bundle_file.write_text(bundle_modules(ASSETS_DIR / "js"))

    result = subprocess.run(["node", "--check", str(bundle_file)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_bundle_orders_dependencies_first(tmp_path):
    """A module's scope comes after the modules it imports, and its exports are returned."""
    (tmp_path / "a.js").write_text('export const A = 1;\nexport function twice(x) {\n  return 2 * x;\n}\n')
    (tmp_path / "main.js").write_text('import { A, twice } from "./a.js";\nconsole.log(twice(A));\n')

    actual = bundle_modules(tmp_path)

    expected_import = 'const { A, twice } = __m["a.js"];'
    expected_return = "return { A, twice };"
    assert expected_import in actual
    assert expected_return in actual
    assert actual.index('__m["a.js"] =') < actual.index(expected_import)


@pytest.mark.parametrize("line", ['import * as a from "./a.js";', "export default 1;", 'export { A } from "./a.js";'])
def test_bundle_rejects_other_module_forms(tmp_path, line):
    """An import or export form the bundler does not handle raises ValueError naming the file."""
    (tmp_path / "a.js").write_text("export const A = 1;\n")
    (tmp_path / "main.js").write_text(f"{line}\n")

    with pytest.raises(ValueError, match="main.js"):
        bundle_modules(tmp_path)


def test_bundle_rejects_import_cycles(tmp_path):
    """Two modules importing each other raise ValueError, since scopes need a dependency order."""
    (tmp_path / "a.js").write_text('import { B } from "./b.js";\nexport const A = 1;\n')
    (tmp_path / "b.js").write_text('import { A } from "./a.js";\nexport const B = 2;\n')
    (tmp_path / "main.js").write_text('import { A } from "./a.js";\n')

    with pytest.raises(ValueError, match="cycle"):
        bundle_modules(tmp_path)
