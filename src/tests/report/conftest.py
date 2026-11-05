import pytest
from report.helpers import REPO
from validate_helpers import build_dataset


@pytest.fixture(scope="module")
def built_7riv(tmp_path_factory, ccd_path):
    """Real finetune-prep output of the 7riv fixture, with NPZ."""
    return build_dataset(tmp_path_factory.mktemp("bonds"), [REPO / "src/tests/fixtures/mmcifs/7riv.cif"], ccd_path)
