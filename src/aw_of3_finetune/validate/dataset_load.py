"""V21: every dataset row loads through OF3's WeightedPDBDataset and passes its feature check."""

from __future__ import annotations

import warnings

from openfold3.core.data.framework.single_datasets.abstract_single import DATASET_REGISTRY
from openfold3.core.data.framework.single_datasets.dataset_utils import check_invalid_feature_dict
from openfold3.projects.of3_all_atom.config.dataset_configs import DATASET_CONFIG_REGISTRY

from aw_of3_finetune.training_config import MSA_SETTINGS, build_dataset_paths
from aw_of3_finetune.validate.context import ValidationContext
from aw_of3_finetune.validate.findings import Finding, Severity

_NO_CROP = {"token_crop": {"enabled": False}, "chain_crop": {"enabled": False}}


def check_v21_dataset_loads(ctx: ValidationContext) -> list[Finding]:
    """V21: build one datapoint per entry through WeightedPDBDataset without cropping.

    Each sample also goes through OF3's check_invalid_feature_dict, which training
    skips with debug_mode false.

    Args:
        ctx: Validation context.

    Returns:
        One E per entry whose first row fails (crops are off, so one row covers the entry); one INFO when there is no msa_dir.
    """
    if ctx.msa_dir is None:
        return [Finding("V21", Severity.INFO, None, "skipped: no --organized-msa-dir (OF3 requires an alignments directory)")]
    paths = build_dataset_paths(ctx.output_dir, ctx.msa_dir)["weighted-pdb"]
    paths["dataset_cache_file"] = str(ctx.cache_path)
    paths["target_structures_directory"] = str(ctx.npz_dir)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No template paths provided")
        config = DATASET_CONFIG_REGISTRY.get("WeightedPDBDataset")(
            name="validate", dataset_paths=paths, msa=MSA_SETTINGS, crop=_NO_CROP
        )
    dataset = DATASET_REGISTRY["WeightedPDBDataset"](config)
    failed: dict[str, Finding] = {}
    seen: set[str] = set()
    for i in range(len(dataset)):
        entry = str(dataset.datapoint_cache.iloc[i]["pdb_id"])
        if entry in seen:
            continue
        seen.add(entry)
        try:
            check_invalid_feature_dict(dataset[i])
        except Exception as exc:
            failed[entry] = Finding(
                "V21", Severity.ERROR, entry, f"{type(exc).__name__}: {exc}", {"row": i}
            )
    return list(failed.values())
