"""Shared constants for the fine-tuning dataset pipeline."""

from __future__ import annotations

# No corpus-level clustering: every chain and interface gets this one dummy cluster id.
# DUMMY_CLUSTER_SIZE is for tests and legacy use only; build_structure_data sets
# cluster_size = datapoints (chains + interfaces) per structure. Safe because
# WeightedPDBDataset uses cluster_size as the sampling-weight divisor and never regroups
# by cluster_id. NOT safe if the cache is fed through OF3's
# create_pdb-weighted_training_dataset_cache.py, which regroups by cluster_id.
DUMMY_CLUSTER_ID = "unused_dummy_cluster"
DUMMY_CLUSTER_SIZE = 1

# finetune-prep writes the structures it dropped here ([{"id", "reason"}]); the report shows the reasons.
PREP_DROPPED_FILE = "prep_dropped.json"

# prepare-inputs writes which stems got the default release date here; validate V17 reads it.
PREPARE_RECORD_FILE = "prepare_inputs.json"

# preprocess-templates records --template-cutoff-date here, beside the template cache; V16 reads it.
TEMPLATE_CUTOFF_FILENAME = "template_cutoff_date.json"
# V16 starts its finding with this when a cutoff was used; the user report shows it as a note.
TEMPLATE_CUTOFF_NOTE = "templates use a fixed cutoff of"

# finetune-prep writes the cleaned structures (augmented mmCIFs) into this subdirectory of the output directory.
CLEANED_STRUCTURES_DIRNAME = "cleaned_structures"

# finetune-pipeline writes the cleaned structures as OF3 NPZ files (<id>/<id>.npz) here; train.yml, split-dataset and the report read them from here.
COMPRESSED_STRUCTURES_DIRNAME = "of3_compressed_structures"
