#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

CLASSES = 100
HIDDEN_SIZE = 5120

RANK_CANDIDATES = (
    128,
    192,
    256,
)

PROVISIONAL_RANK = max(
    RANK_CANDIDATES
)

MINIMUM_FREE_GPU_GIB = 4.0

PPLX_BASE_GPU_GIB = 48.73
A100_AVAILABLE_GIB = 79.25

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

checkpoint = Path(
    os.environ["PPLX_CHECKPOINT"]
).resolve()

model_id = os.environ["PPLX_MODEL_ID"]
revision = os.environ["PPLX_MODEL_REVISION"]

assert checkpoint.is_dir(), (
    f"Missing PPLX checkpoint: {checkpoint}"
)

index_path = (
    checkpoint
    / "model.safetensors.index.json"
)

assert index_path.is_file(), (
    f"Missing model index: {index_path}"
)

supervision_root = (
    root
    / "data"
    / "target_supervision"
    / "exact_laya_description_aware_v1"
)

required_files = {
    "complete_target":
        supervision_root / "complete_target.json",
    "train_complete":
        supervision_root / "train_complete.json",
    "train":
        supervision_root / "train.jsonl",
    "validation":
        supervision_root / "validation.jsonl",
    "test":
        supervision_root / "test.jsonl",
    "cluster_descriptions":
        supervision_root / "cluster_descriptions.json",
    "laya_reference_config":
        supervision_root
        / "laya_reference_training_config.json",
    "manifest":
        supervision_root / "MANIFEST.json",
}

for name, path in required_files.items():
    assert path.is_file(), (
        f"Missing required {name} file: {path}"
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as error:
                raise RuntimeError(
                    f"Invalid JSON at "
                    f"{path}:{line_number}"
                ) from error

            records.append(record)

    return records


laya_config = load_json(
    required_files[
        "laya_reference_config"
    ]
)

expected_laya_values = {
    "epochs": 4,
    "max_length": 512,
    "description_max_length": 256,
    "description_loss_weight": 0.5,
    "description_logit_scale": 10.0,
    "dropout": 0.1,
    "encoder_learning_rate": 2.5e-5,
    "head_learning_rate": 1.0e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "gradient_clip_norm": 1.0,
    "seed": 42,
}

for key, expected in expected_laya_values.items():
    actual = laya_config.get(key)

    assert actual == expected, (
        f"Reference LAYA configuration changed: "
        f"{key}={actual!r}, expected={expected!r}"
    )


raw_catalogue = load_json(
    required_files[
        "cluster_descriptions"
    ]
)

if isinstance(raw_catalogue, dict):
    if "clusters" in raw_catalogue:
        raw_catalogue = raw_catalogue[
            "clusters"
        ]
    elif (
        "complete_cluster_catalog"
        in raw_catalogue
    ):
        raw_catalogue = raw_catalogue[
            "complete_cluster_catalog"
        ]

if isinstance(raw_catalogue, list):
    catalogue = {
        record["cluster_id"]: record
        for record in raw_catalogue
    }
elif isinstance(raw_catalogue, dict):
    catalogue = raw_catalogue
else:
    raise TypeError(
        "Unsupported cluster-description structure"
    )

expected_clusters = {
    f"{field}_{index}"
    for field in FIELDS
    for index in range(CLASSES)
}

assert set(catalogue) == expected_clusters, (
    "Description catalogue does not contain "
    "exactly the expected 400 clusters"
)

catalogue_counts = Counter()
description_word_counts = []

for cluster_id, record in catalogue.items():
    field = cluster_id.split("_", 1)[0]

    catalogue_counts[field] += 1

    description = (
        record.get("description")
        or record.get("refined_description")
    )

    assert isinstance(description, str)
    assert description.strip(), (
        f"Empty description: {cluster_id}"
    )

    description_word_counts.append(
        len(description.split())
    )

for field in FIELDS:
    assert catalogue_counts[field] == CLASSES


split_records = {
    split: load_jsonl(
        required_files[split]
    )
    for split in (
        "train",
        "validation",
        "test",
    )
}

expected_split_sizes = {
    "train": 3978,
    "validation": 850,
    "test": 858,
}

actual_split_sizes = {
    split: len(records)
    for split, records
    in split_records.items()
}

assert (
    actual_split_sizes
    == expected_split_sizes
), (
    f"Split sizes changed: "
    f"{actual_split_sizes}"
)


def validate_target(
    record,
    field,
):
    targets = record.get("targets")

    assert isinstance(targets, dict), (
        "Record has no targets dictionary"
    )

    target = targets.get(field)

    assert isinstance(target, dict), (
        f"Record has no {field} target"
    )

    indices = target.get("indices")
    probabilities = target.get(
        "probabilities"
    )
    cluster_ids = target.get(
        "cluster_ids"
    )
    descriptions = target.get(
        "descriptions"
    )

    assert isinstance(indices, list)
    assert isinstance(
        probabilities,
        list,
    )
    assert isinstance(cluster_ids, list)

    assert len(indices) >= 1
    assert len(indices) == len(
        probabilities
    )
    assert len(indices) == len(
        cluster_ids
    )

    if descriptions is not None:
        assert isinstance(
            descriptions,
            list,
        )
        assert len(descriptions) == len(
            indices
        )

    assert len(indices) == len(
        set(indices)
    ), (
        f"Duplicate class index in {field}"
    )

    for index in indices:
        assert isinstance(index, int)
        assert 0 <= index < CLASSES

    expected_ids = [
        f"{field}_{index}"
        for index in indices
    ]

    assert cluster_ids == expected_ids, (
        f"Cluster IDs do not match indices "
        f"for {field}"
    )

    probability_sum = sum(
        float(value)
        for value in probabilities
    )

    assert abs(
        probability_sum - 1.0
    ) < 1e-5, (
        f"{field} probabilities sum to "
        f"{probability_sum}"
    )

    assert all(
        float(value) >= 0.0
        for value in probabilities
    )

    return len(indices)


target_cardinality = {
    split: {
        field: Counter()
        for field in FIELDS
    }
    for split in split_records
}

for split, records in split_records.items():
    for record in records:
        assert isinstance(
            record.get("text"),
            str,
        )

        assert record["text"].strip()

        for field in FIELDS:
            cardinality = validate_target(
                record,
                field,
            )

            target_cardinality[
                split
            ][field][cardinality] += 1


# These are all verified text-backbone Linear
# module groups found in the exact loaded
# pplx-decider-v1.2 checkpoint.
#
# Visual modules are intentionally excluded
# because the target and source datasets are
# text-only.
lora_module_groups = {
    "down_proj": {
        "shape": [5120, 17408],
        "count": 64,
    },
    "gate_proj": {
        "shape": [17408, 5120],
        "count": 64,
    },
    "up_proj": {
        "shape": [17408, 5120],
        "count": 64,
    },
    "in_proj_qkv": {
        "shape": [10240, 5120],
        "count": 48,
    },
    "in_proj_z": {
        "shape": [6144, 5120],
        "count": 48,
    },
    "out_proj": {
        "shape": [5120, 6144],
        "count": 48,
    },
    "q_proj": {
        "shape": [12288, 5120],
        "count": 16,
    },
    "k_proj": {
        "shape": [1024, 5120],
        "count": 16,
    },
    "v_proj": {
        "shape": [1024, 5120],
        "count": 16,
    },
    "o_proj": {
        "shape": [5120, 6144],
        "count": 16,
    },
}

four_head_parameters = (
    len(FIELDS)
    * (
        HIDDEN_SIZE * CLASSES
        + CLASSES
    )
)

shared_layer_norm_parameters = (
    2 * HIDDEN_SIZE
)

prototype_values = (
    len(FIELDS)
    * CLASSES
    * HIDDEN_SIZE
)

head_initialization_norm = math.sqrt(
    2.0
    * HIDDEN_SIZE
    / (
        HIDDEN_SIZE
        + CLASSES
    )
)


def calculate_lora_parameters(rank):
    total = 0

    for specification in (
        lora_module_groups.values()
    ):
        output_features, input_features = (
            specification["shape"]
        )

        count = specification["count"]

        total += (
            count
            * rank
            * (
                input_features
                + output_features
            )
        )

    return total


capacity_plan = {}

for rank in RANK_CANDIDATES:
    lora_parameters = (
        calculate_lora_parameters(rank)
    )

    total_trainable = (
        lora_parameters
        + four_head_parameters
        + shared_layer_norm_parameters
    )

    # Conservative estimate for trainable state:
    # FP32 parameter or master copy: 4 bytes
    # FP32 gradient:                 4 bytes
    # Adam first moment:            4 bytes
    # Adam second moment:           4 bytes
    #
    # Activations and CUDA workspace are not
    # included; therefore an actual smoke test
    # is still mandatory.
    conservative_training_state_gib = (
        total_trainable
        * 16
        / (1024 ** 3)
    )

    estimated_before_activations_gib = (
        PPLX_BASE_GPU_GIB
        + conservative_training_state_gib
    )

    estimated_remaining_before_activations = (
        A100_AVAILABLE_GIB
        - estimated_before_activations_gib
    )

    capacity_plan[str(rank)] = {
        "rank": rank,
        "lora_parameters":
            lora_parameters,
        "four_head_parameters":
            four_head_parameters,
        "shared_layer_norm_parameters":
            shared_layer_norm_parameters,
        "total_trainable_parameters":
            total_trainable,
        "conservative_training_state_gib":
            conservative_training_state_gib,
        "estimated_gpu_gib_before_activations":
            estimated_before_activations_gib,
        "estimated_remaining_gib_before_activations":
            estimated_remaining_before_activations,
        "requires_real_smoke_test": True,
    }


training_config = {
    "experiment": (
        "pplx_decider_v1_2_"
        "description_four_head_run_01"
    ),
    "model": {
        "model_id": model_id,
        "revision": revision,
        "checkpoint": str(checkpoint),
        "hidden_size": HIDDEN_SIZE,
        "backbone_parameters":
            26_085_330_160,
        "native_readout_parameters":
            1_305_600,
        "start_from_exact_base_checkpoint":
            True,
        "resume_checkpoint": None,
        "native_readout_policy":
            "frozen_and_preserved",
        "visual_backbone_policy":
            "frozen",
    },
    "data": {
        "supervision_root":
            str(supervision_root),
        "complete_target_file":
            str(
                required_files[
                    "complete_target"
                ]
            ),
        "train_complete_file":
            str(
                required_files[
                    "train_complete"
                ]
            ),
        "train_file":
            str(required_files["train"]),
        "validation_file":
            str(
                required_files[
                    "validation"
                ]
            ),
        "test_file":
            str(required_files["test"]),
        "description_catalogue":
            str(
                required_files[
                    "cluster_descriptions"
                ]
            ),
        "manifest":
            str(
                required_files["manifest"]
            ),
        "split_sizes":
            expected_split_sizes,
        "dataset_regenerated": False,
        "assignments_changed": False,
        "descriptions_changed": False,
        "split_membership_changed": False,
        "truncate_consolidated_targets":
            False,
    },
    "architecture": {
        "fields": list(FIELDS),
        "classes_per_field": CLASSES,
        "new_decision_heads": {
            field: CLASSES
            for field in FIELDS
        },
        "representation": (
            "official_pplx_question_marker_"
            "hidden_state"
        ),
        "shared_layer_norm": True,
        "dropout": 0.1,
        "head_initialization": (
            "normalized_description_"
            "prototype"
        ),
        "head_initialization_norm":
            head_initialization_norm,
        "description_prototypes": {
            "shape_per_field": [
                CLASSES,
                HIDDEN_SIZE,
            ],
            "total_values":
                prototype_values,
            "trainable": False,
            "normalization": "l2",
            "maximum_length": 256,
            "logit_scale": 10.0,
            "text_template": (
                "Field: {field}. "
                "Title: {title}. "
                "Description: {description} "
                "Representative concepts: "
                "{keywords}."
            ),
        },
    },
    "adaptation": {
        "method": "lora",
        "capacity_objective": (
            "maximum_trainable_capacity_"
            "that_is_stable_on_one_a100_80gb"
        ),
        "rank_selection_status":
            "pending_gpu_smoke_test",
        "rank_candidates":
            list(RANK_CANDIDATES),
        "testing_order": [
            256,
            192,
            128,
        ],
        "provisional_rank":
            PROVISIONAL_RANK,
        "selected_rank": None,
        "minimum_free_gpu_memory_after_update_gib":
            MINIMUM_FREE_GPU_GIB,
        "alpha": 16,
        "dropout": 0.1,
        "target_module_suffixes":
            list(lora_module_groups),
        "freeze_original_backbone_weights":
            True,
        "freeze_visual_backbone":
            True,
        "freeze_native_readout":
            True,
        "train_new_four_heads":
            True,
        "train_shared_layer_norm":
            True,
        "capacity_plan":
            capacity_plan,
    },
    "loss": {
        "target_type": (
            "complete_normalized_"
            "soft_distribution"
        ),
        "classification": (
            "mean_of_four_"
            "soft_cross_entropies"
        ),
        "description": (
            "mean_of_four_description_"
            "prototype_soft_cross_entropies"
        ),
        "description_loss_weight": 0.5,
        "description_logit_scale": 10.0,
        "total": (
            "classification_loss + "
            "0.5 * description_loss"
        ),
    },
    "training": {
        "epochs": 4,
        "seed": 42,
        "maximum_length": 512,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 64,
        "effective_batch_size": 64,
        "evaluation_batch_size": 1,
        "lora_learning_rate": 2.5e-5,
        "head_learning_rate": 1.0e-4,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "gradient_clip_norm": 1.0,
        "precision": "bfloat16",
        "gradient_checkpointing": True,
        "logging_updates": 5,
        "save_checkpoint_each_epoch":
            True,
    },
    "capacity_smoke_test": {
        "required_before_full_training":
            True,
        "train_examples": 8,
        "validation_examples": 4,
        "optimizer_updates_per_rank": 1,
        "test_forward": True,
        "test_backward": True,
        "test_optimizer_step": True,
        "clear_cuda_cache_between_ranks":
            True,
        "test_order": [
            256,
            192,
            128,
        ],
        "selection_rule": (
            "largest rank completing a real "
            "optimizer update while retaining "
            "the required GPU reserve"
        ),
    },
}

config_path = (
    root
    / "configs"
    / "09_pplx_description_four_head_training.json"
)

report_path = (
    root
    / "metadata"
    / "training_design"
    / "09_pplx_description_training_contract.json"
)

config_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with config_path.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        training_config,
        handle,
        indent=2,
        sort_keys=True,
    )

    handle.write("\n")

report = {
    "status": "PASSED",
    "model_id": model_id,
    "revision": revision,
    "checkpoint": str(checkpoint),
    "input_hashes": {
        name: sha256(path)
        for name, path
        in required_files.items()
    },
    "split_sizes":
        actual_split_sizes,
    "catalogue_counts":
        dict(catalogue_counts),
    "description_word_range": {
        "minimum":
            min(description_word_counts),
        "maximum":
            max(description_word_counts),
    },
    "target_cardinality": {
        split: {
            field: dict(
                sorted(counter.items())
            )
            for field, counter
            in fields.items()
        }
        for split, fields
        in target_cardinality.items()
    },
    "rank_candidates":
        list(RANK_CANDIDATES),
    "provisional_rank":
        PROVISIONAL_RANK,
    "selected_rank": None,
    "capacity_plan":
        capacity_plan,
    "configuration":
        str(config_path),
}

with report_path.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        report,
        handle,
        indent=2,
        sort_keys=True,
    )

    handle.write("\n")


print("=" * 78)
print(
    "PPLX MAXIMUM-CAPACITY "
    "DESCRIPTION TRAINING CONTRACT"
)
print("=" * 78)
print("Status: PASSED")
print("Model:", model_id)
print("Revision:", revision)
print("Checkpoint:", checkpoint)
print()
print("DATA")
print("  Splits:", actual_split_sizes)
print(
    "  Catalogue:",
    dict(catalogue_counts),
)
print("  Assignments changed: NO")
print("  Descriptions changed: NO")
print("  Targets truncated: NO")
print()
print("ARCHITECTURE")
print("  New decision heads: 4")
print("  Classes per head: 100")
print(
    "  Native PPLX readout: "
    "FROZEN AND PRESERVED"
)
print("  Visual backbone: FROZEN")
print(
    "  Text backbone adaptation: "
    "LoRA on all verified linear groups"
)
print()
print("LOSS")
print(
    "  Classification: mean of four "
    "soft cross-entropies"
)
print(
    "  Description: mean of four "
    "prototype soft cross-entropies"
)
print(
    "  Total: classification + "
    "0.5 × description"
)
print()
print("CAPACITY CANDIDATES")

for rank in sorted(
    RANK_CANDIDATES,
    reverse=True,
):
    entry = capacity_plan[str(rank)]

    print()
    print(f"  Rank {rank}")
    print(
        "    LoRA parameters:",
        f"{entry['lora_parameters']:,}",
    )
    print(
        "    Total trainable:",
        f"{entry['total_trainable_parameters']:,}",
    )
    print(
        "    Conservative training state:",
        f"{entry['conservative_training_state_gib']:.2f}",
        "GiB",
    )
    print(
        "    Estimated memory remaining "
        "before activations:",
        f"{entry['estimated_remaining_gib_before_activations']:.2f}",
        "GiB",
    )

print()
print(
    "Provisional maximum rank:",
    PROVISIONAL_RANK,
)
print(
    "Final rank: PENDING REAL GPU "
    "FORWARD/BACKWARD/OPTIMIZER TEST"
)
print()
print("Config:", config_path)
print("Report:", report_path)
print()
print(
    "MAXIMUM-CAPACITY TRAINING "
    "CONTRACT PASSED"
)
