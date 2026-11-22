#!/usr/bin/env python3

from pathlib import Path
import gc
import json
import os

import laya
import pyarrow.parquet as pq
import torch
from safetensors import safe_open


restart_root = Path(
    os.environ.get(
        "BASE_RESTART_ROOT",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "laya_tads_anchor_classifier/laya_base_restart",
    )
).resolve()

exp_root = Path(
    os.environ.get(
        "EXP_ROOT",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "laya_tads_anchor_classifier",
    )
).resolve()

paper_work = Path(
    os.environ.get(
        "PAPER_WORK",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "tads_paper_faithful",
    )
).resolve()

base_directory = (
    exp_root
    / "models"
    / "laya_base_english"
)

fine_checkpoint = (
    restart_root
    / "models"
    / "laya_base_description_four_head_run_01"
    / "epoch_4"
    / "model.safetensors"
)

prediction_path = (
    restart_root
    / "data"
    / "source_predictions"
    / "source_predictions_all_300932.parquet"
)

source_path = (
    paper_work
    / "data"
    / "train-00000-of-00001.parquet"
)

catalog_path = (
    restart_root
    / "data"
    / "complete_target_with_cluster_descriptions.json"
)

output_path = (
    restart_root
    / "metadata"
    / "11_base_vs_hybrid_scoring_smoke.json"
)

for path in (
    base_directory,
    fine_checkpoint,
    prediction_path,
    source_path,
    catalog_path,
):
    assert path.exists(), f"Missing required input: {path}"


complete_target = json.loads(
    catalog_path.read_text(encoding="utf-8")
)

catalog = {
    item["cluster_id"]: item
    for item in complete_target[
        "complete_cluster_catalog"
    ]
}

prediction = (
    pq.ParquetFile(prediction_path)
    .read_row_group(0)
    .slice(0, 1)
    .to_pylist()[0]
)

source = (
    pq.ParquetFile(source_path)
    .read_row_group(
        0,
        columns=["dataset", "id", "messages"],
    )
    .slice(0, 1)
    .to_pylist()[0]
)

assert prediction["source_position"] == 0
assert prediction["dataset"] == source["dataset"]
assert prediction["id"] == source["id"]


def conversation_text(messages):
    parts = []

    for message in messages:
        role = str(
            message.get("role", "unknown")
        ).strip()

        content = str(
            message.get("content", "")
        ).strip()

        parts.append(f"{role}: {content}")

    return "\n".join(parts)


def anchor_text(field):
    indices = prediction[f"{field}_indices"]

    lines = []

    for index in indices:
        cluster_id = f"{field}_{int(index)}"

        assert cluster_id in catalog, (
            f"Missing catalog entry: {cluster_id}"
        )

        item = catalog[cluster_id]

        title = item.get("title", cluster_id)

        description = item[
            "refined_description"
        ].strip()

        lines.append(
            f"- {cluster_id}: {title}. "
            f"{description}"
        )

    return "\n".join(lines)


state = f"""Evaluate the quality of this training example.

TRAINING EXAMPLE
{conversation_text(source["messages"])}

ASSIGNED TASK
{anchor_text("Task")}

ASSIGNED TOPICS
{anchor_text("Topic")}

ASSIGNED STYLES
{anchor_text("Style")}

ASSIGNED AUDIENCES
{anchor_text("Audience")}

Use the assigned semantic anchors as reference points when
judging the example, especially for the Overall Score.
"""


QUESTIONS = {
    "Completeness": {
        "type": "score",
        "instructions": (
            "How complete is the response for the user's "
            "request?"
        ),
        "criteria": [
            "Completely missing or unusable",
            "Almost entirely incomplete",
            "Very incomplete",
            "Several essential parts are missing",
            "Partially complete",
            "Adequate but has noticeable omissions",
            "Mostly complete",
            "Complete and useful",
            "Very complete and detailed",
            "Exceptionally complete",
        ],
    },
    "Information Richness": {
        "type": "score",
        "instructions": (
            "How much useful and informative content does "
            "the response contain?"
        ),
        "criteria": [
            "No useful information",
            "Almost no useful information",
            "Very little useful information",
            "Limited information",
            "Some useful information",
            "Adequate information",
            "Good amount of useful information",
            "Rich and informative",
            "Very rich and detailed",
            "Exceptionally informative",
        ],
    },
    "Rarity": {
        "type": "score",
        "instructions": (
            "How uncommon, distinctive, or rare is this "
            "type of training content?"
        ),
        "criteria": [
            "Extremely common",
            "Very common",
            "Common",
            "Mostly common",
            "Slightly uncommon",
            "Moderately uncommon",
            "Uncommon",
            "Rare",
            "Very rare",
            "Exceptionally rare",
        ],
    },
    "Complexity": {
        "type": "score",
        "instructions": (
            "How complex is the task and the reasoning or "
            "knowledge needed for its response?"
        ),
        "criteria": [
            "Trivial",
            "Extremely simple",
            "Very simple",
            "Simple",
            "Moderately simple",
            "Moderate",
            "Moderately complex",
            "Complex",
            "Very complex",
            "Exceptionally complex",
        ],
    },
    "Overall Score": {
        "type": "score",
        "instructions": (
            "Considering completeness, usefulness, and "
            "compatibility with the assigned Task, Topic, "
            "Style, and Audience anchors, what is the "
            "overall quality and relevance?"
        ),
        "criteria": [
            "Completely unsuitable",
            "Almost entirely unsuitable",
            "Very poor",
            "Poor",
            "Below average",
            "Adequate",
            "Good",
            "Very good",
            "Excellent",
            "Exceptional",
        ],
    },
}


def make_json_safe(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    return str(value)


def run_model(use_finetuned_encoder):
    agent = laya.load(
        str(base_directory),
        device="cuda",
    )

    replacement_report = None

    if use_finetuned_encoder:
        encoder_state = {}

        with safe_open(
            str(fine_checkpoint),
            framework="pt",
            device="cpu",
        ) as handle:
            for key in handle.keys():
                if key.startswith("encoder."):
                    encoder_state[
                        key[len("encoder."):]
                    ] = handle.get_tensor(key)

        expected_keys = set(
            agent.model.encoder.state_dict()
        )

        supplied_keys = set(encoder_state)

        assert supplied_keys == expected_keys, {
            "missing": sorted(
                expected_keys - supplied_keys
            ),
            "unexpected": sorted(
                supplied_keys - expected_keys
            ),
        }

        incompatible = (
            agent.model.encoder.load_state_dict(
                encoder_state,
                strict=True,
            )
        )

        replacement_report = {
            "encoder_tensors_loaded": len(
                encoder_state
            ),
            "missing_keys": list(
                incompatible.missing_keys
            ),
            "unexpected_keys": list(
                incompatible.unexpected_keys
            ),
        }

    result = agent.predict(
        state,
        QUESTIONS,
        max_len=2048,
    )

    safe_result = make_json_safe(result)

    del agent

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "encoder_replacement": (
            replacement_report
        ),
        "result": safe_result,
    }


print("Running untouched base Laya...", flush=True)

base_result = run_model(
    use_finetuned_encoder=False
)

print(
    "Running fine-tuned encoder with base scorer...",
    flush=True,
)

hybrid_result = run_model(
    use_finetuned_encoder=True
)

report = {
    "status": "PASSED",
    "sample": {
        "source_position": (
            prediction["source_position"]
        ),
        "dataset": source["dataset"],
        "id": source["id"],
        "state_characters": len(state),
        "assigned_indices": {
            field: prediction[f"{field}_indices"]
            for field in (
                "Task",
                "Topic",
                "Style",
                "Audience",
            )
        },
    },
    "base_laya": base_result,
    "finetuned_encoder_base_scorer": (
        hybrid_result
    ),
    "warning": (
        "The hybrid scorer is not yet calibrated. "
        "This smoke test only verifies execution and "
        "exposes the raw score format."
    ),
}

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

output_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("\n" + "=" * 72)
print("BASE LAYA RESULT")
print("=" * 72)
print(
    json.dumps(
        base_result["result"],
        indent=2,
        sort_keys=True,
    )
)

print("\n" + "=" * 72)
print("FINE-TUNED ENCODER + BASE SCORER RESULT")
print("=" * 72)
print(
    json.dumps(
        hybrid_result["result"],
        indent=2,
        sort_keys=True,
    )
)

print("\nReport:", output_path)
print("\nBASE VS HYBRID SCORING SMOKE TEST PASSED")
