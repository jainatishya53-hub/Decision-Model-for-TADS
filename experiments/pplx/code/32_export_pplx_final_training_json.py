#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import time

import pyarrow.parquet as pq


EXPECTED_ROWS = 10000


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                16 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def atomic_text(path, value):
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        value,
        encoding="utf-8",
    )

    temporary_path.replace(path)


def atomic_json(
    path,
    value,
    *,
    indent=None,
):
    atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            sort_keys=(
                indent is not None
            ),
        ),
    )


start_time = time.time()

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

input_path = (
    root
    / "data"
    / "distribution_matching"
    / "original_tads_pplx"
    / "final_original_tads_pplx_10000.parquet"
)

selection_report_path = (
    root
    / "metadata"
    / "distribution_matching"
    / "30_original_tads_pplx_distribution_matching.json"
)

audit_report_path = (
    root
    / "metadata"
    / "distribution_matching"
    / "31_final_original_tads_pplx_10000_audit.json"
)

output_path = (
    root
    / "data"
    / "model_finetune"
    / "tads_pplx_10k_gte_5_dataset.json"
)

mapping_path = (
    root
    / "data"
    / "model_finetune"
    / "tads_pplx_10k_id_mapping.json"
)

report_path = (
    root
    / "metadata"
    / "model_finetune"
    / "32_pplx_training_json_export.json"
)

for required_path in (
    input_path,
    selection_report_path,
    audit_report_path,
):
    assert required_path.is_file(), (
        f"Missing required file: {required_path}"
    )

with selection_report_path.open(
    encoding="utf-8"
) as handle:
    selection_report = json.load(handle)

with audit_report_path.open(
    encoding="utf-8"
) as handle:
    audit_report = json.load(handle)

assert selection_report["status"] == "PASSED"
assert audit_report["status"] == "PASSED"

assert (
    selection_report["fusion"]["final_count"]
    == EXPECTED_ROWS
)

assert audit_report["rows"] == EXPECTED_ROWS

assert (
    selection_report["laya_model_used"]
    is False
)

assert (
    selection_report["qwen_model_used"]
    is False
)

assert (
    selection_report[
        "teacher_scores_used"
    ]
    is False
)

assert (
    selection_report[
        "original_external_score_columns_used"
    ]
    is False
)

table = pq.read_table(
    input_path,
    columns=[
        "dataset",
        "id",
        "messages",
        "source_position",
        "tads_adapter_id",
    ],
)

assert table.num_rows == EXPECTED_ROWS

rows = table.to_pylist()

training_records = []
id_mapping = []

dataset_counts = Counter()
role_counts = Counter()
empty_message_role_counts = Counter()

records_without_nonempty_assistant = 0
records_with_system_messages = 0
multi_turn_records = 0

for position, row in enumerate(rows):
    dataset = str(row["dataset"])
    original_id = str(row["id"])

    adapter_id = str(
        row["tads_adapter_id"]
    )

    source_position = int(
        row["source_position"]
    )

    messages = row["messages"]

    expected_adapter_id = (
        f"{dataset}::{original_id}::"
        f"{source_position:09d}"
    )

    assert adapter_id == expected_adapter_id
    assert isinstance(messages, list)
    assert len(messages) >= 2

    normalized_messages = []

    for message in messages:
        assert isinstance(message, dict)

        role = str(message["role"])
        content = str(message["content"])

        assert role in {
            "system",
            "user",
            "assistant",
        }

        if not content.strip():
            empty_message_role_counts[
                role
            ] += 1

        normalized_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        role_counts[role] += 1

    assert any(
        message["role"] == "assistant"
        for message in normalized_messages
    ), (
        f"No assistant response in "
        f"{adapter_id}"
    )

    if not any(
        (
            message["role"]
            == "assistant"
        )
        and message[
            "content"
        ].strip()
        for message in normalized_messages
    ):
        records_without_nonempty_assistant += 1

    if any(
        message["role"] == "system"
        for message in normalized_messages
    ):
        records_with_system_messages += 1

    assistant_turns = sum(
        message["role"] == "assistant"
        for message in normalized_messages
    )

    if assistant_turns > 1:
        multi_turn_records += 1

    # Exact structure consumed by the original
    # TADS finetune.py implementation.
    training_records.append(
        {
            "dataset": dataset,
            "id": adapter_id,
            "messages": normalized_messages,
        }
    )

    id_mapping.append(
        {
            "id": adapter_id,
            "original_id": original_id,
            "dataset": dataset,
            "source_position": (
                source_position
            ),
            "training_position": position,
        }
    )

    dataset_counts[dataset] += 1

training_ids = [
    record["id"]
    for record in training_records
]

assert len(training_records) == EXPECTED_ROWS

assert len(training_ids) == len(
    set(training_ids)
), "Training IDs are not unique"

atomic_json(
    output_path,
    training_records,
)

atomic_json(
    mapping_path,
    id_mapping,
    indent=2,
)

reloaded = json.loads(
    output_path.read_text(
        encoding="utf-8"
    )
)

assert len(reloaded) == EXPECTED_ROWS

assert [
    record["id"]
    for record in reloaded
] == training_ids

assert all(
    set(record.keys())
    == {
        "dataset",
        "id",
        "messages",
    }
    for record in reloaded
)

assert all(
    isinstance(record["messages"], list)
    and len(record["messages"]) >= 2
    for record in reloaded
)

report = {
    "status": "PASSED",
    "experiment": (
        "Original TADS selection using "
        "PPLX cluster assignments and "
        "base-PPLX quality scores"
    ),
    "input": str(input_path),
    "output": str(output_path),
    "id_mapping": str(mapping_path),
    "records": len(training_records),
    "unique_ids": len(
        set(training_ids)
    ),
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "message_role_counts": dict(
        sorted(role_counts.items())
    ),
    "empty_message_role_counts": dict(
        sorted(
            empty_message_role_counts.items()
        )
    ),
    "records_without_nonempty_assistant": (
        records_without_nonempty_assistant
    ),
    "records_with_system_messages": (
        records_with_system_messages
    ),
    "multi_turn_records": (
        multi_turn_records
    ),
    "empty_message_policy": (
        "Preserved exactly as present in "
        "the original source dataset"
    ),
    "format": [
        "dataset",
        "id",
        "messages",
    ],
    "selection_algorithm": (
        "Original TADS distribution matching "
        "and equal-weight fusion"
    ),
    "cluster_assignment_model": (
        "Fine-tuned perplexity-ai/"
        "pplx-decider-v1.2 epoch 8"
    ),
    "quality_scoring_model": (
        "Base/native perplexity-ai/"
        "pplx-decider-v1.2"
    ),
    "quality_filter": (
        "All five PPLX scores >= 5"
    ),
    "laya_model_used": False,
    "qwen_model_used": False,
    "teacher_scores_used": False,
    "original_external_scores_used": False,
    "sha256": {
        "input_parquet": sha256_file(
            input_path
        ),
        "training_json": sha256_file(
            output_path
        ),
        "id_mapping": sha256_file(
            mapping_path
        ),
    },
    "elapsed_seconds": (
        time.time() - start_time
    ),
}

atomic_json(
    report_path,
    report,
    indent=2,
)

print("=" * 72)
print(
    "TADS-PPLX FINETUNING JSON EXPORT"
)
print("=" * 72)
print("Status: PASSED")
print(
    "Training records:",
    len(training_records),
)
print(
    "Unique IDs:",
    len(set(training_ids)),
)
print(
    "Dataset counts:",
    dict(sorted(dataset_counts.items())),
)
print(
    "Message role counts:",
    dict(sorted(role_counts.items())),
)
print(
    "Empty messages by role:",
    dict(
        sorted(
            empty_message_role_counts.items()
        )
    ),
)
print(
    "Records without a nonempty "
    "assistant response:",
    records_without_nonempty_assistant,
)
print(
    "Multi-turn records:",
    multi_turn_records,
)
print(
    "Records with system messages:",
    records_with_system_messages,
)
print("LAYA/Qwen/teacher use: NO")
print("Training JSON:", output_path)
print("ID mapping:", mapping_path)
print("Report:", report_path)
print(
    "Training JSON SHA256:",
    report["sha256"]["training_json"],
)
print(
    "Elapsed seconds:",
    report["elapsed_seconds"],
)
print()
print(
    "PPLX FINETUNING DATA EXPORT PASSED"
)
