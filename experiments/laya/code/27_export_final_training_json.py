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
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


start_time = time.time()

restart_root = Path(
    os.environ["BASE_RESTART_ROOT"]
).resolve()

input_path = (
    restart_root
    / "data"
    / "distribution_matching"
    / "original_tads"
    / "final_original_tads_laya_10000.parquet"
)

output_path = (
    restart_root
    / "data"
    / "model_finetune"
    / "tads_laya_10k_gte_5_dataset.json"
)

mapping_path = (
    restart_root
    / "data"
    / "model_finetune"
    / "tads_laya_10k_id_mapping.json"
)

report_path = (
    restart_root
    / "metadata"
    / "model_finetune"
    / "27_training_json_export.json"
)

assert input_path.is_file(), (
    f"Missing final Parquet: {input_path}"
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

for position, row in enumerate(rows):
    dataset = str(row["dataset"])
    original_id = str(row["id"])
    adapter_id = str(row["tads_adapter_id"])
    source_position = int(row["source_position"])
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
            empty_message_role_counts[role] += 1

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
    ), f"No assistant response in {adapter_id}"

    if not any(
        message["role"] == "assistant"
        and message["content"].strip()
        for message in normalized_messages
    ):
        records_without_nonempty_assistant += 1

    # This is the exact record structure consumed
    # by the original TADS finetune.py program.
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
            "source_position": source_position,
            "training_position": position,
        }
    )

    dataset_counts[dataset] += 1

training_ids = [
    record["id"]
    for record in training_records
]

assert len(training_records) == EXPECTED_ROWS
assert len(training_ids) == len(set(training_ids))

output_path.write_text(
    json.dumps(
        training_records,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

mapping_path.write_text(
    json.dumps(
        id_mapping,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# Reload the output to verify its serialized form.
reloaded = json.loads(
    output_path.read_text(encoding="utf-8")
)

assert len(reloaded) == EXPECTED_ROWS
assert [
    record["id"]
    for record in reloaded
] == training_ids

assert all(
    set(record.keys())
    == {"dataset", "id", "messages"}
    for record in reloaded
)

report = {
    "status": "PASSED",
    "input": str(input_path),
    "output": str(output_path),
    "id_mapping": str(mapping_path),
    "records": len(training_records),
    "unique_ids": len(set(training_ids)),
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "message_role_counts": dict(
        sorted(role_counts.items())
    ),
    "empty_message_role_counts": dict(
        sorted(empty_message_role_counts.items())
    ),
    "records_without_nonempty_assistant": (
        records_without_nonempty_assistant
    ),
    "empty_message_policy": (
        "Preserved exactly as present in the "
        "original source dataset"
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
    "qwen_scores_used_for_selection": False,
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
    "elapsed_seconds": time.time() - start_time,
}

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print("=" * 72)
print("TADS-LAYA FINETUNING JSON EXPORT")
print("=" * 72)
print("Status: PASSED")
print("Training records:", len(training_records))
print("Unique IDs:", len(set(training_ids)))
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
    dict(sorted(empty_message_role_counts.items())),
)
print(
    "Records without a nonempty assistant response:",
    records_without_nonempty_assistant,
)
print("Training JSON:", output_path)
print("ID mapping:", mapping_path)
print("Report:", report_path)
print("Training JSON SHA256:", report["sha256"]["training_json"])
print("Elapsed seconds:", report["elapsed_seconds"])
print("\nFINETUNING DATA EXPORT PASSED")
