#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os
import time

import pyarrow.parquet as pq
import torch


FIELDS = ("Task", "Topic", "Style", "Audience")
CRITERIA = (
    "completeness",
    "information_richness",
    "rarity",
    "complexity",
    "overall_score",
)
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

data_directory = (
    restart_root
    / "data"
    / "distribution_matching"
    / "original_tads"
)

parquet_path = (
    data_directory
    / "final_original_tads_laya_10000.parquet"
)

pt_path = (
    data_directory
    / "final_original_tads_laya_10000.pt"
)

fused_path = (
    data_directory
    / "fused_ids_original_tads_10000.json"
)

selection_report_path = (
    restart_root
    / "metadata"
    / "distribution_matching"
    / "25_original_tads_distribution_matching.json"
)

audit_report_path = (
    restart_root
    / "metadata"
    / "distribution_matching"
    / "26_final_original_tads_10000_audit.json"
)

for path in (
    parquet_path,
    pt_path,
    fused_path,
    selection_report_path,
):
    assert path.is_file(), f"Missing file: {path}"

print("Loading final Parquet dataset...")

table = pq.read_table(parquet_path)

assert table.num_rows == EXPECTED_ROWS, (
    f"Expected {EXPECTED_ROWS} rows, "
    f"found {table.num_rows}"
)

required_columns = {
    "dataset",
    "id",
    "messages",
    "source_position",
    "tads_adapter_id",
    "text_sha256",
    "Task_indices",
    "Topic_indices",
    "Style_indices",
    "Audience_indices",
    "all_five_expected_gte_5",
}

for criterion in CRITERIA:
    required_columns.add(
        f"{criterion}_score_1_to_10"
    )

missing_columns = (
    required_columns
    - set(table.column_names)
)

assert not missing_columns, (
    f"Missing final columns: {missing_columns}"
)

records = table.to_pylist()

adapter_ids = [
    str(record["tads_adapter_id"])
    for record in records
]

source_positions = [
    int(record["source_position"])
    for record in records
]

assert len(adapter_ids) == len(set(adapter_ids)), (
    "Final adapter IDs are not unique"
)

assert len(source_positions) == len(
    set(source_positions)
), "Final source positions are not unique"

dataset_counter = Counter()
truncated_count = 0
field_coverage = {}
score_statistics = {}

for row_index, record in enumerate(records):
    dataset = str(record["dataset"])
    original_id = str(record["id"])
    source_position = int(
        record["source_position"]
    )

    expected_adapter_id = (
        f"{dataset}::{original_id}::"
        f"{source_position:09d}"
    )

    assert (
        record["tads_adapter_id"]
        == expected_adapter_id
    ), (
        f"Adapter-ID mismatch at row {row_index}"
    )

    messages = record["messages"]

    assert isinstance(messages, list), (
        f"Messages are not a list at row {row_index}"
    )

    assert len(messages) >= 2, (
        f"Fewer than two messages at row {row_index}"
    )

    for message in messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert isinstance(message["content"], str)

    assert record["all_five_expected_gte_5"], (
        f"Ineligible sample at row {row_index}"
    )

    dataset_counter[dataset] += 1

    if bool(record.get("truncated", False)):
        truncated_count += 1

    for field in FIELDS:
        indices = record[f"{field}_indices"]

        expected_length = (
            1 if field == "Task" else 3
        )

        assert isinstance(indices, list)

        assert len(indices) == expected_length, (
            f"{field} cardinality error "
            f"at row {row_index}"
        )

        normalized = [
            int(index)
            for index in indices
        ]

        assert len(normalized) == len(
            set(normalized)
        ), (
            f"Repeated {field} cluster "
            f"at row {row_index}"
        )

        assert all(
            0 <= index < 100
            for index in normalized
        ), (
            f"Invalid {field} cluster "
            f"at row {row_index}"
        )

    for criterion in CRITERIA:
        score = float(
            record[
                f"{criterion}_score_1_to_10"
            ]
        )

        assert math.isfinite(score), (
            f"Non-finite {criterion} score "
            f"at row {row_index}"
        )

        assert 5.0 <= score <= 10.0, (
            f"{criterion} score outside "
            f"[5,10] at row {row_index}: {score}"
        )

print("Checking fused-ID order...")

fused_data = json.loads(
    fused_path.read_text(encoding="utf-8")
)

fused_ids = fused_data["fused_ids"]

assert len(fused_ids) == EXPECTED_ROWS
assert fused_ids == adapter_ids, (
    "Parquet row order does not match "
    "original TADS fusion order"
)

print("Checking PT copy...")

pt_records = torch.load(
    pt_path,
    map_location="cpu",
    weights_only=False,
)

assert isinstance(pt_records, list)
assert len(pt_records) == EXPECTED_ROWS

pt_adapter_ids = [
    str(record["tads_adapter_id"])
    for record in pt_records
]

assert pt_adapter_ids == adapter_ids, (
    "PT and Parquet row orders differ"
)

selection_report = json.loads(
    selection_report_path.read_text(
        encoding="utf-8"
    )
)

assert (
    selection_report["status"] == "PASSED"
)

assert (
    selection_report["qwen_scores_used"]
    is False
)

assert (
    selection_report["fusion"]["final_count"]
    == EXPECTED_ROWS
)

for field in FIELDS:
    observed = set()

    for record in records:
        observed.update(
            int(index)
            for index in record[
                f"{field}_indices"
            ]
        )

    field_coverage[field] = {
        "observed_clusters": len(observed),
        "missing_clusters": [
            f"{field}_{index}"
            for index in range(100)
            if index not in observed
        ],
    }

for criterion in CRITERIA:
    values = [
        float(
            record[
                f"{criterion}_score_1_to_10"
            ]
        )
        for record in records
    ]

    score_statistics[criterion] = {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }

audit_report = {
    "status": "PASSED",
    "rows": table.num_rows,
    "unique_adapter_ids": len(
        set(adapter_ids)
    ),
    "unique_source_positions": len(
        set(source_positions)
    ),
    "dataset_counts": dict(
        sorted(dataset_counter.items())
    ),
    "truncated_samples": truncated_count,
    "field_coverage": field_coverage,
    "score_statistics": score_statistics,
    "all_five_expected_scores_gte_5": True,
    "fusion_order_verified": True,
    "pt_parquet_order_verified": True,
    "qwen_scores_used_for_selection": False,
    "sha256": {
        "final_parquet": sha256_file(
            parquet_path
        ),
        "final_pt": sha256_file(pt_path),
        "fused_ids": sha256_file(fused_path),
    },
    "files": {
        "final_parquet": str(parquet_path),
        "final_pt": str(pt_path),
        "fused_ids": str(fused_path),
    },
    "elapsed_seconds": time.time() - start_time,
}

audit_report_path.write_text(
    json.dumps(
        audit_report,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print("=" * 72)
print("FINAL ORIGINAL-TADS 10K AUDIT")
print("=" * 72)
print("Status: PASSED")
print("Rows:", table.num_rows)
print(
    "Unique adapter IDs:",
    len(set(adapter_ids)),
)
print(
    "Unique source positions:",
    len(set(source_positions)),
)
print(
    "Dataset counts:",
    dict(sorted(dataset_counter.items())),
)
print("Truncated samples:", truncated_count)

print("\nCluster coverage:")

for field in FIELDS:
    print(
        f"  {field}: "
        f"{field_coverage[field]['observed_clusters']}"
        "/100"
    )

print("\nScore statistics:")

for criterion in CRITERIA:
    statistics = score_statistics[criterion]

    print(
        f"  {criterion}: "
        f"min={statistics['minimum']:.4f}, "
        f"mean={statistics['mean']:.4f}, "
        f"max={statistics['maximum']:.4f}"
    )

print("\nFusion order: VERIFIED")
print("PT/Parquet agreement: VERIFIED")
print("All five Laya scores >= 5: VERIFIED")
print("Qwen scores used for selection: NO")
print("Report:", audit_report_path)
print(
    "Elapsed seconds:",
    audit_report["elapsed_seconds"],
)
print("\nFINAL 10K TRAINING DATASET AUDIT PASSED")
