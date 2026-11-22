#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import statistics
import time

import pyarrow.parquet as pq
import torch


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

SCORE_COLUMNS = (
    "completeness",
    "information_richness",
    "rarity",
    "complexity",
    "overall_score",
)

EXPECTED_ROWS = 10000
EXPECTED_ELIGIBLE_ROWS = 75197


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                16 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_json(path, value):
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(path)


start_time = time.time()

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

data_directory = (
    root
    / "data"
    / "distribution_matching"
    / "original_tads_pplx"
)

metadata_directory = (
    root
    / "metadata"
    / "distribution_matching"
)

final_parquet_path = (
    data_directory
    / "final_original_tads_pplx_10000.parquet"
)

final_pt_path = (
    data_directory
    / "final_original_tads_pplx_10000.pt"
)

fused_ids_path = (
    data_directory
    / "fused_ids_original_tads_pplx_10000.json"
)

selection_report_path = (
    metadata_directory
    / "30_original_tads_pplx_distribution_matching.json"
)

for required_path in (
    final_parquet_path,
    final_pt_path,
    fused_ids_path,
    selection_report_path,
):
    assert required_path.is_file(), (
        f"Missing required file: {required_path}"
    )

print("Loading final Parquet dataset...")

table = pq.read_table(
    final_parquet_path
)

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
    "Task_indices",
    "Topic_indices",
    "Style_indices",
    "Audience_indices",
    *SCORE_COLUMNS,
}

missing_columns = (
    required_columns
    - set(table.column_names)
)

assert not missing_columns, (
    f"Missing columns: {missing_columns}"
)

for forbidden_column in (
    "gpt_scores",
    "llama_scores",
    "mistral_scores",
    "laya_scores",
    "qwen_scores",
):
    assert (
        forbidden_column
        not in table.column_names
    ), (
        f"Forbidden score column present: "
        f"{forbidden_column}"
    )

threshold_column = None

for candidate in (
    "all_five_gte_5",
    "all_five_expected_gte_5",
):
    if candidate in table.column_names:
        threshold_column = candidate
        break

assert threshold_column is not None

adapter_ids = table[
    "tads_adapter_id"
].to_pylist()

source_positions = [
    int(value)
    for value in table[
        "source_position"
    ].to_pylist()
]

assert len(set(adapter_ids)) == EXPECTED_ROWS
assert len(set(source_positions)) == EXPECTED_ROWS

assert all(
    bool(value)
    for value in table[
        threshold_column
    ].to_pylist()
), "At least one final sample fails >= 5"

print("Checking fused-ID order...")

with fused_ids_path.open(
    encoding="utf-8"
) as handle:
    fused_payload = json.load(handle)

fused_ids = fused_payload["fused_ids"]

assert len(fused_ids) == EXPECTED_ROWS
assert len(set(fused_ids)) == EXPECTED_ROWS
assert adapter_ids == fused_ids, (
    "Parquet order differs from fusion order"
)

print("Checking PT copy...")

pt_rows = torch.load(
    final_pt_path,
    map_location="cpu",
    weights_only=False,
)

assert isinstance(pt_rows, list)
assert len(pt_rows) == EXPECTED_ROWS

for index, row in enumerate(pt_rows):
    assert (
        row["tads_adapter_id"]
        == adapter_ids[index]
    )

    assert int(
        row["source_position"]
    ) == source_positions[index]

    assert (
        str(row["dataset"])
        == str(
            table["dataset"][index].as_py()
        )
    )

    assert (
        str(row["id"])
        == str(
            table["id"][index].as_py()
        )
    )

cluster_coverage = {}

for field in FIELDS:
    values = table[
        f"{field}_indices"
    ].to_pylist()

    expected_length = (
        1 if field == "Task" else 3
    )

    observed = set()

    for row_indices in values:
        assert isinstance(row_indices, list)
        assert (
            len(row_indices)
            == expected_length
        )

        converted = [
            int(value)
            for value in row_indices
        ]

        assert len(converted) == len(
            set(converted)
        )

        assert all(
            0 <= value < 100
            for value in converted
        )

        observed.update(converted)

    cluster_coverage[field] = {
        "observed": len(observed),
        "total": 100,
        "missing": [
            f"{field}_{index}"
            for index in range(100)
            if index not in observed
        ],
    }

score_statistics = {}

for column_name in SCORE_COLUMNS:
    values = [
        float(value)
        for value in table[
            column_name
        ].to_pylist()
    ]

    assert all(
        1.0 <= value <= 10.0
        for value in values
    ), (
        f"{column_name} contains a score "
        "outside 1–10"
    )

    assert min(values) >= 5.0, (
        f"{column_name} contains a score < 5"
    )

    score_statistics[column_name] = {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
        "standard_deviation": (
            statistics.pstdev(values)
        ),
    }

dataset_counts = dict(
    sorted(
        Counter(
            table["dataset"].to_pylist()
        ).items()
    )
)

truncated_rows = None

if "truncated" in table.column_names:
    truncated_rows = sum(
        bool(value)
        for value in table[
            "truncated"
        ].to_pylist()
    )

with selection_report_path.open(
    encoding="utf-8"
) as handle:
    selection_report = json.load(handle)

assert selection_report["status"] == "PASSED"

assert (
    selection_report["input"]["eligible_rows"]
    == EXPECTED_ELIGIBLE_ROWS
)

assert (
    selection_report["fusion"]["final_count"]
    == EXPECTED_ROWS
)

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

parquet_sha256 = sha256_file(
    final_parquet_path
)

pt_sha256 = sha256_file(
    final_pt_path
)

assert (
    selection_report[
        "outputs"
    ]["final_parquet_sha256"]
    == parquet_sha256
)

assert (
    selection_report[
        "outputs"
    ]["final_pt_sha256"]
    == pt_sha256
)

audit_report = {
    "status": "PASSED",
    "rows": table.num_rows,
    "unique_adapter_ids": len(
        set(adapter_ids)
    ),
    "unique_source_positions": len(
        set(source_positions)
    ),
    "dataset_counts": dataset_counts,
    "truncated_rows": truncated_rows,
    "cluster_coverage": cluster_coverage,
    "score_statistics": score_statistics,
    "fusion_order_verified": True,
    "pt_parquet_identity_verified": True,
    "all_five_pplx_scores_gte_5": True,
    "original_tads_provenance_verified": True,
    "laya_model_used": False,
    "qwen_model_used": False,
    "teacher_scores_used": False,
    "original_external_scores_used": False,
    "files": {
        "parquet": str(
            final_parquet_path
        ),
        "parquet_sha256": parquet_sha256,
        "pt": str(final_pt_path),
        "pt_sha256": pt_sha256,
        "fused_ids": str(
            fused_ids_path
        ),
        "fused_ids_sha256": (
            sha256_file(fused_ids_path)
        ),
        "selection_report": str(
            selection_report_path
        ),
    },
    "elapsed_seconds": (
        time.time() - start_time
    ),
}

audit_report_path = (
    metadata_directory
    / "31_final_original_tads_pplx_10000_audit.json"
)

atomic_json(
    audit_report_path,
    audit_report,
)

print("=" * 72)
print("FINAL ORIGINAL-TADS PPLX 10K AUDIT")
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
print("Dataset counts:", dataset_counts)
print("Truncated samples:", truncated_rows)

print("\nCluster coverage:")

for field in FIELDS:
    result = cluster_coverage[field]

    print(
        f"  {field}: "
        f"{result['observed']}/100"
    )

print("\nScore statistics:")

for column_name in SCORE_COLUMNS:
    result = score_statistics[
        column_name
    ]

    print(
        f"  {column_name}: "
        f"min={result['minimum']:.4f}, "
        f"mean={result['mean']:.4f}, "
        f"max={result['maximum']:.4f}"
    )

print("\nFusion order: VERIFIED")
print("PT/Parquet agreement: VERIFIED")
print("All five PPLX scores >= 5: VERIFIED")
print("Original TADS code provenance: VERIFIED")
print("LAYA/Qwen/teacher scores used: NO")
print(
    "Original GPT/LLaMA/Mistral scores used: NO"
)
print("Report:", audit_report_path)
print(
    "Elapsed seconds:",
    audit_report["elapsed_seconds"],
)
print()
print(
    "FINAL PPLX 10K TRAINING DATASET "
    "AUDIT PASSED"
)
