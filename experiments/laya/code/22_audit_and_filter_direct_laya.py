#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


THRESHOLD = 5.0

FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

METRICS = (
    "completeness",
    "information_richness",
    "rarity",
    "complexity",
    "overall_score",
)

root = Path(
    os.environ.get(
        "BASE_RESTART_ROOT",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "laya_tads_anchor_classifier/laya_base_restart",
    )
).resolve()

input_path = (
    root
    / "data"
    / "quality_scoring"
    / "direct_laya_full"
    / "direct_laya_scores_all_300932.parquet"
)

output_directory = (
    root
    / "data"
    / "quality_scoring"
    / "direct_laya_full"
    / "filtered"
)

filtered_path = (
    output_directory
    / "direct_laya_scores_gte_5.parquet"
)

positions_path = (
    output_directory
    / "eligible_source_positions_gte_5.npy"
)

report_path = (
    root
    / "metadata"
    / "quality_scoring"
    / "22_direct_laya_filter_audit.json"
)

output_directory.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)


print("Loading complete direct-Laya scores...")

table = pq.read_table(input_path).combine_chunks()

assert len(table) == 300932

positions = table[
    "source_position"
].to_numpy(zero_copy_only=False)

assert np.array_equal(
    positions,
    np.arange(len(table)),
), "Source ordering is incorrect"


expected_columns = [
    f"{metric}_score_1_to_10"
    for metric in METRICS
]

argmax_columns = [
    f"{metric}_argmax_1_to_10"
    for metric in METRICS
]

expected_scores = np.column_stack(
    [
        table[column].to_numpy(
            zero_copy_only=False
        )
        for column in expected_columns
    ]
)

argmax_scores = np.column_stack(
    [
        table[column].to_numpy(
            zero_copy_only=False
        )
        for column in argmax_columns
    ]
)

assert np.isfinite(expected_scores).all()
assert expected_scores.min() >= 1.0
assert expected_scores.max() <= 10.0

assert argmax_scores.min() >= 1
assert argmax_scores.max() <= 10


expected_threshold_counts = {}
argmax_threshold_counts = {}

for threshold in range(1, 11):
    expected_threshold_counts[str(threshold)] = int(
        np.all(
            expected_scores >= float(threshold),
            axis=1,
        ).sum()
    )

    argmax_threshold_counts[str(threshold)] = int(
        np.all(
            argmax_scores >= threshold,
            axis=1,
        ).sum()
    )


eligible_mask = np.all(
    expected_scores >= THRESHOLD,
    axis=1,
)

eligible_positions = positions[eligible_mask]

assert len(eligible_positions) == 35476

filtered_table = table.filter(
    pa.array(eligible_mask)
)

assert len(filtered_table) == len(eligible_positions)

pq.write_table(
    filtered_table,
    filtered_path,
    compression="zstd",
)

np.save(
    positions_path,
    eligible_positions,
    allow_pickle=False,
)


dataset_counts = Counter(
    filtered_table[
        "dataset"
    ].to_pylist()
)

field_coverage = {}

for field in FIELDS:
    values = filtered_table[
        f"{field}_indices"
    ].to_pylist()

    observed = sorted(
        {
            int(index)
            for row in values
            for index in row
        }
    )

    field_coverage[field] = {
        "observed_clusters": len(observed),
        "missing_clusters": [
            index
            for index in range(100)
            if index not in observed
        ],
    }


truncated = table[
    "truncated"
].to_numpy(zero_copy_only=False)

eligible_truncated = filtered_table[
    "truncated"
].to_numpy(zero_copy_only=False)


metric_statistics = {}

for index, metric in enumerate(METRICS):
    values = expected_scores[:, index]
    selected_values = values[eligible_mask]

    metric_statistics[metric] = {
        "all_samples": {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        },
        "eligible_samples": {
            "mean": float(selected_values.mean()),
            "standard_deviation": float(
                selected_values.std()
            ),
            "minimum": float(
                selected_values.min()
            ),
            "median": float(
                np.median(selected_values)
            ),
            "maximum": float(
                selected_values.max()
            ),
        },
    }


report = {
    "status": "PASSED",
    "method": "direct_base_laya_native_scoring",
    "qwen_scores_used": False,
    "input_rows": len(table),
    "threshold": THRESHOLD,
    "threshold_operator": (
        "all five expected scores >= threshold"
    ),
    "eligible_rows": len(eligible_positions),
    "eligible_percentage": (
        len(eligible_positions) / len(table) * 100
    ),
    "expected_threshold_counts": (
        expected_threshold_counts
    ),
    "argmax_threshold_counts": (
        argmax_threshold_counts
    ),
    "all_truncated_rows": int(truncated.sum()),
    "eligible_truncated_rows": int(
        eligible_truncated.sum()
    ),
    "eligible_dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "eligible_field_coverage": field_coverage,
    "metric_statistics": metric_statistics,
    "filtered_output": str(filtered_path),
    "eligible_positions": str(positions_path),
}

temporary = report_path.with_suffix(
    report_path.suffix + ".tmp"
)

temporary.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

os.replace(temporary, report_path)


print("=" * 72)
print("DIRECT LAYA FILTER AUDIT")
print("=" * 72)
print("Input rows:", len(table))
print("Threshold:", THRESHOLD)
print("Eligible rows:", len(eligible_positions))
print(
    "Eligible percentage:",
    f"{report['eligible_percentage']:.4f}%",
)
print(
    "Eligible dataset counts:",
    dict(sorted(dataset_counts.items())),
)
print()

for field in FIELDS:
    details = field_coverage[field]

    print(
        f"{field}: "
        f"{details['observed_clusters']}/100 clusters, "
        f"missing={details['missing_clusters']}"
    )

print()
print("Qwen scores used: NO")
print("Filtered output:", filtered_path)
print("Eligible positions:", positions_path)
print("Report:", report_path)
print()
print("DIRECT LAYA FILTER AUDIT PASSED")
