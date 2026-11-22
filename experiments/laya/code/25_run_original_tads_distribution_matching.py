#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import inspect
import json
import os
import random
import subprocess
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tads.step4.distribution_match import (
    _sample_to_match_distribution,
)
from tads.step4.fusion import (
    _equal_weight_fusion,
)
from tads.utils.tags import (
    analyze_all_items_from_data,
)


FIELDS = ("Task", "Topic", "Style", "Audience")
CLASSES = 100
TARGET_COUNT = 10000
RANDOM_SEED = 42

EXPECTED_COMMIT = (
    "5459de6e411fdd2997156a53beae3df68705b9ec"
)

EXPECTED_DISTRIBUTION_HASH = (
    "4cff4c48782ad0472beb58d54322bade"
    "7ea1d8fd790d2834c6eb9849eec14ca7"
)

EXPECTED_FUSION_HASH = (
    "189a0ea40bc0c9adceb1b53d15e773ef"
    "68f68be028a679a04911e938bb7d8d97"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def git_output(repository, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
    ).strip()


def distribution_statistics(
    reference_data,
    selected_data,
    field,
):
    reference_counter = (
        analyze_all_items_from_data(
            reference_data
        )[field]
    )

    selected_counter = (
        analyze_all_items_from_data(
            selected_data
        )[field]
    )

    reference_total = sum(reference_counter.values())
    selected_total = sum(selected_counter.values())

    reference_ratio = {
        tag: count / reference_total
        for tag, count in reference_counter.items()
    }

    selected_ratio = {
        tag: (
            selected_counter.get(tag, 0)
            / selected_total
        )
        for tag in reference_counter
    }

    absolute_differences = {
        tag: abs(
            reference_ratio[tag]
            - selected_ratio[tag]
        )
        for tag in reference_ratio
    }

    return {
        "reference_assignments": reference_total,
        "selected_assignments": selected_total,
        "reference_clusters": len(reference_counter),
        "selected_clusters": sum(
            selected_counter.get(tag, 0) > 0
            for tag in reference_counter
        ),
        "l1_distance": sum(
            absolute_differences.values()
        ),
        "maximum_absolute_difference": max(
            absolute_differences.values()
        ),
        "missing_clusters": sorted(
            tag
            for tag in reference_counter
            if selected_counter.get(tag, 0) == 0
        ),
    }


start_time = time.time()

restart_root = Path(
    os.environ.get(
        "BASE_RESTART_ROOT",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "laya_tads_anchor_classifier/"
        "laya_base_restart",
    )
).resolve()

paper_work = Path(
    os.environ.get(
        "PAPER_WORK",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "tads_paper_faithful",
    )
).resolve()

tads_repository = Path(
    os.environ.get(
        "TADS_REPO",
        "/home/gpuuser2/gpuuser2_a/atishya/tads",
    )
).resolve()

filtered_path = (
    restart_root
    / "data"
    / "quality_scoring"
    / "direct_laya_full"
    / "filtered"
    / "direct_laya_scores_gte_5.parquet"
)

source_path = (
    paper_work
    / "data"
    / "train-00000-of-00001.parquet"
)

reference_path = (
    paper_work
    / "data"
    / "cluster_ids.json"
)

distribution_source = (
    tads_repository
    / "src"
    / "tads"
    / "step4"
    / "distribution_match.py"
)

fusion_source = (
    tads_repository
    / "src"
    / "tads"
    / "step4"
    / "fusion.py"
)

output_directory = (
    restart_root
    / "data"
    / "distribution_matching"
    / "original_tads"
)

metadata_directory = (
    restart_root
    / "metadata"
    / "distribution_matching"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

metadata_directory.mkdir(
    parents=True,
    exist_ok=True,
)

for required_path in (
    filtered_path,
    source_path,
    reference_path,
    distribution_source,
    fusion_source,
):
    assert required_path.is_file(), (
        f"Missing required file: {required_path}"
    )

# Verify that the exact audited TADS source is used.
git_commit = git_output(
    tads_repository,
    "rev-parse",
    "HEAD",
)

git_changes = git_output(
    tads_repository,
    "status",
    "--porcelain",
)

distribution_hash = sha256_file(
    distribution_source
)

fusion_hash = sha256_file(
    fusion_source
)

assert git_commit == EXPECTED_COMMIT, (
    f"Unexpected TADS commit: {git_commit}"
)

assert not git_changes, (
    "The original TADS repository contains "
    "uncommitted changes."
)

assert (
    distribution_hash
    == EXPECTED_DISTRIBUTION_HASH
), "distribution_match.py hash changed"

assert fusion_hash == EXPECTED_FUSION_HASH, (
    "fusion.py hash changed"
)

loaded_distribution_source = Path(
    inspect.getsourcefile(
        _sample_to_match_distribution
    )
).resolve()

loaded_fusion_source = Path(
    inspect.getsourcefile(
        _equal_weight_fusion
    )
).resolve()

assert (
    loaded_distribution_source
    == distribution_source.resolve()
)

assert (
    loaded_fusion_source
    == fusion_source.resolve()
)

print("Loading filtered Laya examples...")

filtered_table = pq.read_table(filtered_path)

assert filtered_table.num_rows == 35476, (
    "Unexpected filtered row count: "
    f"{filtered_table.num_rows}"
)

required_columns = {
    "source_position",
    "dataset",
    "id",
    "text_sha256",
    "Task_indices",
    "Topic_indices",
    "Style_indices",
    "Audience_indices",
    "all_five_expected_gte_5",
}

missing_columns = (
    required_columns
    - set(filtered_table.column_names)
)

assert not missing_columns, (
    f"Missing filtered columns: {missing_columns}"
)

filtered_records = filtered_table.to_pylist()

assert all(
    bool(record["all_five_expected_gte_5"])
    for record in filtered_records
), "Filtered file contains an ineligible row"

source_positions = [
    int(record["source_position"])
    for record in filtered_records
]

assert len(source_positions) == len(
    set(source_positions)
), "Filtered source positions are not unique"

print("Building the TADS input adapter...")

adapter_records = []
adapter_id_to_filtered_index = {}

for filtered_index, record in enumerate(
    filtered_records
):
    source_position = int(
        record["source_position"]
    )

    dataset = str(record["dataset"])
    original_id = str(record["id"])

    adapter_id = (
        f"{dataset}::{original_id}::"
        f"{source_position:09d}"
    )

    assert adapter_id not in (
        adapter_id_to_filtered_index
    ), f"Duplicate adapter ID: {adapter_id}"

    generated_content = {}

    for field in FIELDS:
        raw_indices = record[f"{field}_indices"]

        assert isinstance(raw_indices, list), (
            f"{field} indices are not a list"
        )

        expected_length = (
            1 if field == "Task" else 3
        )

        assert len(raw_indices) == expected_length, (
            f"{adapter_id}, {field}: expected "
            f"{expected_length} indices, found "
            f"{len(raw_indices)}"
        )

        indices = [
            int(index)
            for index in raw_indices
        ]

        assert len(indices) == len(set(indices)), (
            f"{adapter_id}, {field}: repeated index"
        )

        assert all(
            0 <= index < CLASSES
            for index in indices
        ), (
            f"{adapter_id}, {field}: "
            "index outside 0–99"
        )

        generated_content[field] = [
            f"{field}_{index}"
            for index in indices
        ]

    adapter_record = {
        "id": adapter_id,
        "original_id": original_id,
        "dataset": dataset,
        "source_position": source_position,
        "text_sha256": str(
            record["text_sha256"]
        ),
        "generated_content": generated_content,
    }

    adapter_id_to_filtered_index[
        adapter_id
    ] = filtered_index

    adapter_records.append(adapter_record)

assert len(adapter_records) == 35476
assert len(adapter_id_to_filtered_index) == 35476

adapter_path = (
    output_directory
    / "eligible_laya_gte_5_tads_adapter.pt"
)

torch.save(adapter_records, adapter_path)

with reference_path.open(
    "r",
    encoding="utf-8",
) as handle:
    reference_data = json.load(handle)

assert isinstance(reference_data, list)
assert len(reference_data) == 6279

# The original run() function applies these seeds
# before calling the selection function.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

selected_by_field = {}
selected_ids_by_field = {}
field_statistics = {}

print("\nRunning original TADS distribution matching...")

for field in FIELDS:
    field_start = time.time()

    selected = _sample_to_match_distribution(
        adapter_records,
        reference_data,
        field,
        target_total=TARGET_COUNT,
    )

    assert len(selected) == TARGET_COUNT, (
        f"{field}: original TADS selected "
        f"{len(selected)}, expected {TARGET_COUNT}"
    )

    selected_ids = [
        item["id"]
        for item in selected
    ]

    assert len(selected_ids) == len(
        set(selected_ids)
    ), f"{field}: duplicate selected IDs"

    selected_path = (
        output_directory
        / f"{field.lower()}_selected_10000.pt"
    )

    torch.save(selected, selected_path)

    selected_by_field[field] = selected
    selected_ids_by_field[field] = selected_ids

    field_statistics[field] = {
        **distribution_statistics(
            reference_data,
            selected,
            field,
        ),
        "selected_file": str(selected_path),
        "elapsed_seconds": (
            time.time() - field_start
        ),
    }

print("\nRunning original TADS equal-weight fusion...")

fused_ids = _equal_weight_fusion(
    *[
        selected_ids_by_field[field]
        for field in FIELDS
    ],
    target_count=TARGET_COUNT,
)

assert len(fused_ids) == TARGET_COUNT, (
    f"Fusion returned {len(fused_ids)} IDs"
)

assert len(fused_ids) == len(
    set(fused_ids)
), "Fusion returned duplicate IDs"

all_votes = Counter()

for field in FIELDS:
    all_votes.update(
        selected_ids_by_field[field]
    )

vote_distribution = Counter(
    all_votes[adapter_id]
    for adapter_id in fused_ids
)

fused_json_path = (
    output_directory
    / "fused_ids_original_tads_10000.json"
)

fused_json_path.write_text(
    json.dumps(
        {
            "fusion_config": {
                "dimensions": list(FIELDS),
                "gte_threshold": 5,
                "target_count": TARGET_COUNT,
                "random_seed": RANDOM_SEED,
                "total_methods": len(FIELDS),
                "final_count": len(fused_ids),
                "algorithm": (
                    "Original TADS equal-weight voting"
                ),
            },
            "vote_distribution_in_final": {
                str(vote): count
                for vote, count in sorted(
                    vote_distribution.items()
                )
            },
            "fused_ids": fused_ids,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("Materializing final original source samples...")

filtered_row_indices = [
    adapter_id_to_filtered_index[
        adapter_id
    ]
    for adapter_id in fused_ids
]

selected_filtered_table = filtered_table.take(
    pa.array(
        filtered_row_indices,
        type=pa.int64(),
    )
)

final_source_positions = [
    int(value)
    for value in selected_filtered_table[
        "source_position"
    ].to_pylist()
]

source_table = pq.read_table(source_path)

assert max(final_source_positions) < (
    source_table.num_rows
)

selected_source_table = source_table.take(
    pa.array(
        final_source_positions,
        type=pa.int64(),
    )
)

source_datasets = selected_source_table[
    "dataset"
].to_pylist()

source_ids = selected_source_table[
    "id"
].to_pylist()

filtered_datasets = selected_filtered_table[
    "dataset"
].to_pylist()

filtered_ids = selected_filtered_table[
    "id"
].to_pylist()

for index in range(TARGET_COUNT):
    assert (
        source_datasets[index]
        == filtered_datasets[index]
    ), (
        f"Dataset mismatch at final row {index}"
    )

    assert (
        source_ids[index]
        == filtered_ids[index]
    ), (
        f"Source ID mismatch at final row {index}"
    )

final_table = selected_source_table.append_column(
    "source_position",
    pa.array(
        final_source_positions,
        type=pa.int64(),
    ),
)

final_table = final_table.append_column(
    "tads_adapter_id",
    pa.array(fused_ids),
)

identity_columns = {
    "source_position",
    "dataset",
    "id",
}

for column_name in (
    selected_filtered_table.column_names
):
    if column_name in identity_columns:
        continue

    output_name = column_name

    if output_name in final_table.column_names:
        output_name = f"laya_{output_name}"

    final_table = final_table.append_column(
        output_name,
        selected_filtered_table[column_name],
    )

final_parquet_path = (
    output_directory
    / "final_original_tads_laya_10000.parquet"
)

pq.write_table(
    final_table,
    final_parquet_path,
    compression="zstd",
)

final_pt_path = (
    output_directory
    / "final_original_tads_laya_10000.pt"
)

torch.save(
    final_table.to_pylist(),
    final_pt_path,
)

report = {
    "status": "PASSED",
    "method": (
        "Original TADS distribution matching "
        "and equal-weight fusion"
    ),
    "qwen_scores_used": False,
    "quality_filter": (
        "All five direct Laya expected scores >= 5"
    ),
    "input": {
        "filtered_file": str(filtered_path),
        "eligible_rows": filtered_table.num_rows,
        "reference_file": str(reference_path),
        "reference_rows": len(reference_data),
        "source_file": str(source_path),
        "source_rows": source_table.num_rows,
    },
    "original_tads_provenance": {
        "repository": str(tads_repository),
        "git_commit": git_commit,
        "working_tree_clean": not bool(git_changes),
        "distribution_source": str(
            distribution_source
        ),
        "distribution_sha256": distribution_hash,
        "fusion_source": str(fusion_source),
        "fusion_sha256": fusion_hash,
    },
    "configuration": {
        "fields": list(FIELDS),
        "target_count": TARGET_COUNT,
        "random_seed": RANDOM_SEED,
        "threshold": 5,
        "adapter_id_format": (
            "dataset::original_id::"
            "zero-padded-source-position"
        ),
    },
    "per_field": field_statistics,
    "fusion": {
        "final_count": len(fused_ids),
        "vote_distribution_in_final": {
            str(vote): count
            for vote, count in sorted(
                vote_distribution.items()
            )
        },
    },
    "outputs": {
        "adapter": str(adapter_path),
        "fused_ids": str(fused_json_path),
        "final_parquet": str(
            final_parquet_path
        ),
        "final_pt": str(final_pt_path),
    },
    "elapsed_seconds": time.time() - start_time,
}

report_path = (
    metadata_directory
    / "25_original_tads_distribution_matching.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print("=" * 72)
print("ORIGINAL TADS + LAYA FINAL SELECTION PASSED")
print("=" * 72)
print("Eligible Laya samples:", len(adapter_records))

for field in FIELDS:
    statistics = field_statistics[field]

    print(f"\n{field}")
    print(
        "  Selected:",
        len(selected_by_field[field]),
    )
    print(
        "  Observed clusters:",
        f"{statistics['selected_clusters']}/100",
    )
    print(
        "  L1 distance:",
        statistics["l1_distance"],
    )
    print(
        "  Maximum absolute difference:",
        statistics[
            "maximum_absolute_difference"
        ],
    )

print("\nFinal fused samples:", len(fused_ids))
print(
    "Final vote distribution:",
    dict(sorted(vote_distribution.items())),
)
print("Qwen scores used: NO")
print("Final Parquet:", final_parquet_path)
print("Final PT:", final_pt_path)
print("Fused IDs:", fused_json_path)
print("Report:", report_path)
print(
    "Elapsed seconds:",
    report["elapsed_seconds"],
)
print(
    "\nORIGINAL TADS DISTRIBUTION "
    "MATCHING AND FUSION VERIFIED"
)
