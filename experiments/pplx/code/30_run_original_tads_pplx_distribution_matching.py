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


EXPECTED_COMMIT = (
    "5459de6e411fdd2997156a53beae3df68705b9ec"
)

EXPECTED_DISTRIBUTION_HASH = (
    "4cff4c48782ad0472beb58d54322bade"
    "7ea1d8fd790d2834c6eb9849eec14ca7"
)

EXPECTED_FUSION_HASH = (
    "189a0ea40bc0c9adceb1b53d15e773e"
    "f68f68be028a679a04911e938bb7d8d97"
)

EXPECTED_ELIGIBLE_ROWS = 75197

FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

CLASSES = 100
TARGET_COUNT = 10000
RANDOM_SEED = 42


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def git_output(repository, *arguments):
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


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

    clusters = [
        f"{field}_{index}"
        for index in range(CLASSES)
    ]

    reference_total = sum(
        reference_counter.values()
    )

    selected_total = sum(
        selected_counter.values()
    )

    assert reference_total > 0
    assert selected_total > 0

    reference_ratio = {
        cluster: (
            reference_counter.get(cluster, 0)
            / reference_total
        )
        for cluster in clusters
    }

    selected_ratio = {
        cluster: (
            selected_counter.get(cluster, 0)
            / selected_total
        )
        for cluster in clusters
    }

    absolute_differences = {
        cluster: abs(
            reference_ratio[cluster]
            - selected_ratio[cluster]
        )
        for cluster in clusters
    }

    return {
        "reference_assignments": reference_total,
        "selected_assignments": selected_total,
        "reference_clusters": sum(
            reference_counter.get(
                cluster,
                0,
            ) > 0
            for cluster in clusters
        ),
        "selected_clusters": sum(
            selected_counter.get(
                cluster,
                0,
            ) > 0
            for cluster in clusters
        ),
        "missing_selected_clusters": [
            cluster
            for cluster in clusters
            if (
                selected_counter.get(
                    cluster,
                    0,
                )
                == 0
            )
        ],
        "l1_distance": sum(
            absolute_differences.values()
        ),
        "maximum_absolute_difference": max(
            absolute_differences.values()
        ),
    }


start_time = time.time()

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

paper_work = Path(
    os.environ["PAPER_WORK"]
).resolve()

tads_repository = Path(
    os.environ["TADS_REPO"]
).resolve()

filtered_path = (
    root
    / "data"
    / "quality_scoring"
    / "filtered"
    / "pplx_scores_all_five_gte_5_75197.parquet"
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

status_path = (
    output_directory
    / "30_original_tads_pplx_status.json"
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

atomic_json(
    status_path,
    {
        "status": "STARTING",
        "stage": "validation",
        "filtered_file": str(filtered_path),
        "target_count": TARGET_COUNT,
    },
)

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
    "Original TADS repository contains "
    "uncommitted changes."
)

assert (
    distribution_hash
    == EXPECTED_DISTRIBUTION_HASH
), "distribution_match.py hash changed"

assert (
    fusion_hash
    == EXPECTED_FUSION_HASH
), "fusion.py hash changed"

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

print("Loading PPLX-filtered eligible samples...")

filtered_table = pq.read_table(
    filtered_path
)

assert (
    filtered_table.num_rows
    == EXPECTED_ELIGIBLE_ROWS
), (
    "Unexpected eligible row count: "
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
}

missing_columns = (
    required_columns
    - set(filtered_table.column_names)
)

assert not missing_columns, (
    f"Missing columns: {missing_columns}"
)

threshold_column = None

for candidate in (
    "all_five_gte_5",
    "all_five_expected_gte_5",
):
    if candidate in filtered_table.column_names:
        threshold_column = candidate
        break

assert threshold_column is not None, (
    "All-five >= 5 threshold column was not found"
)

filtered_records = filtered_table.to_pylist()

assert all(
    bool(record[threshold_column])
    for record in filtered_records
), "Filtered dataset contains an ineligible row"

source_positions = [
    int(record["source_position"])
    for record in filtered_records
]

assert len(source_positions) == len(
    set(source_positions)
), "Filtered source positions are not unique"

print("Building original-TADS PPLX input adapter...")

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

    assert (
        adapter_id
        not in adapter_id_to_filtered_index
    ), f"Duplicate adapter ID: {adapter_id}"

    generated_content = {}

    for field in FIELDS:
        raw_indices = record[
            f"{field}_indices"
        ]

        assert isinstance(raw_indices, list), (
            f"{adapter_id}, {field}: "
            "indices are not a list"
        )

        expected_length = (
            1 if field == "Task" else 3
        )

        assert (
            len(raw_indices)
            == expected_length
        ), (
            f"{adapter_id}, {field}: expected "
            f"{expected_length} indices; found "
            f"{len(raw_indices)}"
        )

        indices = [
            int(index)
            for index in raw_indices
        ]

        assert len(indices) == len(
            set(indices)
        ), (
            f"{adapter_id}, {field}: "
            "repeated cluster index"
        )

        assert all(
            0 <= index < CLASSES
            for index in indices
        ), (
            f"{adapter_id}, {field}: "
            "cluster index outside 0–99"
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

    adapter_records.append(
        adapter_record
    )

assert (
    len(adapter_records)
    == EXPECTED_ELIGIBLE_ROWS
)

assert (
    len(adapter_id_to_filtered_index)
    == EXPECTED_ELIGIBLE_ROWS
)

adapter_path = (
    output_directory
    / "eligible_pplx_gte_5_tads_adapter.pt"
)

torch.save(
    adapter_records,
    adapter_path,
)

with reference_path.open(
    "r",
    encoding="utf-8",
) as handle:
    reference_data = json.load(handle)

assert isinstance(reference_data, list)
assert len(reference_data) == 6279

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

selected_by_field = {}
selected_ids_by_field = {}
field_statistics = {}

atomic_json(
    status_path,
    {
        "status": "RUNNING",
        "stage": "distribution_matching",
        "eligible_rows": len(adapter_records),
        "target_count": TARGET_COUNT,
    },
)

print()
print(
    "Running original TADS distribution matching..."
)

for field in FIELDS:
    field_start = time.time()

    selected = (
        _sample_to_match_distribution(
            adapter_records,
            reference_data,
            field,
            target_total=TARGET_COUNT,
        )
    )

    assert len(selected) == TARGET_COUNT, (
        f"{field}: original TADS selected "
        f"{len(selected)}; expected "
        f"{TARGET_COUNT}"
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

    torch.save(
        selected,
        selected_path,
    )

    selected_by_field[field] = selected
    selected_ids_by_field[field] = (
        selected_ids
    )

    field_statistics[field] = {
        **distribution_statistics(
            reference_data,
            selected,
            field,
        ),
        "selected_file": str(
            selected_path
        ),
        "elapsed_seconds": (
            time.time() - field_start
        ),
    }

print()
print(
    "Running original TADS equal-weight fusion..."
)

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
    / "fused_ids_original_tads_pplx_10000.json"
)

atomic_json(
    fused_json_path,
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
)

print(
    "Materializing final original source samples..."
)

filtered_row_indices = [
    adapter_id_to_filtered_index[
        adapter_id
    ]
    for adapter_id in fused_ids
]

selected_filtered_table = (
    filtered_table.take(
        pa.array(
            filtered_row_indices,
            type=pa.int64(),
        )
    )
)

final_source_positions = [
    int(value)
    for value in selected_filtered_table[
        "source_position"
    ].to_pylist()
]

source_table = pq.read_table(
    source_path,
    columns=[
        "dataset",
        "id",
        "messages",
    ],
)

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

final_table = (
    selected_source_table.append_column(
        "source_position",
        pa.array(
            final_source_positions,
            type=pa.int64(),
        ),
    )
)

final_table = final_table.append_column(
    "tads_adapter_id",
    pa.array(fused_ids),
)

identity_columns = {
    "source_position",
    "dataset",
    "id",
    "messages",
}

for column_name in (
    selected_filtered_table.column_names
):
    if column_name in identity_columns:
        continue

    output_name = column_name

    if output_name in final_table.column_names:
        output_name = (
            f"pplx_{output_name}"
        )

    final_table = final_table.append_column(
        output_name,
        selected_filtered_table[
            column_name
        ],
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
        not in final_table.column_names
    ), (
        f"Forbidden score column present: "
        f"{forbidden_column}"
    )

assert final_table.num_rows == TARGET_COUNT

assert len(
    set(
        final_table[
            "tads_adapter_id"
        ].to_pylist()
    )
) == TARGET_COUNT

assert all(
    bool(value)
    for value in final_table[
        threshold_column
    ].to_pylist()
)

final_parquet_path = (
    output_directory
    / "final_original_tads_pplx_10000.parquet"
)

pq.write_table(
    final_table,
    final_parquet_path,
    compression="zstd",
)

final_pt_path = (
    output_directory
    / "final_original_tads_pplx_10000.pt"
)

torch.save(
    final_table.to_pylist(),
    final_pt_path,
)

dataset_counts = dict(
    sorted(
        Counter(
            final_table[
                "dataset"
            ].to_pylist()
        ).items()
    )
)

report = {
    "status": "PASSED",
    "method": (
        "Original TADS distribution matching "
        "and equal-weight fusion"
    ),
    "runtime_model_family": (
        "perplexity-ai/pplx-decider-v1.2"
    ),
    "cluster_assignment_model": (
        "Fine-tuned PPLX epoch 8"
    ),
    "quality_scoring_model": (
        "Base/native PPLX"
    ),
    "quality_filter": (
        "All five PPLX scores >= 5"
    ),
    "laya_model_used": False,
    "qwen_model_used": False,
    "teacher_scores_used": False,
    "original_external_score_columns_used": False,
    "input": {
        "filtered_file": str(
            filtered_path
        ),
        "filtered_sha256": sha256_file(
            filtered_path
        ),
        "eligible_rows": (
            filtered_table.num_rows
        ),
        "reference_file": str(
            reference_path
        ),
        "reference_rows": len(
            reference_data
        ),
        "source_file": str(source_path),
        "source_rows": source_table.num_rows,
    },
    "original_tads_provenance": {
        "repository": str(
            tads_repository
        ),
        "git_commit": git_commit,
        "working_tree_clean": (
            not bool(git_changes)
        ),
        "distribution_source": str(
            distribution_source
        ),
        "distribution_sha256": (
            distribution_hash
        ),
        "fusion_source": str(
            fusion_source
        ),
        "fusion_sha256": fusion_hash,
    },
    "configuration": {
        "fields": list(FIELDS),
        "classes_per_field": CLASSES,
        "target_count": TARGET_COUNT,
        "random_seed": RANDOM_SEED,
        "threshold": 5,
        "threshold_operator": ">=",
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
    "final_dataset_counts": (
        dataset_counts
    ),
    "outputs": {
        "adapter": str(adapter_path),
        "fused_ids": str(
            fused_json_path
        ),
        "final_parquet": str(
            final_parquet_path
        ),
        "final_parquet_sha256": (
            sha256_file(
                final_parquet_path
            )
        ),
        "final_pt": str(final_pt_path),
        "final_pt_sha256": sha256_file(
            final_pt_path
        ),
    },
    "elapsed_seconds": (
        time.time() - start_time
    ),
}

report_path = (
    metadata_directory
    / "30_original_tads_pplx_distribution_matching.json"
)

atomic_json(
    report_path,
    report,
)

atomic_json(
    status_path,
    {
        "status": "PASSED",
        "stage": "complete",
        "eligible_rows": (
            filtered_table.num_rows
        ),
        "final_rows": (
            final_table.num_rows
        ),
        "final_parquet": str(
            final_parquet_path
        ),
        "final_pt": str(final_pt_path),
        "report": str(report_path),
        "elapsed_seconds": (
            report["elapsed_seconds"]
        ),
    },
)

print("=" * 72)
print(
    "ORIGINAL TADS + PPLX FINAL SELECTION PASSED"
)
print("=" * 72)
print(
    "Eligible PPLX samples:",
    len(adapter_records),
)

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

print(
    "\nFinal fused samples:",
    len(fused_ids),
)
print(
    "Final vote distribution:",
    dict(
        sorted(
            vote_distribution.items()
        )
    ),
)
print(
    "Final dataset counts:",
    dataset_counts,
)
print("LAYA model used: NO")
print("Qwen model used: NO")
print(
    "Original GPT/LLaMA/Mistral scores used: NO"
)
print(
    "Final Parquet:",
    final_parquet_path,
)
print("Final PT:", final_pt_path)
print("Fused IDs:", fused_json_path)
print("Report:", report_path)
print(
    "Elapsed seconds:",
    report["elapsed_seconds"],
)
print()
print(
    "ORIGINAL TADS PPLX DISTRIBUTION "
    "MATCHING AND FUSION VERIFIED"
)
