#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

TOP_K = {
    "Task": 1,
    "Topic": 3,
    "Style": 3,
    "Audience": 3,
}

EXPECTED_ROWS = 300932
EXPECTED_SHARDS = 301
EXPECTED_CHECKPOINT_SHA256 = (
    "293343768aa1f238f1a8a0e7795a270e7fed49282d198b46bcd157f9123d95f2"
)

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

runtime_root = Path(
    os.environ["PPLX_RUNTIME_ROOT"]
).resolve()

shard_root = (
    runtime_root
    / "cluster_predictions"
    / "pplx_epoch8_clusters_full_300932"
)

output_path = (
    root
    / "data"
    / "source_predictions"
    / "pplx_epoch8_source_predictions_all_300932.parquet"
)

temporary_path = output_path.with_suffix(
    ".parquet.incomplete"
)

status_path = (
    root
    / "data"
    / "source_predictions"
    / "22_pplx_cluster_merge_status.json"
)

report_path = (
    root
    / "metadata"
    / "source_predictions"
    / "22_pplx_cluster_merge_audit.json"
)


def write_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


schema_fields = [
    pa.field(
        "source_position",
        pa.int64(),
        nullable=False,
    ),
    pa.field(
        "dataset",
        pa.string(),
        nullable=False,
    ),
    pa.field(
        "id",
        pa.string(),
        nullable=False,
    ),
    pa.field(
        "text_sha256",
        pa.string(),
        nullable=False,
    ),
]

for field in FIELDS:
    top_k = TOP_K[field]

    schema_fields.extend(
        [
            pa.field(
                f"{field}_indices",
                pa.list_(
                    pa.int16(),
                    top_k,
                ),
                nullable=False,
            ),
            pa.field(
                f"{field}_probabilities",
                pa.list_(
                    pa.float32(),
                    top_k,
                ),
                nullable=False,
            ),
            pa.field(
                f"{field}_distribution",
                pa.list_(
                    pa.float32(),
                    100,
                ),
                nullable=False,
            ),
            pa.field(
                f"{field}_max_probability",
                pa.float32(),
                nullable=False,
            ),
            pa.field(
                f"{field}_entropy",
                pa.float32(),
                nullable=False,
            ),
        ]
    )

schema = pa.schema(schema_fields)

started = time.time()

write_json(
    status_path,
    {
        "status": "STARTING",
        "stage": "loading",
        "processed_rows": 0,
        "requested_rows": EXPECTED_ROWS,
        "input": str(shard_root),
        "output": str(output_path),
        "qwen_used": False,
        "laya_model_used": False,
        "laya_predictions_used": False,
        "laya_scores_used": False,
    },
)

assert shard_root.is_dir(), (
    f"Missing shard directory: {shard_root}"
)

shards = sorted(
    shard_root.glob("shard_*.jsonl")
)

manifests = sorted(
    shard_root.glob(
        "shard_*.complete.json"
    )
)

assert len(shards) == EXPECTED_SHARDS, (
    len(shards)
)

assert len(manifests) == EXPECTED_SHARDS, (
    len(manifests)
)

dataset_counts = Counter()
truncated_rows = 0
expected_position = 0
maximum_sum_error = {
    field: 0.0
    for field in FIELDS
}
top1_classes = {
    field: set()
    for field in FIELDS
}
topk_classes = {
    field: set()
    for field in FIELDS
}
sum_max_probability = {
    field: 0.0
    for field in FIELDS
}
sum_entropy = {
    field: 0.0
    for field in FIELDS
}

if temporary_path.exists():
    temporary_path.unlink()

writer = pq.ParquetWriter(
    where=str(temporary_path),
    schema=schema,
    compression="zstd",
    use_dictionary=[
        "dataset",
        "id",
    ],
    write_statistics=True,
)

try:
    for shard_index, shard_path in enumerate(
        shards
    ):
        expected_manifest = (
            shard_root
            / f"shard_{shard_index:06d}.complete.json"
        )

        assert expected_manifest.is_file(), (
            f"Missing manifest: {expected_manifest}"
        )

        with expected_manifest.open(
            encoding="utf-8"
        ) as handle:
            manifest = json.load(handle)

        assert manifest["status"] == "COMPLETE"
        assert int(manifest["shard_index"]) == shard_index
        assert int(manifest["start_position"]) == (
            expected_position
        )
        assert manifest["checkpoint_sha256"] == (
            EXPECTED_CHECKPOINT_SHA256
        )

        parquet_rows = []

        with shard_path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                if not line.strip():
                    continue

                row = json.loads(line)

                position = int(
                    row["source_position"]
                )

                assert position == expected_position, (
                    position,
                    expected_position,
                )

                assert len(row["text_sha256"]) == 64

                dataset_counts[
                    str(row["dataset"])
                ] += 1

                truncated_rows += int(
                    bool(row["truncated"])
                )

                output_row = {
                    "source_position": position,
                    "dataset": str(
                        row["dataset"]
                    ),
                    "id": str(row["id"]),
                    "text_sha256": str(
                        row["text_sha256"]
                    ),
                }

                for field in FIELDS:
                    indices = [
                        int(value)
                        for value in row[
                            f"{field}_indices"
                        ]
                    ]

                    probabilities = [
                        float(value)
                        for value in row[
                            f"{field}_probabilities"
                        ]
                    ]

                    distribution = [
                        float(value)
                        for value in row[
                            f"{field}_distribution"
                        ]
                    ]

                    assert len(indices) == TOP_K[field]
                    assert len(probabilities) == TOP_K[field]
                    assert len(distribution) == 100
                    assert all(
                        0 <= value < 100
                        for value in indices
                    )
                    assert all(
                        math.isfinite(value)
                        and value >= 0.0
                        for value in distribution
                    )

                    probability_sum = sum(
                        distribution
                    )

                    maximum_sum_error[field] = max(
                        maximum_sum_error[field],
                        abs(probability_sum - 1.0),
                    )

                    assert abs(
                        probability_sum - 1.0
                    ) <= 1.0e-5

                    for index, probability in zip(
                        indices,
                        probabilities,
                    ):
                        assert abs(
                            probability
                            - distribution[index]
                        ) <= 1.0e-5

                    top1_classes[field].add(
                        indices[0]
                    )
                    topk_classes[field].update(
                        indices
                    )

                    max_probability = float(
                        row[
                            f"{field}_max_probability"
                        ]
                    )
                    entropy = float(
                        row[f"{field}_entropy"]
                    )

                    assert math.isfinite(
                        max_probability
                    )
                    assert math.isfinite(entropy)

                    sum_max_probability[
                        field
                    ] += max_probability
                    sum_entropy[field] += entropy

                    output_row[
                        f"{field}_indices"
                    ] = indices
                    output_row[
                        f"{field}_probabilities"
                    ] = probabilities
                    output_row[
                        f"{field}_distribution"
                    ] = distribution
                    output_row[
                        f"{field}_max_probability"
                    ] = max_probability
                    output_row[
                        f"{field}_entropy"
                    ] = entropy

                parquet_rows.append(output_row)
                expected_position += 1

        assert len(parquet_rows) == int(
            manifest["rows"]
        )

        assert expected_position == int(
            manifest["end_position_exclusive"]
        )

        table = pa.Table.from_pylist(
            parquet_rows,
            schema=schema,
        )

        writer.write_table(table)

        if (
            (shard_index + 1) % 25 == 0
            or shard_index + 1 == len(shards)
        ):
            print(
                f"shards={shard_index + 1}/"
                f"{len(shards)} "
                f"rows={expected_position}/"
                f"{EXPECTED_ROWS}",
                flush=True,
            )

            write_json(
                status_path,
                {
                    "status": "RUNNING",
                    "stage": "merging",
                    "processed_rows": (
                        expected_position
                    ),
                    "requested_rows": EXPECTED_ROWS,
                    "completed_shards": (
                        shard_index + 1
                    ),
                    "total_shards": len(shards),
                    "input": str(shard_root),
                    "output": str(output_path),
                    "qwen_used": False,
                    "laya_model_used": False,
                    "laya_predictions_used": False,
                    "laya_scores_used": False,
                },
            )
finally:
    writer.close()

assert expected_position == EXPECTED_ROWS

temporary_path.replace(output_path)

parquet_file = pq.ParquetFile(
    output_path
)

assert parquet_file.metadata.num_rows == (
    EXPECTED_ROWS
)

assert parquet_file.schema_arrow == schema

elapsed = time.time() - started
output_sha256 = sha256_file(output_path)

report = {
    "status": "PASSED",
    "stage": "complete",
    "rows": EXPECTED_ROWS,
    "shards": len(shards),
    "schema_columns": schema.names,
    "schema_column_count": len(schema.names),
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "truncated_rows": truncated_rows,
    "truncated_rate": (
        truncated_rows / EXPECTED_ROWS
    ),
    "observed_top1_classes": {
        field: len(top1_classes[field])
        for field in FIELDS
    },
    "observed_any_topk_classes": {
        field: len(topk_classes[field])
        for field in FIELDS
    },
    "mean_maximum_probability": {
        field: (
            sum_max_probability[field]
            / EXPECTED_ROWS
        )
        for field in FIELDS
    },
    "mean_entropy": {
        field: (
            sum_entropy[field]
            / EXPECTED_ROWS
        )
        for field in FIELDS
    },
    "maximum_probability_sum_error": (
        maximum_sum_error
    ),
    "checkpoint_sha256": (
        EXPECTED_CHECKPOINT_SHA256
    ),
    "input": str(shard_root),
    "output": str(output_path),
    "output_bytes": output_path.stat().st_size,
    "output_sha256": output_sha256,
    "elapsed_seconds": elapsed,
    "qwen_used": False,
    "laya_model_used": False,
    "laya_predictions_used": False,
    "laya_scores_used": False,
}

write_json(report_path, report)

write_json(
    status_path,
    {
        **report,
        "report": str(report_path),
    },
)

print("=" * 78)
print("PPLX CLUSTER-PREDICTION MERGE AND AUDIT PASSED")
print("=" * 78)
print("Rows:", EXPECTED_ROWS)
print("Shards:", len(shards))
print("Columns:", len(schema.names))
print("Dataset counts:", dict(dataset_counts))
print("Truncated rows:", truncated_rows)
print("Output:", output_path)
print("SHA256:", output_sha256)
print("Report:", report_path)
print("Elapsed seconds:", elapsed)
print("LAYA model/predictions used: NO")
print("Qwen used: NO")
