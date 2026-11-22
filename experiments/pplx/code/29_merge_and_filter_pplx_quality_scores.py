#!/usr/bin/env python3

from collections import Counter
from itertools import zip_longest
from pathlib import Path
import hashlib
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq


TOTAL_ROWS = 300932
EXPECTED_ELIGIBLE_ROWS = 75197

METRICS = (
    "completeness",
    "information_richness",
    "rarity",
    "complexity",
    "overall_score",
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(
        path.suffix + ".incomplete"
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


def parquet_rows(
    path,
    columns,
    batch_size=1000,
):
    parquet_file = pq.ParquetFile(path)
    position = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        for row in batch.to_pylist():
            yield position, row
            position += 1

    assert position == TOTAL_ROWS, (
        path,
        position,
        TOTAL_ROWS,
    )


def score_rows(shard_paths):
    expected_position = 0

    for shard_path in shard_paths:
        with shard_path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)

                position = int(
                    row["source_position"]
                )

                assert position == (
                    expected_position
                ), (
                    shard_path,
                    position,
                    expected_position,
                )

                yield position, row
                expected_position += 1

    assert expected_position == TOTAL_ROWS


root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

paper_work = Path(
    os.environ["PAPER_WORK"]
).resolve()

source_path = (
    paper_work
    / "data"
    / "train-00000-of-00001.parquet"
)

prediction_path = (
    root
    / "data"
    / "source_predictions"
    / "pplx_epoch8_source_predictions_all_300932.parquet"
)

scoring_root = (
    root
    / "data"
    / "quality_scoring"
    / "base_pplx_full_300932_chunk16_optimized"
    / "shards"
)

scoring_report_path = (
    root
    / "metadata"
    / "quality_scoring"
    / "base_pplx_full_300932_chunk16_optimized.json"
)

full_output_path = (
    root
    / "data"
    / "quality_scoring"
    / "merged"
    / "pplx_scored_all_300932.parquet"
)

filtered_output_path = (
    root
    / "data"
    / "quality_scoring"
    / "filtered"
    / "pplx_scores_all_five_gte_5_75197.parquet"
)

status_path = (
    root
    / "data"
    / "quality_scoring"
    / "29_pplx_score_merge_filter_status.json"
)

report_path = (
    root
    / "metadata"
    / "quality_scoring"
    / "29_pplx_score_merge_filter.json"
)

for required_path in (
    source_path,
    prediction_path,
    scoring_report_path,
):
    assert required_path.is_file(), (
        required_path
    )

shard_paths = sorted(
    scoring_root.glob("shard_*.jsonl")
)

assert len(shard_paths) == 301
assert shard_paths[0].name == (
    "shard_000000_000999.jsonl"
)
assert shard_paths[-1].name == (
    "shard_300000_300931.jsonl"
)

with scoring_report_path.open(
    encoding="utf-8"
) as handle:
    scoring_report = json.load(handle)

assert scoring_report["status"] == "PASSED"
assert scoring_report["rows"] == TOTAL_ROWS
assert scoring_report["shards"] == 301
assert scoring_report[
    "native_readout_unchanged"
] is True
assert scoring_report[
    "laya_model_used"
] is False
assert scoring_report["qwen_used"] is False

expected_selected = scoring_report[
    "summary"
]["all_five_gte_5"]["count"]

assert expected_selected == (
    EXPECTED_ELIGIBLE_ROWS
)

source_file = pq.ParquetFile(
    source_path
)

prediction_file = pq.ParquetFile(
    prediction_path
)

assert source_file.metadata.num_rows == (
    TOTAL_ROWS
)

assert prediction_file.metadata.num_rows == (
    TOTAL_ROWS
)

source_columns = [
    "dataset",
    "id",
    "messages",
]

prediction_columns = (
    prediction_file.schema_arrow.names
)

source_iterator = parquet_rows(
    source_path,
    source_columns,
)

prediction_iterator = parquet_rows(
    prediction_path,
    prediction_columns,
)

scoring_iterator = score_rows(
    shard_paths
)

output_fields = [
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
    source_file.schema_arrow.field(
        "messages"
    ),
    pa.field(
        "text_sha256",
        pa.string(),
        nullable=False,
    ),
]

duplicate_prediction_columns = {
    "source_position",
    "dataset",
    "id",
    "text_sha256",
}

for field in (
    prediction_file.schema_arrow
):
    if field.name not in (
        duplicate_prediction_columns
    ):
        output_fields.append(field)

output_fields.extend(
    [
        pa.field(
            "scoring_index",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "truncated",
            pa.bool_(),
            nullable=False,
        ),
        pa.field(
            "original_sequence_length",
            pa.int64(),
            nullable=False,
        ),
        pa.field(
            "sequence_length",
            pa.int64(),
            nullable=False,
        ),
    ]
)

for metric in METRICS:
    output_fields.append(
        pa.field(
            metric,
            pa.float64(),
            nullable=False,
        )
    )

for threshold in (5, 6, 7):
    output_fields.append(
        pa.field(
            f"all_five_gte_{threshold}",
            pa.bool_(),
            nullable=False,
        )
    )

output_fields.extend(
    [
        pa.field(
            "quality_scoring_model",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "quality_scoring_arm",
            pa.string(),
            nullable=False,
        ),
        pa.field(
            "optimized_kernels_used",
            pa.bool_(),
            nullable=False,
        ),
    ]
)

output_schema = pa.schema(
    output_fields
)

full_temporary_path = (
    full_output_path.with_suffix(
        ".parquet.incomplete"
    )
)

filtered_temporary_path = (
    filtered_output_path.with_suffix(
        ".parquet.incomplete"
    )
)

started = time.time()

atomic_json(
    status_path,
    {
        "status": "RUNNING",
        "stage": "merging",
        "processed_rows": 0,
        "eligible_rows": 0,
        "requested_rows": TOTAL_ROWS,
    },
)

full_writer = pq.ParquetWriter(
    full_temporary_path,
    output_schema,
    compression="zstd",
    use_dictionary=True,
)

filtered_writer = pq.ParquetWriter(
    filtered_temporary_path,
    output_schema,
    compression="zstd",
    use_dictionary=True,
)

full_buffer = []
filtered_buffer = []

dataset_counts = Counter()
eligible_dataset_counts = Counter()

processed_rows = 0
eligible_rows = 0
truncated_rows = 0
eligible_truncated_rows = 0

score_sums = {
    metric: 0.0
    for metric in METRICS
}

eligible_score_sums = {
    metric: 0.0
    for metric in METRICS
}

sentinel = object()

try:
    combined_iterator = zip_longest(
        source_iterator,
        prediction_iterator,
        scoring_iterator,
        fillvalue=sentinel,
    )

    for (
        source_item,
        prediction_item,
        scoring_item,
    ) in combined_iterator:
        assert source_item is not sentinel
        assert prediction_item is not sentinel
        assert scoring_item is not sentinel

        source_position, source = (
            source_item
        )
        prediction_position, prediction = (
            prediction_item
        )
        scoring_position, scoring = (
            scoring_item
        )

        assert (
            source_position
            == prediction_position
            == scoring_position
            == processed_rows
        )

        assert int(
            prediction["source_position"]
        ) == source_position

        assert int(
            scoring["source_position"]
        ) == source_position

        assert source["dataset"] == (
            prediction["dataset"]
        )

        assert source["dataset"] == (
            scoring["dataset"]
        )

        assert source["id"] == prediction["id"]
        assert source["id"] == scoring["id"]

        assert prediction[
            "text_sha256"
        ] == scoring["text_sha256"]

        row = {
            "source_position":
                source_position,
            "dataset": str(
                source["dataset"]
            ),
            "id": str(source["id"]),
            "messages": source["messages"],
            "text_sha256": str(
                prediction["text_sha256"]
            ),
        }

        for column in prediction_columns:
            if column not in (
                duplicate_prediction_columns
            ):
                row[column] = prediction[
                    column
                ]

        row["scoring_index"] = int(
            scoring["scoring_index"]
        )

        row["truncated"] = bool(
            scoring["truncated"]
        )

        row[
            "original_sequence_length"
        ] = int(
            scoring[
                "original_sequence_length"
            ]
        )

        row["sequence_length"] = int(
            scoring["sequence_length"]
        )

        scores = []

        for metric in METRICS:
            value = float(
                scoring[metric][
                    "score_1_to_10"
                ]
            )

            assert 1.0 <= value <= 10.0

            row[metric] = value
            scores.append(value)
            score_sums[metric] += value

        for threshold in (5, 6, 7):
            expected_flag = all(
                value >= threshold
                for value in scores
            )

            score_flag = bool(
                scoring[
                    f"all_five_gte_"
                    f"{threshold}"
                ]
            )

            assert score_flag == (
                expected_flag
            )

            row[
                f"all_five_gte_{threshold}"
            ] = score_flag

        row["quality_scoring_model"] = (
            "perplexity-ai/"
            "pplx-decider-v1.2"
        )

        row["quality_scoring_arm"] = (
            "base_native_pplx"
        )

        row[
            "optimized_kernels_used"
        ] = True

        full_buffer.append(row)

        dataset_counts[
            row["dataset"]
        ] += 1

        truncated_rows += int(
            row["truncated"]
        )

        if row["all_five_gte_5"]:
            filtered_buffer.append(row)
            eligible_rows += 1

            eligible_dataset_counts[
                row["dataset"]
            ] += 1

            eligible_truncated_rows += int(
                row["truncated"]
            )

            for metric in METRICS:
                eligible_score_sums[
                    metric
                ] += row[metric]

        processed_rows += 1

        if (
            len(full_buffer) >= 1000
            or processed_rows == TOTAL_ROWS
        ):
            full_table = pa.Table.from_pylist(
                full_buffer,
                schema=output_schema,
            )

            full_writer.write_table(
                full_table,
                row_group_size=1000,
            )

            full_buffer.clear()

            if filtered_buffer:
                filtered_table = (
                    pa.Table.from_pylist(
                        filtered_buffer,
                        schema=output_schema,
                    )
                )

                filtered_writer.write_table(
                    filtered_table,
                    row_group_size=1000,
                )

                filtered_buffer.clear()

        if (
            processed_rows % 10000 == 0
            or processed_rows == TOTAL_ROWS
        ):
            elapsed = time.time() - started

            print(
                f"processed={processed_rows}/"
                f"{TOTAL_ROWS} "
                f"eligible={eligible_rows} "
                f"rate="
                f"{processed_rows / elapsed:.2f} "
                f"rows/s",
                flush=True,
            )

            atomic_json(
                status_path,
                {
                    "status": "RUNNING",
                    "stage": "merging",
                    "processed_rows":
                        processed_rows,
                    "eligible_rows":
                        eligible_rows,
                    "requested_rows":
                        TOTAL_ROWS,
                    "elapsed_seconds":
                        elapsed,
                },
            )
finally:
    full_writer.close()
    filtered_writer.close()

assert processed_rows == TOTAL_ROWS
assert eligible_rows == (
    EXPECTED_ELIGIBLE_ROWS
)
assert truncated_rows == 20161

full_temporary_path.replace(
    full_output_path
)

filtered_temporary_path.replace(
    filtered_output_path
)

full_rows = pq.ParquetFile(
    full_output_path
).metadata.num_rows

filtered_rows = pq.ParquetFile(
    filtered_output_path
).metadata.num_rows

assert full_rows == TOTAL_ROWS
assert filtered_rows == (
    EXPECTED_ELIGIBLE_ROWS
)

elapsed = time.time() - started

report = {
    "status": "PASSED",
    "stage": "complete",
    "rows": processed_rows,
    "eligible_rows": eligible_rows,
    "selection_rule": (
        "all five PPLX scores >= 5"
    ),
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "eligible_dataset_counts": dict(
        sorted(
            eligible_dataset_counts.items()
        )
    ),
    "truncated_rows": truncated_rows,
    "eligible_truncated_rows":
        eligible_truncated_rows,
    "score_means": {
        metric: (
            score_sums[metric]
            / processed_rows
        )
        for metric in METRICS
    },
    "eligible_score_means": {
        metric: (
            eligible_score_sums[metric]
            / eligible_rows
        )
        for metric in METRICS
    },
    "full_output": str(
        full_output_path
    ),
    "full_output_sha256": sha256_file(
        full_output_path
    ),
    "filtered_output": str(
        filtered_output_path
    ),
    "filtered_output_sha256":
        sha256_file(
            filtered_output_path
        ),
    "source_parquet": str(source_path),
    "prediction_parquet": str(
        prediction_path
    ),
    "scoring_report": str(
        scoring_report_path
    ),
    "omitted_source_score_columns": [
        "gpt_scores",
        "llama_scores",
        "mistral_scores",
    ],
    "quality_scoring_model": (
        "perplexity-ai/"
        "pplx-decider-v1.2"
    ),
    "quality_scoring_arm": (
        "base_native_pplx"
    ),
    "optimized_kernels_used": True,
    "laya_model_used": False,
    "laya_predictions_used": False,
    "laya_scores_used": False,
    "qwen_used": False,
    "external_teacher_scores_used":
        False,
    "elapsed_seconds": elapsed,
    "rows_per_second": (
        processed_rows / elapsed
    ),
}

atomic_json(
    report_path,
    report,
)

atomic_json(
    status_path,
    {
        **report,
        "report": str(report_path),
    },
)

print("=" * 78)
print("PPLX SCORE MERGE AND FILTER PASSED")
print("=" * 78)
print("Full rows:", processed_rows)
print("Eligible rows:", eligible_rows)
print(
    "Eligible dataset counts:",
    dict(eligible_dataset_counts),
)
print("Truncated rows:", truncated_rows)
print(
    "Eligible truncated rows:",
    eligible_truncated_rows,
)
print("Full output:", full_output_path)
print("Filtered output:", filtered_output_path)
print("Report:", report_path)
print("LAYA/Qwen runtime use: NO")
