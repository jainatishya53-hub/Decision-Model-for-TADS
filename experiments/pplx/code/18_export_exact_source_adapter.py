#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import time

import pyarrow.parquet as pq


EXPECTED_ROWS = 300932
READ_BATCH_SIZE = 4096


root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

runtime_root = Path(
    os.environ["PPLX_RUNTIME_ROOT"]
).resolve()

paper_work = Path(
    os.environ["PAPER_WORK"]
).resolve()

source_path = (
    paper_work
    / "data"
    / "train-00000-of-00001.parquet"
)

output_path = (
    runtime_root
    / "source_adapter"
    / "source_300932_exact_laya_text.jsonl"
)

status_path = (
    root
    / "metadata"
    / "source_adapter"
    / "18_source_adapter_status.json"
)

report_path = (
    root
    / "metadata"
    / "source_adapter"
    / "18_source_adapter_report.json"
)


def atomic_json(path, value):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    temporary.replace(path)


def flatten_messages(messages):
    # Exact function used by the LAYA classifier.
    parts = []

    for message in messages:
        role = str(
            message.get("role", "")
        ).strip()

        content = str(
            message.get("content", "")
        ).strip()

        if not role and not content:
            continue

        parts.append(
            f"{role}: {content}"
        )

    return "\n".join(parts)


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


started = time.time()

assert source_path.is_file(), (
    f"Missing source Parquet: {source_path}"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

atomic_json(
    status_path,
    {
        "status": "STARTING",
        "stage": "loading",
        "source": str(source_path),
        "output": str(output_path),
        "expected_rows": EXPECTED_ROWS,
    },
)

parquet = pq.ParquetFile(
    source_path
)

assert parquet.metadata.num_rows == (
    EXPECTED_ROWS
)

temporary = output_path.with_suffix(
    output_path.suffix + ".tmp"
)

digest = hashlib.sha256()
dataset_counts = Counter()

processed = 0
empty_texts = 0
maximum_text_characters = 0
first_identity = None
last_identity = None

with temporary.open(
    "wb"
) as output_handle:
    for batch in parquet.iter_batches(
        batch_size=READ_BATCH_SIZE,
        columns=[
            "dataset",
            "id",
            "messages",
        ],
    ):
        rows = batch.to_pylist()

        for source_row in rows:
            dataset = str(
                source_row["dataset"]
            )

            source_id = str(
                source_row["id"]
            )

            messages = (
                source_row["messages"]
                or []
            )

            text = flatten_messages(
                messages
            )

            if not text:
                empty_texts += 1

            maximum_text_characters = max(
                maximum_text_characters,
                len(text),
            )

            record = {
                "source_position":
                    processed,
                "dataset":
                    dataset,
                "id":
                    source_id,
                "text":
                    text,
                "text_sha256":
                    sha256_text(text),
            }

            identity = {
                "source_position":
                    processed,
                "dataset":
                    dataset,
                "id":
                    source_id,
                "text_sha256":
                    record["text_sha256"],
            }

            if first_identity is None:
                first_identity = identity

            last_identity = identity

            encoded_line = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")

            output_handle.write(
                encoded_line
            )

            digest.update(
                encoded_line
            )

            dataset_counts[
                dataset
            ] += 1

            processed += 1

        print(
            f"processed={processed}/"
            f"{EXPECTED_ROWS}",
            flush=True,
        )

        atomic_json(
            status_path,
            {
                "status": "RUNNING",
                "stage": "exporting",
                "processed_rows":
                    processed,
                "expected_rows":
                    EXPECTED_ROWS,
                "elapsed_seconds":
                    time.time() - started,
                "output":
                    str(output_path),
            },
        )

assert processed == EXPECTED_ROWS

temporary.replace(
    output_path
)

report = {
    "status": "PASSED",
    "stage": "complete",
    "source": str(source_path),
    "output": str(output_path),
    "rows": processed,
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "empty_texts": empty_texts,
    "maximum_text_characters":
        maximum_text_characters,
    "first_identity":
        first_identity,
    "last_identity":
        last_identity,
    "adapter_sha256":
        digest.hexdigest(),
    "text_policy": (
        "Exact LAYA flatten_messages: "
        "one stripped 'role: content' "
        "line per nonempty message"
    ),
    "order_policy":
        "Original Parquet row order",
    "elapsed_seconds":
        time.time() - started,
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
print("EXACT SOURCE ADAPTER BUILD PASSED")
print("=" * 78)
print("Rows:", processed)
print(
    "Dataset counts:",
    dict(sorted(dataset_counts.items())),
)
print("Empty texts:", empty_texts)
print(
    "Maximum text characters:",
    maximum_text_characters,
)
print(
    "Adapter SHA256:",
    digest.hexdigest(),
)
print("Output:", output_path)
print("Report:", report_path)
print(
    "Elapsed seconds:",
    report["elapsed_seconds"],
)
