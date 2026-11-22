#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

TOTAL_ROWS = 300932
PILOT_ROWS = 1000
SEED = 42

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

target_path = (
    root
    / "data"
    / "target_supervision"
    / "exact_laya_description_aware_v1"
    / "complete_target.json"
)

contract_path = (
    root
    / "configs"
    / "21_pplx_quality_scoring_contract.json"
)

output_path = (
    root
    / "data"
    / "quality_scoring"
    / "pplx_scoring_pilot_1000_adapter.jsonl"
)

positions_path = (
    root
    / "data"
    / "quality_scoring"
    / "pplx_scoring_pilot_1000_positions.json"
)

report_path = (
    root
    / "metadata"
    / "quality_scoring"
    / "23_pplx_scoring_pilot_adapter.json"
)

for path in (
    source_path,
    prediction_path,
    target_path,
    contract_path,
):
    assert path.is_file(), (
        f"Missing required input: {path}"
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


def selected_parquet_rows(
    path,
    columns,
    selected_positions,
):
    selected_set = set(
        selected_positions
    )

    results = {}
    offset = 0

    parquet_file = pq.ParquetFile(
        path
    )

    for batch in parquet_file.iter_batches(
        batch_size=4096,
        columns=columns,
    ):
        rows = batch.to_pylist()

        for local_index, row in enumerate(rows):
            position = offset + local_index

            if position in selected_set:
                results[position] = row

        offset += len(rows)

    assert offset == TOTAL_ROWS, (
        path,
        offset,
    )

    assert set(results) == selected_set

    return results


def extract_sample(messages):
    instruction = ""
    input_text = ""
    response = ""

    system_messages = [
        message
        for message in messages
        if message.get("role") == "system"
    ]

    if system_messages:
        instruction = str(
            system_messages[0].get(
                "content",
                "",
            )
        )

    if len(messages) <= 2:
        for message in messages:
            role = message.get("role")
            content = str(
                message.get(
                    "content",
                    "",
                )
            )

            if role == "user":
                input_text = content
            elif role == "assistant":
                response = content
    else:
        for message in messages[-2:]:
            role = message.get("role")
            content = str(
                message.get(
                    "content",
                    "",
                )
            )

            if role == "user":
                input_text = content
            elif role == "assistant":
                response = content

        history = []

        for message in messages[:-2]:
            role = message.get("role")

            if role == "system":
                continue

            content = str(
                message.get(
                    "content",
                    "",
                )
            ).strip()

            if content:
                history.append(
                    f"Previous "
                    f"{str(role).title()}: "
                    f"{content}"
                )

        if history:
            history_text = "\n".join(
                history
            )

            if instruction.strip():
                instruction = (
                    instruction
                    + "\n\nConversation History:\n"
                    + history_text
                )
            else:
                instruction = (
                    "Conversation History:\n"
                    + history_text
                )

    sections = []

    if instruction.strip():
        sections.append(
            "Instruction:\n"
            + instruction.strip()
        )

    sections.append(
        "Input:\n"
        + input_text.strip()
    )

    sections.append(
        "Response:\n"
        + response.strip()
    )

    return "\n\n".join(sections)


def cluster_descriptions(
    indices,
    field,
    catalog,
):
    descriptions = []

    for index in indices:
        cluster_id = (
            f"{field}_{int(index)}"
        )

        assert cluster_id in catalog, (
            f"Missing catalog item: "
            f"{cluster_id}"
        )

        item = catalog[cluster_id]

        title = str(
            item.get("title", "")
        ).strip()

        description = str(
            item.get(
                "refined_description",
                "",
            )
        ).strip()

        assert description

        if title:
            text = (
                f"{cluster_id} — "
                f"{title}: {description}"
            )
        else:
            text = (
                f"{cluster_id}: "
                f"{description}"
            )

        descriptions.append(
            "- " + text
        )

    return "\n".join(descriptions)


def build_state(
    source_row,
    prediction_row,
    catalog,
):
    sample_text = extract_sample(
        source_row["messages"]
    )

    return f"""Evaluate this instruction-response training example.

SAMPLE TO EVALUATE
{sample_text}

ASSIGNED TASK
{cluster_descriptions(
    prediction_row["Task_indices"],
    "Task",
    catalog,
)}

ASSIGNED TOPICS
{cluster_descriptions(
    prediction_row["Topic_indices"],
    "Topic",
    catalog,
)}

ASSIGNED STYLES
{cluster_descriptions(
    prediction_row["Style_indices"],
    "Style",
    catalog,
)}

ASSIGNED AUDIENCES
{cluster_descriptions(
    prediction_row["Audience_indices"],
    "Audience",
    catalog,
)}

Use all assigned semantic anchors as reference points when
judging the example, especially for the Overall Score.
"""



from itertools import zip_longest


FULL_OUTPUT_PATH = (
    root
    / "data"
    / "quality_scoring"
    / "pplx_scoring_full_300932_adapter.jsonl"
)

FULL_STATUS_PATH = (
    root
    / "data"
    / "quality_scoring"
    / "25_pplx_scoring_full_adapter_status.json"
)

FULL_REPORT_PATH = (
    root
    / "metadata"
    / "quality_scoring"
    / "25_pplx_scoring_full_adapter.json"
)


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


def parquet_rows(path, columns):
    parquet_file = pq.ParquetFile(path)

    position = 0

    for batch in parquet_file.iter_batches(
        batch_size=4096,
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


started = time.time()

FULL_OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

FULL_REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

source_parquet = pq.ParquetFile(
    source_path
)

prediction_parquet = pq.ParquetFile(
    prediction_path
)

assert source_parquet.metadata.num_rows == (
    TOTAL_ROWS
)

assert prediction_parquet.metadata.num_rows == (
    TOTAL_ROWS
)

with target_path.open(
    encoding="utf-8"
) as handle:
    target_data = json.load(handle)

catalog = {
    str(item["cluster_id"]): item
    for item in target_data[
        "complete_cluster_catalog"
    ]
}

assert len(catalog) == 400

with contract_path.open(
    encoding="utf-8"
) as handle:
    contract = json.load(handle)

assert contract[
    "runtime_policy"
]["pplx_only"] is True

assert contract[
    "score_policy"
]["selection_threshold"] == 5.0

source_columns = [
    "dataset",
    "id",
    "messages",
]

prediction_columns = [
    "source_position",
    "dataset",
    "id",
    "text_sha256",
]

for field in FIELDS:
    prediction_columns.append(
        f"{field}_indices"
    )

atomic_json(
    FULL_STATUS_PATH,
    {
        "status": "RUNNING",
        "stage": "building",
        "processed_rows": 0,
        "requested_rows": TOTAL_ROWS,
        "output": str(FULL_OUTPUT_PATH),
        "model": (
            "perplexity-ai/"
            "pplx-decider-v1.2"
        ),
        "laya_model_used": False,
        "laya_predictions_used": False,
        "laya_scores_used": False,
        "qwen_used": False,
    },
)

dataset_counts = Counter()

minimum_state_characters = None
maximum_state_characters = 0
total_state_characters = 0

temporary_path = (
    FULL_OUTPUT_PATH.with_suffix(
        ".jsonl.incomplete"
    )
)

source_iterator = parquet_rows(
    source_path,
    source_columns,
)

prediction_iterator = parquet_rows(
    prediction_path,
    prediction_columns,
)

sentinel = object()
processed_rows = 0

with temporary_path.open(
    "w",
    encoding="utf-8",
) as output:
    for source_item, prediction_item in zip_longest(
        source_iterator,
        prediction_iterator,
        fillvalue=sentinel,
    ):
        assert source_item is not sentinel
        assert prediction_item is not sentinel

        source_position, source = source_item
        prediction_position, prediction = (
            prediction_item
        )

        assert source_position == (
            prediction_position
        )

        assert int(
            prediction["source_position"]
        ) == source_position

        assert prediction["dataset"] == (
            source["dataset"]
        )

        assert prediction["id"] == (
            source["id"]
        )

        state = build_state(
            source,
            prediction,
            catalog,
        )

        assert state.strip()

        record = {
            # Kept for exact compatibility with the
            # validated scoring parser. For the full
            # dataset it equals source_position.
            "pilot_index": source_position,
            "scoring_index": source_position,
            "source_position": source_position,
            "dataset": str(
                source["dataset"]
            ),
            "id": str(source["id"]),
            "text_sha256": str(
                prediction["text_sha256"]
            ),
            "state": state,
        }

        for field in FIELDS:
            record[
                f"{field}_indices"
            ] = [
                int(value)
                for value in prediction[
                    f"{field}_indices"
                ]
            ]

        output.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

        dataset_counts[
            record["dataset"]
        ] += 1

        state_characters = len(state)

        if minimum_state_characters is None:
            minimum_state_characters = (
                state_characters
            )
        else:
            minimum_state_characters = min(
                minimum_state_characters,
                state_characters,
            )

        maximum_state_characters = max(
            maximum_state_characters,
            state_characters,
        )

        total_state_characters += (
            state_characters
        )

        processed_rows += 1

        if (
            processed_rows % 4096 == 0
            or processed_rows == TOTAL_ROWS
        ):
            elapsed = time.time() - started

            print(
                f"processed={processed_rows}/"
                f"{TOTAL_ROWS} "
                f"rate="
                f"{processed_rows / elapsed:.2f} "
                f"rows/s",
                flush=True,
            )

            atomic_json(
                FULL_STATUS_PATH,
                {
                    "status": "RUNNING",
                    "stage": "building",
                    "processed_rows":
                        processed_rows,
                    "requested_rows":
                        TOTAL_ROWS,
                    "elapsed_seconds":
                        elapsed,
                    "rows_per_second": (
                        processed_rows
                        / elapsed
                    ),
                    "output": str(
                        FULL_OUTPUT_PATH
                    ),
                    "laya_model_used": False,
                    "qwen_used": False,
                },
            )

assert processed_rows == TOTAL_ROWS

temporary_path.replace(
    FULL_OUTPUT_PATH
)

elapsed = time.time() - started
output_sha256 = sha256_file(
    FULL_OUTPUT_PATH
)

report = {
    "status": "PASSED",
    "stage": "complete",
    "rows": processed_rows,
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "minimum_state_characters":
        minimum_state_characters,
    "maximum_state_characters":
        maximum_state_characters,
    "mean_state_characters": (
        total_state_characters
        / processed_rows
    ),
    "source_parquet": str(
        source_path
    ),
    "prediction_parquet": str(
        prediction_path
    ),
    "contract": str(contract_path),
    "output": str(FULL_OUTPUT_PATH),
    "output_sha256": output_sha256,
    "elapsed_seconds": elapsed,
    "rows_per_second": (
        processed_rows / elapsed
    ),
    "prompt_construction": (
        "Exact helper functions preserved from "
        "23_build_pplx_scoring_pilot_adapter.py"
    ),
    "sample_selection": (
        "All source positions 0 through 300931 "
        "in original order"
    ),
    "model": (
        "perplexity-ai/"
        "pplx-decider-v1.2"
    ),
    "laya_model_used": False,
    "laya_predictions_used": False,
    "laya_scores_used": False,
    "qwen_used": False,
    "external_teacher_scores_used": False,
}

atomic_json(
    FULL_REPORT_PATH,
    report,
)

atomic_json(
    FULL_STATUS_PATH,
    {
        **report,
        "report": str(
            FULL_REPORT_PATH
        ),
    },
)

print("=" * 78)
print("PPLX FULL SCORING ADAPTER PASSED")
print("=" * 78)
print("Rows:", processed_rows)
print(
    "Dataset counts:",
    dict(dataset_counts),
)
print(
    "State character range:",
    minimum_state_characters,
    "-",
    maximum_state_characters,
)
print("Output:", FULL_OUTPUT_PATH)
print("SHA256:", output_sha256)
print("Report:", FULL_REPORT_PATH)
print("Elapsed seconds:", elapsed)
print("LAYA/Qwen runtime use: NO")
