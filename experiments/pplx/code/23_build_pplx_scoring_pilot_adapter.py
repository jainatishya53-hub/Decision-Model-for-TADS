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


started = time.time()

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

rng = np.random.default_rng(SEED)

positions = sorted(
    int(value)
    for value in rng.choice(
        TOTAL_ROWS,
        size=PILOT_ROWS,
        replace=False,
    )
)

source_rows = selected_parquet_rows(
    source_path,
    columns=[
        "dataset",
        "id",
        "messages",
    ],
    selected_positions=positions,
)

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

prediction_rows = selected_parquet_rows(
    prediction_path,
    columns=prediction_columns,
    selected_positions=positions,
)

dataset_counts = Counter()
state_lengths = []
temporary_path = output_path.with_suffix(
    ".jsonl.incomplete"
)

with temporary_path.open(
    "w",
    encoding="utf-8",
) as output:
    for output_index, position in enumerate(
        positions
    ):
        source = source_rows[position]
        prediction = prediction_rows[
            position
        ]

        assert int(
            prediction["source_position"]
        ) == position

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
            "pilot_index": output_index,
            "source_position": position,
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

        state_lengths.append(
            len(state)
        )

temporary_path.replace(
    output_path
)

positions_path.write_text(
    json.dumps(
        {
            "seed": SEED,
            "positions": positions,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

elapsed = time.time() - started

report = {
    "status": "PASSED",
    "rows": PILOT_ROWS,
    "seed": SEED,
    "dataset_counts": dict(
        sorted(dataset_counts.items())
    ),
    "minimum_state_characters": min(
        state_lengths
    ),
    "maximum_state_characters": max(
        state_lengths
    ),
    "mean_state_characters": (
        sum(state_lengths)
        / len(state_lengths)
    ),
    "source_parquet": str(source_path),
    "prediction_parquet": str(
        prediction_path
    ),
    "contract": str(contract_path),
    "output": str(output_path),
    "output_sha256": sha256_file(
        output_path
    ),
    "positions": str(positions_path),
    "elapsed_seconds": elapsed,
    "model": (
        "perplexity-ai/pplx-decider-v1.2"
    ),
    "laya_model_used": False,
    "laya_predictions_used": False,
    "laya_scores_used": False,
    "qwen_used": False,
}

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("=" * 78)
print("PPLX-ONLY SCORING PILOT ADAPTER PASSED")
print("=" * 78)
print("Rows:", PILOT_ROWS)
print("Seed:", SEED)
print("Dataset counts:", dict(dataset_counts))
print(
    "State character range:",
    min(state_lengths),
    "-",
    max(state_lengths),
)
print("Output:", output_path)
print("SHA256:", report["output_sha256"])
print("Report:", report_path)
print("Elapsed seconds:", elapsed)
print("LAYA/Qwen runtime use: NO")
