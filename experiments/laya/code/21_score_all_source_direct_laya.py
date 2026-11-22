#!/usr/bin/env python3

from pathlib import Path
import ast
import json
import os
import time
import traceback

import laya
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


FIELDS = ("Task", "Topic", "Style", "Audience")

METRICS = (
    "Completeness",
    "Information Richness",
    "Rarity",
    "Complexity",
    "Overall Score",
)

METRIC_COLUMNS = {
    "Completeness": "completeness",
    "Information Richness": "information_richness",
    "Rarity": "rarity",
    "Complexity": "complexity",
    "Overall Score": "overall_score",
}

TOTAL_EXPECTED_ROWS = 300932
INFERENCE_BATCH_SIZE = 16
SHARD_SIZE = 2000
MAX_LENGTH = 2048
HEAD_MAX_LENGTH = 512


restart_root = Path(
    os.environ.get(
        "BASE_RESTART_ROOT",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "laya_tads_anchor_classifier/laya_base_restart",
    )
).resolve()

exp_root = Path(
    os.environ.get(
        "EXP_ROOT",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "laya_tads_anchor_classifier",
    )
).resolve()

paper_work = Path(
    os.environ.get(
        "PAPER_WORK",
        "/home/gpuuser2/gpuuser2_a/atishya/"
        "tads_paper_faithful",
    )
).resolve()


source_path = (
    paper_work
    / "data"
    / "train-00000-of-00001.parquet"
)

prediction_path = (
    restart_root
    / "data"
    / "source_predictions"
    / "source_predictions_all_300932.parquet"
)

catalog_path = (
    restart_root
    / "data"
    / "complete_target_with_cluster_descriptions.json"
)

rubric_program = (
    restart_root
    / "code"
    / "11_smoke_base_vs_hybrid_scoring.py"
)

base_checkpoint = (
    exp_root
    / "models"
    / "laya_base_english"
)

output_directory = (
    restart_root
    / "data"
    / "quality_scoring"
    / "direct_laya_full"
)

shard_directory = output_directory / "shards"

output_path = (
    output_directory
    / "direct_laya_scores_all_300932.parquet"
)

status_path = (
    output_directory
    / "21_direct_laya_full_status.json"
)

report_path = (
    restart_root
    / "metadata"
    / "quality_scoring"
    / "21_direct_laya_full_report.json"
)


output_directory.mkdir(parents=True, exist_ok=True)
shard_directory.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)


def write_json_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


def load_questions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue

        if not any(
            isinstance(target, ast.Name)
            and target.id == "QUESTIONS"
            for target in targets
        ):
            continue

        questions = ast.literal_eval(value_node)

        if set(questions) != set(METRICS):
            continue

        for metric in METRICS:
            question = questions[metric]

            assert question["type"] == "score"
            assert isinstance(question["criteria"], list)
            assert len(question["criteria"]) == 10

        return questions

    raise AssertionError(
        "Validated five-question Laya rubric was not found."
    )


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
            system_messages[0].get("content", "")
        )

    if len(messages) <= 2:
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))

            if role == "user":
                input_text = content
            elif role == "assistant":
                response = content
    else:
        for message in messages[-2:]:
            role = message.get("role")
            content = str(message.get("content", ""))

            if role == "user":
                input_text = content
            elif role == "assistant":
                response = content

        history = []

        for message in messages[:-2]:
            role = message.get("role")

            if role == "system":
                continue

            content = str(message.get("content", "")).strip()

            if content:
                history.append(
                    f"Previous {str(role).title()}: {content}"
                )

        if history:
            history_text = "\n".join(history)

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
            "Instruction:\n" + instruction.strip()
        )

    sections.append(
        "Input:\n" + input_text.strip()
    )

    sections.append(
        "Response:\n" + response.strip()
    )

    return "\n\n".join(sections)


def cluster_descriptions(indices, field, catalog):
    descriptions = []

    for index in indices:
        cluster_id = f"{field}_{int(index)}"

        assert cluster_id in catalog, (
            f"Missing description for {cluster_id}"
        )

        item = catalog[cluster_id]

        title = str(
            item.get("title", "")
        ).strip()

        description = str(
            item.get("refined_description", "")
        ).strip()

        assert description, (
            f"Empty description for {cluster_id}"
        )

        if title:
            text = (
                f"{cluster_id} — {title}: {description}"
            )
        else:
            text = (
                f"{cluster_id}: {description}"
            )

        descriptions.append(text)

    return "\n".join(
        f"- {description}"
        for description in descriptions
    )


def build_state(source_row, prediction_row, catalog):
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


def read_probability(probabilities, index):
    if str(index) in probabilities:
        return float(probabilities[str(index)])

    if index in probabilities:
        return float(probabilities[index])

    raise AssertionError(
        f"Missing score probability for level {index}"
    )


def parse_answer(result, metric):
    assert "answers" in result
    assert metric in result["answers"]

    answer = result["answers"][metric]

    assert answer["type"] == "score"

    probabilities = np.asarray(
        [
            read_probability(
                answer["probabilities"],
                index,
            )
            for index in range(10)
        ],
        dtype=np.float64,
    )

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)

    probability_sum = float(probabilities.sum())

    assert abs(probability_sum - 1.0) <= 0.01

    raw_score = float(answer["score"])

    assert 0.0 <= raw_score <= 9.0

    expected_score = float(
        np.dot(
            probabilities,
            np.arange(10, dtype=np.float64),
        )
    )

    assert abs(expected_score - raw_score) <= 0.02

    return {
        "score_1_to_10": raw_score + 1.0,
        "argmax_level_1_to_10": (
            int(np.argmax(probabilities)) + 1
        ),
        "answer_confidence": float(
            answer["answer_confidence"]
        ),
        "confidence": float(
            answer["confidence"]
        ),
        "probabilities": probabilities.tolist(),
    }


def verify_existing_shard(path, start, end):
    if not path.is_file():
        return False

    table = pq.read_table(
        path,
        columns=["source_position"],
    )

    expected_rows = end - start

    if len(table) != expected_rows:
        return False

    positions = table[
        "source_position"
    ].to_numpy(zero_copy_only=False)

    return np.array_equal(
        positions,
        np.arange(start, end),
    )


def merge_shards(shard_paths, destination):
    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    writer = None

    try:
        for shard_path in shard_paths:
            table = pq.read_table(shard_path)

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                )

            assert table.schema == writer.schema
            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    assert temporary.is_file()
    os.replace(temporary, destination)


start_time = time.time()

try:
    write_json_atomic(
        status_path,
        {
            "status": "STARTING",
            "stage": "loading",
            "processed_rows": 0,
            "requested_rows": TOTAL_EXPECTED_ROWS,
            "qwen_scores_used": False,
        },
    )

    questions = load_questions(rubric_program)

    complete_target = json.loads(
        catalog_path.read_text(encoding="utf-8")
    )

    catalog = {
        item["cluster_id"]: item
        for item in complete_target[
            "complete_cluster_catalog"
        ]
    }

    assert len(catalog) == 400

    print("Loading source dataset...")

    source_table = pq.read_table(
        source_path,
        columns=[
            "dataset",
            "id",
            "messages",
        ],
    ).combine_chunks()

    print("Loading fine-tuned Laya anchor predictions...")

    prediction_table = pq.read_table(
        prediction_path,
        columns=[
            "source_position",
            "dataset",
            "id",
            "text_sha256",
            "Task_indices",
            "Topic_indices",
            "Style_indices",
            "Audience_indices",
        ],
    ).combine_chunks()

    total_rows = len(source_table)

    assert total_rows == TOTAL_EXPECTED_ROWS
    assert len(prediction_table) == total_rows

    positions = prediction_table[
        "source_position"
    ].to_numpy(zero_copy_only=False)

    assert np.array_equal(
        positions,
        np.arange(total_rows),
    )

    print("Loading base Laya native decision model...")

    agent = laya.load(
        str(base_checkpoint),
        device="cuda",
        fast=False,
    )

    print("Base checkpoint:", base_checkpoint)
    print("Rows:", total_rows)
    print("Inference batch size:", INFERENCE_BATCH_SIZE)
    print("Shard size:", SHARD_SIZE)
    print("Maximum length:", MAX_LENGTH)
    print("Qwen scores used: NO")

    shard_paths = []
    processed_rows = 0

    for start in range(0, total_rows, SHARD_SIZE):
        end = min(start + SHARD_SIZE, total_rows)

        shard_path = (
            shard_directory
            / f"part_{start:06d}_{end:06d}.parquet"
        )

        shard_paths.append(shard_path)

        if verify_existing_shard(
            shard_path,
            start,
            end,
        ):
            processed_rows = end

            print(
                f"reused={end}/{total_rows} "
                f"elapsed={time.time() - start_time:.1f}s"
            )

            continue

        states = []
        identities = []

        for position in range(start, end):
            source_row = {
                "dataset": source_table[
                    "dataset"
                ][position].as_py(),
                "id": source_table[
                    "id"
                ][position].as_py(),
                "messages": source_table[
                    "messages"
                ][position].as_py(),
            }

            prediction_row = {
                "source_position": prediction_table[
                    "source_position"
                ][position].as_py(),
                "dataset": prediction_table[
                    "dataset"
                ][position].as_py(),
                "id": prediction_table[
                    "id"
                ][position].as_py(),
                "text_sha256": prediction_table[
                    "text_sha256"
                ][position].as_py(),
                "Task_indices": prediction_table[
                    "Task_indices"
                ][position].as_py(),
                "Topic_indices": prediction_table[
                    "Topic_indices"
                ][position].as_py(),
                "Style_indices": prediction_table[
                    "Style_indices"
                ][position].as_py(),
                "Audience_indices": prediction_table[
                    "Audience_indices"
                ][position].as_py(),
            }

            assert (
                source_row["dataset"]
                == prediction_row["dataset"]
            )

            assert (
                source_row["id"]
                == prediction_row["id"]
            )

            states.append(
                build_state(
                    source_row,
                    prediction_row,
                    catalog,
                )
            )

            identities.append(prediction_row)

        outputs = agent.predict_batch(
            states,
            questions,
            batch_size=INFERENCE_BATCH_SIZE,
            lang="en",
            max_len=MAX_LENGTH,
            head_max_len=HEAD_MAX_LENGTH,
            sort_by_length=True,
        )

        assert len(outputs) == len(states)

        output_rows = []

        for offset, result in enumerate(outputs):
            identity = identities[offset]
            usage = result.get("usage", {})

            row = {
                "source_position": int(
                    identity["source_position"]
                ),
                "dataset": identity["dataset"],
                "id": identity["id"],
                "text_sha256": identity["text_sha256"],
                "Task_indices": identity[
                    "Task_indices"
                ],
                "Topic_indices": identity[
                    "Topic_indices"
                ],
                "Style_indices": identity[
                    "Style_indices"
                ],
                "Audience_indices": identity[
                    "Audience_indices"
                ],
                "truncated": bool(
                    usage.get("truncated", False)
                ),
                "state_tokens": int(
                    usage.get("state_tokens", -1)
                ),
            }

            expected_scores = []
            argmax_scores = []

            for metric in METRICS:
                parsed = parse_answer(
                    result,
                    metric,
                )

                prefix = METRIC_COLUMNS[metric]

                row[
                    f"{prefix}_score_1_to_10"
                ] = parsed["score_1_to_10"]

                row[
                    f"{prefix}_argmax_1_to_10"
                ] = parsed[
                    "argmax_level_1_to_10"
                ]

                row[
                    f"{prefix}_answer_confidence"
                ] = parsed["answer_confidence"]

                row[
                    f"{prefix}_confidence"
                ] = parsed["confidence"]

                row[
                    f"{prefix}_probabilities"
                ] = parsed["probabilities"]

                expected_scores.append(
                    parsed["score_1_to_10"]
                )

                argmax_scores.append(
                    parsed[
                        "argmax_level_1_to_10"
                    ]
                )

            row["all_five_expected_gte_5"] = bool(
                all(score >= 5.0 for score in expected_scores)
            )

            row["all_five_argmax_gte_5"] = bool(
                all(score >= 5 for score in argmax_scores)
            )

            output_rows.append(row)

        shard_table = pa.Table.from_pylist(output_rows)

        temporary = shard_path.with_suffix(
            shard_path.suffix + ".tmp"
        )

        pq.write_table(
            shard_table,
            temporary,
            compression="zstd",
        )

        os.replace(temporary, shard_path)

        processed_rows = end
        elapsed = time.time() - start_time
        rate = processed_rows / max(elapsed, 1e-9)

        write_json_atomic(
            status_path,
            {
                "status": "RUNNING",
                "stage": "direct_laya_scoring",
                "processed_rows": processed_rows,
                "requested_rows": total_rows,
                "rows_per_second": rate,
                "elapsed_seconds": elapsed,
                "qwen_scores_used": False,
                "last_shard": str(shard_path),
            },
        )

        print(
            f"processed={processed_rows}/{total_rows} "
            f"rate={rate:.2f} rows/s"
        )

    print("Merging completed shards...")

    merge_shards(
        shard_paths,
        output_path,
    )

    score_columns = [
        f"{METRIC_COLUMNS[metric]}_score_1_to_10"
        for metric in METRICS
    ]

    score_table = pq.read_table(
        output_path,
        columns=score_columns,
    )

    score_matrix = np.column_stack(
        [
            score_table[column].to_numpy(
                zero_copy_only=False
            )
            for column in score_columns
        ]
    )

    assert score_matrix.shape == (
        total_rows,
        len(METRICS),
    )

    threshold_counts = {}

    for threshold in range(1, 11):
        selected = np.all(
            score_matrix >= float(threshold),
            axis=1,
        )

        threshold_counts[str(threshold)] = int(
            selected.sum()
        )

    elapsed = time.time() - start_time

    report = {
        "status": "PASSED",
        "method": "direct_base_laya_native_scoring",
        "qwen_scores_used": False,
        "source_rows": total_rows,
        "elapsed_seconds": elapsed,
        "rows_per_second": (
            total_rows / max(elapsed, 1e-9)
        ),
        "base_checkpoint": str(base_checkpoint),
        "anchor_prediction_source": str(
            prediction_path
        ),
        "cluster_catalog": str(catalog_path),
        "rubric_source": str(rubric_program),
        "output": str(output_path),
        "threshold_rule": (
            "keep sample when all five expected "
            "scores are greater than or equal "
            "to threshold"
        ),
        "all_five_gte_threshold_counts": (
            threshold_counts
        ),
        "metrics": {
            metric: {
                "mean": float(
                    score_matrix[:, index].mean()
                ),
                "standard_deviation": float(
                    score_matrix[:, index].std()
                ),
                "minimum": float(
                    score_matrix[:, index].min()
                ),
                "maximum": float(
                    score_matrix[:, index].max()
                ),
            }
            for index, metric in enumerate(METRICS)
        },
    }

    write_json_atomic(report_path, report)

    write_json_atomic(
        status_path,
        {
            "status": "COMPLETE",
            "stage": "complete",
            "processed_rows": total_rows,
            "requested_rows": total_rows,
            "elapsed_seconds": elapsed,
            "rows_per_second": report[
                "rows_per_second"
            ],
            "qwen_scores_used": False,
            "output": str(output_path),
            "report": str(report_path),
        },
    )

    print("=" * 72)
    print("DIRECT LAYA FULL QUALITY SCORING COMPLETE")
    print("=" * 72)
    print("Rows:", total_rows)
    print("Elapsed seconds:", elapsed)
    print(
        "Rows per second:",
        report["rows_per_second"],
    )
    print(
        "All five expected scores >= 5:",
        threshold_counts["5"],
    )
    print("Qwen scores used: NO")
    print("Output:", output_path)
    print("Report:", report_path)

except Exception as error:
    write_json_atomic(
        status_path,
        {
            "status": "FAILED",
            "stage": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": (
                time.time() - start_time
            ),
            "qwen_scores_used": False,
        },
    )

    traceback.print_exc()
    raise
