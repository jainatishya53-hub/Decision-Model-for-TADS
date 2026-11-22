#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch


TOTAL_ROWS = 300932

EXPECTED_ADAPTER_SHA256 = (
    "bfa10450c786093dc771f3c7b71fb0af"
    "77fa22a03cc7702236a8f84bcda3f4a7"
)


def validate_completed_shard(
    path,
    expected_start,
    expected_end,
):
    count = 0
    first_position = None
    last_position = None

    with path.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            position = int(
                row["source_position"]
            )

            expected_position = (
                expected_start + count
            )

            assert position == (
                expected_position
            ), (
                path,
                position,
                expected_position,
            )

            if first_position is None:
                first_position = position

            last_position = position
            count += 1

    expected_count = (
        expected_end - expected_start
    )

    assert count == expected_count, (
        path,
        count,
        expected_count,
    )

    assert first_position == expected_start
    assert last_position == expected_end - 1

    return count


def score_records(
    decider,
    records,
    questions,
    core,
    chunk_size,
    completed_before,
    total_rows,
    inference_started,
):
    prepared_states = []

    for record in records:
        prepared_states.append(
            core.prepare_state(
                decider,
                record["state"],
                questions,
            )
        )

    output_rows = []

    with torch.inference_mode():
        for start in range(
            0,
            len(records),
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                len(records),
            )

            requests = [
                {
                    "state": prepared_states[
                        index
                    ]["state"],
                    "questions": questions,
                }
                for index in range(start, end)
            ]

            responses = decider.decide(
                requests
            )

            if isinstance(responses, dict):
                responses = [responses]

            assert len(responses) == (
                end - start
            )

            for local_index, response in enumerate(
                responses
            ):
                index = start + local_index
                record = records[index]
                prepared = prepared_states[index]

                assert set(
                    response["answers"]
                ) == set(core.METRICS)

                output = {
                    "scoring_index": int(
                        record.get(
                            "scoring_index",
                            record["source_position"],
                        )
                    ),
                    "source_position": int(
                        record["source_position"]
                    ),
                    "dataset": record["dataset"],
                    "id": record["id"],
                    "text_sha256":
                        record["text_sha256"],
                    "truncated": bool(
                        prepared["truncated"]
                    ),
                    "original_sequence_length":
                        int(
                            prepared[
                                "original_length"
                            ]
                        ),
                    "sequence_length": int(
                        prepared["final_length"]
                    ),
                    "usage": response["usage"],
                }

                scores = []

                for metric in core.METRICS:
                    parsed = (
                        core.parse_score_answer(
                            response["answers"][
                                metric
                            ],
                            questions[metric],
                        )
                    )

                    column = (
                        core.METRIC_COLUMNS[
                            metric
                        ]
                    )

                    output[column] = parsed

                    scores.append(
                        float(
                            parsed[
                                "score_1_to_10"
                            ]
                        )
                    )

                for threshold in (5, 6, 7):
                    output[
                        f"all_five_gte_{threshold}"
                    ] = all(
                        value >= threshold
                        for value in scores
                    )

                output_rows.append(output)

            global_processed = (
                completed_before + end
            )

            elapsed = (
                time.time()
                - inference_started
            )

            rate = (
                global_processed / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                total_rows
                - global_processed
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else None
            )

            print(
                f"processed={global_processed}/"
                f"{total_rows} "
                f"rate={rate:.2f} samples/s "
                f"eta_hours="
                f"{eta_seconds / 3600:.2f}",
                flush=True,
            )

    return output_rows


def aggregate_shards(
    shard_paths,
    core,
    total_rows,
):
    metrics = {
        column: {
            "sum": 0.0,
            "sum_squares": 0.0,
            "minimum": None,
            "maximum": None,
        }
        for column in (
            core.METRIC_COLUMNS.values()
        )
    }

    dataset_counts = Counter()
    threshold_counts = Counter()
    truncated_rows = 0
    count = 0

    for shard_path in shard_paths:
        with shard_path.open(
            encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)

                assert int(
                    row["source_position"]
                ) == count

                dataset_counts[
                    row["dataset"]
                ] += 1

                truncated_rows += int(
                    bool(row["truncated"])
                )

                for threshold in (5, 6, 7):
                    threshold_counts[
                        threshold
                    ] += int(
                        bool(
                            row[
                                f"all_five_gte_"
                                f"{threshold}"
                            ]
                        )
                    )

                for column in metrics:
                    value = float(
                        row[column][
                            "score_1_to_10"
                        ]
                    )

                    accumulator = metrics[column]
                    accumulator["sum"] += value
                    accumulator[
                        "sum_squares"
                    ] += value * value

                    if (
                        accumulator["minimum"]
                        is None
                    ):
                        accumulator[
                            "minimum"
                        ] = value
                        accumulator[
                            "maximum"
                        ] = value
                    else:
                        accumulator[
                            "minimum"
                        ] = min(
                            accumulator[
                                "minimum"
                            ],
                            value,
                        )
                        accumulator[
                            "maximum"
                        ] = max(
                            accumulator[
                                "maximum"
                            ],
                            value,
                        )

                count += 1

    assert count == total_rows

    summary = {
        "rows": count,
        "truncated_rows": truncated_rows,
        "dataset_counts": dict(
            sorted(dataset_counts.items())
        ),
    }

    for column, accumulator in (
        metrics.items()
    ):
        mean = (
            accumulator["sum"] / count
        )

        variance = max(
            0.0,
            (
                accumulator["sum_squares"]
                / count
            )
            - mean * mean,
        )

        summary[column] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "minimum":
                accumulator["minimum"],
            "maximum":
                accumulator["maximum"],
        }

    for threshold in (5, 6, 7):
        selected = threshold_counts[
            threshold
        ]

        summary[
            f"all_five_gte_{threshold}"
        ] = {
            "count": selected,
            "rate": selected / count,
        }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=TOTAL_ROWS,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    assert 1 <= args.limit <= TOTAL_ROWS
    assert 1 <= args.shard_size <= 5000
    assert 1 <= args.chunk_size <= 16

    root = Path(
        os.environ["PPLX_TADS_ROOT"]
    ).resolve()

    source_dir = Path(
        os.environ["PPLX_SOURCE_DIR"]
    ).resolve()

    checkpoint_path = Path(
        os.environ["PPLX_CHECKPOINT"]
    ).resolve()

    sys.path.insert(
        0,
        str(source_dir / "source" / "src"),
    )

    sys.path.insert(
        0,
        str(root / "code"),
    )

    core_path = (
        root
        / "code"
        / "24_run_dual_pplx_quality_scoring_smoke.py"
    )

    specification = (
        importlib.util.spec_from_file_location(
            "validated_pplx_scoring_core",
            core_path,
        )
    )

    core = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(core)

    from pplx_decider import Decider

    adapter_path = (
        root
        / "data"
        / "quality_scoring"
        / "pplx_scoring_full_300932_adapter.jsonl"
    )

    contract_path = (
        root
        / "configs"
        / "21_pplx_quality_scoring_contract.json"
    )

    decision_path = (
        root
        / "final_results"
        / "pplx_quality_scoring"
        / "quality_scoring_arm_selection.json"
    )

    assert checkpoint_path.is_dir()
    assert adapter_path.is_file()
    assert contract_path.is_file()
    assert decision_path.is_file()

    assert core.sha256_file(
        adapter_path
    ) == EXPECTED_ADAPTER_SHA256

    assert core.sha256_file(
        contract_path
    ) == core.EXPECTED_CONTRACT_SHA256

    with contract_path.open(
        encoding="utf-8"
    ) as handle:
        contract = json.load(handle)

    with decision_path.open(
        encoding="utf-8"
    ) as handle:
        decision = json.load(handle)

    assert decision[
        "quality_scoring_arm"
    ]["selected"] == "base_native_pplx"

    assert decision[
        "final_filter"
    ]["threshold"] == 5

    questions = (
        core.questions_from_contract(
            contract
        )
    )

    output_root = (
        root
        / "data"
        / "quality_scoring"
        / args.run_name
    )

    shard_root = (
        output_root / "shards"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        output_root / "status.json"
    )

    report_path = (
        root
        / "metadata"
        / "quality_scoring"
        / f"{args.run_name}.json"
    )

    started = time.time()

    core.atomic_json(
        status_path,
        {
            "status": "STARTING",
            "stage": "loading",
            "requested_rows": args.limit,
            "processed_rows": 0,
            "shard_size": args.shard_size,
            "chunk_size": args.chunk_size,
            "arm": "base_native_pplx",
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

    print(
        "Loading base/native PPLX once...",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.time()

    decider = Decider(
        checkpoint_path,
        device="cuda",
        fast=False,
        max_length=core.MAXIMUM_LENGTH,
        batch_tokens=core.BATCH_TOKENS,
        cpu_threads=8,
    )

    decider.backbone.eval()
    decider.readout.eval()

    readout_sha256 = (
        core.tensor_sha256(
            decider.readout.weight
        )
    )

    torch.cuda.synchronize()

    load_seconds = (
        time.time() - load_started
    )

    inference_started = time.time()
    completed_rows = 0
    shard_paths = []

    with adapter_path.open(
        encoding="utf-8"
    ) as adapter:
        shard_index = 0

        while completed_rows < args.limit:
            shard_start = completed_rows
            shard_end = min(
                shard_start + args.shard_size,
                args.limit,
            )

            records = []

            for expected_position in range(
                shard_start,
                shard_end,
            ):
                line = adapter.readline()

                assert line, (
                    "Adapter ended early",
                    expected_position,
                )

                record = json.loads(line)

                assert int(
                    record["source_position"]
                ) == expected_position

                records.append(record)

            shard_path = (
                shard_root
                / (
                    f"shard_{shard_start:06d}_"
                    f"{shard_end - 1:06d}.jsonl"
                )
            )

            shard_paths.append(shard_path)

            if shard_path.is_file():
                validate_completed_shard(
                    shard_path,
                    shard_start,
                    shard_end,
                )

                completed_rows = shard_end

                print(
                    f"resume: verified existing "
                    f"shard={shard_index} "
                    f"rows={completed_rows}/"
                    f"{args.limit}",
                    flush=True,
                )
            else:
                output_rows = score_records(
                    decider=decider,
                    records=records,
                    questions=questions,
                    core=core,
                    chunk_size=args.chunk_size,
                    completed_before=(
                        shard_start
                    ),
                    total_rows=args.limit,
                    inference_started=(
                        inference_started
                    ),
                )

                assert len(output_rows) == (
                    shard_end - shard_start
                )

                core.write_jsonl(
                    shard_path,
                    output_rows,
                )

                validate_completed_shard(
                    shard_path,
                    shard_start,
                    shard_end,
                )

                completed_rows = shard_end

            elapsed = time.time() - started

            core.atomic_json(
                status_path,
                {
                    "status": "RUNNING",
                    "stage": "scoring",
                    "arm": "base_native_pplx",
                    "requested_rows":
                        args.limit,
                    "processed_rows":
                        completed_rows,
                    "completed_shards":
                        len(shard_paths),
                    "shard_size":
                        args.shard_size,
                    "chunk_size":
                        args.chunk_size,
                    "elapsed_seconds":
                        elapsed,
                    "rows_per_second": (
                        completed_rows
                        / elapsed
                    ),
                    "last_completed_shard":
                        str(shard_path),
                    "native_readout_sha256":
                        readout_sha256,
                    "laya_model_used": False,
                    "qwen_used": False,
                },
            )

            shard_index += 1

    assert completed_rows == args.limit

    print(
        "Aggregating completed shards...",
        flush=True,
    )

    summary = aggregate_shards(
        shard_paths,
        core,
        args.limit,
    )

    readout_after = core.tensor_sha256(
        decider.readout.weight
    )

    assert readout_after == readout_sha256

    torch.cuda.synchronize()

    elapsed = time.time() - started

    report = {
        "status": "PASSED",
        "stage": "complete",
        "arm": "base_native_pplx",
        "rows": args.limit,
        "summary": summary,
        "runtime": {
            "load_seconds": load_seconds,
            "elapsed_seconds": elapsed,
            "rows_per_second": (
                args.limit / elapsed
            ),
            "peak_gpu_memory_reserved_gib": (
                torch.cuda.max_memory_reserved()
                / (1024 ** 3)
            ),
        },
        "maximum_length":
            core.MAXIMUM_LENGTH,
        "batch_tokens":
            core.BATCH_TOKENS,
        "chunk_size": args.chunk_size,
        "shard_size": args.shard_size,
        "shards": len(shard_paths),
        "output_root": str(output_root),
        "shard_root": str(shard_root),
        "adapter": str(adapter_path),
        "adapter_sha256":
            EXPECTED_ADAPTER_SHA256,
        "contract": str(contract_path),
        "contract_sha256":
            core.EXPECTED_CONTRACT_SHA256,
        "native_readout_sha256_before":
            readout_sha256,
        "native_readout_sha256_after":
            readout_after,
        "native_readout_unchanged": True,
        "selection_threshold": 5,
        "model": (
            "perplexity-ai/"
            "pplx-decider-v1.2"
        ),
        "laya_model_used": False,
        "laya_predictions_used": False,
        "laya_scores_used": False,
        "qwen_used": False,
        "external_teacher_scores_used":
            False,
    }

    core.atomic_json(
        report_path,
        report,
    )

    core.atomic_json(
        status_path,
        {
            **report,
            "report": str(report_path),
        },
    )

    print("=" * 78)
    print("BASE PPLX FULL QUALITY SCORING PASSED")
    print("=" * 78)
    print("Rows:", args.limit)
    print("Shards:", len(shard_paths))
    print(
        "Truncated rows:",
        summary["truncated_rows"],
    )

    for threshold in (5, 6, 7):
        value = summary[
            f"all_five_gte_{threshold}"
        ]

        print(
            f"All five >= {threshold}: "
            f"{value['count']} "
            f"({100 * value['rate']:.2f}%)"
        )

    print(
        "Rows per second:",
        report["runtime"][
            "rows_per_second"
        ],
    )
    print("Output:", shard_root)
    print("Report:", report_path)
    print("LAYA/Qwen runtime use: NO")


if __name__ == "__main__":
    main()
