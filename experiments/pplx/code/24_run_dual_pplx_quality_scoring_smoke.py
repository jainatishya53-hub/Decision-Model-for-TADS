#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open


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

EXPECTED_ADAPTER_SHA256 = (
    "398305b2ed29d0a6a7889e7410971d89"
    "fb5bc0ca142b3690c955857727ba3701"
)

EXPECTED_CONTRACT_SHA256 = (
    "6539e8307b25e1706998657d8d7f197"
    "f6410ef227a3ab214c22574bcf394a173"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "293343768aa1f238f1a8a0e7795a270"
    "e7fed49282d198b46bcd157f9123d95f2"
)

MAXIMUM_LENGTH = 2048
BATCH_TOKENS = 8192


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


def load_jsonl(path, limit):
    records = []

    with path.open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            records.append(
                json.loads(line)
            )

            if len(records) >= limit:
                break

    assert len(records) == limit

    for index, record in enumerate(records):
        assert record["pilot_index"] == index

    return records


def recursive_value(value, aliases):
    if isinstance(value, dict):
        for alias in aliases:
            if alias in value:
                return value[alias]

        for child in value.values():
            result = recursive_value(
                child,
                aliases,
            )

            if result is not None:
                return result

    return None


def tensor_sha256(tensor):
    raw = (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )

    return hashlib.sha256(raw).hexdigest()


def questions_from_contract(contract):
    questions = {}

    for metric in METRICS:
        question = copy.deepcopy(
            contract["questions"][metric]
        )

        assert question["type"] == "score"
        assert len(question["criteria"]) == 10
        assert question.get("instructions")

        questions[metric] = question

    return questions


def split_state(state):
    sample_marker = (
        "SAMPLE TO EVALUATE\n"
    )
    anchor_marker = (
        "\n\nASSIGNED TASK\n"
    )

    assert sample_marker in state
    assert anchor_marker in state

    prefix, remainder = state.split(
        sample_marker,
        1,
    )

    sample_text, suffix = remainder.split(
        anchor_marker,
        1,
    )

    prefix = prefix + sample_marker
    suffix = anchor_marker + suffix

    return prefix, sample_text, suffix


def prepare_state(
    decider,
    original_state,
    questions,
):
    request = {
        "state": original_state,
        "questions": questions,
    }

    encoded = decider.encoder.encode(
        request
    )

    original_length = int(
        encoded.length
    )

    if original_length <= MAXIMUM_LENGTH:
        return {
            "state": original_state,
            "truncated": False,
            "original_length": original_length,
            "final_length": original_length,
        }

    prefix, sample_text, suffix = (
        split_state(original_state)
    )

    tokenizer = (
        decider.processor.tokenizer
    )

    sample_tokens = tokenizer.encode(
        sample_text,
        add_special_tokens=False,
    )

    def candidate(count):
        retained_text = tokenizer.decode(
            sample_tokens[:count],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        state = (
            prefix
            + retained_text
            + suffix
        )

        length = int(
            decider.encoder.encode(
                {
                    "state": state,
                    "questions": questions,
                }
            ).length
        )

        return state, length

    zero_state, zero_length = candidate(0)

    assert zero_length <= MAXIMUM_LENGTH, (
        "Prompt and anchors alone exceed "
        "maximum length"
    )

    low = 0
    high = len(sample_tokens)
    best_state = zero_state
    best_length = zero_length

    while low <= high:
        middle = (
            low + high
        ) // 2

        state, length = candidate(
            middle
        )

        if length <= MAXIMUM_LENGTH:
            best_state = state
            best_length = length
            low = middle + 1
        else:
            high = middle - 1

    assert best_length <= MAXIMUM_LENGTH

    return {
        "state": best_state,
        "truncated": True,
        "original_length": original_length,
        "final_length": best_length,
    }


def parse_score_answer(
    answer,
    question,
):
    assert answer["type"] == "score"

    criteria = question["criteria"]
    legend = answer["legend"]
    probabilities = answer[
        "probabilities"
    ]

    assert set(legend) == set(
        probabilities
    )

    indexed_probabilities = [
        None
    ] * len(criteria)

    for code, description in legend.items():
        assert description in criteria, (
            description,
            criteria,
        )

        index = criteria.index(
            description
        )

        assert indexed_probabilities[
            index
        ] is None

        indexed_probabilities[index] = (
            float(probabilities[code])
        )

    assert all(
        value is not None
        for value in indexed_probabilities
    )

    probability_sum = sum(
        indexed_probabilities
    )

    assert abs(
        probability_sum - 1.0
    ) <= 1.0e-5

    raw_expected = sum(
        index * probability
        for index, probability in enumerate(
            indexed_probabilities
        )
    )

    native_score = float(
        answer["score"]
    )

    assert abs(
        native_score - raw_expected
    ) <= 0.02, (
        native_score,
        raw_expected,
    )

    assert 0.0 <= native_score <= 9.0

    final_score = native_score + 1.0

    return {
        "score_1_to_10": final_score,
        "native_expected_level_0_to_9":
            native_score,
        "argmax_level_1_to_10": (
            int(
                np.argmax(
                    indexed_probabilities
                )
            )
            + 1
        ),
        "confidence": float(
            answer["confidence"]
        ),
        "probabilities": (
            indexed_probabilities
        ),
    }


def install_epoch8_lora(
    decider,
    components,
    config,
    checkpoint_path,
):
    adaptation = config[
        "adaptation"
    ]

    rank = int(
        adaptation["selected_rank"]
    )

    alpha = recursive_value(
        adaptation,
        (
            "alpha",
            "lora_alpha",
        ),
    )

    dropout = recursive_value(
        adaptation,
        (
            "dropout",
            "lora_dropout",
        ),
    )

    assert rank == 256
    assert alpha is not None
    assert dropout is not None

    counts = components.install_lora(
        decider.backbone,
        rank=rank,
        alpha=float(alpha),
        dropout=float(dropout),
    )

    assert sum(counts.values()) == 400

    parameters = dict(
        decider.backbone.named_parameters()
    )

    loaded_names = set()

    with safe_open(
        str(checkpoint_path),
        framework="pt",
        device="cpu",
    ) as handle:
        for key in handle.keys():
            prefix = (
                "parameter::backbone."
            )

            if not key.startswith(prefix):
                continue

            name = key[len(prefix):]

            assert name in parameters, (
                f"Unknown LoRA parameter: {name}"
            )

            destination = parameters[name]
            value = handle.get_tensor(key)

            with torch.no_grad():
                destination.copy_(
                    value.to(
                        device=destination.device,
                        dtype=destination.dtype,
                    )
                )

            loaded_names.add(name)

    assert len(loaded_names) == 800, (
        len(loaded_names)
    )

    expected_lora_names = {
        name
        for name in parameters
        if (
            name.endswith(".lora_A")
            or name.endswith(".lora_B")
        )
    }

    assert loaded_names == (
        expected_lora_names
    )

    decider.backbone.eval()
    decider.readout.eval()

    return {
        "rank": rank,
        "alpha": float(alpha),
        "dropout": float(dropout),
        "modules": counts,
        "loaded_lora_tensors": len(
            loaded_names
        ),
    }


def score_arm(
    arm,
    records,
    questions,
    checkpoint_path,
    epoch_checkpoint_path,
    config,
    components,
    chunk_size,
    prepared_states=None,
):
    from pplx_decider import Decider

    print(
        f"\nLoading {arm} PPLX...",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.time()

    decider = Decider(
        checkpoint_path,
        device="cuda",
        fast=False,
        max_length=MAXIMUM_LENGTH,
        batch_tokens=BATCH_TOKENS,
        cpu_threads=8,
    )

    decider.backbone.eval()
    decider.readout.eval()

    readout_before = tensor_sha256(
        decider.readout.weight
    )

    lora_report = None

    if arm == "finetuned":
        lora_report = install_epoch8_lora(
            decider,
            components,
            config,
            epoch_checkpoint_path,
        )

    readout_after = tensor_sha256(
        decider.readout.weight
    )

    assert readout_before == readout_after

    torch.cuda.synchronize()

    load_seconds = (
        time.time() - load_started
    )

    if prepared_states is None:
        print(
            "Preparing exact scoring states...",
            flush=True,
        )

        prepared_states = []

        for index, record in enumerate(
            records
        ):
            prepared = prepare_state(
                decider,
                record["state"],
                questions,
            )

            prepared_states.append(
                prepared
            )

            if (
                (index + 1) % 100 == 0
                or index + 1 == len(records)
            ):
                print(
                    f"prepared={index + 1}/"
                    f"{len(records)}",
                    flush=True,
                )

    output_rows = []
    inference_started = time.time()

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
                for index in range(
                    start,
                    end,
                )
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

                assert set(
                    response["answers"]
                ) == set(METRICS)

                output = {
                    "pilot_index": (
                        record["pilot_index"]
                    ),
                    "source_position": (
                        record["source_position"]
                    ),
                    "dataset": (
                        record["dataset"]
                    ),
                    "id": record["id"],
                    "text_sha256": (
                        record["text_sha256"]
                    ),
                    "truncated": (
                        prepared_states[
                            index
                        ]["truncated"]
                    ),
                    "original_sequence_length": (
                        prepared_states[
                            index
                        ]["original_length"]
                    ),
                    "sequence_length": (
                        prepared_states[
                            index
                        ]["final_length"]
                    ),
                    "usage": response["usage"],
                }

                final_scores = []

                for metric in METRICS:
                    parsed = parse_score_answer(
                        response["answers"][
                            metric
                        ],
                        questions[metric],
                    )

                    column = METRIC_COLUMNS[
                        metric
                    ]

                    output[column] = parsed
                    final_scores.append(
                        parsed[
                            "score_1_to_10"
                        ]
                    )

                output[
                    "all_five_gte_5"
                ] = all(
                    value >= 5.0
                    for value in final_scores
                )

                output[
                    "all_five_gte_6"
                ] = all(
                    value >= 6.0
                    for value in final_scores
                )

                output[
                    "all_five_gte_7"
                ] = all(
                    value >= 7.0
                    for value in final_scores
                )

                output_rows.append(output)

            elapsed = (
                time.time()
                - inference_started
            )

            print(
                f"{arm}: processed={end}/"
                f"{len(records)} "
                f"rate={end / elapsed:.2f} "
                f"samples/s",
                flush=True,
            )

    torch.cuda.synchronize()

    inference_seconds = (
        time.time() - inference_started
    )

    peak_gib = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    runtime = {
        "arm": arm,
        "load_seconds": load_seconds,
        "inference_seconds": (
            inference_seconds
        ),
        "rows_per_second": (
            len(records)
            / inference_seconds
        ),
        "peak_gpu_memory_reserved_gib":
            peak_gib,
        "readout_sha256_before":
            readout_before,
        "readout_sha256_after":
            readout_after,
        "native_readout_unchanged": True,
        "lora": lora_report,
    }

    del decider
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return (
        output_rows,
        runtime,
        prepared_states,
    )


def arm_summary(rows):
    summary = {
        "rows": len(rows),
        "truncated_rows": sum(
            bool(row["truncated"])
            for row in rows
        ),
    }

    for metric, column in (
        METRIC_COLUMNS.items()
    ):
        values = np.asarray(
            [
                row[column][
                    "score_1_to_10"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        summary[column] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }

    for threshold in (5, 6, 7):
        key = f"all_five_gte_{threshold}"

        count = sum(
            bool(row[key])
            for row in rows
        )

        summary[key] = {
            "count": count,
            "rate": count / len(rows),
        }

    return summary


def write_jsonl(path, rows):
    temporary = path.with_suffix(
        ".jsonl.incomplete"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    args = parser.parse_args()

    assert 1 <= args.limit <= 1000
    assert 1 <= args.chunk_size <= 16

    root = Path(
        os.environ["PPLX_TADS_ROOT"]
    ).resolve()

    runtime_root = Path(
        os.environ["PPLX_RUNTIME_ROOT"]
    ).resolve()

    source_dir = Path(
        os.environ["PPLX_SOURCE_DIR"]
    ).resolve()

    sys.path.insert(
        0,
        str(source_dir / "source" / "src"),
    )

    sys.path.insert(
        0,
        str(root / "code"),
    )

    import pplx_training_components as C

    checkpoint_path = Path(
        os.environ["PPLX_CHECKPOINT"]
    ).resolve()

    adapter_path = (
        root
        / "data"
        / "quality_scoring"
        / "pplx_scoring_pilot_1000_adapter.jsonl"
    )

    contract_path = (
        root
        / "configs"
        / "21_pplx_quality_scoring_contract.json"
    )

    config_path = (
        root
        / "configs"
        / "09_pplx_description_four_head_training.json"
    )

    epoch_checkpoint_path = (
        root
        / "final_results"
        / "pplx_decider_v1_2_rank256_microbatch2_10epoch"
        / "best_epoch_08_model.safetensors"
    )

    # The official PPLX checkpoint is a directory containing
    # the sharded base-model weights. The remaining inputs are files.
    assert checkpoint_path.is_dir(), checkpoint_path

    checkpoint_index = (
        checkpoint_path
        / "model.safetensors.index.json"
    )
    native_readout = (
        checkpoint_path
        / "readout.safetensors"
    )

    assert checkpoint_index.is_file(), checkpoint_index
    assert native_readout.is_file(), native_readout

    for input_file in (
        adapter_path,
        contract_path,
        config_path,
        epoch_checkpoint_path,
    ):
        assert input_file.is_file(), input_file

    assert sha256_file(adapter_path) == (
        EXPECTED_ADAPTER_SHA256
    )

    assert sha256_file(contract_path) == (
        EXPECTED_CONTRACT_SHA256
    )

    with contract_path.open(
        encoding="utf-8"
    ) as handle:
        contract = json.load(handle)

    with config_path.open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)

    questions = questions_from_contract(
        contract
    )

    records = load_jsonl(
        adapter_path,
        args.limit,
    )

    output_root = (
        root
        / "data"
        / "quality_scoring"
        / args.run_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        root
        / "metadata"
        / "quality_scoring"
        / f"{args.run_name}.json"
    )

    status_path = (
        output_root
        / "status.json"
    )

    atomic_json(
        status_path,
        {
            "status": "STARTING",
            "stage": "loading",
            "rows": args.limit,
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

    started = time.time()

    base_rows, base_runtime, prepared = (
        score_arm(
            arm="base",
            records=records,
            questions=questions,
            checkpoint_path=checkpoint_path,
            epoch_checkpoint_path=(
                epoch_checkpoint_path
            ),
            config=config,
            components=C,
            chunk_size=args.chunk_size,
            prepared_states=None,
        )
    )

    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "stage": "base_complete",
            "rows": args.limit,
        },
    )

    hybrid_rows, hybrid_runtime, _ = (
        score_arm(
            arm="finetuned",
            records=records,
            questions=questions,
            checkpoint_path=checkpoint_path,
            epoch_checkpoint_path=(
                epoch_checkpoint_path
            ),
            config=config,
            components=C,
            chunk_size=args.chunk_size,
            prepared_states=prepared,
        )
    )

    assert len(base_rows) == len(
        hybrid_rows
    )

    comparison = {}

    for metric, column in (
        METRIC_COLUMNS.items()
    ):
        base_values = np.asarray(
            [
                row[column][
                    "score_1_to_10"
                ]
                for row in base_rows
            ],
            dtype=np.float64,
        )

        hybrid_values = np.asarray(
            [
                row[column][
                    "score_1_to_10"
                ]
                for row in hybrid_rows
            ],
            dtype=np.float64,
        )

        if (
            base_values.std() > 0
            and hybrid_values.std() > 0
        ):
            correlation = float(
                np.corrcoef(
                    base_values,
                    hybrid_values,
                )[0, 1]
            )
        else:
            correlation = None

        comparison[column] = {
            "hybrid_minus_base_mean": float(
                (
                    hybrid_values
                    - base_values
                ).mean()
            ),
            "mean_absolute_difference": float(
                np.abs(
                    hybrid_values
                    - base_values
                ).mean()
            ),
            "pearson_r": correlation,
        }

    base_path = (
        output_root
        / "base_scores.jsonl"
    )

    hybrid_path = (
        output_root
        / "finetuned_scores.jsonl"
    )

    write_jsonl(
        base_path,
        base_rows,
    )

    write_jsonl(
        hybrid_path,
        hybrid_rows,
    )

    report = {
        "status": "PASSED",
        "stage": "complete",
        "rows": args.limit,
        "contract_sha256": (
            EXPECTED_CONTRACT_SHA256
        ),
        "adapter_sha256": (
            EXPECTED_ADAPTER_SHA256
        ),
        "epoch_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA256
        ),
        "maximum_length": MAXIMUM_LENGTH,
        "batch_tokens": BATCH_TOKENS,
        "chunk_size": args.chunk_size,
        "base": {
            "summary": arm_summary(
                base_rows
            ),
            "runtime": base_runtime,
            "output": str(base_path),
        },
        "finetuned": {
            "summary": arm_summary(
                hybrid_rows
            ),
            "runtime": hybrid_runtime,
            "output": str(hybrid_path),
        },
        "comparison": comparison,
        "selection_policy": (
            "No arm is selected from unlabeled "
            "smoke data alone; inspect native "
            "validity, distributions, stability, "
            "and architectural compatibility."
        ),
        "elapsed_seconds": (
            time.time() - started
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
    print("DUAL PPLX QUALITY-SCORING SMOKE PASSED")
    print("=" * 78)
    print("Rows:", args.limit)

    for arm in (
        "base",
        "finetuned",
    ):
        summary = report[arm][
            "summary"
        ]

        print(f"\n{arm.upper()}")

        for metric, column in (
            METRIC_COLUMNS.items()
        ):
            values = summary[column]

            print(
                f"  {metric}: "
                f"mean={values['mean']:.4f} "
                f"std={values['std']:.4f} "
                f"range="
                f"{values['minimum']:.4f}-"
                f"{values['maximum']:.4f}"
            )

        for threshold in (5, 6, 7):
            value = summary[
                f"all_five_gte_{threshold}"
            ]

            print(
                f"  All five >= {threshold}: "
                f"{value['count']} "
                f"({100 * value['rate']:.2f}%)"
            )

    print("\nBASE–FINETUNED COMPARISON")

    for metric, column in (
        METRIC_COLUMNS.items()
    ):
        value = comparison[column]

        print(
            f"  {metric}: "
            f"delta="
            f"{value['hybrid_minus_base_mean']:+.4f}, "
            f"MAE="
            f"{value['mean_absolute_difference']:.4f}, "
            f"r={value['pearson_r']}"
        )

    print("\nReport:", report_path)
    print("LAYA/Qwen runtime use: NO")


if __name__ == "__main__":
    main()
