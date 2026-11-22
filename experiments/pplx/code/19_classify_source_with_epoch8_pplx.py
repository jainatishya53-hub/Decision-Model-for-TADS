#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback

import torch
from safetensors.torch import load_file


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

CLASSES = 100

QUESTION_TEMPLATE = (
    "Produce the semantic representation used "
    "to determine the {field} cluster."
)

OPTION_TEMPLATE = (
    "Semantic evidence relevant to the "
    "{field} cluster."
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


def make_request(text):
    # Exact request used for PPLX fine-tuning.
    return {
        "state": text,
        "questions": {
            field: {
                "type": "choice",
                "instructions":
                    QUESTION_TEMPLATE.format(
                        field=field
                    ),
                "criteria": {
                    "semantic_representation":
                        OPTION_TEMPLATE.format(
                            field=field
                        )
                },
            }
            for field in FIELDS
        },
    }


def encode_with_truncation(
    encoder,
    tokenizer,
    original_text,
    maximum_length,
):
    # Exact truncation policy used during training.
    full_encoded = encoder.encode(
        make_request(original_text)
    )

    original_length = (
        full_encoded.length
    )

    if original_length <= maximum_length:
        state_tokens = len(
            tokenizer.encode(
                original_text,
                add_special_tokens=False,
            )
        )

        return (
            full_encoded,
            False,
            original_length,
            state_tokens,
            state_tokens,
        )

    raw_token_ids = tokenizer.encode(
        original_text,
        add_special_tokens=False,
    )

    original_state_tokens = len(
        raw_token_ids
    )

    assert original_state_tokens >= 1

    retained = list(raw_token_ids)
    previous_size = None

    for _ in range(20):
        truncated_text = tokenizer.decode(
            retained,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        encoded = encoder.encode(
            make_request(truncated_text)
        )

        if encoded.length <= maximum_length:
            return (
                encoded,
                True,
                original_length,
                original_state_tokens,
                len(retained),
            )

        overflow = (
            encoded.length
            - maximum_length
        )

        new_size = max(
            1,
            len(retained)
            - overflow
            - 8,
        )

        if (
            previous_size == new_size
            or new_size >= len(retained)
        ):
            new_size = max(
                1,
                len(retained) - 1,
            )

        previous_size = len(retained)
        retained = retained[:new_size]

    raise RuntimeError(
        "Could not fit record within "
        f"{maximum_length} tokens"
    )


def load_epoch_checkpoint(
    model,
    checkpoint_path,
):
    tensors = load_file(
        str(checkpoint_path),
        device="cpu",
    )

    assert len(tensors) == 814, (
        f"Expected 814 checkpoint tensors, "
        f"found {len(tensors)}"
    )

    parameters = dict(
        model.named_parameters()
    )

    buffers = dict(
        model.named_buffers()
    )

    loaded_parameters = 0
    loaded_buffers = 0

    for key, value in tensors.items():
        if key.startswith("parameter::"):
            name = key.split("::", 1)[1]

            assert name in parameters, (
                f"Unknown parameter: {name}"
            )

            destination = parameters[name]

            with torch.no_grad():
                destination.copy_(
                    value.to(
                        device=destination.device,
                        dtype=destination.dtype,
                    )
                )

            loaded_parameters += 1

        elif key.startswith("buffer::"):
            name = key.split("::", 1)[1]

            assert name in buffers, (
                f"Unknown buffer: {name}"
            )

            destination = buffers[name]

            with torch.no_grad():
                destination.copy_(
                    value.to(
                        device=destination.device,
                        dtype=destination.dtype,
                    )
                )

            loaded_buffers += 1

        else:
            raise AssertionError(
                f"Unexpected tensor key: {key}"
            )

    assert loaded_parameters == 810
    assert loaded_buffers == 4

    del tensors
    gc.collect()

    return {
        "parameters": loaded_parameters,
        "buffers": loaded_buffers,
    }


def adapter_records(path, limit):
    with path.open(
        encoding="utf-8"
    ) as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break

            record = json.loads(line)

            assert (
                record["source_position"]
                == index
            )

            yield record


def encode_record(
    source_record,
    encoder,
    tokenizer,
    maximum_length,
):
    (
        encoded,
        truncated,
        original_sequence_length,
        original_state_tokens,
        retained_state_tokens,
    ) = encode_with_truncation(
        encoder,
        tokenizer,
        source_record["text"],
        maximum_length,
    )

    assert encoded.length <= (
        maximum_length
    )

    assert tuple(encoded.keys) == FIELDS
    assert len(encoded.markers) == 4
    assert encoded.pixel_values is None
    assert encoded.image_grid_thw is None

    markers = [
        int(value)
        for value
        in encoded.markers.tolist()
    ]

    assert markers == sorted(markers)

    assert all(
        0 <= marker < encoded.length
        for marker in markers
    )

    return {
        "source_position":
            source_record[
                "source_position"
            ],
        "dataset":
            source_record["dataset"],
        "id":
            source_record["id"],
        "text_sha256":
            source_record["text_sha256"],
        "input_ids": [
            int(value)
            for value
            in encoded.input_ids.tolist()
        ],
        "markers": markers,
        "sequence_length":
            encoded.length,
        "original_sequence_length":
            original_sequence_length,
        "original_state_tokens":
            original_state_tokens,
        "retained_state_tokens":
            retained_state_tokens,
        "truncated":
            truncated,
    }


def prediction_rows(
    model,
    encoded_records,
):
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits, _ = model(
                encoded_records
            )

        field_probabilities = {
            field: torch.softmax(
                logits[field].float(),
                dim=-1,
            ).cpu()
            for field in FIELDS
        }

    output_rows = []

    for batch_index, record in enumerate(
        encoded_records
    ):
        output = {
            "source_position":
                record["source_position"],
            "dataset":
                record["dataset"],
            "id":
                record["id"],
            "text_sha256":
                record["text_sha256"],
            "sequence_length":
                record["sequence_length"],
            "original_sequence_length":
                record[
                    "original_sequence_length"
                ],
            "original_state_tokens":
                record[
                    "original_state_tokens"
                ],
            "retained_state_tokens":
                record[
                    "retained_state_tokens"
                ],
            "truncated":
                record["truncated"],
        }

        for field in FIELDS:
            probability = (
                field_probabilities[field][
                    batch_index
                ]
            )

            probability_sum = float(
                probability.sum()
            )

            assert abs(
                probability_sum - 1.0
            ) < 1e-5

            top_k = TOP_K[field]

            values, indices = torch.topk(
                probability,
                k=top_k,
                dim=-1,
            )

            entropy = float(
                -(
                    probability
                    * probability.clamp_min(
                        1e-30
                    ).log()
                ).sum()
            )

            distribution = [
                float(value)
                for value
                in probability.tolist()
            ]

            output[
                f"{field}_indices"
            ] = [
                int(value)
                for value
                in indices.tolist()
            ]

            output[
                f"{field}_probabilities"
            ] = [
                float(value)
                for value
                in values.tolist()
            ]

            output[
                f"{field}_distribution"
            ] = distribution

            output[
                f"{field}_max_probability"
            ] = float(values[0])

            output[
                f"{field}_entropy"
            ] = entropy

        output_rows.append(output)

    return output_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    assert 1 <= args.limit <= 300932
    assert args.batch_size in {1, 2, 4, 8}, (
        "Supported inference batch sizes are 1, 2, 4, and 8; "
        f"received {args.batch_size}"
    )
    assert args.shard_size >= args.batch_size

    root = Path(
        os.environ["PPLX_TADS_ROOT"]
    ).resolve()

    runtime_root = Path(
        os.environ["PPLX_RUNTIME_ROOT"]
    ).resolve()

    code_root = root / "code"

    sys.path.insert(
        0,
        str(code_root),
    )

    import pplx_training_components as C

    config_path = (
        root
        / "configs"
        / "09_pplx_description_four_head_training.json"
    )

    adapter_path = (
        runtime_root
        / "source_adapter"
        / "source_300932_exact_laya_text.jsonl"
    )

    checkpoint_path = (
        root
        / "final_results"
        / "pplx_decider_v1_2_rank256_microbatch2_10epoch"
        / "best_epoch_08_model.safetensors"
    )

    prototype_path = (
        root
        / "data"
        / "description_prototypes"
        / "pplx_base_description_prototypes.safetensors"
    )

    experiment_card_path = (
        root
        / "final_results"
        / "pplx_decider_v1_2_rank256_microbatch2_10epoch"
        / "EXPERIMENT_CARD.json"
    )

    for path in (
        config_path,
        adapter_path,
        checkpoint_path,
        prototype_path,
        experiment_card_path,
    ):
        assert path.is_file(), (
            f"Missing required file: {path}"
        )

    with config_path.open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)

    with experiment_card_path.open(
        encoding="utf-8"
    ) as handle:
        experiment_card = json.load(
            handle
        )

    adaptation = config["adaptation"]
    training = config["training"]
    loss_config = config["loss"]

    rank = int(
        adaptation["selected_rank"]
    )

    maximum_length = int(
        training["maximum_length"]
    )

    assert rank == 256
    assert maximum_length == 768
    assert experiment_card[
        "best_checkpoint"
    ]["sha256"] == (
        "293343768aa1f238f1a8a0e7795a270"
        "e7fed49282d198b46bcd157f9123d95f2"
    )

    output_root = (
        runtime_root
        / "cluster_predictions"
        / args.run_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        root
        / "data"
        / "source_predictions"
        / f"{args.run_name}_status.json"
    )

    report_path = (
        root
        / "metadata"
        / "source_predictions"
        / f"{args.run_name}.json"
    )

    started = time.time()

    atomic_json(
        status_path,
        {
            "status": "STARTING",
            "stage": "loading",
            "limit": args.limit,
            "batch_size":
                args.batch_size,
            "checkpoint":
                str(checkpoint_path),
            "output_root":
                str(output_root),
        },
    )

    assert torch.cuda.is_available()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading exact base PPLX checkpoint...",
        flush=True,
    )

    load_started = time.time()

    prototypes = load_file(
        str(prototype_path),
        device="cpu",
    )

    decider = C.Decider(
        C.checkpoint,
        device="cuda",
        fast=False,
        max_length=maximum_length,
        batch_tokens=maximum_length,
        cpu_threads=8,
    )

    decider.backbone.requires_grad_(
        False
    )

    decider.readout.requires_grad_(
        False
    )

    module_counts = C.install_lora(
        decider.backbone.language_model,
        rank=rank,
        alpha=adaptation["alpha"],
        dropout=adaptation["dropout"],
    )

    if hasattr(
        decider.backbone.config,
        "use_cache",
    ):
        decider.backbone.config.use_cache = (
            False
        )

    model = C.PPLXFourHeadModel(
        backbone=decider.backbone,
        prototypes=prototypes,
        dropout=config[
            "architecture"
        ]["dropout"],
        description_scale=loss_config[
            "description_logit_scale"
        ],
    )

    checkpoint_load = (
        load_epoch_checkpoint(
            model,
            checkpoint_path,
        )
    )

    model.requires_grad_(False)
    model.eval()

    tokenizer = (
        decider.processor.tokenizer
    )

    load_seconds = (
        time.time() - load_started
    )

    print(
        f"Model loaded in "
        f"{load_seconds:.2f} seconds",
        flush=True,
    )

    print(
        "LoRA module counts:",
        module_counts,
        flush=True,
    )

    dataset_counts = Counter()
    truncated_count = 0

    observed_top1 = {
        field: set()
        for field in FIELDS
    }

    observed_any_topk = {
        field: set()
        for field in FIELDS
    }

    probability_sum_error = {
        field: 0.0
        for field in FIELDS
    }

    max_probability_sum = {
        field: 0.0
        for field in FIELDS
    }

    entropy_sum = {
        field: 0.0
        for field in FIELDS
    }

    processed = 0
    inference_started = time.time()

    source_iterator = adapter_records(
        adapter_path,
        args.limit,
    )

    shard_records = []
    shard_start = 0

    def write_shard(records, start):
        nonlocal processed
        nonlocal truncated_count

        shard_index = (
            start // args.shard_size
        )

        shard_path = (
            output_root
            / f"shard_{shard_index:06d}.jsonl"
        )

        completion_path = (
            output_root
            / f"shard_{shard_index:06d}.complete.json"
        )

        temporary = shard_path.with_suffix(
            shard_path.suffix + ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for batch_start in range(
                0,
                len(records),
                args.batch_size,
            ):
                source_batch = records[
                    batch_start:
                    batch_start
                    + args.batch_size
                ]

                encoding_started = time.time()

                encoded_batch = [
                    encode_record(
                        source_record,
                        decider.encoder,
                        tokenizer,
                        maximum_length,
                    )
                    for source_record
                    in source_batch
                ]

                predicted = prediction_rows(
                    model,
                    encoded_batch,
                )

                for row in predicted:
                    handle.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                    dataset_counts[
                        row["dataset"]
                    ] += 1

                    truncated_count += int(
                        row["truncated"]
                    )

                    for field in FIELDS:
                        indices = row[
                            f"{field}_indices"
                        ]

                        observed_top1[
                            field
                        ].add(indices[0])

                        observed_any_topk[
                            field
                        ].update(indices)

                        distribution = row[
                            f"{field}_distribution"
                        ]

                        sum_error = abs(
                            sum(distribution)
                            - 1.0
                        )

                        probability_sum_error[
                            field
                        ] = max(
                            probability_sum_error[
                                field
                            ],
                            sum_error,
                        )

                        max_probability_sum[
                            field
                        ] += row[
                            f"{field}_max_probability"
                        ]

                        entropy_sum[
                            field
                        ] += row[
                            f"{field}_entropy"
                        ]

                    processed += 1

                elapsed = (
                    time.time()
                    - inference_started
                )

                print(
                    f"processed={processed}/"
                    f"{args.limit} "
                    f"rate="
                    f"{processed / max(elapsed, 1e-9):.2f} "
                    f"samples/s",
                    flush=True,
                )

        temporary.replace(shard_path)

        atomic_json(
            completion_path,
            {
                "status": "COMPLETE",
                "shard_index":
                    shard_index,
                "start_position":
                    start,
                "rows": len(records),
                "end_position_exclusive":
                    start + len(records),
                "checkpoint_sha256":
                    experiment_card[
                        "best_checkpoint"
                    ]["sha256"],
            },
        )

        atomic_json(
            status_path,
            {
                "status": "RUNNING",
                "stage": "inference",
                "processed_rows":
                    processed,
                "requested_rows":
                    args.limit,
                "batch_size":
                    args.batch_size,
                "truncated_rows":
                    truncated_count,
                "elapsed_seconds":
                    time.time() - started,
                "output_root":
                    str(output_root),
            },
        )

    for source_record in source_iterator:
        if not shard_records:
            shard_start = (
                source_record[
                    "source_position"
                ]
            )

        shard_records.append(
            source_record
        )

        if len(shard_records) == (
            args.shard_size
        ):
            write_shard(
                shard_records,
                shard_start,
            )

            shard_records = []

    if shard_records:
        write_shard(
            shard_records,
            shard_start,
        )

    assert processed == args.limit

    inference_seconds = (
        time.time() - inference_started
    )

    report = {
        "status": "PASSED",
        "stage": "complete",
        "run_name": args.run_name,
        "requested_rows":
            args.limit,
        "processed_rows":
            processed,
        "batch_size":
            args.batch_size,
        "maximum_length":
            maximum_length,
        "rank": rank,
        "best_epoch": 8,
        "checkpoint":
            str(checkpoint_path),
        "checkpoint_sha256":
            experiment_card[
                "best_checkpoint"
            ]["sha256"],
        "adapter":
            str(adapter_path),
        "dataset_counts":
            dict(
                sorted(
                    dataset_counts.items()
                )
            ),
        "truncated_rows":
            truncated_count,
        "observed_top1_classes": {
            field:
                len(observed_top1[field])
            for field in FIELDS
        },
        "observed_any_topk_classes": {
            field:
                len(
                    observed_any_topk[field]
                )
            for field in FIELDS
        },
        "maximum_probability_sum_error":
            probability_sum_error,
        "mean_maximum_probability": {
            field:
                max_probability_sum[field]
                / processed
            for field in FIELDS
        },
        "mean_entropy": {
            field:
                entropy_sum[field]
                / processed
            for field in FIELDS
        },
        "load_seconds":
            load_seconds,
        "inference_seconds":
            inference_seconds,
        "elapsed_seconds":
            time.time() - started,
        "rows_per_second":
            processed
            / max(
                inference_seconds,
                1e-9,
            ),
        "peak_gpu_memory_reserved_gib":
            torch.cuda
            .max_memory_reserved()
            / (1024 ** 3),
        "output_root":
            str(output_root),
        "checkpoint_tensors_loaded":
            checkpoint_load,
        "prompt_policy": {
            "question_template":
                QUESTION_TEMPLATE,
            "option_template":
                OPTION_TEMPLATE,
            "field_order":
                list(FIELDS),
            "source_text":
                "Exact LAYA flatten_messages",
        },
        "qwen_scores_used": False,
        "laya_predictions_used": False,
    }

    atomic_json(
        report_path,
        report,
    )

    atomic_json(
        status_path,
        {
            **report,
            "report":
                str(report_path),
        },
    )

    print("=" * 78)
    print(
        "PPLX EPOCH-8 CLUSTER "
        "ASSIGNMENT PASSED"
    )
    print("=" * 78)
    print("Rows:", processed)
    print(
        "Dataset counts:",
        report["dataset_counts"],
    )
    print(
        "Truncated rows:",
        truncated_count,
    )
    print(
        "Observed top-1 classes:",
        report[
            "observed_top1_classes"
        ],
    )
    print(
        "Observed anywhere in top-k:",
        report[
            "observed_any_topk_classes"
        ],
    )
    print(
        "Rows per second:",
        report["rows_per_second"],
    )
    print(
        "Peak GPU GiB:",
        report[
            "peak_gpu_memory_reserved_gib"
        ],
    )
    print("Output:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"FAILED: "
            f"{type(error).__name__}: "
            f"{error}",
            flush=True,
        )

        traceback.print_exc()
        raise
