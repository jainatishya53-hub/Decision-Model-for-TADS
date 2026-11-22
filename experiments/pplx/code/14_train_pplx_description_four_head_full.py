#!/usr/bin/env python3

from pathlib import Path
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
import traceback

import torch
from safetensors.torch import (
    load_file,
    save_file,
)

import pplx_training_components as C


root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

runtime_root = Path(
    os.environ.get(
        "PPLX_RUNTIME_ROOT",
        "/tmp/gpuuser2_a_pplx_decider_v1_2",
    )
).resolve()

config_path = (
    root
    / "configs"
    / "09_pplx_description_four_head_training.json"
)

prototype_path = (
    root
    / "data"
    / "description_prototypes"
    / "pplx_base_description_prototypes.safetensors"
)

cache_root = (
    root
    / "data"
    / "training_cache"
)

status_path = (
    root
    / "data"
    / "full_training"
    / "14_pplx_full_training_status.json"
)

report_path = (
    root
    / "metadata"
    / "full_training"
    / "14_pplx_full_training.json"
)

csv_path = (
    root
    / "metadata"
    / "full_training"
    / "14_pplx_epoch_metrics.csv"
)

for path in (
    status_path,
    report_path,
    csv_path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_json(path, value):
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


def load_all(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def save_trainable_checkpoint(
    model,
    path,
    metadata,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".safetensors.tmp"
    )

    tensors = {}

    for name, parameter in (
        model.named_parameters()
    ):
        if parameter.requires_grad:
            tensors[
                f"parameter::{name}"
            ] = (
                parameter
                .detach()
                .cpu()
                .contiguous()
            )

    for field in C.FIELDS:
        name = (
            f"description_prototype_"
            f"{field}"
        )

        tensors[
            f"buffer::{name}"
        ] = (
            getattr(model, name)
            .detach()
            .cpu()
            .contiguous()
        )

    save_file(
        tensors,
        str(temporary),
        metadata={
            key: str(value)
            for key, value
            in metadata.items()
        },
    )

    temporary.replace(path)

    del tensors
    gc.collect()


def load_trainable_checkpoint(
    model,
    path,
):
    tensors = load_file(
        str(path),
        device="cpu",
    )

    parameters = dict(
        model.named_parameters()
    )

    for key, value in tensors.items():
        if not key.startswith(
            "parameter::"
        ):
            continue

        name = key.split(
            "::",
            1,
        )[1]

        assert name in parameters

        with torch.no_grad():
            parameters[name].copy_(
                value.to(
                    device=parameters[
                        name
                    ].device,
                    dtype=parameters[
                        name
                    ].dtype,
                )
            )

    del tensors
    gc.collect()


def is_better(
    candidate,
    best,
):
    if best is None:
        return True

    candidate_loss = candidate[
        "validation"
    ]["total_loss"]

    best_loss = best[
        "validation"
    ]["total_loss"]

    if candidate_loss < (
        best_loss - 1e-10
    ):
        return True

    if abs(
        candidate_loss - best_loss
    ) <= 1e-10:
        return (
            candidate["validation"][
                "task_argmax_accuracy"
            ]
            > best["validation"][
                "task_argmax_accuracy"
            ]
        )

    return False


def write_csv(
    path,
    history,
):
    columns = [
        "epoch",
        "train_total_loss",
        "train_classification_loss",
        "train_description_loss",
        "validation_total_loss",
        "validation_classification_loss",
        "validation_description_loss",
        "validation_task_argmax_accuracy",
        "validation_task_soft_accuracy",
        "validation_topic_top3_mass",
        "validation_style_top3_mass",
        "validation_audience_top3_mass",
        "epoch_seconds",
        "peak_reserved_gib",
        "checkpoint",
        "best_after_epoch",
    ]

    temporary = path.with_suffix(
        ".csv.tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for item in history:
            validation = item[
                "validation"
            ]

            writer.writerow(
                {
                    "epoch":
                        item["epoch"],
                    "train_total_loss":
                        item["train"][
                            "total_loss"
                        ],
                    "train_classification_loss":
                        item["train"][
                            "classification_loss"
                        ],
                    "train_description_loss":
                        item["train"][
                            "description_loss"
                        ],
                    "validation_total_loss":
                        validation[
                            "total_loss"
                        ],
                    "validation_classification_loss":
                        validation[
                            "classification_loss"
                        ],
                    "validation_description_loss":
                        validation[
                            "description_loss"
                        ],
                    "validation_task_argmax_accuracy":
                        validation[
                            "task_argmax_accuracy"
                        ],
                    "validation_task_soft_accuracy":
                        validation[
                            "task_soft_accuracy"
                        ],
                    "validation_topic_top3_mass":
                        validation[
                            "top3_target_mass"
                        ]["Topic"],
                    "validation_style_top3_mass":
                        validation[
                            "top3_target_mass"
                        ]["Style"],
                    "validation_audience_top3_mass":
                        validation[
                            "top3_target_mass"
                        ]["Audience"],
                    "epoch_seconds":
                        item[
                            "epoch_seconds"
                        ],
                    "peak_reserved_gib":
                        item[
                            "peak_reserved_gib"
                        ],
                    "checkpoint":
                        item[
                            "checkpoint"
                        ],
                    "best_after_epoch":
                        item[
                            "best_after_epoch"
                        ],
                }
            )

    temporary.replace(path)


started = time.time()

save_json(
    status_path,
    {
        "status": "STARTING",
        "stage": "loading",
    },
)

try:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)

    experiment = config[
        "experiment"
    ]

    training = config["training"]
    adaptation = config["adaptation"]
    loss_config = config["loss"]

    epochs = int(
        training["epochs"]
    )

    rank = int(
        adaptation["selected_rank"]
    )

    maximum_length = int(
        training["maximum_length"]
    )

    micro_batch_size = int(
        training["micro_batch_size"]
    )

    gradient_accumulation = int(
        training[
            "gradient_accumulation_steps"
        ]
    )

    effective_batch_size = int(
        training["effective_batch_size"]
    )

    assert epochs == 10
    assert rank == 256
    assert maximum_length == 768
    assert micro_batch_size == 2
    assert gradient_accumulation == 32
    assert effective_batch_size == 64

    assert (
        micro_batch_size
        * gradient_accumulation
        == effective_batch_size
    )

    runtime_output = (
        runtime_root
        / "training_runs"
        / experiment
    )

    persistent_best_root = (
        root
        / "models"
        / experiment
        / "best"
    )

    runtime_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    persistent_best_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    free_runtime_gib = (
        shutil.disk_usage(
            runtime_output
        ).free
        / (1024 ** 3)
    )

    free_home_gib = (
        shutil.disk_usage(
            persistent_best_root
        ).free
        / (1024 ** 3)
    )

    assert free_runtime_gib >= 45.0, (
        "At least 45 GiB free space is "
        "required for ten epoch checkpoints"
    )

    assert free_home_gib >= 5.0, (
        "At least 5 GiB home storage is "
        "required for the best checkpoint"
    )

    train_records = load_all(
        cache_root
        / "train_pplx_tokens.jsonl"
    )

    validation_records = load_all(
        cache_root
        / "validation_pplx_tokens.jsonl"
    )

    test_records = load_all(
        cache_root
        / "test_pplx_tokens.jsonl"
    )

    assert len(train_records) == 3978
    assert len(validation_records) == 850
    assert len(test_records) == 858

    assert max(
        record["sequence_length"]
        for record in (
            train_records
            + validation_records
            + test_records
        )
    ) <= maximum_length

    prototypes = load_file(
        str(prototype_path),
        device="cpu",
    )

    assert torch.cuda.is_available()

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading exact base PPLX checkpoint...",
        flush=True,
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
        decider.backbone,
        "gradient_checkpointing_enable",
    ):
        decider.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False,
            }
        )

    if hasattr(
        decider.backbone,
        "enable_input_require_grads",
    ):
        decider.backbone.enable_input_require_grads()

    if hasattr(
        decider.backbone.config,
        "use_cache",
    ):
        decider.backbone.config.use_cache = False

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

    lora_parameters = []
    head_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if (
            "lora_A" in name
            or "lora_B" in name
        ):
            lora_parameters.append(
                parameter
            )
        else:
            head_parameters.append(
                parameter
            )

    lora_count = sum(
        parameter.numel()
        for parameter in lora_parameters
    )

    head_count = sum(
        parameter.numel()
        for parameter in head_parameters
    )

    total_trainable = (
        lora_count + head_count
    )

    assert lora_count == 1_740_636_160
    assert head_count == 2_058_640
    assert total_trainable == 1_742_694_800

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    lora_parameters,
                "lr":
                    training[
                        "lora_learning_rate"
                    ],
            },
            {
                "params":
                    head_parameters,
                "lr":
                    training[
                        "head_learning_rate"
                    ],
            },
        ],
        weight_decay=training[
            "weight_decay"
        ],
        fused=True,
    )

    updates_per_epoch = math.ceil(
        len(train_records)
        / effective_batch_size
    )

    total_updates = (
        updates_per_epoch
        * epochs
    )

    warmup_updates = max(
        1,
        math.ceil(
            total_updates
            * training["warmup_ratio"]
        ),
    )

    def lr_lambda(step):
        if step < warmup_updates:
            return (
                step + 1
            ) / warmup_updates

        return max(
            0.0,
            (
                total_updates - step
            )
            / max(
                1,
                total_updates
                - warmup_updates,
            ),
        )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
        )
    )

    print(
        "Running baseline validation...",
        flush=True,
    )

    baseline_validation = C.evaluate(
        model,
        validation_records,
        description_weight=loss_config[
            "description_loss_weight"
        ],
    )

    history = []
    best_record = None
    best_runtime_checkpoint = None
    persistent_best_checkpoint = (
        persistent_best_root
        / "model.safetensors"
    )

    global_update = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_started = time.time()

        shuffled = list(
            train_records
        )

        random.Random(
            42 + epoch
        ).shuffle(shuffled)

        model.train()

        train_total = 0.0
        train_classification = 0.0
        train_description = 0.0
        train_count = 0

        for group_start in range(
            0,
            len(shuffled),
            effective_batch_size,
        ):
            group = shuffled[
                group_start:
                group_start
                + effective_batch_size
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            micro_batch_count = math.ceil(
                len(group)
                / micro_batch_size
            )

            assert (
                micro_batch_count
                <= gradient_accumulation
            )

            for micro_start in range(
                0,
                len(group),
                micro_batch_size,
            ):
                micro_batch = group[
                    micro_start:
                    micro_start
                    + micro_batch_size
                ]

                targets = C.dense_targets(
                    micro_batch
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    (
                        logits,
                        description_logits,
                    ) = model(micro_batch)

                    (
                        loss,
                        classification,
                        description,
                    ) = C.calculate_loss(
                        logits,
                        description_logits,
                        targets,
                        loss_config[
                            "description_loss_weight"
                        ],
                    )

                    # calculate_loss() returns the mean
                    # across this micro-batch. Weight it by
                    # its sample count so the complete
                    # optimizer group remains an exact
                    # per-sample mean.
                    scaled_loss = (
                        loss
                        * len(micro_batch)
                        / len(group)
                    )

                assert torch.isfinite(loss)

                scaled_loss.backward()

                train_total += (
                    float(loss.detach())
                    * len(micro_batch)
                )

                train_classification += (
                    float(
                        classification.detach()
                    )
                    * len(micro_batch)
                )

                train_description += (
                    float(
                        description.detach()
                    )
                    * len(micro_batch)
                )

                train_count += len(
                    micro_batch
                )

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    [
                        *lora_parameters,
                        *head_parameters,
                    ],
                    max_norm=training[
                        "gradient_clip_norm"
                    ],
                )
            )

            assert torch.isfinite(
                gradient_norm
            )

            optimizer.step()
            scheduler.step()

            global_update += 1

            print(
                f"epoch={epoch}/{epochs} "
                f"update="
                f"{global_update}/"
                f"{total_updates} "
                f"rows="
                f"{train_count}/"
                f"{len(train_records)} "
                f"loss="
                f"{train_total / train_count:.6f}",
                flush=True,
            )

            save_json(
                status_path,
                {
                    "status": "RUNNING",
                    "stage": "training",
                    "epoch": epoch,
                    "epochs": epochs,
                    "global_update":
                        global_update,
                    "total_updates":
                        total_updates,
                    "processed_rows":
                        train_count,
                    "epoch_rows":
                        len(train_records),
                    "running_loss":
                        train_total
                        / train_count,
                    "elapsed_seconds":
                        time.time()
                        - started,
                },
            )

        print(
            f"Evaluating validation "
            f"after epoch {epoch}...",
            flush=True,
        )

        validation = C.evaluate(
            model,
            validation_records,
            description_weight=loss_config[
                "description_loss_weight"
            ],
        )

        torch.cuda.synchronize()

        epoch_directory = (
            runtime_output
            / f"epoch_{epoch:02d}"
        )

        checkpoint_path = (
            epoch_directory
            / "model.safetensors"
        )

        train_metrics = {
            "examples": train_count,
            "total_loss":
                train_total
                / train_count,
            "classification_loss":
                train_classification
                / train_count,
            "description_loss":
                train_description
                / train_count,
        }

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "epoch_seconds":
                time.time()
                - epoch_started,
            "global_update":
                global_update,
            "peak_reserved_gib":
                torch.cuda
                .max_memory_reserved()
                / (1024 ** 3),
            "checkpoint":
                str(checkpoint_path),
            "best_after_epoch": False,
        }

        candidate_is_best = is_better(
            epoch_record,
            best_record,
        )

        save_trainable_checkpoint(
            model,
            checkpoint_path,
            metadata={
                "experiment":
                    experiment,
                "epoch": epoch,
                "rank": rank,
                "maximum_length":
                    maximum_length,
                "validation_total_loss":
                    validation[
                        "total_loss"
                    ],
            },
        )

        if candidate_is_best:
            shutil.copy2(
                checkpoint_path,
                persistent_best_checkpoint,
            )

            best_record = epoch_record
            best_runtime_checkpoint = (
                checkpoint_path
            )

            epoch_record[
                "best_after_epoch"
            ] = True

            save_json(
                persistent_best_root
                / "selection.json",
                {
                    "best_epoch": epoch,
                    "validation":
                        validation,
                    "runtime_checkpoint":
                        str(
                            checkpoint_path
                        ),
                    "persistent_checkpoint":
                        str(
                            persistent_best_checkpoint
                        ),
                },
            )

        history.append(
            epoch_record
        )

        write_csv(
            csv_path,
            history,
        )

        save_json(
            report_path,
            {
                "status": "RUNNING",
                "experiment":
                    experiment,
                "history": history,
                "best_epoch":
                    best_record[
                        "epoch"
                    ],
                "best_validation":
                    best_record[
                        "validation"
                    ],
                "baseline_validation":
                    baseline_validation,
            },
        )

        print(
            f"Epoch {epoch} completed. "
            f"Validation loss="
            f"{validation['total_loss']:.6f}. "
            f"Best epoch="
            f"{best_record['epoch']}.",
            flush=True,
        )

    assert best_runtime_checkpoint
    assert persistent_best_checkpoint.is_file()

    print(
        "Loading best checkpoint for "
        "one-time test evaluation...",
        flush=True,
    )

    load_trainable_checkpoint(
        model,
        persistent_best_checkpoint,
    )

    test_metrics = C.evaluate(
        model,
        test_records,
        description_weight=loss_config[
            "description_loss_weight"
        ],
    )

    final_report = {
        "status": "COMPLETE",
        "experiment": experiment,
        "model":
            os.environ.get(
                "PPLX_MODEL_ID",
                "perplexity-ai/"
                "pplx-decider-v1.2",
            ),
        "revision":
            os.environ.get(
                "PPLX_MODEL_REVISION",
                "",
            ),
        "rank": rank,
        "maximum_length":
            maximum_length,
        "epochs": epochs,
        "trainable_parameters":
            total_trainable,
        "updates_per_epoch":
            updates_per_epoch,
        "total_updates":
            total_updates,
        "warmup_updates":
            warmup_updates,
        "baseline_validation":
            baseline_validation,
        "history": history,
        "best_epoch":
            best_record["epoch"],
        "best_validation":
            best_record[
                "validation"
            ],
        "test": test_metrics,
        "runtime_checkpoint_root":
            str(runtime_output),
        "best_checkpoint":
            str(
                persistent_best_checkpoint
            ),
        "best_checkpoint_sha256":
            sha256(
                persistent_best_checkpoint
            ),
        "epoch_metrics_csv":
            str(csv_path),
        "peak_reserved_gib":
            torch.cuda
            .max_memory_reserved()
            / (1024 ** 3),
        "elapsed_seconds":
            time.time() - started,
    }

    save_json(
        report_path,
        final_report,
    )

    save_json(
        status_path,
        {
            **final_report,
            "stage": "complete",
            "report": str(
                report_path
            ),
        },
    )

    print("=" * 78)
    print(
        "PPLX 10-EPOCH FULL TRAINING "
        "COMPLETE"
    )
    print("=" * 78)
    print(
        "Best epoch:",
        final_report["best_epoch"],
    )
    print(
        "Best validation loss:",
        final_report[
            "best_validation"
        ]["total_loss"],
    )
    print(
        "Test loss:",
        test_metrics[
            "total_loss"
        ],
    )
    print(
        "Best checkpoint:",
        persistent_best_checkpoint,
    )
    print("CSV:", csv_path)
    print("Report:", report_path)

except Exception as error:
    failure = {
        "status": "FAILED",
        "stage": "failed",
        "error_type":
            type(error).__name__,
        "error": str(error),
        "elapsed_seconds":
            time.time() - started,
    }

    try:
        failure[
            "peak_reserved_gib"
        ] = (
            torch.cuda.max_memory_reserved()
            / (1024 ** 3)
        )
    except Exception:
        pass

    save_json(
        status_path,
        failure,
    )

    traceback.print_exc()
    raise

finally:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
