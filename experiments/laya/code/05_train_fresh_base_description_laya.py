#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import laya
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)


FIELDS = ("Task", "Topic", "Style", "Audience")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    temporary.replace(path)


def load_rows(path):
    with Path(path).open() as handle:
        rows = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]

    if not rows:
        raise ValueError(f"Empty split: {path}")

    return rows


def target_vector(row, field, classes):
    sparse = row["targets"][field]

    indices = sparse["indices"]
    probabilities = sparse["probabilities"]

    if len(indices) != len(probabilities):
        raise ValueError(
            f"Bad sparse target: "
            f"{row['conversation_id']} {field}"
        )

    vector = np.zeros(
        classes,
        dtype=np.float32,
    )

    for index, probability in zip(
        indices,
        probabilities,
    ):
        index = int(index)
        probability = float(probability)

        if not 0 <= index < classes:
            raise ValueError(
                f"Bad class index: {field} {index}"
            )

        if probability < 0:
            raise ValueError(
                f"Negative probability: "
                f"{field} {probability}"
            )

        vector[index] += probability

    total = float(vector.sum())

    if not np.isclose(
        total,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"Target does not sum to one: "
            f"{field} {total}"
        )

    return vector / total



def build_description_prototypes(
    description_path,
    tokenizer,
    encoder,
    classes,
    batch_size,
    max_length,
    device,
):
    payload = json.loads(
        Path(description_path).read_text()
    )

    catalogue = payload["clusters"]

    expected = {
        f"{field}_{index}"
        for field in FIELDS
        for index in range(classes)
    }

    if set(catalogue) != expected:
        raise ValueError(
            "Description catalogue does not "
            "contain the expected class IDs"
        )

    previous_training_state = (
        encoder.training
    )

    encoder.eval()

    prototypes = {}

    with torch.no_grad():
        for field in FIELDS:
            texts = []

            for index in range(classes):
                cluster_id = (
                    f"{field}_{index}"
                )

                record = catalogue[
                    cluster_id
                ]

                text = (
                    f"Field: {field}. "
                    f"Title: {record['title']}. "
                    f"Description: "
                    f"{record['description']} "
                    "Representative concepts: "
                    f"{', '.join(record['keywords'])}."
                )

                texts.append(text)

            field_vectors = []

            for start in range(
                0,
                len(texts),
                batch_size,
            ):
                batch_texts = texts[
                    start:
                    start + batch_size
                ]

                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

                input_ids = encoded[
                    "input_ids"
                ].to(device)

                attention_mask = encoded[
                    "attention_mask"
                ].to(device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    output = encoder(
                        input_ids=input_ids,
                        attention_mask=
                            attention_mask,
                    )

                vectors = (
                    output
                    .last_hidden_state[:, 0]
                    .float()
                )

                vectors = F.normalize(
                    vectors,
                    p=2,
                    dim=-1,
                )

                field_vectors.append(
                    vectors.detach()
                )

            prototypes[field] = torch.cat(
                field_vectors,
                dim=0,
            )

            if prototypes[field].shape[0] != classes:
                raise RuntimeError(
                    f"Incorrect prototype count "
                    f"for {field}"
                )

    if previous_training_state:
        encoder.train()

    print(
        "Description prototypes:",
        {
            field:
                tuple(value.shape)
            for field, value
            in prototypes.items()
        },
        flush=True,
    )

    return prototypes


class Records(Dataset):

    def __init__(
        self,
        rows,
        tokenizer,
        max_length,
        classes,
    ):
        encoded = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        self.items = []

        for index, row in enumerate(rows):
            self.items.append(
                {
                    "input_ids":
                        encoded["input_ids"][index],
                    "attention_mask":
                        encoded["attention_mask"][index],
                    "labels": {
                        field: target_vector(
                            row,
                            field,
                            classes,
                        )
                        for field in FIELDS
                    },
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Collator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items):
        padded = self.tokenizer.pad(
            {
                "input_ids": [
                    item["input_ids"]
                    for item in items
                ],
                "attention_mask": [
                    item["attention_mask"]
                    for item in items
                ],
            },
            return_tensors="pt",
        )

        padded["labels"] = {
            field: torch.tensor(
                np.stack(
                    [
                        item["labels"][field]
                        for item in items
                    ]
                ),
                dtype=torch.float32,
            )
            for field in FIELDS
        }

        return padded


class Classifier(nn.Module):

    def __init__(
        self,
        encoder,
        hidden_size,
        classes,
        dropout,
        description_prototypes,
        description_loss_weight,
        description_logit_scale,
    ):
        super().__init__()

        self.encoder = encoder

        self.normalization = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.heads = nn.ModuleDict(
            {
                field: nn.Linear(
                    hidden_size,
                    classes,
                )
                for field in FIELDS
            }
        )

        self.description_loss_weight = float(
            description_loss_weight
        )

        self.description_logit_scale = float(
            description_logit_scale
        )

        expected_shape = (
            classes,
            hidden_size,
        )

        initialization_norm = math.sqrt(
            2.0
            * hidden_size
            / (hidden_size + classes)
        )

        for field in FIELDS:
            prototype = (
                description_prototypes[field]
                .detach()
                .float()
            )

            if tuple(prototype.shape) != (
                expected_shape
            ):
                raise ValueError(
                    f"Bad prototype shape for "
                    f"{field}: "
                    f"{tuple(prototype.shape)}"
                )

            self.register_buffer(
                f"description_prototype_"
                f"{field}",
                prototype,
                persistent=True,
            )

            with torch.no_grad():
                self.heads[field].weight.copy_(
                    prototype
                    * initialization_norm
                )

                nn.init.zeros_(
                    self.heads[field].bias
                )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        raw_representation = (
            output.last_hidden_state[:, 0]
        )

        representation = self.normalization(
            raw_representation
        )

        dropped_representation = self.dropout(
            representation
        )

        logits = {
            field: self.heads[field](
                dropped_representation
            )
            for field in FIELDS
        }

        semantic_representation = F.normalize(
            raw_representation.float(),
            p=2,
            dim=-1,
        )

        description_logits = {}

        for field in FIELDS:
            prototype = getattr(
                self,
                f"description_prototype_"
                f"{field}",
            )

            description_logits[field] = (
                self.description_logit_scale
                * semantic_representation
                @ prototype.float().transpose(
                    0,
                    1,
                )
            )

        logits[
            "__description_logits__"
        ] = description_logits

        logits[
            "__description_loss_weight__"
        ] = self.description_loss_weight

        return logits


def calculate_loss(
    logits,
    labels,
):
    classification_field_losses = {
        field: -(
            labels[field]
            * F.log_softmax(
                logits[field],
                dim=-1,
            )
        )
        .sum(dim=-1)
        .mean()
        for field in FIELDS
    }

    classification_loss = torch.stack(
        list(
            classification_field_losses.values()
        )
    ).mean()

    if "__description_logits__" not in logits:
        return (
            classification_loss,
            {
                "classification":
                    classification_field_losses,
            },
        )

    description_logits = logits[
        "__description_logits__"
    ]

    description_field_losses = {
        field: -(
            labels[field]
            * F.log_softmax(
                description_logits[field],
                dim=-1,
            )
        )
        .sum(dim=-1)
        .mean()
        for field in FIELDS
    }

    description_loss = torch.stack(
        list(
            description_field_losses.values()
        )
    ).mean()

    description_weight = float(
        logits[
            "__description_loss_weight__"
        ]
    )

    total_loss = (
        classification_loss
        + description_weight
        * description_loss
    )

    return (
        total_loss,
        {
            "classification":
                classification_field_losses,
            "description":
                description_field_losses,
            "classification_mean":
                classification_loss,
            "description_mean":
                description_loss,
        },
    )


def to_device(
    batch,
    device,
):
    return {
        "input_ids":
            batch["input_ids"].to(
                device,
                non_blocking=True,
            ),
        "attention_mask":
            batch["attention_mask"].to(
                device,
                non_blocking=True,
            ),
        "labels": {
            field: batch["labels"][field].to(
                device,
                non_blocking=True,
            )
            for field in FIELDS
        },
    }


def evaluate(
    model,
    loader,
    device,
    limit_batches=None,
):
    model.eval()

    count = 0
    loss_sum = 0.0

    task_correct = 0.0
    task_soft = 0.0

    top3_mass = {
        field: 0.0
        for field in FIELDS[1:]
    }

    brier = {
        field: 0.0
        for field in FIELDS
    }

    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(
            loader
        ):
            if (
                limit_batches is not None
                and batch_index >= limit_batches
            ):
                break

            batch = to_device(
                raw_batch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                )

                loss, _ = calculate_loss(
                    logits,
                    batch["labels"],
                )

            size = batch["input_ids"].shape[0]

            count += size
            loss_sum += float(loss) * size

            for field in FIELDS:
                probability = torch.softmax(
                    logits[field].float(),
                    dim=-1,
                )

                brier[field] += float(
                    (
                        (
                            probability
                            - batch["labels"][field]
                        )
                        ** 2
                    ).sum()
                )

            task_target = (
                batch["labels"]["Task"]
            )

            task_prediction = (
                logits["Task"].argmax(
                    dim=-1
                )
            )

            selected_mass = (
                task_target.gather(
                    1,
                    task_prediction[:, None],
                ).squeeze(1)
            )

            task_soft += float(
                selected_mass.sum()
            )

            maximum = task_target.max(
                dim=-1
            ).values

            task_correct += float(
                (
                    selected_mass
                    >= maximum - 1e-8
                ).sum()
            )

            for field in FIELDS[1:]:
                top3 = logits[field].topk(
                    3,
                    dim=-1,
                ).indices

                top3_mass[field] += float(
                    batch["labels"][field]
                    .gather(
                        1,
                        top3,
                    )
                    .sum()
                )

    return {
        "examples": count,
        "loss": loss_sum / count,
        "task_argmax_accuracy":
            task_correct / count,
        "task_soft_accuracy":
            task_soft / count,
        "top3_target_mass": {
            field:
                top3_mass[field] / count
            for field in FIELDS[1:]
        },
        "brier": {
            field:
                brier[field] / count
            for field in FIELDS
        },
    }


def save_weights(
    model,
    directory,
):
    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    temporary = (
        directory
        / "model.safetensors.tmp"
    )

    final = (
        directory
        / "model.safetensors"
    )

    state = {
        key:
            value.detach()
            .cpu()
            .contiguous()
        for key, value
        in model.state_dict().items()
    }

    save_file(
        state,
        temporary,
    )

    temporary.replace(final)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    config = json.loads(
        args.config.read_text()
    )

    seed = int(
        config["seed"]
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision(
        "high"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is unavailable"
        )

    device = torch.device("cuda")

    output = Path(
        config["output_dir"]
    )

    status = Path(
        config["status_path"]
    )

    if (
        not args.smoke_test
        and output.exists()
    ):
        raise FileExistsError(
            "Refusing to overwrite existing run: "
            f"{output}"
        )

    if not args.smoke_test:
        output.mkdir(
            parents=True
        )

        atomic_json(
            output / "training_config.json",
            config,
        )

        atomic_json(
            status,
            {
                "status": "STARTING",
                "stage": "loading",
            },
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            Path(
                config["base_model_dir"]
            )
            / "tokenizer",
            local_files_only=True,
        )
    )

    datasets = {
        name: Records(
            load_rows(
                config["splits"][name]
            ),
            tokenizer,
            int(config["max_length"]),
            int(config["num_classes"]),
        )
        for name in (
            "train",
            "validation",
            "test",
        )
    }

    collator = Collator(
        tokenizer
    )

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(
            config["micro_batch_size"]
        ),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    evaluation_loaders = {
        name: DataLoader(
            datasets[name],
            batch_size=int(
                config[
                    "evaluation_batch_size"
                ]
            ),
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        for name in (
            "validation",
            "test",
        )
    }

    print(
        "Loading pretrained "
        "Laya checkpoint on CPU...",
        flush=True,
    )

    agent = laya.load(
        config["base_model_dir"],
        device="cpu",
    )

    encoder = agent.model.encoder

    try:
        encoder.config.reference_compile = (
            False
        )
    except Exception:
        pass

    encoder = encoder.to(device)

    description_prototypes = (
        build_description_prototypes(
            description_path=config[
                "description_catalogue"
            ],
            tokenizer=tokenizer,
            encoder=encoder,
            classes=int(
                config["num_classes"]
            ),
            batch_size=int(
                config[
                    "description_batch_size"
                ]
            ),
            max_length=int(
                config[
                    "description_max_length"
                ]
            ),
            device=device,
        )
    )

    model = Classifier(
        encoder=encoder,
        hidden_size=int(
            encoder.config.hidden_size
        ),
        classes=int(
            config["num_classes"]
        ),
        dropout=float(
            config["dropout"]
        ),
        description_prototypes=
            description_prototypes,
        description_loss_weight=float(
            config[
                "description_loss_weight"
            ]
        ),
        description_logit_scale=float(
            config[
                "description_logit_scale"
            ]
        ),
    ).to(device)

    del agent

    encoder_parameters = list(
        model.encoder.parameters()
    )

    other_parameters = [
        parameter
        for name, parameter
        in model.named_parameters()
        if not name.startswith(
            "encoder."
        )
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    encoder_parameters,
                "lr": float(
                    config[
                        "encoder_learning_rate"
                    ]
                ),
            },
            {
                "params":
                    other_parameters,
                "lr": float(
                    config[
                        "head_learning_rate"
                    ]
                ),
            },
        ],
        weight_decay=float(
            config["weight_decay"]
        ),
    )

    accumulation = int(
        config[
            "gradient_accumulation_steps"
        ]
    )

    epochs = (
        1
        if args.smoke_test
        else int(config["epochs"])
    )

    updates_per_epoch = math.ceil(
        len(train_loader)
        / accumulation
    )

    total_updates = (
        updates_per_epoch
        * epochs
    )

    warmup_steps = round(
        total_updates
        * float(config["warmup_ratio"])
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                total_updates,
        )
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    print(
        "Examples:",
        {
            key: len(value)
            for key, value
            in datasets.items()
        },
        flush=True,
    )

    print(
        "Epochs:",
        epochs,
        flush=True,
    )

    print(
        "Micro batch:",
        config["micro_batch_size"],
        flush=True,
    )

    print(
        "Gradient accumulation:",
        accumulation,
        flush=True,
    )

    print(
        "Total updates:",
        total_updates,
        flush=True,
    )

    print(
        "Parameters:",
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        ),
        flush=True,
    )

    print(
        "Description supervision:",
        {
            "catalogue":
                config[
                    "description_catalogue"
                ],
            "loss_weight":
                config[
                    "description_loss_weight"
                ],
            "logit_scale":
                config[
                    "description_logit_scale"
                ],
            "description_max_length":
                config[
                    "description_max_length"
                ],
        },
        flush=True,
    )

    started = time.time()

    best_loss = float("inf")
    best_path = None
    update = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        epoch_loss = 0.0
        examples = 0

        for (
            batch_number,
            raw_batch,
        ) in enumerate(
            train_loader,
            start=1,
        ):
            batch = to_device(
                raw_batch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                )

                loss, _ = calculate_loss(
                    logits,
                    batch["labels"],
                )

            (
                loss / accumulation
            ).backward()

            size = (
                batch["input_ids"]
                .shape[0]
            )

            epoch_loss += (
                float(loss) * size
            )

            examples += size

            should_update = (
                batch_number
                % accumulation
                == 0
                or batch_number
                == len(train_loader)
            )

            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(
                        config[
                            "gradient_clip_norm"
                        ]
                    ),
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                update += 1

                if (
                    update
                    % int(
                        config[
                            "logging_updates"
                        ]
                    )
                    == 0
                ):
                    print(
                        f"epoch={epoch}/{epochs} "
                        f"update={update}/"
                        f"{total_updates} "
                        f"loss="
                        f"{epoch_loss/examples:.6f} "
                        f"encoder_lr="
                        f"{optimizer.param_groups[0]['lr']:.8g} "
                        f"head_lr="
                        f"{optimizer.param_groups[1]['lr']:.8g}",
                        flush=True,
                    )

            if args.smoke_test:
                break

        validation = evaluate(
            model,
            evaluation_loaders[
                "validation"
            ],
            device,
            limit_batches=(
                1
                if args.smoke_test
                else None
            ),
        )

        metrics = {
            "epoch": epoch,
            "training_loss":
                epoch_loss / examples,
            "validation":
                validation,
            "elapsed_seconds":
                time.time() - started,
            "peak_gpu_memory_gib":
                torch.cuda
                .max_memory_allocated()
                / (1024 ** 3),
        }

        if args.smoke_test:
            print(
                json.dumps(
                    metrics,
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )

            print("=" * 72)
            print(
                "LAYA FOUR-HEAD TRAINING "
                "SMOKE TEST PASSED"
            )
            print("=" * 72)
            return

        epoch_directory = (
            output
            / f"epoch_{epoch}"
        )

        save_weights(
            model,
            epoch_directory,
        )

        atomic_json(
            epoch_directory
            / "metrics.json",
            metrics,
        )

        print(
            json.dumps(
                metrics,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )

        if (
            validation["loss"]
            < best_loss
        ):
            best_loss = (
                validation["loss"]
            )

            best_path = (
                epoch_directory
                / "model.safetensors"
            )

        atomic_json(
            status,
            {
                "status": "RUNNING",
                "stage": "training",
                "completed_epoch": epoch,
                "total_epochs": epochs,
                "update": update,
                "total_updates":
                    total_updates,
                "best_validation_loss":
                    best_loss,
                "best_checkpoint":
                    str(best_path),
                "elapsed_seconds":
                    time.time() - started,
            },
        )

    if best_path is None:
        raise RuntimeError(
            "No checkpoint was produced"
        )

    model.load_state_dict(
        load_file(
            str(best_path),
            device="cpu",
        ),
        strict=True,
    )

    test_metrics = evaluate(
        model,
        evaluation_loaders["test"],
        device,
    )

    final_report = {
        "status": "COMPLETE",
        "stage": "complete",
        "best_checkpoint":
            str(best_path),
        "best_validation_loss":
            best_loss,
        "test": test_metrics,
        "elapsed_seconds":
            time.time() - started,
    }

    atomic_json(
        output
        / "training_complete.json",
        final_report,
    )

    atomic_json(
        status,
        final_report,
    )

    print("=" * 72)
    print(
        "LAYA FOUR-HEAD "
        "TRAINING COMPLETED"
    )
    print("=" * 72)

    print(
        json.dumps(
            final_report,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
