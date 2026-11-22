#!/usr/bin/env python3

from pathlib import Path
import gc
import json
import math
import os
import random
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

TARGET_SUFFIXES = {
    "down_proj",
    "gate_proj",
    "up_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}

CLASSES = 100
HIDDEN_SIZE = 5120
TRAIN_LIMIT = 128
VALIDATION_LIMIT = 64
GRADIENT_ACCUMULATION = 64


root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

checkpoint = Path(
    os.environ["PPLX_CHECKPOINT"]
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

train_path = (
    root
    / "data"
    / "training_cache"
    / "train_pplx_tokens.jsonl"
)

validation_path = (
    root
    / "data"
    / "training_cache"
    / "validation_pplx_tokens.jsonl"
)

status_path = (
    root
    / "data"
    / "training_smoke"
    / "13_actual_training_smoke_status.json"
)

report_path = (
    root
    / "metadata"
    / "training_smoke"
    / "13_actual_training_smoke.json"
)

for path in (
    status_path,
    report_path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

for path in (
    config_path,
    prototype_path,
    train_path,
    validation_path,
):
    assert path.is_file(), (
        f"Missing required file: {path}"
    )

sys.path.insert(
    0,
    str(
        checkpoint
        / "source"
        / "src"
    ),
)

from pplx_decider.model import Decider


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


def load_jsonl(path, limit):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

            if len(records) >= limit:
                break

    assert len(records) == limit

    return records


class LoRALinear(nn.Module):

    def __init__(
        self,
        base,
        rank,
        alpha,
        dropout,
    ):
        super().__init__()

        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(
            dropout
        )

        self.base.requires_grad_(
            False
        )

        device = base.weight.device
        dtype = base.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=device,
                dtype=dtype,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

    def forward(self, inputs):
        base_output = self.base(
            inputs
        )

        adapted = F.linear(
            self.dropout(inputs),
            self.lora_A,
        )

        adapted = F.linear(
            adapted,
            self.lora_B,
        )

        return (
            base_output
            + self.scale * adapted
        )


def get_parent(root_module, name):
    components = name.split(".")

    parent = root_module

    for component in components[:-1]:
        parent = getattr(
            parent,
            component,
        )

    return parent, components[-1]


def install_lora(
    language_model,
    rank,
    alpha,
    dropout,
):
    candidates = []

    for name, module in list(
        language_model.named_modules()
    ):
        suffix = name.rsplit(
            ".",
            1,
        )[-1]

        if (
            suffix in TARGET_SUFFIXES
            and isinstance(module, nn.Linear)
        ):
            candidates.append(
                (name, module)
            )

    assert len(candidates) == 400, (
        f"Expected 400 modules, "
        f"found {len(candidates)}"
    )

    counts = {}

    for name, module in candidates:
        suffix = name.rsplit(
            ".",
            1,
        )[-1]

        counts[suffix] = (
            counts.get(suffix, 0)
            + 1
        )

        parent, attribute = (
            get_parent(
                language_model,
                name,
            )
        )

        setattr(
            parent,
            attribute,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )

    return counts


class PPLXFourHeadModel(nn.Module):

    def __init__(
        self,
        backbone,
        prototypes,
        dropout,
        description_scale,
    ):
        super().__init__()

        self.backbone = backbone

        self.normalization = nn.LayerNorm(
            HIDDEN_SIZE,
            device="cuda",
            dtype=torch.bfloat16,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.heads = nn.ModuleDict(
            {
                field: nn.Linear(
                    HIDDEN_SIZE,
                    CLASSES,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for field in FIELDS
            }
        )

        self.description_scale = float(
            description_scale
        )

        initialization_norm = math.sqrt(
            2.0
            * HIDDEN_SIZE
            / (
                HIDDEN_SIZE
                + CLASSES
            )
        )

        for field in FIELDS:
            key = (
                f"description_prototype_"
                f"{field}"
            )

            prototype = (
                prototypes[key]
                .detach()
                .float()
                .to("cuda")
            )

            assert tuple(
                prototype.shape
            ) == (
                CLASSES,
                HIDDEN_SIZE,
            )

            self.register_buffer(
                key,
                prototype,
                persistent=True,
            )

            with torch.no_grad():
                self.heads[
                    field
                ].weight.copy_(
                    (
                        prototype
                        * initialization_norm
                    ).to(
                        torch.bfloat16
                    )
                )

                self.heads[
                    field
                ].bias.zero_()

    def forward(self, records):
        if isinstance(records, dict):
            records = [records]

        assert isinstance(records, (list, tuple))
        assert len(records) >= 1

        batch_size = len(records)

        lengths = [
            len(record["input_ids"])
            for record in records
        ]

        maximum_length = max(lengths)

        input_ids = torch.zeros(
            batch_size,
            maximum_length,
            device="cuda",
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            batch_size,
            maximum_length,
            device="cuda",
            dtype=torch.long,
        )

        marker_rows = []

        for batch_index, record in enumerate(records):
            record_ids = torch.tensor(
                record["input_ids"],
                device="cuda",
                dtype=torch.long,
            )

            record_length = record_ids.numel()

            input_ids[
                batch_index,
                :record_length,
            ] = record_ids

            attention_mask[
                batch_index,
                :record_length,
            ] = 1

            markers = record["markers"]

            assert len(markers) == len(FIELDS)
            assert min(markers) >= 0
            assert max(markers) < record_length

            marker_rows.append(markers)

        marker_positions = torch.tensor(
            marker_rows,
            device="cuda",
            dtype=torch.long,
        )

        assert tuple(marker_positions.shape) == (
            batch_size,
            len(FIELDS),
        )

        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        batch_positions = torch.arange(
            batch_size,
            device="cuda",
            dtype=torch.long,
        )[:, None]

        raw = output.last_hidden_state[
            batch_positions,
            marker_positions,
        ]

        assert tuple(raw.shape) == (
            batch_size,
            len(FIELDS),
            HIDDEN_SIZE,
        )

        normalized = self.normalization(raw)
        dropped = self.dropout(normalized)

        logits = {}
        description_logits = {}

        for position, field in enumerate(FIELDS):
            field_representation = dropped[
                :,
                position,
                :,
            ]

            raw_field = raw[
                :,
                position,
                :,
            ]

            logits[field] = self.heads[
                field
            ](
                field_representation
            )

            semantic = F.normalize(
                raw_field.float(),
                p=2,
                dim=-1,
            )

            prototype = getattr(
                self,
                f"description_prototype_{field}",
            )

            description_logits[field] = (
                self.description_scale
                * semantic
                @ prototype.transpose(0, 1)
            )

            assert tuple(logits[field].shape) == (
                batch_size,
                CLASSES,
            )

            assert tuple(
                description_logits[field].shape
            ) == (
                batch_size,
                CLASSES,
            )

        return logits, description_logits


def dense_targets(records):
    if isinstance(records, dict):
        records = [records]

    assert isinstance(records, (list, tuple))
    assert len(records) >= 1

    batch_size = len(records)
    targets = {}

    for field in FIELDS:
        vector = torch.zeros(
            batch_size,
            CLASSES,
            device="cuda",
            dtype=torch.float32,
        )

        for batch_index, record in enumerate(records):
            target = record["targets"][field]

            indices = torch.tensor(
                target["indices"],
                device="cuda",
                dtype=torch.long,
            )

            probabilities = torch.tensor(
                target["probabilities"],
                device="cuda",
                dtype=torch.float32,
            )

            assert indices.numel() == probabilities.numel()

            vector[
                batch_index,
                indices,
            ] = probabilities

            assert torch.isclose(
                vector[batch_index].sum(),
                torch.tensor(
                    1.0,
                    device="cuda",
                    dtype=torch.float32,
                ),
                atol=1e-5,
                rtol=1e-5,
            )

        targets[field] = vector

    return targets


def calculate_loss(
    logits,
    description_logits,
    targets,
    description_weight,
):
    classification_fields = {}
    description_fields = {}

    for field in FIELDS:
        classification_fields[field] = -(
            targets[field]
            * F.log_softmax(
                logits[field].float(),
                dim=-1,
            )
        ).sum(dim=-1).mean()

        description_fields[field] = -(
            targets[field]
            * F.log_softmax(
                description_logits[
                    field
                ].float(),
                dim=-1,
            )
        ).sum(dim=-1).mean()

    classification = torch.stack(
        list(
            classification_fields.values()
        )
    ).mean()

    description = torch.stack(
        list(
            description_fields.values()
        )
    ).mean()

    total = (
        classification
        + description_weight
        * description
    )

    return (
        total,
        classification,
        description,
    )


@torch.inference_mode()
def evaluate(
    model,
    records,
    description_weight,
):
    model.eval()

    total_loss = 0.0
    classification_loss = 0.0
    description_loss = 0.0

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

    for index, record in enumerate(
        records
    ):
        targets = dense_targets(
            record
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            (
                logits,
                description_logits,
            ) = model(record)

            (
                loss,
                classification,
                description,
            ) = calculate_loss(
                logits,
                description_logits,
                targets,
                description_weight,
            )

        total_loss += float(loss)
        classification_loss += float(
            classification
        )
        description_loss += float(
            description
        )

        for field in FIELDS:
            probability = torch.softmax(
                logits[field].float(),
                dim=-1,
            )

            brier[field] += float(
                (
                    (
                        probability
                        - targets[field]
                    )
                    ** 2
                ).sum()
            )

        task_target = targets[
            "Task"
        ]

        task_prediction = logits[
            "Task"
        ].argmax(dim=-1)

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
                targets[field]
                .gather(
                    1,
                    top3,
                )
                .sum()
            )

        if (
            (index + 1) % 16 == 0
            or index + 1 == len(records)
        ):
            print(
                f"validation="
                f"{index + 1}/"
                f"{len(records)}",
                flush=True,
            )

    count = len(records)

    return {
        "examples": count,
        "total_loss":
            total_loss / count,
        "classification_loss":
            classification_loss / count,
        "description_loss":
            description_loss / count,
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


