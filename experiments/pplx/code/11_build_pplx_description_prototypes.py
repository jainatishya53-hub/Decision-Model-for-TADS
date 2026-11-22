#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


FIELDS = (
    "Task",
    "Topic",
    "Style",
    "Audience",
)

CLASSES = 100
HIDDEN_SIZE = 5120
MAXIMUM_LENGTH = 256
BATCH_SIZE = 16

QUESTION_TEMPLATE = (
    "Produce the semantic representation used "
    "to determine the {field} cluster."
)

OPTION_TEMPLATE = (
    "Semantic evidence relevant to the "
    "{field} cluster."
)


root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

checkpoint = Path(
    os.environ["PPLX_CHECKPOINT"]
).resolve()

catalogue_path = (
    root
    / "data"
    / "target_supervision"
    / "exact_laya_description_aware_v1"
    / "cluster_descriptions.json"
)

output_path = (
    root
    / "data"
    / "description_prototypes"
    / "pplx_base_description_prototypes.safetensors"
)

input_record_path = (
    root
    / "data"
    / "description_prototypes"
    / "pplx_description_prototype_inputs.json"
)

report_path = (
    root
    / "metadata"
    / "description_prototypes"
    / "11_pplx_description_prototypes.json"
)

status_path = (
    root
    / "data"
    / "description_prototypes"
    / "11_pplx_description_prototypes_status.json"
)

for path in (
    output_path,
    input_record_path,
    report_path,
    status_path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

assert checkpoint.is_dir(), (
    f"Missing checkpoint: {checkpoint}"
)

assert catalogue_path.is_file(), (
    f"Missing catalogue: {catalogue_path}"
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
from pplx_decider.protocol import left_pad


def save_json(
    path,
    value,
):
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


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def normalize_catalogue(
    raw_catalogue,
):
    if isinstance(raw_catalogue, dict):
        if "clusters" in raw_catalogue:
            raw_catalogue = raw_catalogue[
                "clusters"
            ]
        elif (
            "complete_cluster_catalog"
            in raw_catalogue
        ):
            raw_catalogue = raw_catalogue[
                "complete_cluster_catalog"
            ]

    if isinstance(raw_catalogue, list):
        return {
            record["cluster_id"]: record
            for record in raw_catalogue
        }

    if isinstance(raw_catalogue, dict):
        return raw_catalogue

    raise TypeError(
        "Unsupported catalogue structure"
    )


def get_description(
    record,
):
    description = (
        record.get("description")
        or record.get(
            "refined_description"
        )
    )

    assert isinstance(description, str)
    assert description.strip()

    return description.strip()


def get_keywords(
    record,
):
    keywords = record.get(
        "keywords",
        [],
    )

    if isinstance(keywords, str):
        keywords = [
            value.strip()
            for value in keywords.split("_")
            if value.strip()
        ]

    assert isinstance(keywords, list)

    return [
        str(value).strip()
        for value in keywords
        if str(value).strip()
    ]


def build_text(
    field,
    record,
):
    title = str(
        record.get(
            "title",
            "",
        )
    ).strip()

    description = get_description(
        record
    )

    keywords = get_keywords(
        record
    )

    return (
        f"Field: {field}. "
        f"Title: {title}. "
        f"Description: {description} "
        "Representative concepts: "
        f"{', '.join(keywords)}."
    )


def build_request(
    field,
    text,
):
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
        },
    }


started = time.time()

save_json(
    status_path,
    {
        "status": "STARTING",
        "stage": "loading",
        "checkpoint": str(checkpoint),
        "catalogue": str(
            catalogue_path
        ),
    },
)

try:
    with catalogue_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        catalogue = normalize_catalogue(
            json.load(handle)
        )

    expected_ids = {
        f"{field}_{index}"
        for field in FIELDS
        for index in range(CLASSES)
    }

    assert set(catalogue) == expected_ids, (
        "Catalogue must contain exactly "
        "400 expected cluster IDs"
    )

    prototype_inputs = []

    for field in FIELDS:
        for index in range(CLASSES):
            cluster_id = (
                f"{field}_{index}"
            )

            record = catalogue[
                cluster_id
            ]

            text = build_text(
                field,
                record,
            )

            prototype_inputs.append(
                {
                    "cluster_id":
                        cluster_id,
                    "field": field,
                    "index": index,
                    "text": text,
                    "question_instruction":
                        QUESTION_TEMPLATE.format(
                            field=field
                        ),
                    "question_option":
                        OPTION_TEMPLATE.format(
                            field=field
                        ),
                }
            )

    save_json(
        input_record_path,
        {
            "checkpoint":
                str(checkpoint),
            "maximum_length":
                MAXIMUM_LENGTH,
            "inputs":
                prototype_inputs,
        },
    )

    assert torch.cuda.is_available()

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Loading exact base PPLX checkpoint...",
        flush=True,
    )

    load_started = time.time()

    decider = Decider(
        checkpoint,
        device="cuda",
        fast=False,
        max_length=MAXIMUM_LENGTH,
        batch_tokens=4096,
        cpu_threads=8,
    )

    load_seconds = (
        time.time() - load_started
    )

    decider.backbone.eval()
    decider.backbone.requires_grad_(
        False
    )

    encoded_records = []
    token_lengths = []

    print(
        "Encoding 400 cluster descriptions...",
        flush=True,
    )

    for item in prototype_inputs:
        request = build_request(
            item["field"],
            item["text"],
        )

        plan = decider.encoder.plan(
            request,
            MAXIMUM_LENGTH,
        )

        assert len(plan) == 1, (
            f"Description unexpectedly split: "
            f"{item['cluster_id']}"
        )

        encoded = plan[0]

        assert len(encoded.markers) == 1
        assert encoded.keys == (
            item["field"],
        )

        token_lengths.append(
            encoded.length
        )

        encoded_records.append(
            (
                item,
                encoded,
            )
        )

    prototype_lists = {
        field: []
        for field in FIELDS
    }

    save_json(
        status_path,
        {
            "status": "RUNNING",
            "stage": "inference",
            "checkpoint":
                str(checkpoint),
            "processed": 0,
            "total": 400,
        },
    )

    inference_started = time.time()
    processed = 0

    with torch.inference_mode():
        for start in range(
            0,
            len(encoded_records),
            BATCH_SIZE,
        ):
            batch_records = (
                encoded_records[
                    start:
                    start + BATCH_SIZE
                ]
            )

            chunks = [
                encoded
                for _, encoded
                in batch_records
            ]

            padded = left_pad(
                chunks,
                decider.encoder.pad_id,
            )

            model_inputs = {
                key: value.to("cuda")
                for key, value
                in padded.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                output = decider.backbone(
                    **model_inputs,
                    use_cache=False,
                )

            hidden = (
                output.last_hidden_state
            )

            padded_length = (
                model_inputs[
                    "input_ids"
                ].shape[1]
            )

            rows = []
            columns = []

            for batch_index, chunk in enumerate(
                chunks
            ):
                offset = (
                    padded_length
                    - chunk.length
                )

                rows.append(
                    batch_index
                )

                columns.append(
                    int(
                        chunk.markers[0]
                    )
                    + offset
                )

            rows = torch.tensor(
                rows,
                device="cuda",
                dtype=torch.long,
            )

            columns = torch.tensor(
                columns,
                device="cuda",
                dtype=torch.long,
            )

            vectors = hidden[
                rows,
                columns,
            ].float()

            vectors = F.normalize(
                vectors,
                p=2,
                dim=-1,
            )

            assert tuple(
                vectors.shape
            ) == (
                len(batch_records),
                HIDDEN_SIZE,
            )

            vectors = vectors.cpu()

            for (
                item,
                _
            ), vector in zip(
                batch_records,
                vectors,
                strict=True,
            ):
                prototype_lists[
                    item["field"]
                ].append(
                    vector
                )

            processed += len(
                batch_records
            )

            print(
                f"processed={processed}/400",
                flush=True,
            )

            save_json(
                status_path,
                {
                    "status": "RUNNING",
                    "stage": "inference",
                    "processed": processed,
                    "total": 400,
                    "elapsed_seconds":
                        time.time()
                        - inference_started,
                },
            )

    prototypes = {}

    for field in FIELDS:
        tensor = torch.stack(
            prototype_lists[field],
            dim=0,
        ).contiguous()

        assert tuple(tensor.shape) == (
            CLASSES,
            HIDDEN_SIZE,
        )

        norms = torch.linalg.vector_norm(
            tensor,
            dim=-1,
        )

        assert torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
        )

        prototypes[
            f"description_prototype_{field}"
        ] = tensor

    temporary_output = (
        output_path.with_suffix(
            ".safetensors.tmp"
        )
    )

    save_file(
        prototypes,
        str(temporary_output),
        metadata={
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
            "representation":
                "official_question_marker",
            "normalized":
                "true",
        },
    )

    temporary_output.replace(
        output_path
    )

    torch.cuda.synchronize()

    inference_seconds = (
        time.time()
        - inference_started
    )

    peak_allocated_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    peak_reserved_gib = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    report = {
        "status": "PASSED",
        "checkpoint":
            str(checkpoint),
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
        "clusters": 400,
        "fields": {
            field: 100
            for field in FIELDS
        },
        "prototype_shapes": {
            key: list(value.shape)
            for key, value
            in prototypes.items()
        },
        "prototype_dtype":
            "float32",
        "prototype_normalization":
            "l2",
        "question_instruction_template":
            QUESTION_TEMPLATE,
        "question_option_template":
            OPTION_TEMPLATE,
        "maximum_length":
            MAXIMUM_LENGTH,
        "batch_size":
            BATCH_SIZE,
        "token_lengths": {
            "minimum":
                min(token_lengths),
            "maximum":
                max(token_lengths),
            "mean":
                statistics.mean(
                    token_lengths
                ),
            "median":
                statistics.median(
                    token_lengths
                ),
        },
        "load_seconds":
            load_seconds,
        "inference_seconds":
            inference_seconds,
        "elapsed_seconds":
            time.time() - started,
        "peak_allocated_gib":
            peak_allocated_gib,
        "peak_reserved_gib":
            peak_reserved_gib,
        "output":
            str(output_path),
        "output_sha256":
            sha256(output_path),
        "prototype_inputs":
            str(input_record_path),
        "prototype_inputs_sha256":
            sha256(
                input_record_path
            ),
    }

    save_json(
        report_path,
        report,
    )

    save_json(
        status_path,
        {
            **report,
            "stage": "complete",
            "report": str(
                report_path
            ),
        },
    )

    print("=" * 78)
    print(
        "PPLX DESCRIPTION PROTOTYPES "
        "BUILD PASSED"
    )
    print("=" * 78)
    print("Clusters: 400")
    print(
        "Shapes:",
        report["prototype_shapes"],
    )
    print(
        "Token lengths:",
        report["token_lengths"],
    )
    print(
        "Load seconds:",
        load_seconds,
    )
    print(
        "Inference seconds:",
        inference_seconds,
    )
    print(
        "Peak reserved GiB:",
        peak_reserved_gib,
    )
    print("Output:", output_path)
    print(
        "SHA256:",
        report["output_sha256"],
    )
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

    save_json(
        status_path,
        failure,
    )

    traceback.print_exc()
    raise
