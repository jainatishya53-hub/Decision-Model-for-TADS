#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import time

import laya
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer


FIELDS = ("Task", "Topic", "Style", "Audience")
TOP_K = {
    "Task": 1,
    "Topic": 3,
    "Style": 3,
    "Audience": 3,
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def import_training_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "laya_anchor_training",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten_messages(messages):
    parts = []

    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()

        if not role and not content:
            continue

        parts.append(f"{role}: {content}")

    return "\n".join(parts)


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def output_schema(classes):
    fields = [
        pa.field("source_position", pa.int64()),
        pa.field("dataset", pa.string()),
        pa.field("id", pa.string()),
        pa.field("text_sha256", pa.string()),
    ]

    for field in FIELDS:
        k = TOP_K[field]

        fields.extend(
            [
                pa.field(
                    f"{field}_indices",
                    pa.list_(pa.int16(), k),
                ),
                pa.field(
                    f"{field}_probabilities",
                    pa.list_(pa.float32(), k),
                ),
                pa.field(
                    f"{field}_distribution",
                    pa.list_(pa.float32(), classes),
                ),
                pa.field(
                    f"{field}_max_probability",
                    pa.float32(),
                ),
                pa.field(
                    f"{field}_entropy",
                    pa.float32(),
                ),
            ]
        )

    return pa.schema(fields)


def load_temperatures(path):
    value = json.loads(Path(path).read_text())

    temperatures = value.get(
        "temperatures",
        value,
    )

    result = {
        field: float(temperatures[field])
        for field in FIELDS
    }

    for field, temperature in result.items():
        assert np.isfinite(temperature)
        assert temperature > 0.0, (
            field,
            temperature,
        )

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source-parquet",
        required=True,
    )
    parser.add_argument(
        "--training-program",
        required=True,
    )
    parser.add_argument(
        "--training-config",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--calibration",
        required=True,
    )
    parser.add_argument(
        "--output-parquet",
        required=True,
    )
    parser.add_argument(
        "--report",
        required=True,
    )
    parser.add_argument(
        "--status",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    source_path = Path(args.source_parquet).resolve()
    training_program = Path(
        args.training_program
    ).resolve()
    training_config_path = Path(
        args.training_config
    ).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    calibration_path = Path(
        args.calibration
    ).resolve()
    output_path = Path(args.output_parquet).resolve()
    report_path = Path(args.report).resolve()
    status_path = Path(args.status).resolve()

    for required in (
        source_path,
        training_program,
        training_config_path,
        checkpoint,
        calibration_path,
    ):
        assert required.is_file(), (
            f"Missing input: {required}"
        )

    assert not output_path.exists(), (
        f"Output already exists: {output_path}"
    )

    partial_path = output_path.with_suffix(
        output_path.suffix + ".partial"
    )

    assert not partial_path.exists(), (
        f"Partial output already exists: {partial_path}"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    status_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = json.loads(
        training_config_path.read_text()
    )

    classes = int(config["num_classes"])
    max_length = int(config["max_length"])
    dropout = float(config["dropout"])

    assert classes == 100
    assert max_length == 512

    temperatures = load_temperatures(
        calibration_path
    )

    parquet = pq.ParquetFile(source_path)
    total_source_rows = parquet.metadata.num_rows

    requested_rows = total_source_rows

    if args.limit is not None:
        assert args.limit > 0
        requested_rows = min(
            requested_rows,
            args.limit,
        )

    status = {
        "status": "STARTING",
        "stage": "loading",
        "processed_rows": 0,
        "requested_rows": requested_rows,
        "source_rows": total_source_rows,
        "checkpoint": str(checkpoint),
        "output": str(output_path),
    }
    atomic_json(status_path, status)

    print("Loading tokenizer and model...")
    print("Checkpoint:", checkpoint)
    print("Requested rows:", requested_rows)
    print("Batch size:", args.batch_size)
    print("Temperatures:", temperatures)

    training = import_training_module(
        training_program
    )

    tokenizer = AutoTokenizer.from_pretrained(
        Path(config["base_model_dir"])
        / "tokenizer",
        local_files_only=True,
    )

    agent = laya.load(
        config["base_model_dir"],
        device="cpu",
    )

    encoder = agent.model.encoder

    try:
        encoder.config.reference_compile = False
    except Exception:
        pass

    checkpoint_state = load_file(
        str(checkpoint),
        device="cpu",
    )

    description_prototypes = {
        field: checkpoint_state[
            f"description_prototype_{field}"
        ].clone()
        for field in FIELDS
    }

    model = training.Classifier(
        encoder=encoder,
        hidden_size=int(
            encoder.config.hidden_size
        ),
        classes=classes,
        dropout=dropout,
        description_prototypes=description_prototypes,
        description_loss_weight=float(
            config["description_loss_weight"]
        ),
        description_logit_scale=float(
            config["description_logit_scale"]
        ),
    )

    del agent

    model.load_state_dict(
        checkpoint_state,
        strict=True,
    )

    del checkpoint_state
    del description_prototypes

    device = torch.device("cuda")

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1

    model.to(device)
    model.eval()

    use_bf16 = torch.cuda.is_bf16_supported()

    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 inference:", use_bf16)

    schema = output_schema(classes)

    writer = pq.ParquetWriter(
        partial_path,
        schema,
        compression="zstd",
        use_dictionary=True,
    )

    coverage = {
        field: np.zeros(
            classes,
            dtype=np.int64,
        )
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
    batch_number = 0
    started = time.time()

    try:
        batches = parquet.iter_batches(
            batch_size=args.batch_size,
            columns=[
                "dataset",
                "id",
                "messages",
            ],
        )

        for record_batch in batches:
            if processed >= requested_rows:
                break

            rows = record_batch.to_pylist()

            remaining = requested_rows - processed
            rows = rows[:remaining]

            if not rows:
                break

            texts = [
                flatten_messages(row["messages"])
                for row in rows
            ]

            assert all(texts), (
                "A source record produced blank text"
            )

            encoded = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(
                device,
                non_blocking=True,
            )
            attention_mask = (
                encoded["attention_mask"].to(
                    device,
                    non_blocking=True,
                )
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                probabilities = {
                    field: torch.softmax(
                        logits[field].float()
                        / temperatures[field],
                        dim=-1,
                    )
                    for field in FIELDS
                }

            output_rows = []

            for local_index, source_row in enumerate(rows):
                output_row = {
                    "source_position":
                        processed + local_index,
                    "dataset":
                        str(source_row["dataset"]),
                    "id":
                        str(source_row["id"]),
                    "text_sha256":
                        sha256_text(
                            texts[local_index]
                        ),
                }

                for field in FIELDS:
                    vector = (
                        probabilities[field][local_index]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    assert vector.shape == (classes,)
                    assert np.isfinite(vector).all()

                    sum_error = abs(
                        float(vector.sum()) - 1.0
                    )

                    probability_sum_error[field] = max(
                        probability_sum_error[field],
                        sum_error,
                    )

                    k = TOP_K[field]

                    selected = np.argpartition(
                        -vector,
                        k - 1,
                    )[:k]

                    selected = selected[
                        np.argsort(
                            -vector[selected],
                            kind="stable",
                        )
                    ]

                    selected_probabilities = (
                        vector[selected]
                    )

                    for class_index in selected:
                        coverage[field][class_index] += 1

                    safe_vector = np.clip(
                        vector,
                        1e-12,
                        1.0,
                    )

                    entropy = float(
                        -np.sum(
                            safe_vector
                            * np.log(safe_vector)
                        )
                    )

                    max_probability = float(
                        vector.max()
                    )

                    max_probability_sum[field] += (
                        max_probability
                    )
                    entropy_sum[field] += entropy

                    output_row[
                        f"{field}_indices"
                    ] = [
                        int(value)
                        for value in selected
                    ]

                    output_row[
                        f"{field}_probabilities"
                    ] = [
                        float(value)
                        for value in selected_probabilities
                    ]

                    output_row[
                        f"{field}_distribution"
                    ] = [
                        float(value)
                        for value in vector
                    ]

                    output_row[
                        f"{field}_max_probability"
                    ] = max_probability

                    output_row[
                        f"{field}_entropy"
                    ] = entropy

                output_rows.append(output_row)

            table = pa.Table.from_pylist(
                output_rows,
                schema=schema,
            )

            writer.write_table(table)

            processed += len(rows)
            batch_number += 1

            if (
                batch_number == 1
                or batch_number % 20 == 0
                or processed == requested_rows
            ):
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-9)

                status.update(
                    {
                        "status": "RUNNING",
                        "stage": "inference",
                        "processed_rows": processed,
                        "elapsed_seconds": elapsed,
                        "rows_per_second": rate,
                    }
                )
                atomic_json(status_path, status)

                print(
                    f"processed={processed}/"
                    f"{requested_rows} "
                    f"rate={rate:.2f} rows/s"
                )

            del input_ids
            del attention_mask
            del logits
            del probabilities

    finally:
        writer.close()

    assert processed == requested_rows, (
        processed,
        requested_rows,
    )

    os.replace(partial_path, output_path)

    elapsed = time.time() - started

    report = {
        "status": "PASSED",
        "source_parquet": str(source_path),
        "source_rows": total_source_rows,
        "processed_rows": processed,
        "checkpoint": str(checkpoint),
        "calibration": str(calibration_path),
        "temperatures": temperatures,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "elapsed_seconds": elapsed,
        "rows_per_second":
            processed / max(elapsed, 1e-9),
        "output_parquet": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "fields": {},
    }

    for field in FIELDS:
        observed = np.flatnonzero(
            coverage[field] > 0
        )

        report["fields"][field] = {
            "top_k": TOP_K[field],
            "observed_classes": int(
                observed.size
            ),
            "unobserved_classes": [
                int(value)
                for value in np.flatnonzero(
                    coverage[field] == 0
                )
            ],
            "minimum_selected_count": int(
                coverage[field][observed].min()
            ) if observed.size else 0,
            "maximum_selected_count": int(
                coverage[field].max()
            ),
            "mean_max_probability":
                max_probability_sum[field]
                / processed,
            "mean_entropy":
                entropy_sum[field]
                / processed,
            "maximum_probability_sum_error":
                probability_sum_error[field],
        }

    atomic_json(report_path, report)

    status.update(
        {
            "status": "COMPLETE",
            "stage": "complete",
            "processed_rows": processed,
            "elapsed_seconds": elapsed,
            "rows_per_second":
                processed / max(elapsed, 1e-9),
            "report": str(report_path),
        }
    )
    atomic_json(status_path, status)

    print("=" * 72)
    print("DESCRIPTION-AWARE LAYA SOURCE CLASSIFICATION PASSED")
    print("=" * 72)
    print("Rows:", processed)
    print("Elapsed seconds:", elapsed)
    print(
        "Rows per second:",
        processed / max(elapsed, 1e-9),
    )

    for field in FIELDS:
        summary = report["fields"][field]
        print()
        print(field)
        print(
            "  Observed classes:",
            summary["observed_classes"],
        )
        print(
            "  Mean maximum probability:",
            summary["mean_max_probability"],
        )
        print(
            "  Mean entropy:",
            summary["mean_entropy"],
        )

    print("Output:", output_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
