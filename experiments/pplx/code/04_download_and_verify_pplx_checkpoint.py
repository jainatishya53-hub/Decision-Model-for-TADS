#!/usr/bin/env python3

from collections import Counter
from pathlib import Path
import json
import math
import os
import shutil
import time
import traceback

from huggingface_hub import snapshot_download
from safetensors import safe_open


EXPECTED_MODEL = "perplexity-ai/pplx-decider-v1.2"
EXPECTED_REVISION = (
    "cce0f3f80033e9d89d80831d5281fd72124aacce"
)
EXPECTED_PARAMETERS = 26085330160
EXPECTED_WEIGHT_BYTES = 52170660320
EXPECTED_MODEL_SHARDS = 11

root = Path(
    os.environ["PPLX_TADS_ROOT"]
).resolve()

checkpoint = Path(
    os.environ["PPLX_CHECKPOINT"]
).resolve()

model_id = os.environ["PPLX_MODEL_ID"]
revision = os.environ["PPLX_MODEL_REVISION"]

status_path = (
    root
    / "data"
    / "checkpoint_download"
    / "04_pplx_checkpoint_download_status.json"
)

report_path = (
    root
    / "metadata"
    / "checkpoint_download"
    / "04_pplx_checkpoint_verification.json"
)

status_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint.mkdir(
    parents=True,
    exist_ok=True,
)

assert model_id == EXPECTED_MODEL
assert revision == EXPECTED_REVISION

started = time.time()


def write_status(payload):
    temporary = status_path.with_suffix(
        ".json.tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary.replace(status_path)


try:
    free_bytes = shutil.disk_usage(
        checkpoint.parent
    ).free

    required_bytes = (
        EXPECTED_WEIGHT_BYTES
        + 30 * 1024 ** 3
    )

    assert free_bytes >= required_bytes, (
        "Insufficient temporary storage. "
        f"Free GiB={free_bytes / 1024 ** 3:.2f}; "
        f"required GiB={required_bytes / 1024 ** 3:.2f}"
    )

    write_status(
        {
            "status": "STARTING",
            "stage": "downloading",
            "model": model_id,
            "revision": revision,
            "checkpoint": str(checkpoint),
            "expected_weight_bytes": (
                EXPECTED_WEIGHT_BYTES
            ),
            "free_bytes_before_download": free_bytes,
        }
    )

    print("=" * 78)
    print("PPLX DECIDER CHECKPOINT DOWNLOAD")
    print("=" * 78)
    print("Model:", model_id)
    print("Revision:", revision)
    print("Destination:", checkpoint)
    print(
        "Free storage GiB:",
        free_bytes / 1024 ** 3,
    )
    print(
        "Expected BF16 weight GiB:",
        EXPECTED_WEIGHT_BYTES / 1024 ** 3,
    )
    print("The download is resumable.")

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=True,
        local_dir=checkpoint,
        max_workers=4,
        ignore_patterns=[
            "*.gguf",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.ckpt",
        ],
    )

    write_status(
        {
            "status": "RUNNING",
            "stage": "verifying",
            "model": model_id,
            "revision": revision,
            "checkpoint": str(checkpoint),
        }
    )

    index_path = (
        checkpoint
        / "model.safetensors.index.json"
    )

    assert index_path.is_file(), (
        f"Missing weight index: {index_path}"
    )

    index = json.loads(
        index_path.read_text(
            encoding="utf-8"
        )
    )

    metadata = index.get(
        "metadata",
        {},
    )

    weight_map = index.get(
        "weight_map",
        {},
    )

    assert int(
        metadata["total_parameters"]
    ) == EXPECTED_PARAMETERS

    assert int(
        metadata["total_size"]
    ) == EXPECTED_WEIGHT_BYTES

    shard_names = sorted(
        set(weight_map.values())
    )

    assert len(
        shard_names
    ) == EXPECTED_MODEL_SHARDS, (
        f"Expected {EXPECTED_MODEL_SHARDS} shards, "
        f"found {len(shard_names)}"
    )

    missing_shards = [
        name
        for name in shard_names
        if not (checkpoint / name).is_file()
    ]

    assert not missing_shards, (
        f"Missing shards: {missing_shards}"
    )

    tensor_count = 0
    parameter_count = 0
    dtype_counts = Counter()
    prefix_counts = Counter()
    shard_records = []

    for shard_name in shard_names:
        shard_path = (
            checkpoint
            / shard_name
        )

        shard_tensor_count = 0
        shard_parameter_count = 0

        with safe_open(
            shard_path,
            framework="pt",
            device="cpu",
        ) as handle:
            keys = list(handle.keys())

            for key in keys:
                tensor_slice = handle.get_slice(
                    key
                )

                shape = tuple(
                    tensor_slice.get_shape()
                )

                numel = math.prod(shape)

                dtype = str(
                    tensor_slice.get_dtype()
                )

                shard_tensor_count += 1
                shard_parameter_count += numel

                tensor_count += 1
                parameter_count += numel

                dtype_counts[dtype] += 1
                prefix_counts[
                    key.split(".", 1)[0]
                ] += 1

        shard_records.append(
            {
                "name": shard_name,
                "bytes": shard_path.stat().st_size,
                "tensor_count": shard_tensor_count,
                "parameter_count": (
                    shard_parameter_count
                ),
            }
        )

        print(
            f"verified={shard_name} "
            f"tensors={shard_tensor_count} "
            f"parameters={shard_parameter_count}"
        )

    assert tensor_count == len(
        weight_map
    ), (
        f"Tensor-count mismatch: {tensor_count} "
        f"versus {len(weight_map)}"
    )

    assert parameter_count == EXPECTED_PARAMETERS, (
        f"Parameter-count mismatch: {parameter_count} "
        f"versus {EXPECTED_PARAMETERS}"
    )

    additional_safetensors = sorted(
        path
        for path in checkpoint.glob(
            "*.safetensors"
        )
        if path.name not in shard_names
    )

    additional_records = []

    for path in additional_safetensors:
        with safe_open(
            path,
            framework="pt",
            device="cpu",
        ) as handle:
            tensors = []

            for key in handle.keys():
                tensor_slice = handle.get_slice(
                    key
                )

                shape = tuple(
                    tensor_slice.get_shape()
                )

                tensors.append(
                    {
                        "name": key,
                        "shape": list(shape),
                        "parameters": math.prod(
                            shape
                        ),
                        "dtype": str(
                            tensor_slice.get_dtype()
                        ),
                    }
                )

        additional_records.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "tensors": tensors,
            }
        )

    total_checkpoint_bytes = sum(
        path.stat().st_size
        for path in checkpoint.rglob("*")
        if path.is_file()
    )

    report = {
        "status": "PASSED",
        "model": model_id,
        "revision": revision,
        "checkpoint": str(checkpoint),
        "model_shards": len(shard_names),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "weight_bytes_from_index": (
            EXPECTED_WEIGHT_BYTES
        ),
        "total_checkpoint_bytes": (
            total_checkpoint_bytes
        ),
        "dtype_counts": dict(
            sorted(dtype_counts.items())
        ),
        "prefix_counts": dict(
            sorted(prefix_counts.items())
        ),
        "shards": shard_records,
        "additional_safetensors": (
            additional_records
        ),
        "elapsed_seconds": time.time() - started,
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    write_status(
        {
            "status": "COMPLETE",
            "stage": "complete",
            "model": model_id,
            "revision": revision,
            "checkpoint": str(checkpoint),
            "model_shards": len(shard_names),
            "parameter_count": parameter_count,
            "report": str(report_path),
            "elapsed_seconds": (
                time.time() - started
            ),
        }
    )

    print("=" * 78)
    print("PPLX CHECKPOINT DOWNLOAD AND VERIFICATION")
    print("=" * 78)
    print("Status: PASSED")
    print("Model:", model_id)
    print("Revision:", revision)
    print("Model shards:", len(shard_names))
    print("Model tensors:", tensor_count)
    print("Model parameters:", parameter_count)
    print(
        "Indexed weight GiB:",
        EXPECTED_WEIGHT_BYTES / 1024 ** 3,
    )
    print(
        "Checkpoint directory GiB:",
        total_checkpoint_bytes / 1024 ** 3,
    )
    print("Dtypes:", dict(dtype_counts))
    print("Prefixes:", dict(prefix_counts))
    print(
        "Additional safetensors:",
        additional_records,
    )
    print("Checkpoint:", checkpoint)
    print("Report:", report_path)
    print(
        "\nPPLX CHECKPOINT DOWNLOAD "
        "AND VERIFICATION PASSED"
    )

except Exception as error:
    write_status(
        {
            "status": "FAILED",
            "stage": "failed",
            "model": model_id,
            "revision": revision,
            "checkpoint": str(checkpoint),
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": (
                time.time() - started
            ),
        }
    )

    traceback.print_exc()
    raise
