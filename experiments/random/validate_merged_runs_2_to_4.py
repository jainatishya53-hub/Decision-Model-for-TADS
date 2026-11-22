from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import subprocess
import sys

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

paper = Path(
    "/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful"
)
repo = Path(
    "/home/gpuuser2/gpuuser2_a/atishya/tads"
)
base = Path(
    "/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/"
    "huggingface/hub/models--meta-llama--Llama-3.1-8B/"
    "snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
)

runs = {
    2: {
        "seed": 1787701859309839880,
        "adapter_sha256":
            "bc52f149f2bd27b13a633748f60fb4da14a2abaa5ca600ad955deafad2c24a14",
        "lora_marker_sha256":
            "3d200eb1cbefea373b3bb60ddcb2a5fa7c2f13725a0c0d076d32c448e50073b7",
    },
    3: {
        "seed": 1787701873612384727,
        "adapter_sha256":
            "30ac5f08b09d61e9d3ca476c6617639e402895c8645e76b62645259e5ecb98ec",
        "lora_marker_sha256":
            "0d5c0944bfa20edb05acd9ce271067a591e4a96105dc792d8bfa93c2b7bcf351",
    },
    4: {
        "seed": 1787701887533694015,
        "adapter_sha256":
            "b154e8e70df60eb5db4a3cf23b8784f3cac3aea925474341c1d61c10a2f7ef85",
        "lora_marker_sha256":
            "394f28f4865696f5238be9f819ce948eac347b12ee503b641550bf852c2b1a60",
    },
}

tokenizer_hashes_expected = {
    "tokenizer.json":
        "6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b",
    "tokenizer_config.json":
        "babefc68107163310a6cb953878ee7c32e45979729943d291f2604ad9d264793",
    "special_tokens_map.json":
        "cc2e013b7545f183ef03e079a3c91c6f364fa37e4068c512d7dd843e59024535",
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

print("BATCH MERGED-MODEL VALIDATION STARTED")
print("Time:", datetime.now(timezone.utc).isoformat())
print(flush=True)

print("Checking shared locked evidence:")

shared_expected = {
    paper / "model_finetune/merge_lora.py":
        "b0023e4df73867f9f8e1a5ac5191b34946d8268fb5756aaf9b76c2c574b92a11",

    paper / "data/random_10k_4runs/merged_validations/run_1/"
            "RANDOM_RUN_1_MERGED_MODEL_VALIDATED.json":
        "db9e376580b89890d6eec46773c9868254c22d2ea0924a408a11266bb91df475",
}

for path, wanted in shared_expected.items():
    assert path.is_file(), f"Missing: {path}"
    actual = sha256(path)
    assert actual == wanted, f"Checksum mismatch: {path}"
    print(f"OK: {path.relative_to(paper)} {actual}")

commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    text=True,
).strip()

repository_status = subprocess.check_output(
    ["git", "-C", str(repo), "status", "--porcelain"],
    text=True,
)

assert commit == "5459de6e411fdd2997156a53beae3df68705b9ec"
assert not repository_status

base_index = read_json(base / "model.safetensors.index.json")
tensor_name = "model.layers.0.self_attn.q_proj.weight"
base_shard = base / base_index["weight_map"][tensor_name]

with safe_open(
    str(base_shard),
    framework="pt",
    device="cpu",
) as f:
    base_tensor = f.get_tensor(tensor_name).float()

print("Repository and base model: PASSED", flush=True)

for run, run_info in runs.items():
    print()
    print("=" * 72)
    print(f"VALIDATING RANDOM RUN {run}")
    print("=" * 72)
    print(flush=True)

    merged = (
        paper /
        f"model_finetune/model_output/models/random_10k_4runs/"
        f"run_{run}_3ep_merged"
    )
    adapter = (
        paper /
        f"model_finetune/model_output/models/random_10k_4runs/"
        f"run_{run}_3ep_lora/adapter_model.safetensors"
    )
    lora_marker = (
        paper /
        f"data/random_10k_4runs/model_validations/run_{run}/"
        f"RANDOM_RUN_{run}_LORA_VALIDATED.json"
    )
    ready_marker = (
        paper /
        f"data/random_10k_4runs/merge_preparation/run_{run}/"
        f"RUN_{run}_MERGED_OUTPUT_READY.json"
    )
    recovery_marker = (
        paper /
        f"data/random_10k_4runs/merge_recovery/run_{run}/"
        f"RUN_{run}_TOKENIZER_RECOVERY_VALIDATED.json"
    )
    merge_log = (
        paper /
        f"logs/random_10k_4runs/merge_run_{run}_3ep.log"
    )
    validation = (
        paper /
        f"data/random_10k_4runs/merged_validations/run_{run}"
    )

    if validation.exists():
        marker = (
            validation /
            f"RANDOM_RUN_{run}_MERGED_MODEL_VALIDATED.json"
        )
        report = (
            validation /
            f"run_{run}_merged_model_validation.json"
        )

        if marker.is_file() and report.is_file():
            print(
                f"Run {run} already has complete validation; skipping."
            )
            continue

        raise RuntimeError(
            f"Incomplete validation directory exists: {validation}"
        )

    print("Checking locked run evidence:")

    assert sha256(adapter) == run_info["adapter_sha256"]
    assert sha256(lora_marker) == run_info["lora_marker_sha256"]
    assert ready_marker.is_file()
    assert recovery_marker.is_file()
    assert merge_log.is_file()

    ready_data = read_json(ready_marker)
    recovery_data = read_json(recovery_marker)

    assert ready_data["run"] == run
    assert ready_data["weight_shards"] == 7
    assert recovery_data["run"] == run
    assert recovery_data["computational_effect"] == "NONE"

    readiness_hash = sha256(ready_marker)
    recovery_hash = sha256(recovery_marker)

    print("Adapter SHA-256:", run_info["adapter_sha256"])
    print("Readiness marker SHA-256:", readiness_hash)
    print("Recovery marker SHA-256:", recovery_hash)

    log_text = merge_log.read_text(
        errors="replace"
    ).replace("\r", "\n")

    fatal_terms = [
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "RuntimeError",
        "AssertionError",
        "Killed",
        "SIGTERM",
        "SIGKILL",
    ]

    fatal_matches = [
        term for term in fatal_terms if term in log_text
    ]

    required_markers = [
        "Loading the lora model",
        "Merging the lora modules",
        "Saving merged model",
    ]

    missing_markers = [
        marker for marker in required_markers
        if marker not in log_text
    ]

    print("Fatal patterns:", fatal_matches)
    print("Missing merge markers:", missing_markers)

    assert not fatal_matches
    assert not missing_markers

    required_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    for filename in required_files:
        path = merged / filename
        assert path.is_file() and path.stat().st_size > 0

    temporary_files = (
        list(merged.glob("*.tmp"))
        + list(merged.glob("*.part"))
        + list(merged.glob("*.incomplete"))
    )

    assert not temporary_files

    index = read_json(merged / "model.safetensors.index.json")
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))

    print("Indexed tensors:", len(weight_map))
    print("Weight shards:", len(shard_names))

    assert len(weight_map) == 291
    assert len(shard_names) == 7

    for shard_name in shard_names:
        shard = merged / shard_name
        assert shard.is_file() and shard.stat().st_size > 0

    config = read_json(merged / "config.json")

    assert config["model_type"] == "llama"
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["vocab_size"] == 128264

    tokenizer = AutoTokenizer.from_pretrained(
        str(merged),
        local_files_only=True,
    )

    tokenizer_summary = {
        "class": tokenizer.__class__.__name__,
        "length": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    print("Tokenizer:", tokenizer_summary)

    assert len(tokenizer) == 128256
    assert tokenizer.pad_token is None
    assert tokenizer.pad_token_id is None
    assert tokenizer.eos_token_id == 128001

    actual_tokenizer_hashes = {
        filename: sha256(merged / filename)
        for filename in tokenizer_hashes_expected
    }

    assert actual_tokenizer_hashes == tokenizer_hashes_expected

    merged_shard = merged / weight_map[tensor_name]

    with safe_open(
        str(merged_shard),
        framework="pt",
        device="cpu",
    ) as f:
        merged_tensor = f.get_tensor(tensor_name).float()

    difference = (merged_tensor - base_tensor).abs()

    changed_elements = int(
        torch.count_nonzero(difference).item()
    )
    maximum_change = float(difference.max().item())
    mean_change = float(difference.mean().item())

    print("Changed q_proj elements:", changed_elements)
    print("Maximum absolute change:", maximum_change)
    print("Mean absolute change:", mean_change)

    assert changed_elements > 0
    assert math.isfinite(maximum_change) and maximum_change > 0
    assert math.isfinite(mean_change) and mean_change > 0

    del merged_tensor, difference
    gc.collect()

    print("Loading complete model on CPU:", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(merged),
        local_files_only=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    parameter_dtypes = sorted({
        str(parameter.dtype)
        for parameter in model.parameters()
    })
    input_rows = model.get_input_embeddings().weight.shape[0]
    output_rows = model.get_output_embeddings().weight.shape[0]
    remaining_lora = [
        name
        for name, _ in model.named_parameters()
        if "lora_" in name.lower()
    ]

    print("Parameters:", parameter_count)
    print("Parameter dtypes:", parameter_dtypes)
    print("Input rows:", input_rows)
    print("Output rows:", output_rows)
    print("Remaining LoRA parameters:", len(remaining_lora))

    assert parameter_count == 8030326784
    assert parameter_dtypes == ["torch.float32"]
    assert input_rows == 128264
    assert output_rows == 128264
    assert not remaining_lora

    del model
    gc.collect()

    print("Computing complete artifact checksums:", flush=True)

    artifact_hashes = {}

    for filename in required_files + shard_names:
        path = merged / filename
        artifact_hashes[filename] = sha256(path)
        print(filename, artifact_hashes[filename], flush=True)

    report = {
        "experiment": "random_10k_4runs",
        "run": run,
        "status": "merged_model_fully_validated",
        "validated_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "sampling_seed": run_info["seed"],
        "training_seed": 42,
        "repository_commit": commit,
        "repository_clean": True,
        "lora_adapter_sha256":
            run_info["adapter_sha256"],
        "merge_readiness_marker_sha256": readiness_hash,
        "tokenizer_recovery_marker_sha256": recovery_hash,
        "tokenizer_recovery_computational_effect": "NONE",
        "merged_model_directory": str(merged),
        "indexed_tensors": len(weight_map),
        "weight_shards": len(shard_names),
        "parameter_count": parameter_count,
        "parameter_dtypes": parameter_dtypes,
        "input_embedding_rows": input_rows,
        "output_head_rows": output_rows,
        "remaining_lora_parameters": 0,
        "merge_effect": {
            "tensor": tensor_name,
            "changed_elements": changed_elements,
            "maximum_absolute_change": maximum_change,
            "mean_absolute_change": mean_change,
        },
        "tokenizer": tokenizer_summary,
        "artifact_sha256": artifact_hashes,
    }

    temporary = (
        validation.parent /
        f".run_{run}_merged_validation_{os.getpid()}"
    )
    temporary.mkdir(parents=True)

    report_path = (
        temporary /
        f"run_{run}_merged_model_validation.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_hash = sha256(report_path)

    marker_path = (
        temporary /
        f"RANDOM_RUN_{run}_MERGED_MODEL_VALIDATED.json"
    )
    marker_path.write_text(
        json.dumps({
            "status":
                f"RANDOM RUN-{run} MERGED MODEL FULLY VALIDATED",
            "run": run,
            "lora_adapter_sha256":
                run_info["adapter_sha256"],
            "validation_report_sha256": report_hash,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for path in temporary.iterdir():
        path.chmod(0o444)

    temporary.rename(validation)

    print("Installed:")
    print(
        marker_path.name,
        sha256(validation / marker_path.name),
    )
    print(
        report_path.name,
        sha256(validation / report_path.name),
    )
    print(f"RUN-{run} MERGED MODEL VALIDATION PASSED")

del base_tensor
gc.collect()

print()
print("=" * 72)
print("RUNS 2–4 MERGED MODELS PASSED FULL VALIDATION")
print("=" * 72)
