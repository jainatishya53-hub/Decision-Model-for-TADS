#!/usr/bin/env bash

set -u
set -o pipefail
umask 077

ROOT="/home/gpuuser2/gpuuser2_a/atishya/pplx_decider_tads"
TADS_REPO="/home/gpuuser2/gpuuser2_a/atishya/tads"
PAPER_WORK="/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful"
CONDA_ENV="/home/gpuuser2/gpuuser2_a/atishya/tads_private/conda_env"
HF_HOME="/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface"

PYTHON="$CONDA_ENV/bin/python"
ACCELERATE="$CONDA_ENV/bin/accelerate"

BASE_MODEL="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

TRAINER="$ROOT/code/finetune_sequential_epoch.py"
TRAIN_FILE="$ROOT/data/model_finetune/tads_pplx_10k_gte_5_dataset.json"
MERGE="$TADS_REPO/model_finetune/merge_lora.py"
METRIC_EXTRACTOR="$ROOT/code/42_extract_full_epoch_metrics.py"

DATA_ROOT="$PAPER_WORK/data/eval/split_eval"
MMLU_ROOT="$DATA_ROOT/mmlu"

RUNTIME_ROOT="/tmp/gpuuser2_a_pplx_tads_sequential_full_run_01"
LORA_ROOT="$ROOT/models/llama31_8b_pplx_tads_gte5_sequential_5epoch_lora_run_01"
STATE_ROOT="$RUNTIME_ROOT/state"
MERGED_ROOT="$RUNTIME_ROOT/merged"

RESULT_ROOT="$ROOT/data/sequential_training/results/run_01"
LOG_ROOT="$ROOT/logs/sequential_training/run_01"
SUMMARY_ROOT="$ROOT/metadata/sequential_training/full_run_01"

STATUS_FILE="$ROOT/data/sequential_training/43_full_sequential_training_status.json"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PAPER_WORK/model_finetune:$TADS_REPO/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

stage="preflight"
epoch=0
benchmark="none"
completed_evaluations=0
current_log=""
current_output=""

write_status() {
    state="$1"

    STATE="$state" \
    STAGE="$stage" \
    EPOCH="$epoch" \
    BENCHMARK="$benchmark" \
    COMPLETED="$completed_evaluations" \
    CURRENT_LOG="$current_log" \
    CURRENT_OUTPUT="$current_output" \
    STATUS_FILE="$STATUS_FILE" \
    "$PYTHON" - <<'PY'
from pathlib import Path
import datetime
import json
import os

path = Path(os.environ["STATUS_FILE"])
path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "status": os.environ["STATE"],
    "stage": os.environ["STAGE"],
    "epoch": int(os.environ["EPOCH"]),
    "benchmark": os.environ["BENCHMARK"],
    "completed_evaluations": int(
        os.environ["COMPLETED"]
    ),
    "total_evaluations": 25,
    "log": os.environ["CURRENT_LOG"],
    "output": os.environ["CURRENT_OUTPUT"],
    "updated_utc": (
        datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
    ),
}

temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )
)
temporary.replace(path)
PY
}

fail() {
    message="$1"

    echo "FAILURE: $message"
    echo "Stage: $stage"
    echo "Epoch: $epoch"
    echo "Benchmark: $benchmark"

    write_status "FAILED"
    exit 1
}

fatal_log_check() {
    log="$1"

    if grep -aEq \
'Traceback \(most recent call last\)|CUDA out of memory|ChildFailedError|AssertionError|Killed|SIGTERM|SIGKILL' \
"$log"
    then
        fail "Fatal pattern found in $log"
    fi
}

require_file() {
    path="$1"

    test -s "$path" ||
        fail "Missing required file: $path"
}

gpu_pids() {
    nvidia-smi \
    --query-compute-apps=pid \
    --format=csv,noheader,nounits \
    2>/dev/null |
    sed '/^[[:space:]]*$/d' ||
    true
}

run_eval() {
    eval_epoch="$1"
    eval_benchmark="$2"
    result_directory="$3"
    log_path="$4"

    shift 4

    stage="evaluation"
    epoch="$eval_epoch"
    benchmark="$eval_benchmark"
    current_log="$log_path"
    current_output="$result_directory"

    write_status "RUNNING"

    test ! -e "$result_directory" ||
        fail "Result already exists: $result_directory"

    echo
    echo "EVALUATING EPOCH $epoch: $benchmark"

    cd "$PAPER_WORK/model_finetune" ||
        fail "Cannot enter model_finetune"

    CUDA_VISIBLE_DEVICES=0 \
    "$@" > "$log_path" 2>&1

    return_code=$?

    test "$return_code" -eq 0 ||
        fail "$benchmark exited with $return_code"

    fatal_log_check "$log_path"

    test -d "$result_directory" ||
        fail "Missing result directory"

    find "$result_directory" \
        -type f \
        -print -quit |
        grep -q . ||
        fail "Empty result directory"

    completed_evaluations=$((completed_evaluations + 1))

    write_status "BETWEEN_JOBS"
}

echo "============================================================"
echo "FULL FIVE-EPOCH SEQUENTIAL TRAIN/EVALUATE RUN"
echo "============================================================"

for required in \
"$PYTHON" \
"$ACCELERATE" \
"$TRAINER" \
"$TRAIN_FILE" \
"$MERGE" \
"$METRIC_EXTRACTOR" \
"$BASE_MODEL/model.safetensors.index.json"
do
    require_file "$required"
done

test "$(
    sha256sum "$TRAIN_FILE" |
    cut -d' ' -f1
)" = \
"b1c84215a3105f51b9da182989ac7d5b02c9ada1efba4cdff14766b44c5273c2" ||
fail "Production training-file hash changed"

test "$(
    sha256sum "$TRAINER" |
    cut -d' ' -f1
)" = \
"49f55bdcd769e163bdbe9dff9fff12fcaa5e0ecdfcf07d2db67264004b331595" ||
fail "Sequential trainer hash changed"

test -z "$(
    git -C "$TADS_REPO" status --porcelain
)" ||
fail "Original TADS repository is not clean"

test "$(
    git -C "$TADS_REPO" rev-parse HEAD
)" = \
"5459de6e411fdd2997156a53beae3df68705b9ec" ||
fail "Original TADS commit changed"

for planned in \
"$RUNTIME_ROOT" \
"$LORA_ROOT" \
"$RESULT_ROOT" \
"$LOG_ROOT" \
"$SUMMARY_ROOT"
do
    test ! -e "$planned" ||
        fail "Planned path already exists: $planned"
done

active_gpu_pids="$(gpu_pids)"

test -z "$active_gpu_pids" ||
    fail "GPU is busy with PID(s): $active_gpu_pids"

available_kib="$(
    df -Pk /tmp |
    awk 'NR==2 {print $4}'
)"

test "$available_kib" -ge 104857600 ||
    fail "Less than 100 GiB free in /tmp"

mkdir -p \
"$LORA_ROOT" \
"$STATE_ROOT" \
"$MERGED_ROOT" \
"$RESULT_ROOT" \
"$LOG_ROOT" \
"$SUMMARY_ROOT"

write_status "RUNNING"

for epoch_index in 0 1 2 3 4
do
    epoch_number=$((epoch_index + 1))

    stage="training"
    epoch="$epoch_number"
    benchmark="none"

    train_log="$LOG_ROOT/train_epoch_${epoch_number}.log"
    current_log="$train_log"
    current_output="$LORA_ROOT/epoch_${epoch_index}"

    write_status "RUNNING"

    resume_arguments=()

    if test "$epoch_index" -gt 0
    then
        previous_index=$((epoch_index - 1))

        resume_arguments=(
            --resume_from_checkpoint
            "$STATE_ROOT/epoch_${previous_index}"
        )
    fi

    echo
    echo "TRAINING THROUGH EPOCH $epoch_number"

    cd "$PAPER_WORK/model_finetune" ||
        fail "Cannot enter model_finetune"

    CUDA_VISIBLE_DEVICES=0 \
    "$ACCELERATE" launch \
    --num_machines 1 \
    --num_processes 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    "$TRAINER" \
    --model_name_or_path "$BASE_MODEL" \
    --tokenizer_name "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$LORA_ROOT" \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --learning_rate 1e-4 \
    --weight_decay 0 \
    --num_train_epochs 5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --seed 42 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --logging_steps 1 \
    --gradient_checkpointing \
    --low_cpu_mem_usage \
    --use_flash_attn \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --reduce_loss mean \
    --sequential_state_dir "$STATE_ROOT" \
    --stop_after_epoch "$epoch_index" \
    "${resume_arguments[@]}" \
    > "$train_log" 2>&1

    return_code=$?

    test "$return_code" -eq 0 ||
        fail "Training epoch $epoch_number failed"

    fatal_log_check "$train_log"

    adapter_directory="$LORA_ROOT/epoch_${epoch_index}"
    adapter_file="$adapter_directory/adapter_model.safetensors"
    state_directory="$STATE_ROOT/epoch_${epoch_index}"

    require_file "$adapter_file"

    root_adapter="$LORA_ROOT/adapter_model.safetensors"

    if test -s "$root_adapter"
    then
        cmp -s "$root_adapter" "$adapter_file" ||
            fail "Root adapter is not an exact epoch-checkpoint duplicate"

        rm -f "$root_adapter"

        echo "Removed verified duplicate root adapter; retained epoch checkpoint: $adapter_file"
    fi

    test -d "$state_directory" ||
        fail "Missing full training state"

    find "$state_directory" \
        -type f \
        -name 'optimizer*' \
        -print -quit |
        grep -q . ||
        fail "Optimizer state missing"

    find "$state_directory" \
        -type f \
        -name 'scheduler*' \
        -print -quit |
        grep -q . ||
        fail "Scheduler state missing"

    find "$state_directory" \
        -type f \
        -name 'random_states*' \
        -print -quit |
        grep -q . ||
        fail "RNG state missing"

    if test "$epoch_index" -gt 0
    then
        previous_index=$((epoch_index - 1))

        STATE_ROOT="$STATE_ROOT" \
        PREVIOUS_INDEX="$previous_index" \
        "$PYTHON" - <<'PY'
from pathlib import Path
import os
import shutil

root = Path(
    os.environ["STATE_ROOT"]
).resolve()

target = (
    root
    / f"epoch_{os.environ['PREVIOUS_INDEX']}"
).resolve()

assert target.parent == root
assert target.name.startswith("epoch_")
assert target.is_dir()

shutil.rmtree(target)

print(
    "Removed superseded temporary state:",
    target,
)
PY
    fi

    stage="merge"
    current_log="$LOG_ROOT/merge_epoch_${epoch_number}.log"

    merged_model="$MERGED_ROOT/epoch_${epoch_number}"
    current_output="$merged_model"

    write_status "RUNNING"

    echo "MERGING EPOCH $epoch_number"

    "$PYTHON" -u "$MERGE" \
    --base_model_name_or_path "$BASE_MODEL" \
    --lora_model_name_or_path "$adapter_directory" \
    --tokenizer_name_or_path "$BASE_MODEL" \
    --output_dir "$merged_model" \
    --save_tokenizer \
    > "$current_log" 2>&1

    return_code=$?

    test "$return_code" -eq 0 ||
        fail "Merge failed for epoch $epoch_number"

    fatal_log_check "$current_log"

    require_file \
"$merged_model/model.safetensors.index.json"

    epoch_result="$RESULT_ROOT/epoch_${epoch_number}"

    mkdir -p "$epoch_result"

    run_eval \
    "$epoch_number" \
    "mmlu" \
    "$epoch_result/mmlu" \
    "$LOG_ROOT/eval_epoch_${epoch_number}_mmlu.log" \
    "$PYTHON" -u -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir "$MMLU_ROOT" \
    --save_dir "$epoch_result/mmlu" \
    --model_name_or_path "$merged_model" \
    --tokenizer_name_or_path "$merged_model" \
    --eval_batch_size 4

    run_eval \
    "$epoch_number" \
    "gsm8k" \
    "$epoch_result/gsm8k" \
    "$LOG_ROOT/eval_epoch_${epoch_number}_gsm8k.log" \
    "$PYTHON" -u -m eval.gsm.run_eval \
    --data_dir "$DATA_ROOT/gsm" \
    --save_dir "$epoch_result/gsm8k" \
    --model_name_or_path "$merged_model" \
    --tokenizer_name_or_path "$merged_model" \
    --n_shot 8 \
    --use_vllm

    run_eval \
    "$epoch_number" \
    "bbh" \
    "$epoch_result/bbh" \
    "$LOG_ROOT/eval_epoch_${epoch_number}_bbh.log" \
    "$PYTHON" -u -m eval.bbh.run_eval \
    --data_dir "$DATA_ROOT/bbh" \
    --save_dir "$epoch_result/bbh" \
    --model_name_or_path "$merged_model" \
    --tokenizer_name_or_path "$merged_model" \
    --use_vllm

    run_eval \
    "$epoch_number" \
    "truthfulqa" \
    "$epoch_result/truthfulqa" \
    "$LOG_ROOT/eval_epoch_${epoch_number}_truthfulqa.log" \
    "$PYTHON" -u -m eval.truthfulqa.run_eval \
    --data_dir "$DATA_ROOT/truthfulqa" \
    --save_dir "$epoch_result/truthfulqa" \
    --model_name_or_path "$merged_model" \
    --tokenizer_name_or_path "$merged_model" \
    --metrics mc \
    --preset qa \
    --eval_batch_size 4 \
    --load_in_8bit

    run_eval \
    "$epoch_number" \
    "tydiqa" \
    "$epoch_result/tydiqa" \
    "$LOG_ROOT/eval_epoch_${epoch_number}_tydiqa.log" \
    "$PYTHON" -u -m eval.tydiqa.run_eval \
    --data_dir "$DATA_ROOT/tydiqa" \
    --n_shot 1 \
    --max_context_length 512 \
    --save_dir "$epoch_result/tydiqa" \
    --model_name_or_path "$merged_model" \
    --tokenizer_name_or_path "$merged_model" \
    --eval_batch_size 4 \
    --load_in_8bit

    stage="metric_extraction"
    benchmark="all_five"
    current_log=""
    current_output="$SUMMARY_ROOT/epoch_${epoch_number}.json"

    write_status "RUNNING"

    "$PYTHON" -u "$METRIC_EXTRACTOR" \
    --epoch "$epoch_number" \
    --result-root "$epoch_result" \
    --output "$SUMMARY_ROOT/epoch_${epoch_number}.json" ||
    fail "Metric extraction failed"

    MERGED_ROOT="$MERGED_ROOT" \
    EPOCH_NUMBER="$epoch_number" \
    "$PYTHON" - <<'PY'
from pathlib import Path
import os
import shutil

root = Path(
    os.environ["MERGED_ROOT"]
).resolve()

target = (
    root
    / f"epoch_{os.environ['EPOCH_NUMBER']}"
).resolve()

assert target.parent == root
assert target.name.startswith("epoch_")
assert target.is_dir()

shutil.rmtree(target)

print(
    "Removed reproducible temporary "
    "merged model:",
    target,
)
PY

    benchmark="none"
    write_status "EPOCH_COMPLETE"
done

stage="aggregation"
epoch=5
benchmark="all_five"
current_output="$SUMMARY_ROOT/five_epoch_full_results.csv"

write_status "RUNNING"

SUMMARY_ROOT="$SUMMARY_ROOT" \
"$PYTHON" - <<'PY'
from pathlib import Path
import csv
import json
import os

root = Path(
    os.environ["SUMMARY_ROOT"]
)

rows = []

for epoch in range(1, 6):
    path = root / f"epoch_{epoch}.json"

    assert path.is_file(), path

    payload = json.loads(
        path.read_text()
    )

    assert payload["status"] == "PASSED"
    assert payload["epoch"] == epoch

    metrics = payload["metrics"]

    rows.append(
        {
            "epoch": epoch,
            "mmlu": metrics[
                "mmlu"
            ]["percentage"],
            "truthfulqa_mc2": metrics[
                "truthfulqa"
            ]["percentage"],
            "gsm8k": metrics[
                "gsm8k"
            ]["percentage"],
            "bbh": metrics[
                "bbh"
            ]["percentage"],
            "tydiqa": metrics[
                "tydiqa"
            ]["percentage"],
            "average": payload[
                "average_percentage"
            ],
        }
    )

csv_path = (
    root
    / "five_epoch_full_results.csv"
)

with csv_path.open(
    "w",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "epoch",
            "mmlu",
            "truthfulqa_mc2",
            "gsm8k",
            "bbh",
            "tydiqa",
            "average",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)

json_path = (
    root
    / "five_epoch_full_results.json"
)

json_path.write_text(
    json.dumps(
        {
            "status": "PASSED",
            "rows": rows,
            "epoch_count": 5,
            "benchmark_count": 5,
            "completed_evaluations": 25,
            "full_benchmark_scores": True,
        },
        indent=2,
        sort_keys=True,
    )
)

print("=" * 72)
print("FULL FIVE-EPOCH SEQUENTIAL RESULTS")
print("=" * 72)

for row in rows:
    print(row)

print("CSV:", csv_path)
print("JSON:", json_path)
PY

test "$completed_evaluations" -eq 25 ||
    fail "Expected 25 evaluations"

stage="complete"
benchmark="none"
current_log=""
current_output="$SUMMARY_ROOT/five_epoch_full_results.csv"

write_status "PASSED"

echo
echo "============================================================"
echo "FULL FIVE-EPOCH SEQUENTIAL RUN PASSED"
echo "============================================================"
echo "Completed evaluations: $completed_evaluations/25"
echo "Results: $SUMMARY_ROOT"
echo "Persistent LoRA checkpoints: $LORA_ROOT"
echo "Temporary state root: $RUNTIME_ROOT"
