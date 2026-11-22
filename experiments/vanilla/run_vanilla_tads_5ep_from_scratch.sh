#!/bin/bash

set -u
set -o pipefail
umask 077

PAPER_WORK=/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful
TADS_SOURCE=/home/gpuuser2/gpuuser2_a/atishya/tads
CONDA_ENV=/home/gpuuser2/gpuuser2_a/atishya/tads_private/conda_env
HF_HOME=/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface

BASE_MODEL="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
TRAIN_FILE="$PAPER_WORK/model_finetune/selected_data/static_row_uid_gte_5_10k.json"
FINETUNE="$PAPER_WORK/model_finetune/finetune.py"
MERGE="$PAPER_WORK/model_finetune/merge_lora.py"

EXPERIMENT_ROOT="$PAPER_WORK/data/vanilla_tads_5ep_from_scratch"
LOG_ROOT="$PAPER_WORK/logs/vanilla_tads_5ep_from_scratch"
LORA_OUTPUT="$PAPER_WORK/model_finetune/model_output/models/static_row_uid_gte_5_5ep_from_scratch_lora"
MERGED_ROOT="$PAPER_WORK/model_finetune/model_output/models/static_row_uid_gte_5_5ep_from_scratch_merged"
RESULT_ROOT="$PAPER_WORK/model_finetune/model_output/results/static_row_uid_gte_5_5ep_from_scratch"
DATA_ROOT="$PAPER_WORK/data/eval/split_eval"
FINAL_OUTPUT="$PAPER_WORK/metadata/vanilla_tads_5ep_from_scratch"
AGGREGATOR="$PAPER_WORK/aggregate_vanilla_tads_5ep_from_scratch.py"

TRUTH_JUDGE="$HF_HOME/hub/models--allenai--truthfulqa-truth-judge-llama2-7B/snapshots/8f718cb4e68bc30a488727a0e37190835a7c129e"
INFO_JUDGE="$HF_HOME/hub/models--allenai--truthfulqa-info-judge-llama2-7B/snapshots/72601eff294cbf7e8d7188ae82397e19cb9bb9bd"

PYTHON="$CONDA_ENV/bin/python"
ACCELERATE="$CONDA_ENV/bin/accelerate"
STATUS_FILE="$EXPERIMENT_ROOT/status.txt"

export PAPER_WORK TADS_SOURCE CONDA_ENV HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PAPER_WORK:$TADS_SOURCE/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

stage=preflight
epoch=none
benchmark=none
completed=0
current_log=""
current_output=""

write_status() {
    state="$1"
    mkdir -p "$EXPERIMENT_ROOT"
    {
        echo "status=$state"
        echo "stage=$stage"
        echo "epoch=$epoch"
        echo "benchmark=$benchmark"
        echo "completed_evaluations=$completed"
        echo "total_evaluations=25"
        echo "log=$current_log"
        echo "output=$current_output"
        echo "updated=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    } > "$STATUS_FILE.tmp"
    mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

fail() {
    message="$1"
    echo "FAILURE: $message"
    echo "Stage: $stage"
    echo "Epoch: $epoch"
    echo "Benchmark: $benchmark"
    write_status FAILED
    exit 1
}

require_hash() {
    path="$1"
    expected="$2"
    [ -f "$path" ] || fail "Missing locked file: $path"
    actual="$(sha256sum "$path" | cut -d' ' -f1)"
    [ "$actual" = "$expected" ] || fail "Checksum mismatch: $path"
    echo "OK: $path $actual"
}

fatal_log_check() {
    log="$1"
    if grep -aEq \
      'Traceback \(most recent call last\)|CUDA out of memory|ChildFailedError|RuntimeError|AssertionError|Killed|SIGTERM|SIGKILL' \
      "$log"
    then
        fail "Fatal pattern found in $log"
    fi
}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      sed '/^[[:space:]]*$/d' || true
}

wait_for_gpu_idle() {
    stage=waiting_for_gpu
    write_status WAITING_FOR_GPU
    while true; do
        pids="$(gpu_pids)"
        if [ -z "$pids" ]; then
            echo "GPU is idle: $(date -u)"
            return
        fi
        echo "GPU busy with PID(s): $pids; waiting: $(date -u)"
        sleep 60
    done
}

validate_merged_model() {
    model="$1"
    MODEL_TO_CHECK="$model" "$PYTHON" - <<'PY'
from pathlib import Path
import json
import os

model = Path(os.environ["MODEL_TO_CHECK"])
required = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
for name in required:
    path = model / name
    assert path.is_file() and path.stat().st_size > 0, path
index = json.loads((model / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
shards = sorted(set(weight_map.values()))
assert len(weight_map) == 291, len(weight_map)
assert len(shards) == 7, shards
for shard in shards:
    path = model / shard
    assert path.is_file() and path.stat().st_size > 1_000_000_000, path
print("Merged tensors:", len(weight_map))
print("Merged shards:", len(shards))
PY
    [ "$?" -eq 0 ] || fail "Merged-model validation failed: $model"
}

run_eval() {
    epoch="$1"
    benchmark="$2"
    result="$3"
    log="$4"
    shift 4

    stage=evaluation
    current_log="$log"
    current_output="$result"
    write_status RUNNING

    [ ! -e "$result" ] || fail "Result path already exists: $result"
    [ ! -e "$log" ] || fail "Evaluation log already exists: $log"

    wait_for_gpu_idle
    stage=evaluation
    write_status RUNNING
    echo "EVALUATING EPOCH $epoch: $benchmark: $(date -u)"

    cd "$PAPER_WORK/model_finetune" || fail "Cannot enter model_finetune"
    CUDA_VISIBLE_DEVICES=0 "$@" > "$log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "$benchmark evaluator exited with status $rc"
    fatal_log_check "$log"
    [ -d "$result" ] || fail "Missing result directory: $result"
    find "$result" -type f -print -quit | grep -q . || fail "Empty result directory: $result"

    completed=$((completed + 1))
    mkdir -p "$EXPERIMENT_ROOT/execution/epoch_$epoch"
    cat > "$EXPERIMENT_ROOT/execution/epoch_$epoch/$benchmark.json" <<EOF
{
  "epoch": $epoch,
  "benchmark": "$benchmark",
  "status": "completed",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "result": "$result",
  "log": "$log"
}
EOF
    write_status BETWEEN_JOBS
}

echo "VANILLA TADS FIVE-EPOCH FROM-SCRATCH PIPELINE"
echo "Started: $(date -u)"

for planned in "$EXPERIMENT_ROOT" "$LOG_ROOT" "$LORA_OUTPUT" "$MERGED_ROOT" "$RESULT_ROOT" "$FINAL_OUTPUT"; do
    [ ! -e "$planned" ] || { echo "Planned path already exists: $planned"; exit 1; }
done

mkdir -p "$EXPERIMENT_ROOT" "$LOG_ROOT"
write_status PREFLIGHT

require_hash "$FINETUNE" 31ce8030112fe72de83f5d0626cb794e1e1e07f938e7b2a42951f39f48a96816
require_hash "$MERGE" b0023e4df73867f9f8e1a5ac5191b34946d8268fb5756aaf9b76c2c574b92a11
require_hash "$TRAIN_FILE" cc84db71b244660e7aae815e822534bb9676283879b48394a482f4f72708b3cc
require_hash "$PAPER_WORK/model_finetune/eval/mmlu/run_eval.py" 81a441d141359c61c6a3f156a2b2b804e90af396a0fe0cabf115c34e932b6d35
require_hash "$PAPER_WORK/model_finetune/eval/gsm/run_eval.py" 852863714e26ea6bec45bad41dde5a1bd65b7e1121ff77092be3fcbd2b292bfc
require_hash "$PAPER_WORK/model_finetune/eval/bbh/run_eval.py" f9b7ffb2990bb458cbdd772d3c03dff9c57ce57fce9dea12fff671f75009cc55
require_hash "$PAPER_WORK/model_finetune/eval/truthfulqa/run_eval.py" c04837a8b18493fc171795a224bbbe7228263fc17ae95d3ae8c87f19aa6f8100
require_hash "$PAPER_WORK/model_finetune/eval/tydiqa/run_eval.py" 331c578d60849c1eefb462328182ba32e2594636bb1e6a470b747cc099a2a143

[ "$(git -C "$TADS_SOURCE" rev-parse HEAD)" = "5459de6e411fdd2997156a53beae3df68705b9ec" ] || fail "Repository commit mismatch"
[ -z "$(git -C "$TADS_SOURCE" status --porcelain)" ] || fail "Official repository is not clean"
[ -s "$BASE_MODEL/model.safetensors.index.json" ] || fail "Base model is missing"
[ -s "$TRUTH_JUDGE/pytorch_model.bin.index.json" ] || fail "Truth judge is missing"
[ -s "$INFO_JUDGE/pytorch_model.bin.index.json" ] || fail "Info judge is missing"
[ -x "$ACCELERATE" ] || fail "Accelerate executable is missing"
[ -s "$AGGREGATOR" ] || fail "Aggregator is missing"

available_kib="$(df -Pk "$PAPER_WORK" | awk 'NR==2 {print $4}')"
[ "$available_kib" -ge 367001600 ] || fail "Less than 350 GiB storage is available"

wait_for_gpu_idle

stage=training
epoch=all_5
benchmark=none
current_log="$LOG_ROOT/train_5ep.log"
current_output="$LORA_OUTPUT"
write_status RUNNING

echo "TRAINING STARTED: $(date -u)"
cd "$PAPER_WORK/model_finetune" || fail "Cannot enter model_finetune"

CUDA_VISIBLE_DEVICES=0 "$ACCELERATE" launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 1 \
  "$FINETUNE" \
  --model_name_or_path "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --max_seq_length 2048 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --num_train_epochs 5 \
  --seed 42 \
  --use_lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --use_flash_attn \
  --gradient_checkpointing \
  --checkpointing_steps epoch \
  --logging_steps 1 \
  --with_tracking \
  --report_to tensorboard \
  --output_dir "$LORA_OUTPUT" \
  > "$current_log" 2>&1
rc=$?
[ "$rc" -eq 0 ] || fail "Training exited with status $rc"
fatal_log_check "$current_log"

TRAIN_LOG="$current_log" LORA_OUTPUT="$LORA_OUTPUT" "$PYTHON" - <<'PY'
from pathlib import Path
import json
import os
import re

log = Path(os.environ["TRAIN_LOG"]).read_text(errors="replace").replace("\r", "\n")
steps = [int(value) for value in re.findall(r"Step: (\d+), LR:", log)]
assert len(set(steps)) == 390, len(set(steps))
assert min(steps) == 1 and max(steps) == 390

root = Path(os.environ["LORA_OUTPUT"])
for index in range(5):
    checkpoint = root / f"epoch_{index}"
    adapter = checkpoint / "adapter_model.safetensors"
    config_path = checkpoint / "adapter_config.json"
    assert adapter.is_file() and adapter.stat().st_size > 4_000_000_000, adapter
    config = json.loads(config_path.read_text())
    assert config["r"] == 64
    assert float(config["lora_alpha"]) == 16.0
    assert float(config["lora_dropout"]) == 0.1
final = root / "adapter_model.safetensors"
assert final.is_file() and final.stat().st_size > 4_000_000_000
print("Validated optimization steps: 390")
print("Validated epoch checkpoints: 5")
PY
[ "$?" -eq 0 ] || fail "Training-output validation failed"

cmp -s "$LORA_OUTPUT/epoch_4/adapter_model.safetensors" "$LORA_OUTPUT/adapter_model.safetensors" || \
  fail "Final adapter does not equal epoch_4"

cat > "$EXPERIMENT_ROOT/TRAINING_COMPLETE.json" <<EOF
{
  "status": "complete_and_validated",
  "epochs": 5,
  "optimization_steps": 390,
  "checkpoint_mapping": {
    "epoch_0": "after epoch 1",
    "epoch_1": "after epoch 2",
    "epoch_2": "after epoch 3",
    "epoch_3": "after epoch 4",
    "epoch_4": "after epoch 5"
  },
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}
EOF

for epoch_number in 1 2 3 4 5; do
    adapter="$LORA_OUTPUT/epoch_$((epoch_number - 1))"
    model="$MERGED_ROOT/epoch_$epoch_number"
    merge_log="$LOG_ROOT/merge_epoch${epoch_number}.log"

    stage=merge
    epoch="$epoch_number"
    benchmark=none
    current_log="$merge_log"
    current_output="$model"
    write_status RUNNING

    wait_for_gpu_idle
    stage=merge
    write_status RUNNING
    [ ! -e "$model" ] || fail "Merged model already exists: $model"
    echo "MERGING EPOCH $epoch_number: $(date -u)"
    "$PYTHON" -u "$MERGE" \
      --base_model_name_or_path "$BASE_MODEL" \
      --lora_model_name_or_path "$adapter" \
      --tokenizer_name_or_path "$BASE_MODEL" \
      --output_dir "$model" \
      --save_tokenizer \
      > "$merge_log" 2>&1
    rc=$?
    [ "$rc" -eq 0 ] || fail "Epoch-$epoch_number merge exited with status $rc"
    fatal_log_check "$merge_log"
    validate_merged_model "$model"

    epoch_result="$RESULT_ROOT/epoch_$epoch_number"

    run_eval "$epoch_number" mmlu "$epoch_result/mmlu" "$LOG_ROOT/eval_epoch${epoch_number}_mmlu.log" \
      "$PYTHON" -u -m eval.mmlu.run_eval \
        --ntrain 0 --data_dir "$DATA_ROOT/mmlu" --save_dir "$epoch_result/mmlu" \
        --model_name_or_path "$model" --tokenizer_name_or_path "$model" --eval_batch_size 8

    run_eval "$epoch_number" gsm8k "$epoch_result/gsm" "$LOG_ROOT/eval_epoch${epoch_number}_gsm8k.log" \
      "$PYTHON" -u -m eval.gsm.run_eval \
        --data_dir "$DATA_ROOT/gsm" --max_num_examples 200 --save_dir "$epoch_result/gsm" \
        --model_name_or_path "$model" --tokenizer_name_or_path "$model" --n_shot 8 --use_vllm

    run_eval "$epoch_number" bbh "$epoch_result/bbh" "$LOG_ROOT/eval_epoch${epoch_number}_bbh.log" \
      "$PYTHON" -u -m eval.bbh.run_eval \
        --data_dir "$DATA_ROOT/bbh" --save_dir "$epoch_result/bbh" \
        --model_name_or_path "$model" --tokenizer_name_or_path "$model" \
        --max_num_examples_per_task 40 --use_vllm

    run_eval "$epoch_number" truthfulqa "$epoch_result/truthfulqa" "$LOG_ROOT/eval_epoch${epoch_number}_truthfulqa.log" \
      "$PYTHON" -u -m eval.truthfulqa.run_eval \
        --data_dir "$DATA_ROOT/truthfulqa" --save_dir "$epoch_result/truthfulqa" \
        --model_name_or_path "$model" --tokenizer_name_or_path "$model" \
        --metrics truth info mc --preset qa \
        --hf_truth_model_name_or_path "$TRUTH_JUDGE" \
        --hf_info_model_name_or_path "$INFO_JUDGE" \
        --eval_batch_size 40 --load_in_8bit

    run_eval "$epoch_number" tydiqa "$epoch_result/tydiqa" "$LOG_ROOT/eval_epoch${epoch_number}_tydiqa.log" \
      "$PYTHON" -u -m eval.tydiqa.run_eval \
        --data_dir "$DATA_ROOT/tydiqa" --n_shot 1 --max_num_examples_per_lang 100 \
        --max_context_length 512 --save_dir "$epoch_result/tydiqa" \
        --model_name_or_path "$model" --tokenizer_name_or_path "$model" \
        --eval_batch_size 40 --load_in_8bit
done

stage=aggregation
epoch=none
benchmark=none
current_log="$LOG_ROOT/aggregation.log"
current_output="$FINAL_OUTPUT"
write_status RUNNING
"$PYTHON" -u "$AGGREGATOR" > "$current_log" 2>&1
rc=$?
[ "$rc" -eq 0 ] || fail "Aggregation exited with status $rc"
fatal_log_check "$current_log"

stage=complete
current_log=""
current_output="$FINAL_OUTPUT"
write_status COMPLETE

echo "VANILLA TADS FIVE-EPOCH PIPELINE COMPLETED: $(date -u)"
