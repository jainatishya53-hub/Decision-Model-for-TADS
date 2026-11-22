#!/usr/bin/env bash

set -euo pipefail

source \
/home/gpuuser2/gpuuser2_a/atishya/laya_tads_anchor_classifier/laya_base_restart/configs/experiment_paths.sh

BASE_MODEL="/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

TRAIN_FILE="$BASE_RESTART_ROOT/data/model_finetune/tads_laya_10k_gte_5_dataset.json"

OUTPUT_DIR="$BASE_RESTART_ROOT/models/llama31_8b_laya_tads_gte5_5ep_lora_run_01"

FINETUNE_PROGRAM="$TADS_REPO/model_finetune/finetune.py"

CLEAN_TADS_PATH="$TADS_CONDA/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

test -d "$BASE_MODEL"
test -s "$TRAIN_FILE"
test -s "$FINETUNE_PROGRAM"
test -x "$TADS_CONDA/bin/python"
test -x "$TADS_CONDA/bin/accelerate"

if [ -e "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory already exists:"
    echo "$OUTPUT_DIR"
    echo "Training was not started to prevent overwriting."
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

exec env -i \
HOME="$HOME" \
USER="$(id -un)" \
LOGNAME="$(id -un)" \
SHELL=/bin/bash \
PATH="$CLEAN_TADS_PATH" \
PYTHONNOUSERSITE=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0 \
HF_HOME="/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
"$TADS_CONDA/bin/accelerate" launch \
--num_machines 1 \
--num_processes 1 \
--mixed_precision bf16 \
--dynamo_backend no \
"$FINETUNE_PROGRAM" \
--model_name_or_path "$BASE_MODEL" \
--tokenizer_name "$BASE_MODEL" \
--train_file "$TRAIN_FILE" \
--output_dir "$OUTPUT_DIR" \
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
--reduce_loss mean
