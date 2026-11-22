#!/usr/bin/env bash
# Reconstructed from the recorded Random 10K training settings.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'Usage: %s LLAMA_MODEL_DIR TRAIN_JSON NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
test -f "$1/config.json"
test -f "$2"
if [[ -e "$3" ]]; then
    printf 'Output already exists: %s\n' "$3" >&2
    exit 1
fi

accelerate launch --mixed_precision bf16 --num_machines 1 --num_processes 1 \
    "$REPO/shared/upstream_tads/model_finetune/finetune.py" \
    --model_name_or_path "$1" --train_file "$2" --output_dir "$3" \
    --max_seq_length 2048 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 --learning_rate 1e-4 \
    --weight_decay 0 --lr_scheduler_type linear --warmup_ratio 0.03 \
    --num_train_epochs 3 --seed 42 --use_lora \
    --lora_rank 64 --lora_alpha 16 --lora_dropout 0.1 \
    --use_flash_attn --gradient_checkpointing --checkpointing_steps epoch
