#!/usr/bin/env bash

set -euo pipefail

RESTART_ROOT="/home/gpuuser2/gpuuser2_a/atishya/laya_tads_anchor_classifier/laya_base_restart"
TADS_REPO="/home/gpuuser2/gpuuser2_a/atishya/tads"
TADS_CONDA="/home/gpuuser2/gpuuser2_a/atishya/tads_private/conda_env"
PYTHON="$TADS_CONDA/bin/python"

BASE_MODEL="/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"

LORA_ROOT="$RESTART_ROOT/models/llama31_8b_laya_tads_gte5_5ep_lora_run_01"
EVAL_DATA_ROOT="/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful/data/eval/split_eval"

RESULTS_ROOT="$RESTART_ROOT/data/epoch_evaluation/results"
STATE_ROOT="$RESTART_ROOT/data/epoch_evaluation/status"
SCRATCH_ROOT="$RESTART_ROOT/models/epoch_evaluation_scratch"
SUMMARY_ROOT="$RESTART_ROOT/metadata/epoch_evaluation"

mkdir -p \
"$RESULTS_ROOT" \
"$STATE_ROOT" \
"$SCRATCH_ROOT" \
"$SUMMARY_ROOT"

cd "$TADS_REPO/model_finetune"

safe_remove_merged_model() {
    local target="$1"

    if [[ ! -e "$target" ]]; then
        return
    fi

    "$PYTHON" - "$target" "$SCRATCH_ROOT" <<'PY'
from pathlib import Path
import shutil
import sys

target = Path(sys.argv[1]).resolve()
scratch_root = Path(sys.argv[2]).resolve()

assert target.parent == scratch_root, (
    f"Refusing to remove a path outside scratch root: {target}"
)

assert target.name.startswith("merged_epoch_"), (
    f"Unexpected temporary model directory: {target}"
)

shutil.rmtree(target)

print("Removed reproducible temporary merged model:", target)
PY
}

merged_model_is_complete() {
    local model_directory="$1"

    test -s "$model_directory/config.json" &&
    find "$model_directory" \
        -maxdepth 1 \
        -type f \
        -name '*.safetensors' \
        -print -quit |
    grep -q .
}

for EPOCH in 1 2 3 4 5
do
    CHECKPOINT_INDEX=$((EPOCH - 1))

    LORA_CHECKPOINT="$LORA_ROOT/epoch_${CHECKPOINT_INDEX}"
    MERGED_MODEL="$SCRATCH_ROOT/merged_epoch_${EPOCH}"
    TAG="epoch_${EPOCH}"
    EPOCH_STATE="$STATE_ROOT/$TAG"

    mkdir -p "$EPOCH_STATE"

    echo
    echo "========================================================================"
    echo "EVALUATING TRAINING EPOCH $EPOCH"
    echo "LoRA checkpoint: $LORA_CHECKPOINT"
    echo "========================================================================"

    test -s "$LORA_CHECKPOINT/adapter_model.safetensors"

    ALL_BENCHMARKS_COMPLETE=true

    for BENCHMARK in mmlu gsm bbh truthfulqa tydiqa
    do
        if [[ ! -s "$EPOCH_STATE/${BENCHMARK}.complete" ]]; then
            ALL_BENCHMARKS_COMPLETE=false
        fi
    done

    if [[ "$ALL_BENCHMARKS_COMPLETE" == false ]]; then
        if merged_model_is_complete "$MERGED_MODEL"; then
            echo "Reusing existing merged model for epoch $EPOCH"
        else
            safe_remove_merged_model "$MERGED_MODEL"

            echo "Merging epoch $EPOCH LoRA adapter with base Llama..."

            "$PYTHON" merge_lora.py \
                --base_model_name_or_path "$BASE_MODEL" \
                --lora_model_name_or_path "$LORA_CHECKPOINT" \
                --output_dir "$MERGED_MODEL" \
                --tokenizer_name_or_path "$BASE_MODEL" \
                --save_tokenizer

            merged_model_is_complete "$MERGED_MODEL"
        fi

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/merge.complete"
    fi

    if [[ ! -s "$EPOCH_STATE/mmlu.complete" ]]; then
        echo "Running MMLU for epoch $EPOCH..."

        "$PYTHON" -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir "$EVAL_DATA_ROOT/mmlu" \
            --save_dir "$RESULTS_ROOT/mmlu/$TAG" \
            --model_name_or_path "$MERGED_MODEL" \
            --tokenizer_name_or_path "$MERGED_MODEL" \
            --eval_batch_size 8

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/mmlu.complete"
    else
        echo "MMLU already complete for epoch $EPOCH"
    fi

    if [[ ! -s "$EPOCH_STATE/gsm.complete" ]]; then
        echo "Running GSM8K for epoch $EPOCH..."

        "$PYTHON" -m eval.gsm.run_eval \
            --data_dir "$EVAL_DATA_ROOT/gsm/" \
            --max_num_examples 200 \
            --save_dir "$RESULTS_ROOT/gsm/$TAG" \
            --model_name_or_path "$MERGED_MODEL" \
            --tokenizer_name_or_path "$MERGED_MODEL" \
            --n_shot 8 \
            --use_vllm

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/gsm.complete"
    else
        echo "GSM8K already complete for epoch $EPOCH"
    fi

    if [[ ! -s "$EPOCH_STATE/bbh.complete" ]]; then
        echo "Running BBH for epoch $EPOCH..."

        "$PYTHON" -m eval.bbh.run_eval \
            --data_dir "$EVAL_DATA_ROOT/bbh/" \
            --save_dir "$RESULTS_ROOT/bbh/$TAG" \
            --model_name_or_path "$MERGED_MODEL" \
            --tokenizer_name_or_path "$MERGED_MODEL" \
            --max_num_examples_per_task 40 \
            --use_vllm

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/bbh.complete"
    else
        echo "BBH already complete for epoch $EPOCH"
    fi

    if [[ ! -s "$EPOCH_STATE/truthfulqa.complete" ]]; then
        echo "Running TruthfulQA for epoch $EPOCH..."

        "$PYTHON" -m eval.truthfulqa.run_eval \
            --data_dir "$EVAL_DATA_ROOT/truthfulqa" \
            --save_dir "$RESULTS_ROOT/truthfulqa/$TAG" \
            --model_name_or_path "$MERGED_MODEL" \
            --tokenizer_name_or_path "$MERGED_MODEL" \
            --metrics truth info mc \
            --preset qa \
            --hf_truth_model_name_or_path \
            /home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface/hub/models--allenai--truthfulqa-truth-judge-llama2-7B/snapshots/8f718cb4e68bc30a488727a0e37190835a7c129e \
            --hf_info_model_name_or_path \
            /home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface/hub/models--allenai--truthfulqa-info-judge-llama2-7B/snapshots/72601eff294cbf7e8d7188ae82397e19cb9bb9bd \
            --eval_batch_size 40 \
            --load_in_8bit

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/truthfulqa.complete"
    else
        echo "TruthfulQA already complete for epoch $EPOCH"
    fi

    if [[ ! -s "$EPOCH_STATE/tydiqa.complete" ]]; then
        echo "Running TyDiQA for epoch $EPOCH..."

        "$PYTHON" -m eval.tydiqa.run_eval \
            --data_dir "$EVAL_DATA_ROOT/tydiqa/" \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir "$RESULTS_ROOT/tydiqa/$TAG" \
            --model_name_or_path "$MERGED_MODEL" \
            --tokenizer_name_or_path "$MERGED_MODEL" \
            --eval_batch_size 40 \
            --load_in_8bit

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/tydiqa.complete"
    else
        echo "TyDiQA already complete for epoch $EPOCH"
    fi

    if [[ ! -s "$EPOCH_STATE/aggregation.complete" ]]; then
        echo "Aggregating official TADS results for epoch $EPOCH..."

        "$PYTHON" read_results.py \
            --root_result_path "$RESULTS_ROOT" \
            --baseline_tag "$TAG" |
        tee "$SUMMARY_ROOT/${TAG}_official_summary.txt"

        date -u +"%Y-%m-%dT%H:%M:%SZ" \
            > "$EPOCH_STATE/aggregation.complete"
    fi

    safe_remove_merged_model "$MERGED_MODEL"

    date -u +"%Y-%m-%dT%H:%M:%SZ" \
        > "$EPOCH_STATE/epoch.complete"

    echo "Epoch $EPOCH evaluation completed."
done

echo
echo "========================================================================"
echo "ALL FIVE EPOCH EVALUATIONS COMPLETED"
echo "========================================================================"
echo "Results: $RESULTS_ROOT"
echo "Summaries: $SUMMARY_ROOT"
