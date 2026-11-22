#!/usr/bin/env bash

set -uo pipefail

PAPER_WORK=/home/gpuuser2/gpuuser2_a/atishya/tads_paper_faithful
TADS_SOURCE=/home/gpuuser2/gpuuser2_a/atishya/tads
CONDA_ENV=/home/gpuuser2/gpuuser2_a/atishya/tads_private/conda_env
HF_HOME=/home/gpuuser2/gpuuser2_a/atishya/tads_private/cache/huggingface

MODEL_ROOT="$PAPER_WORK/model_finetune/model_output/models/random_10k_4runs"
RESULT_ROOT="$PAPER_WORK/model_finetune/model_output/results/random_10k_4runs"
STATE_ROOT="$PAPER_WORK/data/random_10k_4runs/evaluation_queue"
EVAL_LOG_ROOT="$PAPER_WORK/logs/random_10k_4runs/evaluations"

TRUTH_JUDGE="$HF_HOME/hub/models--allenai--truthfulqa-truth-judge-llama2-7B/snapshots/8f718cb4e68bc30a488727a0e37190835a7c129e"
INFO_JUDGE="$HF_HOME/hub/models--allenai--truthfulqa-info-judge-llama2-7B/snapshots/72601eff294cbf7e8d7188ae82397e19cb9bb9bd"

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PAPER_WORK:$TADS_SOURCE/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$RESULT_ROOT" "$STATE_ROOT" "$EVAL_LOG_ROOT"

CURRENT_STATE="$STATE_ROOT/CURRENT_JOB.txt"

fail_job() {
  local message="$1"

  echo
  echo "============================================================"
  echo "EVALUATION QUEUE STOPPED"
  echo "$message"
  echo "Time: $(date -u)"
  echo "============================================================"

  {
    echo "status=FAILED"
    echo "message=$message"
    echo "time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$CURRENT_STATE"

  exit 1
}

run_job() {
  local benchmark="$1"
  local run="$2"

  local model="$MODEL_ROOT/run_${run}_3ep_merged"
  local result_dir="$RESULT_ROOT/run_${run}/${benchmark}"
  local job_log="$EVAL_LOG_ROOT/eval_${benchmark}_run_${run}.log"
  local marker_dir="$STATE_ROOT/${benchmark}/run_${run}"
  local marker="$marker_dir/EVALUATION_COMPLETE.json"

  if [ -f "$marker" ]; then
    echo "SKIPPING COMPLETED JOB: $benchmark Run-$run"
    return 0
  fi

  if [ -e "$result_dir" ]; then
    fail_job "Unexpected existing result path: $result_dir"
  fi

  if [ -e "$job_log" ]; then
    fail_job "Unexpected existing evaluation log: $job_log"
  fi

  mkdir -p "$(dirname "$result_dir")" "$marker_dir"

  {
    echo "status=RUNNING"
    echo "benchmark=$benchmark"
    echo "run=$run"
    echo "model=$model"
    echo "result=$result_dir"
    echo "log=$job_log"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$CURRENT_STATE"

  echo
  echo "============================================================"
  echo "STARTING: $benchmark — RANDOM RUN $run"
  echo "Time: $(date -u)"
  echo "Model: $model"
  echo "Result: $result_dir"
  echo "Log: $job_log"
  echo "============================================================"

  cd "$PAPER_WORK/model_finetune" || \
    fail_job "Could not enter model_finetune directory"

  local rc

  case "$benchmark" in
    mmlu)
      "$CONDA_ENV/bin/python" -u -m eval.mmlu.run_eval \
        --ntrain 0 \
        --data_dir "$PAPER_WORK/data/eval/split_eval/mmlu" \
        --save_dir "$result_dir" \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --eval_batch_size 8 \
        > "$job_log" 2>&1
      rc=$?
      ;;

    gsm8k)
      "$CONDA_ENV/bin/python" -u -m eval.gsm.run_eval \
        --data_dir "$PAPER_WORK/data/eval/split_eval/gsm" \
        --max_num_examples 200 \
        --save_dir "$result_dir" \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --n_shot 8 \
        --use_vllm \
        > "$job_log" 2>&1
      rc=$?
      ;;

    bbh)
      "$CONDA_ENV/bin/python" -u -m eval.bbh.run_eval \
        --data_dir "$PAPER_WORK/data/eval/split_eval/bbh" \
        --save_dir "$result_dir" \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --max_num_examples_per_task 40 \
        --use_vllm \
        > "$job_log" 2>&1
      rc=$?
      ;;

    truthfulqa)
      "$CONDA_ENV/bin/python" -u -m eval.truthfulqa.run_eval \
        --data_dir "$PAPER_WORK/data/eval/split_eval/truthfulqa" \
        --save_dir "$result_dir" \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --metrics truth info mc \
        --preset qa \
        --hf_truth_model_name_or_path "$TRUTH_JUDGE" \
        --hf_info_model_name_or_path "$INFO_JUDGE" \
        --eval_batch_size 40 \
        --load_in_8bit \
        > "$job_log" 2>&1
      rc=$?
      ;;

    tydiqa)
      "$CONDA_ENV/bin/python" -u -m eval.tydiqa.run_eval \
        --data_dir "$PAPER_WORK/data/eval/split_eval/tydiqa" \
        --n_shot 1 \
        --max_num_examples_per_lang 100 \
        --max_context_length 512 \
        --save_dir "$result_dir" \
        --model_name_or_path "$model" \
        --tokenizer_name_or_path "$model" \
        --eval_batch_size 40 \
        --load_in_8bit \
        > "$job_log" 2>&1
      rc=$?
      ;;

    *)
      fail_job "Unknown benchmark: $benchmark"
      ;;
  esac

  if [ "$rc" -ne 0 ]; then
    tail -n 80 "$job_log" || true
    fail_job "$benchmark Run-$run exited with status $rc"
  fi

  if grep -aEq \
    'Traceback \(most recent call last\)|CUDA out of memory|ChildFailedError|RuntimeError|AssertionError|Killed|SIGTERM|SIGKILL|Segmentation fault|Bus error' \
    "$job_log"
  then
    tail -n 80 "$job_log" || true
    fail_job "Fatal pattern detected in $benchmark Run-$run"
  fi

  if [ ! -d "$result_dir" ]; then
    fail_job "Missing result directory for $benchmark Run-$run"
  fi

  first_file="$(
    find "$result_dir" -type f -size +0c -print -quit 2>/dev/null
  )"

  if [ -z "$first_file" ]; then
    fail_job "No non-empty result files for $benchmark Run-$run"
  fi

  case "$benchmark" in
    mmlu)
      if [ ! -s "$result_dir/metrics.json" ]; then
        fail_job "MMLU metrics.json is missing"
      fi

      csv_count="$(
        find "$result_dir" -maxdepth 1 -type f -name '*.csv' |
        wc -l |
        tr -d ' '
      )"

      if [ "$csv_count" -ne 57 ]; then
        fail_job "MMLU produced $csv_count subject CSV files instead of 57"
      fi
      ;;

    gsm8k)
      if [ ! -s "$result_dir/metrics.json" ]; then
        fail_job "GSM8K metrics.json is missing"
      fi

      if [ ! -s "$result_dir/predictions.jsonl" ]; then
        fail_job "GSM8K predictions.jsonl is missing"
      fi

      prediction_count="$(wc -l < "$result_dir/predictions.jsonl" | tr -d ' ')"

      if [ "$prediction_count" -ne 200 ]; then
        fail_job "GSM8K produced $prediction_count predictions instead of 200"
      fi
      ;;
  esac

  BENCHMARK="$benchmark" \
  RUN_NUMBER="$run" \
  MODEL_PATH="$model" \
  RESULT_PATH="$result_dir" \
  LOG_PATH="$job_log" \
  MARKER_PATH="$marker" \
  "$CONDA_ENV/bin/python" - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

benchmark = os.environ["BENCHMARK"]
run = int(os.environ["RUN_NUMBER"])
model = Path(os.environ["MODEL_PATH"])
result = Path(os.environ["RESULT_PATH"])
log = Path(os.environ["LOG_PATH"])
marker = Path(os.environ["MARKER_PATH"])

files = sorted(path for path in result.rglob("*") if path.is_file())

record = {
    "status": "EVALUATION_PROCESS_COMPLETED",
    "benchmark": benchmark,
    "run": run,
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "model": str(model),
    "result_directory": str(result),
    "log": str(log),
    "log_sha256": sha256(log),
    "result_file_count": len(files),
    "result_total_bytes": sum(path.stat().st_size for path in files),
}

marker.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

  if [ ! -s "$marker" ]; then
    fail_job "Completion marker was not created for $benchmark Run-$run"
  fi

  echo "COMPLETED: $benchmark — RANDOM RUN $run"
  echo "Time: $(date -u)"
  du -sh "$result_dir" || true

  sleep 15
}

echo "RANDOM 10K × 4 UNIFIED EVALUATION QUEUE STARTED"
echo "Time: $(date -u)"
echo "Order: benchmark-major, runs 1–4"
echo "Total jobs: 20"

for benchmark in mmlu gsm8k bbh truthfulqa tydiqa
do
  for run in 1 2 3 4
  do
    run_job "$benchmark" "$run"
  done
done

marker_count="$(
  find "$STATE_ROOT" \
    -type f \
    -name EVALUATION_COMPLETE.json |
  wc -l |
  tr -d ' '
)"

if [ "$marker_count" -ne 20 ]; then
  fail_job "Only $marker_count of 20 completion markers were created"
fi

{
  echo "status=COMPLETE"
  echo "completed_jobs=20"
  echo "completed=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$CURRENT_STATE"

cat > "$STATE_ROOT/RANDOM_10K_4RUNS_ALL_EVALUATIONS_COMPLETE" <<EOF
RANDOM 10K × 4 EVALUATION QUEUE COMPLETE
Completed: $(date -u)
Jobs: 20
Benchmarks: mmlu gsm8k bbh truthfulqa tydiqa
Runs: 1 2 3 4
EOF

echo
echo "============================================================"
echo "ALL 20 RANDOM-SAMPLING EVALUATIONS COMPLETED"
echo "Time: $(date -u)"
echo "Results: $RESULT_ROOT"
echo "============================================================"
